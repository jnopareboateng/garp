"""Keep a committed AgentOS session deletion authoritative across native writers."""

import asyncio
import logging
from threading import Lock
from typing import ClassVar

from agno.db.postgres import PostgresDb
from psycopg import errors, sql
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

_SCHEMA_VERSION = "garp chat deletion v7"
logger = logging.getLogger(__name__)


class ReplayPublicationFence:
    """Order Redis publication against deletion without using the native DB pool."""

    _lock = Lock()
    _loops: ClassVar[dict] = {}

    def __init__(self, db):
        self.url = db.db_engine.url
        self.sessions = sql.Identifier(db.db_schema, db.session_table_name)
        self.runs = sql.Identifier(db.db_schema, db.runs_table_name)
        self.jobs = sql.Identifier(db.db_schema, db.job_table_name)
        self.deleted_runs = sql.Identifier(db.db_schema, "garp_deleted_runs")
        self.query = text(
            f"SELECT session_id FROM {self.sessions.as_string()} "
            f"WHERE session_id = COALESCE((SELECT session_id FROM {self.runs.as_string()} "
            f"WHERE run_id = :run_id), (SELECT session_id FROM {self.jobs.as_string()} "
            "WHERE id = :run_id), :session_id) "
            f"AND NOT EXISTS (SELECT 1 FROM {self.deleted_runs.as_string()} WHERE run_id = :run_id) "
            "FOR KEY SHARE"
        )

    @classmethod
    def open_loop(cls):
        loop = asyncio.get_running_loop()
        with cls._lock:
            state = cls._loops.setdefault(loop, {"refs": 0, "engines": {}, "tasks": set()})
            state["refs"] += 1

    @classmethod
    async def close_loop(cls):
        loop = asyncio.get_running_loop()
        with cls._lock:
            state = cls._loops.get(loop)
            if state is None:
                return
            state["refs"] = max(0, state["refs"] - 1)
            if state["refs"]:
                return
            cls._loops.pop(loop)
        if state["tasks"]:
            await asyncio.gather(*state["tasks"], return_exceptions=True)
        for engine in state["engines"].values():
            await engine.dispose()

    async def __call__(self, run_id, event, publish):
        loop = asyncio.get_running_loop()
        with self._lock:
            state = self._loops.get(loop)
            if state is None:
                return -1
            engine = state["engines"].get(self.url)
            if engine is None:
                engine = create_async_engine(
                    self.url,
                    pool_size=8,
                    max_overflow=0,
                    pool_timeout=5,
                    connect_args={"connect_timeout": 5},
                )
                state["engines"][self.url] = engine
            task = asyncio.current_task()
            state["tasks"].add(task)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SET LOCAL lock_timeout = '5s'"))
                row = await conn.execute(
                    self.query, {"session_id": getattr(event, "session_id", None), "run_id": run_id}
                )
                if row.first() is None:
                    return -1
                # A disconnected or timed-out Redis write may still complete server-side.
                return await asyncio.wait_for(publish(), timeout=10)
        finally:
            state["tasks"].discard(task)


class DeletedRun(RuntimeError):
    pass


class GarpPostgresDb(PostgresDb):
    def deleted_run(self, run_id):
        """Return whether a committed deletion fence exists for a run."""
        if not run_id:
            return False
        table = sql.Identifier(self.db_schema, "garp_deleted_runs").as_string()
        with self.Session() as conn:
            return bool(
                conn.scalar(
                    text(f"SELECT EXISTS (SELECT 1 FROM {table} WHERE run_id = :run_id)"),
                    {"run_id": run_id},
                )
            )

    def session_deleted(self, session_id):
        if not session_id:
            return False
        table = sql.Identifier(self.db_schema, "garp_deleted_sessions").as_string()
        with self.Session() as conn:
            return conn.scalar(
                text(f"SELECT EXISTS (SELECT 1 FROM {table} WHERE session_id = :session_id)"),
                {"session_id": session_id},
            )

    def upsert_session(self, session, deserialize=True):
        try:
            return super().upsert_session(session, deserialize)
        except IntegrityError as exc:
            if not isinstance(exc.orig, errors.CheckViolation) or (
                exc.orig.diag.constraint_name != "garp_deleted_session"
            ):
                raise
            # A late save is a normal no-op after deletion. Observability redacts DB logs.
            return None

    def upsert_run(self, run, session_id, user_id=None, run_index=None):
        try:
            super().upsert_run(run, session_id, user_id, run_index)
        except IntegrityError as exc:
            if not isinstance(exc.orig, errors.ForeignKeyViolation) or (
                exc.orig.diag.constraint_name != f"{self.runs_table_name}_session_id_fkey"
                or not self.session_deleted(session_id)
            ):
                raise
            run_id = run.get("run_id") if isinstance(run, dict) else getattr(run, "run_id", None)
            if run_id:
                table = sql.Identifier(self.db_schema, "garp_deleted_runs").as_string()
                with self.Session() as conn, conn.begin():
                    conn.execute(
                        text(
                            f"INSERT INTO {table} (run_id) VALUES (:run_id) ON CONFLICT DO NOTHING"
                        ),
                        {"run_id": run_id},
                    )

    def enqueue_job(self, job, max_depth=0):
        try:
            return super().enqueue_job(job, max_depth)
        except IntegrityError as exc:
            if isinstance(exc.orig, errors.CheckViolation) and exc.orig.diag.constraint_name in (
                "garp_deleted_session",
                "garp_deleted_run_key",
            ):
                raise DeletedRun("Chat is unavailable") from None
            raise

    def create_schedule(self, schedule_data):
        try:
            return super().create_schedule(schedule_data)
        except IntegrityError as exc:
            if (
                isinstance(exc.orig, errors.CheckViolation)
                and exc.orig.diag.constraint_name == "garp_deleted_session"
            ):
                raise DeletedRun("Scheduled chat is unavailable") from None
            raise

    def delete_session(self, session_id, user_id=None):
        deleted = super().delete_session(session_id, user_id)
        if deleted:
            self.prune_deleted_chat_schedules([session_id], user_id)
            self.prune_deleted_chat_traces([session_id])
        return deleted

    def delete_sessions(self, session_ids, user_id=None):
        super().delete_sessions(session_ids, user_id)
        if session_ids:
            self.prune_deleted_chat_schedules(session_ids, user_id)
            self.prune_deleted_chat_traces(session_ids)

    def prune_deleted_chat_schedules(self, session_ids=None, user_id=None):
        try:
            schedules = self._get_table("schedules")
            if schedules is None:
                return
            table = sql.Identifier(schedules.schema, schedules.name).as_string()
            attempts = self._get_table("schedule_runs")
            markers = sql.Identifier(self.db_schema, "garp_deleted_sessions").as_string()
            target = sql.Identifier(self.db_schema, "garp_schedule_target_chat").as_string()
            if attempts is not None:
                attempt_table = sql.Identifier(attempts.schema, attempts.name).as_string()
                with self.Session() as conn, conn.begin():
                    conn.execute(text("SET LOCAL lock_timeout = '1s'"))
                    conn.execute(
                        text(f"""DELETE FROM {attempt_table} WHERE id IN (
                        SELECT a.id FROM {attempt_table} a JOIN {markers} d
                            ON d.session_id = a.session_id
                        WHERE (:all_sessions OR d.session_id = ANY(CAST(:session_ids AS text[])))
                            AND (CAST(:user_id AS text) IS NULL OR a.user_id = :user_id)
                        FOR UPDATE OF a SKIP LOCKED
                    )"""),
                        {
                            "all_sessions": session_ids is None,
                            "session_ids": session_ids or [],
                            "user_id": user_id,
                        },
                    )
            # Run after the chat transaction. A claimed schedule can hold its row lock
            # while it attempts to admit work against the session. Lock the schedule
            # before removing attempts so the native FK prevents late inserts.
            with self.Session() as conn, conn.begin():
                conn.execute(text("SET LOCAL lock_timeout = '1s'"))
                ids = conn.scalars(
                    text(f"""SELECT s.id FROM {table} s JOIN {markers} d
                        ON d.session_id = {target}(s.method, s.managed_by, s.endpoint, s.payload)
                    WHERE (:all_sessions OR d.session_id = ANY(CAST(:session_ids AS text[])))
                        AND (CAST(:user_id AS text) IS NULL OR s.user_id = :user_id)
                    FOR UPDATE OF s SKIP LOCKED"""),
                    {
                        "all_sessions": session_ids is None,
                        "session_ids": session_ids or [],
                        "user_id": user_id,
                    },
                ).all()
                if ids:
                    if attempts is not None:
                        conn.execute(attempts.delete().where(attempts.c.schedule_id.in_(ids)))
                    conn.execute(schedules.delete().where(schedules.c.id.in_(ids)))
        except Exception as exc:  # noqa: BLE001 - Deletion already committed; maintenance retries.
            logger.warning(
                "Deleted chat schedule cleanup will retry (%s, SQLSTATE %s)",
                type(exc).__name__,
                getattr(getattr(exc, "orig", exc), "sqlstate", None),
            )

    def prune_deleted_chat_traces(self, session_ids=None):
        from garp.observability import _delete_chat_traces

        try:
            _delete_chat_traces(self, session_ids)
        except Exception as exc:  # noqa: BLE001 - The deletion committed; maintenance retries.
            logger.warning("Deleted chat trace cleanup will retry (%s)", type(exc).__name__)

    def cleanup_jobs(self, older_than_seconds=86400):
        count = super().cleanup_jobs(older_than_seconds)
        receipts = sql.Identifier(self.db_schema, "garp_deleted_run_keys").as_string()
        try:
            with self.Session() as conn, conn.begin():
                conn.execute(text("SET LOCAL lock_timeout = '1s'"))
                conn.execute(text(f"DELETE FROM {receipts} WHERE expires_at <= clock_timestamp()"))
        except Exception as exc:  # noqa: BLE001 - Keep the native sweep result and cadence.
            logger.warning(
                "Deleted chat replay key cleanup will retry (%s, SQLSTATE %s)",
                type(exc).__name__,
                getattr(getattr(exc, "orig", exc), "sqlstate", None),
            )
        self.prune_deleted_chat_schedules()
        self.prune_deleted_chat_traces()
        return count


def install_session_deletion(store, db):
    with store._connect() as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext('garp.session_deletion'), hashtext(%s))",
            (db.db_schema,),
        )
        sessions = db._get_table("sessions", create_table_if_not_found=True)
        jobs = db._get_table("jobs", create_table_if_not_found=True)
        schedules = db._get_table("schedules", create_table_if_not_found=True)
        runs = db._get_table("runs", create_table_if_not_found=True)
        attempts = db._get_table("schedule_runs", create_table_if_not_found=True)
        approvals = db._get_table("approvals", create_table_if_not_found=True)
        if (
            sessions is None
            or jobs is None
            or schedules is None
            or runs is None
            or attempts is None
            or approvals is None
        ):
            raise RuntimeError("Session deletion requires native session, job, and schedule tables")
        session_table = sql.Identifier(sessions.schema, sessions.name)
        job_table = sql.Identifier(jobs.schema, jobs.name)
        schedule_table = sql.Identifier(schedules.schema, schedules.name)
        runs_table = sql.Identifier(runs.schema, runs.name)
        attempts_table = sql.Identifier(attempts.schema, attempts.name)
        approval_table = sql.Identifier(approvals.schema, approvals.name)
        markers = sql.Identifier(db.db_schema, "garp_deleted_sessions")
        receipts = sql.Identifier(db.db_schema, "garp_deleted_run_keys")
        deleted_runs = sql.Identifier(db.db_schema, "garp_deleted_runs")
        fence = sql.Identifier(db.db_schema, "garp_fence_deleted_chat")
        conn.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} (trace_id text PRIMARY KEY)").format(
                sql.Identifier(db.db_schema, "garp_deleted_traces")
            )
        )
        if conn.execute(
            """SELECT count(*) = 6 AND to_regclass(%s) IS NOT NULL AND to_regclass(%s) IS NOT NULL
                   AND to_regclass(%s) IS NOT NULL
                   AND obj_description(to_regprocedure(%s), 'pg_proc') = %s AS installed
               FROM pg_catalog.pg_trigger
               WHERE NOT tgisinternal AND tgenabled = 'O' AND (
                   (tgrelid = %s::regclass AND tgname IN ('garp_mark_deleted_chat', 'garp_fence_chat_insert'))
                   OR (tgrelid = %s::regclass AND tgname = 'garp_fence_chat_job')
                   OR (tgrelid = %s::regclass AND tgname = 'garp_fence_chat_schedule')
                   OR (tgrelid = %s::regclass AND tgname = 'garp_fence_chat_attempt')
                   OR (tgrelid = %s::regclass AND tgname = 'garp_fence_chat_approval'))""",
            (
                markers.as_string(conn),
                receipts.as_string(conn),
                deleted_runs.as_string(conn),
                fence.as_string(conn) + "()",
                _SCHEMA_VERSION,
                session_table.as_string(conn),
                job_table.as_string(conn),
                schedule_table.as_string(conn),
                attempts_table.as_string(conn),
                approval_table.as_string(conn),
            ),
        ).fetchone()["installed"]:
            return

        conn.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {markers} (session_id text PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS {receipts} (digest bytea PRIMARY KEY, expires_at timestamptz NOT NULL);
                CREATE TABLE IF NOT EXISTS {deleted_runs} (run_id text PRIMARY KEY, deleted_at timestamptz NOT NULL DEFAULT clock_timestamp());

                CREATE OR REPLACE FUNCTION {schedule_target}(text, text, text, jsonb)
                    RETURNS text LANGUAGE sql IMMUTABLE AS $body$
                    SELECT CASE WHEN $1 = 'POST' THEN CASE
                        WHEN $2 = 'garp-prompt' THEN $4 #>> '{{prompt,session_id}}'
                        WHEN $3 ~ '^/(agents|teams|workflows)/[^/]+/runs/?$' THEN $4 ->> 'session_id'
                        ELSE substring(split_part(split_part($3, '?', 1), '#', 1)
                            FROM '^/garp/sessions/([^/]+)/checks/?$')
                    END END;
                $body$;

                CREATE OR REPLACE FUNCTION {schedule_fence}() RETURNS trigger LANGUAGE plpgsql AS $body$
                DECLARE target_chat text;
                BEGIN
                    target_chat := {schedule_target}(NEW.method, NEW.managed_by, NEW.endpoint, NEW.payload);
                    IF target_chat IS NOT NULL THEN
                        PERFORM 1 FROM {sessions} WHERE session_id = target_chat FOR KEY SHARE;
                        IF EXISTS (SELECT 1 FROM {markers} WHERE session_id = target_chat) THEN
                            RAISE check_violation USING MESSAGE = 'This chat was deleted',
                                CONSTRAINT = 'garp_deleted_session';
                        END IF;
                    END IF;
                    RETURN NEW;
                END
                $body$;

                CREATE OR REPLACE FUNCTION {approval_fence}() RETURNS trigger LANGUAGE plpgsql AS $body$
                BEGIN
                    PERFORM 1 FROM {sessions} WHERE session_id = NEW.session_id FOR KEY SHARE;
                    IF EXISTS (SELECT 1 FROM {markers} WHERE session_id = NEW.session_id) THEN
                        RAISE check_violation USING MESSAGE = 'This chat was deleted',
                            CONSTRAINT = 'garp_deleted_session';
                    END IF;
                    RETURN NEW;
                END
                $body$;

                CREATE OR REPLACE FUNCTION {mark}() RETURNS trigger LANGUAGE plpgsql AS $body$
                DECLARE removed record;
                BEGIN
                    INSERT INTO {markers} VALUES (OLD.session_id) ON CONFLICT DO NOTHING;
                    DELETE FROM {approvals} WHERE session_id = OLD.session_id;
                    INSERT INTO {deleted_runs} (run_id)
                        SELECT run_id FROM {runs} WHERE session_id = OLD.session_id
                        ON CONFLICT DO NOTHING;
                    FOR removed IN DELETE FROM {jobs} WHERE session_id = OLD.session_id
                        RETURNING id, user_id, idempotency_key LOOP
                        INSERT INTO {deleted_runs} (run_id) VALUES (removed.id)
                            ON CONFLICT DO NOTHING;
                        IF removed.idempotency_key IS NOT NULL THEN
                            INSERT INTO {receipts}
                            VALUES (
                                sha256(convert_to(jsonb_build_array(removed.user_id,
                                    removed.idempotency_key)::text, 'UTF8')),
                                clock_timestamp() + interval '1 day'
                            ) ON CONFLICT DO NOTHING;
                        END IF;
                    END LOOP;
                    RETURN OLD;
                END
                $body$;

                CREATE OR REPLACE FUNCTION {fence}() RETURNS trigger LANGUAGE plpgsql AS $body$
                DECLARE parent_session text;
                BEGIN
                    IF TG_NARGS = 1 THEN
                        parent_session := NEW.payload #>> '{{kwargs,metadata,garp_parent_session_id}}';
                        -- Order admission against deletion of the target and of a delegating chat.
                        PERFORM 1 FROM {sessions} WHERE session_id IN (NEW.session_id, parent_session)
                            ORDER BY session_id FOR KEY SHARE;
                        -- An original submission may have omitted its session ID.
                        IF NEW.idempotency_key IS NOT NULL AND EXISTS (
                            SELECT 1 FROM {receipts}
                            WHERE digest = sha256(convert_to(
                                jsonb_build_array(NEW.user_id, NEW.idempotency_key)::text, 'UTF8'))
                            AND expires_at > clock_timestamp()
                        ) THEN
                            RAISE check_violation USING MESSAGE = 'This chat was deleted',
                                CONSTRAINT = 'garp_deleted_run_key';
                        END IF;
                    END IF;
                    IF EXISTS (SELECT 1 FROM {markers} WHERE session_id IN (NEW.session_id, parent_session)) THEN
                        RAISE check_violation USING MESSAGE = 'This chat was deleted',
                            CONSTRAINT = 'garp_deleted_session';
                    END IF;
                    RETURN NEW;
                END
                $body$;
                COMMENT ON FUNCTION {fence}() IS {version};

                CREATE OR REPLACE TRIGGER garp_fence_chat_approval
                    BEFORE INSERT OR UPDATE OF session_id ON {approvals}
                    FOR EACH ROW EXECUTE FUNCTION {approval_fence}();
                CREATE OR REPLACE TRIGGER garp_fence_chat_attempt
                    BEFORE INSERT OR UPDATE OF session_id ON {attempts}
                    FOR EACH ROW EXECUTE FUNCTION {approval_fence}();
                -- Capture native run IDs before AgentOS's session FK cascade removes them.
                CREATE OR REPLACE TRIGGER garp_mark_deleted_chat BEFORE DELETE ON {sessions}
                    FOR EACH ROW EXECUTE FUNCTION {mark}();
                -- A waiting ON CONFLICT upsert must check the marker after insertion.
                CREATE OR REPLACE TRIGGER garp_fence_chat_insert AFTER INSERT ON {sessions}
                    FOR EACH ROW EXECUTE FUNCTION {fence}();
                CREATE OR REPLACE TRIGGER garp_fence_chat_job AFTER INSERT ON {jobs}
                    FOR EACH ROW EXECUTE FUNCTION {fence}('job');
                CREATE OR REPLACE TRIGGER garp_fence_chat_schedule
                    BEFORE INSERT OR UPDATE OF method, managed_by, endpoint, payload ON {schedules}
                    FOR EACH ROW EXECUTE FUNCTION {schedule_fence}();
                DELETE FROM {approvals} a USING {markers} d WHERE a.session_id = d.session_id;
            """).format(
                markers=markers,
                receipts=receipts,
                deleted_runs=deleted_runs,
                sessions=session_table,
                jobs=job_table,
                schedules=schedule_table,
                runs=runs_table,
                attempts=attempts_table,
                approvals=approval_table,
                approval_fence=sql.Identifier(db.db_schema, "garp_fence_deleted_chat_approval"),
                mark=sql.Identifier(db.db_schema, "garp_mark_deleted_chat"),
                fence=fence,
                schedule_target=sql.Identifier(db.db_schema, "garp_schedule_target_chat"),
                schedule_fence=sql.Identifier(db.db_schema, "garp_fence_deleted_chat_schedule"),
                version=sql.Literal(_SCHEMA_VERSION),
            )
        )
