from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from garp.models import default_model, validate_options

NATIVE_CHECK_CHAT_FK = "garp_schedules_native_chat_fkey"


class NotFound(LookupError):
    pass


def lock_cron_claim(conn, table, claim, owner):
    active = conn.execute(
        sql.SQL("""SELECT 1 FROM {} WHERE id = %s AND user_id = %s AND enabled
                AND next_run_at = %s AND locked_by = %s AND locked_at = %s
                FOR UPDATE""").format(sql.Identifier(*table)),
        (claim.id, owner, claim.next_run_at, claim.locked_by, claim.locked_at),
    ).fetchone()
    if active is None:
        raise NotFound("Cron claim is no longer active")


class Store:
    def __init__(self, dsn):
        self.dsn = dsn

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5)

    def initialize(self):
        with self._connect() as conn:
            if conn.info.encoding != "utf-8":
                raise ValueError("Garp requires a UTF-8 Postgres database")
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('garp.schema'))")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS garp_sessions (
                    id uuid PRIMARY KEY, owner text NOT NULL,
                    turn_token uuid, turn_expires timestamptz, turn_text text, title text,
                    created_at timestamptz NOT NULL DEFAULT now()
                );
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS title text;
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS model text;
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS effort text;
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS web_search boolean NOT NULL DEFAULT false;
                CREATE TABLE IF NOT EXISTS garp_events (
                    session_id uuid NOT NULL REFERENCES garp_sessions(id),
                    seq bigint NOT NULL, kind text NOT NULL, data jsonb NOT NULL,
                    PRIMARY KEY (session_id, seq)
                );
                CREATE TABLE IF NOT EXISTS garp_schedules (
                    id uuid PRIMARY KEY, session_id uuid NOT NULL,
                    job_ref text NOT NULL, baseline jsonb NOT NULL, due_at timestamptz NOT NULL,
                    deadline_at timestamptz, status text NOT NULL DEFAULT 'pending',
                    lease_token uuid, lease_expires timestamptz, attempt integer NOT NULL DEFAULT 0,
                    outcome text, evidence jsonb
                );
                ALTER TABLE garp_schedules DROP CONSTRAINT IF EXISTS garp_schedules_session_id_fkey;
                ALTER TABLE garp_schedules ADD COLUMN IF NOT EXISTS native_owner text;
                ALTER TABLE garp_schedules ADD COLUMN IF NOT EXISTS delivered_at timestamptz;
                ALTER TABLE garp_schedules ADD COLUMN IF NOT EXISTS delivery_error text;
                ALTER TABLE garp_schedules ADD COLUMN IF NOT EXISTS next_delivery_at timestamptz;
                CREATE INDEX IF NOT EXISTS garp_schedules_due ON garp_schedules(due_at)
                    WHERE status IN ('pending', 'leased');
            """)

    def install_native_check_reference(self, schema, sessions_table):
        sessions = sql.Identifier(schema, sessions_table)
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('garp.schema'))")
            state = conn.execute(
                """SELECT EXISTS (SELECT 1 FROM pg_attribute
                       WHERE attrelid = to_regclass('garp_schedules')
                         AND attname = 'native_owner' AND NOT attisdropped
                   ) AS present, EXISTS (
                       SELECT 1 FROM pg_constraint
                       WHERE conrelid = to_regclass('garp_schedules') AND conname = %s
                   ) AS installed""",
                (NATIVE_CHECK_CHAT_FK,),
            ).fetchone()
            if not state["present"] or state["installed"]:
                return
            # Legacy occurrences lack a native owner and keep a NULL reference.
            conn.execute(
                sql.SQL("""
                    ALTER TABLE garp_schedules ADD COLUMN native_session_id text GENERATED ALWAYS AS
                        (CASE WHEN native_owner IS NOT NULL THEN session_id::text END) STORED;
                    CREATE INDEX garp_schedules_native_chat ON garp_schedules(native_session_id)
                        WHERE native_session_id IS NOT NULL;
                    ALTER TABLE garp_schedules ADD CONSTRAINT {constraint}
                        FOREIGN KEY (native_session_id) REFERENCES {sessions}(session_id)
                        ON DELETE CASCADE NOT VALID;
                    DELETE FROM garp_schedules AS occurrence WHERE native_session_id IS NOT NULL
                        AND NOT EXISTS (SELECT 1 FROM {sessions} s
                            WHERE s.session_id = occurrence.native_session_id);
                    ALTER TABLE garp_schedules VALIDATE CONSTRAINT {constraint};
                """).format(sessions=sessions, constraint=sql.Identifier(NATIVE_CHECK_CHAT_FK))
            )

    @staticmethod
    def _session(conn, owner, session_id, *, lock=False):
        suffix = " FOR UPDATE" if lock else ""
        row = conn.execute(
            "SELECT * FROM garp_sessions WHERE id = %s AND owner = %s" + suffix,
            (str(session_id), owner),
        ).fetchone()
        if row is None:
            raise NotFound("session not found")
        return row

    def create_session(self, owner):
        session_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO garp_sessions (id, owner) VALUES (%s, %s)", (session_id, owner)
            )
        return session_id

    def get_events(self, owner, session_id, after=0):
        with self._connect() as conn:
            self._session(conn, owner, session_id)
            return conn.execute(
                "SELECT seq, kind, data FROM garp_events WHERE session_id = %s AND seq > %s ORDER BY seq LIMIT 200",
                (str(session_id), after),
            ).fetchall()

    def get_session(self, owner, session_id):
        with self._connect() as conn:
            row = self._session(conn, owner, session_id)
            return {
                "id": str(row["id"]),
                "title": row["title"],
                "model": row["model"],
                "effort": row["effort"],
                "web_search": row["web_search"],
            }

    def list_sessions(self, owner):
        with self._connect() as conn:
            return conn.execute(
                """SELECT p.id::text AS id, p.title, p.model, p.effort, p.web_search
                FROM garp_sessions p WHERE p.owner = %s
                ORDER BY p.created_at DESC, p.id LIMIT 200""",
                (owner,),
            ).fetchall()

    def set_session_options(self, owner, session_id, updates):
        with self._connect() as conn:
            row = self._session(conn, owner, session_id, lock=True)
            prior_model = row["model"] or default_model()
            model = updates.get("model", prior_model)
            same_model = model == prior_model
            effort = updates.get("effort", row["effort"] if same_model else None)
            search = updates.get("web_search", row["web_search"] if same_model else False)
            validate_options(model, effort, search)
            conn.execute(
                "UPDATE garp_sessions SET model = %s, effort = %s, web_search = %s WHERE id = %s",
                (model, effort, search, session_id),
            )
            return {"model": model, "effort": effort, "web_search": search}

    def check_session(self, owner, session_id):
        with self._connect() as conn:
            self._session(conn, owner, session_id)
            return conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]

    @staticmethod
    def _event(conn, session_id, kind, data):
        return conn.execute(
            """INSERT INTO garp_events (session_id, seq, kind, data)
            SELECT %s, COALESCE(MAX(seq), 0) + 1, %s, %s
            FROM garp_events WHERE session_id = %s RETURNING seq, kind, data""",
            (session_id, kind, Jsonb(data), session_id),
        ).fetchone()

    def create_schedule(
        self,
        owner,
        session_id,
        job_ref,
        baseline,
        due_at,
        deadline_at=None,
        *,
        native=False,
        occurrence_id=None,
        cron_claim=None,
    ):
        schedule_id = occurrence_id or str(uuid4())
        with self._connect() as conn:
            if not native:
                self._session(conn, owner, session_id, lock=True)
            if cron_claim is not None:
                table, claim = cron_claim
                lock_cron_claim(conn, table, claim, owner)
            inserted = conn.execute(
                """INSERT INTO garp_schedules
                   (id, session_id, job_ref, baseline, due_at, deadline_at, native_owner)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO NOTHING RETURNING id""",
                (
                    schedule_id,
                    session_id,
                    job_ref,
                    Jsonb(baseline),
                    due_at,
                    deadline_at,
                    owner if native else None,
                ),
            ).fetchone()
            if not inserted:
                existing = conn.execute(
                    """SELECT 1 FROM garp_schedules WHERE id = %s AND native_owner = %s
                       AND session_id = %s AND job_ref = %s AND due_at = %s
                       AND deadline_at IS NOT DISTINCT FROM %s""",
                    (schedule_id, owner, session_id, job_ref, due_at, deadline_at),
                ).fetchone()
                if not occurrence_id or not native or existing is None:
                    raise ValueError("Schedule occurrence ID collision")
        return schedule_id

    def get_schedule(self, owner, schedule_id):
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.*, coalesce(s.native_owner, p.owner) AS owner, clock_timestamp() AS now
                FROM garp_schedules s LEFT JOIN garp_sessions p ON p.id = s.session_id
                WHERE s.id = %s AND coalesce(s.native_owner, p.owner) = %s""",
                (str(schedule_id), owner),
            ).fetchone()
            if row is None:
                raise NotFound("schedule not found")
            return self._ids(row)

    @staticmethod
    def _ids(row):
        for key in ("id", "session_id", "lease_token"):
            if row.get(key) is not None:
                row[key] = str(row[key])
        return row

    def claim_due(self, lease_seconds=30, *, native=False):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._connect() as conn:
            if native:
                row = conn.execute(
                    """WITH next_due AS (
                        SELECT id FROM garp_schedules
                        WHERE native_owner IS NOT NULL AND due_at <= clock_timestamp()
                          AND (status = 'pending' OR (status = 'leased' AND lease_expires <= clock_timestamp()))
                        ORDER BY due_at, id LIMIT 1 FOR UPDATE SKIP LOCKED
                    )
                    UPDATE garp_schedules s SET status = 'leased', lease_token = %s,
                        lease_expires = clock_timestamp() + %s * interval '1 second', attempt = attempt + 1
                    FROM next_due d WHERE s.id = d.id
                    RETURNING s.*, s.native_owner AS owner, clock_timestamp() AS dispatch_at""",
                    (str(uuid4()), lease_seconds),
                ).fetchone()
                return self._ids(row) if row else None
            row = conn.execute(
                """SELECT p.id AS session_id, p.owner, s.id FROM garp_sessions p
                JOIN garp_schedules s ON s.session_id = p.id
                WHERE s.native_owner IS NULL AND s.due_at <= clock_timestamp() AND (s.status = 'pending' OR
                      (s.status = 'leased' AND s.lease_expires <= clock_timestamp()))
                  AND (p.turn_token IS NULL OR p.turn_expires <= clock_timestamp())
                  AND NOT EXISTS (SELECT 1 FROM garp_schedules active WHERE active.session_id = p.id
                      AND active.status = 'leased' AND active.lease_expires > clock_timestamp())
                ORDER BY s.due_at, s.id FOR UPDATE OF p SKIP LOCKED LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            # A competing transaction may commit between the candidate snapshot and row lock.
            valid = conn.execute(
                """SELECT 1 FROM garp_schedules s JOIN garp_sessions p ON p.id = s.session_id
                WHERE s.id = %s AND s.native_owner IS NULL AND s.due_at <= clock_timestamp() AND (s.status = 'pending' OR
                    (s.status = 'leased' AND s.lease_expires <= clock_timestamp()))
                AND (p.turn_token IS NULL OR p.turn_expires <= clock_timestamp())
                AND NOT EXISTS (SELECT 1 FROM garp_schedules active WHERE active.session_id = p.id
                    AND active.status = 'leased' AND active.lease_expires > clock_timestamp())""",
                (row["id"],),
            ).fetchone()
            if valid is None:
                return None
            token = str(uuid4())
            result = conn.execute(
                """UPDATE garp_schedules SET status = 'leased', lease_token = %s,
                lease_expires = clock_timestamp() + %s * interval '1 second', attempt = attempt + 1
                WHERE id = %s RETURNING *, clock_timestamp() AS dispatch_at""",
                (token, lease_seconds, row["id"]),
            ).fetchone()
            result["owner"] = row["owner"]
            return self._ids(result)

    def finish_schedule(self, schedule_id, lease_token, outcome, evidence):
        if outcome not in ("complete", "denied", "expired", "failed"):
            raise ValueError("invalid outcome")
        with self._connect() as conn:
            # Lock the session first, matching both admission paths.
            session = conn.execute(
                """SELECT p.id FROM garp_sessions p JOIN garp_schedules s ON s.session_id = p.id
                WHERE s.id = %s AND s.native_owner IS NULL FOR UPDATE OF p""",
                (str(schedule_id),),
            ).fetchone()
            row = conn.execute(
                """UPDATE garp_schedules SET status = %s, outcome = %s, evidence = %s,
                lease_token = NULL, lease_expires = NULL WHERE id = %s AND status = 'leased'
                AND lease_token = %s AND lease_expires > clock_timestamp()
                RETURNING session_id, native_owner""",
                (outcome, outcome, Jsonb(evidence), str(schedule_id), str(lease_token)),
            ).fetchone()
            if row is None:
                return False
            if row["native_owner"] is None:
                if session is None:
                    raise NotFound("schedule not found")
                self._event(
                    conn,
                    row["session_id"],
                    "schedule.result",
                    {"schedule_id": str(schedule_id), "outcome": outcome, "evidence": evidence},
                )
            return True

    def cancel_native_schedule(self, owner, schedule_id):
        with self._connect() as conn:
            result = conn.execute(
                """UPDATE garp_schedules s SET status = 'cancelled', outcome = 'cancelled',
                           evidence = jsonb_build_object('job_ref', s.job_ref),
                           lease_token = NULL, lease_expires = NULL
                   WHERE s.native_owner = %s AND s.id = %s
                     AND s.status IN ('pending', 'leased')
                   RETURNING s.id""",
                (owner, schedule_id),
            ).fetchone()
            return result is not None

    def pending_native_delivery(self):
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.*, s.native_owner AS owner FROM garp_schedules s
                   WHERE s.native_owner IS NOT NULL AND s.outcome IS NOT NULL AND s.delivered_at IS NULL
                          AND (s.next_delivery_at IS NULL OR s.next_delivery_at <= clock_timestamp())
                          AND s.delivery_error IS NULL ORDER BY s.due_at, s.id LIMIT 1"""
            ).fetchone()
            return self._ids(row) if row else None

    def record_native_delivery(self, schedule_id, *, error=None):
        with self._connect() as conn:
            if error:
                conn.execute(
                    """UPDATE garp_schedules SET delivery_error = %s
                       WHERE id = %s AND native_owner IS NOT NULL AND delivered_at IS NULL""",
                    (error, schedule_id),
                )
            else:
                conn.execute(
                    """UPDATE garp_schedules SET delivered_at = clock_timestamp(), delivery_error = NULL
                       WHERE id = %s AND native_owner IS NOT NULL""",
                    (schedule_id,),
                )

    def defer_native_delivery(self, schedule_id):
        with self._connect() as conn:
            conn.execute(
                """UPDATE garp_schedules SET next_delivery_at = clock_timestamp() + interval '5 seconds'
                   WHERE id = %s AND native_owner IS NOT NULL AND delivered_at IS NULL""",
                (schedule_id,),
            )

    def promote_local_schedules(self, schema, sessions_table):
        with self._connect() as conn:
            marker = sql.Identifier(schema, "garp_chat_imports")
            if not conn.execute(
                "SELECT to_regclass(%s) AS marker", (marker.as_string(conn),)
            ).fetchone()["marker"]:
                return
            conn.execute(
                sql.SQL(
                    """WITH imported AS (
                        SELECT p.id FROM garp_sessions p
                        JOIN {marker} i ON i.session_id = p.id::text
                        JOIN {sessions} n ON n.session_id = i.session_id
                        WHERE p.owner = 'local' AND n.user_id = 'local'
                            AND n.session_type = 'agent' AND n.agent_id = 'garp'
                        FOR KEY SHARE OF n
                    )
                    UPDATE garp_schedules s SET native_owner = 'local'
                    FROM imported WHERE s.session_id = imported.id AND s.native_owner IS NULL
                        AND s.status IN ('pending', 'leased')"""
                ).format(marker=marker, sessions=sql.Identifier(schema, sessions_table))
            )

    def start_turn(self, owner, session_id, text, lease_seconds=120):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._connect() as conn:
            session = self._session(conn, owner, session_id, lock=True)
            busy = conn.execute(
                """SELECT 1 FROM garp_sessions p WHERE p.id = %s AND
                ((p.turn_token IS NOT NULL AND p.turn_expires > clock_timestamp()) OR
                 EXISTS (SELECT 1 FROM garp_schedules s WHERE s.session_id = p.id AND s.status = 'leased'
                         AND s.lease_expires > clock_timestamp()))""",
                (str(session_id),),
            ).fetchone()
            if busy:
                raise RuntimeError("session busy")
            token = str(uuid4())
            conn.execute(
                """UPDATE garp_sessions SET turn_token = %s, turn_text = %s,
                turn_expires = clock_timestamp() + %s * interval '1 second' WHERE id = %s""",
                (token, text, lease_seconds, str(session_id)),
            )
            history = conn.execute(
                """SELECT CASE WHEN kind = 'turn.message' THEN data->>'role'
                ELSE 'assistant' END AS role, CASE WHEN kind = 'turn.message' THEN data->>'content'
                ELSE 'Scheduled check result: ' || data::text END AS content
                FROM garp_events WHERE session_id = %s
                AND (kind = 'turn.message' OR
                    (kind = 'schedule.result' AND data->>'outcome' = 'complete'))
                ORDER BY seq""",
                (str(session_id),),
            ).fetchall()
            return {
                "token": token,
                "history": history,
                "title": session["title"],
                "model": session["model"],
                "effort": session["effort"],
                "web_search": session["web_search"],
            }

    def finish_turn(self, owner, session_id, token, text, title=None, details=None):
        with self._connect() as conn:
            session = self._session(conn, owner, session_id, lock=True)
            row = conn.execute(
                """UPDATE garp_sessions SET turn_token = NULL, turn_expires = NULL, turn_text = NULL,
                title = COALESCE(title, %s)
                WHERE id = %s AND turn_token = %s AND turn_expires > clock_timestamp()
                RETURNING id""",
                (title if text is not None else None, str(session_id), str(token)),
            ).fetchone()
            if row is None:
                return False
            if text is None:
                return True
            return [
                self._event(
                    conn,
                    session_id,
                    "turn.message",
                    {"role": "user", "content": session["turn_text"]},
                ),
                self._event(
                    conn,
                    session_id,
                    "turn.message",
                    {"role": "assistant", "content": text, **(details or {})},
                ),
            ]
