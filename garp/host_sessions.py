"""Atomic hosted updates of user-editable session fields."""

import json
import time
from functools import wraps

from agno.db.schemas.scheduler import strip_reserved_run_metadata
from agno.db.utils import deserialize_session, merge_runs_table_with_legacy_blob
from agno.os.middleware.user_scope import resolve_db_and_scope
from agno.os.schema import (
    AgentSessionDetailSchema,
    TeamSessionDetailSchema,
    WorkflowSessionDetailSchema,
)
from agno.session import AgentSession, TeamSession
from agno.session.summary import SessionSummary
from agno.utils.string import sanitize_postgres_strings
from fastapi import HTTPException
from psycopg import sql
from sqlalchemy import ARRAY, Text, case, cast, func, literal, select, text, update
from sqlalchemy.dialects.postgresql import JSONB

REVISION_KEY = "garp_session_revisions"


def install_host_session_revisions(store, db):
    sessions = db._get_table("sessions", create_table_if_not_found=True)
    if sessions is None:
        raise RuntimeError("Hosted session revisions require a session table")
    with store._connect() as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext('garp.host_session_revisions'), hashtext(%s))",
            (db.db_schema,),
        )
        function_name = sql.Identifier(db.db_schema, "garp_host_session_revision_fence").as_string(
            conn
        )
        table_name = sql.Identifier(sessions.schema, sessions.name).as_string(conn)
        installed = conn.execute(
            """SELECT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgrelid = %s::regclass AND tgname = 'garp_host_session_revisions'
                  AND NOT tgisinternal AND tgenabled = 'O'
                  AND tgfoid = to_regprocedure(%s)
            ) AND obj_description(to_regprocedure(%s), 'pg_proc') = %s AS installed""",
            (
                table_name,
                function_name + "()",
                function_name + "()",
                "garp-host-session-revisions-v1",
            ),
        ).fetchone()["installed"]
        if installed:
            return
        conn.execute(
            sql.SQL("""
            CREATE OR REPLACE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $$
            DECLARE
              old_rev jsonb := coalesce(nullif(OLD.session_data -> 'garp_session_revisions', 'null'::jsonb), '{{}}'::jsonb);
              incoming jsonb := coalesce(nullif(NEW.session_data -> 'garp_session_revisions', 'null'::jsonb), '{{}}'::jsonb);
              trusted jsonb := coalesce(nullif(current_setting('garp.session_update_fields', true), ''), '[]')::jsonb;
              result jsonb := old_rev;
              k text;
            BEGIN
              FOREACH k IN ARRAY ARRAY['session_name', 'session_state', 'metadata', 'summary'] LOOP
                IF trusted ? k THEN
                  result := jsonb_set(result, ARRAY[k], to_jsonb(coalesce((old_rev ->> k)::bigint, 0) + 1), true);
                ELSIF coalesce(incoming ->> k, '0') <> coalesce(old_rev ->> k, '0') THEN
                  IF k IN ('session_name', 'session_state') THEN
                    NEW.session_data := coalesce(nullif(NEW.session_data, 'null'::jsonb), '{{}}'::jsonb) - k;
                    IF OLD.session_data ? k THEN
                      NEW.session_data := jsonb_set(NEW.session_data, ARRAY[k], OLD.session_data -> k, true);
                    END IF;
                  ELSIF k = 'metadata' THEN NEW.metadata := OLD.metadata;
                  ELSIF k = 'summary' THEN NEW.summary := OLD.summary;
                  END IF;
                END IF;
              END LOOP;
              NEW.session_data := jsonb_set(coalesce(nullif(NEW.session_data, 'null'::jsonb), '{{}}'::jsonb), '{{garp_session_revisions}}', result, true);
              RETURN NEW;
            END $$;
            CREATE OR REPLACE TRIGGER garp_host_session_revisions BEFORE UPDATE ON {sessions}
            FOR EACH ROW EXECUTE FUNCTION {function}();
            COMMENT ON FUNCTION {function}() IS 'garp-host-session-revisions-v1';
        """).format(
                function=sql.Identifier(db.db_schema, "garp_host_session_revision_fence"),
                sessions=sql.Identifier(sessions.schema, sessions.name),
            )
        )


def _redact_workflow(result):
    if isinstance(result, WorkflowSessionDetailSchema) and result.session_data:
        result.session_data = {k: v for k, v in result.session_data.items() if k != REVISION_KEY}
    return result


def attach_host_session_updates(app, db):
    def routes(router):
        for route in router.routes:
            if hasattr(route, "original_router"):
                yield from routes(route.original_router)
            else:
                yield route

    by_operation = {getattr(route, "operation_id", None): route for route in routes(app.router)}
    route = by_operation["update_session"]

    @wraps(route.endpoint)
    async def update_session(
        request, session_id, session_type, update_data, user_id, db_id, table, *, rename=False
    ):
        selected, owner = await resolve_db_and_scope(
            request, {db.id: [db]}, db_id, table, fallback_user_id=user_id
        )
        sessions = selected._get_table("sessions")
        if sessions is None:
            raise HTTPException(404, f"Session with id '{session_id}' not found")
        values = {} if rename else {"updated_at": int(time.time())}
        session_data = func.coalesce(
            func.nullif(sessions.c.session_data, literal(None, type_=JSONB)),
            cast(literal("{}"), JSONB),
        )
        for key in ("session_name", "session_state"):
            value = getattr(update_data, key)
            if value is not None:
                session_data = func.jsonb_set(
                    session_data,
                    cast(literal([key]), ARRAY(Text)),
                    cast(literal(sanitize_postgres_strings(value), type_=JSONB), JSONB),
                    True,
                )
                values["session_data"] = session_data
        if update_data.metadata is not None:
            metadata = strip_reserved_run_metadata(update_data.metadata)
            pairs = func.jsonb_each(
                func.coalesce(
                    func.nullif(sessions.c.metadata, literal(None, type_=JSONB)),
                    cast(literal("{}"), JSONB),
                )
            ).table_valued("key", "value")
            reserved = (
                select(func.jsonb_object_agg(pairs.c.key, pairs.c.value))
                .where(pairs.c.key.startswith("garp_parent_", autoescape=True))
                .scalar_subquery()
            )
            if metadata is None:
                values["metadata"] = case(
                    (reserved.is_(None), literal(None, type_=JSONB)), else_=reserved
                )
            else:
                values["metadata"] = literal(sanitize_postgres_strings(metadata), type_=JSONB).op(
                    "||"
                )(func.coalesce(reserved, cast(literal("{}"), JSONB)))
        if update_data.summary is not None:
            values["summary"] = sanitize_postgres_strings(
                SessionSummary.from_dict(update_data.summary).to_dict()
            )
        stmt = (
            update(sessions)
            .where(sessions.c.session_id == session_id, sessions.c.user_id == owner)
            .values(**values)
            .returning(sessions)
        )
        if session_type is not None:
            stmt = stmt.where(sessions.c.session_type == session_type.value)
        with selected.Session() as sess, sess.begin():
            fields = [
                key
                for key in ("session_name", "session_state", "metadata", "summary")
                if getattr(update_data, key) is not None
            ]
            sess.execute(
                text("SELECT set_config('garp.session_update_fields', :fields, true)"),
                {"fields": json.dumps(fields)},
            )
            row = sess.execute(stmt).first()
            if row is None:
                raise HTTPException(404, f"Session with id '{session_id}' not found")
            data = dict(row._mapping)
            runs = selected._get_table("runs")
            if runs is not None:
                data["runs"] = merge_runs_table_with_legacy_blob(
                    selected._get_session_runs_data(
                        sess=sess, runs_table=runs, session_id=session_id
                    ),
                    data.get("runs"),
                )
        result = deserialize_session(session_type, data)
        if isinstance(result, AgentSession):
            return AgentSessionDetailSchema.from_session(result)
        if isinstance(result, TeamSession):
            return TeamSessionDetailSchema.from_session(result)
        return _redact_workflow(WorkflowSessionDetailSchema.from_session(result))

    route.endpoint = update_session
    route.dependant.call = update_session

    rename_route = by_operation["rename_session"]

    @wraps(rename_route.endpoint)
    async def rename_session(
        request, session_id, session_type, session_name, user_id, db_id, table
    ):
        from agno.os.schema import UpdateSessionRequest

        return await update_session(
            request,
            session_id,
            session_type,
            UpdateSessionRequest(session_name=session_name),
            user_id,
            db_id,
            table,
            rename=True,
        )

    rename_route.endpoint = rename_session
    rename_route.dependant.call = rename_session

    get_route = by_operation["get_session_by_id"]
    native_get = get_route.endpoint

    @wraps(native_get)
    async def get_session(*args, **kwargs):
        return _redact_workflow(await native_get(*args, **kwargs))

    get_route.endpoint = get_session
    get_route.dependant.call = get_session
