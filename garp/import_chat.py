"""One-time import of committed local chats; call only under the workspace lock.

Pending turns and schedule occurrences are not imported. The legacy writer must be
stopped before this runs. Each destination session and its marker commit together.
"""

import json
from uuid import NAMESPACE_URL, uuid4, uuid5

import psycopg
from agno.db.postgres import PostgresDb
from agno.db.utils import build_single_run_row
from agno.models.message import Citations, Message
from agno.run.agent import RunInput, RunOutput
from agno.run.base import RunStatus
from psycopg.rows import dict_row
from sqlalchemy import Column, MetaData, String, Table, select, text


def _runs(session_id, events, created_at):
    runs = []
    pending = None
    for event in events:
        kind, data = event["kind"], event["data"]
        if kind == "turn.message":
            if (
                not isinstance(data, dict)
                or data.get("role") not in ("user", "assistant")
                or not isinstance(data.get("content"), str)
            ):
                raise ValueError(f"Invalid legacy message in {session_id}, sequence {event['seq']}")
            if data["role"] == "user":
                if pending is not None:
                    raise ValueError(f"Unpaired legacy message in {session_id}")
                pending = data
                continue
            if pending is None:
                raise ValueError(f"Unpaired legacy message in {session_id}")
            citations = Citations(urls=data["sources"]) if data.get("sources") else None
            messages = [
                Message(role="user", content=pending["content"]),
                Message(role="assistant", content=data["content"], citations=citations),
            ]
            run = RunOutput(
                run_id=str(uuid4()),
                agent_id="garp",
                session_id=session_id,
                user_id="local",
                input=RunInput(input_content=pending["content"]),
                content=data["content"],
                model=data.get("model"),
                citations=citations,
                messages=messages,
                metadata={"garp_legacy_user": pending, "garp_legacy_message": data},
                status=RunStatus.completed,
                created_at=created_at,
            )
            pending = None
        elif kind == "schedule.result":
            if (
                pending is not None
                or not isinstance(data, dict)
                or not isinstance(data.get("schedule_id"), str)
            ):
                raise ValueError(f"Invalid legacy schedule result in {session_id}")
            content = "Scheduled check result: " + json.dumps(
                data, ensure_ascii=False, sort_keys=True
            )
            run = RunOutput(
                run_id=str(uuid5(NAMESPACE_URL, f"garp:schedule:{data['schedule_id']}")),
                agent_id="garp",
                session_id=session_id,
                user_id="local",
                content=content,
                messages=[Message(role="assistant", content=content)],
                metadata={"garp_legacy_schedule_result": data},
                status=RunStatus.completed,
                created_at=created_at,
            )
        else:
            raise ValueError(f"Unknown legacy event {kind!r} in {session_id}")
        runs.append(run)
    if pending is not None:
        raise ValueError(f"Unpaired legacy message in {session_id}")
    return runs


def import_local_chats(dsn: str, db: PostgresDb) -> int:
    """Import committed local chats once, without altering legacy data or overwriting native sessions."""
    with psycopg.connect(dsn, row_factory=dict_row) as source:
        source.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        presence = source.execute(
            "SELECT to_regclass('garp_sessions') AS sessions, to_regclass('garp_events') AS events"
        ).fetchone()
        if not presence["sessions"] and not presence["events"]:
            return 0
        if not presence["sessions"] or not presence["events"]:
            raise RuntimeError("Incomplete legacy chat schema")
        # Prove both connections point at the same database before writing native rows.
        lock_key = uuid4().int % (2**63)
        source.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
        source_db = source.execute("SELECT current_database() AS name").fetchone()["name"]
        with db.Session() as target, target.begin():
            target_db, acquired = target.execute(
                text("SELECT current_database(), pg_try_advisory_xact_lock(:key)"),
                {"key": lock_key},
            ).one()
            if target_db != source_db or acquired:
                raise ValueError("Legacy and native stores must use the same Postgres database")
        sessions = source.execute(
            "SELECT id, title, created_at, model, effort, web_search FROM garp_sessions WHERE owner = 'local' ORDER BY created_at, id"
        ).fetchall()
        if not sessions:
            return 0
        session_table = db._get_table(table_type="sessions", create_table_if_not_found=True)
        runs_table = db._get_table(table_type="runs", create_table_if_not_found=True)
        if session_table is None or runs_table is None:
            raise RuntimeError("Unable to initialize native chat tables")
        marker = Table(
            "garp_chat_imports",
            MetaData(schema=db.db_schema),
            Column("session_id", String, primary_key=True),
        )
        marker.create(db.db_engine, checkfirst=True)
        count = 0
        for row in sessions:
            sid = str(row["id"])
            with db.Session() as target, target.begin():
                if target.execute(
                    select(marker.c.session_id).where(marker.c.session_id == sid)
                ).first():
                    continue
                if target.execute(
                    select(session_table.c.session_id).where(session_table.c.session_id == sid)
                ).first():
                    raise RuntimeError(f"Native session collision: {sid}")
                events = source.execute(
                    "SELECT seq, kind, data FROM garp_events WHERE session_id = %s ORDER BY seq",
                    (row["id"],),
                ).fetchall()
                created_at = int(row["created_at"].timestamp())
                runs = _runs(sid, events, created_at)
                options = {
                    "model": row["model"],
                    "effort": row["effort"],
                    "web_search": row["web_search"],
                }
                target.execute(
                    session_table.insert().values(
                        session_id=sid,
                        session_type="agent",
                        agent_id="garp",
                        user_id="local",
                        session_data={
                            "session_name": row["title"],
                            "session_state": {"garp_options": options},
                        },
                        created_at=created_at,
                        updated_at=created_at,
                    )
                )
                for index, run in enumerate(runs):
                    target.execute(
                        runs_table.insert().values(**build_single_run_row(run, sid, "local", index))
                    )
                target.execute(marker.insert().values(session_id=sid))
            count += 1
        return count


def adopt_local_preview_chats(db: PostgresDb) -> None:
    """Assign unowned preview chats to the single-user workspace principal."""
    sessions = db._get_table(table_type="sessions")
    if sessions is None:
        return
    runs = db._get_table(table_type="runs")
    with db.Session() as target, target.begin():
        adopted = (
            target.execute(
                sessions.update()
                .where(
                    sessions.c.session_type == "agent",
                    sessions.c.agent_id == "garp",
                    sessions.c.user_id.is_(None),
                )
                .values(user_id="local")
                .returning(sessions.c.session_id)
            )
            .scalars()
            .all()
        )
        if adopted and runs is not None:
            target.execute(
                runs.update()
                .where(runs.c.session_id.in_(adopted), runs.c.user_id.is_(None))
                .values(user_id="local")
            )
