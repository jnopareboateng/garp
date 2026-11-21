import time
from uuid import uuid4

import pytest
from agno.db.postgres import PostgresDb
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import URL

from garp.import_chat import adopt_local_preview_chats, import_local_chats
from garp.store import Store


def native(dsn):
    return PostgresDb(
        db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(dsn)).render_as_string(
            hide_password=False
        )
    )


def test_existing_native_preview_keeps_context_under_local_principal(postgres_dsn):
    from agno.models.message import Message
    from agno.run.agent import RunOutput
    from agno.run.base import RunStatus
    from agno.session import AgentSession

    db = native(postgres_dsn)
    sid, other, run_id = (str(uuid4()) for _ in range(3))
    created = int(time.time())
    db.upsert_session(
        AgentSession(
            session_id=sid,
            agent_id="garp",
            session_data={"session_name": "Preview Chat"},
            created_at=created,
            updated_at=created,
        )
    )
    run = RunOutput(
        run_id=run_id,
        agent_id="garp",
        session_id=sid,
        status=RunStatus.completed,
        messages=[Message(role="user", content="Preview context")],
    )
    assert db.append_run_to_session_if_absent(sid, run.to_dict())
    db.upsert_session(
        AgentSession(
            session_id=other,
            agent_id="garp",
            user_id="other",
            created_at=created,
            updated_at=created,
        )
    )
    adopt_local_preview_chats(db)
    adopt_local_preview_chats(db)
    session = db.get_session(sid, user_id="local")
    assert session.session_data["session_name"] == "Preview Chat"
    assert session.get_messages()[0].content == "Preview context"
    assert db.get_run(run_id, deserialize=False)["user_id"] == "local"
    assert db.get_session(other, user_id="other") is not None


def test_import_and_deletion_marker(postgres_dsn):
    store = Store(postgres_dsn)
    store.initialize()
    sid = store.create_session("local")
    other = store.create_session("other")
    schedule_id = str(uuid4())
    with store._connect() as conn:
        conn.execute(
            "UPDATE garp_sessions SET title = 'Prior', model = 'example', effort = 'high', web_search = true WHERE id = %s",
            (sid,),
        )
        for kind, data in [
            ("turn.message", {"role": "user", "content": "Question"}),
            (
                "turn.message",
                {
                    "role": "assistant",
                    "content": "Answer",
                    "model": "example",
                    "sources": [{"url": "https://example.org", "title": "Source"}],
                },
            ),
            (
                "schedule.result",
                {"schedule_id": schedule_id, "outcome": "complete", "evidence": {"ok": True}},
            ),
        ]:
            Store._event(conn, sid, kind, data)
        Store._event(conn, other, "turn.message", {"role": "user", "content": "private"})
    db = native(postgres_dsn)
    assert import_local_chats(postgres_dsn, db) == 1
    session = db.get_session(sid)
    assert session.user_id == "local"
    assert session.session_data["session_name"] == "Prior"
    assert session.session_data["session_state"]["garp_options"] == {
        "model": "example",
        "effort": "high",
        "web_search": True,
    }
    assert [m.content for m in session.get_messages()] == [
        "Question",
        "Answer",
        'Scheduled check result: {"evidence": {"ok": true}, "outcome": "complete", "schedule_id": "'
        + schedule_id
        + '"}',
    ]
    assert session.runs[0].citations.urls[0].url == "https://example.org"
    assert session.runs[0].model == "example"
    assert session.runs[1].metadata["garp_legacy_schedule_result"]["evidence"] == {"ok": True}
    assert db.get_session(other) is None
    assert import_local_chats(postgres_dsn, db) == 0
    db.delete_session(sid)
    assert import_local_chats(postgres_dsn, db) == 0
    assert db.get_session(sid) is None


def test_incomplete_event_rolls_back_and_resumes(postgres_dsn):
    store = Store(postgres_dsn)
    store.initialize()
    sid = store.create_session("local")
    with store._connect() as conn:
        Store._event(conn, sid, "turn.message", {"role": "user", "content": "orphan"})
    db = native(postgres_dsn)
    with pytest.raises(ValueError, match="Unpaired"):
        import_local_chats(postgres_dsn, db)
    assert db.get_session(sid) is None
    with store._connect() as conn:
        Store._event(conn, sid, "turn.message", {"role": "assistant", "content": "fixed"})
    assert import_local_chats(postgres_dsn, db) == 1


def test_collision_never_overwrites_native_session(postgres_dsn):
    from agno.session import AgentSession

    store = Store(postgres_dsn)
    store.initialize()
    sid = store.create_session("local")
    db = native(postgres_dsn)
    db.upsert_session(
        AgentSession(
            session_id=sid,
            agent_id="garp",
            user_id="local",
            session_data={"session_name": "Native"},
            created_at=1,
        )
    )
    with pytest.raises(RuntimeError, match="collision"):
        import_local_chats(postgres_dsn, db)
    assert db.get_session(sid).session_data["session_name"] == "Native"
    with store._connect() as conn:
        conn.execute("DELETE FROM garp_sessions WHERE id = %s", (sid,))


def test_failure_during_runs_rolls_back_session_and_marker(postgres_dsn, monkeypatch):
    from garp import import_chat

    store = Store(postgres_dsn)
    store.initialize()
    sid = store.create_session("local")
    with store._connect() as conn:
        Store._event(conn, sid, "turn.message", {"role": "user", "content": "hello"})
        Store._event(conn, sid, "turn.message", {"role": "assistant", "content": "world"})
    db = native(postgres_dsn)
    original = import_chat.build_single_run_row
    with monkeypatch.context() as patch:

        def fail(*args):
            raise RuntimeError("simulated failure")

        patch.setattr(import_chat, "build_single_run_row", fail)
        with pytest.raises(RuntimeError, match="simulated failure"):
            import_local_chats(postgres_dsn, db)
    assert db.get_session(sid) is None
    assert import_chat.build_single_run_row is original
    assert import_local_chats(postgres_dsn, db) == 1
