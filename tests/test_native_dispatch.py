from concurrent.futures import ThreadPoolExecutor

import pytest
from agno.db.postgres import PostgresDb
from agno.run.agent import RunOutput
from agno.run.base import RunStatus
from agno.session import AgentSession
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import URL

from garp.native_dispatch import publish_check


def test_native_check_retries_preserve_concurrent_runs_and_session(postgres_dsn):
    url = URL.create("postgresql+psycopg", query=conninfo_to_dict(postgres_dsn))
    db = PostgresDb(db_url=url.render_as_string(hide_password=False))
    db.upsert_session(
        AgentSession(
            session_id="chat",
            agent_id="garp",
            user_id="owner",
            session_data={"session_name": "Title"},
            created_at=1,
            updated_at=1,
        )
    )
    db.upsert_session(
        AgentSession(
            session_id="other", agent_id="garp", user_id="owner", created_at=1, updated_at=1
        )
    )
    db.upsert_run(
        RunOutput(
            run_id="before",
            agent_id="garp",
            session_id="chat",
            user_id="owner",
            content="before",
            status=RunStatus.completed,
        ),
        "chat",
        user_id="owner",
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: publish_check(
                    db, "chat", "occurrence", "owner", "complete", {"state": "finished"}
                ),
                range(4),
            )
        )
    assert sorted(results) == [False, False, False, True]
    db.upsert_run(
        RunOutput(
            run_id="after",
            agent_id="garp",
            session_id="chat",
            user_id="owner",
            content="after",
            status=RunStatus.completed,
        ),
        "chat",
        user_id="owner",
    )
    session = db.get_session("chat")
    assert session.session_data["session_name"] == "Title"
    assert [run.run_id for run in session.runs][::2] == ["before", "after"]
    assert '"state": "finished"' in session.runs[1].content
    assert session.runs[1].messages[0].content == session.runs[1].content
    with pytest.raises(ValueError, match="collision"):
        publish_check(db, "other", "occurrence", "owner", "complete", {"state": "finished"})
    with pytest.raises(ValueError, match="owner mismatch"):
        publish_check(db, "chat", "new", "someone else", "complete", {})
    assert db.get_session("other").runs == []
