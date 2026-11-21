import json
import os
import secrets
import subprocess
import sys
import time
from uuid import uuid4

import httpx
import jwt
import psycopg
from agno.db.postgres import PostgresDb
from agno.run.agent import RunInput, RunOutput
from agno.run.base import RunStatus
from agno.run.team import TeamRunOutput
from agno.session import AgentSession, TeamSession
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.types.json import Jsonb
from sqlalchemy import URL

from garp.agentos import HostAuth, create_agentos_app
from garp.demo import _serve_app


def test_installed_cli_finds_and_reopens_a_saved_chat(postgres_dsn, monkeypatch):
    name = "transcript_search_" + uuid4().hex
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    dsn = make_conninfo(postgres_dsn, dbname=name)
    db = PostgresDb(db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(dsn)))
    modern, legacy, team = (str(uuid4()) for _ in range(3))
    db.upsert_session(
        AgentSession(
            session_id=modern,
            agent_id="garp",
            user_id="local",
            created_at=10,
            session_data={"session_name": "Saved Launch Notes"},
        )
    )
    db.upsert_session(
        AgentSession(session_id=legacy, agent_id="garp", user_id="local", created_at=1)
    )
    db.upsert_session(
        TeamSession(session_id=team, team_id="garp-team", user_id="local", created_at=2)
    )

    def add(run, index=0):
        db.upsert_run(run, run.session_id, user_id="local", run_index=index)

    add(
        RunOutput(
            run_id=str(uuid4()),
            agent_id="garp",
            session_id=modern,
            user_id="local",
            status=RunStatus.completed,
            input=RunInput(input_content="Find the rollout notes"),
            content="The saved cobalt lantern launch is on Friday.",
            reasoning_content="hidden chain marker",
        )
    )
    add(
        TeamRunOutput(
            run_id=str(uuid4()),
            team_id="garp-team",
            session_id=team,
            user_id="local",
            status=RunStatus.completed,
            content="The team used a cobalt lantern as well.",
        )
    )
    add(
        RunOutput(
            run_id=str(uuid4()),
            agent_id="reviewer",
            session_id=team,
            user_id="local",
            parent_run_id=str(uuid4()),
            status=RunStatus.completed,
            content="hidden member marker",
        ),
        1,
    )
    add(
        RunOutput(
            run_id=str(uuid4()),
            agent_id="garp",
            session_id=modern,
            user_id="local",
            status=RunStatus.error,
            content="hidden error marker",
        ),
        1,
    )

    shared = {
        "agent_id": "garp",
        "session_id": legacy,
        "user_id": "local",
        "status": RunStatus.completed,
    }
    old_run = RunOutput(run_id=str(uuid4()), content="Historical paper map", **shared)
    changed = RunOutput(run_id=str(uuid4()), content="Outdated shadow answer", **shared)
    add(RunOutput(run_id=changed.run_id, content="Current corrected answer", **shared), 1)
    unrelated = [str(uuid4()) for _ in range(15)]
    alice, bob = "tenant-a:owner", "tenant-b:owner"
    alice_session, bob_legacy, bob_team = (str(uuid4()) for _ in range(3))
    db.upsert_session(
        AgentSession(session_id=alice_session, agent_id="garp", user_id=alice, created_at=1)
    )
    db.upsert_run(
        RunOutput(
            run_id=str(uuid4()),
            agent_id="garp",
            session_id=alice_session,
            user_id=alice,
            status=RunStatus.completed,
            content="cross-tenant needle Axim",
        ),
        alice_session,
        user_id=alice,
        run_index=0,
    )
    db.upsert_session(
        TeamSession(session_id=bob_team, team_id="garp-team", user_id=bob, created_at=1)
    )
    db.upsert_run(
        TeamRunOutput(
            run_id=str(uuid4()),
            team_id="garp-team",
            session_id=bob_team,
            user_id=bob,
            status=RunStatus.completed,
            content="cross-tenant needle Salaga",
        ),
        bob_team,
        user_id=bob,
        run_index=0,
    )
    with psycopg.connect(dsn) as conn:
        conn.execute("ALTER TABLE ai.agno_sessions ADD COLUMN runs jsonb")
        conn.execute(
            "UPDATE ai.agno_sessions SET runs=%s WHERE session_id=%s",
            (Jsonb(json.dumps([old_run.to_dict(), changed.to_dict()])), legacy),
        )
        rows = [(sid, "local", "An unrelated historical reply") for sid in unrelated] + [
            (bob_legacy, bob, "cross-tenant needle Kpando")
        ]
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO ai.agno_sessions (session_id, session_type, agent_id, user_id, created_at, runs) "
                "VALUES (%s, 'agent', 'garp', %s, 1, %s)",
                [
                    (
                        sid,
                        owner,
                        Jsonb(
                            [
                                RunOutput(
                                    run_id=str(uuid4()),
                                    agent_id="garp",
                                    session_id=sid,
                                    user_id=owner,
                                    status=RunStatus.completed,
                                    content=body,
                                ).to_dict()
                            ]
                        ),
                    )
                    for sid, owner, body in rows
                ],
            )

    loads = []
    get_session = PostgresDb.get_session

    def record_load(self, session_id, *args, **kwargs):
        loads.append(session_id)
        return get_session(self, session_id, *args, **kwargs)

    monkeypatch.setattr(PostgresDb, "get_session", record_load)
    key = secrets.token_urlsafe(40)
    auth = HostAuth("https://search.example", "garp-search", key, "HS256")
    with TestClient(create_agentos_app(dsn, host_auth=auth)) as hosted:
        for owner, impersonated, expected in (
            (alice, bob, {alice_session}),
            (bob, alice, {bob_legacy, bob_team}),
        ):
            signed = jwt.encode(
                {
                    "sub": owner,
                    "iss": auth.issuer,
                    "aud": auth.audience,
                    "exp": int(time.time()) + 180,
                    "scopes": ["sessions:read"],
                },
                key,
                algorithm="HS256",
            )
            result = hosted.post(
                "/garp/transcripts/search",
                params={"user_id": impersonated},
                headers={"Authorization": f"Bearer {signed}"},
                json={"query": "cross-tenant needle"},
            )
            assert {
                row["session_id"] for row in result.raise_for_status().json()["matches"]
            } == expected

    token = secrets.token_urlsafe(32)
    headers = {"Authorization": f"Bearer {token}"}
    with _serve_app(create_agentos_app(dsn, token)) as base:
        with httpx.Client(base_url=base, headers=headers, trust_env=False) as client:

            def find(query, **filters):
                response = client.post("/garp/transcripts/search", json={"query": query, **filters})
                return [row["session_id"] for row in response.raise_for_status().json()["matches"]]

            assert set(find("cobalt lantern")) == {modern, team}
            assert find("cobalt lantern", session_type="team") == [team]
            assert find("rollout notes") == [modern]
            assert find("paper map") == [legacy]
            assert find("corrected answer") == [legacy]
            assert not find("shadow answer") and not find("hidden")
            assert client.delete(f"/sessions/{legacy}").status_code == 204
            assert not find("paper map")

        assert not set(unrelated).intersection(loads)

        process = subprocess.run(
            [sys.executable, "-m", "garp.cli", "--url", base, "chat"],
            input="/find COBALT LANTERN\n1\n/history\n/exit\n",
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "GARP_TOKEN": token},
            check=True,
        )
        assert "Saved Launch Notes" in process.stdout
        assert process.stdout.count("The saved cobalt lantern launch is on Friday.") == 3
        assert "team used" not in process.stdout
