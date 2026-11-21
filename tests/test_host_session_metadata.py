"""Hosted session mutation contract against two app instances."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import jwt
import psycopg
from fastapi.testclient import TestClient
from sqlalchemy import event

from garp.agentos import HostAuth, create_agentos_app


def test_host_session_tenancy_and_reserved_metadata(postgres_dsn):
    import time

    key = "fixture-key-longer-than-32-bytes-for-jwt"
    auth = HostAuth("https://fixture.local", "metadata", key, "HS256")

    def headers(subject):
        token = jwt.encode(
            {
                "iss": auth.issuer,
                "aud": auth.audience,
                "sub": subject,
                "exp": int(time.time()) + 120,
                "scopes": ["sessions:read", "sessions:write"],
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": f"Bearer {token}"}

    alice, peer = headers("tenant:alice"), headers("tenant:peer")
    with TestClient(create_agentos_app(postgres_dsn, host_auth=auth)) as client:
        response = client.post(
            "/sessions",
            headers=alice,
            json={"agent_id": "garp", "metadata": {"garp_parent_run_id": "forged"}},
        )
        assert response.status_code == 422
        response = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
        sid = response.json()["session_id"]
        path = f"/sessions/{sid}"
        for body in (
            {"metadata": {"garp_parent_session_id": str(uuid4())}},
            {"metadata": {"garp_parent_run_id": "fake"}},
        ):
            assert client.patch(path, headers=alice, json=body).status_code == 422
        assert (
            client.patch(
                path, headers=peer, json={"session_name": "stolen", "user_id": "tenant:peer"}
            ).status_code
            == 404
        )
        assert (
            client.patch(
                path, headers=alice, json={"session_name": "mine", "user_id": "tenant:peer"}
            ).status_code
            == 200
        )
        assert client.get(path, headers=peer).status_code == 404
        assert (
            client.post(path + "/rename", headers=peer, json={"session_name": "stolen"}).status_code
            == 404
        )
        assert (
            client.post(path + "/rename", headers=alice, json={"session_name": []}).status_code
            == 422
        )
        assert client.get(path, headers=alice).status_code == 200
        assert client.patch(path, headers=alice, json={"session_state": []}).status_code == 422
        assert (
            client.patch(
                path + "?type=workflow", headers=alice, json={"session_name": "x"}
            ).status_code
            == 404
        )
        assert client.patch(path + "?db_id=unknown", headers=alice, json={}).status_code == 404
        response = client.patch(
            path,
            headers=alice,
            json={
                "session_name": "updated",
                "session_state": {"garp_options": {"model": "example"}},
                "metadata": {"tag": "one"},
                "summary": {"summary": "summary"},
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["session_name"] == "updated"
        assert data["session_state"] == {"garp_options": {"model": "example"}}
        assert data["metadata"] == {"tag": "one"}
        assert data["session_summary"]["summary"] == "summary"
        replacement = client.patch(path, headers=alice, json={"metadata": {"other": True}})
        assert replacement.json()["metadata"] == {"other": True}


def test_concurrent_host_session_updates_preserve_unrelated_fields(postgres_dsn, monkeypatch):
    import time

    key = "fixture-key-longer-than-32-bytes-for-jwt"
    auth = HostAuth("https://fixture.local", "metadata", key, "HS256")
    token = jwt.encode(
        {
            "iss": auth.issuer,
            "aud": auth.audience,
            "sub": "tenant:alice",
            "exp": int(time.time()) + 120,
            "scopes": ["sessions:read", "sessions:write"],
        },
        key,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}
    with (
        TestClient(create_agentos_app(postgres_dsn, host_auth=auth)) as a,
        TestClient(create_agentos_app(postgres_dsn, host_auth=auth)) as b,
    ):
        sid = a.post("/sessions", headers=headers, json={"agent_id": "garp"}).json()["session_id"]
        barrier = Barrier(2, timeout=10)

        def gate(conn, cursor, statement, parameters, context, executemany):
            if statement.startswith("UPDATE ai.agno_sessions SET") and (
                "session_data" in statement or "metadata" in statement
            ):
                barrier.wait()

        # Each instance has its own engine. Wait immediately before both SQL updates.
        # Listen globally on Engine so both instance pools meet at the barrier.
        from sqlalchemy.engine import Engine

        event.listen(Engine, "before_cursor_execute", gate)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(
                    a.patch, f"/sessions/{sid}", headers=headers, json={"session_name": "renamed"}
                )
                second = pool.submit(
                    b.patch, f"/sessions/{sid}", headers=headers, json={"metadata": {"tag": "kept"}}
                )
                assert first.result().status_code == second.result().status_code == 200
        finally:
            event.remove(Engine, "before_cursor_execute", gate)
        with psycopg.connect(postgres_dsn) as conn:
            row = conn.execute(
                "SELECT session_data, metadata FROM ai.agno_sessions WHERE session_id=%s", (sid,)
            ).fetchone()
        assert (row[0] or {}).get("session_name") == "renamed" and row[1] == {"tag": "kept"}, row


def test_host_metadata_update_preserves_existing_system_metadata(postgres_dsn):
    import time

    key = "fixture-key-longer-than-32-bytes-for-jwt"
    auth = HostAuth("https://fixture.local", "metadata", key, "HS256")
    token = jwt.encode(
        {
            "iss": auth.issuer,
            "aud": auth.audience,
            "sub": "tenant:alice",
            "exp": int(time.time()) + 120,
            "scopes": ["sessions:read", "sessions:write"],
        },
        key,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(create_agentos_app(postgres_dsn, host_auth=auth)) as client:
        sid = client.post("/sessions", headers=headers, json={"agent_id": "garp"}).json()[
            "session_id"
        ]
        with psycopg.connect(postgres_dsn) as conn:
            conn.execute(
                "UPDATE ai.agno_sessions SET metadata = %s::json WHERE session_id = %s",
                ('{"garp_parent_run_id":"system-owned"}', sid),
            )
        response = client.patch(
            f"/sessions/{sid}", headers=headers, json={"metadata": {"tag": "changed"}}
        )
        assert response.status_code == 200, response.text
        with psycopg.connect(postgres_dsn) as conn:
            metadata = conn.execute(
                "SELECT metadata FROM ai.agno_sessions WHERE session_id = %s", (sid,)
            ).fetchone()[0]
        assert metadata["garp_parent_run_id"] == "system-owned", metadata


def test_worker_saves_preserve_host_edits(postgres_dsn):
    import time

    from agno.agent import Agent
    from agno.agent._session import save_session
    from agno.db.postgres import PostgresDb
    from psycopg.conninfo import conninfo_to_dict
    from sqlalchemy import URL

    key = "fixture-key-longer-than-32-bytes-for-jwt"
    auth = HostAuth("https://fixture.local", "metadata", key, "HS256")
    token = jwt.encode(
        {
            "iss": auth.issuer,
            "aud": auth.audience,
            "sub": "tenant:alice",
            "exp": int(time.time()) + 120,
            "scopes": ["sessions:read", "sessions:write"],
        },
        key,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}
    db = PostgresDb(db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(postgres_dsn)))
    with TestClient(create_agentos_app(postgres_dsn, host_auth=auth)) as client:
        sid = client.post("/sessions", headers=headers, json={"agent_id": "garp"}).json()[
            "session_id"
        ]
        stale = db.get_session(sid, user_id="tenant:alice")
        patch = client.patch(
            f"/sessions/{sid}",
            headers=headers,
            json={
                "session_name": "host",
                "session_state": {"host": True},
                "metadata": {"host": True},
                "summary": {"summary": "host"},
            },
        )
        assert patch.status_code == 200, patch.text
        stale.session_data = {"session_state": {"stale": True}, "other": "native"}
        stale.metadata = {"stale": True}
        save_session(Agent(db=db), stale)
        current = db.get_session(sid, user_id="tenant:alice")
        assert current.metadata == {"host": True}
        assert current.summary.summary == "host"
        assert current.session_data["session_name"] == "host"
        assert current.session_data["session_state"] == {"host": True}
        assert current.session_data["other"] == "native"
        current.metadata = {"native": 1}
        current.session_data["session_state"] = {"native": 1}
        db.upsert_session(current)
        current.metadata = {"native": 2}
        current.session_data["session_state"] = {"native": 2}
        db.upsert_session(current)
        assert db.get_session(sid, user_id="tenant:alice").metadata == {"native": 2}
        stale = db.get_session(sid, user_id="tenant:alice")
        renamed = client.post(
            f"/sessions/{sid}/rename", headers=headers, json={"session_name": "renamed"}
        )
        assert renamed.status_code == 200, renamed.text
        stale.session_data["session_state"] = {"native": 3}
        save_session(Agent(db=db), stale)
        result = db.get_session(sid, user_id="tenant:alice")
        assert result.session_data["session_name"] == "renamed"
        assert result.session_data["session_state"] == {"native": 3}
        assert result.metadata == {"native": 2}


def test_workflow_revision_hidden_and_state_not_trusted(postgres_dsn):
    import time

    key = "fixture-key-longer-than-32-bytes-for-jwt"
    auth = HostAuth("https://fixture.local", "metadata", key, "HS256")
    token = jwt.encode(
        {
            "iss": auth.issuer,
            "aud": auth.audience,
            "sub": "tenant:alice",
            "exp": int(time.time()) + 120,
            "scopes": ["sessions:read", "sessions:write"],
        },
        key,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(create_agentos_app(postgres_dsn, host_auth=auth)) as client:
        sid = client.post(
            "/sessions?type=workflow", headers=headers, json={"workflow_id": "w"}
        ).json()["session_id"]
        state = {"garp_session_revisions": {"session_state": 999}}
        response = client.patch(
            f"/sessions/{sid}?type=workflow", headers=headers, json={"session_state": state}
        )
        assert response.status_code == 200, response.text
        assert response.json()["session_state"] == state
        assert "garp_session_revisions" not in response.json()["session_data"]
        response = client.get(f"/sessions/{sid}?type=workflow", headers=headers)
        assert "garp_session_revisions" not in response.json()["session_data"]
        response = client.post(
            f"/sessions/{sid}/rename?type=workflow",
            headers=headers,
            json={"session_name": "renamed"},
        )
        assert "garp_session_revisions" not in response.json()["session_data"]
        with psycopg.connect(postgres_dsn) as conn:
            data = conn.execute(
                "SELECT session_data FROM ai.agno_sessions WHERE session_id = %s", (sid,)
            ).fetchone()[0]
        assert data["garp_session_revisions"]["session_state"] == 1
        assert data["garp_session_revisions"]["session_name"] == 1


def test_revision_install_is_safe_across_concurrent_starts(postgres_dsn):
    key = "fixture-key-longer-than-32-bytes-for-jwt"
    auth = HostAuth("https://fixture.local", "metadata", key, "HS256")
    first = create_agentos_app(postgres_dsn, host_auth=auth)
    with TestClient(first):
        pass
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("DROP TRIGGER garp_host_session_revisions ON ai.agno_sessions")
    barrier = Barrier(2, timeout=15)

    def start():
        app = create_agentos_app(postgres_dsn, host_auth=auth)
        barrier.wait()
        with TestClient(app):
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: start(), range(2)))
    with psycopg.connect(postgres_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM pg_trigger WHERE tgrelid='ai.agno_sessions'::regclass AND tgname='garp_host_session_revisions' AND tgenabled='O'"
        ).fetchone()[0]
    assert count == 1
