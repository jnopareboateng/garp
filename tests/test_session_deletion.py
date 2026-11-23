import asyncio
import json
import secrets
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import httpx
import jwt
import psycopg
import pytest
from agno.db.postgres import PostgresDb
from agno.os.event_streams.redis import RedisEventStream
from fastapi.testclient import TestClient
from psycopg.conninfo import conninfo_to_dict
from redis.asyncio import Redis
from sqlalchemy import URL
from sqlalchemy.exc import IntegrityError

from garp import agentos
from garp.session_deletion import GarpPostgresDb, ReplayPublicationFence


@pytest.mark.parametrize(("background", "delegate"), [(True, False), (False, False), (False, True)])
def test_deleted_chat_cannot_return_when_a_run_saves(
    postgres_dsn, monkeypatch, capfd, background, delegate
):
    started, release, blocked_save = Event(), Event(), Event()
    marker = "retention-in-flight-fixture"
    provider_calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)

    async def reply(request):
        payload = json.loads(request.content)
        assert request.url.host == "openrouter.ai"
        assert payload["provider"]["zdr"] is True
        last = payload["messages"][-1]
        provider_calls.append(last["content"])
        message = {"role": "assistant", "content": marker}
        finish = "stop"
        if last["role"] == "tool":
            assert last["content"] == "Specialist dispatch is unavailable."
        else:
            assert last["content"] == "check deletion"
            started.set()
            assert await asyncio.to_thread(release.wait, 15)
            if delegate:
                message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "late-child",
                            "type": "function",
                            "function": {
                                "name": "delegate_task",
                                "arguments": json.dumps({"agent_id": "reviewer", "task": marker}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish,
                    }
                ],
            },
        )

    class Fixture(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(reply), **kwargs)

    async def title(*_args, **_kwargs):
        return "Retention Fixture"

    monkeypatch.setattr(agentos, "AsyncClient", Fixture)
    monkeypatch.setattr(agentos, "generate_title", title)
    save_session = PostgresDb.upsert_session

    def watch_save(self, session, *args, **kwargs):
        try:
            return save_session(self, session, *args, **kwargs)
        except IntegrityError as exc:
            if exc.orig.diag.constraint_name == "garp_deleted_session":
                blocked_save.set()
            raise

    monkeypatch.setattr(PostgresDb, "upsert_session", watch_save)
    token = secrets.token_urlsafe(30)
    app = agentos.create_agentos_app(postgres_dsn, token)
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app, headers=headers) as client:
        sid = (
            client.post("/sessions", json={"agent_id": "garp", "user_id": "local"})
            .raise_for_status()
            .json()["session_id"]
        )
        payload = {
            "message": "check deletion",
            "session_id": sid,
            "user_id": "local",
            "background": str(background).lower(),
            "stream": "false",
        }
        keyed = {"Idempotency-Key": str(uuid4())}
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            TestClient(agentos.create_agentos_app(postgres_dsn, token), headers=headers) as second,
        ):
            request = pool.submit(client.post, "/agents/garp/runs", data=payload, headers=keyed)
            if background:
                accepted = request.result(6)
                assert accepted.status_code == 202, accepted.text
            try:
                assert started.wait(7)
                deleted = second.delete(f"/sessions/{sid}")
                assert deleted.status_code == 204, deleted.text
                assert second.get(f"/sessions/{sid}").status_code == 404
            finally:
                release.set()
            assert blocked_save.wait(7)
            result = request.result(8)
            if background:
                rid = result.json()["run_id"]
                with psycopg.connect(postgres_dsn) as conn:
                    assert conn.execute(
                        "SELECT 1 FROM ai.garp_deleted_runs WHERE run_id=%s", (rid,)
                    ).fetchone() == (1,)
                assert (
                    second.get(f"/agents/garp/runs/{rid}", params={"session_id": sid}).status_code
                    == 404
                )
            else:
                assert result.status_code == 200 and result.json()["content"] == marker
            assert second.get(f"/sessions/{sid}").status_code == 404

    if background:
        with psycopg.connect(postgres_dsn) as conn:
            conn.execute(
                "COMMENT ON FUNCTION ai.garp_fence_deleted_chat() IS 'garp chat deletion v1'"
            )
            conn.execute("DROP TRIGGER garp_fence_chat_schedule ON ai.agno_schedules")
    restarted_app = agentos.create_agentos_app(postgres_dsn, token)
    with TestClient(restarted_app, headers=headers) as restarted:
        assert restarted.get(f"/sessions/{sid}").status_code == 404
        result = restarted.post("/garp/transcripts/search", json={"query": marker})
        assert not result.raise_for_status().json()["matches"]
        replay = restarted.post("/agents/garp/runs", data=payload, headers=keyed)
        assert replay.status_code == 403, replay.text
        assert replay.json()["detail"] == "Chat is unavailable"
        if background:
            new_session_input = {
                field: value for field, value in payload.items() if field != "session_id"
            }
            replay_without_id = restarted.post(
                "/agents/garp/runs", data=new_session_input, headers=keyed
            )
            assert replay_without_id.status_code == 409, replay_without_id.text
            assert replay_without_id.json()["detail"] == "Chat is unavailable"
            with psycopg.connect(postgres_dsn) as conn:
                assert not conn.execute(
                    "SELECT 1 FROM ai.agno_jobs WHERE session_id=%s OR idempotency_key=%s",
                    (sid, keyed["Idempotency-Key"]),
                ).fetchone()
                assert conn.execute(
                    "SELECT digest FROM ai.garp_deleted_run_keys WHERE expires_at > now()"
                ).fetchone()
                assert conn.execute(
                    "SELECT obj_description('ai.garp_fence_deleted_chat()'::regprocedure, 'pg_proc')"
                ).fetchone() == ("garp chat deletion v7",)
                assert conn.execute(
                    "SELECT 1 FROM pg_trigger WHERE tgrelid='ai.agno_schedules'::regclass "
                    "AND tgname='garp_fence_chat_schedule'"
                ).fetchone() == (1,)
                conn.execute(
                    "UPDATE ai.garp_deleted_run_keys SET expires_at=now() - interval '1 day' "
                    "WHERE digest=sha256(convert_to(jsonb_build_array('local', %s::text)::text, 'UTF8'))",
                    (keyed["Idempotency-Key"],),
                )
            restarted.portal.call(restarted_app.state.queue_worker.store.cleanup_jobs, 86400)
            with psycopg.connect(postgres_dsn) as conn:
                assert not conn.execute(
                    "SELECT 1 FROM ai.garp_deleted_run_keys WHERE expires_at <= now()"
                ).fetchone()
    expected_calls = ["check deletion"]
    if delegate:
        expected_calls.append("Specialist dispatch is unavailable.")
    assert provider_calls == expected_calls
    captured = capfd.readouterr()
    logs = captured.out + captured.err
    assert "[parameters:" not in logs
    assert "check deletion" not in logs and marker not in logs


def test_deleted_chat_tombstones_native_run_before_fk_cascade(postgres_dsn):
    token = secrets.token_urlsafe(30)
    app = agentos.create_agentos_app(postgres_dsn, token)
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app, headers=headers) as client:
        sid = (
            client.post("/sessions", json={"agent_id": "garp", "user_id": "local"})
            .raise_for_status()
            .json()["session_id"]
        )
        db = GarpPostgresDb(
            db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(postgres_dsn))
        )
        run_id = str(uuid4())
        db.upsert_run(
            {
                "run_id": run_id,
                "run_type": "agent",
                "agent_id": "garp",
                "status": "COMPLETED",
            },
            sid,
            user_id="local",
            run_index=0,
        )
        assert db.delete_session(sid, user_id="local") is True
        assert db.deleted_run(run_id) is True
        late_run_id = str(uuid4())
        db.upsert_run(
            {"run_id": late_run_id, "run_type": "agent", "agent_id": "garp", "status": "COMPLETED"},
            sid,
            user_id="local",
            run_index=1,
        )
        assert db.deleted_run(late_run_id) is True
        with psycopg.connect(postgres_dsn) as conn:
            assert not conn.execute(
                "SELECT 1 FROM ai.agno_runs WHERE run_id=%s", (run_id,)
            ).fetchone()
            assert conn.execute(
                "SELECT 1 FROM ai.garp_deleted_runs WHERE run_id=%s", (run_id,)
            ).fetchone() == (1,)


def test_publication_finishes_before_delete_commits(postgres_dsn, tmp_path):
    if not shutil.which("redis-server"):
        pytest.skip("redis-server is required")
    socket = str(tmp_path / "redis.sock")
    server = subprocess.Popen(
        ["redis-server", "--port", "0", "--unixsocket", socket, "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if (tmp_path / "redis.sock").exists():
                break
            time.sleep(0.02)
        _check_publication_order(postgres_dsn, socket)
    finally:
        server.terminate()
        server.wait(timeout=5)


def _check_publication_order(postgres_dsn, socket):
    token = secrets.token_urlsafe(30)
    with TestClient(
        agentos.create_agentos_app(postgres_dsn, token),
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        sid = (
            client.post("/sessions", json={"agent_id": "garp", "user_id": "local"})
            .raise_for_status()
            .json()["session_id"]
        )
        db = GarpPostgresDb(
            db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(postgres_dsn))
        )
        db.upsert_run(
            {"run_id": "mapped-run", "run_type": "agent", "agent_id": "garp", "status": "RUNNING"},
            sid,
            user_id="local",
            run_index=0,
        )
        guard = ReplayPublicationFence(db)

        async def check():
            entered, release = asyncio.Event(), asyncio.Event()
            redis = Redis(unix_socket_path=socket)
            stream = RedisEventStream(redis)
            await stream.register_run("fresh-run")

            async def mapped_publish():
                return 42

            no_session = type("Event", (), {})()
            assert await guard("mapped-run", no_session, mapped_publish) == 42
            mismatched = type("Event", (), {"session_id": "different-session"})()
            assert await guard("mapped-run", mismatched, mapped_publish) == 42
            assert await guard("unknown-run", no_session, mapped_publish) == -1

            async def publish():
                entered.set()
                await release.wait()
                return await stream.add_event("fresh-run", event)

            event = type(
                "Event",
                (),
                {
                    "session_id": sid,
                    "event": "Message",
                    "to_dict": lambda self: {"event": "Message"},
                },
            )()
            task = asyncio.create_task(guard("fresh-run", event, publish))
            await asyncio.wait_for(entered.wait(), 5)
            deletion = asyncio.create_task(asyncio.to_thread(db.delete_session, sid, "local"))
            try:
                for _ in range(100):
                    with psycopg.connect(postgres_dsn, autocommit=True) as monitor:
                        waiting = monitor.execute(
                            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE wait_event_type = 'Lock' AND query LIKE 'DELETE FROM %agno_sessions%')"
                        ).fetchone()[0]
                    if waiting:
                        break
                    await asyncio.sleep(0.02)
                assert waiting
                assert not deletion.done()
            finally:
                release.set()
            assert await asyncio.wait_for(task, 5) == 0
            assert await asyncio.wait_for(deletion, 5) is True
            assert await guard("fresh-run", event, publish) == -1
            assert len(await stream.replay("fresh-run")) == 1
            await redis.aclose()
            engine = guard._loops[asyncio.get_running_loop()]["engines"][guard.url]
            assert engine.pool.checkedin() == 1

        async def checked():
            guard.open_loop()
            try:
                await check()
            finally:
                await guard.close_loop()

            async def unexpected_publish():
                pytest.fail("A publisher started outside the AgentOS lifespan")

            assert await guard("fresh-run", None, unexpected_publish) == -1
            assert asyncio.get_running_loop() not in guard._loops

        asyncio.run(checked())


def test_publication_waits_for_uncommitted_delete(postgres_dsn):
    token = secrets.token_urlsafe(30)
    with TestClient(
        agentos.create_agentos_app(postgres_dsn, token),
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        sid = (
            client.post("/sessions", json={"agent_id": "garp", "user_id": "local"})
            .raise_for_status()
            .json()["session_id"]
        )
        db = GarpPostgresDb(
            db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(postgres_dsn))
        )
        guard = ReplayPublicationFence(db)

        async def check():
            published = []

            async def publish():
                published.append(True)
                return 0

            event = type("Event", (), {"session_id": sid})()
            with psycopg.connect(postgres_dsn) as conn:
                conn.execute("DELETE FROM ai.agno_sessions WHERE session_id = %s", (sid,))
                task = asyncio.create_task(guard("unknown-run", event, publish))
                for _ in range(100):
                    with psycopg.connect(postgres_dsn, autocommit=True) as monitor:
                        waiting = monitor.execute(
                            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE wait_event_type = 'Lock' AND query LIKE 'SELECT session_id FROM %FOR KEY SHARE%')"
                        ).fetchone()[0]
                    if waiting:
                        break
                    await asyncio.sleep(0.02)
                assert waiting
                assert not task.done()
                conn.commit()
                assert await asyncio.wait_for(task, 5) == -1
            assert not published

        async def checked():
            guard.open_loop()
            try:
                await check()
            finally:
                await guard.close_loop()

        asyncio.run(checked())


def test_host_bulk_deletion_removes_only_owned_sessions_and_tickets(postgres_dsn):
    key = secrets.token_urlsafe(40)
    auth = agentos.HostAuth("https://fixture.example", "deletion", key, "HS256")

    def identity(owner):
        encoded = jwt.encode(
            {
                "sub": owner,
                "iss": auth.issuer,
                "aud": auth.audience,
                "exp": int(time.time()) + 120,
                "scopes": ["sessions:write", "sessions:read", "sessions:delete"],
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": f"Bearer {encoded}"}

    alice, bob = "delete-tenant:alice", "other-tenant:bob"
    owners = (alice, alice, bob)
    with TestClient(agentos.create_agentos_app(postgres_dsn, host_auth=auth)) as client:
        chats = [
            client.post("/sessions", headers=identity(owner), json={"agent_id": "garp"})
            .raise_for_status()
            .json()["session_id"]
            for owner in owners
        ]
        # Keep queue rows ahead of their admission time so the background worker leaves them alone.
        payload = '{"input":"fixture marker"}'
        with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO ai.agno_jobs (id, component_id, component_type, job_type, session_id,
                       user_id, status, payload, available_at, created_at, attempt, max_attempts)
                       VALUES (%s, 'garp', 'agent', 'run', %s, %s, 'queued', %s::jsonb, %s, %s, 0, 1)""",
                [
                    (str(uuid4()), sid, owner, payload, int(time.time()) + 3600, int(time.time()))
                    for sid, owner in zip(chats, owners, strict=True)
                ],
            )
        bulk = {"session_ids": chats, "session_types": ["agent"] * len(chats)}
        for _ in range(2):
            result = client.request("DELETE", "/sessions", json=bulk, headers=identity(alice))
            assert result.status_code == 204, result.text
        for sid in chats[:2]:
            assert client.get(f"/sessions/{sid}", headers=identity(alice)).status_code == 404
        assert client.get(f"/sessions/{chats[2]}", headers=identity(bob)).status_code == 200
        with psycopg.connect(postgres_dsn) as conn:
            marked = conn.execute(
                "SELECT session_id FROM ai.garp_deleted_sessions WHERE session_id = ANY(%s)",
                (chats,),
            ).fetchall()
            tickets = conn.execute(
                "SELECT session_id FROM ai.agno_jobs WHERE session_id = ANY(%s)", (chats,)
            ).fetchall()
        assert {row[0] for row in marked} == set(chats[:2])
        assert tickets == [(chats[2],)]


@pytest.mark.parametrize(
    "operation", ["session", "job", "job-key", "job-child", "schedule", "approval"]
)
def test_native_write_waiting_on_delete_is_fenced(postgres_dsn, operation):
    token = secrets.token_urlsafe(30)
    with TestClient(
        agentos.create_agentos_app(postgres_dsn, token),
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        sid = (
            client.post("/sessions", json={"agent_id": "garp", "user_id": "local"})
            .raise_for_status()
            .json()["session_id"]
        )
        name = "garp-deletion-race-" + uuid4().hex
        key = str(uuid4()) if operation == "job-key" else None
        queue_insert = """INSERT INTO ai.agno_jobs (id, component_id, component_type, job_type,
                          session_id, user_id, status, payload, available_at, created_at, attempt,
                          max_attempts, idempotency_key)
                          VALUES (%s,'garp','agent','run',%s,'local','queued',%s::jsonb,%s,1,0,1,%s)"""
        if key:
            with psycopg.connect(postgres_dsn) as conn:
                conn.execute(queue_insert, (str(uuid4()), sid, "{}", int(time.time()) + 3600, key))

        def writer():
            with psycopg.connect(postgres_dsn, application_name=name) as conn:
                if operation == "session":
                    conn.execute(
                        """INSERT INTO ai.agno_sessions (session_id, session_type, agent_id, user_id, created_at)
                           VALUES (%s, 'agent', 'garp', 'local', 1)
                           ON CONFLICT (session_id) DO UPDATE SET updated_at=1""",
                        (sid,),
                    )
                elif operation == "approval":
                    conn.execute(
                        "INSERT INTO ai.agno_approvals (id, run_id, session_id, status, created_at) "
                        "VALUES (%s, %s, %s, 'pending', 1)",
                        (str(uuid4()), str(uuid4()), sid),
                    )
                elif operation == "schedule":
                    conn.execute(
                        """INSERT INTO ai.agno_schedules
                            (id, name, endpoint, method, payload, enabled, cron_expr, timezone,
                             max_retries, retry_delay_seconds, created_at, updated_at, user_id)
                           VALUES (%s, 'Fixture', %s, 'POST', '{}', true, '', 'UTC', 0, 1, 1, 1,
                                   'local')""",
                        (str(uuid4()), f"/garp/sessions/{sid}/checks?source=fixture"),
                    )
                else:
                    delegated = operation == "job-child"
                    metadata = {"garp_parent_session_id": sid} if delegated else {}
                    payload = json.dumps({"kwargs": {"metadata": metadata}})
                    conn.execute(
                        queue_insert,
                        (
                            str(uuid4()),
                            str(uuid4()) if key or delegated else sid,
                            payload,
                            int(time.time()) + 3600,
                            key,
                        ),
                    )

        with ThreadPoolExecutor(max_workers=1) as pool, psycopg.connect(postgres_dsn) as removing:
            removing.execute("DELETE FROM ai.agno_sessions WHERE session_id=%s", (sid,))
            future = pool.submit(writer)
            try:
                with psycopg.connect(postgres_dsn, autocommit=True) as monitor:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        if monitor.execute(
                            "SELECT wait_event_type FROM pg_stat_activity WHERE application_name=%s",
                            (name,),
                        ).fetchone() == ("Lock",):
                            break
                        time.sleep(0.05)
                    else:
                        pytest.fail("The writer did not wait on the deleting transaction")
            finally:
                removing.commit()
            with pytest.raises(psycopg.errors.CheckViolation) as error:
                future.result(5)
            expected = "garp_deleted_run_key" if key else "garp_deleted_session"
            assert error.value.diag.constraint_name == expected
        assert client.get(f"/sessions/{sid}").status_code == 404


def test_host_approval_records_follow_bulk_deletion(postgres_dsn):
    key = secrets.token_urlsafe(40)
    auth = agentos.HostAuth("https://fixture.example", "approval-deletion", key, "HS256")

    def identity(owner, scopes):
        return {
            "Authorization": "Bearer "
            + jwt.encode(
                {
                    "sub": owner,
                    "iss": auth.issuer,
                    "aud": auth.audience,
                    "exp": int(time.time()) + 120,
                    "scopes": scopes,
                },
                key,
                algorithm="HS256",
            )
        }

    owner, peer = "approval-tenant:owner", "other-tenant:owner"
    owner_headers = identity(
        owner,
        ["sessions:write", "sessions:read", "sessions:delete", "approvals:read", "agents:garp:run"],
    )
    peer_headers = identity(peer, ["sessions:write", "sessions:read"])
    reviewer = identity("approval-tenant:reviewer", ["approvals:write"])
    from garp.extensions import Extensions

    def record_note(note: str) -> str:
        """Record a note."""
        return note

    app = agentos.create_agentos_app(
        postgres_dsn,
        host_auth=auth,
        extensions={"approval-tenant": Extensions(approval_tools=(record_note,))},
    )
    with TestClient(app) as client:
        chats = [
            client.post("/sessions", headers=h, json={"agent_id": "garp"})
            .raise_for_status()
            .json()["session_id"]
            for h in (owner_headers, owner_headers, peer_headers)
        ]
        ids = [str(uuid4()) for _ in range(4)]
        runs = [str(uuid4()) for _ in ids]
        with psycopg.connect(postgres_dsn) as conn:
            for approval_id, run, sid, user, status in zip(
                ids,
                runs,
                [chats[0], chats[0], chats[1], chats[2]],
                [owner, owner, owner, peer],
                ["pending", "approved", "pending", "pending"],
                strict=True,
            ):
                conn.execute(
                    """INSERT INTO ai.agno_approvals
                    (id, run_id, session_id, user_id, agent_id, status, source_type, approval_type, pause_type, created_at, tool_args)
                    VALUES (%s, %s, %s, %s, 'garp', %s, 'agent', 'required', 'approval', 1, '{"secret":"fixture"}'::jsonb)""",
                    (approval_id, run, sid, user, status),
                )
        assert client.get("/garp/approvals", headers=reviewer).json()["data"]
        result = client.request(
            "DELETE",
            "/sessions",
            headers=owner_headers,
            json={"session_ids": [chats[0], chats[2]], "session_types": ["agent", "agent"]},
        )
        assert result.status_code == 204, result.text
        assert (
            client.get("/approvals", headers=owner_headers, params={"run_id": runs[0]}).json()[
                "data"
            ]
            == []
        )
        assert {
            row["id"] for row in client.get("/garp/approvals", headers=reviewer).json()["data"]
        } == {ids[2]}
        path = f"/garp/approvals/{ids[0]}"
        assert client.get(path, headers=reviewer).status_code == 404
        assert (
            client.post(
                path + "/resolve", headers=reviewer, json={"status": "approved"}
            ).status_code
            == 404
        )
        assert client.post(path + "/continue", headers=owner_headers).status_code == 404
        with psycopg.connect(postgres_dsn) as conn:
            rows = conn.execute(
                "SELECT id FROM ai.agno_approvals WHERE id = ANY(%s)", (ids,)
            ).fetchall()
        assert {row[0] for row in rows} == set(ids[2:])


def test_upgrade_removes_approvals_for_previously_deleted_chats(postgres_dsn):
    token = secrets.token_urlsafe(30)
    app = agentos.create_agentos_app(postgres_dsn, token)
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as client:
        sid = client.post("/sessions", json={"agent_id": "garp", "user_id": "local"}).json()[
            "session_id"
        ]
        assert client.delete(f"/sessions/{sid}").status_code == 204
    approval_id = str(uuid4())
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("DROP TRIGGER garp_fence_chat_approval ON ai.agno_approvals")
        conn.execute("COMMENT ON FUNCTION ai.garp_fence_deleted_chat() IS 'garp chat deletion v3'")
        conn.execute(
            """INSERT INTO ai.agno_approvals
            (id, run_id, session_id, status, source_type, approval_type, pause_type, created_at)
            VALUES (%s, %s, %s, 'pending', 'agent', 'required', 'approval', 1)""",
            (approval_id, str(uuid4()), sid),
        )
    with TestClient(agentos.create_agentos_app(postgres_dsn, token)):
        pass
    with psycopg.connect(postgres_dsn) as conn:
        assert not conn.execute(
            "SELECT 1 FROM ai.agno_approvals WHERE id = %s", (approval_id,)
        ).fetchone()


def test_approval_can_precede_first_session_save(postgres_dsn):
    token = secrets.token_urlsafe(30)
    with (
        TestClient(agentos.create_agentos_app(postgres_dsn, token)),
        psycopg.connect(postgres_dsn) as conn,
    ):
        sid, approval_id = str(uuid4()), str(uuid4())
        conn.execute(
            """INSERT INTO ai.agno_approvals
            (id, run_id, session_id, status, source_type, approval_type, pause_type, created_at)
            VALUES (%s, %s, %s, 'pending', 'agent', 'required', 'approval', 1)""",
            (approval_id, str(uuid4()), sid),
        )
        assert conn.execute(
            "SELECT 1 FROM ai.agno_approvals WHERE id=%s", (approval_id,)
        ).fetchone()
