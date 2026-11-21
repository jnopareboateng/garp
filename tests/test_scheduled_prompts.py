import asyncio
import json
import secrets
import time
from collections import Counter
from threading import Event
from uuid import uuid4

import httpx
import jwt
import psycopg
import pytest
from agno.db.postgres import PostgresDb
from agno.db.schemas.scheduler import Schedule
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from garp import agentos
from garp.agentos import HostAuth, create_agentos_app
from garp.cli import main
from garp.demo import _serve_app
from garp.scheduled_prompts import PromptScheduleExecutor

OWNER = "prompt-org:owner"
OTHER = "another-org:user"
MODEL = "deepseek/deepseek-v4.1-flash-1017"
SCOPES = [
    "agents:garp:run",
    "agents:read",
    "sessions:read",
    "sessions:write",
    "sessions:delete",
    "memories:read",
    "schedules:read",
    "schedules:write",
    "schedules:delete",
]


@pytest.fixture
def service(postgres_dsn, monkeypatch):
    name = "scheduled_prompts_" + uuid4().hex
    with psycopg.connect(postgres_dsn, autocommit=True) as db:
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    dsn = make_conninfo(postgres_dsn, dbname=name)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", MODEL)
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)
    counts = Counter()
    started, released = Event(), Event()

    async def respond(request):
        body = json.loads(request.content)
        assert request.url.host == "openrouter.ai"
        assert body["provider"] == {"zdr": True, "only": ["fireworks/us"], "allow_fallbacks": False}
        prompt = body["messages"][-1]["content"]
        counts[prompt] += 1
        if prompt == "Occupied chat":
            started.set()
            assert await asyncio.to_thread(released.wait, 35)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"Reply to {prompt}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    async def title(*args, **kwargs):
        return "Scheduled Prompt Fixture"

    monkeypatch.setattr(agentos, "AsyncClient", Provider)
    monkeypatch.setattr(agentos, "generate_title", title)
    yield dsn, counts, started, released
    released.set()


@pytest.fixture
def host(service):
    dsn, *_ = service
    key = secrets.token_urlsafe(40)
    auth = HostAuth("https://host.example", "scheduled-prompts", key, "HS256")
    grants = {OWNER: SCOPES, OTHER: SCOPES}

    def headers(subject=OWNER):
        token = jwt.encode(
            {
                "iss": auth.issuer,
                "aud": auth.audience,
                "sub": subject,
                "exp": int(time.time()) + 180,
                "scopes": SCOPES,
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": "Bearer " + token}

    def create():
        return create_agentos_app(
            dsn, host_auth=auth, scheduled_scopes=lambda owner: grants.get(owner, ())
        )

    return create, headers, grants


def eventually(fetch, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = fetch()
        if response.status_code == 200 and predicate(response.json()):
            return response.json()
        time.sleep(0.1)
    pytest.fail(f"Condition not met: {response.status_code} {response.text}")


def advance(dsn, schedule, tick=None):
    tick = tick or int(time.time()) - 2
    with psycopg.connect(dsn) as db:
        db.execute("UPDATE ai.garp_host_schedules SET next_run_at=%s WHERE id=%s", (tick, schedule))
    return tick


def schedule(client, headers, **body):
    response = client.post("/garp/prompt-schedules", headers=headers, json=body)
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize("retry_by_restart", [True, False])
def test_deleted_chat_schedules_retry_cleanup_without_waiting_on_a_claim(
    service, host, retry_by_restart
):
    dsn, calls, *_ = service
    create, headers, _ = host
    alice, bob = headers(), headers(OTHER)
    app = create()

    with TestClient(app) as client:

        def chat(identity):
            return (
                client.post("/sessions", headers=identity, json={"agent_id": "garp"})
                .raise_for_status()
                .json()["session_id"]
            )

        sid, peer = chat(alice), chat(bob)
        protected = schedule(
            client, bob, name="Peer", session_id=peer, message="peer prompt", after_seconds=3600
        )["id"]
        standalone = schedule(
            client, alice, name="New chat", message="standalone prompt", after_seconds=3600
        )["id"]
        unlocked = schedule(
            client,
            alice,
            name="Removable one-shot",
            session_id=sid,
            message="pending prompt",
            after_seconds=3600,
        )["id"]
        locked = schedule(
            client,
            alice,
            name="Claimed recurring prompt",
            session_id=sid,
            message="retained pending prompt",
            cron_expr="0 12 1 * *",
        )["id"]
        with psycopg.connect(dsn) as conn:
            if retry_by_restart:
                conn.execute("""ALTER TABLE ai.garp_host_schedule_runs
                    DROP CONSTRAINT garp_host_schedule_runs_schedule_id_fkey,
                    ADD FOREIGN KEY (schedule_id) REFERENCES ai.garp_host_schedules(id)""")
            conn.execute(
                """INSERT INTO ai.garp_host_schedule_runs
                    (id, schedule_id, attempt, user_id, status, input, created_at)
                    VALUES (%s, %s, 1, %s, 'success', '{"message":"prior prompt"}', 1)""",
                (str(uuid4()), locked, OWNER),
            )

        with psycopg.connect(dsn) as claim:
            claim.execute("SET LOCAL idle_in_transaction_session_timeout = '12s'")
            claim.execute("SELECT 1 FROM ai.garp_host_schedules WHERE id=%s FOR UPDATE", (locked,))
            if retry_by_restart:
                deleted = client.delete(f"/sessions/{sid}", headers=alice)
            else:
                deleted = client.request(
                    "DELETE",
                    "/sessions",
                    headers=alice,
                    json={"session_ids": [sid, peer], "session_types": ["agent", "agent"]},
                )
            assert deleted.status_code == 204, deleted.text
            assert client.get(f"/schedules/{unlocked}", headers=alice).status_code == 404
            assert client.get(f"/schedules/{protected}", headers=bob).status_code == 200
            assert client.get(f"/schedules/{standalone}", headers=alice).status_code == 200
            assert claim.execute(
                "SELECT 1 FROM ai.garp_host_schedules WHERE id=%s", (locked,)
            ).fetchone()

        if not retry_by_restart:
            client.portal.call(app.state.queue_worker.store.cleanup_jobs, 86400)
            assert client.get(f"/schedules/{locked}", headers=alice).status_code == 404

    if retry_by_restart:
        with TestClient(create()) as client:
            assert client.get(f"/schedules/{locked}", headers=alice).status_code == 404
    with psycopg.connect(dsn) as conn:
        assert not conn.execute(
            "SELECT 1 FROM ai.garp_host_schedule_runs WHERE schedule_id=%s", (locked,)
        ).fetchone()
    assert not calls


def test_committed_scheduler_claim_cannot_revive_a_deleted_chat(service, host, monkeypatch):
    dsn, calls, *_ = service
    create, headers, _ = host
    alice = headers()
    app = create()
    with TestClient(app) as client:
        sid = (
            client.post("/sessions", headers=alice, json={"agent_id": "garp"})
            .raise_for_status()
            .json()["session_id"]
        )
        linked = schedule(
            client, alice, name="Due later", session_id=sid, message="Discard", after_seconds=3600
        )["id"]
        standalone = schedule(client, alice, name="New", message="Retain", after_seconds=3600)["id"]
        executor = app.state.scheduler_executor
        db = executor.checks.db
        claim_due = db.claim_due_schedule
        monkeypatch.setattr(
            db,
            "claim_due_schedule",
            lambda worker: claim_due(worker) if worker == "test-worker" else None,
        )
        tick = advance(dsn, linked)
        claim = db.claim_due_schedule("test-worker")
        assert (claim["id"], claim["next_run_at"]) == (linked, tick)
        assert client.delete(f"/sessions/{sid}", headers=alice).status_code == 204
        result = client.portal.call(executor._call_endpoint, Schedule.from_dict(claim))
        assert result == {"status": "cancelled", "error": "Schedule claim is no longer active"}
        with pytest.raises(psycopg.errors.CheckViolation) as error, psycopg.connect(dsn) as conn:
            conn.execute(
                """UPDATE ai.garp_host_schedules SET payload = jsonb_set(
                   payload, '{prompt,session_id}', to_jsonb(%s::text)) WHERE id=%s""",
                (sid, standalone),
            )
        assert error.value.diag.constraint_name == "garp_deleted_session"
        assert client.get(f"/schedules/{standalone}", headers=alice).status_code == 200
        assert client.get(f"/schedules/{linked}", headers=alice).status_code == 404
    assert not calls


def test_host_schedules_follow_existing_and_new_chats_after_restart(service, host, monkeypatch):
    dsn, calls, *_ = service
    create, headers, grants = host
    alice, bob = headers(), headers(OTHER)
    grants[OWNER] = [scope for scope in SCOPES if scope != "memories:read"]
    with TestClient(create()) as client:
        chat = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
        assert chat.status_code == 201, chat.text
        sid = chat.json()["session_id"]
        future = schedule(
            client,
            alice,
            name="Later in this chat",
            session_id=sid,
            message="Future existing prompt",
            after_seconds=3600,
            retry_delay_seconds=1,
        )
        assert "memories:read" not in future["payload"]["scopes"]
        recurring = schedule(
            client,
            alice,
            name="Open new monthly chat",
            cron_expr="0 12 1 * *",
            message="New recurring prompt",
        )
        failed = client.post(
            "/garp/prompt-schedules",
            headers=bob,
            json={
                "name": "Forged session",
                "session_id": sid,
                "message": "Other chat",
                "after_seconds": 1,
            },
        )
        assert failed.status_code == 404, failed.text
        assert client.get(f"/schedules/{future['id']}", headers=bob).status_code == 404
        with psycopg.connect(dsn) as db:
            assert db.execute("SELECT count(*) FROM ai.agno_jobs").fetchone() == (0,)
        early = client.post(
            "/agents/garp/runs",
            headers=alice,
            data={
                "session_id": sid,
                "message": "Interactive while pending",
                "background": "true",
                "stream": "false",
            },
        )
        assert early.status_code == 202, early.text
        eventually(
            lambda: client.get(f"/sessions/{sid}/runs", headers=alice),
            lambda rows: any(
                item["content"] == "Reply to Interactive while pending" for item in rows
            ),
        )

    send = PromptScheduleExecutor._call_endpoint
    acknowledgements = []

    async def lose_ack(self, row):
        result = await send(self, row)
        if row.id == future["id"] and result.get("status") == "success":
            acknowledgements.append(result["run_id"])
            if len(acknowledgements) == 1:
                return {
                    "status": "failed",
                    "status_code": 503,
                    "error": "Synthetic lost acknowledgement",
                }
        return result

    monkeypatch.setattr(PromptScheduleExecutor, "_call_endpoint", lose_ack)
    grants[OWNER] = SCOPES
    advance(dsn, future["id"])
    advance(dsn, recurring["id"])
    with TestClient(create()) as client:

        def history(item):
            return eventually(
                lambda: client.get(f"/schedules/{item['id']}/runs", headers=alice),
                lambda result: any(row["status"] == "success" for row in result["data"]),
            )["data"]

        attempts = history(future)
        assert {row["status"] for row in attempts} == {"failed", "success"}
        assert len(acknowledgements) == 2 and len(set(acknowledgements)) == 1
        existing = next(row for row in attempts if row["status"] == "success")
        separate = history(recurring)[0]
        assert existing["session_id"] == sid and separate["session_id"] != sid
        for run, message in (
            (existing, "Future existing prompt"),
            (separate, "New recurring prompt"),
        ):
            endpoint = f"/agents/garp/runs/{run['run_id']}"
            params = {"session_id": run["session_id"]}
            reply = eventually(
                lambda endpoint=endpoint, params=params: client.get(
                    endpoint, headers=alice, params=params
                ),
                lambda result: result["status"] == "COMPLETED",
            )
            assert reply["content"] == f"Reply to {message}" and calls[message] == 1
            assert client.get(endpoint, headers=bob, params=params).status_code == 404
        eventually(
            lambda: client.get(f"/schedules/{future['id']}", headers=alice),
            lambda result: result["next_run_at"] is None and not result["enabled"],
        )
        assert (
            client.post(f"/garp/prompt-schedules/{future['id']}/enable", headers=alice).status_code
            == 409
        )
        with psycopg.connect(dsn) as db:
            admitted = db.execute(
                "SELECT payload FROM ai.agno_jobs WHERE id=%s", (existing["run_id"],)
            ).fetchone()[0]
        assert "memories:read" not in admitted["scope"]["context"]["scopes"]


def test_overlapping_prompt_checks_live_grant_again_when_claimed(service, host):
    _dsn, calls, started, release = service
    create, headers, grants = host
    alice = headers()
    with TestClient(create()) as client:
        session = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
        assert session.status_code == 201, session.text
        sid = session.json()["session_id"]
        busy = client.post(
            "/agents/garp/runs",
            headers=alice,
            data={
                "message": "Occupied chat",
                "background": "true",
                "stream": "false",
                "session_id": sid,
            },
        )
        assert busy.status_code == 202 and started.wait(10), busy.text
        try:
            planned = schedule(
                client,
                alice,
                name="Requires fresh grant",
                session_id=sid,
                message="Revoked prompt",
                after_seconds=0,
            )
            successful = eventually(
                lambda: client.get(f"/schedules/{planned['id']}/runs", headers=alice),
                lambda result: any(row["status"] == "success" for row in result["data"]),
            )["data"][0]
            assert calls["Revoked prompt"] == 0
            grants[OWNER] = ["schedules:write", "sessions:read"]
        finally:
            release.set()
        endpoint = f"/agents/garp/runs/{successful['run_id']}"
        eventually(
            lambda: client.get(endpoint, headers=alice, params={"session_id": sid}),
            lambda result: result["status"] == "ERROR",
        )
        assert calls["Revoked prompt"] == 0
        assert (
            client.post(
                "/garp/prompt-schedules",
                headers=alice,
                json={"name": "Denied", "message": "Denied", "after_seconds": 0},
            ).status_code
            == 403
        )


def test_overlapping_recurring_ticks_reuse_the_chat_after_restart(service, host):
    dsn, calls, started, release = service
    create, headers, _ = host
    alice = headers()
    with TestClient(create()) as client:
        session = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
        assert session.status_code == 201, session.text
        sid = session.json()["session_id"]
        recurring = schedule(
            client,
            alice,
            name="Follow this chat every month",
            message="Occupied chat",
            session_id=sid,
            cron_expr="0 9 1 * *",
        )
        once = schedule(
            client,
            alice,
            name="Start a chat after restart",
            message="Separate report",
            after_seconds=3600,
        )

    first_tick = advance(dsn, recurring["id"], int(time.time()) - 90)
    advance(dsn, once["id"])
    with TestClient(create()) as client:

        def successful(schedule_id, count):
            response = eventually(
                lambda: client.get(f"/schedules/{schedule_id}/runs", headers=alice),
                lambda body: sum(row["status"] == "success" for row in body["data"]) == count,
            )
            return [row for row in response["data"] if row["status"] == "success"]

        def run(run_id, session_id):
            return client.get(
                f"/agents/garp/runs/{run_id}", headers=alice, params={"session_id": session_id}
            )

        try:
            first = successful(recurring["id"], 1)[0]
            assert first["session_id"] == sid and started.wait(10)
            eventually(
                lambda: client.get(f"/schedules/{recurring['id']}", headers=alice),
                lambda body: body["next_run_at"] > first_tick,
            )
            assert advance(dsn, recurring["id"]) > first_tick
            ticks = successful(recurring["id"], 2)
            second = next(row for row in ticks if row["run_id"] != first["run_id"])
            assert second["session_id"] == sid
            assert run(second["run_id"], sid).json()["status"] == "PENDING"
            separate = successful(once["id"], 1)[0]
            assert separate["session_id"] != sid
            reply = eventually(
                lambda: run(separate["run_id"], separate["session_id"]),
                lambda body: body["status"] == "COMPLETED",
            )
            assert reply["content"] == "Reply to Separate report"
            assert calls["Occupied chat"] == 1
        finally:
            release.set()
        for tick in ticks:
            eventually(
                lambda run_id=tick["run_id"]: run(run_id, sid),
                lambda body: body["status"] == "COMPLETED",
            )
    assert calls == Counter({"Occupied chat": 2, "Separate report": 1})


@pytest.mark.parametrize("pause_during_retry", [False, True])
def test_one_shot_can_resume_after_failed_dispatch(service, host, pause_during_retry):
    dsn, calls, *_ = service
    create, headers, grants = host
    alice = headers()
    with TestClient(create()) as client:
        timer = schedule(
            client,
            alice,
            name="Retry a one-shot",
            message="Recover this prompt",
            after_seconds=3600,
            max_retries=int(pause_during_retry),
            retry_delay_seconds=3,
        )
        url = f"/schedules/{timer['id']}"
        grants[OWNER] = []
        tick = advance(dsn, timer["id"])
        eventually(
            lambda: client.get(f"{url}/runs", headers=alice),
            lambda body: any(row.get("status_code") == 403 for row in body["data"]),
        )
        if pause_during_retry:
            assert client.post(f"{url}/disable", headers=alice).status_code == 200
            eventually(
                lambda: client.get(f"{url}/runs", headers=alice),
                lambda body: any(row["status"] == "cancelled" for row in body["data"]),
            )
        state = eventually(
            lambda: client.get(url, headers=alice),
            lambda body: not body["enabled"] and body["disabled_reason"] == "dispatch_failed",
        )
        assert state["next_run_at"] == tick
        assert calls["Recover this prompt"] == 0
        grants[OWNER] = SCOPES
        response = client.post(f"/garp/prompt-schedules/{timer['id']}/enable", headers=alice)
        assert response.status_code == 200, response.text
        assert response.json()["next_run_at"] == tick
        attempts = eventually(
            lambda: client.get(f"{url}/runs", headers=alice),
            lambda body: any(row["status"] == "success" for row in body["data"]),
        )
        run = next(row for row in attempts["data"] if row["status"] == "success")
        eventually(
            lambda: client.get(
                f"/agents/garp/runs/{run['run_id']}",
                params={"session_id": run["session_id"]},
                headers=alice,
            ),
            lambda body: body["status"] == "COMPLETED",
        )
        state = eventually(
            lambda: client.get(url, headers=alice),
            lambda body: body["disabled_reason"] == "admitted",
        )
        assert state["next_run_at"] is None
    assert calls["Recover this prompt"] == 1


def test_cancel_admitted_prompt_after_deleting_schedule(service, host):
    _dsn, calls, started, release = service
    create, headers, _ = host
    alice, bob = headers(), headers(OTHER)
    with TestClient(create()) as client:
        session = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
        assert session.status_code == 201, session.text
        sid = session.json()["session_id"]
        busy = client.post(
            "/agents/garp/runs",
            headers=alice,
            data={
                "message": "Occupied chat",
                "background": "true",
                "stream": "false",
                "session_id": sid,
            },
        )
        assert busy.status_code == 202 and started.wait(10), busy.text
        try:
            timer = schedule(
                client,
                alice,
                name="Cancel the admitted prompt",
                session_id=sid,
                message="Cancel before inference",
                after_seconds=0,
            )
            url = f"/schedules/{timer['id']}"
            attempts = eventually(
                lambda: client.get(f"{url}/runs", headers=alice),
                lambda body: any(row["status"] == "success" for row in body["data"]),
            )
            run = next(row for row in attempts["data"] if row["status"] == "success")
            endpoint = f"/agents/garp/runs/{run['run_id']}"
            params = {"session_id": sid}
            deleted = client.delete(url, headers=alice)
            assert deleted.status_code in (200, 204), deleted.text
            assert client.get(endpoint, params=params, headers=alice).json()["status"] == "PENDING"
            assert client.post(endpoint + "/cancel", params=params, headers=bob).status_code == 404
            cancelled = client.post(endpoint + "/cancel", params=params, headers=alice)
            assert cancelled.status_code == 200, cancelled.text
        finally:
            release.set()
        eventually(
            lambda: client.get(endpoint, params=params, headers=alice),
            lambda body: body["status"] == "CANCELLED",
        )
    assert calls["Cancel before inference"] == 0


def test_one_shot_stays_admitted_if_pending_write_fails(service, host, monkeypatch, caplog):
    _dsn, calls, *_ = service
    create, headers, _ = host
    alice = headers()
    failed = Event()
    append = PostgresDb.append_run_to_session_if_absent

    def fail_once(self, session_id, run_dict, user_id=None):
        if not failed.is_set() and run_dict.get("queue_attempt") == 0:
            failed.set()
            return None
        return append(self, session_id, run_dict, user_id=user_id)

    monkeypatch.setattr(PostgresDb, "append_run_to_session_if_absent", fail_once)
    with TestClient(create()) as client:
        timer = schedule(
            client,
            alice,
            name="Stay admitted",
            message="Reply after a pending row failure",
            after_seconds=0,
            max_retries=0,
        )
        url = f"/schedules/{timer['id']}"
        eventually(
            lambda: client.get(url, headers=alice),
            lambda body: not body["enabled"] and body["next_run_at"] is None,
        )
        assert (
            failed.is_set() and "Pending row unavailable for admitted scheduled run" in caplog.text
        )
        attempts = client.get(f"{url}/runs", headers=alice).json()["data"]
        assert [row["status"] for row in attempts] == ["success"]
        run = attempts[0]
        eventually(
            lambda: client.get(
                f"/agents/garp/runs/{run['run_id']}",
                headers=alice,
                params={"session_id": run["session_id"]},
            ),
            lambda body: body["status"] == "COMPLETED",
        )
    assert calls["Reply after a pending row failure"] == 1


def test_cli_schedules_a_prompt_on_the_local_service(service, monkeypatch, capsys):
    dsn, calls, *_ = service
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("GARP_TOKEN", token)
    app = create_agentos_app(dsn, token=token, scheduler_url="http://127.0.0.1:8787")
    with _serve_app(app) as base:
        assert (
            main(
                [
                    "--url",
                    base,
                    "schedule-prompt",
                    "CLI timer prompt",
                    "--name",
                    "CLI timer",
                    "--after",
                    "0",
                ]
            )
            == 0
        )
        result = json.loads(capsys.readouterr().out)
        with httpx.Client(
            base_url=base, headers={"Authorization": f"Bearer {token}"}, trust_env=False
        ) as client:
            attempts = eventually(
                lambda: client.get(f"/schedules/{result['id']}/runs"),
                lambda body: any(row["status"] == "success" for row in body["data"]),
            )
            run = attempts["data"][0]
            reply = eventually(
                lambda: client.get(
                    f"/agents/garp/runs/{run['run_id']}", params={"session_id": run["session_id"]}
                ),
                lambda body: body["status"] == "COMPLETED",
            )
            assert reply["content"] == "Reply to CLI timer prompt"
            commands = iter(("/schedule 0 Follow up in terminal", "/exit"))
            monkeypatch.setattr("builtins.input", lambda _: next(commands))
            assert main(["--url", base, "chat", run["session_id"]]) == 0
            terminal = capsys.readouterr().out
            assert "Prompt scheduled in 0s" in terminal
            terminal_key = terminal.split("Schedule ID  ", 1)[1].splitlines()[0]
            accepted = eventually(
                lambda: client.get(f"/schedules/{terminal_key}/runs"),
                lambda body: any(row["status"] == "success" for row in body["data"]),
            )["data"][0]
            assert accepted["session_id"] == run["session_id"]
            eventually(
                lambda: client.get(f"/sessions/{run['session_id']}/runs"),
                lambda rows: any(
                    row["content"] == "Reply to Follow up in terminal" for row in rows
                ),
            )
            future = (
                client.post(
                    "/garp/prompt-schedules",
                    json={
                        "name": "Keep the original due time",
                        "message": "Later",
                        "after_seconds": 3600,
                    },
                )
                .raise_for_status()
                .json()
            )
            url = f"/schedules/{future['id']}"
            assert client.post(f"{url}/disable").status_code == 200
            assert client.post(f"{url}/enable").status_code == 409
            assert client.patch(url, json={"timezone": "UTC"}).status_code == 409
            paused = client.get(url).raise_for_status().json()
            assert not paused["enabled"] and paused["next_run_at"] == future["next_run_at"]
            resumed = (
                client.post(f"/garp/prompt-schedules/{future['id']}/enable")
                .raise_for_status()
                .json()
            )
            assert resumed["enabled"] and resumed["next_run_at"] == future["next_run_at"]
            native = (
                client.post(
                    "/schedules",
                    json={
                        "name": "Native cron",
                        "cron_expr": "0 9 1 * *",
                        "endpoint": "/agents/garp/runs",
                    },
                )
                .raise_for_status()
                .json()
            )
            assert client.post(f"/schedules/{native['id']}/disable").status_code == 200
            assert client.post(f"/schedules/{native['id']}/enable").status_code == 200
    assert calls["CLI timer prompt"] == 1
    assert calls["Follow up in terminal"] == 1


def test_interrupted_one_shot_preserves_due_time_and_then_expires(service, host, monkeypatch):
    dsn, calls, *_ = service
    create, headers, _ = host
    alice = headers()
    dispatch = PromptScheduleExecutor._call_endpoint
    started = Event()

    async def interrupted(self, timer):
        if not started.is_set():
            started.set()
            await asyncio.Event().wait()
        return await dispatch(self, timer)

    monkeypatch.setattr(PromptScheduleExecutor, "_call_endpoint", interrupted)
    with TestClient(create()) as client:
        timer = schedule(
            client,
            alice,
            name="Expired during outage",
            message="Should never infer",
            after_seconds=3600,
            deadline_seconds=10,
            max_retries=0,
        )
        tick = advance(dsn, timer["id"], int(time.time()) - 3600)
        assert started.wait(10)

    with psycopg.connect(dsn) as db:
        assert db.execute(
            "SELECT enabled, locked_by, next_run_at FROM ai.garp_host_schedules WHERE id=%s",
            (timer["id"],),
        ).fetchone() == (True, None, tick)
        assert db.execute("SELECT count(*) FROM ai.agno_jobs").fetchone() == (0,)

    with TestClient(create()) as client:
        attempts = eventually(
            lambda: client.get(f"/schedules/{timer['id']}/runs", headers=alice),
            lambda result: any(row.get("status_code") == 410 for row in result["data"]),
        )
        assert {row["status"] for row in attempts["data"]} == {"failed", "cancelled"}
        expired = eventually(
            lambda: client.get(f"/schedules/{timer['id']}", headers=alice),
            lambda result: not result["enabled"] and result["next_run_at"] is None,
        )
        assert expired["disabled_reason"] == "expired"
        response = client.post(f"/garp/prompt-schedules/{timer['id']}/enable", headers=alice)
        assert (
            response.status_code == 409
            and response.json()["detail"] == "This one-shot prompt expired"
        )
    assert calls["Should never infer"] == 0
