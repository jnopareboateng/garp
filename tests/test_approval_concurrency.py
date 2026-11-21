import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import jwt
from fastapi.testclient import TestClient

from garp import agentos
from garp import approvals as approval_routes
from garp.agentos import HostAuth, create_agentos_app
from garp.extensions import Extensions


def test_hosted_approval_races(postgres_dsn, monkeypatch):
    key = "fixture-verification-key-with-at-least-32-bytes"
    auth = HostAuth("https://fixture.example", "approval-test", key, "HS256")
    owner = "tenant:owner"
    grants = ["agents:garp:run", "sessions:read", "actions:record:run", "approvals:read"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)
    monkeypatch.setattr(agentos, "generate_title", _title)
    actions = []
    lock = threading.Lock()

    def record(note: str) -> str:
        with lock:
            actions.append(note)
        return note

    async def respond(request):
        payload = json.loads(request.content)
        last = payload["messages"][-1]
        if (
            last["role"] == "tool"
            and any(
                msg.get("content") == "cycles"
                for msg in payload["messages"]
                if msg["role"] == "user"
            )
            and sum(msg["role"] == "tool" for msg in payload["messages"]) == 1
        ):
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-next",
                        "type": "function",
                        "function": {"name": "record", "arguments": json.dumps({"note": "next"})},
                    }
                ],
            }
            finish = "tool_calls"
        elif last["role"] == "tool":
            message, finish = {"role": "assistant", "content": "Done."}, "stop"
        else:
            notes = ["first", "second"] if last.get("content") == "two" else ["race"]
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call-{note}",
                        "type": "function",
                        "function": {"name": "record", "arguments": json.dumps({"note": note})},
                    }
                    for note in notes
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
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            },
        )

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(agentos, "AsyncClient", Provider)

    def headers(subject=owner, scopes=grants):
        token = jwt.encode(
            {
                "iss": auth.issuer,
                "aud": auth.audience,
                "exp": int(time.time()) + 120,
                "sub": subject,
                "scopes": scopes,
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": "Bearer " + token}

    app = create_agentos_app(
        postgres_dsn, host_auth=auth, extensions={"tenant": Extensions(approval_tools=(record,))}
    )
    with TestClient(app) as client:

        def wait(run, status):
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                response = client.get(
                    f"/sessions/{run['session_id']}/runs/{run['run_id']}", headers=headers()
                )
                if response.status_code == 200 and response.json()["status"] == status:
                    return
                time.sleep(0.05)
            raise AssertionError(response.text)

        def submit(prompt):
            response = client.post(
                "/agents/garp/runs",
                headers=headers(),
                data={"message": prompt, "stream": "false", "background": "true"},
            )
            assert response.status_code == 202, response.text
            run = response.json()
            wait(run, "PAUSED")
            # Wait for the worker to park the durable ticket after storing the paused run.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                job = client.portal.call(app.state.queue_worker.store.get_job, run["run_id"])
                if job and job["status"] == "paused":
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(f"ticket did not pause: {job}")
            response = client.get("/approvals", headers=headers(), params={"run_id": run["run_id"]})
            assert response.status_code == 200, response.text
            return run, response.json()["data"]

        def race(calls):
            barrier = threading.Barrier(len(calls))

            def execute(call):
                barrier.wait(timeout=5)
                return call()

            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                return list(pool.map(execute, calls))

        run, approvals = submit("one")
        assert len(approvals) == 1
        path = "/garp/approvals/" + approvals[0]["id"]
        reviewers = [
            headers("tenant:reviewer-a", ["approvals:write"]),
            headers("tenant:reviewer-b", ["approvals:write"]),
        ]
        decisions = ["approved", "rejected"]
        responses = race(
            [
                lambda i=i: client.post(
                    path + "/resolve", headers=reviewers[i], json={"status": decisions[i]}
                )
                for i in range(2)
            ]
        )
        assert sorted(r.status_code for r in responses) == [200, 409]
        row = client.get(path, headers=reviewers[0]).json()
        winner = next(i for i, r in enumerate(responses) if r.status_code == 200)
        assert row["status"] == decisions[winner] and row["resolved_by"].endswith(
            ("reviewer-a", "reviewer-b")
        )

        run, approvals = submit("one")
        assert len(approvals) == 1
        db = app.state.queue_worker.store._store
        extra = db.get_approval(approvals[0]["id"])
        extra["id"] = str(uuid4())
        db.create_approval(extra)
        paths = [
            "/garp/approvals/" + approval_id for approval_id in (approvals[0]["id"], extra["id"])
        ]
        assert (
            client.post(
                paths[0] + "/resolve", headers=reviewers[0], json={"status": "approved"}
            ).status_code
            == 200
        )
        assert client.post(paths[0] + "/continue", headers=headers()).status_code == 409

        def fail_pending_query(*args, **kwargs):
            raise RuntimeError("synthetic database failure")

        with monkeypatch.context() as patch:
            patch.setattr(db, "get_approvals", fail_pending_query)
            assert client.post(paths[0] + "/continue", headers=headers()).status_code == 500
        assert not actions
        assert (
            client.post(
                paths[1] + "/resolve", headers=reviewers[1], json={"status": "approved"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                paths[0] + "/continue", headers=headers(scopes=["agents:garp:run"])
            ).status_code
            == 403
        )
        responses = race(
            [lambda: client.post(paths[0] + "/continue", headers=headers()) for _ in range(8)]
        )
        assert any(r.status_code == 202 for r in responses), [
            (r.status_code, r.text) for r in responses
        ]
        assert all(r.status_code in (202, 409) for r in responses), [
            (r.status_code, r.text) for r in responses
        ]
        wait(run, "COMPLETED")
        assert actions == ["race"]

        run, approvals = submit("cycles")
        first_path = "/garp/approvals/" + approvals[0]["id"]
        assert (
            client.post(
                first_path + "/resolve", headers=reviewers[0], json={"status": "approved"}
            ).status_code
            == 200
        )
        original_continue = approval_routes.acontinue_via_queue
        blocked, release = threading.Event(), threading.Event()

        async def delay_one(*args, **kwargs):
            if not blocked.is_set():
                blocked.set()
                assert await asyncio.to_thread(release.wait, 12)
            return await original_continue(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(approval_routes, "acontinue_via_queue", delay_one)
            with ThreadPoolExecutor(max_workers=1) as pool:
                delayed = pool.submit(client.post, first_path + "/continue", headers=headers())
                assert blocked.wait(5)
                assert client.post(first_path + "/continue", headers=headers()).status_code == 409
                release.set()
                assert delayed.result(timeout=5).status_code == 202
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    rows = client.get(
                        "/approvals", headers=headers(), params={"run_id": run["run_id"]}
                    ).json()["data"]
                    job = client.portal.call(app.state.queue_worker.store.get_job, run["run_id"])
                    if (
                        any(row["status"] == "pending" for row in rows)
                        and job["status"] == "paused"
                    ):
                        break
                    time.sleep(0.05)
                else:
                    raise AssertionError("second pause not observed")
                next_id = next(row["id"] for row in rows if row["status"] == "pending")
                next_path = "/garp/approvals/" + next_id
                assert client.post(first_path + "/continue", headers=headers()).status_code == 409
                assert (
                    client.post(
                        next_path + "/resolve", headers=reviewers[0], json={"status": "approved"}
                    ).status_code
                    == 200
                )
                assert client.post(first_path + "/continue", headers=headers()).status_code == 409
                assert client.post(next_path + "/continue", headers=headers()).status_code == 202
                wait(run, "COMPLETED")
                assert actions == ["race", "race", "next"]

        async def check_cancelled_lock():
            entered = asyncio.Event()

            async def hold_lock():
                async with approval_routes._continuation_lock(postgres_dsn, run["run_id"]):
                    entered.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(hold_lock())
            await entered.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            async with approval_routes._continuation_lock(postgres_dsn, run["run_id"]):
                pass

        client.portal.call(check_cancelled_lock)

        # Lock admission remains available even when the application's pool is exhausted.
        pool = db.db_engine.pool
        connections = [db.db_engine.connect() for _ in range(pool.size() + pool._max_overflow)]
        try:

            async def lock_with_exhausted_pool():
                async with approval_routes._continuation_lock(postgres_dsn, run["run_id"]):
                    pass

            client.portal.call(lock_with_exhausted_pool)
        finally:
            for connection in connections:
                connection.close()


async def _title(_message, **_kwargs):
    return "Approval Race"
