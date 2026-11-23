"""Exercise native Redis transports across independent AgentOS processes."""

import asyncio
import multiprocessing
import os
import socket
import subprocess
import time
from queue import Empty

import pytest


def replica(dsn, redis_url, prefix, commands, results, started, release):
    import httpx
    import jwt
    from agno.job_queue.config import RedisCoordination
    from fastapi.testclient import TestClient

    from garp import agentos

    os.environ.update(OPENROUTER_API_KEY="fixture-only", GARP_OPENROUTER_ZDR_ONLY="true")

    async def reply(request):
        started.put(os.getpid())
        await asyncio.to_thread(release.wait, 20)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fixture result"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(reply), **kwargs)

    async def title(*args, **kwargs):
        return "Fixture"

    agentos.AsyncClient = Provider
    agentos.generate_title = title
    key = "fixture-key-longer-than-32-bytes-for-jwt"
    auth = agentos.HostAuth("https://fixture.local", "replicas", key, "HS256")
    app = agentos.create_agentos_app(
        dsn, host_auth=auth, redis_coordination=RedisCoordination(url=redis_url, key_prefix=prefix)
    )
    with TestClient(app) as client:
        token = jwt.encode(
            {
                "iss": auth.issuer,
                "aud": auth.audience,
                "sub": "tenant:owner",
                "exp": int(time.time()) + 120,
                "scopes": ["sessions:read", "sessions:write", "agents:garp:run", "agents:read"],
            },
            key,
            algorithm="HS256",
        )
        client.headers["Authorization"] = f"Bearer {token}"
        results.put(("ready", os.getpid()))
        while True:
            command = commands.get()
            if command[0] == "stop":
                return
            method, path, kwargs = command
            response = client.request(method, path, **kwargs)
            results.put((response.status_code, response.text))


def test_cross_process_cancellation(postgres_dsn, tmp_path):
    if not os.path.exists("/usr/bin/redis-server"):
        pytest.skip("redis-server unavailable")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen(
        [
            "/usr/bin/redis-server",
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
    )
    ctx = multiprocessing.get_context("spawn")
    queues = [ctx.Queue(), ctx.Queue()]
    results = [ctx.Queue(), ctx.Queue()]
    started, release = ctx.Queue(), ctx.Event()
    processes = [
        ctx.Process(
            target=replica,
            args=(
                postgres_dsn,
                f"redis://127.0.0.1:{port}",
                "test-" + tmp_path.name,
                queues[i],
                results[i],
                started,
                release,
            ),
        )
        for i in range(2)
    ]
    try:
        for p in processes:
            p.start()
        for p, result in zip(processes, results, strict=True):
            # Cold imports and concurrent schema initialization precede readiness.
            try:
                assert result.get(timeout=90)[0] == "ready"
            except Empty:
                pytest.fail(f"Replica {p.pid} missed its readiness deadline (exit={p.exitcode}).")
        queues[0].put(("POST", "/sessions", {"json": {"agent_id": "garp"}}))
        status, body = results[0].get(timeout=10)
        assert status == 201, body
        import json

        sid = json.loads(body)["session_id"]
        queues[0].put(
            (
                "POST",
                "/agents/garp/runs",
                {
                    "data": {
                        "session_id": sid,
                        "message": "fixture",
                        "background": "true",
                        "stream": "false",
                    }
                },
            )
        )
        status, body = results[0].get(timeout=10)
        assert status == 202, body
        rid = json.loads(body)["run_id"]
        worker_pid = started.get(timeout=15)
        peer = 1 if processes[0].pid == worker_pid else 0
        assert processes[peer].pid != worker_pid
        queues[peer].put(
            ("POST", f"/agents/garp/runs/{rid}/cancel", {"params": {"session_id": sid}})
        )
        status, body = results[peer].get(timeout=10)
        assert status == 200, body
        release.set()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            queues[peer].put(("GET", f"/agents/garp/runs/{rid}", {"params": {"session_id": sid}}))
            status, body = results[peer].get(timeout=5)
            if status == 200 and json.loads(body).get("status", "").upper() == "CANCELLED":
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"run did not cancel: {status} {body}")
    finally:
        release.set()
        for q in queues:
            q.put(("stop",))
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
        server.terminate()
        server.wait(timeout=10)


def streaming_replica(dsn, redis_url, prefix, port, started, release):
    import json

    import httpx
    import uvicorn
    from agno.job_queue.config import RedisCoordination

    from garp import agentos

    os.environ.update(OPENROUTER_API_KEY="fixture-only", GARP_OPENROUTER_ZDR_ONLY="true")

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for content in ("first ", "second"):
                if content == "second":
                    await asyncio.to_thread(release.wait, 30)
                event = {
                    "id": "fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture",
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(event)}\n\n".encode()
            event = {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()

    async def reply(request):
        started.put(os.getpid())
        return httpx.Response(200, stream=Chunks(), headers={"content-type": "text/event-stream"})

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(reply), **kwargs)

    async def title(*args, **kwargs):
        return "Fixture"

    agentos.AsyncClient = Provider
    agentos.generate_title = title
    auth = agentos.HostAuth(
        "https://fixture.local", "replicas", "fixture-key-longer-than-32-bytes-for-jwt", "HS256"
    )
    app = agentos.create_agentos_app(
        dsn, host_auth=auth, redis_coordination=RedisCoordination(url=redis_url, key_prefix=prefix)
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_cross_process_stream_reconnect(postgres_dsn, tmp_path):
    import json
    import queue
    from uuid import uuid4

    import httpx
    import jwt

    if not os.path.exists("/usr/bin/redis-server"):
        pytest.skip("redis-server unavailable")
    redis_port, port_a, port_b = free_port(), free_port(), free_port()
    server = subprocess.Popen(
        [
            "/usr/bin/redis-server",
            "--port",
            str(redis_port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
    )
    ctx = multiprocessing.get_context("spawn")
    started, release = ctx.Queue(), ctx.Event()
    processes = [
        ctx.Process(
            target=streaming_replica,
            args=(
                postgres_dsn,
                f"redis://127.0.0.1:{redis_port}",
                "test-" + tmp_path.name,
                port,
                started,
                release,
            ),
        )
        for port in (port_a, port_b)
    ]
    key = "fixture-key-longer-than-32-bytes-for-jwt"

    def token(subject):
        return jwt.encode(
            {
                "iss": "https://fixture.local",
                "aud": "replicas",
                "sub": subject,
                "exp": int(time.time()) + 120,
                "scopes": ["sessions:read", "sessions:write", "agents:garp:run", "agents:read"],
            },
            key,
            algorithm="HS256",
        )

    a, b = f"http://127.0.0.1:{port_a}", f"http://127.0.0.1:{port_b}"
    owner = {"Authorization": "Bearer " + token("tenant:owner")}
    try:
        for p in processes:
            p.start()
        with httpx.Client(timeout=15) as client:
            for base in (a, b):
                deadline = time.monotonic() + 30
                while True:
                    try:
                        client.get(base + "/health")
                        break
                    except httpx.ConnectError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.1)
            created = client.post(a + "/sessions", headers=owner, json={"agent_id": "garp"})
            created.raise_for_status()
            sid = created.json()["session_id"]
            idem = str(uuid4())
            headers = {**owner, "Idempotency-Key": idem}
            data = {"session_id": sid, "message": "fixture", "background": "true", "stream": "true"}
            events = []
            with client.stream(
                "POST", a + "/agents/garp/runs", headers=headers, data=data
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        event = json.loads(line[5:].strip())
                        events.append(event)
                        if event.get("event") == "RunContent" and "first " in str(
                            event.get("content")
                        ):
                            break
            rid = next(event["run_id"] for event in events if event.get("run_id"))
            index = max(event.get("event_index", -1) for event in events)
            assert started.get(timeout=5) in [p.pid for p in processes]
            resume = client.post(
                b + f"/agents/garp/runs/{rid}/resume",
                headers=owner,
                data={"session_id": sid, "last_event_index": str(index)},
            )
            # Native factory agents may require replaying the idempotent submission.
            assert resume.status_code == 400, resume.text
            peer = client.post(
                b + "/agents/garp/runs",
                headers={
                    "Authorization": "Bearer " + token("tenant:peer"),
                    "Idempotency-Key": idem,
                },
                data=data,
            )
            assert peer.status_code in (403, 404, 409), peer.text
            release.set()
            with client.stream(
                "POST", b + "/agents/garp/runs", headers=headers, data=data
            ) as replay:
                replay.raise_for_status()
                for line in replay.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:].strip()))
            chunks = [
                event.get("content")
                for event in events
                if event.get("event") == "RunContent" and event.get("event_index", -1) > index
            ]
            assert "".join(chunks) == "second"
            stored = client.get(
                b + f"/agents/garp/runs/{rid}", headers=owner, params={"session_id": sid}
            )
            assert stored.status_code == 200 and stored.json()["content"] == "first second", (
                stored.text
            )
            with pytest.raises(queue.Empty):
                started.get(timeout=1)
    finally:
        release.set()
        for p in processes:
            p.terminate()
            p.join(timeout=10)
        server.terminate()
        server.wait(timeout=10)
