import asyncio
import json
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import jwt
import psycopg
import pytest
from agno.run.base import RunContext
from agno.tools.function import FunctionCall
from fastapi.testclient import TestClient

from garp import agentos
from garp.extensions import Extensions, RemoteMCP, _RemoteMCPTools


def test_remote_mcp_user_isolation_and_lifecycle(tmp_path, monkeypatch, postgres_dsn):
    server = tmp_path / "server.py"
    server.write_text('''
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
import asyncio

mcp = FastMCP("Fixture")

@mcp.tool
async def whoami(delay: float = 0, fail: bool = False) -> str:
    """Return the identity presented to the synthetic server."""
    await asyncio.sleep(delay)
    if fail:
        raise ValueError("fixture error")
    token = get_http_headers(include={"authorization"}).get("authorization", "missing")
    return {"Bearer synthetic-tenant:one": "one", "Bearer synthetic-tenant:two": "two"}.get(token, "denied")

mcp.run(transport="streamable-http", host="127.0.0.1", port=int(__import__("sys").argv[1]), show_banner=False)
''')
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    process = subprocess.Popen(
        [sys.executable, str(server), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise AssertionError("Fixture server exited")
            try:
                httpx.get(url, timeout=0.1)
                break
            except httpx.TransportError:
                time.sleep(0.05)
        else:
            raise AssertionError("Fixture server did not start")

        seen = []
        active = set()
        build_client = _RemoteMCPTools._remote_client

        @asynccontextmanager
        async def tracked_client(*args, **kwargs):
            client = build_client(*args, **kwargs)
            async with client as session:
                active.add(id(client))
                try:
                    yield session
                finally:
                    active.remove(id(client))

        monkeypatch.setattr(_RemoteMCPTools, "_remote_client", tracked_client)

        def resolve(principal):
            seen.append(principal)
            return {"Authorization": "Bearer synthetic-" + principal}

        ext = Extensions(remote_mcp=(RemoteMCP("catalog", url, ("whoami",), resolve),))

        def ctx(user, scopes):
            return SimpleNamespace(user_id=user, trusted=SimpleNamespace(scopes=scopes))

        async def check():
            denied, _, _ = await ext.for_run(
                ctx("tenant:denied", []), True, approval_tools=False, mcp_attempted={}
            )
            assert denied == [] and seen == []
            user1, _, _ = await ext.for_run(
                ctx("tenant:one", ["mcp:catalog:run"]), True, approval_tools=False, mcp_attempted={}
            )
            user2, _, _ = await ext.for_run(
                ctx("tenant:two", ["mcp:catalog:run"]), True, approval_tools=False, mcp_attempted={}
            )
            assert seen == ["tenant:one", "tenant:two"]
            assert not active
            empty = Extensions(remote_mcp=(RemoteMCP("empty", url, ("missing",), resolve),))
            tools, instructions, _ = await empty.for_run(
                ctx("tenant:one", ["mcp:empty:run"]), True, approval_tools=False, mcp_attempted={}
            )
            assert not tools and instructions == ["The empty MCP extension is unavailable."]
            seen.clear()
            run1 = RunContext(run_id="one", session_id="one", user_id="tenant:one")
            run2 = RunContext(run_id="two", session_id="two", user_id="tenant:two")
            results = await asyncio.gather(
                user1[0].entrypoint(_agno_run_context=run1),
                user2[0].entrypoint(_agno_run_context=run2),
            )
            assert results[0].content == "one"
            assert results[1].content == "two"
            assert "Bearer synthetic-tenant" not in str(user1[0].to_dict())
            user1[0]._run_context = run1
            call = FunctionCall(function=user1[0], arguments={})
            executed = await call.aexecute()
            assert executed.status == "success" and executed.result.content == "one"
            assert not active
            await user1[0].entrypoint(_agno_run_context=run1, fail=True)
            assert not active
            with pytest.raises(PermissionError):
                await user1[0].entrypoint(_agno_run_context=run2)
            task = asyncio.create_task(user1[0].entrypoint(_agno_run_context=run1, delay=10))
            await asyncio.sleep(0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not active

        asyncio.run(check())

        seen.clear()
        key = "remote-mcp-fixture-jwt-key-at-least-32-bytes"
        auth = agentos.HostAuth("https://host.example", "mcp-fixture", key, "HS256")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
        monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
        monkeypatch.setenv("GARP_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")
        monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)

        async def respond(request):
            assert request.url.host == "openrouter.ai"
            assert b"Bearer synthetic-tenant" not in request.content
            payload = json.loads(request.content)
            assert payload["provider"]["zdr"] is True
            last = payload["messages"][-1]
            tool = any(item["function"]["name"] == "whoami" for item in payload.get("tools", ()))
            finish = "stop"
            if last["role"] == "tool":
                content = last["content"]
                if isinstance(content, list):
                    content = content[0]["text"]
                message = {"role": "assistant", "content": content}
            elif tool:
                finish = "tool_calls"
                message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "read-my-catalog",
                            "type": "function",
                            "function": {"name": "whoami", "arguments": "{}"},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "No catalog"}
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

        async def title(*args, **kwargs):
            return "Remote MCP Fixture"

        monkeypatch.setattr(agentos, "AsyncClient", Provider)
        monkeypatch.setattr(agentos, "generate_title", title)

        def identity(subject, allowed):
            scopes = ["agents:garp:run", "sessions:read"]
            if allowed:
                scopes.append("mcp:catalog:run")
            token = jwt.encode(
                {
                    "iss": auth.issuer,
                    "aud": auth.audience,
                    "sub": subject,
                    "exp": int(time.time()) + 120,
                    "scopes": scopes,
                },
                key,
                algorithm="HS256",
            )
            return {"Authorization": f"Bearer {token}"}

        app = agentos.create_agentos_app(postgres_dsn, host_auth=auth, extensions={"tenant": ext})
        with TestClient(app) as client:
            for principal, expected, allowed in (
                ("tenant:one", "one", True),
                ("tenant:two", "two", True),
                ("tenant:denied", "No catalog", False),
            ):
                headers = identity(principal, allowed)
                run = client.post(
                    "/agents/garp/runs",
                    headers=headers,
                    data={"message": "Identify me", "background": "true", "stream": "false"},
                )
                assert run.status_code == 202, run.text
                accepted = run.json()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    response = client.get(
                        f"/sessions/{accepted['session_id']}/runs/{accepted['run_id']}",
                        headers=headers,
                    )
                    if response.status_code == 200 and response.json()["status"] == "COMPLETED":
                        break
                    time.sleep(0.05)
                assert response.json()["status"] == "COMPLETED", response.text
                assert expected in response.json()["content"], response.text
        assert set(seen) == {"tenant:one", "tenant:two"}
        assert not active
        with psycopg.connect(postgres_dsn) as db:
            assert db.execute(
                "SELECT count(*) FROM ai.agno_jobs WHERE user_id LIKE 'tenant:%' "
                "AND payload::text LIKE '%Bearer synthetic-tenant%'"
            ).fetchone() == (0,)
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/mcp",
        "https://user:secret@example.org/mcp",
        "https://example.org/mcp?token=x",
        "https://example.org/mcp#fragment",
        "https://example.org:99999/mcp",
        "https://example.org\\@other/mcp",
        "https://",
        "http://localhost.example/mcp",
        "http://127.0.0.1.evil/mcp",
    ],
)
def test_remote_mcp_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        Extensions(remote_mcp=(RemoteMCP("remote", url, ("tool",), lambda _: {}),))


def test_remote_mcp_never_follows_redirects_and_resolver_does_not_block():
    received = []

    class Target(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/leak")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    for server in (target, redirect):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    entered, release = threading.Event(), threading.Event()

    def resolve(_principal):
        entered.set()
        assert release.wait(5)
        return {"X-Credential": "synthetic-secret", "Authorization": "Bearer synthetic-secret"}

    ext = Extensions(
        remote_mcp=(
            RemoteMCP("remote", f"http://127.0.0.1:{redirect.server_port}/mcp", ("tool",), resolve),
        )
    )
    ctx = SimpleNamespace(user_id="tenant:user", trusted=SimpleNamespace(scopes=["mcp:remote:run"]))

    async def check():
        task = asyncio.create_task(ext.for_run(ctx, True, approval_tools=False, mcp_attempted={}))
        assert await asyncio.to_thread(entered.wait, 2)
        release.set()
        tools, instructions, _ = await task
        assert not tools and instructions == ["The remote MCP extension is unavailable."]

    try:
        asyncio.run(check())
        assert not received
    finally:
        release.set()
        for server in (redirect, target):
            server.shutdown()
            server.server_close()
