import asyncio
import json
import secrets
import threading
import time

import httpx
import jwt
import psycopg
import pytest
from agno.factory import FactoryPermissionError, QueuedRequest, RequestContext, TrustedContext
from fastapi.testclient import TestClient

from garp import agentos, executor
from garp.host_grants import check_queued_grants


def test_queued_grant_validation():
    ctx = RequestContext(
        user_id="tenant:user",
        request=QueuedRequest(),
        trusted=TrustedContext(scopes=frozenset({"agents:garp:run", "memories:read"})),
    )

    async def check():
        await check_queued_grants(ctx, None)
        await check_queued_grants(ctx, lambda _: [*ctx.trusted.scopes, "knowledge:read"])
        assert "knowledge:read" not in ctx.trusted.scopes
        specialist = RequestContext(
            user_id="tenant:user",
            request=QueuedRequest(),
            trusted=TrustedContext(scopes=frozenset({"agents:reviewer:run"})),
        )
        await check_queued_grants(specialist, lambda _: ["agents:*:run"])
        with pytest.raises(FactoryPermissionError):
            await check_queued_grants(specialist, lambda _: ["agents:other:run"])
        for resolver in (
            lambda _: [],
            lambda _: None,
            lambda _: "agents:garp:run",
            lambda _: [*ctx.trusted.scopes, "agent_os:admin"],
        ):
            with pytest.raises(FactoryPermissionError):
                await check_queued_grants(ctx, resolver)

        def unavailable(_):
            raise RuntimeError("private host error")

        with pytest.raises(FactoryPermissionError, match="unavailable"):
            await check_queued_grants(ctx, unavailable)
        await check_queued_grants(RequestContext(user_id="tenant:user"), unavailable)

    asyncio.run(check())


@pytest.mark.parametrize("configured", [True, False])
def test_revoked_queued_job_never_reaches_provider(postgres_dsn, monkeypatch, configured):
    from agno.os import job_queue

    key = secrets.token_urlsafe(48)
    host = agentos.HostAuth("https://host.example", "garp", key, "HS256")
    scopes = ["agents:garp:run", "sessions:read", "sessions:write", "memories:read"]
    grants = {"tenant:user": list(scopes)}
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)
    received = []

    async def provider(request):
        received.append(request)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash-1017",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(provider), **kwargs)

    monkeypatch.setattr(agentos, "AsyncClient", Provider)
    monkeypatch.setattr(executor, "AsyncClient", Provider)
    entered, release = threading.Event(), threading.Event()
    original = job_queue.aresolve_factory_from_scope

    async def gated(*args, **kwargs):
        entered.set()
        await asyncio.to_thread(release.wait, 10)
        return await original(*args, **kwargs)

    monkeypatch.setattr(job_queue, "aresolve_factory_from_scope", gated)
    token = jwt.encode(
        {
            "iss": host.issuer,
            "aud": host.audience,
            "sub": "tenant:user",
            "exp": int(time.time()) + 120,
            "scopes": scopes,
            "private": "must-not-enter-ticket",
        },
        key,
        algorithm="HS256",
    )
    resolver = (lambda owner: grants.get(owner, ())) if configured else None
    app = agentos.create_agentos_app(postgres_dsn, host_auth=host, current_grants=resolver)
    with TestClient(app) as client:
        response = client.post(
            "/agents/garp/runs",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": "test", "background": "true", "stream": "false"},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        try:
            assert entered.wait(10)
            with psycopg.connect(postgres_dsn) as db:
                payload = db.execute(
                    "SELECT payload FROM ai.agno_jobs WHERE id=%s", (run_id,)
                ).fetchone()[0]
            assert payload["scope"]["context"]["user_id"] == "tenant:user"
            assert set(payload["scope"]["context"]["scopes"]) == set(scopes)
            assert "must-not-enter-ticket" not in json.dumps(payload)
            grants["tenant:user"].remove("memories:read")
        finally:
            release.set()
        deadline = time.monotonic() + 10
        status = None
        while time.monotonic() < deadline:
            with psycopg.connect(postgres_dsn) as db:
                status = db.execute(
                    "SELECT status FROM ai.agno_jobs WHERE id=%s", (run_id,)
                ).fetchone()[0]
            if status in {"failed", "completed", "cancelled"}:
                break
            time.sleep(0.05)
        assert status == ("failed" if configured else "completed")
        assert bool(received) is not configured
