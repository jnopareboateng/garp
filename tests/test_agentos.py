import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from garp.agentos import AgentSettings, create_agentos_app


def test_settings_reject_unapproved_models_and_options():
    with pytest.raises(ValidationError):
        AgentSettings(model_id="meta/muse-ember")
    with pytest.raises(ValidationError):
        AgentSettings(model_id="deepseek/deepseek-v4.1-flash-1017", web_search=True)
    with pytest.raises(ValidationError):
        AgentSettings(unexpected=True)


def test_native_app_auth_and_sessions(postgres_dsn, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    app = create_agentos_app(postgres_dsn, "secret")
    with TestClient(app) as client:
        assert client.get("/garp/models").status_code == 401
        assert client.get("/sessions", headers={"Authorization": "Bearer wrong"}).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/garp/models", headers=headers).status_code == 200
        assert client.post("/optimize-memories", headers=headers, json={}).status_code == 404
        response = client.post(
            "/sessions?type=agent", headers=headers, json={"agent_id": "garp", "user_id": "user"}
        )
        assert response.status_code in (200, 201), response.text
        session_id = response.json()["session_id"]
        response = client.get(f"/sessions/{session_id}?user_id=user", headers=headers)
        assert response.status_code in (200, 201), response.text
        response = client.post(
            "/agents/garp/runs",
            headers=headers,
            data={"message": "hello", "factory_input": '{"model_id":"meta/muse-ember"}'},
        )
        assert response.status_code == 400


@pytest.mark.parametrize("hosted", [False, True])
def test_durable_factory_stream_reattaches_and_reopens(postgres_dsn, monkeypatch, hosted):
    import json
    import secrets
    import time
    from contextlib import contextmanager

    import httpx
    import jwt
    import psycopg

    from garp import agentos, executor
    from garp.demo import _serve_app
    from garp.native_client import NativeChatClient

    terra = "openai/gpt-6.1-terra"
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", terra)
    monkeypatch.setenv("GARP_OPENROUTER_NATIVE_SEARCH_ONLY", "true")
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)
    wire = []

    async def respond(request):
        payload = json.loads(request.content)
        wire.append(payload)
        assert request.url.host == "openrouter.ai"
        assert payload["provider"]["zdr"] is True
        assert payload["provider"]["only"] == ["azure"]
        assert payload["provider"]["allow_fallbacks"] is False
        if not payload.get("stream"):
            return httpx.Response(
                200,
                json={
                    "id": "title",
                    "object": "chat.completion",
                    "created": 1,
                    "model": terra,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Native Stream Proof Extra",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        events = []
        for delta, reason in [
            ({"role": "assistant", "content": "first "}, None),
            (
                {
                    "content": "second",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.org/evidence",
                                "title": "Fixture source",
                                "start_index": 0,
                                "end_index": 6,
                            },
                        }
                    ],
                },
                None,
            ),
            ({}, "stop"),
        ]:
            event = {
                "id": "response",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": terra,
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            events.append(f"data: {json.dumps(event)}\n\n")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(events) + "data: [DONE]\n\n",
        )

    class FixtureProvider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(agentos, "AsyncClient", FixtureProvider)
    monkeypatch.setattr(executor, "AsyncClient", FixtureProvider)

    token = "local-fixture-token"
    config = {"token": token}
    if hosted:
        key = secrets.token_urlsafe(48)
        config = {"host_auth": agentos.HostAuth("https://host.local", "garp", key, "HS256")}

        def host_token(principal):
            return jwt.encode(
                {
                    "iss": "https://host.local",
                    "aud": "garp",
                    "sub": principal,
                    "exp": int(time.time()) + 120,
                    "scopes": ["agents:garp:run", "sessions:read", "sessions:write"],
                    "email": "private-profile-claim@example.invalid",
                },
                key,
                algorithm="HS256",
            )

        token = host_token("org-a:user")

    app = create_agentos_app(postgres_dsn, **config)
    with _serve_app(app) as base:
        client = NativeChatClient(base, token)
        try:
            assert client.catalog()["durable_stream_reconnect"]
            session_id = client.create_session()
            client.set_options(session_id, {"model": terra, "effort": "high", "web_search": True})
            stream = client.client.stream
            requests = []

            @contextmanager
            def interrupt_once(method, url, **kwargs):
                requests.append((url, kwargs["headers"]["Idempotency-Key"]))
                with stream(method, url, **kwargs) as response:
                    if len(requests) == 1:
                        iter_lines = response.iter_lines

                        def broken():
                            yield from iter_lines()
                            raise httpx.ReadError("fixture transport cut")

                        def stop_after_first_chunk():
                            seen_delta = False
                            for line in broken():
                                if (
                                    '"event":"RunContent"' in line
                                    or '"event": "RunContent"' in line
                                ):
                                    seen_delta = True
                                yield line
                                if seen_delta and not line:
                                    raise httpx.ReadError("fixture transport cut")

                        response.iter_lines = stop_after_first_chunk
                    yield response

            client.client.stream = interrupt_once
            printed = []
            turn = client.turn(
                session_id,
                "Check native reconnect",
                on_delta=printed.append,
                model=terra,
                effort="high",
                web_search=True,
            )
            assert printed == ["first ", "second"]
            assert turn["content"] == "first second"
            assert turn["sources"] == [
                {"title": "Fixture source", "url": "https://example.org/evidence"}
            ]
            assert len(requests) == 3 and requests[1][0].endswith("/resume")
            assert requests[0] == requests[2]
            assert client.session(session_id)["title"] == "Native Stream Proof"
            client.set_options(session_id, {"effort": "xhigh"})
            assert client.session(session_id)["title"] == "Native Stream Proof"
        finally:
            client.close()

    assert len(wire) == 2
    assert wire[0]["reasoning"] == {"effort": "high"}
    assert {"type": "openrouter:web_search", "parameters": {"engine": "native"}} in wire[0]["tools"]
    assert any(
        tool.get("function", {}).get("name") == "search_past_sessions" for tool in wire[0]["tools"]
    )
    assert "tools" not in wire[1]
    with psycopg.connect(postgres_dsn) as connection:
        rows = connection.execute(
            "SELECT id, status, payload FROM ai.agno_jobs WHERE session_id = %s", (session_id,)
        ).fetchall()
    assert len(rows) == 1 and rows[0][:2] == (turn["run_id"], "completed")
    assert "private-profile-claim" not in json.dumps(rows[0][2])

    with _serve_app(create_agentos_app(postgres_dsn, **config)) as base:
        client = NativeChatClient(base, token)
        try:
            assert any(row["id"] == session_id for row in client.list_sessions())
            assert client.list_sessions("Stream Proof") == [
                {"id": session_id, "title": "Native Stream Proof"}
            ]
            assert client.list_sessions("title that does not match") == []
            assert client.session(session_id)["effort"] == "xhigh"
            assert client.history(session_id) == [
                {"role": "user", "content": "Check native reconnect"},
                {
                    "role": "assistant",
                    "content": "first second",
                    "sources": [{"title": "Fixture source", "url": "https://example.org/evidence"}],
                },
            ]
            if hosted:
                other = NativeChatClient(base, host_token("org-b:user"))
                try:
                    assert other.list_sessions() == []
                    with pytest.raises(RuntimeError, match="HTTP 404"):
                        other.history(session_id)
                finally:
                    other.close()
        finally:
            client.close()
