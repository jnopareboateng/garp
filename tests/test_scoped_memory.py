import json
import secrets
import time

import httpx
import jwt
import psycopg
from fastapi.testclient import TestClient

from garp import agentos, executor
from garp.agentos import HostAuth, create_agentos_app


def test_host_memory_is_scoped_and_recalled_across_sessions(postgres_dsn, monkeypatch):
    key = secrets.token_urlsafe(48)
    issuer, audience = "https://issuer.example", "garp-test"
    scopes = [
        "agents:run",
        "sessions:read",
        "sessions:write",
        "sessions:delete",
        "memories:read",
        "memories:write",
        "memories:delete",
    ]

    def headers(principal, allowed=scopes):
        token = jwt.encode(
            {
                "iss": issuer,
                "aud": audience,
                "sub": principal,
                "iat": int(time.time()),
                "exp": int(time.time()) + 300,
                "scopes": allowed,
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": f"Bearer {token}"}

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)
    wire = []

    async def respond(request):
        payload = json.loads(request.content)
        wire.append(payload)
        assert request.url.host == "openrouter.ai"
        assert payload["provider"]["zdr"] is True
        assert payload["provider"]["allow_fallbacks"] is False
        assert len(payload["provider"]["only"]) == 1
        messages = payload["messages"]
        message = {"role": "assistant", "content": "Fixture response"}
        finish = "stop"
        prompt = messages[-1].get("content")
        if prompt == "Find my past chats" and payload.get("tools"):
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_history",
                        "type": "function",
                        "function": {"name": "search_past_sessions", "arguments": "{}"},
                    }
                ],
            }
            finish = "tool_calls"
        if prompt == "Find the old itinerary" and payload.get("tools"):
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_transcript",
                        "type": "function",
                        "function": {
                            "name": "search_chat_transcripts",
                            "arguments": json.dumps({"query": "literal 100%_match"}),
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

    class FixtureProvider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(agentos, "AsyncClient", FixtureProvider)
    monkeypatch.setattr(executor, "AsyncClient", FixtureProvider)
    app = create_agentos_app(
        postgres_dsn,
        host_auth=HostAuth(
            issuer=issuer, audience=audience, verification_key=key, algorithm="HS256"
        ),
    )
    alice, bob = headers("tenant-a:user"), headers("tenant-b:user")
    alice_fact = "Alice's preferred fruit is starfruit."
    bob_fact = "Bob's preferred fruit is cloudberry."

    with TestClient(app) as client:

        def remember(auth, text):
            response = client.post("/memories", headers=auth, json={"memory": text})
            assert response.status_code == 200, response.text
            return response.json()["memory_id"]

        def chat(auth, prompt, session_id=None):
            start = len(wire)
            response = client.post(
                "/agents/garp/runs",
                headers=auth,
                data={"message": prompt, "stream": "false", "session_id": session_id or ""},
            )
            assert response.status_code == 200, response.text
            # The first call handles the turn. The title call repeats the user input.
            return response.json()["session_id"], wire[start]

        alice_memory = remember(alice, alice_fact)
        bob_memory = remember(bob, bob_fact)
        for auth, own, other in [(alice, alice_fact, bob_fact), (bob, bob_fact, alice_fact)]:
            response = client.get("/memories", headers=auth)
            assert response.status_code == 200, response.text
            assert [item["memory"] for item in response.json()["data"]] == [own]
            assert other not in response.text
        assert client.get(f"/memories/{alice_memory}", headers=bob).status_code == 404
        assert client.get(f"/memories/{bob_memory}", headers=alice).status_code == 404
        # A forged user_id cannot widen reads or deletion.
        response = client.get("/memories", headers=bob, params={"user_id": "tenant-a:user"})
        assert response.status_code == 200, response.text
        assert alice_fact not in response.text
        response = client.delete(
            f"/memories/{alice_memory}", headers=bob, params={"user_id": "tenant-a:user"}
        )
        assert response.status_code in (204, 404), response.text
        assert client.get(f"/memories/{alice_memory}", headers=alice).status_code == 200

        sessions = []
        for auth, own, other in [(alice, alice_fact, bob_fact), (bob, bob_fact, alice_fact)]:
            for _ in range(2):
                session, provider = chat(auth, "What fruit do I prefer?")
                sessions.append(session)
                prompt = json.dumps(provider["messages"])
                assert own in prompt
                assert other not in prompt
        assert len(set(sessions)) == 4

        alice_trip, _ = chat(alice, "My private trip was to Accra")
        chat(bob, "My private trip was to Kumasi")
        start = len(wire)
        _, provider = chat(alice, "Find my past chats")
        assert any(t["function"]["name"] == "search_past_sessions" for t in provider["tools"])
        preview = next(
            m["content"] for call in wire[start:] for m in call["messages"] if m["role"] == "tool"
        )
        assert "Accra" in preview and "Kumasi" not in preview

        marker = "literal 100%_match"
        old, _ = chat(alice, f"{marker} in the Axim itinerary")
        chat(alice, f"{marker} again in the same itinerary", old)
        chat(bob, f"{marker} in a private Salaga itinerary")
        chat(alice, "literal 100XYmatch is a different phrase")
        for _ in range(21):
            assert (
                client.post("/sessions", headers=alice, json={"agent_id": "garp"}).status_code
                == 201
            )
        with psycopg.connect(postgres_dsn) as db:
            db.execute("UPDATE ai.agno_sessions SET created_at=1 WHERE session_id=%s", (old,))

        start = len(wire)
        chat(alice, "Find my past chats")
        preview = next(
            m["content"] for call in wire[start:] for m in call["messages"] if m["role"] == "tool"
        )
        assert old not in preview
        query = {"query": marker.upper()}
        endpoint = "/garp/transcripts/search"
        result = client.post(endpoint, headers=alice, json=query).raise_for_status().json()
        assert [match["session_id"] for match in result["matches"]] == [old]
        assert "again in the same itinerary" in result["matches"][0]["excerpt"]
        assert "Salaga" not in json.dumps(result)
        foreign = client.post(endpoint + "?user_id=tenant-a:user", headers=bob, json=query)
        assert foreign.status_code == 200 and "Salaga" in foreign.text and old not in foreign.text
        assert client.post(endpoint, json=query).status_code == 401
        no_history = headers("tenant-a:user", ["agents:run"])
        assert client.post(endpoint, headers=no_history, json=query).status_code == 403
        assert (
            client.post(
                endpoint, headers=alice, json={**query, "user_id": "tenant-b:user"}
            ).status_code
            == 422
        )
        for invalid in (" ", "a" * 201, "x\0y"):
            assert client.post(endpoint, headers=alice, json={"query": invalid}).status_code == 422
        hidden = (
            client.post(endpoint, headers=alice, json={"query": alice_fact})
            .raise_for_status()
            .json()
        )
        assert not hidden["matches"]

        start = len(wire)
        chat(alice, "Find the old itinerary")
        tool_result = next(
            m["content"] for call in wire[start:] for m in call["messages"] if m["role"] == "tool"
        )
        assert old in tool_result and "Salaga" not in tool_result
        assert client.delete(f"/sessions/{old}", headers=alice).status_code == 204
        assert (
            not client.post(endpoint, headers=alice, json=query)
            .raise_for_status()
            .json()["matches"]
        )

        run_only = headers("tenant-a:user", ["agents:run"])
        _, provider = chat(run_only, "What fruit do I prefer?", alice_trip)
        prompt = json.dumps(provider["messages"])
        assert alice_fact not in prompt and "Accra" not in prompt
        assert "tools" not in provider

        assert client.delete(f"/memories/{alice_memory}", headers=alice).status_code == 204
        assert client.get(f"/memories/{alice_memory}", headers=alice).status_code == 404
        assert client.get(f"/memories/{bob_memory}", headers=bob).status_code == 200
        _, provider = chat(alice, "What fruit do I prefer?")
        assert alice_fact not in json.dumps(provider["messages"])
