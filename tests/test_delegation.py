import asyncio
import json
import secrets
import threading
import time

import httpx
import jwt
import psycopg
from fastapi.testclient import TestClient

from garp import agentos, executor
from garp.agentos import HostAuth, create_agentos_app
from garp.delegation import Specialist
from garp.native_client import NativeChatClient


def test_queued_parent_delegates_to_resumable_tenant_owned_child(postgres_dsn, monkeypatch):
    key = secrets.token_urlsafe(48)
    host = HostAuth("https://host.example", "garp", key, "HS256")
    model = "deepseek/deepseek-v4.1-flash-1017"
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", model)
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)
    monkeypatch.setattr("garp.agentos.generate_title", _fixture_title)
    specialist = Specialist("reviewer", "Reviewer", "Use the specialist fixture instructions.")
    received = []
    child_calls = []
    delegation_clock_delta = 0
    monkeypatch.setattr("garp.delegation.time", lambda: time.time() + delegation_clock_delta)
    started = threading.Event()
    released = threading.Event()
    abandoned = threading.Event()

    def headers(subject, child=True):
        scopes = [
            "agents:garp:run",
            "teams:garp-team:run",
            "sessions:read",
            "sessions:write",
            "sessions:delete",
        ]
        if child:
            scopes.append("agents:reviewer:run")
        token = jwt.encode(
            {
                "iss": host.issuer,
                "aud": host.audience,
                "sub": subject,
                "exp": int(time.time()) + 120,
                "scopes": scopes,
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": f"Bearer {token}"}

    def call_tool(name, args, tool_id):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }

    async def respond(request):
        nonlocal delegation_clock_delta
        body = json.loads(request.content)
        assert request.url.host == "openrouter.ai"
        assert body["provider"] == {"zdr": True, "only": ["fireworks/us"], "allow_fallbacks": False}
        messages = body["messages"]
        last = messages[-1]
        text = last.get("content")
        tools = {tool["function"]["name"] for tool in body.get("tools", [])}
        if "Use the specialist fixture instructions" in messages[0].get("content", ""):
            child_calls.append(messages)
            assert not tools
            if text == "Wait for cancellation.":
                started.set()
                await asyncio.to_thread(released.wait, 8)
            answer = {"role": "assistant", "content": "Independent child result"}
        elif last["role"] == "tool":
            data = json.loads(text) if text.startswith("{") else {"result": text}
            received.append(data)
            answer = {"role": "assistant", "content": data.get("result") or "Specialist queued"}
            if len(received) == 1:
                delegation_clock_delta = 1000
                answer = call_tool(
                    "delegate_task",
                    {"agent_id": "reviewer", "task": "Check the draft for gaps."},
                    "spawn-again",
                )
            elif len(received) == 2:
                delegation_clock_delta = 0
                answer = {"role": "assistant", "content": "Specialist queued"}
        elif text == "Ask a reviewer to check the draft.":
            assert "delegate_task" in tools
            answer = call_tool(
                "delegate_task",
                {"agent_id": "reviewer", "task": "Check the draft for gaps."},
                "spawn",
            )
        elif text == "Read the child result.":
            answer = call_tool("read_specialist_result", {"run_id": received[0]["run_id"]}, "read")
        elif text == "Read the team's child result.":
            answer = call_tool(
                "read_specialist_result", {"run_id": received[-1]["run_id"]}, "team-read"
            )
        elif text == "Use the team for a quick review.":
            assert "delegate_task_to_member" in tools
            answer = call_tool(
                "delegate_task_to_member",
                {"member_id": "reviewer", "task": "Quick inline review."},
                "team",
            )
        elif text == "Start a cancellable specialist.":
            answer = call_tool(
                "delegate_task",
                {"agent_id": "reviewer", "task": "Wait for cancellation."},
                "cancel-probe",
            )
        elif text == "Start a recoverable specialist.":
            answer = call_tool(
                "delegate_task",
                {"agent_id": "reviewer", "task": "Recover my review."},
                "recovery-probe",
            )
        else:
            assert "delegate_task" not in tools
            answer = {"role": "assistant", "content": "No specialist permission"}
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": answer,
                        "finish_reason": "tool_calls" if "tool_calls" in answer else "stop",
                    }
                ],
            },
        )

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(agentos, "AsyncClient", Provider)
    monkeypatch.setattr(executor, "AsyncClient", Provider)
    owner = headers("org-a:delegate-owner")
    other = headers("org-b:delegate-owner")
    restricted = headers("org-a:delegate-owner", child=False)

    def app():
        return create_agentos_app(postgres_dsn, host_auth=host, specialists=(specialist,))

    def submit(client, auth, agent, message, *, session=None, background=False):
        response = client.post(
            f"/agents/{agent}/runs",
            headers=auth,
            data={
                "message": message,
                "stream": "false",
                "session_id": session or "",
                "background": str(background).lower(),
            },
        )
        assert response.status_code == (202 if background else 200), response.text
        return response.json()

    def completed(client, auth, session, run):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = client.get(f"/sessions/{session}/runs/{run}", headers=auth)
            if response.status_code == 200 and response.json()["status"] == "COMPLETED":
                return response.json()
            time.sleep(0.05)
        raise AssertionError(response.text)

    with TestClient(app()) as client:
        assert client.get("/garp/models", headers=restricted).json()["specialists"] == []
        reply = submit(client, restricted, "garp", "Who can help?")
        assert reply["content"] == "No specialist permission"
        denied = client.post(
            "/agents/reviewer/runs", headers=restricted, data={"message": "Denied"}
        )
        assert denied.status_code == 403
        parent = submit(
            client, owner, "garp", "Ask a reviewer to check the draft.", background=True
        )
        parent_run = completed(client, owner, parent["session_id"], parent["run_id"])
        assert parent_run["content"] == "Specialist queued"
        child = received[0]
        assert received[1]["run_id"] == child["run_id"]
        assert child["agent_id"] == "reviewer" and child["session_id"] != parent["session_id"]
        child_run = completed(client, owner, child["session_id"], child["run_id"])
        assert child_run["metadata"]["garp_parent_run_id"] == parent["run_id"]
        assert child_run["content"] == "Independent child result"

    with psycopg.connect(postgres_dsn) as connection:
        (ticket_payload,) = connection.execute(
            "SELECT payload FROM ai.agno_jobs WHERE id = %s", (child["run_id"],)
        ).fetchone()
        assert ticket_payload["scope"]["context"]["input"]["model_id"] == model
        connection.execute("DELETE FROM ai.agno_jobs WHERE id = %s", (child["run_id"],))

    with TestClient(app()) as client:
        assert client.get(f"/sessions/{child['session_id']}", headers=other).status_code == 404
        resumed = submit(
            client, owner, "reviewer", "Continue reviewing.", session=child["session_id"]
        )
        assert resumed["session_id"] == child["session_id"]
        assert any(msg.get("content") == "Check the draft for gaps." for msg in child_calls[-1])
        summary = submit(
            client, owner, "garp", "Read the child result.", session=parent["session_id"]
        )
        assert summary["content"] == "Independent child result"
        new_chat = submit(client, owner, "garp", "Read the child result.")
        assert new_chat["content"] == "Specialist run not found."

        submit(
            client, owner, "garp", "Start a cancellable specialist.", session=parent["session_id"]
        )
        active_child = received[-1]
        try:
            assert started.wait(5)
            sid, rid = active_child["session_id"], active_child["run_id"]
            cancel = f"/agents/reviewer/runs/{rid}/cancel"
            assert client.post(cancel, headers=other, params={"session_id": sid}).status_code == 404
            assert (
                client.post(cancel, headers=restricted, params={"session_id": sid}).status_code
                == 403
            )
            native = NativeChatClient("http://localhost", "")
            native.close()
            native.base = "http://testserver"
            native.client = client
            client.headers.update(owner)
            assert native.session(sid)["agent_id"] == "reviewer"
            assert native.cancel_pending(sid)
        finally:
            released.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            stopped = client.get(f"/sessions/{sid}/runs/{rid}", headers=owner)
            if stopped.json().get("status") == "CANCELLED":
                break
            time.sleep(0.05)
        assert stopped.json()["status"] == "CANCELLED", stopped.text

        payload = {"message": "Use the team for a quick review.", "stream": "false"}
        denied_team = client.post("/teams/garp-team/runs", headers=restricted, data=payload)
        assert denied_team.status_code == 403
        team = client.post("/teams/garp-team/runs", headers=owner, data=payload)
        assert team.status_code == 200, team.text
        assert team.json()["content"] == "Independent child result"
        assert team.json()["member_responses"][0]["agent_id"] == "reviewer"
        assert team.json()["member_responses"][0]["session_id"] == team.json()["session_id"]
        team_id = team.json()["session_id"]
        team_child = client.post(
            "/teams/garp-team/runs",
            headers=owner,
            data={
                "message": "Ask a reviewer to check the draft.",
                "stream": "false",
                "background": "true",
                "session_id": team_id,
            },
        )
        assert team_child.status_code == 202, team_child.text
        team_run = completed(client, owner, team_id, team_child.json()["run_id"])
        assert team_run["content"] == "Specialist queued"
        spawned = received[-1]
        team_child_run = completed(client, owner, spawned["session_id"], spawned["run_id"])
        assert team_child_run["metadata"]["garp_parent_session_id"] == team_id
        read = client.post(
            "/teams/garp-team/runs",
            headers=owner,
            data={
                "message": "Read the team's child result.",
                "stream": "false",
                "session_id": team_id,
            },
        )
        assert read.status_code == 200 and read.json()["content"] == "Independent child result", (
            read.text
        )

        claim = agentos.PostgresDb.claim_job

        def abandon_child(db, *args):
            ticket = claim(db, *args)
            if ticket and ticket["payload"].get("input") == "Recover my review.":
                # The real Postgres claim committed. Simulate process loss before the worker runs it.
                abandoned.set()
                return None
            return ticket

        with monkeypatch.context() as stopped_worker:
            stopped_worker.setattr(agentos.PostgresDb, "claim_job", abandon_child)
            submit(
                client,
                owner,
                "garp",
                "Start a recoverable specialist.",
                session=parent["session_id"],
            )
            lost = received[-1]
            assert abandoned.wait(5)
            assert not any(
                messages[-1].get("content") == "Recover my review." for messages in child_calls
            )

    with psycopg.connect(postgres_dsn) as connection:
        (attempt,) = connection.execute(
            "UPDATE ai.agno_jobs SET locked_at = 0 WHERE id = %s AND status = 'running' RETURNING attempt",
            (lost["run_id"],),
        ).fetchone()
        assert attempt == 1
    with TestClient(app()) as client:
        result = completed(client, owner, lost["session_id"], lost["run_id"])
        assert result["content"] == "Independent child result"
        assert client.delete(f"/sessions/{parent['session_id']}", headers=owner).status_code == 204
        assert client.get(f"/sessions/{lost['session_id']}", headers=owner).status_code == 200
    with psycopg.connect(postgres_dsn) as connection:
        (attempt, status) = connection.execute(
            "SELECT attempt, status FROM ai.agno_jobs WHERE id = %s", (lost["run_id"],)
        ).fetchone()
        assert (attempt, status) == (2, "completed")


async def _fixture_title(_message, **_kwargs):
    return "Fixture Chat"
