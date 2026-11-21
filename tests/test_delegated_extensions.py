import json
import shlex
import sys
import threading
import time

import httpx
import jwt
import psycopg
import pytest
from agno.skills import Skills
from agno.skills.loaders import LocalSkills
from agno.tools.mcp import MCPTools
from fastapi.testclient import TestClient

from garp import agentos
from garp.agentos import HostAuth, create_agentos_app
from garp.delegation import Specialist, specialist_extension_scopes
from garp.extensions import Extensions


def test_specialist_grant_validation():
    for scopes in [("mcp:*:run",), ("actions:write:run",), ("skills:read", "skills:read")]:
        with pytest.raises(ValueError):
            Specialist("reviewer", "Reviewer", "Review", scopes)
    spec = Specialist("reviewer", "Reviewer", "Review", ("mcp:catalog:run", "skills:read"))
    assert specialist_extension_scopes(spec, ["mcp:*:run"], hosted=True) == ["mcp:catalog:run"]
    assert specialist_extension_scopes(spec, [], hosted=True) == []


def test_specialist_extension_grants(postgres_dsn, tmp_path, monkeypatch):
    key = "fixture-key-with-at-least-32-bytes-long"
    auth = HostAuth("https://fixture.example", "garp", key, "HS256")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)

    async def title(*args, **kwargs):
        return "Fixture"

    monkeypatch.setattr(agentos, "generate_title", title)
    skill = tmp_path / "checklist"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: checklist\ndescription: fixture checklist\n---\nUse the blue checklist.\n"
    )
    server = tmp_path / "fixture_server.py"
    server.write_text(
        'from fastmcp import FastMCP\ns = FastMCP("fixture")\n@s.tool\ndef catalog() -> str:\n    """Find fixture."""\n    return "cedar catalog"\ns.run(transport="stdio", show_banner=False)\n'
    )
    retrievals = []

    def retrieve(query, num_documents, user_id):
        retrievals.append(user_id)
        return [{"name": "handbook", "content": "two witnesses"}]

    def record_note(note: str):
        return note

    config = Extensions(
        skills=Skills([LocalSkills(str(skill))]),
        mcp=(
            MCPTools(
                name="catalog",
                command=f"{shlex.quote(sys.executable)} {shlex.quote(str(server))}",
                include_tools=["catalog"],
                timeout_seconds=10,
            ),
        ),
        retrieve=retrieve,
        approval_tools=(record_note,),
    )
    specs = (
        Specialist(
            "reviewer",
            "Reviewer",
            "Specialist fixture.",
            ("skills:read", "mcp:catalog:run", "knowledge:read"),
        ),
        Specialist("plain", "Plain", "Specialist fixture."),
    )
    offered = {}
    children = []
    calls = {
        "skill": ("get_skill_instructions", {"skill_name": "checklist"}),
        "mcp": ("catalog", {}),
        "knowledge": ("search_knowledge_base", {"query": "witnesses"}),
    }

    async def respond(request):
        body = json.loads(request.content)
        messages = body["messages"]
        last = messages[-1]
        content = last.get("content") or ""
        names = {t["function"]["name"] for t in body.get("tools", [])}
        if last["role"] == "tool":
            if content.startswith('{"agent_id"'):
                children.append(json.loads(content))
                answer = {"role": "assistant", "content": "queued"}
            else:
                answer = {"role": "assistant", "content": content}
        else:
            offered[content] = names
            if content.startswith("queue "):
                tool, args = "delegate_task", {"agent_id": "reviewer", "task": content[6:]}
            elif content == "inline":
                tool, args = (
                    "delegate_task_to_member",
                    {"member_id": "reviewer", "task": "inline-check"},
                )
            elif content in calls and calls[content][0] in names:
                tool, args = calls[content]
            else:
                tool = None
            answer = (
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {"name": tool, "arguments": json.dumps(args)},
                        }
                    ],
                }
                if tool
                else {"role": "assistant", "content": "no tool"}
            )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
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

    def headers(subject="tenant:owner", grants=()):
        token = jwt.encode(
            {
                "iss": auth.issuer,
                "aud": auth.audience,
                "exp": int(time.time()) + 120,
                "sub": subject,
                "scopes": [
                    "agents:garp:run",
                    "agents:reviewer:run",
                    "agents:plain:run",
                    "teams:garp-team:run",
                    "sessions:read",
                    *grants,
                ],
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": "Bearer " + token}

    full = ("skills:read", "mcp:catalog:run", "knowledge:read", "actions:record_note:run")

    def run(client, agent, text, actor):
        result = client.post(
            f"/agents/{agent}/runs", headers=actor, data={"message": text, "stream": "false"}
        )
        assert result.status_code == 200, result.text
        return result.json()

    with TestClient(
        create_agentos_app(
            postgres_dsn, host_auth=auth, specialists=specs, extensions={"tenant": config}
        )
    ) as client:
        assert "blue checklist" in run(client, "reviewer", "skill", headers(grants=full))["content"]
        assert "cedar catalog" in run(client, "reviewer", "mcp", headers(grants=full))["content"]
        assert (
            "two witnesses" in run(client, "reviewer", "knowledge", headers(grants=full))["content"]
        )
        assert retrievals == ["tenant:owner"]
        assert "record_note" not in offered["knowledge"]
        assert run(client, "reviewer", "mcp", headers())["content"] == "no tool"
        assert run(client, "plain", "mcp", headers(grants=full))["content"] == "no tool"
        assert run(client, "reviewer", "mcp", headers("other:owner", full))["content"] == "no tool"
        assert "cedar catalog" in run(client, "garp", "mcp", headers(grants=full))["content"]
        team = client.post(
            "/teams/garp-team/runs",
            headers=headers(grants=full),
            data={"message": "inline", "stream": "false"},
        )
        assert team.status_code == 200, team.text
        assert not (
            {"catalog", "get_skill_instructions", "search_knowledge_base"} & offered["inline-check"]
        )
        for task, actor, expected, attempts_expected in (
            ("skill", headers(grants=full), "blue checklist", 1),
            ("mcp", headers(grants=full), "cedar catalog", 1),
            ("knowledge", headers(grants=full), "two witnesses", 1),
            ("mcp", headers(), "no tool", 2),
            ("mcp", headers("other:owner", full), "no tool", 1),
        ):
            run(client, "garp", "queue " + task, actor)
            child = children[-1]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                result = client.get(
                    f"/sessions/{child['session_id']}/runs/{child['run_id']}", headers=actor
                )
                if result.status_code == 200 and result.json().get("status") == "COMPLETED":
                    break
                time.sleep(0.05)
            assert expected in result.json()["content"], result.text
            with psycopg.connect(postgres_dsn) as connection:
                payload, attempts = connection.execute(
                    "SELECT payload, max_attempts FROM ai.agno_jobs WHERE id = %s",
                    (child["run_id"],),
                ).fetchone()
                assert attempts == attempts_expected
                serialized = json.dumps(payload)
                assert "Bearer" not in serialized and "Authorization" not in serialized

        claimed = threading.Event()
        original_claim = agentos.PostgresDb.claim_job

        def lose_claim(db, *args):
            ticket = original_claim(db, *args)
            if ticket and ticket["payload"].get("input") == "retry-probe":
                claimed.set()
                return None
            return ticket

        monkeypatch.setattr(agentos.PostgresDb, "claim_job", lose_claim)
        run(client, "garp", "queue retry-probe", headers(grants=full))
        lost = children[-1]
        assert claimed.wait(5)
        with psycopg.connect(postgres_dsn) as connection:
            connection.execute(
                "UPDATE ai.agno_jobs SET locked_at = 0 WHERE id = %s", (lost["run_id"],)
            )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with psycopg.connect(postgres_dsn) as connection:
                attempt, status = connection.execute(
                    "SELECT attempt, status FROM ai.agno_jobs WHERE id = %s", (lost["run_id"],)
                ).fetchone()
            if status == "failed":
                break
            time.sleep(0.1)
        assert (attempt, status) == (1, "failed")
        assert "retry-probe" not in offered
