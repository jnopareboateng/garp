import json
import os
import shlex
import signal
import sys
import time

import httpx
import jwt
from agno.skills import Skills
from agno.skills.loaders import LocalSkills
from agno.tools.mcp import MCPTools
from fastapi.testclient import TestClient

from garp import agentos
from garp.agentos import HostAuth, create_agentos_app
from garp.extensions import Extensions
from garp.native_client import NativeChatClient
from garp.terminal import native_chat

TENANT = "extension_tenant"
OWNER = TENANT + ":requester"
REQUEST = ["agents:garp:run", "sessions:read"]
GRANTS = [
    "skills:read",
    "mcp:catalog:run",
    "knowledge:read",
    "actions:record_note:run",
    "approvals:read",
]


def test_tenant_extensions_and_persisted_approval(postgres_dsn, tmp_path, monkeypatch, capfd):
    key = "test-verification-key-with-32-bytes-or-more"
    auth = HostAuth("https://fixture.example", "extensions", key, "HS256")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)
    monkeypatch.setattr(agentos, "generate_title", _title)

    skill = tmp_path / "review-note"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: review-note\ndescription: Review notes using the fixture checklist.\n---\n"
        "Verify that the note has a concrete example.\n"
    )
    skills = Skills([LocalSkills(str(skill))])
    mcp_server = tmp_path / "fixture_mcp.py"
    offline_server = tmp_path / "unavailable_mcp.py"
    offline_server.write_text("raise SystemExit(1)\n")
    mcp_server.write_text('''
import os
from pathlib import Path

from fastmcp import FastMCP

Path(__file__).with_suffix(".pid").write_text(str(os.getpid()))
server = FastMCP("Fixture Catalog")

@server.tool
def fetch_catalog() -> str:
    """Read the fixture catalog."""
    return "The fixture catalog contains cedar."

@server.tool
def private_tool() -> str:
    return "This tool must never be exposed."

server.run(transport="stdio", show_banner=False)
''')
    retrieved = []
    executed = []
    offered = {}
    resumed_models = []
    failed_connections = []

    class UnavailableMCP(MCPTools):
        async def connect(self, force=False):
            if not failed_connections:
                failed_connections.append(True)
                raise RuntimeError("Synthetic transport startup failure")
            return await super().connect(force=force)

    def retrieve(query, num_documents, user_id):
        retrieved.append((query, user_id))
        return [
            {"name": "Fixture Handbook", "content": "The fixture handbook requires two witnesses."}
        ]

    def record_note(note: str) -> str:
        """Record the approved synthetic note."""
        executed.append(note)
        return note

    def create():
        return create_agentos_app(
            postgres_dsn,
            host_auth=auth,
            extensions={
                TENANT: Extensions(
                    skills,
                    (
                        MCPTools(
                            name="catalog",
                            command=f"{shlex.quote(sys.executable)} {shlex.quote(str(mcp_server))}",
                            include_tools=["fetch_catalog"],
                            timeout_seconds=20,
                        ),
                    ),
                    retrieve,
                    (record_note,),
                ),
                "offline-tenant": Extensions(
                    mcp=(
                        UnavailableMCP(
                            name="catalog",
                            command=f"{shlex.quote(sys.executable)} {shlex.quote(str(offline_server))}",
                            include_tools=["fetch_catalog"],
                            timeout_seconds=2,
                        ),
                    )
                ),
            },
        )

    def headers(sub=OWNER, scopes=REQUEST + GRANTS):
        value = jwt.encode(
            {
                "iss": auth.issuer,
                "aud": auth.audience,
                "exp": int(time.time()) + 120,
                "sub": sub,
                "scopes": scopes,
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": f"Bearer {value}"}

    calls = {
        "Read the skill.": ("get_skill_instructions", {"skill_name": "review-note"}),
        "Read the catalog.": ("fetch_catalog", {}),
        "Search the handbook.": ("search_knowledge_base", {"query": "witnesses"}),
        "Record the note.": ("record_note", {"note": "Approval fixture note."}),
    }

    async def respond(request):
        payload = json.loads(request.content)
        assert request.url.host == "openrouter.ai"
        assert (
            payload["provider"]["zdr"] is True and payload["provider"]["allow_fallbacks"] is False
        )
        messages = payload["messages"]
        last = messages[-1]
        text = last.get("content")
        if last["role"] == "tool" and "Approval fixture" in text:
            resumed_models.append((payload["model"], payload.get("reasoning")))
        if last["role"] == "tool":
            answer, finish = {"role": "assistant", "content": text}, "stop"
        else:
            names = {tool["function"]["name"] for tool in payload.get("tools", [])}
            offered[text] = names
            assert "private_tool" not in names and "get_skill_script" not in names
            if text in calls and (tool := calls[text])[0] in names:
                answer = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "fixture-tool",
                            "type": "function",
                            "function": {"name": tool[0], "arguments": json.dumps(tool[1])},
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                answer, finish = (
                    {"role": "assistant", "content": "No permitted fixture tool."},
                    "stop",
                )
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
                        "message": answer,
                        "finish_reason": finish,
                    }
                ],
            },
        )

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(agentos, "AsyncClient", Provider)

    def submit(client, prompt, actor=None, *, background=False, model_options=None):
        response = client.post(
            "/agents/garp/runs",
            headers=actor or headers(),
            data={
                "message": prompt,
                "stream": "false",
                "background": str(background).lower(),
                "factory_input": json.dumps(model_options or {}),
            },
        )
        assert response.status_code == (202 if background else 200), response.text
        return response.json()

    def await_run(client, run, status):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            response = client.get(
                f"/sessions/{run['session_id']}/runs/{run['run_id']}", headers=headers()
            )
            if response.status_code == 200 and response.json()["status"] == status:
                return response.json()
            time.sleep(0.05)
        raise AssertionError(response.text)

    with TestClient(create()) as client:
        assert "concrete example" in submit(client, "Read the skill.")["content"]
        assert "contains cedar" in submit(client, "Read the catalog.")["content"]
        assert (
            submit(client, "Read the catalog.", headers("offline-tenant:requester"))["content"]
            == "No permitted fixture tool."
        )
        assert failed_connections
        pid_file = mcp_server.with_suffix(".pid")
        first_pid = pid_file.read_text()
        os.kill(int(first_pid), signal.SIGTERM)
        # Allow health detection, reconnect backoff and a cold child-process import.
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if (
                pid_file.read_text() != first_pid
                and "contains cedar" in submit(client, "Read the catalog.")["content"]
            ):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("The terminated fixture MCP server was not reconnected")
        assert "two witnesses" in submit(client, "Search the handbook.")["content"]
        assert retrieved == [("witnesses", OWNER)]
        submit(client, "Scope probe.", headers(scopes=REQUEST), background=False)
        submit(client, "Tenant probe.", headers("other-tenant:requester"), background=False)
        denied = {tool[0] for tool in calls.values()}
        assert not (offered["Scope probe."] & denied or offered["Tenant probe."] & denied)

        assert submit(client, "Record the note.")["content"] == "No permitted fixture tool."
        run = submit(client, "Record the note.", background=True, model_options={"effort": "high"})
        await_run(client, run, "PAUSED")
        listing = client.get("/approvals", headers=headers(), params={"run_id": run["run_id"]})
        assert listing.status_code == 200, listing.text
        approval = listing.json()["data"][0]
        assert approval["status"] == "pending" and not executed

    reviewer = headers(TENANT + ":reviewer", ["approvals:write"])
    other = headers("extension-tenant:reviewer", ["approvals:write"])
    # The old default can support effort=high even though the new one cannot.
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", "xiaomi/mimo-v2.6-flash")
    with TestClient(create()) as client:
        assert client.get("/garp/approvals", headers=headers()).status_code == 403
        disguised = client.get("/garp/approvals", headers=other)
        assert disguised.status_code == 200 and disguised.json()["data"] == [], disguised.text
        inbox = client.get("/garp/approvals", headers=reviewer)
        assert inbox.status_code == 200, inbox.text
        found = inbox.json()["data"]
        assert [row["id"] for row in found] == [approval["id"]], found
        assert "tool_args" not in found[0]
        path = f"/garp/approvals/{found[0]['id']}"
        data = {"status": "approved"}
        assert client.get(path, headers=other).status_code == 404
        assert client.post(f"{path}/resolve", headers=other, json=data).status_code == 404
        assert (
            client.post(
                f"{path}/resolve",
                headers=headers(scopes=REQUEST + GRANTS + ["approvals:write"]),
                json=data,
            ).status_code
            == 403
        )
        assert client.post(f"{path}/continue", headers=headers()).status_code == 409
        resolved = client.post(f"{path}/resolve", headers=reviewer, json=data)
        assert (
            resolved.status_code == 200 and resolved.json()["resolved_by"] == TENANT + ":reviewer"
        ), resolved.text
        assert client.get("/garp/approvals", headers=reviewer).json()["data"] == []
        assert not executed
        assert client.post(f"{path}/resolve", headers=reviewer, json=data).status_code == 409
        assert client.post(f"{path}/continue", headers=headers(scopes=REQUEST)).status_code == 403
        assert (
            client.post(
                f"/agents/garp/runs/{run['run_id']}/continue",
                headers=headers(),
                data={"session_id": run["session_id"], "background": "true", "stream": "false"},
            ).status_code
            == 404
        )
        terminal = NativeChatClient("http://localhost", "")
        terminal.close()
        terminal.base, terminal.client = "http://testserver", client
        client.headers.update(headers())
        commands = iter(("/approvals", f"/continue {approval['id']}", "/exit"))
        monkeypatch.setattr("builtins.input", lambda _: next(commands))
        assert native_chat(terminal, run["session_id"]) == 0
        output = capfd.readouterr().out
        assert "approved  record_note" in output
        assert "Continuation queued" in output
        result = await_run(client, run, "COMPLETED")
        assert result["content"] == "Approval fixture note."
        assert executed == ["Approval fixture note."]
        assert resumed_models == [("deepseek/deepseek-v4.1-flash-1017", {"effort": "high"})]
        assert client.post(f"{path}/continue", headers=headers()).status_code == 409

        monkeypatch.setenv("GARP_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")
        rejected = submit(client, "Record the note.", background=True)
        await_run(client, rejected, "PAUSED")
        listing = client.get("/approvals", headers=headers(), params={"run_id": rejected["run_id"]})
        rejected_path = f"/garp/approvals/{listing.json()['data'][0]['id']}"
        assert (
            client.post(
                f"{rejected_path}/resolve", headers=reviewer, json={"status": "rejected"}
            ).status_code
            == 200
        )
        assert client.post(f"{rejected_path}/continue", headers=headers()).status_code == 202
        await_run(client, rejected, "COMPLETED")
        assert executed == ["Approval fixture note."]


async def _title(_message, **_kwargs):
    return "Extension Fixture Chat"
