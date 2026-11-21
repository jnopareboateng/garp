import json
import logging
import secrets
import time
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import uuid4

import httpx
import jwt
import psycopg
import pytest
from agno.agent import Agent
from fastapi.testclient import TestClient
from openinference.instrumentation import using_user
from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from sqlalchemy import URL
from sqlalchemy.exc import IntegrityError

from garp import agentos, observability
from garp.agentos import HostAuth, create_agentos_app
from garp.delegation import Specialist
from garp.executor import _model
from garp.extensions import Extensions


def test_hosted_traces_are_redacted_and_owner_scoped(postgres_dsn, monkeypatch, capfd, caplog):
    owner, other = "trace-org:owner", "other-org:owner"
    prompt, output, tool_input = "PRIVATE_PROMPT", "PRIVATE_OUTPUT", "PRIVATE_TOOL_INPUT"
    key = secrets.token_urlsafe(40)
    model = "deepseek/deepseek-v4.1-flash-1017"
    auth = HostAuth("https://fixture.example", "trace-fixture", key, "HS256")
    markers = (
        prompt,
        output,
        tool_input,
        "PRIVATE_EXCEPTION",
        "PRIVATE_TEAM_PROMPT",
        "PRIVATE_DB",
        "PRIVATE_UNHANDLED",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")
    monkeypatch.setenv("GARP_OPENROUTER_MODEL", model)
    monkeypatch.delenv("GARP_OPENROUTER_ENDPOINT", raising=False)

    async def title(*args, **kwargs):
        return "Private Fixture Name"

    def fixture_tool(query: str) -> str:
        """Raise the synthetic tool error."""
        raise ValueError("PRIVATE_EXCEPTION")

    extension_tools = Extensions.for_run

    async def for_run(configured, *args, **kwargs):
        tools, instructions, retrieve = await extension_tools(configured, *args, **kwargs)
        return [*tools, fixture_tool], instructions, retrieve

    async def respond(request):
        payload = json.loads(request.content)
        last = payload["messages"][-1]
        if last.get("content") == prompt:
            answer = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "fixture-tool",
                        "type": "function",
                        "function": {
                            "name": "fixture_tool",
                            "arguments": json.dumps({"query": tool_input}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            answer, finish = {"role": "assistant", "content": output}, "stop"
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "created": 1,
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": answer, "finish_reason": finish}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26},
            },
        )

    class Provider(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(agentos, "AsyncClient", Provider)
    monkeypatch.setattr(agentos, "generate_title", title)
    monkeypatch.setattr(Extensions, "for_run", for_run)
    app = create_agentos_app(
        postgres_dsn,
        host_auth=auth,
        tracing=True,
        specialists=(Specialist("trace-reviewer", "Trace Reviewer", "Review fixture."),),
    )

    @app.get("/_fixture-error", operation_id="health_check")
    async def broken_route():
        raise RuntimeError("PRIVATE_UNHANDLED")

    async def ambient_host(scope, receive, send):
        if scope["type"] == "http":
            with using_user(other):
                await app(scope, receive, send)
        else:
            await app(scope, receive, send)

    scopes = [
        "agents:garp:run",
        "teams:garp-team:run",
        "agents:trace-reviewer:run",
        "sessions:read",
        "sessions:delete",
        "knowledge:read",
        "traces:read",
    ]

    def headers(subject, grants=scopes):
        token = jwt.encode(
            {
                "sub": subject,
                "iss": auth.issuer,
                "aud": auth.audience,
                "exp": int(time.time()) + 120,
                "scopes": grants,
            },
            key,
            algorithm="HS256",
        )
        return {"Authorization": f"Bearer {token}"}

    original_factory = logging.getLogRecordFactory()
    published = []
    with TestClient(ambient_host) as client:
        broken = client.get("/_fixture-error", headers=headers(owner))
        assert broken.status_code == 500 and broken.json()["detail"] == "AgentOS request failed"
        own = client.post(
            "/agents/garp/runs",
            headers=headers(owner),
            data={"message": prompt, "user_id": other, "stream": "false"},
        )
        assert own.status_code == 200, own.text
        assert own.json()["content"] == output, own.text
        queued = client.post(
            "/teams/garp-team/runs",
            headers=headers(other),
            data={"message": "PRIVATE_TEAM_PROMPT", "background": "true", "stream": "false"},
        )
        assert queued.status_code == 202, queued.text
        for run, actor in ((own.json(), owner), (queued.json(), other)):
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                listing = client.get(
                    "/traces", headers=headers(actor), params={"run_id": run["run_id"]}
                )
                assert listing.status_code == 200, listing.text
                traces = listing.json()["data"]
                if traces:
                    path = f"/traces/{traces[0]['trace_id']}"
                    detail = client.get(path, headers=headers(actor))
                    assert detail.status_code == 200, detail.text
                    if detail.json()["tree"]:
                        break
                time.sleep(0.1)
            assert len(traces) == 1 and traces[0]["user_id"] == actor, listing.text
            assert model in detail.text and "input_tokens" in detail.text, detail.text
            if actor == owner:
                assert "fixture_tool" in detail.text and '"status":"ERROR"' in detail.text
            assert not any(value in detail.text for value in markers)
            stranger = other if actor == owner else owner
            span = detail.json()["tree"][0]["id"]
            published.append((path, span, actor))
            assert (
                client.get(path, headers=headers(actor), params={"span_id": span}).status_code
                == 200
            )
            assert client.get(path, headers=headers(stranger)).status_code == 404
            assert (
                client.get(path, headers=headers(stranger), params={"span_id": span}).status_code
                == 404
            )
            assert client.get(path, headers=headers(actor, [])).status_code == 403
            forged = client.get("/traces", headers=headers(stranger), params={"user_id": actor})
            assert forged.status_code == 200, forged.text
            assert all(row["user_id"] == stranger for row in forged.json()["data"])

        path, _, actor = published[0]
        foreign_span = published[1][1]
        response = client.get(path, headers=headers(actor), params={"span_id": foreign_span})
        assert response.status_code == 404, response.text
        missing = client.get(path, headers=headers(actor), params={"span_id": "missing"})
        assert response.json() == missing.json()

        def broken_list(*args, **kwargs):
            raise RuntimeError("PRIVATE_DB")

        with monkeypatch.context() as errors:
            errors.setattr(agentos.PostgresDb, "get_traces", broken_list)
            unavailable = client.get("/traces", headers=headers(owner))
            assert unavailable.status_code == 500, unavailable.text
            assert unavailable.json()["detail"] == "Trace service unavailable"

        async def unrelated():
            async with Provider() as provider:
                return await Agent(model=_model(provider), telemetry=False).arun(
                    "UNRELATED", user_id=owner
                )

        unrelated_run = client.portal.call(unrelated)
        assert (
            client.delete(
                f"/sessions/{own.json()['session_id']}", headers=headers(owner)
            ).status_code
            == 204
        )
        assert client.get(published[0][0], headers=headers(owner)).status_code == 404
        assert client.get(published[1][0], headers=headers(other)).status_code == 200

    assert logging.getLogRecordFactory() is original_factory
    captured = capfd.readouterr()
    logs = captured.out + captured.err + caplog.text
    assert "Runtime detail redacted" in logs
    assert not any(value in logs for value in markers)
    with psycopg.connect(postgres_dsn) as db:
        spans = db.execute("SELECT attributes, status_message FROM ai.garp_spans").fetchall()
        assert any(row[0].get("llm.token_count.total") == 26 for row in spans)
        assert all(message is None for _, message in spans)
        for fields, _ in spans:
            assert (
                not {"input.value", "output.value", "tool.parameters", "exception.message"}
                & fields.keys()
            )
        assert not db.execute(
            "SELECT 1 FROM ai.garp_traces WHERE run_id=%s", (unrelated_run.run_id,)
        ).fetchone()

    with TestClient(create_agentos_app(postgres_dsn, host_auth=auth)) as disabled:
        assert disabled.get("/traces", headers=headers(owner)).status_code == 404


def test_tracing_start_failure_preserves_the_error_and_logging_factory(monkeypatch):
    original_factory = logging.getLogRecordFactory()
    monkeypatch.setattr(
        AgnoInstrumentor, "is_instrumented_by_opentelemetry", property(lambda _: True)
    )
    app = create_agentos_app("postgresql://unused", "test-token", tracing=True)
    with pytest.raises(RuntimeError, match="Garp tracing requires control"), TestClient(app):
        pytest.fail("Conflicting instrumentation allowed Garp to start")
    assert logging.getLogRecordFactory() is original_factory


def test_trace_retention_prunes_in_batches_and_recovers_on_restart(
    postgres_dsn, monkeypatch, caplog
):
    name = "garp_trace_retention_" + uuid4().hex
    with psycopg.connect(postgres_dsn, autocommit=True) as source:
        source.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    dsn = make_conninfo(postgres_dsn, dbname=name)
    db = agentos.PostgresDb(
        db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(dsn)),
        traces_table="garp_traces",
        spans_table="garp_spans",
    )
    traces = db._get_table("traces", create_table_if_not_found=True)
    spans = db._get_table("spans", create_table_if_not_found=True)
    now = datetime.now(UTC)

    def record(trace_id, owner, ended):
        start = (now - timedelta(days=6)).isoformat()
        finish = ended.isoformat()
        with db.Session() as conn, conn.begin():
            conn.execute(
                traces.insert().values(
                    trace_id=trace_id,
                    name="Agent",
                    status="OK",
                    start_time=start,
                    end_time=finish,
                    duration_ms=0,
                    created_at=start,
                    user_id=owner,
                )
            )
            conn.execute(
                spans.insert().values(
                    span_id=trace_id,
                    trace_id=trace_id,
                    name="Agent",
                    span_kind="INTERNAL",
                    status_code="OK",
                    attributes={},
                    start_time=start,
                    end_time=finish,
                    duration_ms=0,
                    created_at=finish,
                )
            )

    def remaining():
        with db.Session() as conn:
            return set(
                conn.execute(traces.select().with_only_columns(traces.c.trace_id)).scalars()
            ), set(conn.execute(spans.select().with_only_columns(spans.c.trace_id)).scalars())

    for i in range(3):
        record(f"expired-{i}", "trace-org:owner", now - timedelta(days=2))
    record("recent", "other-org:owner", now - timedelta(minutes=20))
    assert remaining() == ({"expired-0", "expired-1", "expired-2", "recent"},) * 2

    with TestClient(create_agentos_app(dsn, "fixture-token")):
        assert len(remaining()[0]) == 4

    monkeypatch.setattr(observability, "_TRACE_DELETE_BATCH", 2)
    monkeypatch.setattr(observability, "_TRACE_SWEEP_INTERVAL", 0.05)
    fail_next = Event()
    prune = observability._delete_expired_traces

    def unstable(*args):
        if fail_next.is_set():
            fail_next.clear()
            raise OSError("Trace retention fixture detail")
        return prune(*args)

    monkeypatch.setattr(observability, "_delete_expired_traces", unstable)
    with TestClient(
        create_agentos_app(dsn, "fixture-token", tracing=True, trace_retention_days=1)
    ) as client:
        assert remaining() == ({"recent"},) * 2
        auth = {"Authorization": "Bearer fixture-token"}
        assert client.get("/traces/expired-0", headers=auth).status_code == 404
        assert client.get("/traces/recent", headers=auth).status_code == 200
        fail_next.set()
        record("arrived-late", "trace-org:owner", now - timedelta(days=3))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and remaining() != ({"recent"},) * 2:
            time.sleep(0.05)
        assert remaining() == ({"recent"},) * 2
        assert "Trace retention sweep will retry" in caplog.text
        assert "Trace retention fixture detail" not in caplog.text

    record("while-down", "trace-org:owner", now - timedelta(days=3))
    with TestClient(create_agentos_app(dsn, "fixture-token", trace_retention_days=1)):
        assert remaining() == ({"recent"},) * 2

    def exported_span(trace, age):
        ended = int((now - age).timestamp() * 1_000_000_000)
        return ReadableSpan(
            name="Synthetic span",
            context=SpanContext(trace, trace, False, TraceFlags(1)),
            start_time=ended - 1_000_000,
            end_time=ended,
        )

    exporter = observability._RedactedExporter(db, object(), timedelta(days=1))
    assert (
        exporter.export([exported_span(1, timedelta(days=2)), exported_span(2, timedelta())])
        == SpanExportResult.SUCCESS
    )
    assert remaining() == ({"recent", f"{2:032x}"},) * 2

    record("locked", "trace-org:owner", now - timedelta(days=3))
    cutoff = now - timedelta(days=1)
    with psycopg.connect(dsn) as locker:
        locker.execute("SELECT trace_id FROM ai.garp_traces WHERE trace_id='locked' FOR UPDATE")
        assert observability._delete_expired_traces(db, cutoff) == 0
    assert "locked" in remaining()[1]
    with psycopg.connect(dsn) as setup:
        setup.execute(
            "CREATE TABLE ai.retention_fixture (id text REFERENCES ai.garp_traces(trace_id))"
        )
        setup.execute("INSERT INTO ai.retention_fixture VALUES ('locked')")
    with pytest.raises(IntegrityError):
        observability._delete_expired_traces(db, cutoff)
    assert all("locked" in rows for rows in remaining())
    with psycopg.connect(dsn) as setup:
        setup.execute("DELETE FROM ai.retention_fixture")
    assert observability._delete_expired_traces(db, cutoff) == 1
    assert remaining() == ({"recent", f"{2:032x}"},) * 2


@pytest.mark.parametrize("value", [0, -1, True, "30", 10**20])
def test_invalid_trace_retention_window_is_rejected(value):
    with pytest.raises(ValueError, match="Trace retention"):
        create_agentos_app("postgresql://unused", "fixture-token", trace_retention_days=value)


def test_invalid_trace_retention_env_does_not_echo_its_value(monkeypatch):
    monkeypatch.setenv("GARP_TOKEN", "fixture-token-at-least-24-bytes")
    monkeypatch.setenv("GARP_TRACE_RETENTION_DAYS", "invalid-value-fixture")
    with pytest.raises(
        ValueError, match="^GARP_TRACE_RETENTION_DAYS must be a positive whole number$"
    ):
        agentos.serve(8787)
