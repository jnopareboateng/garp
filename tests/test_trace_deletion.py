import secrets
import time
from datetime import UTC, datetime

import jwt
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import URL

from garp import agentos, observability


def test_hosted_trace_deletion_and_late_batches(postgres_dsn):
    key = secrets.token_urlsafe(40)
    auth = agentos.HostAuth("https://fixture.example", "fixture", key, "HS256")

    def identity(owner):
        return {
            "Authorization": "Bearer "
            + jwt.encode(
                {
                    "sub": owner,
                    "iss": auth.issuer,
                    "aud": auth.audience,
                    "exp": int(time.time()) + 120,
                    "scopes": ["sessions:write", "sessions:read", "sessions:delete", "traces:read"],
                },
                key,
                algorithm="HS256",
            )
        }

    db = agentos.PostgresDb(
        db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(postgres_dsn)),
        traces_table="garp_traces",
        spans_table="garp_spans",
    )
    now = int(datetime.now(UTC).timestamp() * 1_000_000_000)
    trace_base = secrets.randbits(120) << 8
    fixture_ids = {f"{trace_base + i:032x}" for i in range(1, 6)}

    def span(trace, n, session=None, *, parent=None):
        attrs = {"session.id": session} if session else {}
        trace_id = trace_base + trace
        return ReadableSpan(
            name="fixture",
            context=SpanContext(trace_id, n, False, TraceFlags(1)),
            parent=SpanContext(trace_id, parent, False, TraceFlags(1)) if parent else None,
            attributes=attrs,
            start_time=now - 10,
            end_time=now,
        )

    with TestClient(agentos.create_agentos_app(postgres_dsn, host_auth=auth)) as client:
        alice, bob = identity("tenant:alice"), identity("other:bob")

        def create(headers):
            result = client.post("/sessions", headers=headers, json={"agent_id": "garp"})
            assert result.status_code == 201, result.text
            return result.json()["session_id"]

        first, second, child, other = create(alice), create(alice), create(alice), create(bob)
        exporter = observability._RedactedExporter(db, object(), None)
        assert (
            exporter.export(
                [span(i, i, session) for i, session in enumerate((first, second, child, other), 1)]
            )
            == SpanExportResult.SUCCESS
        )
        # A child-only batch can arrive before its source session is identifiable.
        assert exporter.export([span(5, 15, parent=5)]) == SpanExportResult.SUCCESS
        partial = db.get_trace(f"{trace_base + 5:032x}")
        assert partial is not None and partial.session_id is None
        assert client.delete(f"/sessions/{first}", headers=alice).status_code == 204
        assert exporter.export([span(5, 5, first)]) == SpanExportResult.SUCCESS
        assert exporter.export([span(5, 16, parent=5)]) == SpanExportResult.SUCCESS
        bulk = {"session_ids": [second, other], "session_types": ["agent", "agent"]}
        assert client.request("DELETE", "/sessions", headers=alice, json=bulk).status_code == 204
        assert exporter.export([span(1, 11), span(2, 12)]) == SpanExportResult.SUCCESS
        with db.Session() as conn:
            traces = db._get_table("traces")
            spans = db._get_table("spans")
            trace_ids = set(
                conn.scalars(
                    traces.select()
                    .with_only_columns(traces.c.trace_id)
                    .where(traces.c.trace_id.in_(fixture_ids))
                )
            )
            span_ids = set(
                conn.scalars(
                    spans.select()
                    .with_only_columns(spans.c.trace_id)
                    .where(spans.c.trace_id.in_(fixture_ids))
                )
            )
        assert trace_ids == span_ids == {f"{trace_base + i:032x}" for i in (3, 4)}
        assert client.get(f"/sessions/{child}", headers=alice).status_code == 200
        assert client.get(f"/sessions/{other}", headers=bob).status_code == 200


def test_export_and_cleanup_do_not_starve_single_connection_pool(postgres_dsn, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    from sqlalchemy import create_engine

    with TestClient(agentos.create_agentos_app(postgres_dsn, "fixture-token")):
        pass
    engine = create_engine(
        URL.create("postgresql+psycopg", query=conninfo_to_dict(postgres_dsn)),
        pool_size=1,
        max_overflow=0,
        pool_timeout=2,
    )
    db = agentos.PostgresDb(db_engine=engine, traces_table="garp_traces", spans_table="garp_spans")
    entered, proceed, waiters_ready = Event(), Event(), Event()
    lock, count = Lock(), [0]
    trace_lock = observability._trace_lock

    def watch_lock(conn, database):
        with lock:
            count[0] += 1
            if count[0] == 3:
                waiters_ready.set()
        return trace_lock(conn, database)

    monkeypatch.setattr(observability, "_trace_lock", watch_lock)
    original = db.upsert_trace

    def blocked(trace):
        entered.set()
        assert proceed.wait(5)
        original(trace)

    monkeypatch.setattr(db, "upsert_trace", blocked)
    now = int(datetime.now(UTC).timestamp() * 1_000_000_000)
    fixture = ReadableSpan(
        name="fixture",
        context=SpanContext(99, 99, False, TraceFlags(1)),
        start_time=now - 10,
        end_time=now,
    )
    exporter = observability._RedactedExporter(db, object(), None)
    try:
        with ThreadPoolExecutor(max_workers=3) as threads:
            write = threads.submit(exporter.export, [fixture])
            assert entered.wait(5)
            cleanup = threads.submit(observability._delete_chat_traces, db)
            retention = threads.submit(observability._delete_expired_traces, db, datetime.now(UTC))
            assert waiters_ready.wait(5)
            proceed.set()
            assert write.result(timeout=5) == SpanExportResult.SUCCESS
            cleanup.result(timeout=5)
            retention.result(timeout=5)
    finally:
        proceed.set()
        engine.dispose()
