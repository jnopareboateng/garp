"""Keep AgentOS tracing useful without persisting model or tool payloads."""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from threading import Lock

from agno.tracing import DatabaseSpanExporter
from agno.tracing.processor import AgnoSpanFilterProcessor
from fastapi import HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry.context import Context, attach, detach
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON, Sampler
from opentelemetry.trace import Status
from psycopg import sql
from sqlalchemy import DateTime, cast, create_engine, select, text
from sqlalchemy.pool import NullPool

TRACE_OPERATIONS = {"get_traces", "get_trace"}
logger = logging.getLogger(__name__)
_TRACE_DELETE_BATCH = 500
_TRACE_SWEEP_INTERVAL = 3600

_app_scope = ContextVar("garp_observability_app", default=None)
_log_lock = Lock()
_instrument_lock = Lock()
_log_users = 0
_log_factory = None
_previous_factory = None
_log_names = {
    "agno",
    "agno-team",
    "agno-workflow",
    "openinference",
    "opentelemetry",
    "openai",
    "httpx",
    "httpcore",
    "mcp",
    "fastmcp",
    "sqlalchemy",
}
_span_fields = {
    "openinference.span.kind",
    "agno.agent.id",
    "agno.team.id",
    "agno.run.id",
    "session.id",
    "user.id",
    "llm.model_name",
    "llm.provider",
    "tool.name",
    "llm.token_count.prompt",
    "llm.token_count.completion",
    "llm.token_count.total",
    "llm.token_count.prompt_details.cache_read",
    "llm.token_count.completion_details.reasoning",
}


@contextmanager
def _protect_logs():
    global _log_users, _log_factory, _previous_factory
    with _log_lock:
        if not _log_users:
            previous = logging.getLogRecordFactory()

            def factory(name, level, filename, line, message, args, exc, func=None, sinfo=None):
                if _app_scope.get() is not None and name.partition(".")[0] in _log_names:
                    message, args, exc, sinfo = "Runtime detail redacted", (), None, None
                return previous(name, level, filename, line, message, args, exc, func, sinfo)

            _previous_factory, _log_factory = previous, factory
            logging.setLogRecordFactory(factory)
        _log_users += 1
    try:
        yield
    finally:
        with _log_lock:
            _log_users -= 1
            if not _log_users and logging.getLogRecordFactory() is _log_factory:
                logging.setLogRecordFactory(_previous_factory)


@contextmanager
def _scope(scope):
    token = _app_scope.set(scope)
    # Host trace headers and OpenInference context must not supply run ownership.
    context = attach(Context())
    try:
        yield
    finally:
        detach(context)
        _app_scope.reset(token)


class _AppSampler(Sampler):
    def __init__(self, scope):
        self.scope = scope

    def should_sample(self, *args, **kwargs):
        sampler = ALWAYS_ON if _app_scope.get() is self.scope else ALWAYS_OFF
        return sampler.should_sample(*args, **kwargs)

    def get_description(self):
        return "Garp AgentOS application scope"


class _RedactedExporter(DatabaseSpanExporter):
    def __init__(self, db, scope, retention):
        super().__init__(db)
        self.scope = scope
        self.retention = retention

    def export(self, spans):
        safe_spans = []
        cutoff = _trace_cutoff(self.retention).timestamp() if self.retention is not None else None
        for span in spans:
            # Late delivery must not recreate an expired trace.
            if cutoff is not None and (span.end_time or 0) < cutoff * 1_000_000_000:
                continue
            attributes = {k: v for k, v in (span.attributes or {}).items() if k in _span_fields}
            kind = attributes.get("openinference.span.kind", "CHAIN")
            name = (
                attributes.get("tool.name")
                or attributes.get("llm.model_name")
                or attributes.get("agno.team.id")
                or attributes.get("agno.agent.id")
                or kind
            )
            safe_spans.append(
                ReadableSpan(
                    name=name,
                    context=span.context,
                    parent=span.parent,
                    resource=Resource.get_empty(),
                    attributes=attributes,
                    kind=span.kind,
                    status=Status(span.status.status_code),
                    start_time=span.start_time,
                    end_time=span.end_time,
                )
            )
        token = _app_scope.set(self.scope)
        try:
            return _export_fenced(self.db, safe_spans, super().export)
        finally:
            _app_scope.reset(token)


def _trace_tables(db):
    markers = sql.Identifier(db.db_schema, "garp_deleted_sessions").as_string()
    tombstones = sql.Identifier(db.db_schema, "garp_deleted_traces").as_string()
    return markers, tombstones


def _trace_lock(conn, db):
    conn.execute(
        text("SELECT pg_advisory_xact_lock(hashtext('garp.trace_deletion'), hashtext(:schema))"),
        {"schema": db.db_schema},
    )


def _delete_trace_ids(conn, traces, spans, tombstones, ids):
    if not ids:
        return
    conn.execute(
        text(
            f"INSERT INTO {tombstones} SELECT unnest(CAST(:ids AS text[])) ON CONFLICT DO NOTHING"
        ),
        {"ids": list(ids)},
    )
    if spans is not None:
        conn.execute(spans.delete().where(spans.c.trace_id.in_(ids)))
    if traces is not None:
        conn.execute(traces.delete().where(traces.c.trace_id.in_(ids)))


@contextmanager
def _trace_connection(db):
    # Waiters must not occupy the pool needed by Agno's separate trace and span writes.
    engine = create_engine(db.db_engine.url, poolclass=NullPool)
    try:
        with engine.begin() as conn:
            yield conn
    finally:
        engine.dispose()


def _delete_chat_traces(db, session_ids=None):
    traces, spans = db._get_table("traces"), db._get_table("spans")
    if traces is None and spans is None:
        return
    markers, tombstones = _trace_tables(db)
    with _trace_connection(db) as conn:
        _trace_lock(conn, db)
        sources = select(text("session_id")).select_from(text(markers))
        if session_ids is not None:
            sources = sources.where(text("session_id = ANY(CAST(:session_ids AS text[]))"))
        params = {"session_ids": session_ids} if session_ids is not None else {}
        ids = set()
        if traces is not None:
            ids.update(
                conn.scalars(
                    select(traces.c.trace_id).where(traces.c.session_id.in_(sources)), params
                )
            )
        if spans is not None:
            ids.update(
                conn.scalars(
                    select(spans.c.trace_id).where(
                        spans.c.attributes["session.id"].astext.in_(sources)
                    ),
                    params,
                )
            )
        _delete_trace_ids(conn, traces, spans, tombstones, ids)


def _export_fenced(db, spans, export):
    if not spans:
        return export(spans)
    markers, tombstones = _trace_tables(db)
    traces, stored_spans = db._get_table("traces"), db._get_table("spans")
    with _trace_connection(db) as conn:
        _trace_lock(conn, db)
        sessions = {
            span.attributes.get("session.id") for span in spans if span.attributes.get("session.id")
        }
        deleted = set(
            conn.scalars(
                text(
                    f"SELECT session_id FROM {markers} WHERE session_id = ANY(CAST(:ids AS text[]))"
                ),
                {"ids": list(sessions)},
            )
        )
        ids = {
            f"{span.context.trace_id:032x}"
            for span in spans
            if span.attributes.get("session.id") in deleted
        }
        _delete_trace_ids(conn, traces, stored_spans, tombstones, ids)
        ids.update(
            conn.scalars(
                text(
                    f"SELECT trace_id FROM {tombstones} WHERE trace_id = ANY(CAST(:ids AS text[]))"
                ),
                {"ids": [f"{span.context.trace_id:032x}" for span in spans]},
            )
        )
        return export([span for span in spans if f"{span.context.trace_id:032x}" not in ids])


@asynccontextmanager
async def _trace(db, scope, retention):
    instrumentor = AgnoInstrumentor()
    provider = None
    try:
        with _instrument_lock:
            if instrumentor.is_instrumented_by_opentelemetry:
                raise RuntimeError(
                    "Garp tracing requires control of Agno instrumentation in this process"
                )
            provider = TracerProvider(sampler=_AppSampler(scope), shutdown_on_exit=False)
            provider.add_span_processor(
                AgnoSpanFilterProcessor(BatchSpanProcessor(_RedactedExporter(db, scope, retention)))
            )
            try:
                instrumentor.instrument(
                    tracer_provider=provider,
                    config=TraceConfig(
                        hide_inputs=True,
                        hide_outputs=True,
                        hide_llm_invocation_parameters=True,
                        hide_llm_tools=True,
                        hide_embeddings_vectors=True,
                    ),
                )
            except BaseException:
                # OpenTelemetry sets its active flag only after the native wrappers are installed.
                with suppress(Exception):
                    instrumentor._uninstrument()
                raise
            if not instrumentor.is_instrumented_by_opentelemetry:
                raise RuntimeError("Garp could not start Agno instrumentation")
        try:
            yield
        finally:
            with _instrument_lock:
                instrumentor.uninstrument()
    finally:
        if provider is not None:
            await asyncio.to_thread(provider.shutdown)


def _trace_cutoff(retention):
    now = datetime.now(UTC)
    return now - min(retention, now - datetime.min.replace(tzinfo=UTC))


def _delete_expired_traces(db, cutoff):
    traces = db._get_table("traces")
    if traces is None:
        return 0
    spans = db._get_table("spans")
    with _trace_connection(db) as connection:
        _trace_lock(connection, db)
        ids = connection.scalars(
            select(traces.c.trace_id)
            .where(cast(traces.c.end_time, DateTime(timezone=True)) < cutoff)
            .order_by(traces.c.trace_id)
            .limit(_TRACE_DELETE_BATCH)
            .with_for_update(skip_locked=True)
        ).all()
        if spans is not None and ids:
            connection.execute(spans.delete().where(spans.c.trace_id.in_(ids)))
        if ids:
            connection.execute(traces.delete().where(traces.c.trace_id.in_(ids)))
        return len(ids)


async def _prune_traces(db, retention, stop):
    cutoff = _trace_cutoff(retention)
    while not stop.is_set():
        if await asyncio.to_thread(_delete_expired_traces, db, cutoff) < _TRACE_DELETE_BATCH:
            return


async def _maintain_retention(db, retention, stop):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), _TRACE_SWEEP_INTERVAL)
        except TimeoutError:
            try:
                await _prune_traces(db, retention, stop)
            except Exception:  # noqa: BLE001 - Retry the maintenance pass without leaking SQL details.
                logger.warning("Trace retention sweep will retry")


def configure_observability(app, db, *, tracing, trace_retention_days=None):
    retention = None
    if trace_retention_days is not None:
        if type(trace_retention_days) is not int or trace_retention_days <= 0:
            raise ValueError("Trace retention must be a positive whole number of days")
        try:
            retention = timedelta(days=trace_retention_days)
        except OverflowError as exc:
            raise ValueError("Trace retention is out of range") from exc
    scope = object()
    agentos_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        ready = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()

        async def run_agentos():
            with _scope(scope), _protect_logs():
                async with AsyncExitStack() as stack:
                    if tracing:
                        await stack.enter_async_context(_trace(db, scope, retention))
                    await stack.enter_async_context(agentos_lifespan(app))
                    if hasattr(db, "prune_deleted_chat_traces"):
                        await asyncio.to_thread(db.prune_deleted_chat_traces)
                    if retention is not None:
                        await _prune_traces(db, retention, stop)
                        sweep = asyncio.create_task(_maintain_retention(db, retention, stop))
                        stack.push_async_callback(lambda: sweep)
                    ready.set_result(None)
                    await stop.wait()

        # Queue and scheduler tasks inherit the scope without passing it to the embedding host.
        task = asyncio.create_task(run_agentos())
        try:
            await asyncio.wait((ready, task), return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                task.result()
            yield
        finally:
            stop.set()
            await task

    def protect_requests(app):
        async def handle(asgi_scope, receive, send):
            if asgi_scope["type"] != "http":
                await app(asgi_scope, receive, send)
                return
            started = False

            async def send_response(message):
                nonlocal started
                if message["type"] == "http.response.start":
                    started = True
                await send(message)

            with _scope(scope):
                try:
                    await app(asgi_scope, receive, send_response)
                except Exception:  # noqa: BLE001 - Redact before ASGI error logging.
                    logger.error("AgentOS request failed")
                    if started:
                        raise RuntimeError("AgentOS stream failed") from None
                    response = JSONResponse({"detail": "AgentOS request failed"}, status_code=500)
                    await response(asgi_scope, receive, send)

        return handle

    app.router.lifespan_context = lifespan
    app.add_middleware(protect_requests)

    if tracing:
        previous_handler = app.exception_handlers.get(HTTPException, http_exception_handler)

        @app.exception_handler(HTTPException)
        async def trace_error(request, exc):
            route = request.scope.get("route")
            if getattr(route, "operation_id", None) in TRACE_OPERATIONS:
                if exc.status_code == 404:
                    return JSONResponse({"detail": "Trace or span not found"}, status_code=404)
                if exc.status_code >= 500:
                    return JSONResponse({"detail": "Trace service unavailable"}, status_code=500)
            return await previous_handler(request, exc)
