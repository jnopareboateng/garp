"""Self-hosted AgentOS using native sessions and the durable run queue."""

import asyncio
import hmac
import logging
import os
import secrets
from collections.abc import Callable, Collection, Mapping
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from dataclasses import replace
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass

import uvicorn
from agno.agent import Agent
from agno.agent.factory import AgentFactory
from agno.factory import FactoryPermissionError, FactoryValidationError
from agno.factory.utils import QueuedRequest
from agno.job_queue.config import QueueConfig, RedisCoordination
from agno.os import AgentOS
from agno.os.event_streams import get_event_stream, set_event_stream
from agno.os.scopes import has_required_scopes
from agno.os.settings import AgnoAPISettings
from agno.team import Team
from agno.team.factory import TeamFactory
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import URL

from garp.approvals import attach_host_approvals
from garp.delegation import (
    DEFAULT_SPECIALISTS,
    DELEGATION_KEY_PREFIX,
    Delegation,
    specialist_allowed,
    specialist_extension_scopes,
)
from garp.executor import _model, _require_success, generate_title, normalize_title
from garp.extensions import Extensions
from garp.host_auth import SCHEDULE_KEY_PREFIX, HostAuth, host_request_guard, is_host_principal
from garp.host_grants import check_queued_grants
from garp.host_sessions import attach_host_session_updates, install_host_session_revisions
from garp.import_chat import adopt_local_preview_chats, import_local_chats
from garp.models import AgentSettings, default_model, model_choices, validate_options
from garp.native_checks import NativeChecks
from garp.observability import configure_observability
from garp.recurring_checks import HostedCheckExecutor, attach_recurring_checks, run_scheduler
from garp.replay_retention import DeletedRunEventStream
from garp.scheduled_prompts import PromptScheduleExecutor, ScheduledPrompts
from garp.session_deletion import DeletedRun, install_session_deletion
from garp.session_deletion import GarpPostgresDb as PostgresDb
from garp.store import Store
from garp.transcript_search import attach_transcript_search, transcript_tool

logger = logging.getLogger(__name__)


def create_agentos_app(
    dsn: str,
    token: str | None = None,
    *,
    host_auth: HostAuth | None = None,
    jobs=None,
    import_legacy=False,
    scheduler_url: str | None = None,
    specialists=None,
    extensions: Extensions | Mapping[str, Extensions] | None = None,
    tracing: bool = False,
    trace_retention_days: int | None = None,
    scheduled_scopes: Callable[[str], Collection[str]] | None = None,
    current_grants: Callable[[str], Collection[str]] | None = None,
    redis_coordination: RedisCoordination | None = None,
    _demo_model=None,
) -> FastAPI:
    if not dsn.strip() or bool(token) == bool(host_auth) or (token and not token.strip()):
        raise ValueError("Provide a Postgres DSN and exactly one of a local token or HostAuth")
    if host_auth and (import_legacy or scheduler_url or _demo_model is not None):
        raise ValueError(
            "Migration, scheduler callbacks and demo models require local operator mode"
        )
    if scheduled_scopes is not None and not host_auth:
        raise ValueError("Scheduled scopes apply to hosted identities")
    if current_grants is not None and not host_auth:
        raise ValueError("Current grants apply to hosted identities")
    if redis_coordination is not None and not (redis_coordination.key_prefix or "").strip():
        raise ValueError("Redis coordination requires a deployment-specific key prefix")
    authorization = host_auth.authorization() if host_auth else False
    if host_auth and isinstance(extensions, Extensions):
        raise ValueError("Hosted extensions must be assigned to explicit tenant IDs")
    if not host_auth and isinstance(extensions, Mapping):
        raise ValueError("Local mode takes one Extensions configuration")
    empty_extensions = Extensions()
    extension_sets = (
        dict(extensions or {}) if host_auth else {"local": extensions or empty_extensions}
    )
    mcp_attempts = {}
    if any(
        not isinstance(tenant, str) or not is_host_principal(f"{tenant}:host")
        for tenant in extension_sets
    ):
        raise ValueError("Invalid extension tenant ID")
    connections = [tool for configured in extension_sets.values() for tool in configured.mcp]
    if len({id(tool) for tool in connections}) != len(connections):
        raise ValueError("MCP connections cannot be shared across tenant configurations")
    managed_approvals = bool(
        host_auth and any(config.approval_tools for config in extension_sets.values())
    )
    specialists = (
        tuple(specialists)
        if specialists is not None
        else (() if host_auth else DEFAULT_SPECIALISTS)
    )
    if len({specialist.id for specialist in specialists}) != len(specialists):
        raise ValueError("Specialist IDs must be unique")
    if scheduler_url:
        target = urlsplit(scheduler_url)
        if target.scheme != "http" or target.hostname != "127.0.0.1":
            raise ValueError("The scheduler callback must use 127.0.0.1")
        proxies = getproxies()
        if (proxies.get("http") or proxies.get("all")) and not proxy_bypass("127.0.0.1"):
            raise ValueError("Add 127.0.0.1 to NO_PROXY for local scheduler callbacks")

    connection = conninfo_to_dict(dsn.replace("postgresql+psycopg://", "postgresql://", 1))
    db_url = URL.create("postgresql+psycopg", query=connection).render_as_string(
        hide_password=False
    )
    db = PostgresDb(
        id="garp-db",
        db_url=db_url,
        traces_table="garp_traces",
        spans_table="garp_spans",
        schedules_table="garp_host_schedules" if host_auth else None,
        schedule_runs_table="garp_host_schedule_runs" if host_auth else None,
    )
    store = Store(dsn)
    checks = NativeChecks(store, db, jobs)
    recurring_checks = host_auth is not None and jobs is not None
    prompt_schedules = scheduled_scopes is not None or scheduler_url is not None
    prompts = ScheduledPrompts(
        checks,
        hosted=host_auth is not None,
        current_scopes=scheduled_scopes,
        worker=lambda: getattr(app.state, "queue_worker", None),
    )
    scheduler_token = secrets.token_urlsafe() if scheduler_url else None
    if _demo_model is not None:
        from garp.demo import _scripted_transport

        demo_transport = _scripted_transport()
    else:
        demo_transport = None
    client = AsyncClient(
        **({"transport": demo_transport} if demo_transport is not None else {}),
        follow_redirects=False,
        trust_env=False,
        timeout=60,
        event_hooks={"response": [_require_success]},
    )

    @asynccontextmanager
    async def lifespan(_app):
        stop = asyncio.Event()
        worker = None
        try:
            await asyncio.to_thread(install_session_deletion, store, db)
            if host_auth:
                await asyncio.to_thread(install_host_session_revisions, store, db)
            await asyncio.to_thread(db.prune_deleted_chat_schedules)
            if import_legacy or jobs is not None:
                await asyncio.to_thread(store.initialize)
            await asyncio.to_thread(
                store.install_native_check_reference, db.db_schema, db.session_table_name
            )
            if import_legacy:
                await asyncio.to_thread(import_local_chats, dsn, db)
                await asyncio.to_thread(adopt_local_preview_chats, db)
                await asyncio.to_thread(
                    store.promote_local_schedules, db.db_schema, db.session_table_name
                )
            if jobs is not None or import_legacy:
                worker = asyncio.create_task(checks.work(stop))
            async with AsyncExitStack() as stack:
                for tenant, configured in extension_sets.items():
                    mcp_attempts[tenant] = await stack.enter_async_context(configured.connected())
                yield
        finally:
            try:
                if worker is not None:
                    stop.set()
                    await worker
            finally:
                await client.aclose()

    def model_for(settings, admitted_default=None):
        model_id = settings.model_id or admitted_default or default_model()
        try:
            validate_options(model_id, settings.effort, settings.web_search)
        except ValueError as exc:
            raise FactoryValidationError(str(exc)) from exc
        if _demo_model is not None:
            return _demo_model(model_id, client)
        try:
            return _model(client, model_id, effort=settings.effort, web_search=settings.web_search)
        except ValueError:
            if not host_auth:
                raise
            logger.warning("Hosted model configuration is unavailable")
            raise RuntimeError("Model service is unavailable") from None

    def title_hook(model_id, history_access):
        async def name_session(session, run_output):
            if (session.session_data or {}).get("session_name"):
                return
            if await asyncio.to_thread(db.session_deleted, session.session_id):
                return
            if _demo_model is not None:
                session.session_data = {
                    **(session.session_data or {}),
                    "session_name": "Local Harness Demo",
                }
                return
            messages = session.get_messages() if history_access else ()
            first = next(
                (m.content for m in messages if m.role == "user" and m.content),
                run_output.input.input_content if run_output.input else None,
            )
            if not isinstance(first, str):
                return
            try:
                async with asyncio.timeout(4):
                    name = normalize_title(await generate_title(first, model_id=model_id))
            except Exception:  # noqa: BLE001 - A failed title must not abort a completed reply.
                logger.info("Chat title generation failed")
                return
            if name:
                session.session_data = {**(session.session_data or {}), "session_name": name}

        return name_session

    def build_specialist(
        specialist,
        owner,
        settings,
        history_access,
        *,
        name_session=True,
        tools=None,
        instructions=None,
        retriever=None,
    ):
        if host_auth and not is_host_principal(owner):
            raise FactoryPermissionError("A verified host principal is required")
        settings = AgentSettings.model_validate(settings)
        model = model_for(settings)
        return Agent(
            id=specialist.id,
            name=specialist.name,
            instructions=[specialist.instructions, *(instructions or [])],
            user_id=owner or "local",
            model=model,
            db=db,
            store_events=True,
            add_history_to_context=history_access,
            tools=tools,
            knowledge_retriever=retriever,
            post_hooks=[title_hook(model.id, history_access)] if name_session else None,
            telemetry=False,
        )

    delegation = Delegation(
        db, lambda: getattr(app.state, "queue_worker", None), specialists, build_specialist
    )

    def specialist_factory(spec):
        async def build(ctx):
            await check_queued_grants(ctx, current_grants)
            if await asyncio.to_thread(db.session_deleted, ctx.session_id):
                raise FactoryPermissionError("Chat is unavailable")
            if host_auth and not is_host_principal(ctx.user_id):
                raise FactoryPermissionError("A verified host principal is required")
            tenant = ctx.user_id.split(":", 1)[0] if host_auth else "local"
            scopes = specialist_extension_scopes(
                spec,
                ctx.trusted.scopes,
                hosted=bool(host_auth) or isinstance(ctx.request, QueuedRequest),
            )
            restricted = replace(ctx, trusted=replace(ctx.trusted, scopes=frozenset(scopes)))
            configured = extension_sets.get(tenant, empty_extensions)
            tools, instructions, retriever = await configured.for_run(
                restricted, True, approval_tools=False, mcp_attempted=mcp_attempts.get(tenant, {})
            )
            return build_specialist(
                spec,
                ctx.user_id,
                ctx.input.model_dump(),
                not host_auth or "sessions:read" in ctx.trusted.scopes,
                tools=tools,
                instructions=instructions,
                retriever=retriever,
            )

        return AgentFactory(
            id=spec.id, name=spec.name, db=db, factory=build, input_schema=AgentSettings
        )

    def available_members(ctx):
        return frozenset(
            spec.id
            for spec in specialists
            if not host_auth or specialist_allowed(ctx.trusted.scopes, spec.id)
        )

    async def make_team(ctx):
        await check_queued_grants(ctx, current_grants)
        if await asyncio.to_thread(db.session_deleted, ctx.session_id):
            raise FactoryPermissionError("Chat is unavailable")
        allowed = available_members(ctx)
        if host_auth and (not is_host_principal(ctx.user_id) or not allowed):
            raise FactoryPermissionError("At least one authorized team specialist is required")
        history_access = not host_auth or has_required_scopes(ctx.trusted.scopes, ["sessions:read"])
        settings = ctx.input
        model = model_for(settings)
        tools, _ = delegation.tools(ctx, allowed, hosted=host_auth is not None, model_id=model.id)
        if history_access:
            tools.append(transcript_tool(db, ctx.user_id or "local"))
        members = [
            build_specialist(
                spec,
                ctx.user_id,
                {**settings.model_dump(), "model_id": model.id, "web_search": False},
                False,
                name_session=False,
            )
            for spec in specialists
            if spec.id in allowed
        ]
        return Team(
            id="garp-team",
            name="Garp Team",
            members=members,
            db=db,
            user_id=ctx.user_id or "local",
            model=model,
            tools=tools,
            instructions=(
                "Use the native team delegation tool for a result in this turn. Use delegate_task "
                "when work needs its own durable specialist chat. Report outstanding work as pending."
            ),
            add_history_to_context=history_access,
            store_events=True,
            store_member_responses=True,
            post_hooks=[title_hook(model.id, history_access)],
            telemetry=False,
        )

    async def make_agent(ctx):
        await check_queued_grants(ctx, current_grants)
        if await asyncio.to_thread(db.session_deleted, ctx.session_id):
            raise FactoryPermissionError("Chat is unavailable")
        if host_auth and not is_host_principal(ctx.user_id):
            raise FactoryPermissionError("A verified host principal is required")
        scheduled = "garp_scheduled_prompt" in ctx.trusted.claims
        if scheduled:
            ctx = replace(ctx, trusted=await prompts.check_run(ctx))
        settings = ctx.input
        admitted_default = ctx.trusted.claims.get("garp_default_model")
        history_access = not host_auth or has_required_scopes(ctx.trusted.scopes, ["sessions:read"])
        model = model_for(settings, admitted_default)
        allowed = available_members(ctx)
        tenant = ctx.user_id.split(":", 1)[0] if host_auth else "local"
        configured = extension_sets.get(tenant, empty_extensions)
        tools, instructions, retriever = await configured.for_run(
            ctx,
            host_auth is not None,
            approval_tools=not host_auth
            or ctx.trusted.claims.get("garp_durable_approval_tools") is True,
            mcp_attempted=mcp_attempts.get(tenant, {}),
        )
        if history_access:
            tools.append(transcript_tool(db, ctx.user_id or "local"))
        if allowed:
            delegation_tools, choices = delegation.tools(
                ctx, allowed, hosted=host_auth is not None, model_id=model.id
            )
            tools.extend(delegation_tools)
            instructions.append(
                f"Available specialists: {choices}. When delegating, explain what was queued and keep "
                "the run ID. Use read_specialist_result for follow-up. Report the task as pending "
                "until the tool returns a completed result."
            )

        return Agent(
            id="garp",
            user_id=ctx.user_id or "local",
            model=model,
            db=db,
            add_history_to_context=history_access,
            add_memories_to_context=not host_auth or "memories:read" in ctx.trusted.scopes,
            search_past_sessions=history_access,
            tools=tools,
            instructions=instructions or None,
            knowledge_retriever=retriever,
            pre_hooks=[prompts.start_check(ctx)] if scheduled else None,
            post_hooks=[title_hook(model.id, history_access)],
            store_events=True,
            telemetry=False,
        )

    factory = AgentFactory(id="garp", db=db, factory=make_agent, input_schema=AgentSettings)
    base_app = None
    if host_auth:
        base_app = FastAPI(
            dependencies=[
                Depends(
                    host_request_guard(
                        db,
                        managed_approvals=managed_approvals,
                        tracing=tracing,
                        recurring_checks=recurring_checks,
                        prompt_schedules=prompt_schedules,
                    )
                )
            ],
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
    agent_os = AgentOS(
        agents=[factory, *(specialist_factory(spec) for spec in specialists)],
        teams=[TeamFactory(id="garp-team", db=db, factory=make_team, input_schema=AgentSettings)]
        if specialists
        else [],
        db=db,
        settings=AgnoAPISettings(os_security_key=token),
        authorization=authorization,
        user_isolation=host_auth is not None,
        base_app=base_app,
        queue=QueueConfig(durable=True, redis=redis_coordination),
        lifespan=lifespan,
        telemetry=False,
        internal_service_token=scheduler_token,
    )
    # Keep native Redis/in-memory transport semantics, but make deletion a
    # committed write/read fence across replicas.
    native_stream = get_event_stream()
    while isinstance(native_stream, DeletedRunEventStream):
        native_stream = native_stream.inner

    async def deleted_run(run_id):
        return await asyncio.to_thread(db.deleted_run, run_id)

    set_event_stream(DeletedRunEventStream(native_stream, deleted_run))
    app = agent_os.get_app()
    if host_auth:
        attach_host_session_updates(app, db)

    @app.exception_handler(DeletedRun)
    async def reject_deleted_run(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    if token:

        @app.middleware("http")
        async def authenticate(request: Request, call_next):
            supplied = request.headers.get("authorization", "")
            allowed = hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode())
            if not allowed and scheduler_token:
                allowed = hmac.compare_digest(
                    supplied.encode(), f"Bearer {scheduler_token}".encode()
                )
            if not allowed:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            if request.headers.get("idempotency-key", "").startswith(
                (DELEGATION_KEY_PREFIX, SCHEDULE_KEY_PREFIX)
            ):
                return JSONResponse({"detail": "Reserved idempotency key"}, status_code=422)
            if request.url.path.rstrip("/").endswith("/optimize-memories"):
                return JSONResponse({"detail": "Endpoint is unavailable"}, status_code=404)
            prefix = "/schedules/"
            path = request.url.path.rstrip("/")
            if prompt_schedules and path.startswith(prefix):
                schedule_id, slash, action = path.removeprefix(prefix).partition("/")
                edits_prompt = (request.method == "PATCH" and not slash) or (
                    request.method == "POST" and action == "enable"
                )
                if edits_prompt and await asyncio.to_thread(prompts.lookup, schedule_id, "local"):
                    return JSONResponse(
                        {"detail": "Manage this prompt through /garp/prompt-schedules"},
                        status_code=409,
                    )
            return await call_next(request)

    @app.get("/garp/models", operation_id="garp_model_catalog")
    async def models(request: Request):
        return {
            "default": default_model(),
            "models": model_choices(),
            "durable_stream_reconnect": getattr(app.state, "queue_worker", None) is not None,
            "job_checks": jobs is not None,
            "prompt_schedules": prompt_schedules
            and (
                not host_auth
                or (
                    has_required_scopes(request.state.scopes, ["schedules:write"])
                    and has_required_scopes(request.state.scopes, ["agents:run"], "agents", "garp")
                )
            ),
            "hosted_approvals": managed_approvals
            and has_required_scopes(request.state.scopes, ["approvals:read"]),
            "specialists": [
                {"id": spec.id, "name": spec.name}
                for spec in specialists
                if not host_auth or specialist_allowed(request.state.scopes, spec.id)
            ],
        }

    if not host_auth or jobs is not None:
        checks.attach(app, hosted=host_auth is not None)
    if recurring_checks:
        attach_recurring_checks(app, checks)
    if prompt_schedules:
        prompts.attach(app)
    if recurring_checks or prompt_schedules:
        executor = (
            PromptScheduleExecutor(
                checks, prompts, local_url=scheduler_url, local_token=scheduler_token
            )
            if prompt_schedules
            else HostedCheckExecutor(checks)
        )
        agentos_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def scheduled_lifespan(app):
            async with agentos_lifespan(app), run_scheduler(app, db, executor):
                yield

        app.router.lifespan_context = scheduled_lifespan
    if managed_approvals:
        attach_host_approvals(app, db, dsn.replace("postgresql+psycopg://", "postgresql://", 1))
    attach_transcript_search(app, db, hosted=host_auth is not None)
    configure_observability(app, db, tracing=tracing, trace_retention_days=trace_retention_days)
    return app


def serve(port: int):
    from garp.demo import _local_database, configured_jobs

    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    token = os.environ.get("GARP_TOKEN", "")
    if (
        len(token) < 24
        or not token.isascii()
        or not token.isprintable()
        or any(c.isspace() for c in token)
    ):
        raise ValueError("GARP_TOKEN must be at least 24 printable ASCII characters without spaces")
    external_dsn = os.environ.get("GARP_DATABASE_URL")
    retention = os.environ.get("GARP_TRACE_RETENTION_DAYS")
    try:
        retention_days = int(retention) if retention else None
    except ValueError:
        raise ValueError("GARP_TRACE_RETENTION_DAYS must be a positive whole number") from None
    redis_url = os.environ.get("GARP_REDIS_URL")
    redis_prefix = os.environ.get("GARP_REDIS_KEY_PREFIX")
    if bool(redis_url) != bool(redis_prefix):
        raise ValueError("Set both GARP_REDIS_URL and GARP_REDIS_KEY_PREFIX")
    coordination = RedisCoordination(url=redis_url, key_prefix=redis_prefix) if redis_url else None
    database = nullcontext((None, external_dsn)) if external_dsn else _local_database()
    with database as (_, dsn):
        target = f"http://127.0.0.1:{port}"
        app = create_agentos_app(
            dsn,
            token,
            jobs=configured_jobs(),
            import_legacy=not external_dsn,
            scheduler_url=target,
            tracing=os.environ.get("GARP_TRACING", "").strip().lower() == "true",
            trace_retention_days=retention_days,
            redis_coordination=coordination,
        )
        uvicorn.run(app, host="127.0.0.1", port=port)
    return 0
