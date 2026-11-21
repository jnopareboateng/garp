"""Hold scheduled prompts outside the ordered AgentOS run queue until due."""

import asyncio
import logging
from collections.abc import Collection, Mapping
from math import ceil
from time import time
from types import SimpleNamespace
from typing import Annotated
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from agno.db.base import SessionType
from agno.db.schemas.jobs import QueuedJob
from agno.db.schemas.scheduler import Schedule
from agno.db.utils import is_unique_violation
from agno.exceptions import InputCheckError
from agno.factory import FactoryPermissionError, TrustedContext
from agno.os.job_queue import factory_queue_scope, wake_queue_worker
from agno.os.routers.schedules.schema import ScheduleResponse
from agno.os.scopes import AgentOSScope, has_required_scopes
from agno.run.agent import RunInput, RunOutput
from agno.run.base import RunStatus
from agno.scheduler import ScheduleExecutor
from agno.scheduler.cron import next_run_for, validate_cron_expr, validate_timezone
from agno.session import AgentSession
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from garp.host_auth import SCHEDULE_KEY_PREFIX, is_host_principal, require_host_scope
from garp.models import AgentSettings, default_model, validate_options
from garp.recurring_checks import HostedCheckExecutor, enable_schedule
from garp.store import NotFound, lock_cron_claim

_MANAGED_BY = "garp-prompt"
_ENDPOINT = "/agents/garp/runs"
_RUN_SCOPE = "agents:run"
logger = logging.getLogger(__name__)


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32000)
    ]
    session_id: UUID | None = None
    options: AgentSettings = Field(default_factory=AgentSettings)
    deadline_seconds: int | None = Field(default=None, strict=True, ge=0, le=31536000)


class PromptSchedule(PromptSpec):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$", max_length=255)
    after_seconds: int | None = Field(default=None, strict=True, ge=0, le=31536000)
    cron_expr: str | None = Field(default=None, min_length=1, max_length=128)
    timezone: str = Field(default="UTC", max_length=64)
    max_retries: int = Field(default=1, strict=True, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, strict=True, ge=1, le=3600)

    @model_validator(mode="after")
    def timing(self):
        if (self.after_seconds is None) == (self.cron_expr is None):
            raise ValueError("Supply after_seconds or cron_expr")
        if self.cron_expr and (
            len(self.cron_expr.split()) != 5 or not validate_cron_expr(self.cron_expr)
        ):
            raise ValueError("Use a valid five-field cron expression")
        if not validate_timezone(self.timezone):
            raise ValueError("Use an IANA timezone")
        return self


class ScheduledPrompts:
    def __init__(self, checks, *, hosted, current_scopes, worker):
        self.checks = checks
        self.hosted = hosted
        self.current_scopes = current_scopes
        self.worker = worker

    def lookup(self, schedule_id, owner):
        row = self.checks.db.get_schedule(schedule_id, user_id=owner)
        return row if row and row["managed_by"] == _MANAGED_BY else None

    def effective_scopes(self, owner, bounds):
        if not self.hosted:
            return ()
        if not is_host_principal(owner) or self.current_scopes is None:
            raise FactoryPermissionError("Scheduled prompts require a current host grant")
        try:
            current = self.current_scopes(owner)
        except Exception as exc:
            raise FactoryPermissionError("Scheduled prompt grant is unavailable") from exc
        if not isinstance(current, Collection) or isinstance(current, (str, bytes, Mapping)):
            raise FactoryPermissionError("Scheduled prompt grant is unavailable")
        scopes = sorted(
            set(bounds).intersection(scope for scope in current if isinstance(scope, str))
        )
        if (
            AgentOSScope.ADMIN.value in scopes
            or not has_required_scopes(scopes, ["schedules:write"])
            or not has_required_scopes(scopes, [_RUN_SCOPE], "agents", "garp")
        ):
            raise FactoryPermissionError("Scheduled prompt is no longer authorized")
        return scopes

    async def check_run(self, ctx):
        scheduled = ctx.trusted.claims.get("garp_scheduled_prompt")
        if scheduled is None:
            return ctx.trusted
        if scheduled["deadline"] is not None and time() > scheduled["deadline"]:
            raise FactoryPermissionError("Scheduled prompt expired before execution")
        scopes = await asyncio.to_thread(self.effective_scopes, ctx.user_id, ctx.trusted.scopes)
        if scheduled["existing_session"] and not await asyncio.to_thread(
            self.checks.session_allowed,
            ctx.session_id,
            ctx.user_id,
        ):
            raise FactoryPermissionError("Scheduled chat is unavailable")
        return TrustedContext(claims=ctx.trusted.claims, scopes=frozenset(scopes))

    def start_check(self, ctx):
        async def check_authority():
            try:
                current = await self.check_run(ctx)
            except FactoryPermissionError as exc:
                raise InputCheckError(str(exc)) from None
            if current.scopes != ctx.trusted.scopes:
                raise InputCheckError("Scheduled prompt permissions changed before execution")

        return check_authority

    def _prepare_pending(self, spec, run_id, sid, owner):
        db = self.checks.db
        try:
            if spec.session_id is None:
                created = db.insert_session_if_absent(
                    AgentSession(
                        session_id=sid, agent_id="garp", user_id=owner, created_at=int(time())
                    )
                )
                if created is None:
                    raise RuntimeError("Scheduled chat could not be prepared")
            if not self.checks.session_allowed(sid, owner):
                return  # The worker checks ownership again before running in an existing chat.
            run = RunOutput(
                run_id=run_id,
                session_id=sid,
                agent_id="garp",
                user_id=owner,
                input=RunInput(input_content=spec.message),
                status=RunStatus.pending,
                queue_attempt=0,
            )
            prepared = db.append_run_to_session_if_absent(sid, run.to_dict(), user_id=owner)
        except Exception:  # noqa: BLE001 - A committed ticket remains admitted on storage errors.
            prepared = None
        if prepared is None:
            logger.warning("Pending row unavailable for admitted scheduled run %s", run_id)

    def submit_due(self, schedule, worker):
        owner, stamp = schedule.user_id, schedule.next_run_at
        if (
            schedule.endpoint != _ENDPOINT
            or schedule.method != "POST"
            or stamp is None
            or (not self.hosted and owner != "local")
        ):
            raise ValueError("Invalid scheduled prompt")
        payload = schedule.payload or {}
        spec = PromptSpec.model_validate(payload["prompt"])
        key = f"garp:scheduled-prompt:{schedule.id}:{stamp}"
        run_id = str(uuid5(NAMESPACE_URL, key + ":run"))
        sid = str(spec.session_id or uuid5(NAMESPACE_URL, key + ":session"))
        deadline = None if spec.deadline_seconds is None else stamp + spec.deadline_seconds
        db = self.checks.db
        grant_error = None
        try:
            scopes = self.effective_scopes(owner, payload.get("scopes", ()))
        except FactoryPermissionError as exc:
            grant_error = exc
        with self.checks.store._connect() as conn:
            lock_cron_claim(conn, (db.db_schema, db.schedules_table_name), schedule, owner)
            ticket = db.get_job(run_id, strict=True)
            previous = ticket or db.get_run(run_id, deserialize=False)
            if previous:
                component = previous.get("component_id") or previous.get("agent_id")
                if (previous["user_id"], previous["session_id"], component) != (owner, sid, "garp"):
                    raise ValueError("Scheduled run ID collision")
                if (
                    ticket
                    and ticket["status"] == "queued"
                    and not db.get_run(run_id, deserialize=False)
                ):
                    self._prepare_pending(spec, run_id, sid, owner)
                return run_id, sid
            if deadline is not None and time() > deadline:
                raise HTTPException(410, "Scheduled prompt expired")
            if spec.session_id and not self.checks.session_allowed(sid, owner):
                raise HTTPException(404, "Scheduled chat is unavailable")
            if grant_error is not None:
                raise grant_error
            settings = spec.options.model_dump()
            state = SimpleNamespace(
                scopes=scopes,
                claims={
                    "sub": owner,
                    "garp_durable_approval_tools": True,
                    "garp_scheduled_prompt": {
                        "deadline": deadline,
                        "existing_session": spec.session_id is not None,
                    },
                },
            )
            kwargs = {"metadata": {"garp_prompt_schedule_id": schedule.id}}
            if spec.session_id is None:
                kwargs["session_state"] = {
                    "garp_options": {
                        "model": settings["model_id"],
                        "effort": settings["effort"],
                        "web_search": settings["web_search"],
                    }
                }
            job = QueuedJob(
                id=run_id,
                component_type="agent",
                component_id="garp",
                session_id=sid,
                user_id=owner,
                deployment_id=worker.config.deployment_id,
                idempotency_key=SCHEDULE_KEY_PREFIX + run_id,
                payload={
                    "input": spec.message,
                    "serialize_session": worker.config.queue_per_session,
                    "kwargs": kwargs,
                    "scope": factory_queue_scope(state, owner, sid, settings),
                },
            ).to_dict()
            accepted = db.enqueue_job(job, max_depth=worker.config.max_queue_depth)
            if not accepted.get("job") or accepted["job"]["id"] != run_id:
                raise HTTPException(503, "Scheduled prompt queue is unavailable")
            self._prepare_pending(spec, run_id, sid, owner)
        return run_id, sid

    def attach(self, app):
        db = self.checks.db

        def access(request):
            if not self.hosted:
                return "local", []
            owner = require_host_scope(request, "schedules:write")
            require_host_scope(request, _RUN_SCOPE, resource_type="agents", resource_id="garp")
            scopes = sorted(request.state.scopes)
            try:
                scopes = self.effective_scopes(owner, scopes)
            except FactoryPermissionError as exc:
                raise HTTPException(403, str(exc)) from None
            return owner, scopes

        @app.post(
            "/garp/prompt-schedules",
            status_code=201,
            operation_id="garp_create_prompt_schedule",
            response_model=ScheduleResponse,
        )
        def create(request: Request, body: PromptSchedule):
            owner, scopes = access(request)
            options = body.options.model_dump()
            if body.session_id:
                chat = db.get_session(
                    str(body.session_id), session_type=SessionType.AGENT, user_id=owner
                )
                if not chat or chat.agent_id != "garp":
                    raise HTTPException(404, "Scheduled chat is unavailable")
                if "options" not in body.model_fields_set:
                    saved = ((chat.session_data or {}).get("session_state") or {}).get(
                        "garp_options"
                    ) or {}
                    options = {
                        "model_id": saved.get("model"),
                        "effort": saved.get("effort"),
                        "web_search": saved.get("web_search", False),
                    }
            options["model_id"] = options["model_id"] or default_model()
            try:
                validate_options(options["model_id"], options["effort"], options["web_search"])
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
            prompt = body.model_dump(mode="json", include=set(PromptSpec.model_fields))
            prompt["options"] = options
            due = (
                next_run_for(body.cron_expr, body.timezone)
                if body.cron_expr
                else (ceil(time()) + body.after_seconds if body.after_seconds else int(time()))
            )
            row = Schedule(
                id=str(uuid4()),
                name=body.name,
                endpoint=_ENDPOINT,
                method="POST",
                user_id=owner,
                managed_by=_MANAGED_BY,
                cron_expr=body.cron_expr or "",
                timezone=body.timezone,
                max_retries=body.max_retries,
                retry_delay_seconds=body.retry_delay_seconds,
                next_run_at=due,
                payload={"prompt": prompt, "scopes": scopes},
            )
            try:
                return db.create_schedule(row.to_dict())
            except Exception as exc:
                if is_unique_violation(exc):
                    raise HTTPException(409, "Schedule name is already in use") from None
                raise

        @app.post(
            "/garp/prompt-schedules/{schedule_id}/enable",
            operation_id="garp_enable_prompt_schedule",
            response_model=ScheduleResponse,
        )
        def enable(request: Request, schedule_id: UUID):
            owner, _ = access(request)
            row = self.lookup(str(schedule_id), owner)
            if row is None:
                raise HTTPException(404, "Prompt schedule not found")
            loaded = Schedule.from_dict(row)
            if not loaded.cron_expr and loaded.next_run_at is None:
                detail = (
                    "This one-shot prompt expired"
                    if loaded.disabled_reason == "expired"
                    else "This one-shot prompt is closed"
                )
                raise HTTPException(409, detail)
            due = (
                next_run_for(loaded.cron_expr, loaded.timezone)
                if loaded.cron_expr
                else loaded.next_run_at
            )
            return enable_schedule(db, loaded, owner, due)


class PromptScheduleExecutor(HostedCheckExecutor):
    def __init__(self, checks, prompts, *, local_url=None, local_token=None):
        super().__init__(checks)
        self.prompts = prompts
        self.local = ScheduleExecutor(local_url, local_token) if local_url else None

    async def close(self):
        if self.local:
            await self.local.close()

    async def execute(self, schedule, db, release_schedule=True):
        if isinstance(schedule, dict):
            schedule = Schedule.from_dict(schedule)
        if schedule.managed_by != _MANAGED_BY and self.local:
            return await self.local.execute(schedule, db, release_schedule=release_schedule)
        if schedule.managed_by == _MANAGED_BY and not release_schedule:
            raise HTTPException(409, "Prompt schedules run when due")
        return await super().execute(schedule, db, release_schedule=release_schedule)

    async def _call_endpoint(self, schedule):
        if schedule.managed_by != _MANAGED_BY:
            return await super()._call_endpoint(schedule)
        try:
            worker = self.prompts.worker()
            if worker is None:
                raise HTTPException(503, "Scheduled prompt queue is unavailable")
            run_id, session_id = await asyncio.to_thread(self.prompts.submit_due, schedule, worker)
            wake_queue_worker(worker)
            return {
                "status": "success",
                "status_code": 202,
                "run_id": run_id,
                "session_id": session_id,
            }
        except FactoryPermissionError as exc:
            return {"status": "failed", "status_code": 403, "error": str(exc)}
        except HTTPException as exc:
            return {"status": "failed", "status_code": exc.status_code, "error": exc.detail}
        except NotFound:
            return {"status": "cancelled", "error": "Schedule claim is no longer active"}
        except Exception:  # noqa: BLE001 - Attempt history must not store provider or database exceptions.
            return {
                "status": "failed",
                "status_code": 503,
                "error": "Scheduled prompt dispatch unavailable",
            }
