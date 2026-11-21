"""Dispatch hosted AgentOS cron occurrences to Garp's durable check worker."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import time
from uuid import NAMESPACE_URL, UUID, uuid5

from agno.db.schemas.scheduler import Schedule
from agno.db.utils import is_unique_violation
from agno.os.routers.schedules.schema import ScheduleResponse
from agno.scheduler import ScheduleExecutor, ScheduleManager, SchedulePoller
from agno.scheduler.cron import next_run_for
from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field

from garp.api import ScheduleInput
from garp.host_auth import is_host_principal, require_host_scope
from garp.jobs import JobAccessError
from garp.store import NotFound

_MANAGED_BY = "garp-check"


class RecurringCheck(ScheduleInput):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$", max_length=255)
    cron_expr: str = Field(min_length=1, max_length=128)
    timezone: str = Field(default="UTC", max_length=64)
    delay_seconds: int = Field(default=0, strict=True, ge=0, le=31_536_000)
    max_retries: int = Field(default=1, strict=True, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, strict=True, ge=1, le=3600)


def _check_target(schedule: Schedule):
    if schedule.managed_by != _MANAGED_BY or schedule.method != "POST":
        raise ValueError("Unmanaged schedule")
    session_id = str(
        UUID(schedule.endpoint.removeprefix("/garp/sessions/").removesuffix("/checks"))
    )
    if schedule.endpoint != f"/garp/sessions/{session_id}/checks":
        raise ValueError("Invalid scheduled check target")
    return session_id, ScheduleInput.model_validate(schedule.payload)


class HostedCheckExecutor(ScheduleExecutor):
    def __init__(self, checks):
        super().__init__(base_url="", internal_service_token="")
        self.checks = checks

    async def execute(self, schedule, db, release_schedule=True):
        if isinstance(schedule, dict):
            schedule = Schedule.from_dict(schedule)
        if not release_schedule:
            return await super().execute(schedule, db, release_schedule=False)
        if schedule.locked_by is None or schedule.locked_at is None:
            raise ValueError("A native scheduler claim is required")
        next_due = schedule.next_run_at
        disabled_reason = None
        try:
            result = await super().execute(schedule, db, release_schedule=False)
            if schedule.cron_expr:
                next_due = next_run_for(schedule.cron_expr, schedule.timezone)
            elif result["status"] == "success":
                next_due, disabled_reason = None, "admitted"
            elif result.get("status_code") == 410:
                next_due, disabled_reason = None, "expired"
            else:
                disabled_reason = "dispatch_failed"
            return result
        finally:
            # Native release advances cron even when cancellation interrupts dispatch.
            # Keep this tick claimable after shutdown unless its attempt finished.
            await asyncio.to_thread(
                self._release,
                db,
                schedule,
                next_due,
                disabled_reason,
            )

    def _release(self, db, schedule, next_due, disabled_reason):
        if schedule.cron_expr:
            db.release_schedule(
                schedule.id,
                next_run_at=next_due,
                worker_id=schedule.locked_by,
                locked_at=schedule.locked_at,
            )
            return
        # Retain the occurrence ID and deadline for a retry after re-enabling.
        table = db._get_table("schedules")
        values = {"locked_by": None, "locked_at": None, "next_run_at": next_due}
        if disabled_reason is not None:
            values.update(enabled=False, disabled_reason=disabled_reason)
        with db.Session() as session, session.begin():
            session.execute(
                table.update()
                .where(
                    table.c.id == schedule.id,
                    table.c.next_run_at == schedule.next_run_at,
                    table.c.locked_by == schedule.locked_by,
                    table.c.locked_at == schedule.locked_at,
                )
                .values(**values)
            )

    async def _call_endpoint(self, schedule):
        # Agno owns attempts and releases claims. This callback uses the tick's
        # persisted due time, which stays fixed across retries and stale claims.
        try:
            return await asyncio.to_thread(self._dispatch_check, schedule)
        except HTTPException as exc:
            return {"status": "failed", "status_code": exc.status_code, "error": exc.detail}
        except NotFound:
            return {"status": "cancelled", "error": "Schedule claim is no longer active"}
        except (ValueError, AttributeError):
            return {"status": "failed", "error": "Invalid hosted check schedule"}
        except Exception:  # noqa: BLE001 - Native schedule history must not store raw exceptions.
            return {"status": "failed", "status_code": 503, "error": "Check dispatch unavailable"}

    def _dispatch_check(self, schedule):
        owner = schedule.user_id
        if not is_host_principal(owner) or schedule.next_run_at is None:
            raise ValueError("A claimed hosted schedule is required")
        session_id, body = _check_target(schedule)
        if not self.checks.session_allowed(session_id, owner):
            raise HTTPException(404, "Scheduled session is unavailable")
        stamp = schedule.next_run_at
        occurrence_id = str(uuid5(NAMESPACE_URL, f"garp:cron-check:{schedule.id}:{stamp}"))
        db = self.checks.db
        result = self.checks.admit(
            owner,
            session_id,
            body,
            scheduled_at=datetime.fromtimestamp(stamp, UTC),
            occurrence_id=occurrence_id,
            cron_claim=((db.db_schema, db.schedules_table_name), schedule),
        )
        return {
            "status": "success",
            "status_code": 201,
            "session_id": session_id,
            "output": {"check_id": result["id"]},
        }


@asynccontextmanager
async def run_scheduler(app, db, executor):
    poller = SchedulePoller(db, executor, poll_interval=5)
    app.state.scheduler_executor = executor
    await poller.start()
    try:
        yield
    finally:
        await poller.stop()


def enable_schedule(db, loaded, owner, due):
    if loaded.enabled:
        return loaded.to_dict()
    table = db._get_table("schedules")
    with db.Session() as session, session.begin():
        updated = (
            session.execute(
                table.update()
                .where(
                    table.c.id == loaded.id,
                    table.c.user_id == owner,
                    table.c.enabled.is_(False),
                    table.c.locked_by.is_(None),
                )
                .values(enabled=True, next_run_at=due, disabled_reason=None, updated_at=int(time()))
                .returning(*table.c)
            )
            .mappings()
            .one_or_none()
        )
    if updated is None:
        raise HTTPException(409, "Schedule changed or still has an active claim")
    return dict(updated)


def attach_recurring_checks(app, checks):
    manager = ScheduleManager(checks.db)

    def require_target(owner, session_id, job_ref):
        if not checks.session_allowed(session_id, owner):
            raise HTTPException(404, "Scheduled session is unavailable")
        try:
            checks.jobs.authorize(owner, job_ref)
        except JobAccessError as exc:
            raise HTTPException(403, "job access unavailable or denied") from exc

    @app.post(
        "/garp/sessions/{session_id}/recurring-checks",
        status_code=201,
        operation_id="garp_create_recurring_check",
        response_model=ScheduleResponse,
    )
    def create(request: Request, session_id: UUID, body: RecurringCheck):
        owner = require_host_scope(request, "schedules:write")
        require_host_scope(request, "checks:write")
        sid = str(session_id)
        require_target(owner, sid, body.job_ref)
        if len(body.cron_expr.split()) != 5:
            raise HTTPException(422, "Use a five-field cron expression")
        if body.deadline_seconds is not None and body.deadline_seconds < body.delay_seconds:
            raise HTTPException(422, "deadline precedes the due time")
        try:
            schedule = manager.create(
                name=body.name,
                cron=body.cron_expr,
                endpoint=f"/garp/sessions/{sid}/checks",
                user_id=owner,
                timezone=body.timezone,
                max_retries=body.max_retries,
                retry_delay_seconds=body.retry_delay_seconds,
                payload=body.model_dump(include=set(ScheduleInput.model_fields)),
                provenance={"managed_by": _MANAGED_BY},
            )
        except ValueError as exc:
            if checks.db.get_schedule_by_name(body.name, user_id=owner):
                raise HTTPException(409, "Schedule name is already in use") from exc
            raise HTTPException(422, "Invalid cron expression or timezone") from exc
        except Exception as exc:
            if is_unique_violation(exc):
                raise HTTPException(409, "Schedule name is already in use") from exc
            raise
        return schedule.to_dict()

    @app.post(
        "/garp/recurring-checks/{schedule_id}/enable",
        operation_id="garp_enable_recurring_check",
        response_model=ScheduleResponse,
    )
    def enable(request: Request, schedule_id: UUID):
        owner = require_host_scope(request, "schedules:write")
        require_host_scope(request, "checks:write")
        loaded = manager.get(str(schedule_id), user_id=owner)
        if loaded is None:
            raise HTTPException(404, "Schedule not found")
        try:
            session_id, body = _check_target(loaded)
        except ValueError as exc:
            raise HTTPException(404, "Schedule not found") from exc
        require_target(owner, session_id, body.job_ref)
        return enable_schedule(
            checks.db, loaded, owner, next_run_for(loaded.cron_expr, loaded.timezone)
        )
