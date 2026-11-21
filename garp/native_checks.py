"""One-shot job checks and delivery into the authoritative AgentOS chat."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from agno.db.base import SessionType
from fastapi import HTTPException, Request
from psycopg import OperationalError
from psycopg.errors import ForeignKeyViolation
from sqlalchemy.exc import SQLAlchemyError

from garp.api import ScheduleInput
from garp.host_auth import require_host_scope
from garp.jobs import JobAccessError, JobEvidenceError
from garp.native_dispatch import publish_check
from garp.store import NATIVE_CHECK_CHAT_FK, NotFound
from garp.worker import run_once

logger = logging.getLogger(__name__)


class NativeChecks:
    def __init__(self, store, db, jobs):
        self.store, self.db, self.jobs = store, db, jobs

    def _owner(self, session_id):
        session = self.db.get_session(session_id, session_type=SessionType.AGENT, runs_limit=1)
        return session.user_id if session and session.agent_id == "garp" else None

    def session_allowed(self, session_id, owner):
        return self._owner(session_id) == owner

    def admit(
        self, owner, session_id, body, *, scheduled_at=None, occurrence_id=None, cron_claim=None
    ):
        if self.jobs is None:
            raise HTTPException(503, "job checks are not configured")
        delay, deadline = body.delay_seconds, body.deadline_seconds
        if deadline is not None and deadline < delay:
            raise HTTPException(422, "deadline precedes the due time")
        now = datetime.now(UTC)
        start = scheduled_at or now
        due_at = start + timedelta(seconds=delay)
        deadline_at = start + timedelta(seconds=deadline) if deadline is not None else None
        if occurrence_id is not None:
            try:
                previous = self.store.get_schedule(owner, occurrence_id)
            except NotFound:
                pass
            else:
                if (
                    previous["native_owner"],
                    previous["session_id"],
                    previous["job_ref"],
                    previous["due_at"],
                    previous["deadline_at"],
                ) != (owner, session_id, body.job_ref, due_at, deadline_at):
                    raise ValueError("Schedule occurrence ID collision")
                return {"id": occurrence_id, "due_at": due_at}
        try:
            self.jobs.authorize(owner, body.job_ref)
            if deadline_at and deadline_at < now:
                baseline = {}
            else:
                baseline = self.jobs.observe(owner, body.job_ref, now)
        except JobAccessError as exc:
            raise HTTPException(403, "job access unavailable or denied") from exc
        except JobEvidenceError as exc:
            raise HTTPException(502, "job evidence unavailable") from exc
        if baseline.get("freshness") == "missing":
            raise HTTPException(404, "job evidence not found")
        try:
            check = self.store.create_schedule(
                owner,
                session_id,
                body.job_ref,
                baseline,
                due_at,
                deadline_at,
                native=True,
                occurrence_id=occurrence_id,
                cron_claim=cron_claim,
            )
        except ForeignKeyViolation as exc:
            if exc.diag.constraint_name != NATIVE_CHECK_CHAT_FK:
                raise
            raise HTTPException(404, "no Garp session with an owner") from None
        return {"id": check, "due_at": due_at}

    def attach(self, app, *, hosted=False):
        def authorized_owner(request, sid, scope):
            principal = require_host_scope(request, scope) if hosted else self._owner(sid)
            if not principal or (hosted and not self.session_allowed(sid, principal)):
                raise HTTPException(404, "no Garp session with an owner")
            return principal

        @app.post(
            "/garp/sessions/{session_id}/checks",
            status_code=201,
            operation_id="garp_schedule_check",
        )
        def schedule(request: Request, session_id: UUID, body: ScheduleInput):
            sid = str(session_id)
            owner = authorized_owner(request, sid, "checks:write")
            return self.admit(owner, sid, body)

        def status_for(request, sid, key, scope):
            owner = authorized_owner(request, sid, scope)
            try:
                row = self.store.get_schedule(owner, key)
            except NotFound as exc:
                raise HTTPException(404, "check not found") from exc
            if row["session_id"] != sid or row["native_owner"] is None:
                raise HTTPException(404, "check not found")
            return row

        @app.get("/garp/sessions/{session_id}/checks/{check_id}", operation_id="garp_get_check")
        def status(request: Request, session_id: UUID, check_id: UUID):
            return _public_status(
                status_for(request, str(session_id), str(check_id), "checks:read")
            )

        @app.get("/garp/checks/{check_id}", operation_id="garp_get_check_by_id")
        def status_by_id(request: Request, check_id: UUID):
            owner = require_host_scope(request, "checks:read") if hosted else "local"
            try:
                row = self.store.get_schedule(owner, str(check_id))
            except NotFound as exc:
                raise HTTPException(404, "check not found") from exc
            if row["native_owner"] is None or not self.session_allowed(row["session_id"], owner):
                raise HTTPException(404, "check not found")
            return _public_status(row)

        @app.delete(
            "/garp/sessions/{session_id}/checks/{check_id}", operation_id="garp_cancel_check"
        )
        def cancel(request: Request, session_id: UUID, check_id: UUID):
            row = status_for(request, str(session_id), str(check_id), "checks:write")
            if not self.store.cancel_native_schedule(row["owner"], str(check_id)):
                raise HTTPException(409, "check is no longer pending")
            return {"status": "cancelled"}

    def work_once(self):
        result = self.store.pending_native_delivery()
        if result:
            try:
                publish_check(
                    self.db,
                    result["session_id"],
                    result["id"],
                    result["owner"],
                    result["outcome"],
                    result["evidence"],
                )
            except ValueError:
                self.store.record_native_delivery(
                    result["id"], error="Native session or run no longer matches the check"
                )
            except (OperationalError, SQLAlchemyError, RuntimeError):
                self.store.defer_native_delivery(result["id"])
                logger.warning("Scheduled check delivery will retry")
            else:
                self.store.record_native_delivery(result["id"])
            return True
        if self.jobs is None:
            return False
        return run_once(self.store, self.jobs, native=True, session_allowed=self.session_allowed)

    async def work(self, stop):
        while not stop.is_set():
            try:
                if await asyncio.to_thread(self.work_once):
                    continue
            except Exception as exc:  # noqa: BLE001 - A failed iteration must leave the worker alive.
                logger.warning("Scheduled check worker will retry after %s", type(exc).__name__)
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except TimeoutError:
                pass


def _public_status(row):
    status = {
        field: row[field]
        for field in (
            "id",
            "session_id",
            "due_at",
            "deadline_at",
            "status",
            "attempt",
            "outcome",
            "evidence",
            "delivered_at",
            "delivery_error",
        )
    }
    if row["status"] == "pending" and row["due_at"] <= row["now"]:
        status["status"] = "overdue"
    elif row["status"] == "leased" and row["lease_expires"] <= row["now"]:
        status["status"] = "lease_expired"
    return status
