"""Tenant-scoped decisions and guarded continuation of native approvals."""

from contextlib import asynccontextmanager
from time import time
from typing import Literal
from uuid import UUID

import psycopg
from agno.os.job_queue import acontinue_via_queue
from agno.os.routers.approvals.schema import ApprovalResponse
from agno.os.scopes import has_required_scopes
from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import or_, select
from starlette.concurrency import run_in_threadpool

from garp.host_auth import is_host_principal, require_host_scope


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["approved", "rejected"]


@asynccontextmanager
async def _continuation_lock(dsn, run_id):
    async with await psycopg.AsyncConnection.connect(dsn) as connection:
        cursor = await connection.execute(
            "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"garp:approval-continue:{run_id}",),
        )
        if not (await cursor.fetchone())[0]:
            raise HTTPException(409, "Another continuation is in progress")
        yield


def attach_host_approvals(app, db, dsn):
    def load(approval_id):
        row = db.get_approval(str(approval_id))
        if not row or not is_host_principal(row.get("user_id")) or row.get("agent_id") != "garp":
            raise HTTPException(404, "Approval not found")
        return row

    def can_read(request, row):
        actor, scopes = request.state.user_id, request.state.scopes
        if actor == row["user_id"] and has_required_scopes(scopes, ["approvals:read"]):
            return True
        return actor.split(":", 1)[0] == row["user_id"].split(":", 1)[0] and has_required_scopes(
            scopes, ["approvals:write"]
        )

    @app.get("/garp/approvals", operation_id="garp_list_reviewable_approvals")
    def review_queue(
        request: Request, limit: int = Query(50, ge=1, le=100), page: int = Query(1, ge=1)
    ):
        actor = require_host_scope(request, "approvals:write")
        tenant = actor.split(":", 1)[0]
        # Agno's public query accepts one exact subject; the reviewer queue spans one tenant.
        table = db._get_table("approvals")
        if table is None:
            return {"data": []}
        columns = table.c
        statement = (
            select(
                columns.id,
                columns.run_id,
                columns.session_id,
                columns.tool_name,
                columns.user_id,
                columns.created_at,
                columns.expires_at,
            )
            .where(
                columns.agent_id == "garp",
                columns.status == "pending",
                columns.user_id.startswith(f"{tenant}:", autoescape=True),
                columns.user_id != actor,
                or_(columns.expires_at.is_(None), columns.expires_at > int(time())),
            )
            .order_by(columns.created_at, columns.id)
            .limit(limit)
            .offset((page - 1) * limit)
        )
        with db.Session() as session:
            return {"data": [dict(row) for row in session.execute(statement).mappings()]}

    @app.get(
        "/garp/approvals/{approval_id}",
        operation_id="garp_get_approval",
        response_model=ApprovalResponse,
    )
    def get(request: Request, approval_id: UUID):
        row = load(approval_id)
        if not can_read(request, row):
            raise HTTPException(404, "Approval not found")
        return row

    @app.post(
        "/garp/approvals/{approval_id}/resolve",
        operation_id="garp_resolve_approval",
        response_model=ApprovalResponse,
    )
    def resolve(request: Request, approval_id: UUID, decision: Decision):
        actor = require_host_scope(request, "approvals:write")
        row = load(approval_id)
        if not can_read(request, row):
            raise HTTPException(404, "Approval not found")
        if actor == row["user_id"]:
            raise HTTPException(403, "A different reviewer must resolve this approval")
        now = int(time())
        if row["status"] != "pending" or (row.get("expires_at") and row["expires_at"] <= now):
            raise HTTPException(409, "Approval is no longer pending")
        updated = db.update_approval(
            str(approval_id),
            expected_status="pending",
            status=decision.status,
            resolved_by=actor,
            resolved_at=now,
        )
        if not updated:
            raise HTTPException(409, "Approval could not be resolved")
        return updated

    @app.post(
        "/garp/approvals/{approval_id}/continue",
        status_code=202,
        operation_id="garp_continue_approval",
    )
    async def continue_approved(request: Request, approval_id: UUID):
        actor = require_host_scope(
            request, "agents:run", resource_type="agents", resource_id="garp"
        )
        row = await run_in_threadpool(load, approval_id)
        if actor != row["user_id"]:
            raise HTTPException(404, "Approval not found")
        async with _continuation_lock(dsn, row["run_id"]):
            return await continue_locked(request, actor, approval_id)

    async def continue_locked(request, actor, approval_id):
        row = await run_in_threadpool(load, approval_id)
        if actor != row["user_id"]:
            raise HTTPException(404, "Approval not found")
        worker = getattr(app.state, "queue_worker", None)
        if worker is None:
            raise HTTPException(503, "Approval continuation needs the AgentOS queue")

        run_id, session_id = row["run_id"], row["session_id"]
        ticket = await worker.store.get_job(run_id, strict=True)
        if not ticket or (
            ticket["component_type"],
            ticket["component_id"],
            ticket["user_id"],
            ticket["session_id"],
        ) != ("agent", "garp", actor, session_id):
            raise HTTPException(409, "Approval run is no longer in the durable queue")
        claims = (ticket["payload"].get("scope") or {}).get("context") or {}
        if not set(claims.get("scopes") or ()) <= set(request.state.scopes):
            raise HTTPException(
                403, "The current token no longer grants the paused run's capabilities"
            )
        pending, _ = await run_in_threadpool(
            db.get_approvals, user_id=actor, run_id=run_id, status="pending", limit=1
        )
        if row["status"] not in {"approved", "rejected"} or pending:
            raise HTTPException(409, "Resolve the run's pending approvals first")
        run = await run_in_threadpool(db.get_run, run_id)
        if (
            not run
            or str(run.status).lower() not in {"paused", "runstatus.paused"}
            or not any(
                requirement.tool_execution
                and requirement.tool_execution.approval_id == str(approval_id)
                and not requirement.is_resolved()
                for requirement in run.requirements or []
            )
        ):
            raise HTTPException(409, "Approval does not belong to the current pause")
        outcome = await acontinue_via_queue(
            worker,
            run_id,
            {"continue_from": "end", "kwargs": {}},
            component_type="agent",
            component_id="garp",
        )
        if not outcome or outcome["outcome"] not in {"queued", "attach"}:
            raise HTTPException(409, "Run is not ready to continue")
        return {"run_id": run_id, "session_id": session_id, "status": "PENDING"}
