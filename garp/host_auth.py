"""Garp's host identity contract on top of AgentOS JWT authorization."""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from agno.db.base import SessionType
from agno.os import Authorization
from agno.os.middleware.user_scope import get_scoped_user_id
from agno.os.scopes import AgentOSScope, has_required_scopes
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

from garp.delegation import DELEGATION_KEY_PREFIX
from garp.models import default_model
from garp.observability import TRACE_OPERATIONS

PRINCIPAL = re.compile(r"[A-Za-z0-9_.@-]{1,128}:[A-Za-z0-9_.@-]{1,128}")
SCHEDULE_KEY_PREFIX = "garp.schedule."

# AgentOS also mounts operator APIs and model-backed operations. Each hosted surface
# needs an ownership and model-policy check before we expose it to a tenant.
HOST_OPERATIONS = {
    "garp_model_catalog",
    "get_agents",
    "get_agent",
    "create_agent_run",
    "get_agent_run",
    "list_agent_runs",
    "cancel_agent_run",
    "resume_agent_run_stream",
    "get_teams",
    "get_team",
    "create_team_run",
    "get_team_run",
    "list_team_runs",
    "cancel_team_run",
    "resume_team_run_stream",
    "get_sessions",
    "get_sessions_filter_schema",
    "search_sessions",
    "garp_search_transcripts",
    "create_session",
    "get_session_by_id",
    "get_session_runs",
    "get_session_run",
    "get_session_run_previews",
    "rename_session",
    "update_session",
    "delete_session",
    "delete_sessions",
    "create_memory",
    "get_memories",
    "get_memory",
    "get_memory_topics",
    "get_user_memory_stats",
    "update_memory",
    "delete_memory",
    "delete_memories",
    "garp_schedule_check",
    "garp_get_check",
    "garp_get_check_by_id",
    "garp_cancel_check",
}
RUN_FIELDS = {
    "message",
    "stream",
    "background",
    "session_id",
    "user_id",
    "factory_input",
    "version",
    "files",
    "files_metadata",
}
APPROVAL_OPERATIONS = {
    "list_approvals",
    "garp_list_reviewable_approvals",
    "garp_get_approval",
    "garp_resolve_approval",
    "garp_continue_approval",
}
RECURRING_CHECK_OPERATIONS = {
    "garp_create_recurring_check": "schedules:write",
    "garp_enable_recurring_check": "schedules:write",
}
PROMPT_SCHEDULE_OPERATIONS = {
    "garp_create_prompt_schedule": "schedules:write",
    "garp_enable_prompt_schedule": "schedules:write",
}
SCHEDULE_MANAGEMENT_OPERATIONS = {
    "list_schedules": "schedules:read",
    "get_schedule": "schedules:read",
    "list_schedule_runs": "schedules:read",
    "get_schedule_run": "schedules:read",
    "disable_schedule": "schedules:write",
    "delete_schedule": "schedules:delete",
}


@dataclass(frozen=True)
class HostAuth:
    issuer: str
    audience: str
    verification_key: str = field(repr=False)
    algorithm: Literal["RS256", "HS256"] = "RS256"

    def authorization(self) -> Authorization:
        if not all(value.strip() for value in (self.issuer, self.audience, self.verification_key)):
            raise ValueError("Hosted auth requires an issuer, audience and verification key")
        if self.algorithm not in {"HS256", "RS256"}:
            raise ValueError("Hosted auth supports RS256 and HS256")
        if self.algorithm == "HS256" and len(self.verification_key.encode()) < 32:
            raise ValueError("Hosted HS256 keys must be at least 32 bytes")
        if self.algorithm == "RS256":
            try:
                key = load_pem_public_key(self.verification_key.encode())
            except ValueError as exc:
                raise ValueError("Hosted RS256 requires a PEM-encoded RSA public key") from exc
            if not isinstance(key, RSAPublicKey) or key.key_size < 2048:
                raise ValueError("Hosted RS256 requires an RSA public key of at least 2048 bits")
        # AgentOS merges these with an explicitly supplied key. A host must have one trust root.
        if os.getenv("JWT_VERIFICATION_KEY") or os.getenv("JWT_JWKS_FILE"):
            raise ValueError("Unset ambient AgentOS JWT configuration when using HostAuth")
        return Authorization(
            verification_keys=[self.verification_key],
            algorithm=self.algorithm,
            issuer=self.issuer,
            audience=self.audience,
            verify_audience=True,
        )


def require_host_scope(
    request: Request, scope: str, *, resource_type=None, resource_id=None
) -> str:
    principal = get_scoped_user_id(request)
    if not principal or not has_required_scopes(
        request.state.scopes, [scope], resource_type, resource_id
    ):
        raise HTTPException(403, "Insufficient permissions")
    return principal


def is_host_principal(principal) -> bool:
    return isinstance(principal, str) and PRINCIPAL.fullmatch(principal) is not None


def host_request_guard(
    db,
    *,
    managed_approvals=False,
    tracing=False,
    recurring_checks=False,
    prompt_schedules=False,
):
    schedule_operations = {}
    if recurring_checks or prompt_schedules:
        schedule_operations.update(SCHEDULE_MANAGEMENT_OPERATIONS)
    if recurring_checks:
        schedule_operations.update(RECURRING_CHECK_OPERATIONS)
    if prompt_schedules:
        schedule_operations.update(PROMPT_SCHEDULE_OPERATIONS)

    def owns_session(session_id, principal, component, kind):
        session = db.get_session(
            session_id, session_type=SessionType(kind), user_id=principal, runs_limit=1
        )
        return session is not None and getattr(session, f"{kind}_id") == component

    async def require_host_request(request: Request):
        route = request.scope.get("route")
        operation = getattr(route, "operation_id", None) or getattr(route, "name", None)
        if operation == "health_check":
            return
        claims = getattr(request.state, "claims", {})
        principal = getattr(request.state, "user_id", None)
        expires = claims.get("exp")
        if (
            not is_host_principal(principal)
            or type(expires) is not int
            or expires <= time.time()
            or claims.get("session_id") is not None
            or len(request.headers.getlist("authorization")) != 1
        ):
            raise HTTPException(401, "A signed host identity with an expiry is required")
        if AgentOSScope.ADMIN.value in request.state.scopes:
            raise HTTPException(403, "Hosted requests require tenant-scoped permissions")

        allowed = (
            operation in HOST_OPERATIONS
            or (managed_approvals and operation in APPROVAL_OPERATIONS)
            or (tracing and operation in TRACE_OPERATIONS)
            or operation in schedule_operations
        )
        if not allowed:
            raise HTTPException(404, "Endpoint is unavailable in hosted mode")
        if operation in TRACE_OPERATIONS:
            require_host_scope(request, "traces:read")
        if operation in schedule_operations:
            require_host_scope(request, schedule_operations[operation])
        # AgentOS snapshots these claims into durable job payloads. Retain identity only.
        request.state.claims = {key: claims[key] for key in ("iss", "aud", "sub", "exp")}
        if operation in {"create_session", "update_session"} and await request.body():
            try:
                body = await request.json()
            except ValueError:
                raise HTTPException(422, "Expected a JSON session object") from None
            if isinstance(body, dict):
                if operation == "create_session" and body.get("session_id") is not None:
                    raise HTTPException(422, "Hosted session IDs are assigned by the service")
                metadata = body.get("metadata")
                if isinstance(metadata, dict) and any(
                    k.startswith("garp_parent_") for k in metadata
                ):
                    raise HTTPException(422, "Reserved session metadata")
        if operation in {"create_agent_run", "create_team_run"}:
            if request.headers.get("idempotency-key", "").startswith(
                (DELEGATION_KEY_PREFIX, SCHEDULE_KEY_PREFIX)
            ):
                raise HTTPException(422, "Reserved idempotency key")
            form = await request.form()
            if form.keys() - RUN_FIELDS:
                raise HTTPException(422, "Unsupported hosted run fields")
            if (
                managed_approvals
                and operation == "create_agent_run"
                and form.get("background") == "true"
                and not form.getlist("files")
                and getattr(request.app.state, "queue_worker", None) is not None
            ):
                request.state.claims["garp_durable_approval_tools"] = True
                request.state.claims["garp_default_model"] = default_model()
            # Agno accepts new caller-supplied IDs after a preflight read. Require an
            # owned row when a caller supplies an ID, so two tenants cannot race for it.
            session_id = form.get("session_id")
            kind = "agent" if operation == "create_agent_run" else "team"
            if session_id is not None and not isinstance(session_id, str):
                raise HTTPException(422, "Expected a session ID")
            if session_id and not await run_in_threadpool(
                owns_session,
                session_id,
                principal,
                request.path_params[f"{kind}_id"],
                kind,
            ):
                raise HTTPException(404, "Session not found")

    return require_host_request
