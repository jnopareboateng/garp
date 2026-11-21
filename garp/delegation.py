"""Submit independently addressable specialist runs to the AgentOS queue."""

import asyncio
import json
import re
from dataclasses import dataclass
from hashlib import sha256
from time import time
from types import SimpleNamespace
from uuid import UUID, uuid5

from agno.db.schemas.jobs import QueuedJob
from agno.os.job_queue import (
    aenqueue_ticket,
    aprepare_accepted_or_abort,
    factory_queue_scope,
    ticket_status_to_api,
    wake_queue_worker,
)
from agno.os.scopes import has_required_scopes
from agno.run.authorization import authorization_scope
from agno.run.base import RunContext

from garp.session_deletion import DeletedRun

DELEGATION_KEY_PREFIX = "garp.delegate."
_NAMESPACE = UUID("ee129db2-c967-4e18-bfa1-2014fe00e2e3")


@dataclass(frozen=True)
class Specialist:
    id: str
    name: str
    instructions: str
    extension_scopes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.id in {"garp", "garp-team"} or not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", self.id):
            raise ValueError("Specialist IDs must be distinct from garp and URL safe")
        if not self.name.strip() or not self.instructions.strip():
            raise ValueError("Specialists require a name and instructions")
        if (
            not isinstance(self.extension_scopes, tuple)
            or any(not isinstance(scope, str) for scope in self.extension_scopes)
            or any(
                scope not in {"skills:read", "knowledge:read"}
                and not re.fullmatch(r"mcp:[a-z][a-z0-9_-]{0,63}:run", scope)
                for scope in self.extension_scopes
            )
            or len(set(self.extension_scopes)) != len(self.extension_scopes)
        ):
            raise ValueError("Specialist extensions require distinct named read scopes")


DEFAULT_SPECIALISTS = (
    Specialist("reviewer", "Reviewer", "Review the assigned work and return concrete findings."),
    Specialist(
        "researcher", "Researcher", "Investigate the assigned question and report evidence."
    ),
)


def specialist_allowed(scopes, agent_id):
    return has_required_scopes(list(scopes), ["agents:run"], "agents", agent_id)


def specialist_extension_scopes(specialist, parent_scopes, *, hosted):
    """Return only named host-granted capabilities for this specialist."""
    return [
        scope
        for scope in specialist.extension_scopes
        if not hosted
        or (
            has_required_scopes(list(parent_scopes), ["mcp:run"], "mcp", scope.split(":")[1])
            if scope.startswith("mcp:")
            else has_required_scopes(list(parent_scopes), [scope])
        )
    ]


class Delegation:
    def __init__(self, db, worker, specialists, build_specialist):
        self.db = db
        self.worker = worker
        self.specialists = {specialist.id: specialist for specialist in specialists}
        self.build_specialist = build_specialist

    def tools(self, ctx, allowed, *, hosted, model_id):
        choices = ", ".join(f"{self.specialists[key].name} ({key})" for key in sorted(allowed))

        async def delegate_task(agent_id: str, task: str, run_context: RunContext) -> str:
            """Assign a task to a named specialist in its own durable chat and return its IDs."""
            if agent_id not in allowed:
                return "That specialist is unavailable to this chat."
            if not task.strip() or len(task) > 32_000:
                return "Provide a non-empty task of at most 32,000 characters."
            owner = run_context.user_id
            if owner != (ctx.user_id or "local"):
                return "Specialist dispatch is unavailable."
            if hosted and not specialist_allowed(ctx.trusted.scopes, agent_id):
                return "That specialist is unavailable to this chat."
            digest = sha256(task.encode()).hexdigest()
            key = json.dumps([owner, run_context.run_id, agent_id, digest])
            child_id, run_id = (str(uuid5(_NAMESPACE, key + kind)) for kind in ("chat", "run"))
            previous = await asyncio.to_thread(self.db.get_run, run_id)
            if previous is not None:
                return self._result(previous, owner, child_id, agent_id)
            if existing := await self._ticket_result(run_id, run_context, allowed):
                return existing
            if hosted and ctx.trusted.claims.get("exp", 0) <= time():
                return "Sign in again before assigning new specialist work."
            if not (worker := self.worker()):
                return "Specialist dispatch is unavailable."

            settings = {**ctx.input.model_dump(), "model_id": model_id, "web_search": False}
            # Workers, history and external capabilities see the child grant, never the parent's.
            specialist = self.specialists[agent_id]
            scopes = ["sessions:read"] if "sessions:read" in ctx.trusted.scopes else []
            if hosted:
                scopes.append(f"agents:{agent_id}:run")
            extension_scopes = specialist_extension_scopes(
                specialist, ctx.trusted.scopes, hosted=hosted
            )
            scopes.extend(extension_scopes)
            state = SimpleNamespace(claims={"sub": owner}, scopes=scopes)
            child = self.build_specialist(specialist, owner, settings, "sessions:read" in scopes)
            payload = {
                "input": task,
                "kwargs": {
                    "session_state": {
                        "garp_options": {
                            "model": settings["model_id"],
                            "effort": settings["effort"],
                            "web_search": False,
                        }
                    },
                    "metadata": {
                        "garp_parent_session_id": run_context.session_id,
                        "garp_parent_run_id": run_context.run_id,
                    },
                },
                "scope": factory_queue_scope(state, owner, child_id, settings),
            }
            job = QueuedJob(
                id=run_id,
                component_type="agent",
                component_id=agent_id,
                session_id=child_id,
                user_id=owner,
                payload=payload,
                # Extension code may have side effects, so only model-only runs may retry.
                max_attempts=1 if extension_scopes else 2,
                deployment_id=worker.config.deployment_id,
                idempotency_key=DELEGATION_KEY_PREFIX + run_id,
            ).to_dict()
            with authorization_scope(lambda: None):
                try:
                    accepted = await aenqueue_ticket(worker, job, child)
                except DeletedRun:
                    return "Specialist dispatch is unavailable."
            ticket = accepted.get("job")
            if not ticket or ticket["id"] != run_id:
                return "The specialist task could not be queued."
            if accepted["accepted"]:
                await aprepare_accepted_or_abort(
                    worker, child, "agent", run_id, child_id, owner, task
                )
                wake_queue_worker(worker)
            return self._ticket_view(ticket)

        async def read_specialist_result(run_id: str, run_context: RunContext) -> str:
            """Read a specialist result by run ID after assigning work, when it is ready."""
            run = await asyncio.to_thread(self.db.get_run, run_id)
            if (
                run is not None
                and getattr(run, "agent_id", None) in allowed
                and (run.metadata or {}).get("garp_parent_session_id") == run_context.session_id
            ):
                return self._result(run, run_context.user_id, run.session_id, run.agent_id)
            return (
                await self._ticket_result(run_id, run_context, allowed)
                or "Specialist run not found."
            )

        return [delegate_task, read_specialist_result], choices

    async def _ticket_result(self, run_id, run_context, allowed):
        ticket = await asyncio.to_thread(self.db.get_job, run_id)
        if (
            not ticket
            or ticket["component_id"] not in allowed
            or ticket["user_id"] != run_context.user_id
        ):
            return None
        metadata = ticket["payload"].get("kwargs", {}).get("metadata", {})
        if (ticket["idempotency_key"] or "").startswith(DELEGATION_KEY_PREFIX) and (
            metadata.get("garp_parent_session_id") == run_context.session_id
        ):
            return self._ticket_view(ticket)
        return None

    @staticmethod
    def _ticket_view(ticket):
        return json.dumps(
            {
                "agent_id": ticket["component_id"],
                "session_id": ticket["session_id"],
                "run_id": ticket["id"],
                "status": ticket_status_to_api(ticket["status"]),
            }
        )

    @staticmethod
    def _result(run, owner, session_id, agent_id):
        if (getattr(run, "user_id", None), run.session_id, getattr(run, "agent_id", None)) != (
            owner,
            session_id,
            agent_id,
        ):
            return "Specialist run not found."
        status = run.status.value
        return json.dumps(
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "run_id": run.run_id,
                "status": status,
                "result": run.content if status == "COMPLETED" else None,
            }
        )
