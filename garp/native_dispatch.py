"""Publish a scheduled check as a native AgentOS assistant run."""

import json
from uuid import NAMESPACE_URL, uuid5

from agno.db.base import SessionType
from agno.db.postgres import PostgresDb
from agno.models.message import Message
from agno.run.agent import RunOutput
from agno.run.base import RunStatus
from agno.session import AgentSession


def check_run(
    session_id: str, occurrence_id: str, user_id: str, outcome: str, evidence: dict
) -> RunOutput:
    """Build the same native run for live dispatch and historical import."""
    run_id = str(uuid5(NAMESPACE_URL, f"garp:schedule:{occurrence_id}"))
    content = f"Job check: {outcome}\n{json.dumps(evidence, indent=2, sort_keys=True)}"
    return RunOutput(
        run_id=run_id,
        agent_id="garp",
        session_id=session_id,
        user_id=user_id,
        content=content,
        messages=[Message(role="assistant", content=content)],
        metadata={"garp_schedule_id": occurrence_id, "outcome": outcome, "evidence": evidence},
        status=RunStatus.completed,
    )


def publish_check(
    db: PostgresDb, session_id: str, occurrence_id: str, user_id: str, outcome: str, evidence: dict
) -> bool:
    """Insert once; return False on a matching retry, and raise on conflicts or failure."""
    session = db.get_session(session_id, session_type=SessionType.AGENT)
    if not user_id or not isinstance(session, AgentSession) or session.user_id != user_id:
        raise ValueError("native session not found or owner mismatch")
    run = check_run(session_id, occurrence_id, user_id, outcome, evidence)
    inserted = db.append_run_to_session_if_absent(session_id, run.to_dict(), user_id=user_id)
    if inserted is None:
        raise RuntimeError("native run could not be published")
    if inserted:
        return True
    existing = db.get_run(run.run_id, deserialize=False)
    if (
        not existing
        or existing["session_id"] != session_id
        or existing["user_id"] != user_id
        or existing["run_data"].get("metadata") != run.metadata
        or existing["run_data"].get("content") != run.content
    ):
        raise ValueError("native run ID collision")
    return False
