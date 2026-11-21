"""Search the user-visible fields in AgentOS's authoritative chat storage."""

import asyncio
import json
import re
from typing import Annotated, Literal

from agno.run.base import RunStatus
from fastapi import Request
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError
from sqlalchemy import case, cast, func, or_, select, type_coerce
from sqlalchemy.dialects.postgresql import JSONB

from garp.host_auth import require_host_scope

_LIMIT = 20
_CHAT_TYPES = ("agent", "team")
_MATCH_CONTEXT = 40


class TranscriptQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=200, pattern=r"^[^\x00]+$"
        ),
    ]
    session_type: Literal["agent", "team"] | None = None


def search_transcripts(db, owner, request):
    sessions = db._get_table("sessions")
    if sessions is None:
        return {"matches": [], "more": False}
    runs = db._get_table("runs")
    kinds = (request.session_type,) if request.session_type else _CHAT_TYPES
    owned = select(
        sessions.c.session_id,
        sessions.c.session_type,
        sessions.c.session_data["session_name"].as_string().label("title"),
        func.coalesce(sessions.c.updated_at, sessions.c.created_at).label("activity"),
    ).where(sessions.c.user_id == owner, sessions.c.session_type.in_(kinds))
    # A v2 session can still have runs in its legacy JSON column. Let Agno merge those
    # sessions with their newer rows using its native run-ID conflict rules.
    legacy = sessions.c.get("runs")
    if legacy is not None:
        legacy = cast(legacy, JSONB)
        legacy_kind = func.jsonb_typeof(legacy)
        modern = owned.where(~func.coalesce(legacy_kind.in_(("array", "string")), False)).subquery()
    else:
        modern = owned.subquery()

    matches = []
    with db.Session() as connection:
        if runs is not None:
            matches.extend(
                connection.execute(_modern_matches(runs, modern, owner, request.query)).mappings()
            )
        legacy_sessions = []
        if legacy is not None:
            candidates = _legacy_candidates(
                owned, sessions, runs, legacy, legacy_kind, owner, request.query
            )
            legacy_sessions = connection.execute(candidates).mappings().all()
    if legacy_sessions:
        phrase = re.compile(re.escape(request.query), re.IGNORECASE)
        legacy_hits = 0
        for row in legacy_sessions:
            session = db.get_session(row["session_id"], user_id=owner)
            if session and (excerpt := _legacy_excerpt(session, owner, phrase)):
                matches.append({**row, "excerpt": excerpt})
                legacy_hits += 1
                if legacy_hits > _LIMIT:
                    break

    ordered = sorted(matches, key=lambda row: (-row["activity"], row["session_id"]))
    return {
        "matches": [{k: v for k, v in row.items() if k != "activity"} for row in ordered[:_LIMIT]],
        "more": len(ordered) > _LIMIT,
    }


def _contains(field, needle):
    return (func.jsonb_typeof(field) == "string") & (
        func.strpos(func.lower(field.astext), needle) > 0
    )


def _visible_run(runs, owner):
    return (
        (func.coalesce(runs.c.user_id, owner) == owner)
        & runs.c.parent_run_id.is_(None)
        & (runs.c.status == RunStatus.completed.value)
    )


def _legacy_candidates(owned, sessions, runs, legacy, legacy_kind, owner, query):
    needle = func.lower(query)
    # Agno also accepts legacy arrays serialized as JSON strings. Give those to
    # Agno to parse without making malformed historical JSON a query failure.
    elements = func.jsonb_array_elements(
        case((legacy_kind == "array", legacy), else_=cast("[]", JSONB))
    ).column_valued(joins_implicitly=True)
    entry = type_coerce(elements, JSONB)
    # A match in the blob can be stale. Check it after Agno merges by run ID.
    old_match = (
        select(1)
        .where(
            or_(
                _contains(entry["input"]["input_content"], needle),
                _contains(entry["content"], needle),
            )
        )
        .correlate(sessions)
        .exists()
    )
    matches = [legacy_kind == "string", old_match]
    if runs is not None:
        data = runs.c.run_data
        matches.append(
            select(1)
            .where(
                runs.c.session_id == sessions.c.session_id,
                _visible_run(runs, owner),
                or_(
                    _contains(data["input"]["input_content"], needle),
                    _contains(data["content"], needle),
                ),
            )
            .correlate(sessions)
            .exists()
        )
    return owned.where(legacy_kind.in_(("array", "string")), or_(*matches)).order_by(
        func.coalesce(sessions.c.updated_at, sessions.c.created_at).desc(), sessions.c.session_id
    )


def _modern_matches(runs, owned, owner, query):
    data = runs.c.run_data
    prompt, answer = data["input"]["input_content"], data["content"]
    needle = func.lower(query)
    in_prompt, in_answer = _contains(prompt, needle), _contains(answer, needle)
    body = case((in_prompt, prompt.astext), else_=answer.astext)
    start = func.greatest(1, func.strpos(func.lower(body), needle) - _MATCH_CONTEXT)
    rank = func.row_number().over(
        partition_by=owned.c.session_id,
        order_by=(runs.c.run_index.desc().nulls_last(), runs.c.created_at.desc(), runs.c.run_id),
    )
    hits = (
        select(
            *owned.c,
            func.substr(body, start, len(query) + 2 * _MATCH_CONTEXT).label("excerpt"),
            rank.label("rank"),
        )
        .select_from(owned.join(runs, owned.c.session_id == runs.c.session_id))
        .where(
            _visible_run(runs, owner),
            in_prompt | in_answer,
        )
        .subquery()
    )
    return (
        select(*(hits.c[col.name] for col in owned.c), hits.c.excerpt)
        .where(hits.c.rank == 1)
        .order_by(hits.c.activity.desc(), hits.c.session_id)
        .limit(_LIMIT + 1)
    )


def _legacy_excerpt(session, owner, phrase):
    for run in reversed(session.runs or []):
        if (
            run.user_id not in (None, owner)
            or run.parent_run_id
            or run.status != RunStatus.completed
        ):
            continue
        for body in (getattr(run.input, "input_content", None), run.content):
            if isinstance(body, str) and (match := phrase.search(body)):
                return body[max(0, match.start() - _MATCH_CONTEXT) : match.end() + _MATCH_CONTEXT]
    return None


def attach_transcript_search(app, db, *, hosted):
    @app.post("/garp/transcripts/search", operation_id="garp_search_transcripts")
    def search(request: Request, body: TranscriptQuery):
        owner = require_host_scope(request, "sessions:read") if hosted else "local"
        return search_transcripts(db, owner, body)


def transcript_tool(db, owner):
    async def search_chat_transcripts(query: str) -> str:
        """Find a literal word or phrase in your past chats. Returns excerpts and session IDs."""
        try:
            request = TranscriptQuery(query=query)
        except ValidationError:
            return "Enter a search term between 1 and 200 characters."
        result = await asyncio.to_thread(search_transcripts, db, owner, request)
        return json.dumps(result, ensure_ascii=False)

    return search_chat_transcripts
