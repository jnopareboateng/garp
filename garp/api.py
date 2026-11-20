"""Local HTTP API for Garp sessions and one-shot checks."""

import asyncio
import logging
import os
import secrets
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated
from uuid import UUID

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt

from garp.executor import Reply, execute_turn, generate_title, normalize_title, openrouter_config
from garp.jobs import JobAccessError, JobEvidenceError, LocalJobs
from garp.models import default_model, model_choices, validate_options
from garp.store import NotFound, Store
from garp.worker import local_jobs

_TurnExecutor = Callable[[str, str, str, list[dict], str], Awaitable[str | Reply]]
logger = logging.getLogger(__name__)


class TurnInput(BaseModel):
    text: str = Field(min_length=1, max_length=32_000)


class ScheduleInput(BaseModel):
    job_ref: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    delay_seconds: StrictInt = Field(ge=0, le=31_536_000)
    deadline_seconds: StrictInt | None = Field(default=None, ge=0, le=31_536_000)


class ModelInput(BaseModel):
    model: str = Field(min_length=1, max_length=150)


class SettingsInput(BaseModel):
    effort: str | None = Field(default=None, max_length=8)
    web_search: bool = False


def create_app(
    store: Store,
    jobs: LocalJobs,
    identities: dict[str, str],
    executor: _TurnExecutor = execute_turn,
    titler: Callable[[str], Awaitable[str | None]] | None = None,
) -> FastAPI:
    if titler is None and executor is execute_turn:
        titler = generate_title

    @asynccontextmanager
    async def lifespan(_app):
        await asyncio.to_thread(store.initialize)
        yield

    app = FastAPI(lifespan=lifespan)

    def principal(authorization: Annotated[str | None, Header()] = None) -> str:
        if authorization and authorization.startswith("Bearer "):
            credential = authorization[7:].encode()
            for token, owner in identities.items():
                if token and secrets.compare_digest(credential, token.encode()):
                    return owner
        raise HTTPException(401, "invalid credentials", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(NotFound)
    async def not_found(_request, _exc):
        return JSONResponse({"detail": "resource not found"}, status_code=404)

    @app.post("/v1/sessions", status_code=201)
    def create_session(owner: str = Depends(principal)):
        return {"id": store.create_session(owner)}

    @app.get("/v1/sessions")
    def list_sessions(owner: str = Depends(principal)):
        default = default_model()
        return {
            "sessions": [
                {**row, "model": row["model"] or default} for row in store.list_sessions(owner)
            ]
        }

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: UUID, owner: str = Depends(principal)):
        session = store.get_session(owner, str(session_id))
        return {**session, "model": session["model"] or default_model()}

    @app.get("/v1/models")
    def models(_owner: str = Depends(principal)):
        return {"default": default_model(), "models": model_choices()}

    @app.put("/v1/sessions/{session_id}/model")
    def select_model(session_id: UUID, body: ModelInput, owner: str = Depends(principal)):
        try:
            return store.set_session_options(owner, str(session_id), {"model": body.model})
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.patch("/v1/sessions/{session_id}/settings")
    def settings(session_id: UUID, body: SettingsInput, owner: str = Depends(principal)):
        try:
            return store.set_session_options(
                owner, str(session_id), body.model_dump(exclude_unset=True)
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/sessions/{session_id}/events")
    def events(
        session_id: UUID,
        after: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
        owner: str = Depends(principal),
    ):
        result = store.get_events(owner, str(session_id), after)
        return {"events": result, "cursor": result[-1]["seq"] if result else after}

    @app.post("/v1/sessions/{session_id}/turns")
    async def turn(session_id: UUID, body: TurnInput, owner: str = Depends(principal)):
        session = str(session_id)
        try:
            turn = await asyncio.to_thread(store.start_turn, owner, session, body.text)
        except RuntimeError as exc:
            raise HTTPException(409, "session busy") from exc
        model = turn["model"] or default_model()
        live = executor is execute_turn

        async def finish(text, title=None, details=None):
            return await asyncio.to_thread(
                store.finish_turn, owner, session, turn["token"], text, title, details
            )

        if live:
            try:
                validate_options(model, turn["effort"], turn["web_search"])
            except ValueError as exc:
                await finish(None)
                raise HTTPException(422, str(exc)) from exc
            try:
                openrouter_config(model)
            except ValueError as exc:
                await finish(None)
                raise HTTPException(503, str(exc)) from exc
        try:
            async with asyncio.timeout(90):
                arguments = (owner, session, turn["token"], turn["history"], body.text)
                if live:
                    reply = await execute_turn(
                        *arguments,
                        model_id=model,
                        effort=turn["effort"],
                        web_search=turn["web_search"],
                    )
                else:
                    reply = await executor(*arguments)
        except asyncio.CancelledError:
            await finish(None)
            raise
        except TimeoutError as exc:
            await finish(None)
            raise HTTPException(504, "agent turn timed out") from exc
        except Exception as exc:
            await finish(None)
            raise HTTPException(502, "agent turn failed") from exc

        text = reply.text if isinstance(reply, Reply) else reply
        details = (
            {
                "model": model,
                "effort": turn["effort"],
                "web_search": turn["web_search"],
            }
            if live
            else {}
        )
        if isinstance(reply, Reply) and reply.sources:
            details["sources"] = reply.sources
        title = turn["title"]
        try:
            if not title and titler is not None:
                prompts = [entry["content"] for entry in turn["history"] if entry["role"] == "user"]
                if len(prompts) < 2:
                    async with asyncio.timeout(4):
                        opening = prompts[0] if prompts else body.text
                        if titler is generate_title:
                            raw_title = await generate_title(opening, model_id=model)
                        else:
                            raw_title = await titler(opening)
                        title = normalize_title(raw_title) if raw_title else None
        except asyncio.CancelledError:
            await finish(text, details=details)
            raise
        except Exception:  # noqa: BLE001 - A title failure must not lose the completed reply.
            logger.info("Chat title generation failed")
        result = await finish(text, title, details)
        if result is False:
            raise HTTPException(409, "turn lease expired")
        return {
            "events": result,
            "cursor": result[-1]["seq"],
            "title": title,
            "model": model if executor is execute_turn else None,
        }

    @app.post("/v1/sessions/{session_id}/schedules", status_code=201)
    def schedule(session_id: UUID, body: ScheduleInput, owner: str = Depends(principal)):
        session = str(session_id)
        now = store.check_session(owner, session)
        if body.deadline_seconds is not None and body.deadline_seconds < body.delay_seconds:
            raise HTTPException(422, "deadline precedes the due time")
        try:
            baseline = jobs.observe(owner, body.job_ref, now)
        except JobAccessError as exc:
            raise HTTPException(403, "job access unavailable or denied") from exc
        except JobEvidenceError as exc:
            raise HTTPException(502, "job evidence unavailable") from exc
        if baseline.get("freshness") == "missing":
            raise HTTPException(404, "job evidence not found")
        due_at = now + timedelta(seconds=body.delay_seconds)
        deadline_at = (
            now + timedelta(seconds=body.deadline_seconds)
            if body.deadline_seconds is not None
            else None
        )
        schedule_id = store.create_schedule(
            owner, session, body.job_ref, baseline, due_at, deadline_at
        )
        return {"id": schedule_id, "due_at": due_at}

    @app.get("/v1/schedules/{schedule_id}")
    def status(schedule_id: UUID, owner: str = Depends(principal)):
        row = store.get_schedule(owner, str(schedule_id))
        result = {
            key: row[key]
            for key in (
                "id",
                "session_id",
                "due_at",
                "deadline_at",
                "status",
                "attempt",
                "outcome",
                "evidence",
            )
        }
        if row["status"] == "pending" and row["due_at"] <= row["now"]:
            result["status"] = "overdue"
        elif row["status"] == "leased" and row["lease_expires"] <= row["now"]:
            result["status"] = "lease_expired"
        return result

    return app


def main() -> None:
    token = os.environ["GARP_TOKEN"]
    if (
        len(token) < 24
        or not token.isascii()
        or not token.isprintable()
        or any(c.isspace() for c in token)
    ):
        raise SystemExit("GARP_TOKEN must be at least 24 printable ASCII characters without spaces")
    app = create_app(
        Store(os.environ["GARP_DATABASE_URL"]),
        local_jobs(),
        {token: "local:default"},
    )
    uvicorn.run(app, host="127.0.0.1", port=8787)
