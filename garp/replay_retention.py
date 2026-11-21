"""Deletion fence for AgentOS replay buffers.

The database is the deletion authority.  This adapter only adds the
post-commit check; native event stream behaviour remains delegated unchanged.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from agno.os.event_streams.base import BaseEventStream
from agno.run.base import RunStatus


class ReplayFenceUnavailable(RuntimeError):
    """The deletion authority could not be read for a replay operation."""


class DeletedRunEventStream(BaseEventStream):
    """Refuse writes and reads after the durable run deletion marker exists.

    ``is_deleted`` must read committed state and may use any replica-safe
    database connection.  A check is intentionally made for every event: a
    cached decision would reopen the race this fence closes.
    """

    def __init__(
        self,
        inner: BaseEventStream,
        is_deleted: Callable[[str], Awaitable[bool]],
    ) -> None:
        self.inner = inner
        self.is_deleted = is_deleted
        self.shared = inner.shared

    async def _deleted(self, run_id: str, *, fail_closed: bool = True) -> bool:
        try:
            return await self.is_deleted(run_id)
        except Exception:  # noqa: BLE001 - preserve live state across transient fence failures
            # Writes and metadata reads survive a transient fence outage; content reads fail closed.
            return fail_closed

    async def _check(self, run_id: str) -> bool | None:
        try:
            return await self.is_deleted(run_id)
        except Exception:  # noqa: BLE001 - preserve live state across transient fence failures
            return None

    async def _cleanup_deleted(self, run_id: str) -> bool:
        deleted = await self._check(run_id)
        if deleted is None:
            raise ReplayFenceUnavailable("Replay deletion fence is unavailable")
        if deleted:
            await self.inner.cleanup_run(run_id)
            return True
        return False

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        if await self._deleted(run_id, fail_closed=False):
            return None
        return await self.inner.get_run_status(run_id)

    async def register_run(self, run_id: str, status: RunStatus = RunStatus.pending) -> None:
        if not await self._deleted(run_id, fail_closed=False):
            await self.inner.register_run(run_id, status)

    async def set_run_status(
        self, run_id: str, status: RunStatus, generation: int | None = None
    ) -> None:
        if not await self._deleted(run_id, fail_closed=False):
            await self.inner.set_run_status(run_id, status, generation)

    async def complete_run(
        self, run_id: str, status: RunStatus, generation: int | None = None
    ) -> None:
        if not await self._deleted(run_id, fail_closed=False):
            await self.inner.complete_run(run_id, status, generation)

    async def reopen_run(
        self, run_id: str, include_error: bool = False, floor: int | None = None
    ) -> bool:
        if await self._deleted(run_id, fail_closed=False):
            return False
        return await self.inner.reopen_run(run_id, include_error, floor)

    async def cleanup_run(self, run_id: str) -> None:
        await self.inner.cleanup_run(run_id)

    async def reset_run_events(self, run_id: str, generation: int | None = None) -> None:
        if not await self._deleted(run_id, fail_closed=False):
            await self.inner.reset_run_events(run_id, generation)

    async def begin_attempt(self, run_id: str, generation: int) -> None:
        if not await self._deleted(run_id, fail_closed=False):
            await self.inner.begin_attempt(run_id, generation)

    async def add_event(self, run_id: str, event: Any, generation: int | None = None) -> int:
        if await self._deleted(run_id, fail_closed=False):
            return -1
        index = await self.inner.add_event(run_id, event, generation)
        # The database check and native publish cannot share a transaction.
        # If deletion committed in that window, erase the just-published
        # buffer immediately; an already delivered network frame is outside
        # the replay-retention boundary and is reported as such.
        if (await self._check(run_id)) is True:
            await self.inner.cleanup_run(run_id)
            return -1
        return index

    async def replay(self, run_id: str, last_event_index: int | None = None):
        if await self._cleanup_deleted(run_id):
            return []
        return await self.inner.replay(run_id, last_event_index)

    async def get_last_index(self, run_id: str) -> int:
        if await self._deleted(run_id, fail_closed=False):
            return -1
        return await self.inner.get_last_index(run_id)

    async def get_event_count(self, run_id: str) -> int:
        if await self._deleted(run_id, fail_closed=False):
            return 0
        return await self.inner.get_event_count(run_id)

    async def tail(
        self, run_id: str, last_event_index: int | None = None
    ) -> AsyncIterator[tuple[int, str]]:
        if await self._cleanup_deleted(run_id):
            return
        async for item in self.inner.tail(run_id, last_event_index):
            check = await self._check(run_id)
            if check is None:
                raise ReplayFenceUnavailable("Replay deletion fence is unavailable")
            if check:
                await self.inner.cleanup_run(run_id)
                return
            yield item
