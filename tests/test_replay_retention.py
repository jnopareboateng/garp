import asyncio
from dataclasses import dataclass

import pytest
from agno.os.event_streams.in_memory import InMemoryEventStream
from agno.run.base import RunStatus

from garp.replay_retention import DeletedRunEventStream, ReplayFenceUnavailable


@dataclass
class Event:
    event: str = "Message"

    def to_dict(self):
        return {"event": self.event}


def test_deleted_run_fence_blocks_late_writer_and_replay():
    async def check():
        inner = InMemoryEventStream()
        deleted = set()
        stream = DeletedRunEventStream(inner, lambda run_id: _deleted(deleted, run_id))
        await stream.register_run("run")
        await stream.add_event("run", Event())
        deleted.add("run")
        assert await stream.add_event("run", Event()) == -1
        assert await stream.replay("run") == []
        assert await stream.get_event_count("run") == 0
        assert await inner.replay("run") == []

    asyncio.run(check())


def test_transient_deleted_run_read_preserves_live_buffer_and_status():
    async def check():
        inner = InMemoryEventStream()

        async def unavailable(_run_id):
            raise RuntimeError("database unavailable")

        stream = DeletedRunEventStream(inner, unavailable)
        await stream.register_run("transient-status")
        await stream.add_event("transient-status", Event())
        await stream.complete_run("transient-status", status=RunStatus.completed)

        assert await stream.get_run_status("transient-status") == RunStatus.completed
        assert len(await inner.replay("transient-status")) == 1

    asyncio.run(check())


def test_transient_deleted_run_read_fails_closed_without_erasing_buffer():
    async def check():
        inner = InMemoryEventStream()

        async def unavailable(_run_id):
            raise RuntimeError("database unavailable")

        stream = DeletedRunEventStream(inner, unavailable)
        await stream.register_run("transient-read")
        await stream.add_event("transient-read", Event())

        with pytest.raises(ReplayFenceUnavailable):
            await stream.replay("transient-read")
        assert len(await inner.replay("transient-read")) == 1

    asyncio.run(check())


def test_tail_preserves_only_bytes_already_yielded():
    async def check():
        inner = InMemoryEventStream()
        deleted = set()
        stream = DeletedRunEventStream(inner, lambda run_id: _deleted(deleted, run_id))
        await stream.register_run("tail-run")
        await stream.add_event("tail-run", Event())
        tail = stream.tail("tail-run")
        first = await anext(tail)
        assert first[0] == 0
        deleted.add("tail-run")
        await tail.aclose()
        assert [item async for item in stream.tail("tail-run")] == []

    asyncio.run(check())


def test_post_publish_deletion_cleans_buffer():
    async def check():
        inner = InMemoryEventStream()
        await inner.register_run("post-publish")
        deleted = iter((False, True))
        stream = DeletedRunEventStream(inner, lambda _run_id: _next(deleted))

        assert await stream.add_event("post-publish", Event()) == -1
        assert await inner.replay("post-publish") == []

    asyncio.run(check())


def test_cancelled_publisher_finishes_before_guard_releases():
    async def check():
        inner = InMemoryEventStream()
        await inner.register_run("cancelled")
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def guard(_run_id, _event, publish):
            entered.set()
            await release.wait()
            try:
                return await publish()
            finally:
                finished.set()

        stream = DeletedRunEventStream(inner, lambda run_id: _deleted(set(), run_id), guard)
        task = asyncio.create_task(stream.add_event("cancelled", Event()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
        assert len(await inner.replay("cancelled")) == 1

    asyncio.run(check())


def test_tail_deletion_and_fence_outage_are_visible():
    async def check():
        inner = InMemoryEventStream()
        checks = iter((False, True))
        deleted = DeletedRunEventStream(inner, lambda _run_id: _next(checks))
        await inner.register_run("tail-fence")
        task = asyncio.create_task(_collect(deleted.tail("tail-fence"), inner, "tail-fence"))
        assert await asyncio.wait_for(task, 2) == []

        outage = iter((False, RuntimeError("database unavailable")))
        unavailable = DeletedRunEventStream(inner, lambda _run_id: _next(outage))
        await inner.register_run("tail-outage")
        task = asyncio.create_task(_collect(unavailable.tail("tail-outage"), inner, "tail-outage"))
        with pytest.raises(ReplayFenceUnavailable):
            await asyncio.wait_for(task, 2)

    asyncio.run(check())


async def _deleted(deleted, run_id):
    return run_id in deleted


async def _next(values):
    value = next(values)
    if isinstance(value, Exception):
        raise value
    return value


async def _collect(stream, inner, run_id):
    async def produce():
        await asyncio.sleep(0.01)
        await inner.add_event(run_id, Event())

    producer = asyncio.create_task(produce())
    try:
        return [item async for item in stream]
    finally:
        await producer
