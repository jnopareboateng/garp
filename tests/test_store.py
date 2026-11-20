import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from garp.store import Store


@pytest.mark.usefixtures("configure_store")
class StoreTest(unittest.TestCase):
    def setUp(self):
        with self.store._connect() as conn:
            conn.execute("TRUNCATE garp_events, garp_schedules, garp_sessions")
        self.now = datetime.now(UTC)
        self.session = self.store.create_session("alice")

    def schedule(self):
        return self.store.create_schedule("alice", self.session, "job", {"v": 1}, self.now)

    def test_scope_cursor_and_completion(self):
        job = self.schedule()
        with self.assertRaises(LookupError):
            self.store.get_schedule("bob", job)
        with self.assertRaises(LookupError):
            self.store.get_events("bob", self.session)
        with self.assertRaises(LookupError):
            self.store.start_turn("bob", self.session, "x")
        claim = self.store.claim_due()
        self.assertEqual(claim["id"], job)
        self.assertEqual(claim["owner"], "alice")
        self.assertEqual(claim["baseline"], {"v": 1})
        self.assertTrue(
            self.store.finish_schedule(job, claim["lease_token"], "complete", {"ok": True})
        )
        self.assertFalse(self.store.finish_schedule(job, claim["lease_token"], "complete", {}))
        events = self.store.get_events("alice", self.session)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0]["data"], {"schedule_id": job, "outcome": "complete", "evidence": {"ok": True}}
        )
        self.assertEqual(self.store.get_events("alice", self.session, events[0]["seq"]), [])
        self.assertEqual(self.store.get_schedule("alice", job)["status"], "complete")

    def test_lease_reclaim_and_turn_history(self):
        job = self.schedule()
        old = self.store.claim_due(1)
        self.assertIsNone(self.store.claim_due())
        with self.store._connect() as conn:
            conn.execute(
                "UPDATE garp_schedules SET lease_expires = clock_timestamp() WHERE id = %s", (job,)
            )
        fresh = self.store.claim_due()
        self.assertEqual(fresh["attempt"], 2)
        self.assertFalse(self.store.finish_schedule(job, old["lease_token"], "complete", {}))
        with self.assertRaises(RuntimeError):
            self.store.start_turn("alice", self.session, "blocked")
        self.assertTrue(self.store.finish_schedule(job, fresh["lease_token"], "failed", {}))
        turn = self.store.start_turn("alice", self.session, "first", 1)
        self.assertEqual(turn["history"], [])
        with self.store._connect() as conn:
            conn.execute(
                "UPDATE garp_sessions SET turn_expires = clock_timestamp() WHERE id = %s",
                (self.session,),
            )
        newer = self.store.start_turn("alice", self.session, "second")
        self.assertFalse(self.store.finish_turn("alice", self.session, turn["token"], "stale"))
        self.assertTrue(self.store.finish_turn("alice", self.session, newer["token"], "answer"))
        self.assertFalse(self.store.finish_turn("alice", self.session, newer["token"], "duplicate"))
        next_turn = self.store.start_turn("alice", self.session, "third")
        self.assertEqual(
            next_turn["history"],
            [{"role": "user", "content": "second"}, {"role": "assistant", "content": "answer"}],
        )
        self.assertTrue(self.store.finish_turn("alice", self.session, next_turn["token"], None))
        self.assertEqual(
            self.store.start_turn("alice", self.session, "fourth")["history"],
            next_turn["history"],
        )

    def test_future_due_time(self):
        self.store.create_schedule("alice", self.session, "job", {}, self.now + timedelta(hours=1))
        self.assertIsNone(self.store.claim_due())

    def test_turn_blocks_claim(self):
        self.schedule()
        turn = self.store.start_turn("alice", self.session, "hello")
        self.assertIsNone(self.store.claim_due())
        self.assertTrue(self.store.finish_turn("alice", self.session, turn["token"], None))
        self.assertIsNotNone(self.store.claim_due())

    def test_concurrent_claims(self):
        self.schedule()
        barrier = threading.Barrier(2)

        def claim():
            barrier.wait()
            return self.store.claim_due()

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda _: claim(), range(2)))
        self.assertEqual(sum(c is not None for c in claims), 1)


@pytest.fixture(scope="class")
def configure_store(request, postgres_dsn):
    request.cls.store = Store(postgres_dsn)
    request.cls.store.initialize()
