import json
import os
import select
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.conninfo import make_conninfo

from garp.agentos import create_agentos_app
from garp.demo import _serve_app
from garp.jobs import LocalJobs
from garp.native_checks import NativeChecks
from garp.native_client import NativeChatClient
from garp.native_dispatch import check_run
from garp.store import Store


@pytest.fixture
def checks_dsn(postgres_dsn):
    name = "checks_" + uuid4().hex
    with psycopg.connect(postgres_dsn, autocommit=True) as setup:
        setup.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(name)))
    return make_conninfo(postgres_dsn, dbname=name)


@pytest.mark.parametrize("delete_during_observation", [True, False])
def test_native_check_observations_follow_chat_deletion(checks_dsn, delete_during_observation):
    started, release = Event(), Event()

    class Jobs:
        def authorize(self, owner, job_ref):
            assert (owner, job_ref) == ("local", "fixture")

        def observe(self, owner, job_ref, now):
            if delete_during_observation:
                started.set()
                assert release.wait(10)
            return {"freshness": "fresh", "progress": 7}

    token = "local-check-deletion-fixture"
    with TestClient(
        create_agentos_app(checks_dsn, token, jobs=Jobs()),
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        sid = (
            client.post("/sessions", json={"agent_id": "garp", "user_id": "local"})
            .raise_for_status()
            .json()["session_id"]
        )

        def submit():
            return client.post(
                f"/garp/sessions/{sid}/checks", json={"job_ref": "fixture", "delay_seconds": 3600}
            )

        if delete_during_observation:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(submit)
                try:
                    assert started.wait(6)
                    assert client.delete(f"/sessions/{sid}").status_code == 204
                finally:
                    release.set()
                response = pending.result(8)
            assert response.status_code == 404, response.text
            assert response.json() == submit().json() == {"detail": "no Garp session with an owner"}
        else:
            response = submit()
            assert response.status_code == 201, response.text
            check = response.json()["id"]
            with psycopg.connect(checks_dsn) as db:
                assert db.execute(
                    "SELECT baseline->>'progress' FROM garp_schedules WHERE id=%s", (check,)
                ).fetchone() == ("7",)
            assert client.delete(f"/sessions/{sid}").status_code == 204
            assert client.get(f"/garp/checks/{check}").status_code == 404
        with psycopg.connect(checks_dsn) as db:
            assert db.execute(
                "SELECT count(*) FROM garp_schedules WHERE session_id=%s", (sid,)
            ).fetchone() == (0,)


def test_native_check_reference_migrates_old_rows_and_preserves_legacy_checks(checks_dsn):
    store = Store(checks_dsn)
    store.initialize()
    sid = store.create_session("local")
    due = datetime.now(UTC) + timedelta(hours=1)
    valid = store.create_schedule("local", sid, "fixture", {}, due)
    orphan = store.create_schedule("local", str(uuid4()), "orphan", {}, due, native=True)
    token = "local-check-schema-fixture"
    headers = {"Authorization": f"Bearer {token}"}

    def app():
        return create_agentos_app(checks_dsn, token, import_legacy=True)

    with TestClient(app(), headers=headers) as client:
        assert store.get_schedule("local", valid)["native_owner"] == "local"
        with psycopg.connect(checks_dsn) as db:
            assert (
                db.execute("SELECT 1 FROM garp_schedules WHERE id=%s", (orphan,)).fetchone() is None
            )
        assert client.delete(f"/sessions/{sid}").status_code == 204
    legacy = store.create_schedule("local", sid, "legacy", {}, due)
    with TestClient(app(), headers=headers) as client:
        assert client.get(f"/sessions/{sid}").status_code == 404
        assert store.get_schedule("local", legacy)["native_owner"] is None
        with psycopg.connect(checks_dsn) as db:
            assert db.execute(
                "SELECT id::text FROM garp_schedules WHERE session_id=%s", (sid,)
            ).fetchall() == [(legacy,)]


def test_claimed_check_discards_its_late_observation(checks_dsn, monkeypatch):
    started, release, saved = Event(), Event(), Event()
    results = []
    finish_schedule = Store.finish_schedule

    def finish(self, *args):
        results.append(finish_schedule(self, *args))
        saved.set()
        return results[-1]

    monkeypatch.setattr(Store, "finish_schedule", finish)

    class Jobs:
        reads = 0

        def authorize(self, owner, job_ref):
            assert (owner, job_ref) == ("local", "fixture")

        def observe(self, owner, job_ref, now):
            self.reads += 1
            if self.reads == 2:
                started.set()
                assert release.wait(10)
            return {"freshness": "fresh", "observed_at": now.isoformat(), "state": "running"}

    token = "local-check-claim-fixture"
    with TestClient(
        create_agentos_app(checks_dsn, token, jobs=Jobs()),
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        sid = (
            client.post("/sessions", json={"agent_id": "garp", "user_id": "local"})
            .raise_for_status()
            .json()["session_id"]
        )
        client.post(
            f"/garp/sessions/{sid}/checks", json={"job_ref": "fixture", "delay_seconds": 0}
        ).raise_for_status()
        try:
            assert started.wait(6)
            assert client.delete(f"/sessions/{sid}").status_code == 204
        finally:
            release.set()
        assert saved.wait(6)
        assert results == [False]
        assert client.get(f"/sessions/{sid}/runs").status_code == 404
        with psycopg.connect(checks_dsn) as db:
            assert db.execute("SELECT count(*) FROM garp_schedules").fetchone() == (0,)


def test_migrated_chat_receives_check_after_terminal_exits(checks_dsn, tmp_path):
    postgres_dsn = checks_dsn
    store = Store(postgres_dsn)
    store.initialize()
    sid = store.create_session("local")
    turn = store.start_turn("local", sid, "Remember this chat")
    store.finish_turn("local", sid, turn["token"], "Saved before migration", "A Saved Chat")
    policy = tmp_path / "grants.json"
    policy.write_text('{"local":["batch"]}')
    workspace = LocalJobs(tmp_path, policy)

    def job(progress):
        path = tmp_path / "batch.tmp"
        path.write_text(
            json.dumps(
                {
                    "instance": "batch-fixture",
                    "state": "running",
                    "observed_at": datetime.now(UTC).isoformat(),
                    "progress": progress,
                }
            )
        )
        path.replace(tmp_path / "batch.json")

    job(1)
    old = store.create_schedule(
        "local",
        sid,
        "batch",
        workspace.observe("local", "batch", datetime.now(UTC)),
        datetime.now(UTC) - timedelta(seconds=1),
        datetime.now(UTC) - timedelta(milliseconds=100),
    )
    token = "local-fixture-token"
    app = create_agentos_app(postgres_dsn, token, jobs=workspace, import_legacy=True)

    with _serve_app(app) as origin:
        terminal = subprocess.run(
            [sys.executable, "-m", "garp.cli", "--url", origin, "agentos", sid],
            input="/check batch 120\n/exit\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=45,
            env={**os.environ, "GARP_TOKEN": token},
        )
        assert "Saved before migration" in terminal.stdout
        assert "Check scheduled" in terminal.stdout
        check_id = terminal.stdout.split("Check ID  ", 1)[1].split()[0]
        job(2)
        with psycopg.connect(postgres_dsn) as conn:
            # Make the check due only after the terminal exits and the source advances.
            updated = conn.execute(
                "UPDATE garp_schedules SET due_at = clock_timestamp() "
                "WHERE id = %s AND status = 'pending'",
                (check_id,),
            )
            assert updated.rowcount == 1
        client = NativeChatClient(origin, token)
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                status = client.check_status(check_id)
                if status["delivered_at"]:
                    break
                time.sleep(0.05)
            assert status["delivered_at"], status
            assert status["evidence"]["progress"] == "advanced"
            assert client.check_status(old)["outcome"] == "expired"
            assert client.session(sid)["title"] == "A Saved Chat"
            history = client.history(sid)
            checks = [message for message in history if message.get("kind") == "job_check"]
            assert len(checks) == 2
            assert sum("Job check: complete" in message["content"] for message in checks) == 1
            assert '"progress": "advanced"' in checks[-1]["content"]
            assert store.get_events("local", sid)[-1]["kind"] == "turn.message"

            pending = client.schedule(sid, "batch", 120)["id"]
            endpoint = f"{origin}/garp/sessions/{sid}/checks/{pending}"
            assert client.client.delete(endpoint).status_code == 200
            assert client.client.delete(endpoint).status_code == 409
        finally:
            client.close()

    retry = store.create_schedule(
        "local",
        sid,
        "batch",
        workspace.observe("local", "batch", datetime.now(UTC)),
        datetime.now(UTC),
        native=True,
    )
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import json, os, threading\n"
                "from garp.store import Store\n"
                "store = Store(os.environ['GARP_DATABASE_URL'])\n"
                "print(json.dumps(store.claim_due(native=True), default=str), flush=True)\n"
                "threading.Event().wait()\n"
            ),
        ],
        env={**os.environ, "GARP_DATABASE_URL": postgres_dsn},
        stdout=subprocess.PIPE,
        text=True,
    ) as worker:
        try:
            assert select.select([worker.stdout], [], [], 8)[0]
            claim = json.loads(worker.stdout.readline())
            assert claim["id"] == retry
        finally:
            worker.kill()
            worker.wait(8)

    policy.write_text("{}")
    with store._connect() as conn:
        # Make the abandoned lease due for reclaim without waiting the full lease period.
        conn.execute(
            "UPDATE garp_schedules SET lease_expires = now() - interval '1 second' WHERE id = %s",
            (retry,),
        )
        # Simulate process loss after the native append but before recording delivery.
        conn.execute("UPDATE garp_schedules SET delivered_at = NULL WHERE id = %s", (check_id,))
    with _serve_app(
        create_agentos_app(postgres_dsn, token, jobs=workspace, import_legacy=True)
    ) as origin:
        client = NativeChatClient(origin, token)
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                status = client.check_status(check_id)
                denied = client.check_status(retry)
                if status["delivered_at"] and denied["delivered_at"]:
                    break
                time.sleep(0.05)
            assert status["delivered_at"], status
            assert denied["outcome"] == "denied" and denied["attempt"] == 2
            assert client.check_status(pending)["outcome"] == "cancelled"
            assert not store.finish_schedule(
                retry, claim["lease_token"], "complete", {"bogus": True}
            )
            latest_history = client.history(sid)
            assert latest_history[: len(history)] == history
            assert (
                sum("Job check: complete" in message["content"] for message in latest_history) == 1
            )
            native_id = check_run(sid, check_id, "local", "complete", {}).run_id
            with store._connect() as conn:
                assert (
                    conn.execute(
                        "SELECT count(*) AS count FROM ai.agno_runs WHERE run_id = %s", (native_id,)
                    ).fetchone()["count"]
                    == 1
                )
        finally:
            client.close()


def test_standalone_native_cron_delivers_to_chat(checks_dsn, tmp_path):
    policy = tmp_path / "grants.json"
    policy.write_text('{"local":["batch"]}')
    (tmp_path / "batch.json").write_text(
        json.dumps(
            {
                "instance": "batch-fixture",
                "state": "running",
                "observed_at": datetime.now(UTC).isoformat(),
                "progress": 1,
            }
        )
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    token = "scheduled-test-bearer-token"
    origin = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "GARP_TOKEN": token,
        "GARP_DATABASE_URL": checks_dsn,
        "GARP_JOBS_DIR": str(tmp_path),
        "GARP_POLICY_FILE": str(policy),
    }
    log = tmp_path / "service.log"
    with (
        log.open("w") as output,
        subprocess.Popen(
            [sys.executable, "-m", "garp.cli", "serve", "--port", str(port)],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        ) as service,
    ):
        client = NativeChatClient(origin, token)
        try:
            # Allow a cold Python import and database migrations on a busy test host.
            limit = time.monotonic() + 45
            while time.monotonic() < limit and service.poll() is None:
                try:
                    ready = client.catalog()["job_checks"]
                except RuntimeError:
                    time.sleep(0.1)
                else:
                    if ready:
                        break
            else:
                pytest.fail(
                    f"Garp service did not become ready (exit={service.poll()}).\n{log.read_text()}"
                )
            sid = client.create_session()
            response = client.client.post(
                origin + "/schedules",
                json={
                    "name": "Check progress each minute",
                    "cron_expr": "* * * * *",
                    "endpoint": f"/garp/sessions/{sid}/checks?source=fixture",
                    "payload": {"job_ref": "batch", "delay_seconds": 0},
                },
            )
            assert response.status_code == 201, response.text
            key = response.json()["id"]
            with psycopg.connect(checks_dsn) as conn:
                # Move the first due time forward so the native poller can claim it now.
                conn.execute(
                    "UPDATE ai.agno_schedules SET next_run_at = extract(epoch FROM now())::bigint - 1 WHERE id = %s",
                    (key,),
                )
            limit = time.monotonic() + 25
            while time.monotonic() < limit:
                history = client.history(sid)
                if history:
                    break
                time.sleep(0.1)
            assert len(history) == 1 and history[0]["kind"] == "job_check", log.read_text()
            runs = client.client.get(origin + f"/schedules/{key}/runs")
            assert runs.status_code == 200, runs.text
            assert "success" in str(runs.json())
            follow_up = (
                client.client.post(
                    origin + "/schedules",
                    json={
                        "name": "Manual AgentOS follow-up",
                        "endpoint": "/agents/garp/runs",
                        "payload": {"session_id": sid, "message": "Later message"},
                    },
                )
                .raise_for_status()
                .json()["id"]
            )
            assert client.client.delete(origin + f"/sessions/{sid}").status_code == 204
            assert client.client.get(origin + f"/schedules/{key}").status_code == 404
            assert client.client.get(origin + f"/schedules/{follow_up}").status_code == 404
            with psycopg.connect(checks_dsn) as conn:
                assert not conn.execute(
                    "SELECT 1 FROM ai.agno_schedule_runs WHERE schedule_id=%s", (key,)
                ).fetchone()
        finally:
            client.close()
            service.terminate()
            try:
                service.wait(12)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait(5)


def test_failed_delivery_does_not_block_another_due_result(checks_dsn, monkeypatch):
    store = Store(checks_dsn)
    store.initialize()
    sid = str(uuid4())
    now = datetime.now(UTC)
    first = store.create_schedule(
        "local", sid, "batch", {}, now - timedelta(seconds=2), native=True
    )
    second = store.create_schedule(
        "local", sid, "batch", {}, now - timedelta(seconds=1), native=True
    )
    for expected in (first, second):
        claim = store.claim_due(native=True)
        assert claim["id"] == expected
        assert store.finish_schedule(expected, claim["lease_token"], "complete", {})
    attempts = []

    def publish(_db, _sid, occurrence, *_args):
        attempts.append(occurrence)
        if occurrence == first:
            raise RuntimeError("temporary failure")

    monkeypatch.setattr("garp.native_checks.publish_check", publish)
    worker = NativeChecks(store, None, None)
    assert worker.work_once() and worker.work_once()
    assert attempts == [first, second]
    assert store.get_schedule("local", first)["delivered_at"] is None
    assert store.get_schedule("local", second)["delivered_at"] is not None
