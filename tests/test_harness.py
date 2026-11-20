import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import psycopg
import pytest
import uvicorn
from fastapi.testclient import TestClient

from garp.api import create_app
from garp.jobs import LocalJobs, compare_observations
from garp.models import default_model
from garp.store import Store
from garp.worker import run_once

TOKEN = "synthetic-cli-token-aaaaaaaa"
OTHER = "synthetic-other-token-bbbbb"


@pytest.fixture
def fixture(postgres_dsn, tmp_path):
    temp = tmp_path / "job-progress"
    temp.mkdir()
    policy = tmp_path / "jobs-policy.json"
    policy.write_text(json.dumps({"host-a:user": ["job"], "host-b:user": ["job"]}))
    store = Store(postgres_dsn)
    store.initialize()
    with store._connect() as conn:
        conn.execute("TRUNCATE garp_events, garp_schedules, garp_sessions")
    return store, LocalJobs(temp, policy)


def write_job(jobs, *, progress, state="running", at=None, instance="incarnation"):
    path = jobs.directory / ".job-tmp"
    path.write_text(
        json.dumps(
            {
                "state": state,
                "observed_at": (at or datetime.now(UTC)).isoformat(),
                "instance": instance,
                "progress": progress,
            }
        )
    )
    path.replace(jobs.directory / "job.json")


def app_for(store, jobs, executor=None, titler=None):
    options = {"executor": executor} if executor else {}
    return create_app(
        store, jobs, {TOKEN: "host-a:user", OTHER: "host-b:user"}, titler=titler, **options
    )


def headers(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_worker_can_start_before_api(postgres_dsn, tmp_path):
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute("CREATE DATABASE garp_worker_first")
    dsn = postgres_dsn + " dbname=garp_worker_first"
    result = subprocess.run(
        [Path(sys.executable).with_name("garp-worker"), "--once"],
        env={
            **os.environ,
            "GARP_DATABASE_URL": dsn,
            "GARP_JOBS_DIR": str(tmp_path),
            "GARP_POLICY_FILE": str(tmp_path / "unused.json"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=12,
    )
    assert result.returncode == 0, result.stderr
    assert Store(dsn).create_session("host-a:user")


def test_separate_cli_and_worker_replay_one_result(fixture, postgres_dsn):
    store, jobs = fixture
    write_job(jobs, progress=3)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app_for(store, jobs), log_level="warning"))
    server_thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    server_thread.start()
    try:
        for _ in range(300):
            if server.started:
                break
            time.sleep(0.01)
        assert server.started

        env = os.environ.copy()
        env.update(
            GARP_URL=origin,
            GARP_TOKEN=TOKEN,
            GARP_DATABASE_URL=postgres_dsn,
            GARP_POLICY_FILE=str(jobs.policy),
            GARP_JOBS_DIR=str(jobs.directory),
        )
        commands = Path(sys.executable).parent

        def cli(*args, token=TOKEN):
            result = subprocess.run(
                [commands / "garp", *args],
                env={**env, "GARP_TOKEN": token},
                check=True,
                capture_output=True,
                text=True,
                timeout=12,
            )
            return json.loads(result.stdout)

        session = cli("session", "new")["id"]
        occurrence = cli("schedule", session, "job", "--after", "0")["id"]
        assert cli("status", occurrence)["status"] == "overdue"

        # Exercise a real worker-process death after claiming. Its lease expires
        # while both the scheduling CLI and the claimed worker are gone.
        claimant = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                (
                    "import os,time; from garp.store import Store; "
                    "store=Store(os.environ['GARP_DATABASE_URL']); "
                    "assert store.claim_due(1); print('claimed'); time.sleep(30)"
                ),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert claimant.stdout.readline().strip() == "claimed"
        finally:
            claimant.kill()
            claimant.communicate(timeout=5)
        time.sleep(1.1)
        assert cli("status", occurrence)["status"] == "lease_expired"
        write_job(jobs, progress=8)

        for _ in range(2):
            subprocess.run(
                [commands / "garp-worker", "--once"],
                env=env,
                check=True,
                capture_output=True,
                timeout=12,
            )
        replay = cli("events", session)
        assert len(replay["events"]) == 1
        event = replay["events"][0]
        assert event["kind"] == "schedule.result"
        assert event["data"]["schedule_id"] == occurrence
        assert event["data"]["evidence"]["progress"] == "advanced"
        assert event["data"]["evidence"]["counter"] == 8
        assert cli("status", occurrence)["attempt"] == 2
        assert cli("events", session, "--after", str(replay["cursor"]))["events"] == []
        assert (
            httpx.get(f"{origin}/v1/schedules/{occurrence}", headers=headers(OTHER)).status_code
            == 404
        )
        assert (
            httpx.get(f"{origin}/v1/sessions/{session}/events", headers=headers(OTHER)).status_code
            == 404
        )
    finally:
        server.should_exit = True
        server_thread.join(timeout=8)
        listener.close()


def test_worker_rechecks_grants_and_deadlines(fixture, monkeypatch):
    store, jobs = fixture
    store.initialize()
    now = datetime.now(UTC)
    session = store.create_session("host-a:user")
    write_job(jobs, progress=1)
    base = jobs.observe("host-a:user", "job", now)
    denied = store.create_schedule("host-a:user", session, "job", base, now)
    jobs.policy.write_text(json.dumps({"host-a:user": []}))

    def must_not_observe(*_args):
        raise AssertionError("an expired occurrence must not access the job")

    assert run_once(store, jobs)
    state = store.get_schedule("host-a:user", denied)
    assert state["outcome"] == "denied" and "counter" not in state["evidence"]
    expired = store.create_schedule("host-a:user", session, "job", base, now, now)
    monkeypatch.setattr(jobs, "observe", must_not_observe)
    assert run_once(store, jobs)
    assert store.get_schedule("host-a:user", expired)["outcome"] == "expired"


def test_job_evidence_distinguishes_stalled_stale_and_lost(fixture):
    _, jobs = fixture
    now = datetime.now(UTC)
    write_job(jobs, progress=2, at=now - timedelta(minutes=4))
    baseline = jobs.observe("host-a:user", "job", now)
    latest = compare_observations(baseline, baseline, now)
    assert latest["state"] == "unknown" and latest["freshness"] == "stale"
    assert latest["last_seen_at"] == baseline["observed_at"]

    write_job(jobs, progress=2, at=now)
    still_running = jobs.observe("host-a:user", "job", now)
    assert compare_observations(baseline, still_running, now)["progress"] == "unchanged"
    assert compare_observations(still_running, still_running, now)["progress"] == "unmeasurable"
    write_job(jobs, progress=9, instance="replacement", at=now)
    replaced = jobs.observe("host-a:user", "job", now)
    assert compare_observations(baseline, replaced, now)["progress"] == "unmeasurable"
    write_job(jobs, progress=2, state="lost", at=now)
    lost = jobs.observe("host-a:user", "job", now)
    assert compare_observations(baseline, lost, now)["state"] == "lost"


def test_missing_observation_before_and_after_admission(fixture):
    store, jobs = fixture
    with TestClient(app_for(store, jobs)) as client:
        session = client.post("/v1/sessions", headers=headers()).json()["id"]
        path = f"/v1/sessions/{session}/schedules"
        request = {"job_ref": "job", "delay_seconds": 0}
        assert client.post(path, json=request, headers=headers()).status_code == 404
        write_job(jobs, progress=1)
        scheduled = client.post(path, json=request, headers=headers())
        assert scheduled.status_code == 201
        (jobs.directory / "job.json").unlink()
        assert run_once(store, jobs)
        result = store.get_schedule("host-a:user", scheduled.json()["id"])
        assert result["outcome"] == "failed"
        assert result["evidence"]["state"] == "unknown"
        assert result["evidence"]["freshness"] == "missing"
        assert result["evidence"]["last_seen_at"]


def test_turn_history_and_schedule_reenter_same_chat(fixture):
    store, jobs = fixture
    write_job(jobs, progress=1)
    history_seen = []

    async def executor(_owner, _session, _run, history, text):
        history_seen.append(history)
        return f"received {text}"

    with TestClient(app_for(store, jobs, executor)) as client:
        assert client.post("/v1/sessions").status_code == 401
        session = client.post("/v1/sessions", headers=headers()).json()["id"]
        path = f"/v1/sessions/{session}"
        initial = client.post(f"{path}/turns", json={"text": "start"}, headers=headers())
        assert initial.status_code == 200, initial.text
        schedule = client.post(
            f"{path}/schedules", json={"job_ref": "job", "delay_seconds": 0}, headers=headers()
        )
        assert schedule.status_code == 201, schedule.text
        assert run_once(store, jobs)
        follow_up = client.post(f"{path}/turns", json={"text": "what changed?"}, headers=headers())
        assert follow_up.status_code == 200, follow_up.text
        assert history_seen[0] == []
        assert [row["role"] for row in history_seen[1]] == ["user", "assistant", "assistant"]
        assert "Scheduled check result:" in history_seen[1][-1]["content"]
        result = client.get(f"{path}/events", headers=headers()).json()
        assert [event["seq"] for event in result["events"]] == list(range(1, 6))
        assert all("lease_token" not in event["data"] for event in result["events"])


def test_three_word_title_is_owned_by_session_and_title_failure_keeps_reply(fixture):
    store, jobs = fixture
    openings = []

    async def executor(_owner, _session, _run, _history, text):
        return f"received {text}"

    async def titler(opening):
        openings.append(opening)
        if len(openings) == 1:
            raise RuntimeError("fixture failed")
        return "“Reliable Worker Leases Explained”\nextra"

    with TestClient(app_for(store, jobs, executor, titler)) as client:
        sid = client.post("/v1/sessions", headers=headers()).json()["id"]
        path = f"/v1/sessions/{sid}"
        first = client.post(
            f"{path}/turns", json={"text": "Design worker leases"}, headers=headers()
        )
        assert first.status_code == 200
        assert first.json()["title"] is None
        second = client.post(f"{path}/turns", json={"text": "Continue"}, headers=headers())
        assert second.json()["title"] == "Reliable Worker Leases"
        result = client.get(path, headers=headers())
        assert result.json() == {
            "id": sid,
            "title": "Reliable Worker Leases",
            "model": default_model(),
            "effort": None,
            "web_search": False,
        }
        assert client.get(path, headers=headers(OTHER)).status_code == 404
        assert client.get("/v1/sessions", headers=headers(OTHER)).json() == {"sessions": []}
        assert client.get("/v1/sessions", headers=headers()).json()["sessions"] == [result.json()]
        third = client.post(f"{path}/turns", json={"text": "Continue again"}, headers=headers())
        assert third.json()["title"] == "Reliable Worker Leases"
        assert openings == ["Design worker leases", "Design worker leases"]


def test_model_selection_is_owner_scoped_persistent_and_used_for_title(fixture, monkeypatch):
    from garp import executor

    store, jobs = fixture
    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("GARP_OPENROUTER_ZDR_ONLY", "true")

    async def mock_agent(_messages, *, title_agent=False, model_id=None, **_ids):
        calls.append((model_id, title_agent))
        return executor.Reply("Pinned Model" if title_agent else "Reply", [])

    monkeypatch.setattr(executor, "_run_agent", mock_agent)
    with TestClient(app_for(store, jobs)) as client:
        sid = client.post("/v1/sessions", headers=headers()).json()["id"]
        route = f"/v1/sessions/{sid}"
        presets = {
            row["id"]: row for row in client.get("/v1/models", headers=headers()).json()["models"]
        }
        assert presets["meta/muse-ember"]["available"] is False
        assert presets["anthropic/claude-haiku-5.5"]["endpoint"] == "amazon-bedrock"
        backend = {"model": "xiaomi/mimo-v2.6-flash"}
        assert client.put(route + "/model", json=backend, headers=headers(OTHER)).status_code == 404
        assert client.put(route + "/model", json=backend, headers=headers()).status_code == 200
        disabled = client.put(
            route + "/model", json={"model": "meta/muse-ember"}, headers=headers()
        )
        assert disabled.status_code == 422
        turn = client.post(
            route + "/turns", json={"text": "Test the selected model"}, headers=headers()
        )
        assert turn.status_code == 200
        assert turn.json()["model"] == backend["model"]
        assert calls == [(backend["model"], False), (backend["model"], True)]
        assert client.get(route, headers=headers()).json()["model"] == backend["model"]
        unsupported = client.patch(route + "/settings", json={"effort": "high"}, headers=headers())
        assert unsupported.status_code == 422
        monkeypatch.setenv("GARP_OPENROUTER_NATIVE_SEARCH_ONLY", "true")
        terra = "openai/gpt-6.1-terra"
        assert (
            client.put(route + "/model", json={"model": terra}, headers=headers()).status_code
            == 200
        )
        settings = {"effort": "high", "web_search": True}
        assert (
            client.patch(route + "/settings", json=settings, headers=headers(OTHER)).status_code
            == 404
        )
        selected = client.patch(route + "/settings", json=settings, headers=headers())
        assert selected.json() == {"model": terra, **settings}
        turn = client.post(
            route + "/turns", json={"text": "Check sources"}, headers=headers()
        ).json()
        assert turn["events"][-1]["data"] == {
            "role": "assistant",
            "content": "Reply",
            "model": terra,
            **settings,
        }
        switched = client.put(route + "/model", json=backend, headers=headers())
        assert switched.json() == {"model": backend["model"], "effort": None, "web_search": False}
