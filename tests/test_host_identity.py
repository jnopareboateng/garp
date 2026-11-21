import asyncio
import json
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import NAMESPACE_URL, uuid4, uuid5

import jwt
import psycopg
import pytest
from agno.db.postgres import PostgresDb
from agno.db.schemas.scheduler import Schedule
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from sqlalchemy import URL

from garp.agentos import HostAuth, create_agentos_app
from garp.jobs import LocalJobs
from garp.native_checks import NativeChecks
from garp.recurring_checks import HostedCheckExecutor
from garp.store import Store

ISSUER = "https://host.example"
AUDIENCE = "garp-fixture"
ALICE = "tenant-a:alice"
BOB = "tenant-b:bob"
SCOPES = ["sessions:read", "sessions:write", "sessions:delete", "agents:garp:run"]


@pytest.fixture
def host_dsn(postgres_dsn):
    name = "host_identity_" + uuid4().hex
    with psycopg.connect(postgres_dsn, autocommit=True) as setup:
        setup.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(name)))
    return make_conninfo(postgres_dsn, dbname=name)


@pytest.fixture
def host_service(host_dsn, monkeypatch, tmp_path):
    key = secrets.token_urlsafe(48)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    grants = tmp_path / "grants.json"
    grants.write_text(json.dumps({ALICE: ["batch"]}))
    (tmp_path / "batch.json").write_text(
        json.dumps(
            {
                "instance": "batch-1",
                "observed_at": datetime.now(UTC).isoformat(),
                "state": "running",
                "progress": 1,
            }
        )
    )

    def create_app():
        return create_agentos_app(
            host_dsn,
            jobs=LocalJobs(tmp_path, grants),
            host_auth=HostAuth(
                issuer=ISSUER, audience=AUDIENCE, verification_key=key, algorithm="HS256"
            ),
        )

    def token(sub=ALICE, scopes=SCOPES, key_override=None, **claims):
        payload = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": sub,
            "exp": datetime.now(UTC) + timedelta(minutes=5),
            "scopes": scopes,
        }
        payload.update(claims)
        payload = {k: v for k, v in payload.items() if v is not None}
        return jwt.encode(payload, key_override or key, algorithm="HS256")

    return create_app, token


@pytest.fixture
def host_client(host_service):
    create_app, token = host_service
    with TestClient(create_app()) as client:
        yield client, token


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_host_tokens_require_valid_signature_and_claims(host_client):
    client, token = host_client
    assert client.get("/garp/models").status_code == 401
    assert client.get("/garp/models", headers=auth(token())).status_code == 200
    invalid = [
        token(iss="https://other.example"),
        token(aud="other-audience"),
        token(key_override=secrets.token_urlsafe(48)),
        token(exp=datetime.now(UTC) - timedelta(minutes=1)),
        token(exp=None),
        token(sub=None),
        token(sub="alice"),
        token(session_id=str(uuid4())),
        jwt.encode(
            {
                "iss": ISSUER,
                "aud": AUDIENCE,
                "sub": ALICE,
                "exp": datetime.now(UTC) + timedelta(minutes=5),
                "scopes": SCOPES,
            },
            key="",
            algorithm="none",
        ),
    ]
    for bad in invalid:
        assert client.get("/garp/models", headers=auth(bad)).status_code == 401
        assert client.get("/sessions", headers=auth(bad)).status_code == 401


def test_native_scopes_are_enforced(host_client):
    client, token = host_client
    no_scopes = auth(token(scopes=[]))
    assert client.get("/sessions", headers=no_scopes).status_code == 403
    assert client.post("/sessions", headers=no_scopes, json={}).status_code == 403
    assert (
        client.post("/agents/garp/runs", headers=no_scopes, data={"message": "hello"}).status_code
        == 403
    )
    read_only = auth(token(scopes=["sessions:read"]))
    assert client.get("/sessions", headers=read_only).status_code == 200
    assert client.post("/sessions", headers=read_only, json={}).status_code == 403
    assert (
        client.get("/sessions", headers=auth(token(scopes=["agent_os:admin"]))).status_code == 403
    )
    memory_writer = auth(token(scopes=["memories:write"]))
    assert client.post("/optimize-memories", headers=memory_writer, json={}).status_code == 404
    invalid_session = client.post("/sessions", headers=auth(token()), content="invalid")
    assert invalid_session.status_code == 422


def test_host_subject_owns_sessions_despite_forged_user_id(host_client):
    client, token = host_client
    alice, bob = auth(token(ALICE)), auth(token(BOB))

    def create(headers, name, forged):
        response = client.post(
            "/sessions?type=agent",
            headers=headers,
            json={
                "agent_id": "garp",
                "session_name": name,
                "user_id": forged,
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["user_id"] != forged
        return response.json()["session_id"]

    alice_id = create(alice, "Alice private", BOB)
    bob_id = create(bob, "Bob private", ALICE)
    for headers, own, other, spoof in [
        (alice, alice_id, bob_id, BOB),
        (bob, bob_id, alice_id, ALICE),
    ]:
        response = client.get("/sessions", headers=headers, params={"user_id": spoof})
        assert response.status_code in (200, 403), response.text
        if response.status_code == 200:
            ids = {row["session_id"] for row in response.json()["data"]}
            assert own in ids and other not in ids
        assert client.get(f"/sessions/{own}", headers=headers).status_code == 200
        assert client.get(
            f"/sessions/{other}", headers=headers, params={"user_id": spoof}
        ).status_code in (403, 404)
        assert client.get(
            f"/sessions/{other}/runs", headers=headers, params={"user_id": spoof}
        ).status_code in (403, 404)
        assert client.patch(
            f"/sessions/{other}",
            headers=headers,
            params={"user_id": spoof},
            json={"session_name": "stolen"},
        ).status_code in (403, 404)
        response = client.post(
            "/agents/garp/runs",
            headers=headers,
            data={"message": "intrusion", "session_id": other, "user_id": spoof},
        )
        assert response.status_code in (403, 404, 409), response.text
        # Agno may return 204 for a scoped deletion that matched no rows.
        response = client.delete(f"/sessions/{other}", headers=headers, params={"user_id": spoof})
        assert response.status_code in (204, 403, 404)
        assert client.get(f"/sessions/{own}", headers=headers).json()["session_name"] != "stolen"

    assert client.get(f"/sessions/{alice_id}", headers=alice).status_code == 200
    assert client.get(f"/sessions/{bob_id}", headers=bob).status_code == 200
    assert (
        client.post("/sessions", headers=alice, json={"session_id": str(uuid4())}).status_code
        == 422
    )
    response = client.post(
        "/agents/garp/runs", headers=alice, data={"message": "hello", "session_id": str(uuid4())}
    )
    assert response.status_code == 404, response.text
    response = client.post(
        "/agents/garp/runs", headers=alice, data={"message": "hello", "run_id": str(uuid4())}
    )
    assert response.status_code == 422, response.text

    query = {"filter": {"key": "session_name", "op": "CONTAINS", "value": "private"}}
    for headers, expected in [(alice, alice_id), (bob, bob_id)]:
        response = client.post("/sessions/search", headers=headers, json=query)
        assert response.status_code == 200, response.text
        assert [r["session_id"] for r in response.json()["data"]] == [expected]


def test_hosted_check_uses_jwt_owner_and_job_grant(host_client):
    client, token = host_client
    checks = [*SCOPES, "checks:write", "checks:read"]
    alice, bob = auth(token(ALICE, scopes=checks)), auth(token(BOB, scopes=checks))

    def session(headers):
        response = client.post("/sessions", headers=headers, json={"agent_id": "garp"})
        assert response.status_code == 201, response.text
        return response.json()["session_id"]

    alice_id, bob_id = session(alice), session(bob)
    alice_path = f"/garp/sessions/{alice_id}/checks"
    schedule = {"delay_seconds": 0, "job_ref": "batch"}
    assert client.post(alice_path, headers=bob, json=schedule).status_code == 404
    assert client.post(alice_path, headers=auth(token()), json=schedule).status_code == 403
    assert (
        client.post(f"/garp/sessions/{bob_id}/checks", headers=bob, json=schedule).status_code
        == 403
    )

    response = client.post(alice_path, headers=alice, json=schedule)
    assert response.status_code == 201, response.text
    check_id = response.json()["id"]
    assert client.get(f"/garp/checks/{check_id}", headers=bob).status_code == 404
    assert client.delete(f"{alice_path}/{check_id}", headers=bob).status_code == 404
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        status = client.get(f"/garp/checks/{check_id}", headers=alice)
        assert status.status_code == 200, status.text
        if status.json()["delivered_at"]:
            break
        time.sleep(0.05)
    assert status.json()["delivered_at"], status.text
    runs = client.get(f"/sessions/{alice_id}/runs", headers=alice)
    assert runs.status_code == 200 and len(runs.json()) == 1, runs.text
    assert client.get(f"/sessions/{alice_id}/runs", headers=bob).status_code == 404


def test_hosted_cron_reclaims_due_tick_and_deduplicates_lost_ack(
    host_client, host_dsn, tmp_path, monkeypatch
):
    client, token = host_client
    scopes = [
        *SCOPES,
        "checks:read",
        "checks:write",
        "schedules:read",
        "schedules:write",
        "schedules:delete",
    ]
    alice = auth(token(ALICE, scopes=scopes))
    bob = auth(token(BOB, scopes=scopes))
    session = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
    assert session.status_code == 201, session.text
    session_id = session.json()["session_id"]
    path = f"/garp/sessions/{session_id}/recurring-checks"
    spec = {
        "name": "Watch batch",
        "cron_expr": "0 12 1 * *",
        "job_ref": "batch",
        "max_retries": 1,
        "retry_delay_seconds": 1,
    }
    assert client.post(path, headers=bob, json=spec).status_code == 404
    assert client.post(path, headers=auth(token()), json=spec).status_code == 403
    assert client.post(path, headers=alice, json={**spec, "endpoint": "/health"}).status_code == 422
    assert (
        client.post(path, headers=alice, json={**spec, "cron_expr": "0 * * * * *"}).status_code
        == 422
    )
    created = client.post(path, headers=alice, json=spec)
    assert created.status_code == 201, created.text
    schedule_id = created.json()["id"]
    assert created.json()["user_id"] == ALICE
    assert client.post(path, headers=alice, json=spec).status_code == 409
    assert client.post("/schedules", headers=alice, json={}).status_code == 404
    assert client.get("/schedules", headers=bob).json()["data"] == []
    assert client.get(f"/schedules/{schedule_id}", headers=bob).status_code == 404

    admit = NativeChecks.admit
    seen = []

    def lose_ack(self, *args, **kwargs):
        result = admit(self, *args, **kwargs)
        if kwargs.get("occurrence_id"):
            seen.append(result["id"])
            if len(seen) == 1:
                raise RuntimeError("Synthetic error after durable admission")
        return result

    monkeypatch.setattr(NativeChecks, "admit", lose_ack)
    history_url = f"/schedules/{schedule_id}/runs"
    assert client.get(history_url, headers=bob).status_code == 404
    assert client.post(f"/schedules/{schedule_id}/disable", headers=bob).status_code == 404

    def due(stamp, *, stale=False):
        with psycopg.connect(host_dsn) as db:
            db.execute(
                """UPDATE ai.garp_host_schedules SET next_run_at=%s, locked_by=%s, locked_at=%s
                   WHERE id=%s""",
                (
                    stamp,
                    "lost-worker" if stale else None,
                    int(time.time()) - 301 if stale else None,
                    schedule_id,
                ),
            )

    def attempts_until(predicate):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            response = client.get(history_url, headers=alice)
            assert response.status_code == 200, response.text
            attempts = response.json()["data"]
            if predicate(attempts):
                return attempts
            time.sleep(0.1)
        pytest.fail(f"Cron attempts did not settle: {attempts}")

    first_tick = int(time.time()) - 180
    due(first_tick, stale=True)
    attempts = attempts_until(lambda rows: any(row["status"] == "success" for row in rows))
    assert {row["status"] for row in attempts} == {"failed", "success"}
    assert len(seen) == 2 and seen[0] == seen[1]
    successful = next(row for row in attempts if row["status"] == "success")
    assert successful["output"]["check_id"] == seen[0]
    assert "Synthetic error" not in json.dumps(attempts)
    assert client.get(f"{history_url}/{successful['id']}", headers=bob).status_code == 404
    assert client.get(f"/garp/checks/{seen[0]}", headers=bob).status_code == 404
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        occurrence = client.get(f"/garp/checks/{seen[0]}", headers=alice).json()
        if occurrence["delivered_at"]:
            break
        time.sleep(0.1)
    assert occurrence["delivered_at"]
    assert datetime.fromisoformat(occurrence["due_at"]).timestamp() == first_tick
    with psycopg.connect(host_dsn) as db:
        assert db.execute("SELECT count(*) FROM garp_schedules").fetchone() == (1,)
    assert len(client.get(f"/sessions/{session_id}/runs", headers=alice).json()) == 1

    # Revoking the current job grant stops a later tick even though the schedule remains enabled.
    (tmp_path / "grants.json").write_text("{}")
    due(first_tick + 60)
    attempts_until(
        lambda rows: (
            sum(row.get("status_code") == 403 and row["status"] == "failed" for row in rows) == 2
        )
    )
    assert len(seen) == 2
    disabled = client.post(f"/schedules/{schedule_id}/disable", headers=alice)
    assert disabled.status_code == 200 and not disabled.json()["enabled"], disabled.text
    enable = f"/garp/recurring-checks/{schedule_id}/enable"
    assert client.post(enable, headers=bob).status_code == 404
    assert client.post(enable, headers=alice).status_code == 403
    (tmp_path / "grants.json").write_text(json.dumps({ALICE: ["batch"]}))
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        enabled = client.post(enable, headers=alice)
        if enabled.status_code != 409:
            break
        time.sleep(0.05)
    assert enabled.status_code == 200 and enabled.json()["enabled"], enabled.text

    claimed, resume_dispatch = Event(), Event()

    def wait_to_admit(self, *args, **kwargs):
        if kwargs.get("occurrence_id") and not claimed.is_set():
            claimed.set()
            if not resume_dispatch.wait(12):
                raise RuntimeError("Fixture did not release the claim")
        return admit(self, *args, **kwargs)

    monkeypatch.setattr(NativeChecks, "admit", wait_to_admit)
    due(first_tick + 120)
    try:
        assert claimed.wait(10)
        assert client.post(f"/schedules/{schedule_id}/disable", headers=alice).status_code == 200
        assert client.post(enable, headers=alice).status_code == 409
    finally:
        resume_dispatch.set()
    attempts_until(lambda rows: any(row["status"] == "cancelled" for row in rows))
    with psycopg.connect(host_dsn) as db:
        assert db.execute("SELECT count(*) FROM garp_schedules").fetchone() == (1,)
    assert client.delete(f"/schedules/{schedule_id}", headers=bob).status_code == 404
    assert client.delete(f"/schedules/{schedule_id}", headers=alice).status_code == 204
    assert client.get(history_url, headers=alice).status_code == 404


def test_hosted_cron_preserves_tick_when_service_stops(host_service, host_dsn, monkeypatch):
    create_app, token = host_service
    alice = auth(
        token(
            ALICE,
            scopes=[*SCOPES, "checks:write", "checks:read", "schedules:write", "schedules:read"],
        )
    )
    first_claim = Event()
    dispatch = HostedCheckExecutor._call_endpoint

    async def stop_before_admission(self, schedule):
        if not first_claim.is_set():
            first_claim.set()
            await asyncio.Event().wait()
        return await dispatch(self, schedule)

    monkeypatch.setattr(HostedCheckExecutor, "_call_endpoint", stop_before_admission)
    tick = int(time.time()) - 2
    with TestClient(create_app()) as client:
        chat = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
        assert chat.status_code == 201, chat.text
        sid = chat.json()["session_id"]
        created = client.post(
            f"/garp/sessions/{sid}/recurring-checks",
            headers=alice,
            json={"name": "Survive shutdown", "cron_expr": "0 12 1 * *", "job_ref": "batch"},
        )
        assert created.status_code == 201, created.text
        key = created.json()["id"]
        with psycopg.connect(host_dsn) as db:
            db.execute("UPDATE ai.garp_host_schedules SET next_run_at=%s WHERE id=%s", (tick, key))
        assert first_claim.wait(10)

    with psycopg.connect(host_dsn) as db:
        assert db.execute(
            "SELECT next_run_at, locked_by FROM ai.garp_host_schedules WHERE id=%s", (key,)
        ).fetchone() == (tick, None)
        assert db.execute("SELECT count(*) FROM garp_schedules").fetchone() == (0,)

    with TestClient(create_app()) as client:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            history = client.get(f"/sessions/{sid}/runs", headers=alice)
            assert history.status_code == 200, history.text
            runs = history.json()
            if runs:
                break
            time.sleep(0.1)
        assert len(runs) == 1, history.text
        assert "Job check: complete" in runs[0]["content"]


def test_hosted_cron_retries_before_admission_and_expires_missed_tick(
    host_client, host_dsn, monkeypatch
):
    client, token = host_client
    alice = auth(
        token(ALICE, scopes=[*SCOPES, "checks:write", "schedules:write", "schedules:read"])
    )
    session = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
    assert session.status_code == 201, session.text
    sid = session.json()["session_id"]
    created = client.post(
        f"/garp/sessions/{sid}/recurring-checks",
        headers=alice,
        json={
            "name": "Expired tick",
            "cron_expr": "0 12 1 * *",
            "job_ref": "batch",
            "deadline_seconds": 10,
            "retry_delay_seconds": 1,
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["max_retries"] == 1
    key = created.json()["id"]
    admit = NativeChecks.admit
    attempts, observations = [], []

    def fail_once_before_insert(self, *args, **kwargs):
        attempts.append(kwargs["occurrence_id"])
        if len(attempts) == 1:
            raise psycopg.OperationalError("Synthetic private database error")
        return admit(self, *args, **kwargs)

    observe = LocalJobs.observe

    def count_observations(self, *args, **kwargs):
        observations.append(args)
        return observe(self, *args, **kwargs)

    monkeypatch.setattr(NativeChecks, "admit", fail_once_before_insert)
    monkeypatch.setattr(LocalJobs, "observe", count_observations)
    tick = int(time.time()) - 3600
    with psycopg.connect(host_dsn) as db:
        db.execute("UPDATE ai.garp_host_schedules SET next_run_at=%s WHERE id=%s", (tick, key))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        history = client.get(f"/schedules/{key}/runs", headers=alice)
        runs = client.get(f"/sessions/{sid}/runs", headers=alice)
        assert history.status_code == runs.status_code == 200
        rows = history.json()["data"]
        if runs.json() and any(row["status"] == "success" for row in rows):
            break
        time.sleep(0.1)
    assert sorted((row["status"], row["status_code"]) for row in rows) == [
        ("failed", 503),
        ("success", 201),
    ], history.text
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert "private database error" not in history.text
    assert len(runs.json()) == 1 and "Job check: expired" in runs.json()[0]["content"]
    assert observations == []


def test_hosted_cron_dispatch_fences_reclaimed_and_deleted_rows(host_client, host_dsn, tmp_path):
    client, token = host_client
    alice = auth(
        token(ALICE, scopes=[*SCOPES, "checks:write", "schedules:write", "schedules:delete"])
    )
    session = client.post("/sessions", headers=alice, json={"agent_id": "garp"})
    assert session.status_code == 201, session.text
    sid = session.json()["session_id"]
    created = client.post(
        f"/garp/sessions/{sid}/recurring-checks",
        headers=alice,
        json={"name": "Claim fencing", "cron_expr": "0 12 1 * *", "job_ref": "batch"},
    )
    assert created.status_code == 201, created.text
    key = created.json()["id"]
    db = PostgresDb(
        db_url=URL.create("postgresql+psycopg", query=conninfo_to_dict(host_dsn)),
        schedules_table="garp_host_schedules",
    )
    store = Store(host_dsn)
    executor = HostedCheckExecutor(
        NativeChecks(store, db, LocalJobs(tmp_path, tmp_path / "grants.json"))
    )

    # Hold the native poller away from this case. The snapshots represent successive claims.
    def claim(worker, stamp):
        with psycopg.connect(host_dsn, row_factory=dict_row) as sql_db:
            row = sql_db.execute(
                """UPDATE ai.garp_host_schedules SET locked_by=%s, locked_at=%s, next_run_at=%s
                   WHERE id=%s RETURNING *""",
                (worker, int(time.time()), stamp, key),
            ).fetchone()
        return Schedule.from_dict(row)

    tick = int(time.time()) + 3600
    original = claim("old-worker", tick)
    replacement = claim("new-worker", tick)
    assert client.portal.call(executor._call_endpoint, original)["status"] == "cancelled"
    accepted = client.portal.call(executor._call_endpoint, replacement)
    assert accepted["status"] == "success"
    first_check = accepted["output"]["check_id"]
    assert store.get_schedule(ALICE, first_check)["job_ref"] == "batch"
    assert client.portal.call(executor._call_endpoint, original) == accepted
    with psycopg.connect(host_dsn) as sql_db:
        assert sql_db.execute("SELECT count(*) FROM garp_schedules").fetchone() == (1,)

    later = claim("new-worker", tick + 60)
    conflict_id = str(uuid5(NAMESPACE_URL, f"garp:cron-check:{key}:{later.next_run_at}"))
    store.create_schedule(
        ALICE,
        sid,
        "different",
        {},
        datetime.fromtimestamp(later.next_run_at, UTC),
        native=True,
        occurrence_id=conflict_id,
    )
    collision = client.portal.call(executor._call_endpoint, later)
    assert collision["status"] == "failed" and collision["error"] == "Invalid hosted check schedule"
    with pytest.raises(ValueError, match="occurrence ID collision"):
        store.create_schedule(
            ALICE,
            sid,
            "batch",
            {},
            datetime.fromtimestamp(later.next_run_at, UTC),
            native=True,
            occurrence_id=conflict_id,
            cron_claim=((db.db_schema, db.schedules_table_name), later),
        )
    assert store.get_schedule(ALICE, conflict_id)["job_ref"] == "different"

    deleted = claim("new-worker", tick + 120)
    assert client.delete(f"/schedules/{key}", headers=alice).status_code == 204
    assert client.portal.call(executor._call_endpoint, deleted)["status"] == "cancelled"
    with psycopg.connect(host_dsn) as sql_db:
        assert sql_db.execute("SELECT count(*) FROM garp_schedules").fetchone() == (2,)


def test_host_can_sign_with_rsa_and_keep_private_key_out_of_garp(host_dsn, monkeypatch):
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verification_key = signing_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    config = HostAuth(ISSUER, AUDIENCE, verification_key.decode())
    for name in ("JWT_VERIFICATION_KEY", "JWT_JWKS_FILE"):
        with monkeypatch.context() as env:
            env.setenv(name, "unexpected-key-source")
            with pytest.raises(ValueError, match="ambient"):
                create_agentos_app(host_dsn, host_auth=config)
    app = create_agentos_app(host_dsn, host_auth=config)

    @asynccontextmanager
    async def lifespan(_host):
        async with app.router.lifespan_context(app):
            yield

    host = FastAPI(lifespan=lifespan)
    host.mount("/assistant", app)
    token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": ALICE,
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        signing_key,
        algorithm="RS256",
    )
    with TestClient(host) as client:
        assert client.get("/assistant/health").status_code == 200
        response = client.get("/assistant/garp/models", headers=auth(token))
        assert response.status_code == 200 and response.json()["durable_stream_reconnect"]
        assert client.get("/assistant/sessions", headers=auth(token)).status_code == 403
        assert (
            client.get(f"/assistant/garp/checks/{uuid4()}", headers=auth(token)).status_code == 404
        )


def test_host_does_not_return_provider_configuration_details(host_client, monkeypatch):
    client, token = host_client
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    response = client.post(
        "/agents/garp/runs", headers=auth(token()), data={"message": "hello", "stream": "false"}
    )
    assert response.status_code == 500
    assert "Model service is unavailable" in response.text
    assert "OPENROUTER_API_KEY" not in response.text
