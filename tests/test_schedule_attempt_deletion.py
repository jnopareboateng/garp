import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import psycopg
import pytest
import test_scheduled_prompts as helpers
from fastapi.testclient import TestClient
from psycopg import errors

host, service = helpers.host, helpers.service


def test_attempt_update_waits_for_session_deletion(service, host):
    dsn, *_ = service
    create, headers, _ = host
    auth = headers()
    with TestClient(create()) as client:
        sid = client.post("/sessions", headers=auth, json={"agent_id": "garp"}).json()["session_id"]
        sched = helpers.schedule(client, auth, name="Race", cron_expr="0 12 1 * *", message="Test")
        with psycopg.connect(dsn) as conn:
            conn.execute(
                """INSERT INTO ai.garp_host_schedule_runs
                (id, schedule_id, attempt, user_id, status, created_at)
                VALUES ('race', %s, 1, %s, 'running', 1)""",
                (sched["id"], helpers.OWNER),
            )
        started = Event()

        def update():
            with psycopg.connect(dsn) as conn:
                started.set()
                conn.execute(
                    "UPDATE ai.garp_host_schedule_runs SET session_id=%s WHERE id='race'", (sid,)
                )

        with ThreadPoolExecutor() as pool, psycopg.connect(dsn) as deletion:
            deletion.execute("DELETE FROM ai.agno_sessions WHERE session_id=%s", (sid,))
            future = pool.submit(update)
            assert started.wait(5)
            assert not future.done()
            deletion.commit()
            with pytest.raises(errors.CheckViolation):
                future.result(timeout=5)


def test_occurrence_retention(service, host):
    dsn, *_ = service
    create, headers, _ = host
    auth = headers()
    with TestClient(create()) as client:
        sched = helpers.schedule(
            client,
            auth,
            name="Probe recurring",
            cron_expr="0 12 1 * *",
            message="Synthetic schedule secret",
        )
        rows = []
        for tick in (int(time.time()) - 20, int(time.time()) - 10):
            helpers.advance(dsn, sched["id"], tick)
            count = len(rows)
            result = helpers.eventually(
                lambda: client.get(f"/schedules/{sched['id']}/runs", headers=auth),
                lambda x, count=count: (
                    len([r for r in x["data"] if r["status"] == "success"]) > count
                ),
            )
            rows = [r for r in result["data"] if r["status"] == "success"]
        first, second = rows
        assert first["session_id"] != second["session_id"]
        for row in rows:
            helpers.eventually(
                lambda row=row: client.get(
                    f"/agents/garp/runs/{row['run_id']}",
                    headers=auth,
                    params={"session_id": row["session_id"]},
                ),
                lambda x: x["status"] == "COMPLETED",
            )
        assert client.delete(f"/sessions/{first['session_id']}", headers=auth).status_code == 204
        history = client.get(f"/schedules/{sched['id']}/runs", headers=auth).json()
        assert client.get(f"/schedules/{sched['id']}", headers=auth).status_code == 200
        assert client.get(f"/sessions/{second['session_id']}/runs", headers=auth).status_code == 200
        with psycopg.connect(dsn) as conn:
            db_rows = conn.execute(
                "SELECT row_to_json(r) FROM ai.garp_host_schedule_runs r WHERE schedule_id=%s ORDER BY created_at",
                (sched["id"],),
            ).fetchall()
        assert len(db_rows) == 1
        assert [r["session_id"] for r in history["data"]] == [second["session_id"]]
        with psycopg.connect(dsn) as conn:
            conn.execute(
                """INSERT INTO ai.garp_host_schedule_runs
                (id, schedule_id, attempt, user_id, status, created_at)
                VALUES ('late', %s, 1, %s, 'running', 1)""",
                (sched["id"], helpers.OWNER),
            )
        with pytest.raises(errors.CheckViolation), psycopg.connect(dsn) as conn:
            conn.execute(
                "UPDATE ai.garp_host_schedule_runs SET session_id=%s, run_id=%s WHERE id='late'",
                (first["session_id"], first["run_id"]),
            )
        with pytest.raises(errors.CheckViolation), psycopg.connect(dsn) as conn:
            conn.execute(
                """INSERT INTO ai.garp_host_schedule_runs
                (id, schedule_id, attempt, user_id, status, created_at, session_id)
                VALUES ('later', %s, 1, %s, 'running', 1, %s)""",
                (sched["id"], helpers.OWNER, first["session_id"]),
            )
