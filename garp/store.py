from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from garp.models import default_model, validate_options


class NotFound(LookupError):
    pass


class Store:
    def __init__(self, dsn):
        self.dsn = dsn

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5)

    def initialize(self):
        with self._connect() as conn:
            if conn.info.encoding != "utf-8":
                raise ValueError("Garp requires a UTF-8 Postgres database")
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('garp.schema'))")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS garp_sessions (
                    id uuid PRIMARY KEY, owner text NOT NULL,
                    turn_token uuid, turn_expires timestamptz, turn_text text, title text,
                    created_at timestamptz NOT NULL DEFAULT now()
                );
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS title text;
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS model text;
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS effort text;
                ALTER TABLE garp_sessions ADD COLUMN IF NOT EXISTS web_search boolean NOT NULL DEFAULT false;
                CREATE TABLE IF NOT EXISTS garp_events (
                    session_id uuid NOT NULL REFERENCES garp_sessions(id),
                    seq bigint NOT NULL, kind text NOT NULL, data jsonb NOT NULL,
                    PRIMARY KEY (session_id, seq)
                );
                CREATE TABLE IF NOT EXISTS garp_schedules (
                    id uuid PRIMARY KEY, session_id uuid NOT NULL REFERENCES garp_sessions(id),
                    job_ref text NOT NULL, baseline jsonb NOT NULL, due_at timestamptz NOT NULL,
                    deadline_at timestamptz, status text NOT NULL DEFAULT 'pending',
                    lease_token uuid, lease_expires timestamptz, attempt integer NOT NULL DEFAULT 0,
                    outcome text, evidence jsonb
                );
                CREATE INDEX IF NOT EXISTS garp_schedules_due ON garp_schedules(due_at)
                    WHERE status IN ('pending', 'leased');
            """)

    @staticmethod
    def _session(conn, owner, session_id, *, lock=False):
        suffix = " FOR UPDATE" if lock else ""
        row = conn.execute(
            "SELECT * FROM garp_sessions WHERE id = %s AND owner = %s" + suffix,
            (str(session_id), owner),
        ).fetchone()
        if row is None:
            raise NotFound("session not found")
        return row

    def create_session(self, owner):
        session_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO garp_sessions (id, owner) VALUES (%s, %s)", (session_id, owner)
            )
        return session_id

    def get_events(self, owner, session_id, after=0):
        with self._connect() as conn:
            self._session(conn, owner, session_id)
            return conn.execute(
                "SELECT seq, kind, data FROM garp_events WHERE session_id = %s AND seq > %s ORDER BY seq LIMIT 200",
                (str(session_id), after),
            ).fetchall()

    def get_session(self, owner, session_id):
        with self._connect() as conn:
            row = self._session(conn, owner, session_id)
            return {
                "id": str(row["id"]),
                "title": row["title"],
                "model": row["model"],
                "effort": row["effort"],
                "web_search": row["web_search"],
            }

    def list_sessions(self, owner):
        with self._connect() as conn:
            return conn.execute(
                """SELECT p.id::text AS id, p.title, p.model, p.effort, p.web_search
                FROM garp_sessions p WHERE p.owner = %s
                ORDER BY p.created_at DESC, p.id LIMIT 200""",
                (owner,),
            ).fetchall()

    def set_session_options(self, owner, session_id, updates):
        with self._connect() as conn:
            row = self._session(conn, owner, session_id, lock=True)
            prior_model = row["model"] or default_model()
            model = updates.get("model", prior_model)
            same_model = model == prior_model
            effort = updates.get("effort", row["effort"] if same_model else None)
            search = updates.get("web_search", row["web_search"] if same_model else False)
            validate_options(model, effort, search)
            conn.execute(
                "UPDATE garp_sessions SET model = %s, effort = %s, web_search = %s WHERE id = %s",
                (model, effort, search, session_id),
            )
            return {"model": model, "effort": effort, "web_search": search}

    def check_session(self, owner, session_id):
        with self._connect() as conn:
            self._session(conn, owner, session_id)
            return conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]

    @staticmethod
    def _event(conn, session_id, kind, data):
        return conn.execute(
            """INSERT INTO garp_events (session_id, seq, kind, data)
            SELECT %s, COALESCE(MAX(seq), 0) + 1, %s, %s
            FROM garp_events WHERE session_id = %s RETURNING seq, kind, data""",
            (session_id, kind, Jsonb(data), session_id),
        ).fetchone()

    def create_schedule(self, owner, session_id, job_ref, baseline, due_at, deadline_at=None):
        schedule_id = str(uuid4())
        with self._connect() as conn:
            self._session(conn, owner, session_id, lock=True)
            conn.execute(
                """INSERT INTO garp_schedules (id, session_id, job_ref, baseline, due_at, deadline_at)
                            VALUES (%s, %s, %s, %s, %s, %s)""",
                (schedule_id, session_id, job_ref, Jsonb(baseline), due_at, deadline_at),
            )
        return schedule_id

    def get_schedule(self, owner, schedule_id):
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.*, p.owner, clock_timestamp() AS now
                FROM garp_schedules s JOIN garp_sessions p ON p.id = s.session_id
                WHERE s.id = %s AND p.owner = %s""",
                (str(schedule_id), owner),
            ).fetchone()
            if row is None:
                raise NotFound("schedule not found")
            return self._ids(row)

    @staticmethod
    def _ids(row):
        for key in ("id", "session_id", "lease_token"):
            if row.get(key) is not None:
                row[key] = str(row[key])
        return row

    def claim_due(self, lease_seconds=30):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._connect() as conn:
            row = conn.execute(
                """SELECT p.id AS session_id, p.owner, s.id FROM garp_sessions p
                JOIN garp_schedules s ON s.session_id = p.id
                WHERE s.due_at <= clock_timestamp() AND (s.status = 'pending' OR
                      (s.status = 'leased' AND s.lease_expires <= clock_timestamp()))
                  AND (p.turn_token IS NULL OR p.turn_expires <= clock_timestamp())
                  AND NOT EXISTS (SELECT 1 FROM garp_schedules active WHERE active.session_id = p.id
                      AND active.status = 'leased' AND active.lease_expires > clock_timestamp())
                ORDER BY s.due_at, s.id FOR UPDATE OF p SKIP LOCKED LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            # A competing transaction may commit between the candidate snapshot and row lock.
            valid = conn.execute(
                """SELECT 1 FROM garp_schedules s JOIN garp_sessions p ON p.id = s.session_id
                WHERE s.id = %s AND s.due_at <= clock_timestamp() AND (s.status = 'pending' OR
                    (s.status = 'leased' AND s.lease_expires <= clock_timestamp()))
                AND (p.turn_token IS NULL OR p.turn_expires <= clock_timestamp())
                AND NOT EXISTS (SELECT 1 FROM garp_schedules active WHERE active.session_id = p.id
                    AND active.status = 'leased' AND active.lease_expires > clock_timestamp())""",
                (row["id"],),
            ).fetchone()
            if valid is None:
                return None
            token = str(uuid4())
            result = conn.execute(
                """UPDATE garp_schedules SET status = 'leased', lease_token = %s,
                lease_expires = clock_timestamp() + %s * interval '1 second', attempt = attempt + 1
                WHERE id = %s RETURNING *, clock_timestamp() AS dispatch_at""",
                (token, lease_seconds, row["id"]),
            ).fetchone()
            result["owner"] = row["owner"]
            return self._ids(result)

    def finish_schedule(self, schedule_id, lease_token, outcome, evidence):
        if outcome not in ("complete", "denied", "expired", "failed"):
            raise ValueError("invalid outcome")
        with self._connect() as conn:
            # Lock the session first, matching both admission paths.
            session = conn.execute(
                """SELECT p.id FROM garp_sessions p JOIN garp_schedules s ON s.session_id = p.id
                WHERE s.id = %s FOR UPDATE OF p""",
                (str(schedule_id),),
            ).fetchone()
            if session is None:
                raise NotFound("schedule not found")
            row = conn.execute(
                """UPDATE garp_schedules SET status = %s, outcome = %s, evidence = %s,
                lease_token = NULL, lease_expires = NULL WHERE id = %s AND status = 'leased'
                AND lease_token = %s AND lease_expires > clock_timestamp()
                RETURNING session_id""",
                (outcome, outcome, Jsonb(evidence), str(schedule_id), str(lease_token)),
            ).fetchone()
            if row is None:
                return False
            self._event(
                conn,
                row["session_id"],
                "schedule.result",
                {"schedule_id": str(schedule_id), "outcome": outcome, "evidence": evidence},
            )
            return True

    def start_turn(self, owner, session_id, text, lease_seconds=120):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._connect() as conn:
            session = self._session(conn, owner, session_id, lock=True)
            busy = conn.execute(
                """SELECT 1 FROM garp_sessions p WHERE p.id = %s AND
                ((p.turn_token IS NOT NULL AND p.turn_expires > clock_timestamp()) OR
                 EXISTS (SELECT 1 FROM garp_schedules s WHERE s.session_id = p.id AND s.status = 'leased'
                         AND s.lease_expires > clock_timestamp()))""",
                (str(session_id),),
            ).fetchone()
            if busy:
                raise RuntimeError("session busy")
            token = str(uuid4())
            conn.execute(
                """UPDATE garp_sessions SET turn_token = %s, turn_text = %s,
                turn_expires = clock_timestamp() + %s * interval '1 second' WHERE id = %s""",
                (token, text, lease_seconds, str(session_id)),
            )
            history = conn.execute(
                """SELECT CASE WHEN kind = 'turn.message' THEN data->>'role'
                ELSE 'assistant' END AS role, CASE WHEN kind = 'turn.message' THEN data->>'content'
                ELSE 'Scheduled check result: ' || data::text END AS content
                FROM garp_events WHERE session_id = %s
                AND (kind = 'turn.message' OR
                    (kind = 'schedule.result' AND data->>'outcome' = 'complete'))
                ORDER BY seq""",
                (str(session_id),),
            ).fetchall()
            return {
                "token": token,
                "history": history,
                "title": session["title"],
                "model": session["model"],
                "effort": session["effort"],
                "web_search": session["web_search"],
            }

    def finish_turn(self, owner, session_id, token, text, title=None, details=None):
        with self._connect() as conn:
            session = self._session(conn, owner, session_id, lock=True)
            row = conn.execute(
                """UPDATE garp_sessions SET turn_token = NULL, turn_expires = NULL, turn_text = NULL,
                title = COALESCE(title, %s)
                WHERE id = %s AND turn_token = %s AND turn_expires > clock_timestamp()
                RETURNING id""",
                (title if text is not None else None, str(session_id), str(token)),
            ).fetchone()
            if row is None:
                return False
            if text is None:
                return True
            return [
                self._event(
                    conn,
                    session_id,
                    "turn.message",
                    {"role": "user", "content": session["turn_text"]},
                ),
                self._event(
                    conn,
                    session_id,
                    "turn.message",
                    {"role": "assistant", "content": text, **(details or {})},
                ),
            ]
