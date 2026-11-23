"""Start an isolated local Garp workspace for an interactive trial."""

import fcntl
import json
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import httpx
import uvicorn
from agno.models.openrouter import OpenRouter
from psycopg.conninfo import make_conninfo

from garp.executor import openrouter_config
from garp.jobs import LocalJobs
from garp.models import endpoint_for
from garp.native_client import NativeChatClient
from garp.terminal import native_chat


@contextmanager
def _postgres_at(data, socket_dir, *, recover=False):
    initdb = shutil.which("initdb")
    default = Path(initdb).parent if initdb else Path("/usr/lib/postgresql/16/bin")
    binaries = Path(os.environ.get("GARP_POSTGRES_BIN", str(default)))
    if not (binaries / "pg_ctl").is_file() or not (binaries / "initdb").is_file():
        raise ValueError("Postgres tools unavailable. Set GARP_POSTGRES_BIN to the bin directory.")
    if os.geteuid() == 0:
        raise ValueError("Run Garp as your regular user so Postgres can start.")

    ctl = [binaries / "pg_ctl", "-D", data, "-w", "-t", "10"]
    env = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C"}
    start_attempted = False
    try:
        if not data.exists():
            # Publish the cluster only after initdb succeeds.
            with TemporaryDirectory(prefix="garp-init-", dir=data.parent) as staging:
                fresh = Path(staging) / "pg"
                subprocess.run(
                    [
                        binaries / "initdb",
                        "-D",
                        fresh,
                        "-U",
                        "garp",
                        "-A",
                        "trust",
                        "--no-locale",
                        "-E",
                        "UTF8",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    env=env,
                )
                fresh.rename(data)
        # A killed CLI can leave its private Postgres process running.
        if recover and (data / "postmaster.pid").exists():
            status = subprocess.run([*ctl, "status"], check=False, capture_output=True, env=env)
            if status.returncode == 0:
                subprocess.run(
                    [*ctl, "-m", "fast", "stop"], check=True, stdout=subprocess.DEVNULL, env=env
                )
        start_attempted = True
        subprocess.run(
            [
                *ctl,
                "-l",
                data / "server.log",
                "-o",
                f"-h '' -k {shlex.quote(str(socket_dir))}",
                "start",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError):
        raise ValueError(f"Could not start Garp's Postgres. See {data / 'server.log'}.") from None
    else:
        dsn = make_conninfo(user="garp", dbname="postgres", host=str(socket_dir))
        yield dsn
    finally:
        if start_attempted:
            for mode in ("fast", "immediate"):
                try:
                    stopped = subprocess.run(
                        [*ctl, "-m", mode, "stop"], check=False, capture_output=True, env=env
                    )
                except OSError:
                    break
                if stopped.returncode == 0:
                    break
        if (data / "postmaster.pid").exists():
            print(f"Could not stop local Postgres. Its data is at {data}.", file=sys.stderr)
        else:
            shutil.rmtree(socket_dir)


@contextmanager
def _postgres():
    directory = Path(mkdtemp(prefix="garp-demo-"))
    try:
        with _postgres_at(directory / "pg", directory) as dsn:
            yield directory, dsn
    finally:
        if directory.exists() and not (directory / "pg/postmaster.pid").exists():
            shutil.rmtree(directory)


@contextmanager
def _serve_app(app):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=4)
        )
        api_thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        api_thread.start()
        try:
            # AgentOS runs database setup before it accepts connections.
            deadline = time.monotonic() + 45
            while not server.started and api_thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not server.started:
                raise ValueError("The local Garp API did not start.")
            yield origin
        finally:
            server.should_exit = True
            api_thread.join(timeout=6)


@contextmanager
def _local_database():
    root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "garp"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    with (root / "workspace.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("The local Garp workspace is already in use.") from None
        socket_dir = Path(mkdtemp(prefix="garp-socket-"))
        try:
            with _postgres_at(root / "pg", socket_dir, recover=True) as dsn:
                yield root, dsn
        finally:
            if socket_dir.exists() and not (root / "pg/postmaster.pid").exists():
                shutil.rmtree(socket_dir)


def configured_jobs():
    jobs_dir, policy = os.environ.get("GARP_JOBS_DIR"), os.environ.get("GARP_POLICY_FILE")
    if bool(jobs_dir) != bool(policy):
        raise ValueError("Set both GARP_JOBS_DIR and GARP_POLICY_FILE to enable job checks")
    return LocalJobs(Path(jobs_dir), Path(policy)) if jobs_dir and policy else None


@contextmanager
def _quiet_runtime():
    from agno.utils.log import agent_logger, team_logger, workflow_logger

    # Uvicorn resets disabled flags and Agno changes levels during runs.
    loggers = (agent_logger, team_logger, workflow_logger)

    def suppress(_record):
        return False

    for log in loggers:
        log.addFilter(suppress)
    try:
        yield
    finally:
        for log in loggers:
            log.removeFilter(suppress)


@contextmanager
def native_workspace(*, model=None, needs_model=True):
    from garp.agentos import create_agentos_app

    if needs_model:
        openrouter_config(model)
    token = secrets.token_urlsafe()
    with (
        _quiet_runtime(),
        _local_database() as (_, dsn),
        _serve_app(
            create_agentos_app(dsn, token, jobs=configured_jobs(), import_legacy=True)
        ) as origin,
    ):
        yield origin, token


def _scripted_transport():
    async def respond(request):
        payload = json.loads(request.content)
        model_id = payload["model"]
        message = "This is a scripted reply. Try /check demo 5, then /history to see the result."
        if payload.get("stream"):
            events = []
            for delta, reason in [({"role": "assistant", "content": message}, None), ({}, "stop")]:
                event = {
                    "id": "demo",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                }
                events.append(f"data: {json.dumps(event)}\n\n")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(events) + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "demo",
                "object": "chat.completion",
                "created": 1,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": message},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return httpx.MockTransport(respond)


def _scripted_model(model_id, client):
    return OpenRouter(id=model_id, api_key="demo", http_client=client)


def demo(*, live=False, model=None):
    if live:
        openrouter_config(model)
    elif model:
        endpoint_for(model)
    with _postgres() as (root, dsn):
        policy = root / "grants.json"
        policy.write_text(json.dumps({"local": ["demo"]}))
        jobs = LocalJobs(root, policy)
        counter = 0

        def update_demo_job():
            nonlocal counter
            pending = root / "demo.tmp"
            pending.write_text(
                json.dumps(
                    {
                        "instance": "demo-job",
                        "observed_at": datetime.now(UTC).isoformat(),
                        "state": "running",
                        "progress": counter,
                    }
                )
            )
            pending.replace(root / "demo.json")
            counter += 1

        from garp.agentos import create_agentos_app

        token = secrets.token_urlsafe()
        options = {} if live else {"_demo_model": _scripted_model}
        mode = "Live demo · chats expire on exit" if live else "Local demo · scripted replies"
        stop = threading.Event()

        def work():
            while not stop.wait(1):
                update_demo_job()

        update_demo_job()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            with (
                _quiet_runtime(),
                _serve_app(create_agentos_app(dsn, token, jobs=jobs, **options)) as base,
            ):
                client = NativeChatClient(base, token)
                try:
                    return native_chat(client, demo_mode=mode, model=model)
                finally:
                    client.close()
        finally:
            stop.set()
            worker.join(timeout=6)
