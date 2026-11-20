"""Start an isolated local Garp workspace for an interactive trial."""

import json
import logging
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
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

import uvicorn
from psycopg import OperationalError
from psycopg.conninfo import make_conninfo

from garp.api import create_app
from garp.cli import _request
from garp.executor import openrouter_config
from garp.jobs import LocalJobs
from garp.models import endpoint_for
from garp.store import Store
from garp.terminal import chat
from garp.worker import run_once

logger = logging.getLogger(__name__)


@contextmanager
def _postgres():
    local_initdb = shutil.which("initdb")
    default = Path(local_initdb).parent if local_initdb else Path("/usr/lib/postgresql/16/bin")
    binaries = Path(os.environ.get("GARP_POSTGRES_BIN", str(default)))
    if not (binaries / "pg_ctl").is_file() or not (binaries / "initdb").is_file():
        raise ValueError("Postgres tools unavailable. Set GARP_POSTGRES_BIN to the bin directory.")
    if os.geteuid() == 0:
        raise ValueError("Run the demo as your regular user so Postgres can start.")
    directory = Path(mkdtemp(prefix="garp-demo-"))
    data = directory / "pg"
    ctl = [binaries / "pg_ctl", "-D", data, "-w", "-t", "10"]
    env = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C"}
    start_attempted = False
    try:
        try:
            subprocess.run(
                [
                    binaries / "initdb",
                    "-D",
                    data,
                    "-U",
                    "garp_demo",
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
            start_attempted = True
            subprocess.run(
                [
                    *ctl,
                    "-l",
                    data / "server.log",
                    "-o",
                    f"-h '' -k {shlex.quote(str(directory))}",
                    "start",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                env=env,
            )
        except (OSError, subprocess.CalledProcessError):
            raise ValueError("Could not start disposable Postgres.") from None
        yield directory, make_conninfo(user="garp_demo", dbname="postgres", host=str(directory))
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
            print(
                f"Could not stop disposable Postgres. Its files are in {directory}.",
                file=sys.stderr,
            )
        else:
            shutil.rmtree(directory)


async def _scripted_reply(_owner, _session_id, _run_id, _history, _text):
    return "This is a scripted reply. Try /check demo 5, then /history to see the worker's result."


async def _scripted_title(_opening):
    return "Local Harness Demo"


def demo(*, live=False, model=None):
    if live:
        openrouter_config(model)
    elif model:
        endpoint_for(model)
    with _postgres() as (root, dsn):
        policy = root / "grants.json"
        policy.write_text(json.dumps({"demo": ["demo"]}))
        jobs = LocalJobs(root, policy)
        store = Store(dsn)
        token = secrets.token_urlsafe()
        options = {} if live else {"executor": _scripted_reply, "titler": _scripted_title}
        app = create_app(store, jobs, {token: "demo"}, **options)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
            server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
            api_thread = threading.Thread(
                target=server.run, kwargs={"sockets": [listener]}, daemon=True
            )
            stop = threading.Event()

            def run_fixture():
                counter = 0
                while not stop.is_set():
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
                    try:
                        run_once(store, jobs)
                    except OperationalError:
                        logger.warning("Demo worker will retry the database connection")
                    counter += 1
                    stop.wait(1)

            api_thread.start()
            worker = threading.Thread(target=run_fixture, daemon=True)
            try:
                for _ in range(500):
                    if server.started or not api_thread.is_alive():
                        break
                    time.sleep(0.02)
                if not server.started:
                    raise ValueError("The local Garp API did not start.")
                worker.start()
                for _ in range(100):
                    if (root / "demo.json").exists():
                        break
                    time.sleep(0.01)
                if not (root / "demo.json").exists():
                    raise ValueError("The local demo job did not start.")
                mode = (
                    "Live OpenRouter · temporary workspace"
                    if live
                    else "Local demo · scripted replies"
                )
                return chat(
                    origin, None, partial(_request, token=token), demo_mode=mode, model=model
                )
            finally:
                stop.set()
                if worker.ident is not None:
                    worker.join(timeout=6)
                server.should_exit = True
                api_thread.join(timeout=6)
