import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture(scope="session")
def postgres_dsn():
    candidates = [Path("/usr/lib/postgresql/16/bin")]
    if initdb := shutil.which("initdb"):
        candidates.insert(0, Path(initdb).parent)
    binaries = next((path for path in candidates if (path / "pg_ctl").is_file()), None)
    if binaries is None or os.geteuid() == 0:
        pytest.skip("local Postgres binaries and non-root execution are required")

    with TemporaryDirectory(prefix="garp-pg-") as folder:
        data = Path(folder) / "data"
        subprocess.run(
            [
                binaries / "initdb",
                "-D",
                data,
                "-U",
                "garp_test",
                "-A",
                "trust",
                "--no-locale",
                "-E",
                "UTF8",
            ],
            check=True,
            capture_output=True,
        )
        pg_ctl = [binaries / "pg_ctl", "-D", data, "-w"]
        subprocess.run(
            [*pg_ctl, "-l", data / "pg.log", "-o", f"-h '' -k {folder}", "start"],
            check=True,
            capture_output=True,
        )
        try:
            yield f"user=garp_test dbname=postgres host={folder}"
        finally:
            subprocess.run([*pg_ctl, "-m", "immediate", "stop"], check=True, capture_output=True)
