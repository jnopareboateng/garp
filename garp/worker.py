"""Run due one-shot job checks independently of the API and CLI."""

import argparse
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from psycopg import OperationalError

from garp.jobs import JobAccessError, JobEvidenceError, LocalJobs, compare_observations
from garp.store import Store

logger = logging.getLogger(__name__)


def run_once(store: Store, jobs: LocalJobs, clock=lambda: datetime.now(UTC)) -> bool:
    claim = store.claim_due()
    if claim is None:
        return False

    dispatch_at = claim["dispatch_at"]
    evidence = {
        "job_ref": claim["job_ref"],
        "due_at": claim["due_at"].isoformat(),
        "dispatch_at": dispatch_at.isoformat(),
    }
    if claim["deadline_at"] and claim["deadline_at"] < dispatch_at:
        outcome = "expired"
    elif claim["attempt"] > 3:
        outcome = "failed"
        evidence["state"] = "unknown"
        evidence["freshness"] = "attempts_exhausted"
    else:
        try:
            current = jobs.observe(claim["owner"], claim["job_ref"], clock())
            jobs.authorize(claim["owner"], claim["job_ref"])
        except JobAccessError:
            outcome = "denied"
        except JobEvidenceError:
            outcome = "failed"
            evidence["state"] = "unknown"
            evidence["freshness"] = "unavailable"
        else:
            outcome = "failed" if current.get("freshness") == "missing" else "complete"
            evidence.update(compare_observations(claim["baseline"], current, clock()))

    if store.finish_schedule(claim["id"], claim["lease_token"], outcome, evidence):
        logger.info("schedule=%s attempt=%s outcome=%s", claim["id"], claim["attempt"], outcome)
    return True


def local_jobs() -> LocalJobs:
    return LocalJobs(Path(os.environ["GARP_JOBS_DIR"]), Path(os.environ["GARP_POLICY_FILE"]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Process at most one due occurrence")
    args = parser.parse_args()
    store = Store(os.environ["GARP_DATABASE_URL"])
    jobs = local_jobs()
    if args.once:
        store.initialize()
        run_once(store, jobs)
        return
    initialized = False
    while True:
        try:
            if not initialized:
                store.initialize()
                initialized = True
            if run_once(store, jobs):
                continue
        except OperationalError:
            logger.warning("job store temporarily unavailable")
        time.sleep(1)
