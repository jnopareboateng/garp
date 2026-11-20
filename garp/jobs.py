"""Read evidence from a trusted local job adapter."""

import json
import os
import re
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

_JOB_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_STATES = {"running", "finished", "failed", "lost"}


class JobAccessError(PermissionError):
    pass


class JobEvidenceError(Exception):
    pass


class LocalJobs:
    def __init__(self, directory: Path, policy: Path):
        self.directory = directory
        self.policy = policy

    def authorize(self, owner: str, job_ref: str) -> None:
        if not _JOB_REF.fullmatch(job_ref):
            raise ValueError("invalid job reference")
        try:
            policy = json.loads(self.policy.read_bytes())
            grants = policy[owner]
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            raise JobAccessError("job access unavailable or denied") from exc
        if not isinstance(grants, list) or job_ref not in grants:
            raise JobAccessError("job access denied")

    def observe(self, owner: str, job_ref: str, now: datetime) -> dict:
        self.authorize(owner, job_ref)
        path = self.directory / f"{job_ref}.json"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as file:
                if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                    raise JobEvidenceError("job evidence is not a regular file")
                raw = file.read(16_385)
            if len(raw) > 16_384:
                raise JobEvidenceError("job evidence is too large")
            value = json.loads(raw)
            observed = datetime.fromisoformat(value["observed_at"])
            state = value["state"]
            instance = value["instance"]
            progress = value.get("progress")
            if (
                observed.tzinfo is None
                or observed > now + timedelta(seconds=5)
                or state not in _STATES
                or not isinstance(instance, str)
                or not _JOB_REF.fullmatch(instance)
                or (progress is not None and (type(progress) is not int or progress < 0))
            ):
                raise ValueError("invalid observation")
        except FileNotFoundError:
            return {"state": "unknown", "freshness": "missing"}
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            raise JobEvidenceError("invalid or unavailable job evidence") from exc
        return {
            "state": state,
            "observed_at": observed.astimezone(UTC).isoformat(),
            "instance": instance,
            "counter": progress,
        }


def compare_observations(baseline: dict, latest: dict, now: datetime) -> dict:
    result = dict(latest)
    result["progress"] = "unmeasurable"
    last_seen = latest.get("observed_at", baseline.get("observed_at"))
    if last_seen:
        result["last_seen_at"] = last_seen
    if latest["state"] == "running":
        if now - datetime.fromisoformat(latest["observed_at"]) > timedelta(minutes=2):
            result.update(state="unknown", freshness="stale")
            return result
        result["freshness"] = "fresh"
    if (
        result.get("observed_at")
        and baseline.get("observed_at")
        and baseline.get("instance") == result.get("instance")
        and datetime.fromisoformat(result["observed_at"])
        > datetime.fromisoformat(baseline["observed_at"])
        and baseline.get("counter") is not None
        and result.get("counter") is not None
    ):
        if result["counter"] > baseline["counter"]:
            result["progress"] = "advanced"
        elif result["counter"] == baseline["counter"] and result["state"] == "running":
            result["progress"] = "unchanged"
    if baseline.get("observed_at"):
        result["baseline_observed_at"] = baseline["observed_at"]
    return result
