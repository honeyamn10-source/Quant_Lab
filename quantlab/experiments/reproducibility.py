"""Reproducibility envelope.

Every result embeds git commit, Python version, dependency versions, dataset hash
and the exact config + seed. `reproduce()` verifies the environment matches before
re-running.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

from quantlab.data.fingerprint import dataset_fingerprint


def git_commit(path: str = ".") -> str:
    git = shutil.which("git")
    if git is None:
        return "not-a-repo"
    try:
        out = subprocess.run(  # noqa: S603
            [git, "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() or "not-a-repo"
    except Exception:  # noqa: BLE001
        return "not-a-repo"


def python_version() -> str:
    return platform.python_version()


def package_versions(*names: str) -> dict[str, str]:
    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


@dataclass
class EnvSnapshot:
    git_commit: str = field(default_factory=lambda: git_commit())
    python: str = field(default_factory=python_version)
    packages: dict = field(
        default_factory=lambda: package_versions("numpy", "pydantic", "PyYAML", "quantlab")
    )
    platform_: str = field(default_factory=lambda: platform.platform())
    dataset_hash: str = ""
    seed: int = 0
    timestamp_iso: str = field(
        default_factory=lambda: __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z")
    )

    def to_dict(self) -> dict:
        return {
            "git_commit": self.git_commit,
            "python": self.python,
            "packages": self.packages,
            "platform": self.platform_,
            "dataset_hash": self.dataset_hash,
            "seed": self.seed,
            "timestamp_iso": self.timestamp_iso,
        }


def reproduce(
    experiment_record: dict,
    run_fn,
    ledger,
    dataset_fingerprints: dict[str, str] | None = None,
) -> dict:
    """Verify pinned inputs still match the original run, then re-execute.

    `run_fn(experiment_record)` must be the pure, seed-pinned runner. When
    ``dataset_fingerprints`` is supplied, the exact consumed data windows are
    compared with the fingerprints stored in the original result record.
    """
    run_config = experiment_record.get("run_config") or {}
    commit_now = git_commit()
    recorded_commit = run_config.get("_git_commit")
    warnings: list[str] = []
    if recorded_commit and recorded_commit != commit_now and commit_now != "not-a-repo":
        warnings.append(f"git commit changed: recorded={recorded_commit}, now={commit_now}")
    if "seed" not in run_config:
        warnings.append(
            "original run did not record a seed; rerun uses the recorded run_config seed inheritance"
        )
    seed = run_config.get("seed")
    if seed is not None:
        warnings.append(f"re-running pinned seed={seed}")

    recorded_data = (experiment_record.get("results") or {}).get("dataset_fingerprints") or {}
    dataset_hash_now = ""
    if dataset_fingerprints:
        dataset_hash_now = fingerprint_text(
            json.dumps(dataset_fingerprints, sort_keys=True, separators=(",", ":"))
        )
        if recorded_data:
            if set(recorded_data) != set(dataset_fingerprints):
                warnings.append("dataset symbols changed since the original run")
            else:
                mismatches = [
                    symbol
                    for symbol, digest in dataset_fingerprints.items()
                    if recorded_data.get(symbol) != digest
                ]
                if mismatches:
                    warnings.append(
                        "dataset fingerprint changed for: " + ", ".join(sorted(mismatches))
                    )
        else:
            warnings.append("original run did not record dataset fingerprints")

    result = run_fn(experiment_record)
    return {
        "experiment_id": experiment_record["experiment_id"],
        "git_commit_recorded": recorded_commit,
        "git_commit_now": commit_now,
        "dataset_hash_now": dataset_hash_now,
        "environment_matches": not warnings,
        "warnings": warnings,
        "result": result,
    }


def fingerprint_text(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = [
    "EnvSnapshot",
    "dataset_fingerprint",
    "git_commit",
    "package_versions",
    "python_version",
    "reproduce",
]
