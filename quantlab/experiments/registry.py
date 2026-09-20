"""Experiment registry — the permanent ledger.

Every strategy attempt (successful or not) becomes an immutable, JSON-serializable
record. Failed trials are never deleted; the number of attempted configurations is
itself evidence when evaluating backtest overfitting.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from quantlab.experiments.config import RunConfig

DEFAULT_LEDGER_DIR = Path("experiments")


class ExperimentLedger:
    """Append-only JSON Lines ledger of experiments."""

    def __init__(self, base_dir: str | Path = DEFAULT_LEDGER_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "ledger.jsonl"
        self._lock = threading.Lock()
        self._sequence = self._load_max_sequence()

    def _load_max_sequence(self) -> int:
        seq = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    n = int(rec.get("number", 0))
                    seq = max(seq, n)
                except (json.JSONDecodeError, ValueError):
                    continue
        return seq

    def _next_id(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"EXP-{self._sequence:04d}"

    def create(
        self, run_name: str, strategy_name: str, run_config: RunConfig | dict | None = None
    ) -> dict:
        experiment_id = self._next_id()
        record = {
            "experiment_id": experiment_id,
            "number": int(experiment_id.split("-")[1]),
            "run_name": run_name,
            "strategy_name": strategy_name,
            "strategy_version": "0.1.0",
            "created_ns": int(time.time_ns()),
            "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "uid": uuid.uuid4().hex[:12],
            "run_config": (
                run_config.model_dump(mode="json")
                if isinstance(run_config, RunConfig)
                else (dict(run_config) if isinstance(run_config, dict) else {})
            ),
            "results": None,
            "failure_reason": None,
            "status": "created",
        }
        self._append(record)
        return record

    def _append(self, record: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=False) + "\n")

    def complete(
        self, experiment_id: str, results: dict, failure_reason: str | None = None
    ) -> dict:
        record = self.get(experiment_id)
        record["results"] = results
        record["failure_reason"] = failure_reason
        record["status"] = "rejected" if failure_reason else "completed"
        # rewrite: append-only with final state recorded as a new line is unsafe,
        # so we keep ONE line per experiment and patch the final state in place.
        self._update_record(experiment_id, record)
        return record

    def _update_record(self, experiment_id: str, record: dict) -> None:
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("experiment_id") == experiment_id:
                    lines[idx] = json.dumps(record)
                    break
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def get(self, experiment_id: str) -> dict:
        for rec in self.all():
            if rec["experiment_id"] == experiment_id:
                return dict(rec)
        raise KeyError(f"unknown experiment {experiment_id}")

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def count(self) -> int:
        return len(self.all())

    def search(self, **filters) -> list[dict]:
        out = []
        for rec in self.all():
            match = True
            for key, value in filters.items():
                if rec.get(key) != value:
                    match = False
                    break
            if match:
                out.append(rec)
        return out

    def export_json(self) -> list[dict]:
        return self.all()


_global_ledger = ExperimentLedger()


def default_ledger() -> ExperimentLedger:
    return _global_ledger


def scaffold_default() -> None:
    os.makedirs(DEFAULT_LEDGER_DIR, exist_ok=True)
    marker = DEFAULT_LEDGER_DIR / "README.md"
    if not marker.exists():
        marker.write_text(
            "# Experiment outputs\n\nThis directory holds reproducible experiment records. It is gitignored; datasets live under `datasets/`.\n",
            encoding="utf-8",
        )
