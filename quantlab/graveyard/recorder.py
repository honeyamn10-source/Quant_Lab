"""Graveyard recorder — persistent, append-only ledger of failed strategies.

Every buriable strategy attempt becomes one immutable JSON line. Entries are
never edited or deleted; the surviving record itself documents how much was
tried and falsified.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

DEFAULT_GRAVEYARD_DIR = Path("experiments") / "graveyard"


class Graveyard:
    """Append-only JSON Lines ledger for failed strategy entries."""

    def __init__(self, base_dir: str | Path = DEFAULT_GRAVEYARD_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "entries.jsonl"
        self._lock = threading.Lock()
        self._sequence = self._load_max_sequence()

    def _load_max_sequence(self) -> int:
        seq = 0
        if not self.path.exists():
            return seq
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identifier = record.get("id", "")
                if identifier.startswith("STRATEGY-"):
                    seq = max(seq, int(identifier.split("-")[1]))
            except (json.JSONDecodeError, ValueError):
                continue
        return seq

    def _next_id(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"STRATEGY-{self._sequence:05d}"

    def _append(self, record: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def record_entry(self, entry: dict) -> dict:
        """Persist an entry, assigning a monotonic id and creation timestamp."""
        record = dict(entry)
        record["id"] = self._next_id()
        record["created_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._append(record)
        return dict(record)

    def list_entries(self) -> list[dict]:
        """Return every recorded entry in insertion order."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def get_entry(self, identifier: str) -> dict:
        """Return the entry with the given id, raising KeyError if unknown."""
        for record in self.list_entries():
            if record["id"] == identifier:
                return dict(record)
        raise KeyError(f"unknown graveyard entry {identifier}")

    def search(
        self,
        status: str | None = None,
        category: str | None = None,
        symbols: list[str] | None = None,
    ) -> list[dict]:
        """Filter entries by status, failure category, and/or traded symbols."""
        wanted_symbols = set(symbols) if symbols else None
        out = []
        for record in self.list_entries():
            if status is not None and record.get("status") != status:
                continue
            if category is not None and not self._has_category(record, category):
                continue
            if wanted_symbols is not None and not wanted_symbols <= set(record.get("symbols", [])):
                continue
            out.append(record)
        return out

    @staticmethod
    def _has_category(record: dict, category: str) -> bool:
        raw = record.get("failure_categories", record.get("failure_category", []))
        if isinstance(raw, str):
            raw = [raw]
        return any(str(category) == str(candidate) for candidate in raw)

    def count(self) -> int:
        """Number of recorded entries (including inconclusive ones)."""
        return len(self.list_entries())
