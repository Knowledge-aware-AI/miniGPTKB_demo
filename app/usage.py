"""Per-run counts of LLM calls and tokens, by role, kept in the run's
usage.json (model calls per triple, retries, rate-limit hits).

Counts accumulate across resumes: an existing usage.json is loaded first.
The file is rewritten at most every few seconds while the run is going,
and on flush()."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

FIELDS = ("calls", "prompt_tokens", "completion_tokens", "retries", "rate_limited")


class UsageCounter:
    def __init__(self, path: Path, min_interval_seconds: float = 5.0) -> None:
        self.path = path
        self.min_interval = min_interval_seconds
        self.lock = threading.Lock()
        self.roles: dict[str, dict[str, int]] = {}
        self._last_write = 0.0
        try:
            stored = json.loads(path.read_text(encoding="utf-8")).get("roles", {})
            for role, counts in stored.items():
                self.roles[role] = {field: int(counts.get(field, 0)) for field in FIELDS}
        except (OSError, ValueError, AttributeError):
            pass

    def _role(self, role: str) -> dict[str, int]:
        return self.roles.setdefault(role, {field: 0 for field in FIELDS})

    def record_call(self, role: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        with self.lock:
            counts = self._role(role)
            counts["calls"] += 1
            counts["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            counts["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self._maybe_write()

    def record_retry(self, role: str, rate_limited: bool) -> None:
        with self.lock:
            counts = self._role(role)
            counts["retries"] += 1
            counts["rate_limited"] += int(rate_limited)
        self._maybe_write()

    def _maybe_write(self) -> None:
        if time.monotonic() - self._last_write >= self.min_interval:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            payload = {"roles": {role: dict(counts) for role, counts in self.roles.items()}}
            self._last_write = time.monotonic()
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass
