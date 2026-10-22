"""Client-side pacing to stay under an endpoint's per-model rate limits.

The ScaDS.AI endpoint (LiteLLM) limits each model to N requests and T tokens
per minute per key, and answers anything over it with HTTP 429 and a 60 s
retry-after -- which stalls a run for a minute at a time, typically during
NED, whose calls are long. Its responses report both limits in
x-ratelimit-*-limit-requests / -limit-tokens headers, so the throttle learns
them from the first response for each model and from then on delays requests
that would push the model's last 60 seconds past a safety fraction of either
limit. The token limit counts output tokens only (measured on ScaDS: a call is
charged its completion tokens plus one, its prompt is free), so that is what
is counted here. A request's output size is only known after it returns, so
each one reserves an estimate (a running average per model and task) that is
corrected once the response arrives.

Only this process's own usage is counted: other users of the same key can
still trigger a 429, which the caller's retry handles as before. Endpoints that
report no limits are not throttled at all.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

WINDOW_SECONDS = 60.0
SAFETY_FRACTION = 0.9
DEFAULT_COMPLETION_TOKENS = 256


@dataclass
class _Usage:
    started: float
    tokens: float


@dataclass
class _ModelState:
    limit_requests: int | None = None
    limit_tokens: int | None = None
    window: deque[_Usage] = field(default_factory=deque)
    last_paced_log: float = float("-inf")


def limits_from_headers(headers: Mapping[str, str]) -> tuple[int | None, int | None]:
    def read(suffix: str) -> int | None:
        for key, value in headers.items():
            if key.lower().startswith("x-ratelimit") and key.lower().endswith(suffix):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return None
        return None

    return read("limit-requests"), read("limit-tokens")


class ModelThrottle:
    def __init__(
        self,
        log: Callable[[str, str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        enabled: bool = True,
    ) -> None:
        self._log = log or (lambda message, level: None)
        # Disabled: never waits (limits are still learned), for comparing
        # runs with and without pacing.
        self.enabled = enabled
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._models: dict[str, _ModelState] = {}
        # Running mean of actual tokens per (model, task), for estimates.
        self._mean_tokens: dict[tuple[str, str], tuple[float, int]] = {}

    def estimate(self, model: str, task: str) -> float:
        """Expected charged (output) tokens of the next request of this kind."""
        with self._lock:
            mean = self._mean_tokens.get((model, task))
        return mean[0] if mean else DEFAULT_COMPLETION_TOKENS

    def acquire(self, model: str, tokens: float, should_stop: Callable[[], bool], label: str = "") -> _Usage | None:
        """Blocks until `model` has room for one more request of ~`tokens`
        tokens, then reserves it. Returns None if stopped while waiting."""
        while True:
            if should_stop():
                return None
            if not self.enabled:
                return _Usage(self._clock(), tokens)
            with self._lock:
                state = self._models.setdefault(model, _ModelState())
                now = self._clock()
                while state.window and now - state.window[0].started >= WINDOW_SECONDS:
                    state.window.popleft()
                wait = self._wait_needed(state, tokens, now)
                if wait <= 0:
                    usage = _Usage(now, tokens)
                    state.window.append(usage)
                    return usage
                if now - state.last_paced_log >= WINDOW_SECONDS:
                    state.last_paced_log = now
                    limits = ", ".join(
                        part
                        for part in (
                            f"{state.limit_requests} requests" if state.limit_requests else "",
                            f"{state.limit_tokens} tokens" if state.limit_tokens else "",
                        )
                        if part
                    )
                    self._log(
                        f"Pacing {label + ' ' if label else ''}requests to {model} to stay under its limit "
                        f"of {limits} per minute.",
                        "info",
                    )
            self._sleep(min(wait, 1.0))

    def _wait_needed(self, state: _ModelState, tokens: float, now: float) -> float:
        if not state.window:
            return 0.0  # always allow one request, even an oversized one
        waits = [0.0]
        if state.limit_requests:
            allowed = max(1, int(state.limit_requests * SAFETY_FRACTION))
            if len(state.window) + 1 > allowed:
                # Wait until enough of the oldest requests leave the window.
                oldest = state.window[len(state.window) - allowed]
                waits.append(oldest.started + WINDOW_SECONDS - now)
        if state.limit_tokens:
            budget = state.limit_tokens * SAFETY_FRACTION
            used = sum(usage.tokens for usage in state.window)
            if used + tokens > budget:
                freed = 0.0
                for usage in state.window:
                    freed += usage.tokens
                    if used - freed + tokens <= budget:
                        waits.append(usage.started + WINDOW_SECONDS - now)
                        break
                else:
                    waits.append(state.window[-1].started + WINDOW_SECONDS - now)
        return max(waits)

    def record(self, model: str, task: str, usage: _Usage | None, actual_tokens: int | None,
               headers: Mapping[str, Any] | None = None) -> None:
        """Corrects a reservation with the tokens the request really used, and
        learns the model's limits from the response headers."""
        with self._lock:
            state = self._models.setdefault(model, _ModelState())
            if headers is not None:
                requests, tokens = limits_from_headers(headers)
                if requests:
                    state.limit_requests = requests
                if tokens:
                    state.limit_tokens = tokens
            if usage is not None and actual_tokens is not None:
                usage.tokens = actual_tokens
                mean, count = self._mean_tokens.get((model, task), (0.0, 0))
                count = min(count + 1, 50)  # a moving average that follows drift
                self._mean_tokens[(model, task)] = (mean + (actual_tokens - mean) / count, count)
