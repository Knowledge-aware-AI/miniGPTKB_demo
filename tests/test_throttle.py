"""Tests of the per-model rate-limit pacing, on a fake clock."""

from __future__ import annotations

import unittest

from app.disambig.throttle import ModelThrottle, limits_from_headers

SCADS_HEADERS = {
    "x-ratelimit-model_per_key-limit-requests": "10",
    "x-ratelimit-model_per_key-limit-tokens": "1000",
    "x-ratelimit-model_per_key-remaining-requests": "9",
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def throttle_with_limits(headers=SCADS_HEADERS):
    clock = FakeClock()
    logs = []
    throttle = ModelThrottle(lambda message, level: logs.append(message), clock=clock, sleep=clock.sleep)
    throttle.record("m", "t", None, None, headers)  # as from a first response
    return throttle, clock, logs


def never() -> bool:
    return False


class ThrottleTest(unittest.TestCase):
    def test_reads_scads_headers(self):
        self.assertEqual(limits_from_headers(SCADS_HEADERS), (10, 1000))
        self.assertEqual(limits_from_headers({"content-type": "application/json"}), (None, None))

    def test_token_budget(self):
        throttle, clock, logs = throttle_with_limits()
        # 90% of 1000 tokens: two 400-token requests fit, the third waits for the first to leave the window.
        for _ in range(2):
            throttle.acquire("m", 400, never)
        self.assertEqual(clock.now, 0)
        throttle.acquire("m", 400, never)
        self.assertGreaterEqual(clock.now, 60)
        self.assertLess(clock.now, 62)
        self.assertTrue(any("Pacing" in message and "1000 tokens" in message for message in logs))

    def test_request_budget(self):
        throttle, clock, _ = throttle_with_limits()
        for _ in range(9):  # 90% of 10 requests
            throttle.acquire("m", 1, never)
        self.assertEqual(clock.now, 0)
        throttle.acquire("m", 1, never)
        self.assertGreaterEqual(clock.now, 60)

    def test_actual_tokens_replace_estimate(self):
        throttle, clock, _ = throttle_with_limits()
        first = throttle.acquire("m", 800, never)
        throttle.record("m", "t", first, 100)  # it really used 100 tokens
        throttle.acquire("m", 700, never)
        self.assertEqual(clock.now, 0)
        self.assertEqual(throttle.estimate("m", "t"), 100)  # learned from the actual usage
        self.assertEqual(throttle.estimate("m", "unseen task"), 256)  # default before any response

    def test_models_are_independent_and_unknown_limits_not_throttled(self):
        throttle, clock, _ = throttle_with_limits()
        throttle.acquire("m", 900, never)
        for _ in range(50):
            throttle.acquire("other", 900, never)  # no limits known for "other"
        self.assertEqual(clock.now, 0)

    def test_oversized_request_allowed_alone(self):
        throttle, clock, _ = throttle_with_limits()
        throttle.acquire("m", 5000, never)  # larger than the whole budget: allowed into an empty window
        throttle.acquire("m", 10, never)  # but nothing else until it has left the window
        self.assertGreaterEqual(clock.now, 60)

    def test_stop_while_waiting(self):
        throttle, clock, _ = throttle_with_limits()
        throttle.acquire("m", 900, never)
        self.assertIsNone(throttle.acquire("m", 900, lambda: clock.now > 5))


class DisabledThrottleTest(unittest.TestCase):
    def test_never_waits(self):
        clock = FakeClock()
        throttle = ModelThrottle(clock=clock, sleep=clock.sleep, enabled=False)
        throttle.record("m", "t", None, None, SCADS_HEADERS)
        for _ in range(50):
            self.assertIsNotNone(throttle.acquire("m", 900, never))
        self.assertEqual(clock.now, 0)


class UsageCounterTest(unittest.TestCase):
    def test_accumulates_across_sessions(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from app.usage import UsageCounter

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            counter = UsageCounter(path)
            counter.record_call("ner", SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3)))
            counter.record_retry("ner", rate_limited=True)
            counter.flush()
            resumed = UsageCounter(path)  # a resumed run continues the counts
            resumed.record_call("ner", SimpleNamespace(usage=None))
            resumed.flush()
            self.assertEqual(
                UsageCounter(path).roles["ner"],
                {"calls": 2, "prompt_tokens": 10, "completion_tokens": 3, "retries": 1, "rate_limited": 1},
            )


if __name__ == "__main__":
    unittest.main()

