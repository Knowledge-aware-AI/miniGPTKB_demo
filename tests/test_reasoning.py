"""Tests of reasoning-capability probing (app/reasoning.py) against simulated
models that behave like the ones measured on ScaDS.AI and OpenAI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from openai import BadRequestError

from app import reasoning


def bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    return BadRequestError(message, response=httpx.Response(400, request=request), body=None)


class SimulatedModel:
    """Answers chat requests with a reasoning trace whose length depends on
    the request body, like a real model behind vLLM/LiteLLM."""

    def __init__(self, trace_for):
        self.trace_for = trace_for  # body -> trace length, or an exception to raise
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        result = self.trace_for(request.get("extra_body") or {})
        if isinstance(result, Exception):
            raise result
        message = SimpleNamespace(content="42", reasoning_content="x" * result, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def effort_of(body):
    return body.get("reasoning_effort")


def qwen_like(body):
    """Reasons by default; enable_thinking=false turns it off; effort is ignored or rejected."""
    if body.get("chat_template_kwargs", {}).get("enable_thinking") is False:
        return 0
    if effort_of(body) == "high":
        return bad_request("unsupported reasoning effort")
    return 300


def gemma_like(body):
    """No reasoning by default; thinking.type=enabled (or any effort) turns it on, levels make no difference."""
    if body.get("thinking", {}).get("type") == "enabled" or effort_of(body):
        return 1000
    return 0


def glm_like(body):
    """Reasons by default, ignores the switches; low and high both almost stop it, medium does not."""
    return {"low": 20, "medium": 1200, "high": 50}.get(effort_of(body), 1100)


def gpt_oss_like(body):
    """Reasons by default, no switch; effort orders the amount of reasoning."""
    return {"low": 120, "medium": 220, "high": 360}.get(effort_of(body), 220)


def plain(body):
    return 0


class GenericProbeTest(unittest.TestCase):
    def probe(self, behaviour):
        return reasoning._probe_generic(SimulatedModel(behaviour), "m")

    def test_switch_off(self):
        capability = self.probe(qwen_like)
        self.assertEqual(capability["toggle"]["target"], ["extra_body", "chat_template_kwargs", "enable_thinking"])
        self.assertTrue(capability["toggle"]["default_on"])
        self.assertIsNone(capability["effort"])  # a rejected level means no slider

    def test_switch_on(self):
        capability = self.probe(gemma_like)
        self.assertEqual(capability["toggle"]["target"], ["extra_body", "thinking", "type"])
        self.assertFalse(capability["toggle"]["default_on"])
        self.assertIsNone(capability["effort"])

    def test_unordered_effort_is_not_offered(self):
        self.assertIsNone(self.probe(glm_like))

    def test_ordered_effort(self):
        capability = self.probe(gpt_oss_like)
        self.assertIsNone(capability["toggle"])
        self.assertEqual(capability["effort"]["levels"], ["low", "medium", "high"])
        self.assertEqual(capability["effort"]["default"], "low")

    def test_no_reasoning(self):
        self.assertIsNone(self.probe(plain))


class OpenAIProbeTest(unittest.TestCase):
    def client_raising(self, error):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=mock.Mock(side_effect=error))))
        return reasoning._probe_openai(client, "m")

    def test_levels_from_validation_error(self):
        error = bad_request("Unsupported value: 'reasoning_effort' does not support 'probe-invalid-level' with this "
                            "model. Supported values are: 'minimal', 'low', 'medium', and 'high'.")
        capability = self.client_raising(error)
        self.assertEqual(capability["effort"]["levels"], ["minimal", "low", "medium", "high"])
        self.assertEqual(capability["effort"]["default"], "low")
        self.assertEqual(capability["effort"]["targets"], [["reasoning_effort"]])

    def test_not_a_reasoning_model(self):
        error = bad_request("Unsupported parameter: 'reasoning_effort' is not supported with this model.")
        self.assertIsNone(self.client_raising(error))


class CacheTest(unittest.TestCase):
    def test_probed_once_then_cached(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(reasoning, "CACHE_PATH", Path(tmp) / "cache.json"), \
                mock.patch.object(reasoning, "OpenAI", lambda **kwargs: SimulatedModel(qwen_like)) as _, \
                mock.patch.object(reasoning, "_probe_generic", wraps=reasoning._probe_generic) as probe:
            first = reasoning.capability_for("http://local/v1", "key", "qwen")
            second = reasoning.capability_for("http://local/v1/", "key", "qwen")
            self.assertEqual(first, second)
            self.assertEqual(probe.call_count, 1)
            # without a key nothing is probed, but the cached answer is still used
            self.assertEqual(reasoning.capability_for("http://local/v1", "", "qwen"), first)
            self.assertIsNone(reasoning.capability_for("http://local/v1", "", "other"))


if __name__ == "__main__":
    unittest.main()
