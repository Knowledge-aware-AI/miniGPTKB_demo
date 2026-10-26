"""Which reasoning controls a model has, found out by asking the model itself.

Model lists carry no reliable information about reasoning: OpenAI's lists
only names, and the metadata a LiteLLM gateway such as ScaDS.AI publishes
contradicts what the models actually do (advertised effort levels that are
rejected or ignored, switches that have no effect). So the controls are
probed with a few tiny requests and cached per endpoint and model:

- OpenAI validates parameters strictly: one request with an invalid
  `reasoning_effort` is rejected either because the model has no reasoning
  ("unsupported parameter") or with the list of levels it accepts.
- Other OpenAI-compatible endpoints (vLLM, LiteLLM, ...): whether a reasoning
  trace comes back by default, whether one of the known switches turns it
  off (or on), and whether the effort levels really order the amount of
  reasoning (low < medium < high). Levels that do not are not offered.

OpenRouter is not probed: its model list reports reasoning support
reliably (see server.py).

A capability is {"toggle": {...} | None, "effort": {...} | None}, in the
shape apply_reasoning_settings (runner.py) writes into requests; None means
no controls. Results are kept for CACHE_TTL_SECONDS in a JSON file shared by
the server and the run processes. Probes that fail for a transient reason
(network, authentication, rate limit) are not cached.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(os.environ.get("MINIGPTKB_RUNS_DIR") or ROOT / "runs") / "_reasoning_capabilities.json"
CACHE_TTL_SECONDS = 24 * 3600

OPENAI_URL = "https://api.openai.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Effort level used when the user does not choose one: the cheapest real
# reasoning level, as before for gpt-oss and the small GPT-5 models.
DEFAULT_EFFORT = "low"

PROBE_PROMPTS = (
    "A train leaves at 9:40 and arrives at 13:15. How many minutes is the trip? Answer with the number only.",
    "If all bloops are razzies and some razzies are lazzies, are some bloops lazzies? Answer yes, no or unknown.",
)
PROBE_MAX_TOKENS = 1500

# Known on/off switches, tried in this order: (request body, path written
# by apply_reasoning_settings, value for on, value for off).
SWITCHES = (
    ({"chat_template_kwargs": {"enable_thinking": None}}, ("extra_body", "chat_template_kwargs", "enable_thinking"), True, False),
    ({"thinking": {"type": None}}, ("extra_body", "thinking", "type"), "enabled", "disabled"),
    ({"chat_template_kwargs": {"thinking": None}}, ("extra_body", "chat_template_kwargs", "thinking"), True, False),
)
# Effort is sent both as a request field and to the chat template: vLLM's
# templates (e.g. gpt-oss) only see the latter.
EFFORT_TARGETS = (("extra_body", "reasoning_effort"), ("extra_body", "chat_template_kwargs", "reasoning_effort"))
EFFORT_LEVELS = ("low", "medium", "high")

_lock = threading.Lock()
_key_locks: dict[tuple[str, str], threading.Lock] = {}


class ProbeUnavailable(Exception):
    """The endpoint could not be asked right now (network, key, rate limit)."""


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


# ---- cache ------------------------------------------------------------------------


def _read_cache() -> dict[str, Any]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache_entry(key: str, capability: dict[str, Any] | None) -> None:
    with _lock:
        cache = _read_cache()
        cache[key] = {"at": time.time(), "capability": capability}
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_PATH.with_name(CACHE_PATH.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            tmp.replace(CACHE_PATH)
        except OSError:
            pass


def cached_capability(llm_url: str, model: str) -> tuple[bool, dict[str, Any] | None]:
    """(found, capability) from the cache, if fresh."""
    entry = _read_cache().get(f"{normalize_url(llm_url)}|{model.strip()}")
    if entry and time.time() - entry.get("at", 0) < CACHE_TTL_SECONDS:
        return True, entry.get("capability")
    return False, None


# ---- public ---------------------------------------------------------------------


def capability_for(llm_url: str, api_key: str | None, model: str) -> dict[str, Any] | None:
    """The model's reasoning controls; probes the model when nothing is cached.
    Without a key, or when the endpoint cannot be reached, only the cache is
    used (and None is returned if it has nothing)."""
    url, model = normalize_url(llm_url), model.strip()
    if not url or not model or url == OPENROUTER_URL:
        return None
    found, capability = cached_capability(url, model)
    if found:
        return capability
    if not api_key:
        return None
    with _lock:
        key_lock = _key_locks.setdefault((url, model), threading.Lock())
    with key_lock:  # one probe per model at a time; later callers get the cached result
        found, capability = cached_capability(url, model)
        if found:
            return capability
        client = OpenAI(base_url=llm_url.strip().rstrip("/"), api_key=api_key, timeout=90.0, max_retries=1)
        try:
            capability = _probe_openai(client, model) if url == OPENAI_URL else _probe_generic(client, model)
        except ProbeUnavailable:
            return None
        _write_cache_entry(f"{url}|{model}", capability)
        return capability


# ---- OpenAI ------------------------------------------------------------------------


def _probe_openai(client: OpenAI, model: str) -> dict[str, Any] | None:
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Say OK."}],
            reasoning_effort="probe-invalid-level", max_completion_tokens=16,
        )
    except BadRequestError as exc:
        message = str(exc)
        lowered = message.lower()
        if "reasoning_effort" not in lowered and "reasoning.effort" not in lowered:
            return None  # rejected for another reason (e.g. not a chat model): no controls
        supported = re.search(r"supported values (?:are|is)[:\s]*(.*)", message, re.IGNORECASE)
        if not supported:
            return None  # "unsupported parameter": not a reasoning model
        levels = re.findall(r"'([a-z]+)'", supported.group(1))
        if not levels:
            return None
        default = DEFAULT_EFFORT if DEFAULT_EFFORT in levels else levels[0]
        return {"toggle": None, "effort": {"levels": levels, "default": default, "targets": [["reasoning_effort"]]}}
    except (AuthenticationError, RateLimitError, APIConnectionError) as exc:
        raise ProbeUnavailable(str(exc)) from exc
    except APIStatusError as exc:
        if exc.status_code >= 500 or exc.status_code in (401, 403, 429):
            raise ProbeUnavailable(str(exc)) from exc
        return None
    # The invalid level was accepted: the endpoint does not validate it, so
    # there is no way to tell which levels work.
    return None


# ---- other endpoints ------------------------------------------------------------------


def _trace_length(response: Any) -> int:
    message = response.choices[0].message
    extra = getattr(message, "model_extra", None) or {}
    trace = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) \
        or extra.get("reasoning_content") or extra.get("reasoning") or ""
    if not trace:
        content = message.content or ""
        if "<think>" in content:
            trace = content.split("</think>")[0]
    return len(trace)


def _ask(client: OpenAI, model: str, body: dict[str, Any], prompt: str) -> int | None:
    """Length of the reasoning trace for one request; None if the request was rejected."""
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=PROBE_MAX_TOKENS, temperature=0, extra_body=body or None,
        )
    except BadRequestError:
        return None
    except (AuthenticationError, RateLimitError, APIConnectionError) as exc:
        raise ProbeUnavailable(str(exc)) from exc
    except APIStatusError as exc:
        if exc.status_code >= 500 or exc.status_code in (401, 403, 429):
            raise ProbeUnavailable(str(exc)) from exc
        return None
    return _trace_length(response)


def _switch_body(template: dict[str, Any], value: Any) -> dict[str, Any]:
    outer, inner = next(iter(template.items()))
    return {outer: {next(iter(inner)): value}}


def _probe_generic(client: OpenAI, model: str) -> dict[str, Any] | None:
    with ThreadPoolExecutor(max_workers=6) as pool:
        default = list(pool.map(lambda prompt: _ask(client, model, {}, prompt), PROBE_PROMPTS))
    if any(length is None for length in default):
        return None  # the model rejects plain requests: nothing to offer
    reasons_by_default = any(default)

    toggle = None
    for template, target, on_value, off_value in SWITCHES:
        value = off_value if reasons_by_default else on_value
        length = _ask(client, model, _switch_body(template, value), PROBE_PROMPTS[0])
        if length is None:
            continue
        if (reasons_by_default and length == 0) or (not reasons_by_default and length > 0):
            toggle = {"default_on": reasons_by_default, "on_value": on_value, "off_value": off_value, "target": list(target)}
            break

    effort = None
    if reasons_by_default:
        effort = _probe_effort(client, model)

    if not toggle and not effort:
        return None
    return {"toggle": toggle, "effort": effort}


def _probe_effort(client: OpenAI, model: str) -> dict[str, Any] | None:
    """Offer effort levels only if they really order the amount of reasoning."""
    def ask(level_prompt: tuple[str, str]) -> int | None:
        level, prompt = level_prompt
        return _ask(client, model, {"reasoning_effort": level, "chat_template_kwargs": {"reasoning_effort": level}}, prompt)

    jobs = [(level, prompt) for level in EFFORT_LEVELS for prompt in PROBE_PROMPTS]
    with ThreadPoolExecutor(max_workers=6) as pool:
        lengths = list(pool.map(ask, jobs))
    if any(length is None for length in lengths):
        return None
    totals = {level: 0 for level in EFFORT_LEVELS}
    for (level, _), length in zip(jobs, lengths):
        totals[level] += length
    low, medium, high = totals["low"], totals["medium"], totals["high"]
    ordered = high >= 1.8 * max(low, 1) and 0.7 * low <= medium <= 1.3 * high
    if not ordered:
        return None
    return {"levels": list(EFFORT_LEVELS), "default": DEFAULT_EFFORT, "targets": [list(t) for t in EFFORT_TARGETS]}
