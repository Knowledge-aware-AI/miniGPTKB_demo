"""GPTKB 2.0 knowledge base construction with on-the-fly disambiguation,
as a miniGPTKB run (Hu, Nguyen & Razniewski, "Constructing Disambiguated
Knowledge Bases from Large Language Models at Scale", arXiv:2608.03729).

A port of the single-request mode of `Constructor.loop` in
https://github.com/Knowledge-aware-AI/GPTKB2_disambiguation_on_the_fly
(construction.py). Every round runs each stage once, in the repo's order, on
whatever items their status says are ready:

    NEDG -> NED2 -> NED1 -> Elicitation -> PDG -> CDG -> PD -> CD -> NER

with the same candidate retrieval (top-5 by label embedding; same-label
entities ranked by description for NED2), the same caching (a (predicate,
object label) pair resolved to one entity more than 50 times skips NER/NED),
the same guarded parallelization (pending object labels are clustered and
only one per cluster is disambiguated per round, unless its label has been
resolved more than 50 times already), and the same duplicate-triple handling.

What is miniGPTKB's rather than GPTKB 2.0's: the elicitation prompt (topic-
scoped, multilingual, editable -- plus GPTKB 2.0's subject description line),
the endpoints/models (any OpenAI-compatible Chat Completions server, one model
each for elicitation, NER and NED), BFS depth tracking, the stopping criteria,
cost tracking and the live progress events/HTML report of this app.

Like the repo, only LLM calls run concurrently (up to `nthreads` at once);
every database and index write happens on the loop thread, in order.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from openai import OpenAI, RateLimitError

from app.runner import (
    OPENAI_TEXT_PRICING,
    OPENAI_PRICING_LAST_UPDATED,
    OPENROUTER_REASONING_CAPABILITY,
    RequestRateLimiter,
    append_subject_line,
    apply_reasoning_settings,
    build_elicitation_prompt,
    build_elicitation_prompt_body,
    get_reasoning_capability,
    resolve_reasoning_settings,
)
from app import reasoning
from app.disambig import prompts as P
from app.disambig import store as S
from app.disambig.embeddings import EmbeddingClient, VectorIndex, cluster_labels
from app.disambig.report import ReportWriter
from app.disambig.throttle import ModelThrottle
from app.usage import UsageCounter

ProgressCallback = Callable[[dict[str, Any]], None]

SCADS_EMBEDDING_URL = "https://llm.scads.ai/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

# GPTKB 2.0's seed predicate (Constructor.__init__ default). GPTKB 2.0 also
# seeds the class "human", only because its class disambiguation does nothing
# while the class index is empty; here a class (or entity, or relation) with
# no candidates at all skips the LLM and goes straight to "None of above", so
# a topic-specific KB starts without an unrelated seed class.
SEED_PREDICATE = ("instanceOf", "relation of type constraints")

ROLES = ("elicitation", "ner", "ned")
ROLE_NAMES = {"elicitation": "Elicitation", "ner": "NER", "ned": "NED"}
REPORT_INTERVAL_SECONDS = 20.0
STATUS_INTERVAL_SECONDS = 1.0


@dataclass
class DisambigConfig:
    llm_url: str
    api_key: str
    model: str  # elicitation model
    topic: str
    seed_entity: str
    ner_model: str = ""
    ned_model: str = ""
    # Endpoints of the NER/NED models when not the elicitation endpoint
    # (e.g. ScaDS.AI next to OpenAI); empty means the elicitation endpoint.
    ner_llm_url: str = ""
    ner_api_key: str = ""
    ned_llm_url: str = ""
    ned_api_key: str = ""
    language: str = "en"
    schema_mode: str = "open"
    schema_text: str = ""
    facts_multiplier: float = 1.0
    prompt_template: str = ""
    max_entities: int | None = None
    max_bfs_depth: int | None = None
    max_runtime_seconds: int | None = None
    max_cost: float | None = None
    nthreads: int = 5
    request_timeout_seconds: float = 120.0
    max_requests_per_minute: int | None = None
    reasoning_effort: str | None = None
    reasoning_enabled: bool | None = None
    pipeline: str = "gptkb2"
    embedding_url: str = SCADS_EMBEDDING_URL
    embedding_api_key: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # GPTKB 2.0 hyperparameters (paper Section 4.1 / repo single mode).
    num_candidates: int = 5  # relations and classes
    # Entity merging (NED1/NED2): how many candidates the model is shown
    # (GPTKB 2.0: 5), and the minimum label similarity a candidate needs to be
    # shown at all -- 0 is GPTKB 2.0; raising it makes merging more
    # conservative. Entities with the mention's exact label or alias are
    # always candidates.
    ned_candidates: int = 5
    ned_similarity_floor: float = 0.0
    frequency_threshold: int = 50  # caching and guarded parallelization
    cluster_distance: float = 0.15
    chunk_size: int = 20
    fetch_size: int = 500
    max_idle_rounds: int = 3
    # Not in GPTKB 2.0 (which retries a failing item forever): give up on an
    # item after this many failed requests/unparsable answers, so a run
    # always terminates.
    max_attempts: int = 3
    # Per-model pacing below the endpoint's rate limits; can be switched off
    # through the API (`"pacing": false`) to compare runs with and without it.
    pacing: bool = True

    def endpoint_for(self, role: str) -> tuple[str, str]:
        """(URL, API key) of the role's endpoint; NER/NED default to the elicitation endpoint."""
        if role == "ner" and self.ner_llm_url.strip():
            return self.ner_llm_url, self.ner_api_key
        if role == "ned" and self.ned_llm_url.strip():
            return self.ned_llm_url, self.ned_api_key
        return self.llm_url, self.api_key

    def model_for(self, role: str) -> str:
        if role == "ner":
            return self.ner_model.strip() or self.model
        if role == "ned":
            return self.ned_model.strip() or self.model
        return self.model


class StopRequested(Exception):
    pass


OPENAI_URL = "https://api.openai.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


@dataclass
class Endpoint:
    """One OpenAI-compatible server the run talks to. NER and NED can use a
    different endpoint than elicitation (e.g. ScaDS.AI models next to an
    OpenAI elicitation model), so the client keeps one per distinct URL."""

    url: str
    api_key: str
    client: OpenAI
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)  # OpenRouter's, per model

    @property
    def kind(self) -> str:
        normalized = normalize_url(self.url)
        if normalized == OPENAI_URL:
            return "openai"
        if normalized == OPENROUTER_URL:
            return "openrouter"
        if normalized == normalize_url(SCADS_EMBEDDING_URL):
            return "scads"
        return "other"

    @property
    def host(self) -> str:
        return self.url.split("//", 1)[-1].split("/", 1)[0]


class LLMClient:
    """Chat Completions calls for the three roles, each on its own endpoint,
    sharing one request-rate limiter and one cost counter. Mirrors
    LocalKBCRunner's request handling (retries with backoff, reasoning
    controls, temperature 0 except for OpenAI's GPT-5 family, OpenAI/
    OpenRouter pricing; ScaDS.AI is free)."""

    def __init__(
        self,
        config: DisambigConfig,
        stop_event: threading.Event,
        log: Callable[[str, str], None],
        usage: UsageCounter | None = None,
    ):
        self.config = config
        self.usage = usage
        self.stop_event = stop_event
        self.log = log
        self.endpoints: dict[str, Endpoint] = {}
        self.role_endpoint: dict[str, Endpoint] = {}
        for role in ROLES:
            url, api_key = config.endpoint_for(role)
            key = normalize_url(url)
            if key not in self.endpoints:
                timeout = config.request_timeout_seconds
                if key == OPENAI_URL:
                    timeout = max(timeout, 300.0)  # GPT-5 models can take minutes
                self.endpoints[key] = Endpoint(url, api_key, OpenAI(base_url=url, api_key=api_key, timeout=timeout, max_retries=0))
            self.role_endpoint[role] = self.endpoints[key]
        self.rate_limiter = RequestRateLimiter(config.max_requests_per_minute)
        self.throttle = ModelThrottle(log, enabled=config.pacing)
        if not config.pacing:
            log("Pacing is off: requests are sent without waiting for the endpoint's rate limits.", "warning")
        self.cost_lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.capabilities = {role: self._capability(role) for role in ROLES}
        # Elicitation uses the reasoning settings chosen in the form. NER and
        # NED use the lowest effort the model offers -- GPTKB 2.0 runs its
        # disambiguation prompts with minimal reasoning and 128 output tokens.
        self.reasoning: dict[str, tuple[str | None, bool | None]] = {
            "elicitation": (config.reasoning_effort, config.reasoning_enabled),
        }
        for role in ("ner", "ned"):
            effort = (self.capabilities[role] or {}).get("effort")
            self.reasoning[role] = (effort["levels"][0] if effort else None, None)

    def model_label(self, role: str) -> str:
        """The role's model, with its endpoint's host if the run uses several endpoints."""
        model = self.config.model_for(role)
        return f"{model} ({self.role_endpoint[role].host})" if len(self.endpoints) > 1 else model

    def _capability(self, role: str) -> dict[str, Any] | None:
        endpoint = self.role_endpoint[role]
        if endpoint.kind == "openrouter":
            return OPENROUTER_REASONING_CAPABILITY
        model = self.config.model_for(role)
        if not reasoning.cached_capability(endpoint.url, model)[0]:
            self.log(f"Checking the reasoning options of {model} (first use, a few seconds).", "info")
        return get_reasoning_capability(endpoint.url, model, endpoint.api_key)

    def resolved_reasoning(self, role: str) -> tuple[str | None, bool | None]:
        effort, enabled = self.reasoning[role]
        return resolve_reasoning_settings(self.capabilities[role], effort, enabled)

    def cost_tracked(self) -> bool:
        """Whether any role runs on a priced endpoint (OpenAI, OpenRouter)."""
        return any(endpoint.kind in ("openai", "openrouter") for endpoint in self.role_endpoint.values())

    def has_pricing(self) -> bool:
        """Whether every role's cost is known: a priced model, or free (ScaDS.AI)."""
        for role, endpoint in self.role_endpoint.items():
            model = self.config.model_for(role).strip().lower()
            if endpoint.kind == "openai" and model not in OPENAI_TEXT_PRICING:
                return False
            if endpoint.kind == "openrouter" and model not in endpoint.pricing:
                return False
            if endpoint.kind == "other":
                return False
        return self.cost_tracked()

    def load_openrouter_pricing(self) -> None:
        import urllib.request

        for endpoint in self.endpoints.values():
            if endpoint.kind != "openrouter":
                continue
            request = urllib.request.Request(
                endpoint.url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {endpoint.api_key}"}
            )
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                self.log(f"Could not fetch OpenRouter model pricing: {exc}", "warning")
                continue
            for model in data.get("data", []):
                model_id = str(model.get("id", "")).strip().lower()
                price = model.get("pricing") or {}
                try:
                    endpoint.pricing[model_id] = {
                        "prompt_per_token": float(price.get("prompt", 0) or 0),
                        "completion_per_token": float(price.get("completion", 0) or 0),
                    }
                except (TypeError, ValueError):
                    continue

    def chat(self, role: str, system: str, user: str, max_retries: int = 5) -> str:
        model = self.config.model_for(role)
        request_args: dict[str, Any] = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
        }
        effort, enabled = self.reasoning[role]
        apply_reasoning_settings(request_args, self.capabilities[role], effort, enabled)
        endpoint = self.role_endpoint[role]
        if not (endpoint.kind == "openai" and model.lower().startswith("gpt-5")):
            request_args["temperature"] = 0

        # Requests of one kind (same instructions) have similar sizes, so the
        # throttle's token estimate is kept per system prompt.
        task = system[:80] or "elicitation"
        # Rate limits are per model and endpoint.
        throttle_key = self.model_label(role)
        estimate = self.throttle.estimate(throttle_key, task)

        attempt = 0
        while True:
            if not self.rate_limiter.acquire(self.stop_event):
                raise StopRequested()
            reservation = self.throttle.acquire(throttle_key, estimate, self.stop_event.is_set, ROLE_NAMES[role])
            if reservation is None:
                raise StopRequested()
            try:
                raw = endpoint.client.chat.completions.with_raw_response.create(**request_args)
                response = raw.parse()
                usage = getattr(response, "usage", None)
                completion = getattr(usage, "completion_tokens", None)
                # What the endpoint charges against its token limit: output tokens, plus one.
                charged = completion + 1 if completion is not None else None
                self.throttle.record(throttle_key, task, reservation, charged, raw.headers)
                self._record_cost(model, endpoint, response)
                if self.usage:
                    self.usage.record_call(role, response)
                return response.choices[0].message.content or ""
            except Exception as exc:
                if isinstance(exc, RateLimitError) and getattr(exc, "response", None) is not None:
                    self.throttle.record(throttle_key, task, None, None, exc.response.headers)
                if self.stop_event.is_set():
                    raise StopRequested() from exc
                attempt += 1
                if self.usage:
                    self.usage.record_retry(role, isinstance(exc, RateLimitError))
                if attempt > max_retries:
                    raise
                delay = self._retry_delay(exc, attempt)
                if isinstance(exc, RateLimitError):
                    reason = f"rate limit hit{self._rate_limit_detail(exc)}"
                else:
                    reason = f"{type(exc).__name__}: {str(exc)[:160]}"
                self.log(
                    f"{ROLE_NAMES[role]} request to {self.model_label(role)} failed ({reason}); "
                    f"retrying in {delay:.1f}s (attempt {attempt}/{max_retries}).",
                    "warning",
                )
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    if self.stop_event.is_set():
                        raise StopRequested() from exc
                    time.sleep(min(0.5, deadline - time.monotonic()))

    @staticmethod
    def _rate_limit_detail(exc: RateLimitError) -> str:
        """The per-minute limits the endpoint reports (ScaDS/LiteLLM's
        x-ratelimit-* headers), e.g. " (limit 60 requests, 10000 tokens per minute)"."""
        response = getattr(exc, "response", None)
        headers = response.headers if response is not None else {}
        requests = next((v for k, v in headers.items() if k.lower().endswith("limit-requests")), None)
        tokens = next((v for k, v in headers.items() if k.lower().endswith("limit-tokens")), None)
        parts = [f"{requests} requests" if requests else None, f"{tokens} tokens" if tokens else None]
        parts = [part for part in parts if part]
        return f" (limit {', '.join(parts)} per minute)" if parts else ""

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        import random

        if isinstance(exc, RateLimitError):
            response = getattr(exc, "response", None)
            header = response.headers.get("retry-after") if response is not None else None
            if header is not None:
                try:
                    return max(0.5, float(header))
                except ValueError:
                    pass
        return min(60.0, 2**attempt) + random.uniform(0, 1)

    def _record_cost(self, model: str, endpoint: Endpoint, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        key = model.strip().lower()
        if endpoint.kind == "openai":
            pricing = OPENAI_TEXT_PRICING.get(key)
            if pricing is None:
                return
            cost = input_tokens / 1e6 * pricing["input_per_mtok"] + output_tokens / 1e6 * pricing["output_per_mtok"]
        elif endpoint.kind == "openrouter":
            pricing = endpoint.pricing.get(key)
            if pricing is None:
                return
            cost = input_tokens * pricing["prompt_per_token"] + output_tokens * pricing["completion_per_token"]
        else:
            return
        with self.cost_lock:
            self.total_cost_usd += cost


class DisambiguatedKBCRunner:
    def __init__(
        self,
        config: DisambigConfig,
        run_dir: Path,
        emit: ProgressCallback,
        pause_event: threading.Event,
        stop_event: threading.Event,
    ):
        self.config = config
        self.run_dir = run_dir
        self.emit = emit
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.db_path = run_dir / "knowledgeBase.db"
        self.llm = LLMClient(config, stop_event, self._log, UsageCounter(run_dir / "usage.json"))
        self.instance_of_labels = P.instance_of_labels(config.language)
        self.entity_index = VectorIndex()
        self.predicate_index = VectorIndex()
        self.concept_index = VectorIndex()
        self.dirty_entities: set[str] = set()
        self.active_requests = 0
        self.active_lock = threading.Lock()
        self.started_at = 0.0
        self.cost_before_session = 0.0
        self.exploration_closed = False
        self.limit_reason: str | None = None
        self._last_status_emit = 0.0
        self._last_report = 0.0

    # ---- logging / events -------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        self.emit({"type": "log", "level": level, "message": message, "time": time.strftime("%H:%M:%S")})

    def _current(self, text: str) -> None:
        self.emit({"type": "current_subject", "subject": text})

    def total_cost(self) -> float:
        with self.llm.cost_lock:
            return self.cost_before_session + self.llm.total_cost_usd

    def _cost_tracked(self) -> bool:
        return self.llm.cost_tracked()

    def _emit_status(self, status: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_emit < STATUS_INTERVAL_SECONDS:
            return
        self._last_status_emit = now
        counts = S.progress_counts(self.db_path) or {}
        with self.active_lock:
            active = self.active_requests
        self.emit(
            {
                "type": "status",
                "status": status,
                "processed": counts.get("processed", 0),
                "queued": counts.get("queued", 0),
                "active": active,
                "triples": counts.get("triples", 0),
                "literals": counts.get("literals", 0),
                "entities": counts.get("entities", 0),
                "pending": counts.get("pending", 0),
                "failed_subjects": counts.get("failed_subjects", 0),
                # Kept for the legacy dashboard fields.
                "off_domain_entities": 0,
                "total_objects": counts.get("entities", 0),
                "cost_usd": round(self.total_cost(), 4) if self._cost_tracked() else None,
            }
        )

    # ---- main loop ---------------------------------------------------------

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db = S.connect(self.db_path)
        self.embedder = EmbeddingClient(
            self.config.embedding_url,
            self.config.embedding_api_key,
            self.config.embedding_model,
            self.db,
            self.stop_event,
            self._log,
        )
        self.report = ReportWriter(self)
        self.started_at = time.monotonic()
        final_status = "finished"
        try:
            self._log(
                f"Models: elicitation {self.llm.model_label('elicitation')}, NER {self.llm.model_label('ner')}, "
                f"NED {self.llm.model_label('ned')}; embeddings {self.config.embedding_model}."
            )
            if any(endpoint.kind == "openai" for endpoint in self.llm.endpoints.values()):
                self._log(
                    f"OpenAI pricing table last updated: {OPENAI_PRICING_LAST_UPDATED}. "
                    "Verify current prices at https://developers.openai.com/api/docs/pricing before relying on cost tracking.",
                    "warning",
                )
            self.llm.load_openrouter_pricing()
            if self.config.max_cost is not None and not self.llm.has_pricing():
                self._log("The cost of some selected models is unknown; ignoring the cost limit.", "warning")
                self.config.max_cost = None

            self._emit_status("running", force=True)
            self._load_or_seed()
            self._write_run_row("running")
            self.report.write(full=True)
            self._loop()
        except StopRequested:
            pass
        finally:
            if self.stop_event.is_set():
                final_status = "stopped"
            try:
                if self.llm.usage:
                    self.llm.usage.flush()
                self._write_run_row(final_status)
                self.report.write(full=True, final=True)
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as exc:
                self._log(f"Could not write the final report: {exc}", "error")
            self._log(f"Knowledge extraction pipeline {final_status}.")
            self._emit_status(final_status, force=True)
            self.db.close()

    def _loop(self) -> None:
        idle_rounds = 0
        rounds = 0
        while True:
            if self._check_stop():
                return
            rounds += 1
            did_work = False
            if_ned = if_pd = if_cd = True

            if self._stage_nedg():
                did_work, if_ned = True, False
            if self._stage_ned2():
                did_work, if_ned = True, False
            if if_ned and self._stage_ned1():
                did_work = True
            if self._stage_elicitation():
                did_work = True
            if self._stage_pdg():
                did_work, if_pd = True, False
            if self._stage_cdg():
                did_work, if_cd = True, False
            if if_pd and self._stage_pd():
                did_work = True
            if if_cd and self._stage_cd():
                did_work = True
            if self._stage_ner():
                did_work = True

            self._emit_status("running")
            if time.monotonic() - self._last_report >= REPORT_INTERVAL_SECONDS:
                self._write_run_row("running")
                self.report.write()
                self._last_report = time.monotonic()

            if did_work:
                idle_rounds = 0
            else:
                idle_rounds += 1
                if idle_rounds >= self.config.max_idle_rounds:
                    if self.limit_reason:
                        self._log(f"{self.limit_reason} All pending triples are disambiguated.")
                    else:
                        self._log("Nothing left to explore or disambiguate: the knowledge base is complete.")
                    return

    def _check_stop(self) -> bool:
        self._wait_if_paused()
        if self.stop_event.is_set():
            return True
        if self.config.max_runtime_seconds and time.monotonic() - self.started_at >= self.config.max_runtime_seconds:
            self._log("Runtime limit reached.")
            self.stop_event.set()
            return True
        if self.config.max_cost is not None and self.total_cost() >= self.config.max_cost:
            self._log(f"Cost limit reached (${self.total_cost():.4f} spent, limit ${self.config.max_cost:.2f}).")
            self.stop_event.set()
            return True
        return False

    def _wait_if_paused(self) -> None:
        announced = False
        while self.pause_event.is_set() and not self.stop_event.is_set():
            if not announced:
                self._emit_status("paused", force=True)
                announced = True
            time.sleep(0.5)
        if announced and not self.stop_event.is_set():
            self._emit_status("running", force=True)

    def _fan_out(self, role: str, items: list[Any], build: Callable[[Any], tuple[str, str]]) -> Iterator[tuple[Any, str | None, Exception | None]]:
        """Sends one request per item, at most `nthreads` in flight, and
        yields (item, response text, error) as each completes -- on the
        calling (loop) thread, so the caller can write results directly."""
        if not items:
            return
        pending: dict[Future, Any] = {}
        queue = list(items)
        with ThreadPoolExecutor(max_workers=max(1, self.config.nthreads)) as executor:
            def call(item: Any) -> str:
                system, user = build(item)
                with self.active_lock:
                    self.active_requests += 1
                try:
                    return self.llm.chat(role, system, user)
                finally:
                    with self.active_lock:
                        self.active_requests -= 1

            try:
                while queue or pending:
                    while queue and len(pending) < max(1, self.config.nthreads):
                        if self._check_stop():
                            raise StopRequested()
                        item = queue.pop(0)
                        pending[executor.submit(call, item)] = item
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        item = pending.pop(future)
                        try:
                            yield item, future.result(), None
                        except StopRequested:
                            raise
                        except Exception as exc:
                            yield item, None, exc
                    self._emit_status("running")
            except StopRequested:
                for future in pending:
                    future.cancel()
                raise

    # ---- ids / bookkeeping -------------------------------------------------

    def _next_id(self, prefix: str, key: str) -> tuple[str, int]:
        value = int(S.get_meta(self.db, key, "0"))
        S.set_meta(self.db, key, value + 1)
        return f"{prefix}{value}", value

    def _record_failure(self, subject: str, stage: str, error: str, detail: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO failures (subject, stage, error, detail, time) VALUES (?, ?, ?, ?, ?)",
            (subject, stage, error[:2000], detail, time.strftime("%H:%M:%S")),
        )

    def _log_decision(
        self,
        triple_id: str,
        stage: str,
        target: str,
        context: str,
        candidate_ids: list[str],
        answer: str | None,
        outcome: str,
        chosen_id: str | None,
    ) -> None:
        self.db.execute(
            "INSERT INTO decisions (triple_id, stage, target, context, candidates, answer, outcome, chosen_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                triple_id, stage, target, context, json.dumps(candidate_ids), (answer or "")[:500],
                outcome, chosen_id, time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )

    def _write_run_row(self, status: str) -> None:
        try:
            run_config = json.loads((self.run_dir / "runConfig.json").read_text(encoding="utf-8"))
        except Exception:
            run_config = {}
        effort, enabled = self.llm.resolved_reasoning("elicitation")
        c = self.config
        self.db.execute("DELETE FROM run")
        self.db.execute(
            "INSERT INTO run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.run_dir.name, c.pipeline, c.topic, c.seed_entity,
                c.model_for("elicitation"), c.model_for("ner"), c.model_for("ned"), c.embedding_model,
                c.llm_url, c.language, c.schema_mode, c.schema_text, self.elicitation_prompt_body(),
                c.facts_multiplier, c.nthreads, c.max_requests_per_minute,
                c.max_entities, c.max_bfs_depth, c.max_runtime_seconds, c.max_cost,
                run_config.get("started_at"), run_config.get("ended_at"), status,
                round(self.total_cost(), 4) if self._cost_tracked() else None,
                effort, None if enabled is None else int(enabled),
            ),
        )
        S.set_meta(self.db, "cost_usd", self.total_cost())
        self.db.commit()

    def elicitation_prompt_body(self) -> str:
        c = self.config
        return c.prompt_template.strip() or build_elicitation_prompt_body(
            c.language, c.topic, c.facts_multiplier, c.schema_mode, c.schema_text
        )

    # ---- seeding / resume --------------------------------------------------

    def _load_or_seed(self) -> None:
        self.cost_before_session = float(S.get_meta(self.db, "cost_usd", "0") or 0)
        if self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0:
            self._seed()
        else:
            self._log("Resuming from the existing knowledge base.")
        for kind, table, index in (
            ("entities", "entities", self.entity_index),
            ("predicates", "predicates", self.predicate_index),
            ("classes", "concepts", self.concept_index),
        ):
            rows = self.db.execute(f"SELECT id, label FROM {table} ORDER BY seq").fetchall()
            if table == "entities":
                rows += self.db.execute("SELECT entity_id, alias FROM entity_aliases ORDER BY rowid").fetchall()
            if rows:
                index.add([r[0] for r in rows], self.embedder.embed([r[1] for r in rows]))
        self._log(
            f"Indexed {len(self.entity_index)} entities, {len(self.predicate_index)} relations and "
            f"{len(self.concept_index)} classes for candidate retrieval."
        )

    def _seed(self) -> None:
        seed = self.config.seed_entity.strip()
        self._current(f"Describing seed entity {seed}")
        system, user = P.seed_description_messages(seed, self.config.topic)
        try:
            description = P.parse_description(self.llm.chat("ned", system, user))
        except StopRequested:
            raise
        except Exception as exc:
            description = f"{seed}, the seed entity of a knowledge base on {self.config.topic}."
            self._log(f"Could not generate a description for the seed entity ({exc}); using a generic one.", "warning")
        self.db.execute(
            "INSERT INTO entities (id, seq, label, description, status, depth, first_appeared) VALUES (?, ?, ?, ?, ?, 0, ?)",
            ("E0", 0, seed, description, S.ENTITY_UNEXPLORED, "Where the exploration starts"),
        )
        self.db.execute(
            "INSERT INTO predicates (id, seq, label, description, first_appeared) VALUES ('P0', 0, ?, ?, ?)",
            (*SEED_PREDICATE, "Where the exploration starts"),
        )
        S.set_meta(self.db, "next_entity", 1)
        S.set_meta(self.db, "next_predicate", 1)
        S.set_meta(self.db, "next_concept", 0)
        S.set_meta(self.db, "next_triple", 0)
        self.db.commit()
        self.dirty_entities.add("E0")
        self._log(f"Seed entity E0 '{seed}': {description}")

    # ---- shared update helpers --------------------------------------------

    def _delete_triple(self, triple_id: str) -> None:
        self.db.execute("DELETE FROM triples WHERE id = ?", (triple_id,))
        self.db.execute("DELETE FROM instance_triples WHERE triple_id = ?", (triple_id,))
        self.db.execute("DELETE FROM candidates WHERE triple_id = ?", (triple_id,))

    def _update_triple(self, triple_id: str, **fields: Any) -> bool:
        """Applies an update that may give the triple an id-level duplicate of
        an existing triple; GPTKB 2.0 deletes such a triple instead (it checks
        signatures first, the unique index does that here). False if deleted."""
        assignments = ", ".join(f"{name} = ?" for name in fields)
        try:
            self.db.execute(f"UPDATE triples SET {assignments} WHERE id = ?", (*fields.values(), triple_id))
            return True
        except Exception as exc:
            if "UNIQUE" not in str(exc):
                raise
            self._delete_triple(triple_id)
            return False

    def _resolve_object(self, triple: dict[str, Any], entity_id: str, resolution: str) -> None:
        if self._update_triple(
            triple["id"], object_id=entity_id, object_status=S.OBJ_FINISHED, resolution=resolution
        ):
            label = self.db.execute("SELECT label FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if label and label[0] != triple["object_label"]:
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                    (entity_id, triple["object_label"]),
                )
                if cursor.rowcount:
                    # As in the paper (Section 3.2.3): the merged mention's
                    # label joins the entity's aliases "so that future lookups
                    # can find the entity by either name" -- the published
                    # code only stores it, here it is also indexed.
                    self.entity_index.add([entity_id], self.embedder.embed([triple["object_label"]]))
            self.dirty_entities.add(entity_id)
            self.dirty_entities.add(triple["subject_id"])

    def _bump_attempts(self, table: str, key_column: str, key: str) -> int:
        self.db.execute(f"UPDATE {table} SET attempts = attempts + 1 WHERE {key_column} = ?", (key,))
        row = self.db.execute(f"SELECT attempts FROM {table} WHERE {key_column} = ?", (key,)).fetchone()
        return int(row[0]) if row else 0

    def _triple_failed(self, triple: dict[str, Any], stage: str, error: str, detail: str | None, give_up: dict[str, Any]) -> None:
        """Counts a failed request/answer against a triple; after max_attempts
        the triple leaves its stage with `give_up` applied."""
        if self._bump_attempts("triples", "id", triple["id"]) >= self.config.max_attempts:
            self._update_triple(triple["id"], attempts=0, **give_up)
            self._record_failure(triple["object_label"], stage, error, detail)
            self._log(f"{stage}: giving up on '{triple['object_label']}' in triple {triple['id']} ({error[:120]}).", "warning")

    @staticmethod
    def _rows(cursor) -> list[dict[str, Any]]:
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _entity_rows(self, ids: list[str]) -> dict[str, tuple[str, str]]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        return {
            row[0]: (row[1], row[2])
            for row in self.db.execute(f"SELECT id, label, description FROM entities WHERE id IN ({placeholders})", ids)
        }

    def _set_candidates(self, triple_id: str, stage: str, ids: list[str]) -> None:
        self.db.execute("DELETE FROM candidates WHERE triple_id = ? AND stage = ?", (triple_id, stage))
        self.db.executemany(
            "INSERT INTO candidates (triple_id, stage, position, ref_id) VALUES (?, ?, ?, ?)",
            [(triple_id, stage, position, ref_id) for position, ref_id in enumerate(ids)],
        )

    def _cached_object_ids(self, pairs: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """GPTKB 2.0's caching: a (predicate label, object label) pair whose
        object was resolved more than `frequency_threshold` times, always to
        the same entity, resolves to it directly."""
        result: dict[tuple[str, str], str] = {}
        pair_list = list(pairs)
        for start in range(0, len(pair_list), 400):
            chunk = pair_list[start : start + 400]
            values = ",".join("(?, ?)" for _ in chunk)
            params = [value for pair in chunk for value in pair]
            rows = self.db.execute(
                f"""
                SELECT predicate_label, object_label, MIN(object_id) FROM triples
                WHERE object_type = 'ne' AND object_id IS NOT NULL
                  AND (predicate_label, object_label) IN (VALUES {values})
                GROUP BY predicate_label, object_label
                HAVING COUNT(*) > ? AND MIN(object_id) = MAX(object_id)
                """,
                [*params, self.config.frequency_threshold],
            ).fetchall()
            for predicate, obj, object_id in rows:
                result[(predicate, obj)] = object_id
        return result

    # ---- Elicitation -------------------------------------------------------

    def _stage_elicitation(self) -> bool:
        if self.config.max_entities and not self.exploration_closed:
            count = self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            if count >= self.config.max_entities:
                self.exploration_closed = True
                self.limit_reason = f"Entity limit reached ({count:,} entities): no further entities are explored."
                self._log(self.limit_reason + " Finishing the disambiguation of pending triples.")
        if self.exploration_closed:
            return False

        depth_clause = "" if self.config.max_bfs_depth is None else "AND depth <= ?"
        params: list[Any] = [] if self.config.max_bfs_depth is None else [self.config.max_bfs_depth]
        entities = self._rows(
            self.db.execute(
                f"SELECT id, label, description, depth FROM entities WHERE status = 'unexplored' {depth_clause} "
                f"ORDER BY seq LIMIT ?",
                [*params, max(self.config.chunk_size, self.config.nthreads)],
            )
        )
        if not entities:
            return False
        names = ", ".join(e["label"] for e in entities[:3])
        self._current(f"Elicitation: {names}" + (f" (+{len(entities) - 3} more)" if len(entities) > 3 else ""))

        base = self.elicitation_prompt_body()
        language = self.config.language
        def build(entity: dict[str, Any]) -> tuple[str, str]:
            prompt = append_subject_line(base, language, entity["label"]) if self.config.prompt_template.strip() else (
                build_elicitation_prompt(
                    language, self.config.topic, entity["label"], self.config.facts_multiplier,
                    self.config.schema_mode, self.config.schema_text,
                )
            )
            return "", P.elicitation_user_message(prompt, language, entity["description"])

        explored = stored = 0
        for entity, output, error in self._fan_out("elicitation", entities, build):
            try:
                if error:
                    raise error
                facts = P.parse_elicitation(output or "")
            except Exception as exc:
                stage = "llm_request" if error else "json_parse"
                if self._bump_attempts("entities", "id", entity["id"]) >= self.config.max_attempts:
                    self.db.execute("UPDATE entities SET status = 'failed' WHERE id = ?", (entity["id"],))
                    self._record_failure(entity["label"], stage, str(exc), None if error else output)
                    self._log(f"Giving up on {entity['label']} after repeated failures ({str(exc)[:120]}).", "error")
                else:
                    self._log(f"Elicitation failed for {entity['label']} ({str(exc)[:120]}); will retry.", "warning")
                self.db.commit()
                continue
            count = self._commit_new_triples(entity, facts)
            self.db.execute(
                "UPDATE entities SET status = 'explored', num_triples = ? WHERE id = ?", (count, entity["id"])
            )
            self.db.commit()
            self.dirty_entities.add(entity["id"])
            explored += 1
            stored += count
            self._log(f"Processed {entity['label']} ({entity['id']}): stored {count} triples.")
        if explored:
            self._log(f"Elicitation: {explored} entities explored, {stored} new triples.")
        return True

    def _commit_new_triples(self, entity: dict[str, Any], facts: list[dict[str, str]]) -> int:
        seen: set[tuple[str, str, str]] = set()
        unique = []
        for fact in facts:
            key = (fact["subject"], fact["predicate"], fact["object"])
            if key not in seen:
                seen.add(key)
                unique.append(fact)
        cache = self._cached_object_ids({(f["predicate"], f["object"]) for f in unique})

        inserted = 0
        for fact in unique:
            triple_id, seq = self._next_id("T", "next_triple")
            predicate, obj = fact["predicate"], fact["object"]
            row: dict[str, Any] = {
                "id": triple_id, "seq": seq, "subject_id": entity["id"], "subject_label": fact["subject"],
                "predicate_label": predicate, "predicate_id": None, "predicate_status": S.PRED_GENERATED,
                "object_label": obj, "object_id": None, "object_type": S.TYPE_UNRECOGNIZED,
                "object_status": S.OBJ_GENERATED, "resolution": None,
            }
            is_instance = False
            if (predicate, obj) in cache:
                row.update(object_id=cache[(predicate, obj)], object_type=S.TYPE_NE,
                           object_status=S.OBJ_FINISHED, resolution="cache")
            elif predicate in self.instance_of_labels:
                row.update(predicate_id="P0", predicate_status=S.PRED_FINISHED, object_type=S.TYPE_CLASS,
                           object_status=S.OBJ_FINISHED)
                is_instance = True
            columns = ", ".join(row)
            cursor = self.db.execute(
                f"INSERT OR IGNORE INTO triples ({columns}) VALUES ({', '.join('?' * len(row))})", list(row.values())
            )
            if cursor.rowcount:
                inserted += 1
                if is_instance:
                    self.db.execute(
                        "INSERT OR IGNORE INTO instance_triples (triple_id, entity_id, concept_label) VALUES (?, ?, ?)",
                        (triple_id, entity["id"], obj),
                    )
                if row["object_id"]:
                    self.dirty_entities.add(row["object_id"])
        return inserted

    # ---- NER ---------------------------------------------------------------

    def _stage_ner(self) -> bool:
        triples = self._rows(
            self.db.execute(
                "SELECT id, subject_id, subject_label, predicate_label, object_label FROM triples "
                "WHERE object_type = 'unrecognized' AND object_status = 'generated' "
                "AND predicate_status IN ('pd_finished', 'finished') ORDER BY seq LIMIT ?",
                (self.config.fetch_size,),
            )
        )
        if not triples:
            return False
        self._current(f"NER: {len(triples)} triple objects")
        language = self.config.language
        nes = literals = 0
        for triple, output, error in self._fan_out(
            "ner", triples, lambda t: P.ner_messages(language, t["subject_label"], t["predicate_label"], t["object_label"])
        ):
            try:
                if error:
                    raise error
                is_ne = P.parse_ner(output or "")
            except Exception as exc:
                self._triple_failed(
                    triple, "ner", str(exc), output,
                    {"object_type": S.TYPE_LITERAL, "object_status": S.OBJ_FINISHED, "resolution": "failed"},
                )
                self.db.commit()
                continue
            if is_ne:
                self._update_triple(triple["id"], object_type=S.TYPE_NE, object_status=S.OBJ_NER_FINISHED)
                nes += 1
            else:
                self._update_triple(triple["id"], object_type=S.TYPE_LITERAL, object_status=S.OBJ_FINISHED)
                literals += 1
            self.db.commit()
        self._log(f"NER: {nes} named entities and {literals} literals among {len(triples)} triple objects.")
        return True

    # ---- NED1 (triple-context NED) ----------------------------------------

    def _stage_ned1(self) -> bool:
        triples = self._rows(
            self.db.execute(
                "SELECT id, seq, subject_id, subject_label, predicate_id, predicate_label, object_label FROM triples "
                "WHERE object_type = 'ne' AND object_status = 'ner_finished' ORDER BY seq LIMIT ?",
                (self.config.fetch_size,),
            )
        )
        if not triples:
            return False

        cache = self._cached_object_ids({(t["predicate_label"], t["object_label"]) for t in triples})
        not_cached = []
        cached = 0
        for triple in triples:
            object_id = cache.get((triple["predicate_label"], triple["object_label"]))
            if object_id:
                self._resolve_object(triple, object_id, "cache")
                cached += 1
            else:
                not_cached.append(triple)
        self.db.commit()
        if cached:
            self._log(f"NED1: {cached} objects resolved from the cache.")
        if not not_cached:
            return True

        # Guarded parallelization: cluster the pending object labels; from
        # each cluster only its earliest label is disambiguated this round --
        # all of its triples (one per subject) if that label was already
        # resolved more than `frequency_threshold` times, else just one -- so
        # synonymous mentions never race each other into duplicate entities.
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for triple in not_cached:
            by_label[triple["object_label"]].append(triple)
        labels = list(by_label)
        ned_counts: dict[str, int] = {}
        for start in range(0, len(labels), 500):
            chunk = labels[start : start + 500]
            for label, count in self.db.execute(
                f"SELECT object_label, COUNT(*) FROM triples WHERE object_type = 'ne' AND object_id IS NOT NULL "
                f"AND object_label IN ({','.join('?' * len(chunk))}) GROUP BY object_label",
                chunk,
            ):
                ned_counts[label] = count
        label_vectors = self.embedder.embed(labels)
        assignment = cluster_labels(label_vectors, self.config.cluster_distance)
        clusters: dict[int, list[str]] = {}
        for label, cluster_id in zip(labels, assignment):
            clusters.setdefault(cluster_id, []).append(label)
        selected: list[dict[str, Any]] = []
        for cluster in clusters.values():
            first = cluster[0]
            if ned_counts.get(first, 0) > self.config.frequency_threshold:
                subjects: set[str] = set()
                for triple in by_label[first]:
                    if triple["subject_id"] not in subjects:
                        selected.append(triple)
                        subjects.add(triple["subject_id"])
            else:
                selected.append(by_label[first][0])

        for start in range(0, len(selected), self.config.chunk_size):
            self._ned1_chunk(selected[start : start + self.config.chunk_size])
        return True

    def _ned1_chunk(self, triples: list[dict[str, Any]]) -> None:
        self._current(f"NED1: {', '.join(t['object_label'] for t in triples[:3])}" + (f" (+{len(triples) - 3} more)" if len(triples) > 3 else ""))
        vectors = self.embedder.embed([t["object_label"] for t in triples])
        neighbor_ids = self.entity_index.search(vectors, self.config.ned_candidates, self.config.ned_similarity_floor)
        entity_info = self._entity_rows(sorted({i for ids in neighbor_ids for i in ids}))
        candidates: dict[str, list[str]] = {}
        for triple, ids in zip(triples, neighbor_ids):
            ids = [i for i in ids if i in entity_info]
            candidates[triple["id"]] = ids
            self._set_candidates(triple["id"], "ned1", ids)
        self.db.commit()

        for triple in [t for t in triples if not candidates[t["id"]]]:
            self._update_triple(triple["id"], object_status=S.OBJ_NED1_FINISHED)
        triples = [t for t in triples if candidates[t["id"]]]

        def build(t: dict[str, Any]) -> tuple[str, str]:
            ids = candidates[t["id"]]
            return P.ned1_messages(
                t["subject_label"], t["predicate_label"], t["object_label"],
                [entity_info[i][0] for i in ids], [entity_info[i][1] for i in ids],
            )

        merged = new = unsure = 0
        for triple, output, error in self._fan_out("ned", triples, build):
            ids = candidates[triple["id"]]
            context = f"[{triple['subject_label']}, {triple['predicate_label']}, {triple['object_label']}]"
            if error:
                self._triple_failed(triple, "ned1", str(error), None,
                                    {"object_status": S.OBJ_FINISHED, "resolution": "failed"})
                self.db.commit()
                continue
            choice = P.parse_choice(output or "", **P.ned_parse_options("ned1", len(ids)))
            if choice == "unsure":
                self._update_triple(triple["id"], object_status=S.OBJ_FINISHED, resolution="unsure")
                self._log_decision(triple["id"], "ned1", triple["object_label"], context, ids, output, "unsure", None)
                unsure += 1
            elif isinstance(choice, int):
                if 1 <= choice <= len(ids):
                    chosen = ids[choice - 1]
                    self._resolve_object(triple, chosen, "ned1")
                    self._log_decision(triple["id"], "ned1", triple["object_label"], context, ids, output, "merged", chosen)
                    merged += 1
                else:
                    self._triple_failed(triple, "ned1", f"answer {output!r} is not one of the {len(ids)} candidates", output,
                                        {"object_status": S.OBJ_FINISHED, "resolution": "failed"})
            else:
                self._update_triple(triple["id"], object_status=S.OBJ_NED1_FINISHED)
                self._log_decision(triple["id"], "ned1", triple["object_label"], context, ids, output, "none", None)
                new += 1
            self.db.commit()
        self._log(f"NED1: {merged} objects matched existing entities, {new} go to description generation, {unsure} unsure.")

    # ---- NEDG / NED2 (description-context NED) -----------------------------

    def _stage_nedg(self) -> bool:
        triples = self._rows(
            self.db.execute(
                "SELECT id, subject_id, subject_label, predicate_label, object_label FROM triples "
                "WHERE object_type = 'ne' AND object_status = 'ned_finished1' ORDER BY seq LIMIT ?",
                (max(self.config.chunk_size, self.config.nthreads),),
            )
        )
        if not triples:
            return False
        self._current(f"Describing: {', '.join(t['object_label'] for t in triples[:3])}" + (f" (+{len(triples) - 3} more)" if len(triples) > 3 else ""))
        done = 0
        for triple, output, error in self._fan_out(
            "ned", triples,
            lambda t: P.nedg_messages(t["subject_label"], t["predicate_label"], t["object_label"], self.config.topic),
        ):
            try:
                if error:
                    raise error
                description = P.parse_description(output or "")
            except Exception as exc:
                self._triple_failed(triple, "nedg", str(exc), output,
                                    {"object_status": S.OBJ_FINISHED, "resolution": "failed"})
                self.db.commit()
                continue
            self._update_triple(triple["id"], object_description=description, object_status=S.OBJ_DG_FINISHED)
            self.db.commit()
            done += 1
        self._log(f"Description generation: {done} new-entity candidates described.")
        return True

    def _stage_ned2(self) -> bool:
        triples = self._rows(
            self.db.execute(
                "SELECT id, subject_id, subject_label, predicate_label, object_label, object_description FROM triples "
                "WHERE object_type = 'ne' AND object_status = 'dg_finished' AND object_description IS NOT NULL "
                "ORDER BY seq LIMIT ?",
                (max(self.config.chunk_size, self.config.nthreads),),
            )
        )
        if not triples:
            return False
        self._current(f"NED2: {', '.join(t['object_label'] for t in triples[:3])}" + (f" (+{len(triples) - 3} more)" if len(triples) > 3 else ""))
        k = self.config.ned_candidates

        # Entities whose label -- or, with aliases indexed as in the paper,
        # one of whose aliases -- is exactly the object label.
        labels = sorted({t["object_label"] for t in triples})
        placeholders = ",".join("?" * len(labels))
        same_label: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for matched, entity_id, label, description in self.db.execute(
            f"SELECT label, id, label, description FROM entities WHERE label IN ({placeholders}) "
            f"UNION SELECT a.alias, e.id, e.label, e.description FROM entity_aliases a JOIN entities e "
            f"ON e.id = a.entity_id WHERE a.alias IN ({placeholders})",
            [*labels, *labels],
        ):
            if all(existing[0] != entity_id for existing in same_label[matched]):
                same_label[matched].append((entity_id, label, description))

        candidates: dict[str, list[str]] = {}
        info: dict[str, tuple[str, str]] = {}
        dense = [t for t in triples if len(same_label.get(t["object_label"], [])) > k]
        sparse = [t for t in triples if len(same_label.get(t["object_label"], [])) <= k]

        # High-density homonyms (more same-label entities than candidate
        # slots, e.g. "Faculty of Law"): rank the same-label entities by
        # description similarity.
        for triple in dense:
            group = same_label[triple["object_label"]]
            target = self.embedder.embed([triple["object_description"]])[0]
            scores = self.embedder.embed([e[2] for e in group]) @ target
            order = np.argsort(-scores, kind="stable")[:k]
            candidates[triple["id"]] = [group[i][0] for i in order]
            for i in order:
                info[group[i][0]] = (group[i][1], group[i][2])

        # Otherwise: every same-label entity, topped up to k by label similarity.
        if sparse:
            neighbor_ids = self.entity_index.search(
                self.embedder.embed([t["object_label"] for t in sparse]), k, self.config.ned_similarity_floor
            )
            neighbor_info = self._entity_rows(sorted({i for ids in neighbor_ids for i in ids}))
            for triple, ids in zip(sparse, neighbor_ids):
                chosen = [e[0] for e in same_label.get(triple["object_label"], [])]
                for e in same_label.get(triple["object_label"], []):
                    info[e[0]] = (e[1], e[2])
                for entity_id in ids:
                    if entity_id in neighbor_info and entity_id not in chosen and len(chosen) < k:
                        chosen.append(entity_id)
                        info[entity_id] = neighbor_info[entity_id]
                candidates[triple["id"]] = chosen
        for triple in triples:
            self._set_candidates(triple["id"], "ned2", candidates[triple["id"]])
        self.db.commit()
        new_entity_triples: list[dict[str, Any]] = [t for t in triples if not candidates[t["id"]]]
        triples = [t for t in triples if candidates[t["id"]]]

        def build(t: dict[str, Any]) -> tuple[str, str]:
            ids = candidates[t["id"]]
            return P.ned2_messages(t["object_label"], t["object_description"], [info[i][0] for i in ids], [info[i][1] for i in ids])

        merged = 0
        for triple, output, error in self._fan_out("ned", triples, build):
            ids = candidates[triple["id"]]
            context = triple["object_description"]
            if error:
                self._triple_failed(triple, "ned2", str(error), None,
                                    {"object_status": S.OBJ_FINISHED, "resolution": "failed"})
                self.db.commit()
                continue
            choice = P.parse_choice(output or "", **P.ned_parse_options("ned2", len(ids)))
            if isinstance(choice, int):
                if 1 <= choice <= len(ids):
                    chosen = ids[choice - 1]
                    self._resolve_object(triple, chosen, "ned2")
                    self._log_decision(triple["id"], "ned2", triple["object_label"], context, ids, output, "merged", chosen)
                    merged += 1
                else:
                    self._triple_failed(triple, "ned2", f"answer {output!r} is not one of the {len(ids)} candidates", output,
                                        {"object_status": S.OBJ_FINISHED, "resolution": "failed"})
                self.db.commit()
            else:
                self._log_decision(triple["id"], "ned2", triple["object_label"], context, ids, output, "new", None)
                new_entity_triples.append(triple)

        created = self._commit_new_entities(new_entity_triples)
        self.db.commit()
        self._log(f"NED2: {merged} objects matched existing entities, {created} new entities added.")
        return True

    def _commit_new_entities(self, triples: list[dict[str, Any]]) -> int:
        """New entities for the NED2 "None of above" objects; a (label,
        description) pair that already exists -- or was just created for an
        earlier triple of the same batch -- is reused instead."""
        if not triples:
            return 0
        created_ids: list[str] = []
        created_labels: list[str] = []
        for triple in triples:
            label, description = triple["object_label"], triple["object_description"]
            existing = self.db.execute(
                "SELECT id FROM entities WHERE label = ? AND description = ?", (label, description)
            ).fetchone()
            if existing:
                entity_id = existing[0]
                resolution = "ned2"
            else:
                entity_id, seq = self._next_id("E", "next_entity")
                parent = self.db.execute("SELECT depth FROM entities WHERE id = ?", (triple["subject_id"],)).fetchone()
                depth = (parent[0] if parent else 0) + 1
                self.db.execute(
                    "INSERT INTO entities (id, seq, label, description, status, depth, first_appeared) "
                    "VALUES (?, ?, ?, ?, 'unexplored', ?, ?)",
                    (entity_id, seq, label, description, depth, triple["id"]),
                )
                created_ids.append(entity_id)
                created_labels.append(label)
                resolution = "new"
            self._resolve_object(triple, entity_id, resolution)
            self.db.execute(
                "UPDATE decisions SET chosen_id = ? WHERE triple_id = ? AND stage = 'ned2' AND outcome = 'new'",
                (entity_id, triple["id"]),
            )
        if created_ids:
            self.entity_index.add(created_ids, self.embedder.embed(created_labels))
        return len(created_ids)

    # ---- PD / PDG (relations) ----------------------------------------------

    def _stage_pd(self) -> bool:
        triples = self._rows(
            self.db.execute(
                "SELECT id, subject_id, subject_label, predicate_label, object_label FROM triples "
                "WHERE predicate_status = 'generated' ORDER BY seq LIMIT ?",
                (self.config.fetch_size,),
            )
        )
        if not triples:
            return False

        # Cache: a label any triple's predicate was already resolved for is
        # mapped directly (GPTKB 2.0's PD auto-map), deleting triples that
        # would then duplicate an existing (subject, predicate, object label).
        labels = sorted({t["predicate_label"] for t in triples})
        auto_map: dict[str, str] = {}
        for label, predicate_id in self.db.execute(
            f"SELECT predicate_label, predicate_id FROM triples WHERE predicate_id IS NOT NULL "
            f"AND predicate_label IN ({','.join('?' * len(labels))})",
            labels,
        ):
            if len(label) > 1:
                auto_map[label] = predicate_id
        mapped = deleted = 0
        remaining = []
        for triple in triples:
            predicate_id = auto_map.get(triple["predicate_label"])
            if predicate_id is None:
                remaining.append(triple)
                continue
            collision = self.db.execute(
                "SELECT 1 FROM triples WHERE subject_id = ? AND predicate_id = ? AND object_label = ? LIMIT 1",
                (triple["subject_id"], predicate_id, triple["object_label"]),
            ).fetchone()
            if collision:
                self._delete_triple(triple["id"])
                deleted += 1
                continue
            if self._set_predicate(triple, predicate_id):
                mapped += 1
            else:
                deleted += 1
        self.db.commit()
        if mapped or deleted:
            self._log(f"PD cache: {mapped} relations mapped, {deleted} duplicate triples removed.")
        if not remaining:
            return True

        # One triple per label (per 5-character prefix while the KB has few
        # relations) goes to PD per round; the rest follow via the cache.
        predicate_count = self.db.execute("SELECT COUNT(*) FROM predicates").fetchone()[0]
        selected, keys = [], set()
        for triple in remaining:
            label = str(triple["predicate_label"])
            key = label.lower().strip() if predicate_count > 300 else label[:5].lower().strip()
            if key not in keys:
                keys.add(key)
                selected.append(triple)

        for start in range(0, len(selected), self.config.chunk_size):
            chunk = selected[start : start + self.config.chunk_size]
            self._current(f"Relation disambiguation: {', '.join(t['predicate_label'] for t in chunk[:3])}")
            neighbor_ids = self.predicate_index.search(
                self.embedder.embed([t["predicate_label"] for t in chunk]), self.config.num_candidates
            )
            predicate_info = {
                row[0]: (row[1], row[2])
                for row in self.db.execute(
                    f"SELECT id, label, description FROM predicates WHERE id IN "
                    f"({','.join('?' * len({i for ids in neighbor_ids for i in ids}))})",
                    sorted({i for ids in neighbor_ids for i in ids}),
                )
            } if any(neighbor_ids) else {}
            candidates = {t["id"]: [i for i in ids if i in predicate_info] for t, ids in zip(chunk, neighbor_ids)}
            for triple in chunk:
                self._set_candidates(triple["id"], "pd", candidates[triple["id"]])
                if not candidates[triple["id"]]:
                    self._update_triple(triple["id"], predicate_status=S.PRED_PD_FINISHED)
            self.db.commit()
            chunk = [t for t in chunk if candidates[t["id"]]]

            def build(t: dict[str, Any]) -> tuple[str, str]:
                ids = candidates[t["id"]]
                return P.pd_messages(t["subject_label"], t["predicate_label"], t["object_label"],
                                     [predicate_info[i][0] for i in ids], [predicate_info[i][1] for i in ids])

            matched = new = 0
            for triple, output, error in self._fan_out("ned", chunk, build):
                ids = candidates[triple["id"]]
                context = f"[{triple['subject_label']}, {triple['predicate_label']}, {triple['object_label']}]"
                if error:
                    self._predicate_failed(triple, "pd", str(error), None)
                    self.db.commit()
                    continue
                choice = P.parse_choice(output or "", "A-G")
                if isinstance(choice, int):
                    if 1 <= choice <= len(ids):
                        chosen = ids[choice - 1]
                        self._set_predicate(triple, chosen)
                        self._log_decision(triple["id"], "pd", triple["predicate_label"], context, ids, output, "merged", chosen)
                        matched += 1
                    else:
                        self._predicate_failed(triple, "pd", f"answer {output!r} is not one of the {len(ids)} candidates", output)
                else:
                    self._update_triple(triple["id"], predicate_status=S.PRED_PD_FINISHED)
                    self._log_decision(triple["id"], "pd", triple["predicate_label"], context, ids, output, "none", None)
                    new += 1
                self.db.commit()
            self._log(f"Relation disambiguation: {matched} matched existing relations, {new} need a new relation.")
        return True

    def _set_predicate(self, triple: dict[str, Any], predicate_id: str) -> bool:
        if not self._update_triple(triple["id"], predicate_id=predicate_id, predicate_status=S.PRED_FINISHED):
            return False
        label = self.db.execute("SELECT label FROM predicates WHERE id = ?", (predicate_id,)).fetchone()
        if label and label[0] != triple["predicate_label"]:
            self.db.execute(
                "INSERT OR IGNORE INTO predicate_aliases (predicate_id, alias) VALUES (?, ?)",
                (predicate_id, triple["predicate_label"]),
            )
        if predicate_id == "P0":
            # A relation disambiguated to instanceOf makes the object a class.
            self._update_triple(triple["id"], object_type=S.TYPE_CLASS, object_status=S.OBJ_FINISHED)
            self.db.execute(
                "INSERT OR IGNORE INTO instance_triples (triple_id, entity_id, concept_label) VALUES (?, ?, ?)",
                (triple["id"], triple["subject_id"], triple["object_label"]),
            )
        self.dirty_entities.add(triple["subject_id"])
        return True

    def _predicate_failed(self, triple: dict[str, Any], stage: str, error: str, detail: str | None) -> None:
        if self._bump_attempts("triples", "id", triple["id"]) >= self.config.max_attempts:
            # Give the triple a relation of its own rather than drop it.
            self._update_triple(triple["id"], predicate_status=S.PRED_PD_FINISHED, attempts=0)
            self._record_failure(triple["predicate_label"], stage, error, detail)

    def _stage_pdg(self) -> bool:
        triples = self._rows(
            self.db.execute(
                "SELECT id, subject_id, subject_label, predicate_label, object_label FROM triples "
                "WHERE predicate_status = 'pd_finished' ORDER BY seq LIMIT ?",
                (max(self.config.chunk_size, self.config.nthreads),),
            )
        )
        if not triples:
            return False
        self._current(f"Describing relations: {', '.join(t['predicate_label'] for t in triples[:3])}")
        described: list[tuple[dict[str, Any], str]] = []
        for triple, output, error in self._fan_out(
            "ned", triples, lambda t: P.pdg_messages(t["predicate_label"], self.config.topic)
        ):
            try:
                if error:
                    raise error
                described.append((triple, P.parse_description(output or "")))
            except Exception as exc:
                if self._bump_attempts("triples", "id", triple["id"]) >= self.config.max_attempts:
                    described.append((triple, f"Indicates the relation '{triple['predicate_label']}'."))
                    self._record_failure(triple["predicate_label"], "pdg", str(exc), output)
                self.db.commit()
        created = self._commit_new_elements(
            "predicates", "P", "next_predicate", self.predicate_index,
            [(t, t["predicate_label"], d) for t, d in described],
            lambda t, element_id: self._set_predicate(t, element_id),
        )
        self.db.commit()
        self._log(f"Relation description generation: {created} new relations.")
        return True

    def _commit_new_elements(
        self,
        table: str,
        prefix: str,
        counter: str,
        index: VectorIndex,
        items: list[tuple[dict[str, Any], str, str]],
        assign: Callable[[dict[str, Any], str], Any],
    ) -> int:
        """New predicates/classes for (item, label, description) triples,
        reusing an existing element with the same label and description."""
        new_ids, new_labels = [], []
        for item, label, description in items:
            existing = self.db.execute(
                f"SELECT id FROM {table} WHERE label = ? AND description = ?", (label, description)
            ).fetchone()
            if existing:
                element_id = existing[0]
            else:
                element_id, seq = self._next_id(prefix, counter)
                first = item.get("id") or item.get("triple_id")
                self.db.execute(
                    f"INSERT INTO {table} (id, seq, label, description, first_appeared) VALUES (?, ?, ?, ?, ?)",
                    (element_id, seq, label, description, first),
                )
                new_ids.append(element_id)
                new_labels.append(label)
            assign(item, element_id)
        if new_ids:
            index.add(new_ids, self.embedder.embed(new_labels))
        return len(new_ids)

    # ---- CD / CDG (classes) -------------------------------------------------

    def _stage_cd(self) -> bool:
        items = self._rows(
            self.db.execute(
                "SELECT it.triple_id, it.entity_id, it.concept_label, t.subject_label, t.predicate_label "
                "FROM instance_triples it JOIN triples t ON t.id = it.triple_id "
                "WHERE it.status = 'generated' ORDER BY t.seq LIMIT ?",
                (self.config.fetch_size,),
            )
        )
        if not items:
            return False

        labels = sorted({i["concept_label"] for i in items})
        auto_map = {
            label: concept_id
            for label, concept_id in self.db.execute(
                f"SELECT concept_label, concept_id FROM instance_triples WHERE concept_id IS NOT NULL "
                f"AND concept_label IN ({','.join('?' * len(labels))})",
                labels,
            )
        }
        mapped = deleted = 0
        remaining = []
        for item in items:
            concept_id = auto_map.get(item["concept_label"])
            if concept_id is None:
                remaining.append(item)
                continue
            collision = self.db.execute(
                "SELECT 1 FROM instance_triples WHERE entity_id = ? AND concept_id = ? LIMIT 1",
                (item["entity_id"], concept_id),
            ).fetchone()
            if collision:
                self._delete_triple(item["triple_id"])
                deleted += 1
            else:
                self._set_concept(item, concept_id)
                mapped += 1
        self.db.commit()
        if mapped or deleted:
            self._log(f"Class cache: {mapped} classes mapped, {deleted} duplicate class triples removed.")
        if not remaining:
            return True

        concept_count = self.db.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        selected, keys = [], set()
        for item in remaining:
            label = str(item["concept_label"])
            key = label.lower().strip() if concept_count > 100 else label[:3].lower().strip()
            if key not in keys:
                keys.add(key)
                selected.append(item)

        for start in range(0, len(selected), self.config.chunk_size):
            chunk = selected[start : start + self.config.chunk_size]
            self._current(f"Class disambiguation: {', '.join(i['concept_label'] for i in chunk[:3])}")
            neighbor_ids = self.concept_index.search(
                self.embedder.embed([i["concept_label"] for i in chunk]), self.config.num_candidates
            )
            all_ids = sorted({i for ids in neighbor_ids for i in ids})
            concept_info = {
                row[0]: (row[1], row[2])
                for row in self.db.execute(
                    f"SELECT id, label, description FROM concepts WHERE id IN ({','.join('?' * len(all_ids))})", all_ids
                )
            } if all_ids else {}
            candidates = {i["triple_id"]: [c for c in ids if c in concept_info] for i, ids in zip(chunk, neighbor_ids)}
            for item in chunk:
                self._set_candidates(item["triple_id"], "cd", candidates[item["triple_id"]])
                if not candidates[item["triple_id"]]:
                    self.db.execute(
                        "UPDATE instance_triples SET status = 'cd_finished' WHERE triple_id = ?", (item["triple_id"],)
                    )
            self.db.commit()
            chunk = [i for i in chunk if candidates[i["triple_id"]]]

            def build(i: dict[str, Any]) -> tuple[str, str]:
                ids = candidates[i["triple_id"]]
                return P.cd_messages(i["subject_label"], i["predicate_label"], i["concept_label"],
                                     [concept_info[c][0] for c in ids], [concept_info[c][1] for c in ids])

            matched = new = 0
            for item, output, error in self._fan_out("ned", chunk, build):
                ids = candidates[item["triple_id"]]
                context = f"[{item['subject_label']}, {item['predicate_label']}, {item['concept_label']}]"
                if error:
                    self._concept_failed(item, "cd", str(error), None)
                    self.db.commit()
                    continue
                choice = P.parse_choice(output or "", "A-G")
                if isinstance(choice, int):
                    if 1 <= choice <= len(ids):
                        chosen = ids[choice - 1]
                        self._set_concept(item, chosen)
                        self._log_decision(item["triple_id"], "cd", item["concept_label"], context, ids, output, "merged", chosen)
                        matched += 1
                    else:
                        self._concept_failed(item, "cd", f"answer {output!r} is not one of the {len(ids)} candidates", output)
                else:
                    self.db.execute(
                        "UPDATE instance_triples SET status = 'cd_finished' WHERE triple_id = ?", (item["triple_id"],)
                    )
                    self._log_decision(item["triple_id"], "cd", item["concept_label"], context, ids, output, "none", None)
                    new += 1
                self.db.commit()
            self._log(f"Class disambiguation: {matched} matched existing classes, {new} need a new class.")
        return True

    def _set_concept(self, item: dict[str, Any], concept_id: str) -> None:
        self.db.execute(
            "UPDATE instance_triples SET concept_id = ?, status = 'finished' WHERE triple_id = ?",
            (concept_id, item["triple_id"]),
        )
        label = self.db.execute("SELECT label FROM concepts WHERE id = ?", (concept_id,)).fetchone()
        if label and label[0] != item["concept_label"]:
            self.db.execute(
                "INSERT OR IGNORE INTO concept_aliases (concept_id, alias) VALUES (?, ?)", (concept_id, item["concept_label"])
            )
        self.dirty_entities.add(item["entity_id"])

    def _concept_failed(self, item: dict[str, Any], stage: str, error: str, detail: str | None) -> None:
        if self._bump_attempts("instance_triples", "triple_id", item["triple_id"]) >= self.config.max_attempts:
            self.db.execute(
                "UPDATE instance_triples SET status = 'cd_finished', attempts = 0 WHERE triple_id = ?", (item["triple_id"],)
            )
            self._record_failure(item["concept_label"], stage, error, detail)

    def _stage_cdg(self) -> bool:
        items = self._rows(
            self.db.execute(
                "SELECT triple_id, entity_id, concept_label FROM instance_triples WHERE status = 'cd_finished' "
                "ORDER BY rowid LIMIT ?",
                (max(self.config.chunk_size, self.config.nthreads),),
            )
        )
        if not items:
            return False
        self._current(f"Describing classes: {', '.join(i['concept_label'] for i in items[:3])}")
        described: list[tuple[dict[str, Any], str]] = []
        for item, output, error in self._fan_out(
            "ned", items, lambda i: P.cdg_messages(i["concept_label"], self.config.topic)
        ):
            try:
                if error:
                    raise error
                described.append((item, P.parse_description(output or "")))
            except Exception as exc:
                if self._bump_attempts("instance_triples", "triple_id", item["triple_id"]) >= self.config.max_attempts:
                    described.append((item, f"The class of {item['concept_label']}."))
                    self._record_failure(item["concept_label"], "cdg", str(exc), output)
                self.db.commit()
        created = self._commit_new_elements(
            "concepts", "C", "next_concept", self.concept_index,
            [(i, i["concept_label"], d) for i, d in described],
            lambda i, element_id: self._set_concept(i, element_id),
        )
        self.db.commit()
        self._log(f"Class description generation: {created} new classes.")
        return True
