from __future__ import annotations

import asyncio
import dataclasses
import io
import os
import math
import mimetypes
import json
import logging
import multiprocessing as mp
import queue
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile

# Python's mimetypes module doesn't know ".db" -- without this, StaticFiles
# (which serves knowledgeBase.db straight out of a run's directory) would
# label the download "text/plain" instead of a binary type.
mimetypes.add_type("application/vnd.sqlite3", ".db")
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from app.runner import (
    KBCConfig,
    LocalKBCRunner,
    OPENAI_TEXT_PRICING,
    OPENROUTER_REASONING_CAPABILITY,
    PROMPT_LANGUAGES,
    append_subject_line,
    build_elicitation_prompt,
    build_elicitation_prompt_body,
    chunked,
    get_reasoning_capability,
    load_subject_state,
)
from app.disambig.runner import (
    DEFAULT_EMBEDDING_MODEL,
    SCADS_EMBEDDING_URL,
    DisambigConfig,
    DisambiguatedKBCRunner,
)
from app.disambig.store import progress_counts


ROOT = Path(__file__).resolve().parent.parent
# Overridable so a second instance (e.g. a test server on another port) can
# run against its own runs directory.
RUNS_DIR = Path(os.environ.get("MINIGPTKB_RUNS_DIR") or ROOT / "runs")
STATIC_DIR = ROOT / "app" / "static"

SCADS_ENDPOINT = "https://llm.scads.ai/v1"
# Project-provisioned ScaDS AI key, used automatically for the ScaDS endpoint
# whenever a request doesn't supply its own -- never sent to the frontend,
# never persisted in a run's config (see _safe_run_config), and only ever
# used for requests actually targeting SCADS_ENDPOINT. Read from the
# environment (on the server: /etc/minigptkb/env, loaded by the systemd
# unit) so it isn't in the source code.
DEFAULT_SCADS_API_KEY = os.environ.get("SCADS_API_KEY", "").strip()
if not DEFAULT_SCADS_API_KEY:
    logging.getLogger("uvicorn.error").warning(
        "SCADS_API_KEY is not set: ScaDS.AI requests without a user-supplied key, and all embeddings, will fail."
    )
# Leading underscore keeps this out of list_runs()'s directory scan, which
# already skips any RUNS_DIR entry starting with "_".
FOLDERS_PATH = RUNS_DIR / "_folders.json"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
MP_CONTEXT = mp.get_context("spawn")

# Every new run uses the GPTKB 2.0 disambiguation pipeline (app/disambig).
# Runs created before it have no "pipeline" in their runConfig.json and are
# "classic" runs: they stay browsable, and resuming one continues it with
# the original string-identity runner (app/runner.py) so its data stays
# consistent.
CURRENT_PIPELINE = "gptkb2"
CLASSIC_PIPELINE = "classic"


def _pipeline_of(config: Any) -> str:
    if isinstance(config, dict) and config.get("pipeline"):
        return str(config["pipeline"])
    return CLASSIC_PIPELINE

app = FastAPI(title="miniGPTKB UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


class RunRequest(BaseModel):
    llm_url: str = Field(default="https://llm.scads.ai/v1")
    api_key: str = Field(default="")
    # The elicitation model: no default, always chosen by the user.
    model: str = Field(min_length=1)
    # Models for named entity recognition and for disambiguation (NED, plus
    # relation/class disambiguation and description generation); empty means
    # "same as the elicitation model".
    ner_model: str = Field(default="")
    ned_model: str = Field(default="")
    # Endpoints of the NER/NED models: empty (or the elicitation endpoint)
    # means the elicitation endpoint with its key; the only other allowed
    # endpoint is ScaDS.AI, with the shared key.
    ner_llm_url: str = Field(default="")
    ned_llm_url: str = Field(default="")
    topic: str = Field(default="the ancient city of Babylon")
    seed_entity: str = Field(default="Hammurabi")
    language: str = Field(default="en")
    schema_mode: str = Field(default="open")
    schema_text: str = Field(default="")
    prompt_template: str = Field(default="")
    facts_multiplier: float = Field(default=1.0, ge=0.25, le=3.0)
    max_entities: int | None = Field(default=None, ge=1)
    max_bfs_depth: int | None = Field(default=None, ge=0)
    max_runtime_seconds: int | None = Field(default=None, ge=1)
    max_cost: float | None = Field(default=None, ge=0)
    nthreads: int = Field(default=5, ge=1, le=50)
    # Entity merging: candidates shown per NED decision, and the minimum
    # label similarity for a candidate (0 = GPTKB 2.0).
    ned_candidates: int = Field(default=5, ge=1, le=10)
    ned_similarity_floor: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_requests_per_minute: int | None = Field(default=None, ge=1)
    folder_id: str | None = Field(default=None)
    new_folder_name: str | None = Field(default=None)
    reasoning_effort: str | None = Field(default=None)
    reasoning_enabled: bool | None = Field(default=None)
    # API only (not in the UI): "classic" starts a run with the original
    # string-identity pipeline (app/runner.py), e.g. as a baseline. Ignored on
    # resume.
    pipeline: Literal["gptkb2", "classic"] = Field(default=CURRENT_PIPELINE)
    # API only: False sends requests without per-model pacing (for comparisons).
    pacing: bool = Field(default=True)
    # Explicitly no runtime limit, also on the shared ScaDS.AI key (whose
    # runs otherwise stop after FREE_KEY_RUNTIME_SECONDS).
    no_time_limit: bool = Field(default=False)


class FolderRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class RunFolderUpdate(BaseModel):
    folder_id: str | None = Field(default=None)


class BulkDeleteRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


class BulkMoveRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)
    folder_id: str | None = Field(default=None)


class ModelsRequest(BaseModel):
    llm_url: str
    api_key: str = Field(default="")


class ReasoningCapabilityRequest(BaseModel):
    llm_url: str
    model: str
    api_key: str = Field(default="")


class PromptPreviewRequest(BaseModel):
    language: str = Field(default="en")
    topic: str = Field(default="")
    facts_multiplier: float = Field(default=1.0, ge=0.25, le=3.0)
    schema_mode: str = Field(default="open")
    schema_text: str = Field(default="")


class EndpointRateLimiter:
    """Per-client sliding-window rate limiter for FastAPI routes.

    Keyed by client host rather than a global counter so one heavy client
    (e.g. a misbehaving browser tab) can't starve requests from another.
    """

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_calls:
                retry_after = max(1, round(self.window_seconds - (now - hits[0])))
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please slow down.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)


def _resolve_api_key(llm_url: str, api_key: str) -> str:
    """Fills in the project's default ScaDS key when a request targets the
    ScaDS endpoint and didn't supply its own -- any other endpoint, or a
    user-supplied key, passes through unchanged.
    """
    if api_key.strip():
        return api_key
    if llm_url.strip().rstrip("/").lower() == SCADS_ENDPOINT:
        return _require_scads_key()
    return api_key


def _require_scads_key() -> str:
    if not DEFAULT_SCADS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="No shared ScaDS.AI key is configured on this server (SCADS_API_KEY); enter your own API key.",
        )
    return DEFAULT_SCADS_API_KEY


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limited(limiter: EndpointRateLimiter):
    def dependency(request: Request) -> None:
        limiter.check(_client_key(request))

    return dependency


# Endpoints that trigger real work (spawning a run, calling an external LLM
# API for model listing) get tighter caps than the ones the UI just polls.
_run_control_limiter = EndpointRateLimiter(max_calls=10, window_seconds=60)
_models_limiter = EndpointRateLimiter(max_calls=10, window_seconds=60)
_prompt_preview_limiter = EndpointRateLimiter(max_calls=60, window_seconds=60)


def _tail(items: "deque[dict[str, Any]]", count: int) -> list[dict[str, Any]]:
    """Last `count` items of a deque, in original order, without scanning the rest."""
    if count <= 0:
        return []
    return list(reversed([event for event, _ in zip(reversed(items), range(count))]))


def _tail_since(items: "deque[dict[str, Any]]", cursor: int) -> list[dict[str, Any]]:
    """Items with id >= cursor, in original order.

    Ids increase with insertion order, so matches are always a suffix of the
    deque -- walk from the right and stop at the first older id instead of
    scanning the whole deque.
    """
    matches: list[dict[str, Any]] = []
    for event in reversed(items):
        if event["id"] < cursor:
            break
        matches.append(event)
    matches.reverse()
    return matches


class RunController:
    """Tracks one run's live process/events. One instance per run_id, created
    once when that run is launched and kept in the `controllers` registry for
    the rest of this server process's lifetime (so late viewers -- including
    a run started by a different user -- can still fetch its status/log
    history). Unlike the single-controller design this replaced, there's no
    reuse/reset across different runs, so no run-generation token is needed
    to guard against a stale event from a previous occupant.
    """

    # "status"/"current_subject" events fire about as often as "log" events
    # (roughly once per processed subject each), but only the most recent one
    # of those actually matters to a client. "log" events are the run's
    # visible history in the UI's log panel, so they're worth keeping much
    # longer -- a shared count-based cap across all event types would let the
    # high-frequency, low-value ones evict the low-frequency, high-value ones
    # once a run passes a few hundred processed subjects.
    MAX_LOG_EVENTS = 20000
    MAX_TRANSIENT_EVENTS = 200

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.id = run_id
        self.run_dir = run_dir
        # Log lines are also appended to the run folder, so a run's log
        # survives resumes and server restarts; earlier lines are loaded back
        # here as the start of this controller's history.
        self.log_path = run_dir / "log.jsonl" if run_dir else None
        self.process: mp.Process | None = None
        self.listener_thread: threading.Thread | None = None
        self.event_queue: mp.Queue = MP_CONTEXT.Queue()
        # Bounded deques evict their oldest entry in O(1) once full, so appending
        # stays cheap no matter how long a run has been going. Splitting by
        # event type means a high-frequency, low-value event (status/
        # current_subject, fired ~once per processed subject) evicting one of
        # its own only ever touches its own small deque, never the much larger,
        # deliberately long-retained log history.
        self._log_events: deque[dict[str, Any]] = deque(maxlen=self.MAX_LOG_EVENTS)
        self._transient_events: deque[dict[str, Any]] = deque(maxlen=self.MAX_TRANSIENT_EVENTS)
        self.next_event_id = 0
        self.lock = threading.Lock()
        self.pause_event = MP_CONTEXT.Event()
        self.stop_event = MP_CONTEXT.Event()
        self.status = "starting"
        self._load_log_history()

    def _load_log_history(self) -> None:
        if not self.log_path or not self.log_path.exists():
            return
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()[-self.MAX_LOG_EVENTS:]
        except OSError:
            return
        for line in lines:
            try:
                stored = json.loads(line)
            except ValueError:
                continue
            self._log_events.append({"id": self.next_event_id, **stored, "type": "log"})
            self.next_event_id += 1

    def _persist_log(self, event: dict[str, Any]) -> None:
        if not self.log_path:
            return
        stored = {key: event[key] for key in ("created_at", "time", "level", "message", "url") if key in event}
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(stored, ensure_ascii=False) + "\n")
        except OSError:
            pass  # the in-memory log still works; only the file copy is missing

    def emit(self, event: dict[str, Any]) -> None:
        with self.lock:
            event = {"id": self.next_event_id, "created_at": time.time(), **event}
            self.next_event_id += 1
            if event.get("type") == "log":
                self._log_events.append(event)
                self._persist_log(event)
            else:
                self._transient_events.append(event)
            if event.get("type") == "status":
                self.status = str(event.get("status", self.status))

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def events_since(self, cursor: int) -> list[dict[str, Any]]:
        """Events with id >= cursor, id-ordered.

        Ids increase with insertion order within each deque, so the matches are
        always a suffix of each -- walk each from the right and stop as soon as
        an older id is hit, rather than scanning the whole (possibly 20000-long)
        log deque on every poll.
        """
        return sorted(
            (*_tail_since(self._log_events, cursor), *_tail_since(self._transient_events, cursor)),
            key=lambda event: event["id"],
        )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            latest_status = next((e for e in reversed(self._transient_events) if e.get("type") == "status"), None)
            recent = sorted((*_tail(self._log_events, 100), *_tail(self._transient_events, 100)), key=lambda event: event["id"])[-100:]
            # started_at is read fresh from disk (rather than cached at
            # __init__) so it's correct even though runConfig.json is written
            # by the caller just before this controller is registered.
            config = _read_json(self.run_dir / "runConfig.json") if self.run_dir else None
            started_at = config.get("started_at") if isinstance(config, dict) else None
            ended_at = config.get("ended_at") if isinstance(config, dict) else None
            return {
                "run_id": self.id,
                "status": self.status,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "html_index_url": f"/runs/{self.id}/html/index.html" if self.id else None,
                "started_at": started_at,
                "ended_at": ended_at,
                "latest": latest_status,
                "events": recent,
            }


# Registry of every run this server process has launched, keyed by run_id, so
# multiple runs -- from the same user or different ones -- can be active at
# once. Entries are never evicted: each holds only a couple of bounded
# deques, and keeping them lets a late viewer still fetch a finished run's
# final status/log history rather than getting a "not found".
_controllers_lock = threading.Lock()
controllers: dict[str, RunController] = {}


def _is_run_active(run_id: str) -> bool:
    controller = controllers.get(run_id)
    return controller is not None and controller.is_alive()


def _run_process_entry(
    config_data: dict[str, Any],
    run_dir: str,
    event_queue: mp.Queue,
    pause_event: Any,
    stop_event: Any,
) -> None:
    def emit(event: dict[str, Any]) -> None:
        event_queue.put(event)

    if config_data.get("pipeline") == CURRENT_PIPELINE:
        runner_class, config_class = DisambiguatedKBCRunner, DisambigConfig
    else:
        runner_class, config_class = LocalKBCRunner, KBCConfig
    field_names = {field.name for field in dataclasses.fields(config_class)}
    runner = runner_class(
        config=config_class(**{key: value for key, value in config_data.items() if key in field_names}),
        run_dir=Path(run_dir),
        emit=emit,
        pause_event=pause_event,
        stop_event=stop_event,
    )
    runner.run()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/languages")
def list_languages() -> dict[str, Any]:
    languages = [
        {"code": code, "name": info["name"]}
        for code, info in sorted(PROMPT_LANGUAGES.items(), key=lambda item: item[1]["name"])
    ]
    return {"languages": languages}


@app.post("/api/prompt-preview", dependencies=[Depends(rate_limited(_prompt_preview_limiter))])
def prompt_preview(request: PromptPreviewRequest) -> dict[str, Any]:
    body = build_elicitation_prompt_body(
        request.language,
        request.topic,
        request.facts_multiplier,
        request.schema_mode,
        request.schema_text,
    )
    return {"prompt": body}


@app.post("/api/models", dependencies=[Depends(rate_limited(_models_limiter))])
def list_models(request: ModelsRequest) -> dict[str, Any]:
    if request.llm_url.strip().rstrip("/").lower() == "https://openrouter.ai/api/v1":
        return _list_openrouter_models(request)

    try:
        api_key = _resolve_api_key(request.llm_url, request.api_key)
        client = OpenAI(base_url=request.llm_url, api_key=api_key, timeout=15.0)
        response = client.models.list()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not list models: {exc}") from exc

    models = sorted({model.id for model in response.data if getattr(model, "id", None)})
    return {"models": models}


# OpenRouter's /models list (several hundred models with their metadata) is
# needed for the model dropdown and for every reasoning-capability check, so
# it is kept for a few minutes instead of being downloaded each time. Failed
# downloads are not cached.
OPENROUTER_MODELS_TTL_SECONDS = 600
_openrouter_models_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_openrouter_models_lock = threading.Lock()


def _openrouter_models(llm_url: str, api_key: str) -> dict[str, Any]:
    url = llm_url.strip().rstrip("/") + "/models"
    with _openrouter_models_lock:
        cached = _openrouter_models_cache.get(url)
    if cached and time.monotonic() - cached[0] < OPENROUTER_MODELS_TTL_SECONDS:
        return cached[1]
    http_request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(http_request, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
    with _openrouter_models_lock:
        _openrouter_models_cache[url] = (time.monotonic(), data)
    return data


def _list_openrouter_models(request: ModelsRequest) -> dict[str, Any]:
    try:
        data = _openrouter_models(request.llm_url, request.api_key)
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not list models: HTTP {exc.code}.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not list models: {exc}") from exc

    models = sorted({str(item.get("id", "")) for item in data.get("data", []) if item.get("id")})
    return {"models": models}


_NO_REASONING_CAPABILITY = {"toggle": None, "effort": None}


@app.post("/api/reasoning-capability", dependencies=[Depends(rate_limited(_models_limiter))])
def reasoning_capability(request: ReasoningCapabilityRequest) -> dict[str, Any]:
    """Whether this model+endpoint has a tunable reasoning effort and/or an
    on/off switch, so the dashboard can show the matching control(s) --
    never a hard error, since a slider/switch failing to appear is a much
    lower-stakes failure than the model list itself failing to load; any
    problem just resolves to "no controls" instead of raising.
    """
    if request.llm_url.strip().rstrip("/").lower() == "https://openrouter.ai/api/v1":
        return _openrouter_reasoning_capability(request)
    try:
        api_key = _resolve_api_key(request.llm_url, request.api_key)
    except HTTPException:
        api_key = ""
    # The first check of a model asks the model itself (a few seconds); the
    # result is cached, so later checks and runs are instant.
    try:
        return get_reasoning_capability(request.llm_url, request.model, api_key) or _NO_REASONING_CAPABILITY
    except Exception:
        return _NO_REASONING_CAPABILITY


def _openrouter_reasoning_capability(request: ReasoningCapabilityRequest) -> dict[str, Any]:
    """OpenRouter reports per-model parameter support directly in its /models
    response (already used for the model dropdown -- see
    _list_openrouter_models) via a "supported_parameters" list, which
    includes "reasoning" for any model where its normalized reasoning API
    actually does something. That one flag is enough here: OpenRouter uses
    the same request shape (OPENROUTER_REASONING_CAPABILITY) for every model
    it proxies, so no per-model registry entry is needed the way ScaDS/
    OpenAI need one.
    """
    try:
        data = _openrouter_models(request.llm_url, request.api_key)
    except Exception:
        return _NO_REASONING_CAPABILITY

    for item in data.get("data", []):
        if str(item.get("id", "")) == request.model:
            supported = item.get("supported_parameters") or []
            if "reasoning" in supported:
                return OPENROUTER_REASONING_CAPABILITY
            break
    return _NO_REASONING_CAPABILITY


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


_folders_lock = threading.Lock()


def _read_folders_unlocked() -> list[dict[str, Any]]:
    data = _read_json(FOLDERS_PATH)
    folders = data.get("folders") if isinstance(data, dict) else None
    return folders if isinstance(folders, list) else []


def _write_folders_unlocked(folders: list[dict[str, Any]]) -> None:
    FOLDERS_PATH.write_text(json.dumps({"folders": folders}, indent=2, ensure_ascii=False), encoding="utf-8")


def _create_folder(name: str) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty.")
    with _folders_lock:
        folders = _read_folders_unlocked()
        folder = {"id": uuid.uuid4().hex[:10], "name": name, "created_at": datetime.now().isoformat(timespec="seconds")}
        folders.append(folder)
        _write_folders_unlocked(folders)
    return folder


def _folder_exists(folder_id: str) -> bool:
    with _folders_lock:
        return any(folder["id"] == folder_id for folder in _read_folders_unlocked())


def _resolve_folder_id(folder_id: str | None, new_folder_name: str | None) -> str | None:
    if new_folder_name and new_folder_name.strip():
        return _create_folder(new_folder_name)["id"]
    if folder_id and _folder_exists(folder_id):
        return folder_id
    return None


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _infer_seed(depths: dict[str, int], processed: list[Any], queued: list[Any], config: dict[str, Any]) -> str:
    configured_seed = str(config.get("seed_entity", "")).strip()
    if configured_seed:
        return configured_seed

    if depths:
        depth_zero = [str(name) for name, depth in depths.items() if depth == 0]
        if depth_zero:
            return sorted(depth_zero)[0]
        try:
            return min(depths.items(), key=lambda item: int(item[1]))[0]
        except Exception:
            pass

    if isinstance(processed, list) and processed:
        return str(processed[0])
    if isinstance(queued, list) and queued:
        return str(queued[0])
    return ""


def _sample_run_terms(
    run_dir: Path, depths: dict[str, int], processed: list[Any], queued: list[Any], limit: int = 250
) -> set[str]:
    terms = {str(item) for item in (processed or [])[:limit] + (queued or [])[:limit]}
    terms.update(str(key) for key in list(depths.keys())[:limit])

    triples_path = run_dir / "triples.jsonl"
    if triples_path.exists():
        with triples_path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index >= limit:
                    break
                try:
                    triple = json.loads(line)
                except json.JSONDecodeError:
                    continue
                terms.add(str(triple.get("s", "")))
                terms.add(str(triple.get("o", "")))
    return {term for term in terms if term}


def _infer_topic(
    run_dir: Path, seed: str, depths: dict[str, int], processed: list[Any], queued: list[Any], config: dict[str, Any]
) -> str:
    configured_topic = str(config.get("topic", "")).strip()
    if configured_topic:
        return configured_topic
    if not seed:
        return ""

    terms = _sample_run_terms(run_dir, depths, processed, queued)
    lower_terms = " ".join(terms).lower()
    seed_lower = seed.lower()

    if seed_lower == "radebeul" or "radebeul" in lower_terms:
        if "germany" in lower_terms or "saxony" in lower_terms or "sachsen" in lower_terms:
            return "the German city Radebeul"
        return "Radebeul"

    babylon_markers = ["hammurabi", "babylon", "mesopotamia", "marduk", "code of hammurabi"]
    if seed_lower == "hammurabi" or any(marker in lower_terms for marker in babylon_markers):
        return "the ancient city of Babylon"

    return f"Seed: {seed}"


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


# Files whose modification times tell when a run last changed. Checking these
# few instead of every file keeps the run list fast also for runs with tens
# of thousands of report pages.
_RUN_KEY_FILES = (
    "runConfig.json", "knowledgeBase.db", "knowledgeBase.db-wal", "log.jsonl", "usage.json",
    "triples.jsonl", "subjectEvents.jsonl", "html/index.html",
)


def _key_file_stats(run_dir: Path) -> tuple[tuple[str, int, int], ...]:
    stats = []
    for name in _RUN_KEY_FILES:
        try:
            stat = (run_dir / name).stat()
        except OSError:
            continue
        stats.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(stats)


def _last_modified(run_dir: Path) -> float:
    stats = _key_file_stats(run_dir)
    if stats:
        return max(mtime for _, mtime, _ in stats) / 1e9
    return _file_time_span(run_dir)[1]


def _file_time_span(run_dir: Path) -> tuple[float, float]:
    times = [file.stat().st_mtime for file in run_dir.rglob("*") if file.is_file()]
    if not times:
        stat_time = run_dir.stat().st_mtime
        return stat_time, stat_time
    return min(times), max(times)


def _duration_seconds(run_dir: Path, config: dict[str, Any], active: bool) -> int:
    started = _parse_iso_datetime(config.get("started_at"))
    ended = _parse_iso_datetime(config.get("ended_at"))
    if started:
        end = datetime.now() if active else ended
        if end:
            return max(0, round((end - started).total_seconds()))

    first_file_time, last_file_time = _file_time_span(run_dir)
    return max(0, round(last_file_time - first_file_time))


def _normalized_model_name(config: dict[str, Any]) -> str:
    return str(config.get("model", "")).strip().lower()


def _normalized_llm_url(config: dict[str, Any]) -> str:
    return str(config.get("llm_url", "")).strip().rstrip("/").lower()


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _build_elicitation_prompt(config: dict[str, Any], subject: str) -> str:
    language = str(config.get("language", "en")).strip() or "en"
    template = str(config.get("prompt_template", "")).strip()
    if template:
        return append_subject_line(template, language, subject)
    return build_elicitation_prompt(
        language,
        str(config.get("topic", "")).strip(),
        subject,
        float(config.get("facts_multiplier", 1.0) or 1.0),
        str(config.get("schema_mode", "open")).strip(),
        str(config.get("schema_text", "")).strip(),
    )


def _build_classification_prompt(topic: str, phrases: list[str]) -> str:
    return (
        f"I want you to classify object phrases from a knowledge graph about {topic}.\n\n"
        "Classify each phrase into exactly one of these categories:\n"
        '- "domain_entity": a named entity that is relevant to the topic and should be expanded as a future subject.\n'
        '- "off_domain_entity": a named entity, but not relevant to the topic.\n'
        '- "literal": not a named entity, such as a date, number, URL, title, common noun, description, or verbose phrase.\n\n'
        'Return only JSON in this exact shape: {"results": [{"phrase": "...", "category": "domain_entity|off_domain_entity|literal"}]}.\n'
        "Use the exact input phrase strings in the phrase field.\n\n"
        f"Phrases: {json.dumps(phrases, ensure_ascii=False)}"
    )


def _estimate_run_cost(run_dir: Path, config: dict[str, Any]) -> float | None:
    # A disambiguated run tracks its real cost itself (cumulative across
    # resumes, in its database); this reconstruction only fits classic runs.
    if _pipeline_of(config) != CLASSIC_PIPELINE:
        return None
    if _normalized_llm_url(config) != "https://api.openai.com/v1":
        return None

    pricing = OPENAI_TEXT_PRICING.get(_normalized_model_name(config))
    if pricing is None:
        return None

    queued, processed, _depths, _used_event_log = load_subject_state(run_dir)

    triples_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    triples_path = run_dir / "triples.jsonl"
    if triples_path.exists():
        with triples_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    triple = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(triple, dict):
                    continue
                triples_by_subject[str(triple.get("s", ""))].append(triple)

    input_tokens = 0
    output_tokens = 0
    for subject in processed:
        input_tokens += _estimate_tokens(_build_elicitation_prompt(config, str(subject)))
        response_text = json.dumps(triples_by_subject.get(str(subject), []), ensure_ascii=False)
        output_tokens += _estimate_tokens(response_text)

    literals: list[str] = []
    literals_path = run_dir / "literals.jsonl"
    if literals_path.exists():
        with literals_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("text") is not None:
                    literals.append(str(row["text"]))

    off_domain: list[str] = []
    off_domain_path = run_dir / "offDomainEntities.jsonl"
    if off_domain_path.exists():
        with off_domain_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("text") is not None:
                    off_domain.append(str(row["text"]))

    domain_entities = [str(item) for item in processed[1:]] + [str(item) for item in queued]
    classified_rows = (
        [{"phrase": phrase, "category": "literal"} for phrase in literals]
        + [{"phrase": phrase, "category": "off_domain_entity"} for phrase in off_domain]
        + [{"phrase": phrase, "category": "domain_entity"} for phrase in domain_entities]
    )

    batch_size = int(config.get("classification_batch_size") or 30)
    topic = str(config.get("topic", "")).strip()
    for batch in chunked(classified_rows, batch_size):
        phrases = [str(row["phrase"]) for row in batch]
        input_tokens += _estimate_tokens(_build_classification_prompt(topic, phrases))
        output_tokens += _estimate_tokens(json.dumps({"results": batch}, ensure_ascii=False))

    estimated_cost = (
        (input_tokens / 1_000_000) * pricing["input_per_mtok"]
        + (output_tokens / 1_000_000) * pricing["output_per_mtok"]
    )
    return round(estimated_cost, 4)


def _update_run_config(run_dir: Path, **updates: Any) -> None:
    config_path = run_dir / "runConfig.json"
    config = _read_json(config_path) or {}
    if not isinstance(config, dict):
        config = {}
    config.update(updates)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _safe_run_config(
    request: RunRequest,
    *,
    started_at: str | None = None,
    resumed_at: str | None = None,
    folder_id: str | None = None,
    pipeline: str = CURRENT_PIPELINE,
) -> dict[str, Any]:
    safe_config = request.model_dump()
    safe_config["api_key"] = "***"
    safe_config["pipeline"] = pipeline
    if pipeline == CURRENT_PIPELINE:
        safe_config["ner_model"] = request.ner_model.strip() or request.model
        safe_config["ned_model"] = request.ned_model.strip() or request.model
        # Recorded explicitly (also when it is the elicitation endpoint).
        safe_config["ner_llm_url"] = _role_llm_url(request, "ner") if request.ner_model.strip() else _normalize_url(request.llm_url)
        safe_config["ned_llm_url"] = _role_llm_url(request, "ned") if request.ned_model.strip() else _normalize_url(request.llm_url)
        safe_config["embedding_model"] = DEFAULT_EMBEDDING_MODEL
    else:
        for key in ("ner_model", "ned_model", "ner_llm_url", "ned_llm_url"):
            safe_config.pop(key, None)
    safe_config.pop("new_folder_name", None)
    safe_config["folder_id"] = folder_id
    if started_at is not None:
        safe_config["started_at"] = started_at
    if resumed_at is not None:
        safe_config["resumed_at"] = resumed_at
    safe_config["ended_at"] = None
    return safe_config


def _prepare_controller(run_id: str, run_dir: Path) -> RunController:
    ctrl = RunController(run_id, run_dir)
    with _controllers_lock:
        controllers[run_id] = ctrl
    return ctrl


def _viewing_controller(run_id: str) -> RunController | None:
    """The run's controller; for a run this server process has not launched
    (e.g. after a restart) but whose folder has a log file, a passive one
    that only serves that log and the run's last known status."""
    ctrl = controllers.get(run_id)
    if ctrl is not None:
        return ctrl
    try:
        run_dir = _resolve_run_dir(run_id)
    except HTTPException:
        return None
    if not (run_dir / "log.jsonl").exists():
        return None
    ctrl = RunController(run_id, run_dir)
    try:
        ctrl.status = str(_run_summary(run_dir).get("status") or "finished")
    except Exception:
        ctrl.status = "finished"
    with _controllers_lock:
        return controllers.setdefault(run_id, ctrl)


def _launch_runner(ctrl: RunController, run_dir: Path, config_data: dict[str, Any]) -> None:
    event_queue = ctrl.event_queue
    process = MP_CONTEXT.Process(
        target=_run_process_entry,
        args=(config_data, str(run_dir), ctrl.event_queue, ctrl.pause_event, ctrl.stop_event),
        daemon=True,
    )
    ctrl.process = process

    def listen() -> None:
        while process.is_alive() or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            ctrl.emit(event)

        process.join(timeout=0)
        if process.exitcode == 0:
            updates: dict[str, Any] = {"ended_at": datetime.now().isoformat(timespec="seconds")}
            estimated_cost = _latest_live_cost(ctrl)
            if estimated_cost is None:
                estimated_cost = _estimate_run_cost(run_dir, config_data)
            if estimated_cost is not None:
                updates["estimated_cost_usd"] = estimated_cost
            _update_run_config(run_dir, **updates)
        elif ctrl.status != "stopped":
            updates = {
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "error": f"Run process exited with code {process.exitcode}.",
            }
            estimated_cost = _latest_live_cost(ctrl)
            if estimated_cost is None:
                estimated_cost = _estimate_run_cost(run_dir, config_data)
            if estimated_cost is not None:
                updates["estimated_cost_usd"] = estimated_cost
            _update_run_config(run_dir, **updates)
            ctrl.emit(
                {
                    "type": "log",
                    "level": "error",
                    "time": time.strftime("%H:%M:%S"),
                    "message": f"Run process exited with code {process.exitcode}.",
                }
            )
            ctrl.emit({"type": "status", "status": "failed"})

    process.start()
    ctrl.listener_thread = threading.Thread(target=listen, daemon=True)
    ctrl.listener_thread.start()


def _runner_config(request: RunRequest, pipeline: str) -> dict[str, Any]:
    """The config handed to the runner subprocess (with the real API keys,
    unlike the redacted runConfig.json). folder_id/new_folder_name are
    server-side organizational fields, so they're left out."""
    config_data = request.model_dump(exclude={"folder_id", "new_folder_name"})
    config_data["api_key"] = _resolve_api_key(request.llm_url, request.api_key)
    config_data["pipeline"] = pipeline
    if pipeline == CURRENT_PIPELINE:
        for role in ("ner", "ned"):
            url = _role_llm_url(request, role)
            if not getattr(request, f"{role}_model").strip() or url == _normalize_url(request.llm_url):
                config_data[f"{role}_llm_url"], config_data[f"{role}_api_key"] = "", ""
            else:  # ScaDS.AI next to another elicitation endpoint
                config_data[f"{role}_llm_url"], config_data[f"{role}_api_key"] = SCADS_ENDPOINT, _require_scads_key()
        # Embeddings always come from ScaDS.AI's Qwen3-Embedding-4B (the
        # GPTKB 2.0 embedding model), whatever endpoint the LLMs use.
        config_data["embedding_url"] = SCADS_EMBEDDING_URL
        config_data["embedding_api_key"] = _require_scads_key()
        config_data["embedding_model"] = DEFAULT_EMBEDDING_MODEL
    return config_data


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


# Runs on the project's shared ScaDS.AI key stop after an hour unless the
# user set a runtime limit themselves, so a forgotten crawl cannot hold the
# shared quota for days. Any explicit limit (also a very large one) is kept.
FREE_KEY_RUNTIME_SECONDS = 3600


def _uses_shared_key(request: RunRequest, pipeline: str) -> bool:
    """Whether any model of the run is called with the shared ScaDS.AI key:
    the main endpoint is ScaDS.AI and the user gave no key, or (with the
    disambiguation pipeline) NER/NED run on ScaDS.AI next to another main
    endpoint. Embeddings, which always use the shared key, do not count."""
    main = _normalize_url(request.llm_url)
    if main == SCADS_ENDPOINT and not request.api_key.strip():
        return True
    if pipeline != CURRENT_PIPELINE:
        return False
    return any(
        getattr(request, f"{role}_model").strip() and _role_llm_url(request, role) == SCADS_ENDPOINT != main
        for role in ("ner", "ned")
    )


def _apply_free_key_runtime(request: RunRequest, pipeline: str) -> bool:
    """Sets the one-hour runtime limit on shared-key runs without one; True if it did."""
    if request.max_runtime_seconds is None and not request.no_time_limit and _uses_shared_key(request, pipeline):
        request.max_runtime_seconds = FREE_KEY_RUNTIME_SECONDS
        return True
    return False


def _log_free_key_runtime(ctrl: RunController) -> None:
    ctrl.emit({
        "type": "log", "level": "warning", "time": time.strftime("%H:%M:%S"),
        "message": "This run uses the shared free ScaDS.AI key, so it stops after 60 minutes. "
                   "For a longer run, set a runtime limit or choose No time limit under Advanced > Stopping Criteria.",
    })


def _role_llm_url(request: RunRequest, role: str) -> str:
    """The normalized endpoint of the NER or NED model -- the elicitation
    endpoint unless ScaDS.AI was picked; any other URL is refused, so the
    shared ScaDS key and the user's key only ever go to their own endpoint."""
    url = _normalize_url(getattr(request, f"{role}_llm_url") or request.llm_url)
    if url not in (_normalize_url(request.llm_url), SCADS_ENDPOINT):
        raise HTTPException(
            status_code=400,
            detail=f"The {role.upper()} model must come from ScaDS.AI or from the elicitation model's endpoint.",
        )
    return url


def _emit_run_start_logs(
    ctrl: RunController, run_id: str, request: RunRequest, pipeline: str, html_index_url: str, action: str
) -> None:
    ctrl.emit({"type": "log", "level": "info", "time": time.strftime("%H:%M:%S"), "message": f"Run {run_id} {action}."})
    if pipeline == CURRENT_PIPELINE:
        models = (
            f"Models: elicitation {request.model}, NER {request.ner_model.strip() or request.model}, "
            f"NED {request.ned_model.strip() or request.model}"
        )
    else:
        models = f"Model: {request.model} (classic pipeline)"
    ctrl.emit({"type": "log", "level": "info", "time": time.strftime("%H:%M:%S"), "message": models})
    ctrl.emit({"type": "log", "level": "info", "time": time.strftime("%H:%M:%S"), "message": "HTML index", "url": html_index_url})


def _latest_current_subject(ctrl: RunController) -> str | None:
    with ctrl.lock:
        for event in reversed(ctrl._transient_events):
            if event.get("type") == "current_subject":
                subject = str(event.get("subject", "")).strip()
                return subject or None
    return None


def _latest_live_cost(ctrl: RunController) -> float | None:
    with ctrl.lock:
        for event in reversed(ctrl._transient_events):
            if event.get("type") == "status" and event.get("cost_usd") is not None:
                return float(event["cost_usd"])
    return None


def _latest_progress_counts(ctrl: RunController) -> tuple[int, int] | None:
    """(processed, queued) from the most recent live status event, if any --
    lets an active run's History row be built with zero disk/log reads, since
    the runner already pushes these counts over SSE on every status update.
    """
    with ctrl.lock:
        for event in reversed(ctrl._transient_events):
            if event.get("type") == "status" and "processed" in event and "queued" in event:
                try:
                    return int(event["processed"]), int(event["queued"])
                except (TypeError, ValueError):
                    return None
    return None


def _restore_subject_to_queue(run_dir: Path, subject: str | None) -> None:
    """A forcefully-stopped run's in-flight subject is normally recovered
    automatically on the next resume -- replaying subjectEvents.jsonl treats
    a "dequeue" with no matching "finish" as still owed a retry. This is only
    a fallback for a run that predates the event log and hasn't been resumed
    since, where that safety net doesn't exist yet.
    """
    if not subject:
        return
    if (run_dir / "subjectEvents.jsonl").exists():
        return
    if _pipeline_of(_read_json(run_dir / "runConfig.json")) != CLASSIC_PIPELINE:
        return  # disambiguated runs keep their state in the database
    queue_path = run_dir / "subjectQueue.json"
    processed_path = run_dir / "processedSubjects.json"
    queued = _read_json(queue_path) or []
    processed = _read_json(processed_path) or []
    if not isinstance(queued, list):
        queued = []
    if not isinstance(processed, list):
        processed = []
    if subject in queued or subject in processed:
        return
    queue_path.write_text(json.dumps([subject, *queued], ensure_ascii=False), encoding="utf-8")


def _finalize_run_database(run_dir: Path) -> None:
    """Leaves knowledgeBase.db in a clean, complete, single-file state after
    a run's subprocess is gone, one way or another.

    The runner's own final export (runner.py's
    _write_sqlite_export(final=True)) does this already when it gets a
    chance to run -- but SIGTERM's default handler is SIG_DFL (the OS kills
    the process outright, no Python code runs), and process.terminate() is
    called right after the process's worker threads are typically deep in a
    real, slow LLM call, not idle at a stop_event check -- so in practice
    that cooperative path is the exception, not the rule, here. Doing it
    from the server instead, once the process is confirmed dead, works
    every time regardless of how the subprocess ended, and costs nothing
    extra (a fresh connection, not a wait). Idempotent and best-effort: a
    run that never got as far as creating a database, or whose runner did
    already finalize it, is left alone without erroring the stop request.
    """
    db_path = run_dir / "knowledgeBase.db"
    if not db_path.exists():
        return
    try:
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("UPDATE run SET status = 'stopped'")
            connection.commit()
        finally:
            connection.close()
    except Exception:
        pass


def _terminate_active_process(ctrl: RunController) -> bool:
    process = ctrl.process
    if process is None or not process.is_alive():
        return False

    run_dir = ctrl.run_dir
    current_subject = _latest_current_subject(ctrl)
    if run_dir is not None:
        _restore_subject_to_queue(run_dir, current_subject)

    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)

    if run_dir is not None:
        _update_run_config(run_dir, ended_at=datetime.now().isoformat(timespec="seconds"), stopped_by_user=True)
        _finalize_run_database(run_dir)
    return True


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """Builds one History-table row.

    /api/runs calls this for every run directory on every poll, so the
    common cases are kept as cheap as possible: an active run's
    processed/queued counts come from the controller's own live status
    events (already pushed over SSE, no disk touched at all), and an
    inactive run's counts/size/inferred seed-topic are computed once and
    cached into runConfig.json (mirroring the pre-existing estimated_cost_usd
    cache) so a run sitting in history doesn't pay to be re-measured on every
    single poll for the rest of its life. Only a genuinely cold read --
    an active run before its first status event, or an inactive run seen for
    the first time -- replays subjectEvents.jsonl at all.
    """
    run_id = run_dir.name
    config = _read_json(run_dir / "runConfig.json") or {}
    html_dir = run_dir / "html"
    html_index = html_dir / "index.html"
    modified = _last_modified(run_dir)

    ctrl = controllers.get(run_id)
    active = ctrl is not None and ctrl.is_alive()

    seed_entity = str(config.get("seed_entity", "")).strip()
    topic = str(config.get("topic", "")).strip()
    processed_count: int | None = None
    queued_count: int | None = None
    pipeline = _pipeline_of(config)
    # Disambiguated runs: counts come from their database (a few indexed
    # COUNT queries), cached into runConfig.json once the run is inactive.
    db_counts: dict[str, int] | None = None
    if pipeline != CLASSIC_PIPELINE:
        cached_counts = config.get("cached_db_counts")
        if not active and isinstance(cached_counts, dict):
            db_counts = cached_counts
        else:
            db_counts = progress_counts(run_dir / "knowledgeBase.db") or {}
            if not active and db_counts:
                _update_run_config(run_dir, cached_db_counts=db_counts)
        processed_count = int(db_counts.get("processed", 0))
        queued_count = int(db_counts.get("queued", 0))
    elif active:
        live_counts = _latest_progress_counts(ctrl)
        if live_counts is not None:
            processed_count, queued_count = live_counts
    else:
        cached_processed = config.get("cached_processed_count")
        cached_queued = config.get("cached_queued_count")
        if isinstance(cached_processed, int) and isinstance(cached_queued, int):
            processed_count, queued_count = cached_processed, cached_queued
        if not seed_entity:
            seed_entity = str(config.get("cached_seed_entity", "")).strip()
        if not topic:
            topic = str(config.get("cached_topic", "")).strip()

    if db_counts is None and (processed_count is None or queued_count is None or not seed_entity or not topic):
        queue_order, processed_list, depths, _used_event_log = load_subject_state(run_dir)
        if processed_count is None or queued_count is None:
            processed_count, queued_count = len(processed_list), len(queue_order)
        if not seed_entity:
            seed_entity = _infer_seed(depths, processed_list, queue_order, config)
        if not topic:
            topic = _infer_topic(run_dir, seed_entity, depths, processed_list, queue_order, config)
        if not active:
            _update_run_config(
                run_dir,
                cached_processed_count=processed_count,
                cached_queued_count=queued_count,
                cached_seed_entity=seed_entity,
                cached_topic=topic,
            )

    status = ctrl.status if active and ctrl is not None else "finished"
    if not processed_count and queued_count:
        status = "created"
    if db_counts is not None:
        triples_count = int(db_counts.get("triples", 0))
        literals_count = int(db_counts.get("literals", 0))
        failed_count = int(db_counts.get("failed_subjects", 0))
        off_domain_count = 0
    else:
        triples_count = _count_lines(run_dir / "triples.jsonl")
        literals_count = _count_lines(run_dir / "literals.jsonl") + _count_lines(run_dir / "notNE.jsonl")
        failed_count = _count_lines(run_dir / "failedSubjects.jsonl")
        off_domain_count = _count_lines(run_dir / "offDomainEntities.jsonl")

    cost_usd = config.get("estimated_cost_usd")
    if cost_usd is None and db_counts is not None:
        # Written by the runner itself as it goes, so it survives a Stop.
        cost_usd = db_counts.get("cost_usd")
    elif cost_usd is None:
        cost_usd = _estimate_run_cost(run_dir, config)
        if cost_usd is not None and not active:
            _update_run_config(run_dir, estimated_cost_usd=cost_usd)

    size_bytes = config.get("cached_size_bytes")
    if size_bytes is None:
        size_bytes = _directory_size(run_dir)
        if not active:
            _update_run_config(run_dir, cached_size_bytes=size_bytes)

    return {
        "run_id": run_id,
        "status": status,
        "topic": topic,
        "seed_entity": seed_entity,
        "model": config.get("model", ""),
        "ner_model": config.get("ner_model"),
        "ned_model": config.get("ned_model"),
        "ner_llm_url": config.get("ner_llm_url"),
        "ned_llm_url": config.get("ned_llm_url"),
        "pipeline": pipeline,
        "entities": db_counts.get("entities") if db_counts is not None else None,
        "folder_id": config.get("folder_id"),
        "started_at": config.get("started_at", ""),
        "ended_at": config.get("ended_at", ""),
        "duration_seconds": _duration_seconds(run_dir, config, active),
        "cost_usd": cost_usd,
        "modified_at": datetime.fromtimestamp(modified).isoformat(timespec="seconds"),
        "processed": processed_count,
        "queued": queued_count,
        "pending": db_counts.get("pending") if db_counts is not None else None,
        "resumable": bool(queued_count) or bool(db_counts and db_counts.get("pending")),
        "triples": triples_count,
        "literals": literals_count,
        "off_domain_entities": off_domain_count,
        "failed_subjects": failed_count,
        "html_pages": len(list(html_dir.glob("*.html"))) if html_dir.exists() else 0,
        "size_bytes": size_bytes,
        "run_dir": str(run_dir),
        "html_index_url": f"/runs/{run_id}/html/index.html" if html_index.exists() else None,
    }


# Run-list rows of inactive runs, reused until one of the run's key files
# changes (a resume, a rename or folder move rewrites runConfig.json).
# Active runs are always measured afresh.
_summary_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
_summary_cache_lock = threading.Lock()


def _cached_run_summary(run_dir: Path) -> dict[str, Any]:
    if _is_run_active(run_dir.name):
        return _run_summary(run_dir)
    signature = _key_file_stats(run_dir)
    with _summary_cache_lock:
        cached = _summary_cache.get(run_dir.name)
    if cached and cached[0] == signature:
        return cached[1]
    summary = _run_summary(run_dir)
    with _summary_cache_lock:
        # _run_summary may itself update runConfig.json (cached counts), so
        # the signature is taken again after it.
        _summary_cache[run_dir.name] = (_key_file_stats(run_dir), summary)
    return summary


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    runs = [
        _cached_run_summary(path)
        for path in RUNS_DIR.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ]
    runs.sort(key=lambda item: item["modified_at"], reverse=True)
    return {"runs": runs}


@app.get("/api/folders")
def list_folders() -> dict[str, Any]:
    with _folders_lock:
        folders = _read_folders_unlocked()
    return {"folders": sorted(folders, key=lambda folder: folder["name"].lower())}


@app.post("/api/folders", dependencies=[Depends(rate_limited(_run_control_limiter))])
def create_folder(request: FolderRequest) -> dict[str, Any]:
    return _create_folder(request.name)


@app.patch("/api/folders/{folder_id}", dependencies=[Depends(rate_limited(_run_control_limiter))])
def rename_folder(folder_id: str, request: FolderRequest) -> dict[str, Any]:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty.")
    with _folders_lock:
        folders = _read_folders_unlocked()
        folder = next((item for item in folders if item["id"] == folder_id), None)
        if folder is None:
            raise HTTPException(status_code=404, detail="Folder not found.")
        folder["name"] = name
        _write_folders_unlocked(folders)
    return folder


@app.delete("/api/folders/{folder_id}", dependencies=[Depends(rate_limited(_run_control_limiter))])
def delete_folder(folder_id: str) -> dict[str, Any]:
    with _folders_lock:
        folders = _read_folders_unlocked()
        if not any(folder["id"] == folder_id for folder in folders):
            raise HTTPException(status_code=404, detail="Folder not found.")
        folders = [folder for folder in folders if folder["id"] != folder_id]
        _write_folders_unlocked(folders)

    # Deleting a folder never deletes its runs -- they fall back to
    # "No folder" rather than disappearing along with it.
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        config = _read_json(run_dir / "runConfig.json")
        if isinstance(config, dict) and config.get("folder_id") == folder_id:
            _update_run_config(run_dir, folder_id=None)

    return {"deleted": folder_id}


def _resolve_run_dir(run_id: str) -> Path:
    if not run_id or any(char in run_id for char in ('/', '\\')) or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = (RUNS_DIR / run_id).resolve()
    runs_root = RUNS_DIR.resolve()
    if run_dir.parent != runs_root:
        raise HTTPException(status_code=400, detail="Invalid run path.")
    return run_dir


@app.patch("/api/runs/{run_id}", dependencies=[Depends(rate_limited(_run_control_limiter))])
def update_run(run_id: str, request: RunFolderUpdate) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    if request.folder_id is not None and not _folder_exists(request.folder_id):
        raise HTTPException(status_code=404, detail="Folder not found.")

    _update_run_config(run_dir, folder_id=request.folder_id)
    return {"run_id": run_id, "folder_id": request.folder_id}


def _delete_run_dir(run_id: str) -> None:
    """Raises HTTPException on failure; used by both the single- and bulk-delete endpoints."""
    if _is_run_active(run_id):
        raise HTTPException(status_code=409, detail="Stop the active run before deleting it.")

    run_dir = _resolve_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")

    shutil.rmtree(run_dir)


@app.delete("/api/runs/{run_id}", dependencies=[Depends(rate_limited(_run_control_limiter))])
def delete_run(run_id: str) -> dict[str, Any]:
    _delete_run_dir(run_id)
    return {"deleted": run_id}


@app.post("/api/runs/bulk-delete", dependencies=[Depends(rate_limited(_run_control_limiter))])
def bulk_delete_runs(request: BulkDeleteRequest) -> dict[str, Any]:
    deleted: list[str] = []
    skipped: list[dict[str, str]] = []
    for run_id in request.run_ids:
        try:
            _delete_run_dir(run_id)
            deleted.append(run_id)
        except HTTPException as exc:
            skipped.append({"run_id": run_id, "detail": str(exc.detail)})
    return {"deleted": deleted, "skipped": skipped}


@app.post("/api/runs/bulk-move", dependencies=[Depends(rate_limited(_run_control_limiter))])
def bulk_move_runs(request: BulkMoveRequest) -> dict[str, Any]:
    if request.folder_id is not None and not _folder_exists(request.folder_id):
        raise HTTPException(status_code=404, detail="Folder not found.")

    moved: list[str] = []
    skipped: list[dict[str, str]] = []
    for run_id in request.run_ids:
        try:
            run_dir = _resolve_run_dir(run_id)
            if not run_dir.exists() or not run_dir.is_dir():
                raise HTTPException(status_code=404, detail="Run not found.")
            _update_run_config(run_dir, folder_id=request.folder_id)
            moved.append(run_id)
        except HTTPException as exc:
            skipped.append({"run_id": run_id, "detail": str(exc.detail)})
    return {"moved": moved, "skipped": skipped, "folder_id": request.folder_id}


@app.post("/api/run", dependencies=[Depends(rate_limited(_run_control_limiter))])
def start_run(request: RunRequest) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    run_dir = RUNS_DIR / run_id
    html_index_url = f"/runs/{run_id}/html/index.html"
    pipeline = request.pipeline
    free_key_limit = _apply_free_key_runtime(request, pipeline)
    config_data = _runner_config(request, pipeline)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    folder_id = _resolve_folder_id(request.folder_id, request.new_folder_name)
    (run_dir / "runConfig.json").write_text(
        json.dumps(_safe_run_config(request, started_at=started_at, folder_id=folder_id, pipeline=pipeline), indent=2),
        encoding="utf-8",
    )

    ctrl = _prepare_controller(run_id, run_dir)
    _emit_run_start_logs(ctrl, run_id, request, pipeline, html_index_url, "starting")
    if free_key_limit:
        _log_free_key_runtime(ctrl)
    _launch_runner(ctrl, run_dir, config_data)
    return {
        "run_id": run_id,
        "status": "starting",
        "run_dir": str(run_dir),
        "html_index_url": html_index_url,
        "started_at": started_at,
        "folder_id": folder_id,
    }


@app.post("/api/runs/{run_id}/resume", dependencies=[Depends(rate_limited(_run_control_limiter))])
def resume_historical_run(run_id: str, request: RunRequest) -> dict[str, Any]:
    if _is_run_active(run_id):
        raise HTTPException(status_code=409, detail="This run is already active.")

    run_dir = _resolve_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")

    existing_config = _read_json(run_dir / "runConfig.json") or {}
    pipeline = _pipeline_of(existing_config)
    if pipeline == CLASSIC_PIPELINE:
        if not load_subject_state(run_dir)[0]:
            raise HTTPException(status_code=409, detail="This run has no queued objects to resume.")
    else:
        counts = progress_counts(run_dir / "knowledgeBase.db")
        if not counts or (not counts["queued"] and not counts["pending"]):
            raise HTTPException(status_code=409, detail="This run has nothing left to explore or disambiguate.")

    html_index_url = f"/runs/{run_id}/html/index.html"
    free_key_limit = _apply_free_key_runtime(request, pipeline)
    config_data = _runner_config(request, pipeline)
    started_at = str(
        (existing_config.get("started_at") if isinstance(existing_config, dict) else None)
        or datetime.now().isoformat(timespec="seconds")
    )
    # A run's folder assignment is set once (at launch or via the History
    # table's move control) and carried forward across resumes -- the resume
    # form's folder fields (if any) are ignored here so resuming never
    # silently re-files or re-creates a folder for an already-organized run.
    existing_folder_id = existing_config.get("folder_id") if isinstance(existing_config, dict) else None
    safe_config = _safe_run_config(
        request,
        started_at=started_at,
        resumed_at=datetime.now().isoformat(timespec="seconds"),
        folder_id=existing_folder_id,
        pipeline=pipeline,
    )
    (run_dir / "runConfig.json").write_text(json.dumps(safe_config, indent=2), encoding="utf-8")

    ctrl = _prepare_controller(run_id, run_dir)
    _emit_run_start_logs(ctrl, run_id, request, pipeline, html_index_url, "resuming")
    if free_key_limit:
        _log_free_key_runtime(ctrl)
    _launch_runner(ctrl, run_dir, config_data)
    return {
        "run_id": run_id,
        "status": "starting",
        "run_dir": str(run_dir),
        "html_index_url": html_index_url,
        "started_at": started_at,
        "folder_id": existing_folder_id,
    }


def _require_active_controller(run_id: str) -> RunController:
    ctrl = controllers.get(run_id)
    if ctrl is None or not ctrl.is_alive():
        raise HTTPException(status_code=409, detail="This run is not active.")
    return ctrl


@app.post("/api/runs/{run_id}/pause")
def pause_run(run_id: str) -> dict[str, str]:
    ctrl = _require_active_controller(run_id)
    ctrl.pause_event.set()
    ctrl.emit({"type": "status", "status": "paused"})
    ctrl.emit({"type": "log", "level": "info", "time": time.strftime("%H:%M:%S"), "message": "Pause requested."})
    return {"status": "paused"}


@app.post("/api/runs/{run_id}/unpause")
def unpause_run(run_id: str) -> dict[str, str]:
    ctrl = _require_active_controller(run_id)
    ctrl.pause_event.clear()
    ctrl.emit({"type": "status", "status": "running"})
    ctrl.emit({"type": "log", "level": "info", "time": time.strftime("%H:%M:%S"), "message": "Resume requested."})
    return {"status": "running"}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, str]:
    ctrl = controllers.get(run_id)
    if ctrl is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    ctrl.stop_event.set()
    ctrl.pause_event.clear()
    ctrl.status = "stopped"
    killed = _terminate_active_process(ctrl)
    ctrl.emit({"type": "status", "status": "stopped"})
    ctrl.emit({
        "type": "log",
        "level": "warning",
        "time": time.strftime("%H:%M:%S"),
        "message": "Run process killed." if killed else "Stop requested, but no run process was active.",
    })
    return {"status": "stopped"}


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str) -> dict[str, Any]:
    ctrl = _viewing_controller(run_id)
    if ctrl is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return ctrl.snapshot()


def _zip_directory(directory: Path) -> bytes:
    """Zips every file under `directory`, with archive paths relative to it
    (so extracting lands the contents at the zip's root, not nested inside a
    directory-name folder).
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=str(file_path.relative_to(directory)))
    return buffer.getvalue()


def _zip_response(directory: Path, filename: str) -> Response:
    return Response(
        content=_zip_directory(directory),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/runs/{run_id}/download")
def download_run(run_id: str) -> Response:
    """Zips the whole run directory (data files, knowledgeBase.db, and the
    generated HTML report) on the fly and returns it as an attachment.
    api_key is never in there -- runConfig.json already redacts it before
    it's written to disk.
    """
    run_dir = _resolve_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    return _zip_response(run_dir, f"minigptkb-{run_id}.zip")


LOG_REPLAY_LINES = 500


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, before: int, limit: int = 500) -> dict[str, Any]:
    """Log lines older than event id `before`, the newest `limit` of them,
    for a viewer scrolling up past the lines the event stream sent."""
    ctrl = _viewing_controller(run_id)
    if ctrl is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    limit = max(1, min(limit, 2000))
    with ctrl.lock:
        older = [event for event in ctrl._log_events if event["id"] < before]
    return {"events": older[-limit:], "more": len(older) > limit}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request) -> StreamingResponse:
    ctrl = _viewing_controller(run_id)
    if ctrl is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    # A browser that reconnects sends the id of the last event it got; carry
    # on after it. A new viewer gets only the last LOG_REPLAY_LINES log lines
    # (older ones are fetched page by page from /log when scrolling up).
    last_event_id = request.headers.get("last-event-id", "")
    if last_event_id.isdigit():
        start = int(last_event_id) + 1
    else:
        with ctrl.lock:
            logs = ctrl._log_events
            start = logs[-LOG_REPLAY_LINES]["id"] if len(logs) > LOG_REPLAY_LINES else 0

    async def stream():
        cursor = start
        idle_polls = 0
        while True:
            with ctrl.lock:
                pending = ctrl.events_since(cursor)
            if pending:
                idle_polls = 0
            for event in pending:
                cursor = event["id"] + 1
                yield f"id: {event['id']}\ndata: {json.dumps(event)}\n\n"
            if not pending:
                idle_polls += 1
                # Long idle stretches between events (LLM calls can take a while) look
                # identical to a dead connection to browsers/proxies with their own idle
                # timeouts, which then silently reconnect -- and every reconnect restarts
                # this generator's cursor at 0. A periodic comment line keeps the
                # connection visibly alive without affecting the event stream itself.
                if idle_polls % 20 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")
