"""SQLite schema of a disambiguated miniGPTKB run (knowledgeBase.db).

Mirrors GPTKB 2.0's data model (db/db_models.py): entities ("instance
nodes"), predicates and concepts (classes) are first-class elements with ids,
a label and a description, and triples move through the pipeline by status.
The same file is the run's working state (a run resumes from it) and its
downloadable export, so it also carries miniGPTKB's `run` metadata table and
a log of every disambiguation decision for the HTML report.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Object (triple object) statuses, in pipeline order -- GPTKB 2.0's
# ObjectStatusType minus the "on..." statuses only its batch mode uses.
OBJ_GENERATED = "generated"
OBJ_NER_FINISHED = "ner_finished"
OBJ_NED1_FINISHED = "ned_finished1"
OBJ_DG_FINISHED = "dg_finished"
OBJ_FINISHED = "finished"

TYPE_UNRECOGNIZED = "unrecognized"
TYPE_NE = "ne"
TYPE_LITERAL = "literal"
TYPE_CLASS = "class"

PRED_GENERATED = "generated"
PRED_PD_FINISHED = "pd_finished"
PRED_FINISHED = "finished"

CONCEPT_GENERATED = "generated"
CONCEPT_CD_FINISHED = "cd_finished"
CONCEPT_FINISHED = "finished"

ENTITY_UNEXPLORED = "unexplored"
ENTITY_EXPLORED = "explored"
ENTITY_FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS run (
    run_id TEXT, pipeline TEXT, topic TEXT, seed_entity TEXT,
    model TEXT, ner_model TEXT, ned_model TEXT, embedding_model TEXT,
    llm_url TEXT, language TEXT, schema_mode TEXT, schema_text TEXT, prompt_used TEXT,
    facts_multiplier REAL, nthreads INTEGER, max_requests_per_minute INTEGER,
    max_entities INTEGER, max_bfs_depth INTEGER, max_runtime_seconds INTEGER, max_cost REAL,
    started_at TEXT, ended_at TEXT, status TEXT, cost_usd REAL,
    reasoning_effort TEXT, reasoning_enabled INTEGER
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unexplored',
    depth INTEGER NOT NULL DEFAULT 0,
    first_appeared TEXT,
    num_triples INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    UNIQUE (label, description)
);
CREATE INDEX IF NOT EXISTS idx_entities_status ON entities (status, seq);
CREATE INDEX IF NOT EXISTS idx_entities_label ON entities (label);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id TEXT NOT NULL, alias TEXT NOT NULL, PRIMARY KEY (entity_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases (alias);

CREATE TABLE IF NOT EXISTS predicates (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL, label TEXT NOT NULL, description TEXT NOT NULL,
    first_appeared TEXT, UNIQUE (label, description)
);
CREATE TABLE IF NOT EXISTS predicate_aliases (
    predicate_id TEXT NOT NULL, alias TEXT NOT NULL, PRIMARY KEY (predicate_id, alias)
);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL, label TEXT NOT NULL, description TEXT NOT NULL,
    first_appeared TEXT, UNIQUE (label, description)
);
CREATE TABLE IF NOT EXISTS concept_aliases (
    concept_id TEXT NOT NULL, alias TEXT NOT NULL, PRIMARY KEY (concept_id, alias)
);

CREATE TABLE IF NOT EXISTS triples (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    subject_id TEXT NOT NULL,
    subject_label TEXT NOT NULL,
    predicate_label TEXT NOT NULL,
    predicate_id TEXT,
    predicate_status TEXT NOT NULL DEFAULT 'generated',
    object_label TEXT NOT NULL,
    object_id TEXT,
    object_type TEXT NOT NULL DEFAULT 'unrecognized',
    object_status TEXT NOT NULL DEFAULT 'generated',
    object_description TEXT,
    -- How the object got (or didn't get) its id: cache, ned1, ned2, new,
    -- unsure (NED1 option G) or failed (gave up after repeated errors).
    resolution TEXT,
    attempts INTEGER NOT NULL DEFAULT 0
);
-- GPTKB 2.0's triple_unique index: a resolved triple that would duplicate
-- an existing one is deleted instead (NULL ids never collide in SQLite).
CREATE UNIQUE INDEX IF NOT EXISTS triple_unique ON triples
    (subject_id, subject_label, predicate_id, predicate_label, object_id, object_label);
CREATE INDEX IF NOT EXISTS idx_triples_seq ON triples (seq);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples (subject_id);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples (object_id);
CREATE INDEX IF NOT EXISTS idx_triples_object_stage ON triples (object_type, object_status, seq);
CREATE INDEX IF NOT EXISTS idx_triples_predicate_stage ON triples (predicate_status, seq);
CREATE INDEX IF NOT EXISTS idx_triples_pred_obj ON triples (predicate_label, object_label, object_id);
CREATE INDEX IF NOT EXISTS idx_triples_object_label ON triples (object_label);

CREATE TABLE IF NOT EXISTS instance_triples (
    triple_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    concept_label TEXT NOT NULL,
    concept_id TEXT,
    status TEXT NOT NULL DEFAULT 'generated',
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_instance_status ON instance_triples (status);
CREATE INDEX IF NOT EXISTS idx_instance_label ON instance_triples (concept_label);
CREATE INDEX IF NOT EXISTS idx_instance_entity ON instance_triples (entity_id, concept_id);

-- The candidates offered for each disambiguation decision (NED1, NED2, PD,
-- CD), in option order -- the model answers with an option letter.
CREATE TABLE IF NOT EXISTS candidates (
    triple_id TEXT NOT NULL, stage TEXT NOT NULL, position INTEGER NOT NULL, ref_id TEXT NOT NULL,
    PRIMARY KEY (triple_id, stage, position)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triple_id TEXT, stage TEXT, target TEXT, context TEXT,
    candidates TEXT, answer TEXT, outcome TEXT, chosen_id TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_chosen ON decisions (chosen_id);
CREATE INDEX IF NOT EXISTS idx_decisions_triple ON decisions (triple_id);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT, stage TEXT, error TEXT, detail TEXT, time TEXT
);

CREATE TABLE IF NOT EXISTS text_embeddings (text TEXT PRIMARY KEY, vec BLOB NOT NULL);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(SCHEMA)
    connection.commit()
    return connection


def get_meta(connection: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))


def progress_counts(db_path: Path) -> dict[str, Any] | None:
    """Summary counts for the History table, read-only and without touching
    the schema -- None if the run has no database yet."""
    if not db_path.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        def scalar(sql: str) -> int:
            return int(connection.execute(sql).fetchone()[0] or 0)

        return {
            "processed": scalar("SELECT COUNT(*) FROM entities WHERE status != 'unexplored'"),
            "queued": scalar("SELECT COUNT(*) FROM entities WHERE status = 'unexplored'"),
            "entities": scalar("SELECT COUNT(*) FROM entities"),
            "triples": scalar("SELECT COUNT(*) FROM triples"),
            "literals": scalar("SELECT COUNT(*) FROM triples WHERE object_type = 'literal'"),
            "failed_subjects": scalar("SELECT COUNT(*) FROM failures"),
            "pending": scalar(
                "SELECT COUNT(*) FROM triples WHERE object_status != 'finished' OR predicate_status != 'finished'"
            )
            + scalar("SELECT COUNT(*) FROM instance_triples WHERE status != 'finished'"),
            "cost_usd": (connection.execute("SELECT cost_usd FROM run").fetchone() or (None,))[0],
        }
    except sqlite3.Error:
        return None
    finally:
        connection.close()
