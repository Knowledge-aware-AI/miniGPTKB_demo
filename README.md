# miniGPTKB

miniGPTKB is a web application for building small, topic-specific knowledge
bases from what a large language model knows. You choose a topic, a seed
entity and a model; miniGPTKB asks the model for facts about the seed as
subject–predicate–object triples, recognizes the named entities among the
objects, decides whether each one is an entity it already has or a new one,
and then asks about the new ones, round after round, until nothing new is left
or a budget runs out.

Entities, relations and classes are disambiguated during construction with the
pipeline of [GPTKB 2.0](https://github.com/Knowledge-aware-AI/GPTKB2_disambiguation_on_the_fly),
so the result has canonical identifiers rather than strings, and every merge
decision can be inspected.

A hosted instance runs at <http://minigptkb.scads.ai/>.

## Features

- **Any model, any OpenAI-compatible endpoint:** presets for ScaDS.AI, OpenAI
  and OpenRouter, or a custom URL (e.g. a self-hosted vLLM server). One model
  does all three tasks (fact elicitation, named entity recognition,
  disambiguation) unless NER and NED are given models of their own under
  *Advanced*.
- **Topic-scoped, multilingual elicitation:** 11 pre-packaged prompt languages, an editable
  prompt, open, example-driven or fixed-ontology schema.
- **Budgets:** caps on entities, BFS depth, cost (USD) and runtime. Entity and
  depth caps stop the exploration but let pending disambiguation finish.
- **Live view:** metrics, the current pipeline step and the log stream into
  the browser; runs can be paused, stopped and resumed, also after a server
  restart.
- **Controllable merging:** a slider trades fewer wrong merges for more
  duplicates by restricting which existing entities are offered as matches.
- **Results:** a browsable HTML report per run (statistics, an interactive
  graph, one page per entity with its disambiguation trace) and a SQLite
  database with entities, relations, classes, aliases, triples and every
  disambiguation decision.
- **Shared, rate-limited endpoints:** requests are paced per model below the
  limits the endpoint reports.

## How a run works

Each run is a separate worker process. In every round it runs the GPTKB 2.0
stages on whatever items are ready for them:

```
NEDG -> NED2 -> NED1 -> Elicitation -> PDG -> CDG -> PD -> CD -> NER
```

- **Elicitation** asks for the facts about unexplored entities.
- **NER** classifies each triple's object as a named entity or a literal.
- **NED1** matches an object against the most similar existing entities, using
  its triple as context; **NEDG** writes a description for objects without a
  match, and **NED2** matches again using that description, or creates a new
  entity.
- **PD/PDG** and **CD/CDG** do the same for relations and classes.

Candidates are retrieved by label embedding (Qwen3-Embedding-4B). The state of
a run lives in `runs/<run id>/knowledgeBase.db`, which is also its export and
what a resumed run continues from.

The original miniGPTKB crawler, which identifies entities by their names, is
kept in `app/runner.py`; runs made with it stay viewable and resumable, and new
ones can be started through the API with `"pipeline": "classic"`.

## Running it

Requirements: Python 3.11 or newer.

```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

> **Embeddings need a ScaDS.AI key.** Disambiguation retrieves its candidates
> with the embedding model Qwen3-Embedding-4B, which miniGPTKB calls on the
> ScaDS.AI inference service with the key in `SCADS_API_KEY`, whatever endpoint
> the language models use. Without that key, runs with disambiguation fail. To
> run miniGPTKB without one, point the embeddings to another OpenAI-compatible
> embeddings endpoint (e.g. Qwen3-Embedding-4B served locally with vLLM or
> Ollama): the endpoint and model are set in `app/disambig/runner.py`
> (`SCADS_EMBEDDING_URL`, `DEFAULT_EMBEDDING_MODEL`) and the key in
> `app/server.py`. With a different embedding model, the similarity thresholds
> of the entity-merging slider and of label clustering may need adjusting.

Environment variables:

| Variable | Purpose |
|---|---|
| `SCADS_API_KEY` | Key for the ScaDS.AI inference service. Used for label embeddings, and for model calls on ScaDS.AI when the user leaves the API key field empty. Runs on this shared key without a runtime limit stop after 60 minutes. |
| `MINIGPTKB_RUNS_DIR` | Where run folders are kept (default: `runs/`). |

API keys that users enter in the form are passed to the run in memory only:
they are never written to disk, and they are sent only to the endpoint they
were entered for.

`deploy/` contains an example systemd unit and environment file for a hosted
instance.

## Repository layout

| Path | Contents |
|---|---|
| `app/server.py` | FastAPI server: REST API, run processes, event stream, run list and folders, downloads |
| `app/disambig/` | The disambiguation pipeline: runner, prompts and answer parsers, SQLite schema, embeddings, pacing, HTML report |
| `app/runner.py` | The original string-identity crawler |
| `app/usage.py` | Per-run counts of model calls and tokens (`usage.json`) |
| `app/static/` | The browser application (plain JavaScript, no build step) |
| `app/vendor/` | vis-network, for the graph view |
| `tests/` | Unit tests: `python -m unittest discover -s tests` |

## License

miniGPTKB is released under the
[Creative Commons Attribution 4.0 International](LICENSE) license.

The prompts in `app/disambig/prompts/` are taken from the GPTKB 2.0 repository
and remain under the Apache License 2.0 (`app/disambig/prompts/LICENSE-GPTKB2`,
source commit in `app/disambig/prompts/SOURCE`). `app/vendor/vis-network.min.js`
is dual-licensed under the Apache License 2.0 and MIT.
