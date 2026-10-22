"""Offline tests of the GPTKB 2.0 disambiguation runner: a scripted fake LLM
and hashed character-trigram "embeddings" stand in for the real endpoints.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from app.disambig import prompts as P
from app.disambig.embeddings import EMBEDDING_DIM, EmbeddingClient, VectorIndex, cluster_labels
from app.disambig.runner import DisambigConfig, DisambiguatedKBCRunner, LLMClient

# What the fake model "knows": subject label -> facts. Two different Munichs:
# the film (object of notableWork) and the city (object of hasCity/livesIn).
WORLD = {
    ("Mathieu Amalric", None): [
        ("instanceOf", "actor"), ("notableWork", "Munich"), ("placeOfBirth", "Neuilly-sur-Seine"), ("birthYear", "1965"),
    ],
    ("Munich", "film"): [("instanceOf", "film"), ("director", "Steven Spielberg"), ("releaseYear", "2005")],
    ("Munich", "city"): [("instanceOf", "city"), ("country", "Federal Republic of Germany"), ("population", "1.5 million")],
    ("Steven Spielberg", None): [
        ("instanceOf", "human"), ("notableWork", "Munich"), ("directed", "Jaws"), ("born in", "Cincinnati"),
    ],
    ("Neuilly-sur-Seine", None): [("instanceOf", "commune"), ("country", "France"), ("nearCity", "Paris")],
    ("France", None): [("instanceOf", "country"), ("capital", "Paris"), ("neighbour", "Germany")],
    ("Germany", None): [("instanceOf", "country"), ("hasCity", "Munich"), ("capital", "Berlin")],
    ("Paris", None): [("instanceOf", "city"), ("country", "France")],
}
# Synonyms the fake NED accepts as the same entity.
SAME = {frozenset({"Germany", "Federal Republic of Germany"})}
CITY_PREDICATES = {"hasCity", "capital", "nearCity", "placeOfBirth", "born in"}
FILM_PREDICATES = {"notableWork", "directed"}


def kind_for(predicate: str) -> str:
    if predicate in FILM_PREDICATES:
        return "film"
    if predicate in CITY_PREDICATES:
        return "city"
    if predicate in {"country", "neighbour"}:
        return "country"
    return "thing"


def same_entity(a: str, b: str) -> bool:
    return a == b or frozenset({a, b}) in SAME


def options(user: str) -> list[tuple[str, str]]:
    return re.findall(r'^[A-J]\. \w+: "([^"]*)" \nDescription: (.*)$', user, flags=re.M)


def none_letter(user: str) -> str:
    return re.search(r"^([A-L])\. None of above\.", user, flags=re.M).group(1)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.lock = threading.Lock()

    def count(self, kind: str) -> None:
        with self.lock:
            self.calls[kind] = self.calls.get(kind, 0) + 1

    def __call__(self, role: str, system: str, user: str, max_retries: int = 5) -> str:
        if not system:  # elicitation
            self.count("elicitation")
            label = re.search(r'Subject: "(.*)"', user).group(1)
            description = user.rsplit("Description of subject: ", 1)[1]
            kind = "film" if "film" in description else "city" if "city" in description else None
            facts = WORLD.get((label, kind)) or WORLD.get((label, None)) or []
            return json.dumps([{"s": label, "p": p, "o": o} for p, o in facts])
        if system == P.NER_SYSTEM:
            self.count("ner")
            phrase = re.search(r"Phrase: (.*) \| Statement", user).group(1)
            is_ne = phrase[:1].isupper() and not phrase[:1].isdigit()
            return json.dumps({"phrase": phrase, "is_ne": is_ne})
        if system == P.NED1_SYSTEM:
            self.count("ned1")
            target = re.search(r"Target entity: (.*)", user).group(1)
            predicate = re.search(r"Context triple: \[.*?, (.*?), ", user).group(1)
            for index, (label, description) in enumerate(options(user)):
                if same_entity(label, target) and (kind_for(predicate) in description or kind_for(predicate) == "thing"):
                    return chr(65 + index)
            return none_letter(user)
        if system.startswith(P.NEDG_SYSTEM):
            self.count("nedg")
            entity = re.search(r"Entity: (.*)", user).group(1)
            predicate = re.search(r"Triple: \[.*?, (.*?), ", user).group(1)
            return f"Description: {entity} is a {kind_for(predicate)}."
        if system == P.NED2_SYSTEM:
            self.count("ned2")
            target = re.search(r"Target entity: (.*)", user).group(1)
            target_description = re.search(r"Target entity description: (.*)", user).group(1)
            for index, (label, description) in enumerate(options(user)):
                if same_entity(label, target) and description.split(" is a ")[-1] == target_description.split(" is a ")[-1]:
                    return chr(65 + index)
            return none_letter(user)
        if system in (P.PD_SYSTEM, P.CD_SYSTEM):
            self.count("pd" if system == P.PD_SYSTEM else "cd")
            target = re.search(r"Target (?:predicate|class): (.*)", user).group(1)
            for index, (label, _) in enumerate(options(user)):
                if label.lower() == target.lower():
                    return chr(65 + index)
            return "F"
        if system.startswith(P.PDG_SYSTEM):
            self.count("pdg")
            return f"Indicates the relation {re.search(r'Predicate: (.*)', user).group(1)}."
        if system.startswith(P.CDG_SYSTEM):
            self.count("cdg")
            return f"Description: The class of all {re.search(r'Class: (.*)', user).group(1)}s."
        # Seed description (adapted NEDG prompt).
        self.count("seed")
        return "Description: Mathieu Amalric is a French actor and film director."


def fake_embeddings(texts: list[str]) -> list[np.ndarray]:
    out = []
    for text in texts:
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        padded = f"  {text.lower()}  "
        for i in range(len(padded) - 2):
            bucket = int(hashlib.md5(padded[i : i + 3].encode()).hexdigest()[:8], 16) % EMBEDDING_DIM
            vec[bucket] += 1.0
        out.append(vec / (np.linalg.norm(vec) or 1.0))
    return out


class RunnerTest(unittest.TestCase):
    def run_pipeline(self, **overrides):
        tmp = Path(tempfile.mkdtemp())
        run_dir = tmp / "run"
        run_dir.mkdir()
        (run_dir / "runConfig.json").write_text("{}")
        config = DisambigConfig(
            llm_url="http://fake/v1", api_key="x", model="fake", topic="French cinema",
            seed_entity="Mathieu Amalric", nthreads=4, **overrides,
        )
        events = []
        fake = FakeLLM()
        originals = (LLMClient.chat, EmbeddingClient._request)
        LLMClient.chat = lambda self, role, system, user, max_retries=5: fake(role, system, user)
        EmbeddingClient._request = lambda self, texts, max_retries=6: fake_embeddings(texts)
        try:
            runner = DisambiguatedKBCRunner(config, run_dir, events.append, threading.Event(), threading.Event())
            runner.run()
        finally:
            LLMClient.chat, EmbeddingClient._request = originals
        return run_dir, events, fake

    def test_homonyms_and_synonyms(self):
        run_dir, events, fake = self.run_pipeline()
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertEqual(statuses[-1], "finished")

        munichs = db.execute("SELECT id, description FROM entities WHERE label = 'Munich'").fetchall()
        self.assertEqual(len(munichs), 2, munichs)
        self.assertEqual({"film" in d for _, d in munichs}, {True, False})

        film_id = next(i for i, d in munichs if "film" in d)
        city_id = next(i for i, d in munichs if "city" in d)
        work_objects = {r[0] for r in db.execute("SELECT object_id FROM triples WHERE predicate_label = 'notableWork'")}
        self.assertEqual(work_objects, {film_id})  # Amalric's and Spielberg's notableWork are the same film
        self.assertEqual(
            db.execute("SELECT object_id FROM triples WHERE predicate_label = 'hasCity'").fetchone()[0], city_id
        )

        # Germany and "Federal Republic of Germany" are one entity with an alias.
        germanies = db.execute("SELECT id FROM entities WHERE label LIKE '%Germany%'").fetchall()
        self.assertEqual(len(germanies), 1, germanies)
        aliases = {r[0] for r in db.execute("SELECT alias FROM entity_aliases WHERE entity_id = ?", germanies[0])}
        self.assertEqual(aliases, {"Federal Republic of Germany"})

        # One Paris, referenced from three subjects.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM entities WHERE label = 'Paris'").fetchone()[0], 1)

        # Everything is resolved; literals stay literals; classes resolved.
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM triples WHERE object_status != 'finished' OR predicate_status != 'finished'").fetchone()[0],
            0,
        )
        self.assertEqual(db.execute("SELECT object_type FROM triples WHERE object_label = '1965'").fetchone()[0], "literal")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM instance_triples WHERE status != 'finished'").fetchone()[0], 0)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label = 'city'").fetchone()[0], 1)
        # instanceOf is the seed predicate P0; there is no seed class, the
        # first class is created without asking the LLM about zero candidates.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label = 'human'").fetchone()[0], 1)
        self.assertEqual(db.execute("SELECT label FROM concepts WHERE id = 'C0'").fetchone()[0], "actor")
        self.assertEqual(db.execute("SELECT predicate_id FROM triples WHERE predicate_label = 'instanceOf' LIMIT 1").fetchone()[0], "P0")
        # Every entity reachable in the toy world was explored, and depth is tracked.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM entities WHERE status = 'unexplored'").fetchone()[0], 0)
        self.assertEqual(db.execute("SELECT depth FROM entities WHERE id = ?", (film_id,)).fetchone()[0], 1)

        html = run_dir / "html"
        self.assertTrue((html / "index.html").exists())
        self.assertTrue((html / f"{film_id}.html").exists())
        self.assertIn("Munich", (html / "index.html").read_text())
        self.assertTrue((run_dir / "triples.jsonl").exists())
        self.assertEqual(db.execute("SELECT status FROM run").fetchone()[0], "finished")

    def test_similarity_floor_blocks_dissimilar_merges(self):
        # "Federal Republic of Germany" is not similar enough to "Germany" to
        # be offered as a candidate at a floor of 0.99, so it becomes its own entity.
        run_dir, events, _ = self.run_pipeline(ned_similarity_floor=0.99)
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        labels = {r[0] for r in db.execute("SELECT label FROM entities")}
        self.assertIn("Federal Republic of Germany", labels)
        self.assertIn("Germany", labels)
        # Identical labels are still merged: one Paris, one France.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM entities WHERE label = 'Paris'").fetchone()[0], 1)
        self.assertEqual([e["status"] for e in events if e["type"] == "status"][-1], "finished")

    def test_more_than_five_candidates(self):
        run_dir, _, _ = self.run_pipeline(ned_candidates=7)
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM entities WHERE label = 'Munich'").fetchone()[0], 2)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM entities WHERE label LIKE '%Germany%'").fetchone()[0], 1)
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM triples WHERE object_status != 'finished'").fetchone()[0], 0
        )

    def test_bfs_depth_limit(self):
        run_dir, _, _ = self.run_pipeline(max_bfs_depth=1)
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        explored_depths = {r[0] for r in db.execute("SELECT depth FROM entities WHERE status = 'explored'")}
        self.assertEqual(explored_depths, {0, 1})
        self.assertGreater(db.execute("SELECT COUNT(*) FROM entities WHERE depth = 2").fetchone()[0], 0)
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM triples WHERE object_status != 'finished'").fetchone()[0], 0
        )

    def test_resume_is_idempotent(self):
        run_dir, _, _ = self.run_pipeline()
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        before = db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        db.close()
        config = DisambigConfig(llm_url="http://fake/v1", api_key="x", model="fake", topic="French cinema",
                                seed_entity="Mathieu Amalric")
        fake = FakeLLM()
        originals = (LLMClient.chat, EmbeddingClient._request)
        LLMClient.chat = lambda self, role, system, user, max_retries=5: fake(role, system, user)
        EmbeddingClient._request = lambda self, texts, max_retries=6: fake_embeddings(texts)
        try:
            DisambiguatedKBCRunner(config, run_dir, lambda e: None, threading.Event(), threading.Event()).run()
        finally:
            LLMClient.chat, EmbeddingClient._request = originals
        self.assertEqual(fake.calls, {})  # nothing left to do, no LLM calls
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM triples").fetchone()[0], before)


class UnitTest(unittest.TestCase):
    def test_parse_choice(self):
        self.assertEqual(P.parse_choice("B"), 2)
        self.assertEqual(P.parse_choice("B. Entity: Munich"), 2)
        self.assertIsNone(P.parse_choice("F"))
        self.assertIsNone(P.parse_choice("A or B"))  # ambiguous -> treated as none
        self.assertEqual(P.parse_choice("G", unsure=True), "unsure")
        self.assertEqual(P.parse_choice("G"), 7)  # out of range -> caller retries
        self.assertIsNone(P.parse_choice("G", "A-F"))
        self.assertEqual(P.parse_choice("<think>maybe A? or C</think>\nC"), 3)

    def test_parse_elicitation_formats(self):
        mini = P.parse_elicitation('```json\n[{"s": "X", "p": "a", "o": "Y"}]\n```')
        gptkb = P.parse_elicitation('{"facts": [{"subject": "X", "predicate": "a", "object": "Y"}]}')
        self.assertEqual(mini, gptkb)
        self.assertEqual(P.parse_elicitation("Sure! [] "), [])

    def test_parse_ner_and_description(self):
        self.assertTrue(P.parse_ner('{"phrase": "Paris", "is_ne": true}'))
        self.assertFalse(P.parse_ner('{"phrase": "1965", "is_ne": "false"}'))
        self.assertEqual(P.parse_description("Description: A city."), "A city.")

    def test_language_adaptations(self):
        self.assertIn("istanzaDi", P.instance_of_labels("it"))
        self.assertIn("english named entity", P.ner_system_prompt("en"))
        self.assertNotIn("english", P.ner_system_prompt("de"))
        self.assertNotIn("Latin script", P.ner_system_prompt("de"))

    def test_clustering_matches_threshold(self):
        vectors = np.stack(fake_embeddings(["Munich", "Munich ", "München", "Paris"]))
        labels = cluster_labels(vectors, 0.15)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[3])

    def test_parse_choice_verbose_fallback(self):
        # GPTKB 2.0's rule sees "A" (the article) and "D": ambiguous; the stated answer decides.
        self.assertEqual(P.parse_choice("The target is a character in the show, so the answer is D."), 4)
        self.assertEqual(P.parse_choice("F. \n\nThe target is a character, not an actor."), None)
        self.assertIsNone(P.parse_choice("A city is not D."))
        self.assertEqual(P.parse_choice("Answer: C"), 3)

    def test_vector_index_dedupes_aliases(self):
        index = VectorIndex()
        index.add(["E0", "E1", "E0"], np.stack(fake_embeddings(["Germany", "Berlin", "Federal Republic of Germany"])))
        self.assertEqual(index.search(np.stack(fake_embeddings(["Germany"])), 2)[0], ["E0", "E1"])
        self.assertEqual(index.search(np.stack(fake_embeddings(["Federal Republic of Germany"])), 1)[0], ["E0"])

    def test_vector_index_similarity_floor(self):
        index = VectorIndex()
        index.add(["E0", "E1"], np.stack(fake_embeddings(["Paris", "Berlin"])))
        scored = index.search_scored(np.stack(fake_embeddings(["Paris"])), 2)[0]
        self.assertEqual(scored[0][0], "E0")
        self.assertAlmostEqual(scored[0][1], 1.0, places=5)
        self.assertEqual(index.search(np.stack(fake_embeddings(["Paris"])), 2, min_similarity=0.9), [["E0"]])

    def test_vector_index(self):
        index = VectorIndex()
        index.add(["E0", "E1", "E2"], np.stack(fake_embeddings(["Paris", "Berlin", "Parisian"])))
        self.assertEqual(index.search(np.stack(fake_embeddings(["Paris"])), 2)[0], ["E0", "E2"])
        self.assertEqual(len(index.search(np.stack(fake_embeddings(["x"])), 10)[0]), 3)


if __name__ == "__main__":
    unittest.main()


class EndpointRoutingTest(unittest.TestCase):
    """Each role's request goes to its own endpoint, with that endpoint's key."""

    def test_roles_use_their_endpoints(self):
        from types import SimpleNamespace

        config = DisambigConfig(
            llm_url="https://api.openai.com/v1", api_key="sk-user", model="gpt-5.4-mini", topic="t", seed_entity="s",
            ner_model="google/gemma-4-26B-A4B-it", ner_llm_url="https://llm.scads.ai/v1", ner_api_key="scads-shared",
            ned_model="Qwen/Qwen3.8-27B", ned_llm_url="https://llm.scads.ai/v1", ned_api_key="scads-shared",
        )
        client = LLMClient(config, threading.Event(), lambda message, level: None)
        self.assertEqual(len(client.endpoints), 2)
        calls = []

        def fake_endpoint(endpoint):
            def create(**request):
                calls.append((endpoint.url, endpoint.api_key, request["model"], "temperature" in request))
                response = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="A"))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11),
                )
                return SimpleNamespace(parse=lambda: response, headers={})
            completions = SimpleNamespace(with_raw_response=SimpleNamespace(create=create))
            return SimpleNamespace(chat=SimpleNamespace(completions=completions))

        for endpoint in client.endpoints.values():
            endpoint.client = fake_endpoint(endpoint)
        for role in ("elicitation", "ner", "ned"):
            client.chat(role, "system", "user")
        self.assertEqual(calls[0][:3], ("https://api.openai.com/v1", "sk-user", "gpt-5.4-mini"))
        self.assertFalse(calls[0][3])  # no temperature for OpenAI's GPT-5 models
        self.assertEqual(calls[1][:3], ("https://llm.scads.ai/v1", "scads-shared", "google/gemma-4-26B-A4B-it"))
        self.assertEqual(calls[2][:3], ("https://llm.scads.ai/v1", "scads-shared", "Qwen/Qwen3.8-27B"))
        self.assertTrue(calls[1][3])
        self.assertEqual(client.model_label("ned"), "Qwen/Qwen3.8-27B (llm.scads.ai)")
        self.assertTrue(client.cost_tracked())  # the OpenAI part is priced, ScaDS is free
        self.assertTrue(client.has_pricing())
