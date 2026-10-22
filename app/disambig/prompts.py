"""Prompts and response parsers of the GPTKB 2.0 disambiguation pipeline.

The system prompts in ./prompts are copied verbatim from
https://github.com/Knowledge-aware-AI/GPTKB2_disambiguation_on_the_fly
(Apache-2.0, see prompts/LICENSE-GPTKB2 and prompts/SOURCE), and the user
messages and parsers below mirror that repo's prompter_parser/prompt_schema.py
-- adapted from its Responses-API request bodies to plain chat messages, since
this app talks to any OpenAI-compatible Chat Completions endpoint.

Elicitation is the one exception: it keeps miniGPTKB's own topic-scoped,
multilingual prompt (app.runner.build_elicitation_prompt*), extended with
GPTKB 2.0's "Description of subject" line so homonymous subjects are elicited
as the entity they actually denote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.runner import PROMPT_LANGUAGES, remove_json_delimiters

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _load(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8").strip()


NER_SYSTEM = _load("ner_system.md")
NED1_SYSTEM = _load("ned1_separate_system.md")
NED2_SYSTEM = _load("ned2_separate_system.md")
NEDG_SYSTEM = _load("nedg_system.md")
PD_SYSTEM = _load("pd_separate_system.md")
PDG_SYSTEM = _load("pdg_system.md")
CD_SYSTEM = _load("cd_separate_system.md")
CDG_SYSTEM = _load("cdg_system.md")

# The GPTKB 2.0 NER prompt asks for "english named entit[ies] ... in Latin
# script", which would classify every object of a non-English run as a
# literal. For those runs only that qualifier is dropped; English runs use the
# prompt exactly as published.
_NER_ENGLISH_QUALIFIER = re.compile(r"english (named entity \([^)]*\)) in Latin script")


def ner_system_prompt(language: str) -> str:
    if language == "en":
        return NER_SYSTEM
    return _NER_ENGLISH_QUALIFIER.sub(r"\1", NER_SYSTEM)


# Label for the extra elicitation line, per prompt language.
DESCRIPTION_LABELS: dict[str, str] = {
    "en": "Description of subject",
    "de": "Beschreibung des Subjekts",
    "es": "Descripción del sujeto",
    "fr": "Description du sujet",
    "it": "Descrizione del soggetto",
    "pl": "Opis podmiotu",
    "pt": "Descrição do sujeito",
    "ru": "Описание субъекта",
    "sv": "Beskrivning av subjektet",
    "tr": "Konunun açıklaması",
    "cs": "Popis podmětu",
}


def instance_of_labels(language: str) -> set[str]:
    """The predicate label(s) the elicitation prompt asks the model to use
    for class membership -- "instanceOf", plus its translation for a
    non-English prompt (e.g. "istanzaDi"), taken from the same rule text
    the prompt itself is built from so the two can't disagree.
    """
    labels = {"instanceOf"}
    rule = PROMPT_LANGUAGES.get(language, PROMPT_LANGUAGES["en"])["rule_instance_of"]
    match = re.search(r'"([^"]+)"', rule)
    if match:
        labels.add(match.group(1))
    return labels


def elicitation_user_message(base_prompt: str, language: str, description: str) -> str:
    label = DESCRIPTION_LABELS.get(language, DESCRIPTION_LABELS["en"])
    return f"{base_prompt}\n{label}: {description}"


def seed_description_messages(seed_label: str, topic: str) -> tuple[str, str]:
    """The seed entity has no source triple to describe it from (GPTKB 2.0
    seeds its KB with a hand-written description), so it gets the NEDG
    instructions with the run's topic standing in for the context triple.
    """
    system = NEDG_SYSTEM.replace(
        "You are given a context triple in the form (subject, predicate, object), where the object is the target entity.",
        "You are given the topic of the knowledge base the target entity belongs to.",
    ).replace("Use the triple to infer", "Use the topic to infer").replace(
        "Avoid repeating the information from the triple, unless really essential.", ""
    )
    return system, f"Entity: {seed_label}\nTopic: {topic}\n"


def ner_messages(language: str, subject: str, predicate: str, obj: str) -> tuple[str, str]:
    return ner_system_prompt(language), f"Phrase: {obj} | Statement: [{subject}, {predicate}, {obj}]\n"


def _options(kind: str, labels: list[str], descs: list[str]) -> str:
    return "\n\n".join(
        f'{chr(idx + 64)}. {kind}: "{label}" \nDescription: {desc}'
        for idx, (label, desc) in enumerate(zip(labels, descs), 1)
    )


def option_letters(n_candidates: int) -> tuple[str, str]:
    """The letters of the "None of above" and "Unsure" options: GPTKB 2.0's
    fixed F and G for up to 5 candidates (its top-k), the next free letters
    when more candidates are offered."""
    if n_candidates <= 5:
        return "F", "G"
    return chr(65 + n_candidates), chr(66 + n_candidates)


def ned_parse_options(stage: str, n_candidates: int) -> dict[str, Any]:
    """parse_choice() arguments for a NED1/NED2 answer to n candidates."""
    none, unsure = option_letters(n_candidates)
    if stage == "ned1":
        return {"letters": f"A-{unsure}", "unsure": True, "none": none, "unsure_letter": unsure}
    return {"letters": f"A-{none}", "unsure": False, "none": none, "unsure_letter": unsure}


def ned1_messages(subject: str, predicate: str, obj: str, labels: list[str], descs: list[str]) -> tuple[str, str]:
    none, unsure = option_letters(len(labels))
    fixed = (
        f"{none}. None of above.\n\n"
        f"{unsure}. Unsure - the case is ambiguous/there is not enough information to decide.\n"
    )
    user = (
        f"Target entity: {obj}\n"
        f"Context triple: [{subject}, {predicate}, {obj}]\n"
        f"Options:\n{_options('Entity', labels, descs)}\n\n{fixed}"
    )
    return NED1_SYSTEM, user


# miniGPTKB addition to GPTKB 2.0's description prompts (NEDG, PDG, CDG):
# a topic-specific KB knows what its elements are about, and without it a
# description written from one triple is often a guess about some other
# namesake (e.g. a sitcom character described as an unrelated actress),
# which then derails description-context NED.
TOPIC_INSTRUCTION = (
    "The knowledge base is about the topic given below; describe the {kind} in the sense relevant to that topic."
)


def _with_topic(system: str, kind: str) -> str:
    return f"{system}\n# Topic\n{TOPIC_INSTRUCTION.format(kind=kind)}"


def nedg_messages(subject: str, predicate: str, obj: str, topic: str) -> tuple[str, str]:
    return (
        _with_topic(NEDG_SYSTEM, "entity"),
        f"Entity: {obj}\nTriple: [{subject}, {predicate}, {obj}]\nTopic: {topic}\n",
    )


def ned2_messages(obj: str, obj_description: str, labels: list[str], descs: list[str]) -> tuple[str, str]:
    user = (
        f"Target entity: {obj}\n"
        f"Target entity description: {obj_description}\n"
        f"Options:\n{_options('Entity', labels, descs)}\n\n{option_letters(len(labels))[0]}. None of above.\n"
    )
    return NED2_SYSTEM, user


def pd_messages(subject: str, predicate: str, obj: str, labels: list[str], descs: list[str]) -> tuple[str, str]:
    user = (
        f"Target predicate: {predicate}\n"
        f"Context triple: [{subject}, {predicate}, {obj}]\n"
        f"Options:\n{_options('Predicate', labels, descs)}\n\nF. None of above.\n"
    )
    return PD_SYSTEM, user


def pdg_messages(predicate: str, topic: str) -> tuple[str, str]:
    return _with_topic(PDG_SYSTEM, "relationship"), f"Predicate: {predicate}\nTopic: {topic}\n"


def cd_messages(subject: str, predicate: str, obj: str, labels: list[str], descs: list[str]) -> tuple[str, str]:
    user = (
        f"Target class: {obj}\n"
        f"Context triple: [{subject}, {predicate}, {obj}]\n"
        f"Options:\n{_options('Class', labels, descs)}\n\nF. None of above.\n"
    )
    return CD_SYSTEM, user


def cdg_messages(concept: str, topic: str) -> tuple[str, str]:
    return _with_topic(CDG_SYSTEM, "class"), f"Class: {concept}\nTopic: {topic}\n"


# ---- Parsers -------------------------------------------------------------


class ParsingError(ValueError):
    pass


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_output(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


def _first_json_value(text: str) -> Any:
    value = remove_json_delimiters(clean_output(text))
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    # Fall back to the first complete JSON value starting at a "[" or "{",
    # for models that wrap the JSON in prose (or a fenced block followed by
    # an explanation) despite being told not to.
    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for start, char in enumerate(value):
        if char not in "[{":
            continue
        try:
            return decoder.raw_decode(value, start)[0]
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ParsingError(f"No valid JSON found in response ({last_error})" if last_error else "No JSON found in response")


def parse_elicitation(text: str) -> list[dict[str, str]]:
    """Accepts miniGPTKB's {"s","p","o"} list (what its prompt asks for) as
    well as GPTKB 2.0's {"facts": [{"subject","predicate","object"}]}."""
    data = _first_json_value(text)
    if isinstance(data, dict):
        data = data.get("facts", data.get("triples", data.get("results")))
    if not isinstance(data, list):
        raise ParsingError("Expected a JSON list of triples")
    triples = []
    for row in data:
        if not isinstance(row, dict):
            continue
        s = row.get("s", row.get("subject"))
        p = row.get("p", row.get("predicate"))
        o = row.get("o", row.get("object"))
        if s is None or p is None or o is None:
            continue
        s, p, o = str(s).strip(), str(p).strip(), str(o).strip()
        if s and p and o:
            triples.append({"subject": s, "predicate": p, "object": o})
    return triples


def parse_ner(text: str) -> bool:
    data = _first_json_value(text)
    if not isinstance(data, dict) or "is_ne" not in data:
        raise ParsingError("NER response has no 'is_ne' field")
    value = data["is_ne"]
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
        raise ParsingError(f"Unrecognized is_ne value {value!r}")
    return bool(value)


def _unique_letter(text: str, letters: str) -> str | None:
    content = clean_output(text).strip().upper()
    matches = set(re.findall(rf"\b([{letters}])\b", content))
    return matches.pop() if len(matches) == 1 else None


# miniGPTKB fallback, used only where GPTKB 2.0's rule finds no unique
# letter: models that explain their choice ("Rajesh is a character, so the
# answer is D") contain other capital letters -- including the article "a",
# which the rule uppercases -- and would otherwise be read as "None of above".
_LEADING_LETTER = r"^\W*([{letters}])\W*(?:[.):]|$)"
_STATED_LETTER = r"(?i:answer|option|choice)(?:\s+is)?\s*[:\-]?\s*\**\(?([{letters}])\b"


def _stated_letter(text: str, letters: str) -> str | None:
    content = clean_output(text)
    first_line = content.splitlines()[0] if content else ""
    match = re.match(_LEADING_LETTER.format(letters=letters), first_line)
    if match:
        return match.group(1)
    stated = re.findall(_STATED_LETTER.format(letters=letters), content)
    return stated[-1] if stated and len(set(stated)) == 1 else None


def parse_choice(
    text: str, letters: str = "A-G", unsure: bool = False, none: str = "F", unsure_letter: str = "G"
) -> int | str | None:
    """GPTKB 2.0's option parser: the answer must contain exactly one option
    letter (NED2 only looks for A-F, the others for A-G). Returns the 1-based
    candidate index, None for "None of above" (or an unparsable answer, which
    GPTKB 2.0 treats the same way), or "unsure" for option G where it is
    offered (NED1). Elsewhere G comes back as index 7, which is out of range
    for every candidate list, so the caller resets that item for a retry --
    again as GPTKB 2.0 does.
    """
    letter = _unique_letter(text, letters) or _stated_letter(text, letters)
    if letter is None or letter == none:
        return None
    if letter == unsure_letter and unsure:
        return "unsure"
    return ord(letter) - 64


def parse_description(text: str) -> str:
    content = clean_output(text)
    marker = "Description: "
    if content.startswith(marker):
        content = content.split(marker, 1)[1]
    content = content.strip().strip('"').strip()
    if not content:
        raise ParsingError("Empty description")
    return content
