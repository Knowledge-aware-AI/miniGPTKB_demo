from __future__ import annotations

import json
import math
import random
import re
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha1
from html import escape
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from openai import OpenAI, RateLimitError

from app.usage import UsageCounter


ProgressCallback = Callable[[dict[str, Any]], None]

# Source: https://developers.openai.com/api/docs/models/all (pricing page)
# Not fetched at runtime -- OpenAI's API has no pricing endpoint, so this table
# must be updated by hand whenever OpenAI changes prices or ships new models.
OPENAI_PRICING_LAST_UPDATED = "August 2026"

OPENAI_TEXT_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.6-sol": {"input_per_mtok": 5.0, "output_per_mtok": 30.0},
    "gpt-5.6-terra": {"input_per_mtok": 2.0, "output_per_mtok": 12.0},
    "gpt-5.6-luna": {"input_per_mtok": 0.2, "output_per_mtok": 1.2},
    "gpt-5.6-cyber": {"input_per_mtok": 12.5, "output_per_mtok": 75.0},
    "gpt-5.5": {"input_per_mtok": 5.0, "output_per_mtok": 30.0},
    "gpt-5.5-pro": {"input_per_mtok": 30.0, "output_per_mtok": 180.0},
    "gpt-5.5-cyber": {"input_per_mtok": 12.5, "output_per_mtok": 75.0},
    "gpt-5.4": {"input_per_mtok": 2.5, "output_per_mtok": 15.0},
    "gpt-5.4-mini": {"input_per_mtok": 0.75, "output_per_mtok": 4.5},
    "gpt-5.4-nano": {"input_per_mtok": 0.2, "output_per_mtok": 1.25},
    "gpt-5.4-pro": {"input_per_mtok": 30.0, "output_per_mtok": 180.0},
    "gpt-5.2": {"input_per_mtok": 1.75, "output_per_mtok": 14.0},
    "gpt-5.2-pro": {"input_per_mtok": 21.0, "output_per_mtok": 168.0},
    "gpt-5.1": {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
    "gpt-5": {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
    "gpt-5-mini": {"input_per_mtok": 0.25, "output_per_mtok": 2.0},
    "gpt-5-nano": {"input_per_mtok": 0.05, "output_per_mtok": 0.4},
    "gpt-5-pro": {"input_per_mtok": 15.0, "output_per_mtok": 120.0},
    "gpt-4.1": {"input_per_mtok": 2.0, "output_per_mtok": 8.0},
    "gpt-4.1-mini": {"input_per_mtok": 0.4, "output_per_mtok": 1.6},
    "gpt-4.1-nano": {"input_per_mtok": 0.1, "output_per_mtok": 0.4},
    "gpt-4o": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
    "gpt-4o-2024-05-13": {"input_per_mtok": 5.0, "output_per_mtok": 15.0},
    "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.6},
    "o1": {"input_per_mtok": 15.0, "output_per_mtok": 60.0},
    "o1-pro": {"input_per_mtok": 150.0, "output_per_mtok": 600.0},
    "o3-pro": {"input_per_mtok": 20.0, "output_per_mtok": 80.0},
    "o3": {"input_per_mtok": 2.0, "output_per_mtok": 8.0},
    "o4-mini": {"input_per_mtok": 1.1, "output_per_mtok": 4.4},
    "o3-mini": {"input_per_mtok": 1.1, "output_per_mtok": 4.4},
    "gpt-4-turbo-2024-04-09": {"input_per_mtok": 10.0, "output_per_mtok": 30.0},
    "gpt-4-0613": {"input_per_mtok": 30.0, "output_per_mtok": 60.0},
    "gpt-3.5-turbo": {"input_per_mtok": 0.5, "output_per_mtok": 1.5},
    "gpt-3.5-turbo-0125": {"input_per_mtok": 0.5, "output_per_mtok": 1.5},
    "gpt-3.5-turbo-1106": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
    "gpt-3.5-turbo-instruct": {"input_per_mtok": 1.5, "output_per_mtok": 2.0},
    "davinci-002": {"input_per_mtok": 2.0, "output_per_mtok": 2.0},
    "babbage-002": {"input_per_mtok": 0.4, "output_per_mtok": 0.4},
}

# Which reasoning controls a model has (an on/off switch, effort levels, and
# the exact request fields for them) is found out by asking the model itself
# -- see app/reasoning.py; model lists carry no reliable information about
# it. OpenRouter is the exception: it normalizes reasoning across every model
# it proxies into one shape and reports per-model support in its /models
# response, so it is handled in server.py with the capability below.
#
# A capability has up to two independent parts:
#   "toggle": reasoning can be fully switched on/off (a boolean control).
#   "effort": reasoning intensity is tunable (an ordinal control).
# "targets"/"target" are paths into the request kwargs where the resolved
# value gets written -- more than one where a model needs the same value in
# two places (e.g. gpt-oss on vLLM) to actually take effect.

# OpenRouter's own unified reasoning API, used for any model it reports as
# supporting the "reasoning" parameter (see server.py's OpenRouter capability
# lookup) -- one shape for every provider/model it proxies, so no per-model
# entry is needed here the way ScaDS/OpenAI need one above.
OPENROUTER_REASONING_CAPABILITY: dict[str, Any] = {
    "toggle": {
        "default_on": True,
        "on_value": True,
        "off_value": False,
        "target": ("extra_body", "reasoning", "enabled"),
    },
    "effort": {
        "levels": ["low", "medium", "high"],
        "default": "medium",
        "targets": [("extra_body", "reasoning", "effort")],
    },
}


def get_reasoning_capability(llm_url: str, model: str, api_key: str | None = None) -> dict[str, Any] | None:
    """The reasoning controls (if any) of this model on this endpoint -- the
    single source of truth shared by the runners (which build the real
    requests with it) and the /api/reasoning-capability endpoint (which tells
    the dashboard whether to show a switch or slider), so the two can never
    disagree. Probed once per endpoint and model and then cached (see
    app/reasoning.py); None for a model without reasoning controls.
    """
    from app import reasoning

    return reasoning.capability_for(llm_url, api_key, model)


def _set_nested(container: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = container
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def resolve_reasoning_settings(
    capability: dict[str, Any] | None,
    reasoning_effort: str | None,
    reasoning_enabled: bool | None,
) -> tuple[str | None, bool | None]:
    """The (effort_level, enabled) that will actually be used for a request,
    accounting for the capability's own default when the caller didn't
    choose one (or chose something invalid for this model) -- so a run
    started without ever touching the new controls resolves to exactly what
    this model already did before this feature existed. Each element is
    None when the model has no such capability at all (e.g. effort is None
    for a toggle-only model like Kimi). Shared by apply_reasoning_settings
    (to build the real request) and anything reporting what was used (the
    HTML report, the database export).
    """
    if not capability:
        return None, None

    resolved_effort = None
    effort_capability = capability.get("effort")
    if effort_capability:
        levels = effort_capability["levels"]
        resolved_effort = reasoning_effort if reasoning_effort in levels else effort_capability["default"]

    resolved_enabled = None
    toggle_capability = capability.get("toggle")
    if toggle_capability:
        resolved_enabled = reasoning_enabled if reasoning_enabled is not None else toggle_capability["default_on"]

    return resolved_effort, resolved_enabled


def apply_reasoning_settings(
    request_args: dict[str, Any],
    capability: dict[str, Any] | None,
    reasoning_effort: str | None,
    reasoning_enabled: bool | None,
) -> None:
    """Writes the resolved reasoning effort/toggle into request_args (and its
    extra_body, created only if actually needed) per `capability`'s target
    path(s). A no-op when the model has no known reasoning capability.
    """
    if not capability:
        return

    resolved_effort, resolved_enabled = resolve_reasoning_settings(capability, reasoning_effort, reasoning_enabled)

    effort_capability = capability.get("effort")
    if effort_capability and resolved_effort is not None:
        for target in effort_capability["targets"]:
            if target[0] == "extra_body":
                _set_nested(request_args.setdefault("extra_body", {}), target[1:], resolved_effort)
            else:
                _set_nested(request_args, target, resolved_effort)

    toggle_capability = capability.get("toggle")
    if toggle_capability and resolved_enabled is not None:
        value = toggle_capability["on_value"] if resolved_enabled else toggle_capability["off_value"]
        target = toggle_capability["target"]
        if target[0] == "extra_body":
            _set_nested(request_args.setdefault("extra_body", {}), target[1:], value)
        else:
            _set_nested(request_args, target, value)


# Elicitation-prompt translations. Adapted from the per-language prompt
# templates in the babylonGPTKB project (including their per-language
# "instanceOf" predicate translations), extended with the JSON-formatting
# instructions this app needs since it doesn't use a structured
# response_format like those templates did.
PROMPT_LANGUAGES: dict[str, dict[str, str]] = {
    "en": {
        "name": "English",
        "intro": "I want to construct a knowledge graph on the topic of {topic}. Given a subject entity, return all facts that you know for the subject as a list of subject, predicate, object triples. The number of facts may be very high, between {popular_low} to {popular_high} or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like {rare_low} or {rare_high}.",
        "important_header": "Important:",
        "rule_unknown": "If you don't know the subject, return an empty list.",
        "rule_not_named_entity": "If the subject is not a named entity, return an empty list.",
        "rule_off_topic": "If the subject does not belong to the topic of {topic}, return an empty list.",
        "rule_instance_of": 'If the subject is a named entity, include at least one triple where predicate is "instanceOf".',
        "rule_not_wordy": "Do not get too wordy.",
        "rule_separate_objects": "Separate several objects into multiple triples with one object.",
        "rule_language": None,
        "rule_json_format": 'Format the output as a structured JSON, a list of dictionaries with 3 keys "s", "p", and "o" each, and the respective values.',
        "rule_no_extra_text": "Don't generate any other text except for the triples in JSON format.",
        "schema_example_label": "Use these example triples as style/schema guidance:",
        "schema_fixed_label": "Use only this fixed schema/ontology guidance:",
        "subject_label": "Subject",
    },
    "de": {
        "name": "German",
        "intro": "Ich möchte einen Knowledge Graph zum Thema {topic} erstellen. Gegeben ein Subjekt, gib mir alle Fakten, die du über dieses Subjekt kennst, als eine Liste von Subjekt-Prädikat-Objekt-Tripeln zurück. Die Anzahl der Fakten kann sehr hoch sein, zwischen {popular_low} und {popular_high} oder mehr, für sehr populäre Subjekte. Bei weniger populären Subjekten kann die Anzahl der Fakten sehr gering sein, etwa {rare_low} oder {rare_high}.",
        "important_header": "Wichtig:",
        "rule_unknown": "Wenn du das Subjekt nicht kennst, gib eine leere Liste zurück.",
        "rule_not_named_entity": "Wenn das Subjekt keine benannte Entität ist, gib eine leere Liste zurück.",
        "rule_off_topic": "Wenn das Subjekt nicht zum Thema {topic} gehört, gib eine leere Liste zurück.",
        "rule_instance_of": 'Wenn das Subjekt eine benannte Entität ist, füge mindestens ein Tripel hinzu, dessen Prädikat "InstanzVon" ist.',
        "rule_not_wordy": "Sei nicht zu weitschweifig.",
        "rule_separate_objects": "Trenne mehrere Objekte in mehrere Tripel mit jeweils einem Objekt.",
        "rule_language": "Alle Tripel müssen auf Deutsch sein.",
        "rule_json_format": 'Formatiere die Ausgabe als strukturiertes JSON, eine Liste von Objekten mit den drei Schlüsseln "s", "p" und "o" und den jeweiligen Werten.',
        "rule_no_extra_text": "Erzeuge keinen weiteren Text außer den Tripeln im JSON-Format.",
        "schema_example_label": "Verwende diese Beispieltripel als Stil-/Schema-Vorgabe:",
        "schema_fixed_label": "Verwende ausschließlich diese feste Schema-/Ontologie-Vorgabe:",
        "subject_label": "Subjekt",
    },
    "es": {
        "name": "Spanish",
        "intro": "Quiero construir un knowledge graph sobre el tema de {topic}. Dada una entidad sujeto, devuelve todos los hechos que conoces para el sujeto como una lista de tripletas (sujeto, predicado, objeto). El número de hechos puede ser muy alto, entre {popular_low} y {popular_high} o más, para sujetos muy populares. Para sujetos menos populares, el número de hechos puede ser muy bajo, como {rare_low} o {rare_high}.",
        "important_header": "Importante:",
        "rule_unknown": "Si no conoces el sujeto, devuelve una lista vacía.",
        "rule_not_named_entity": "Si el sujeto no es una entidad nombrada, devuelve una lista vacía.",
        "rule_off_topic": "Si el sujeto no pertenece al tema de {topic}, devuelve una lista vacía.",
        "rule_instance_of": 'Si el sujeto es una entidad nombrada, incluye al menos un triple cuyo predicado sea "instanciaDe".',
        "rule_not_wordy": "No te extiendas demasiado.",
        "rule_separate_objects": "Separa varios objetos en múltiples triples con un objeto cada uno.",
        "rule_language": "Todos los triples deben estar en español.",
        "rule_json_format": 'Formatea la salida como un JSON estructurado, una lista de diccionarios con las tres claves "s", "p" y "o" y sus respectivos valores.',
        "rule_no_extra_text": "No generes ningún otro texto aparte de los triples en formato JSON.",
        "schema_example_label": "Usa estos triples de ejemplo como guía de estilo/esquema:",
        "schema_fixed_label": "Usa únicamente esta guía de esquema/ontología fija:",
        "subject_label": "Sujeto",
    },
    "fr": {
        "name": "French",
        "intro": "Je souhaite construire un graphe de connaissances sur le sujet de {topic}. Étant donné une entité sujet, retourne tous les faits que tu connais à propos de ce sujet sous forme de triplets (sujet, prédicat, objet). Le nombre de faits peut être très élevé, entre {popular_low} et {popular_high} ou plus, pour des sujets très populaires. Pour des sujets moins populaires, le nombre de faits peut être très faible, comme {rare_low} ou {rare_high}.",
        "important_header": "Important :",
        "rule_unknown": "Si tu ne connais pas le sujet, retourne une liste vide.",
        "rule_not_named_entity": "Si le sujet n'est pas une entité nommée, retourne une liste vide.",
        "rule_off_topic": "Si le sujet ne relève pas du thème de {topic}, retourne une liste vide.",
        "rule_instance_of": 'Si le sujet est une entité nommée, inclus au moins un triplet où le prédicat est "instanceDe".',
        "rule_not_wordy": "Ne sois pas trop verbeux.",
        "rule_separate_objects": "Sépare plusieurs objets en plusieurs triplets avec un seul objet chacun.",
        "rule_language": "Tous les triplets doivent être en français.",
        "rule_json_format": 'Formate la sortie en JSON structuré, une liste de dictionnaires avec les trois clés "s", "p" et "o" et leurs valeurs respectives.',
        "rule_no_extra_text": "Ne génère aucun autre texte que les triplets au format JSON.",
        "schema_example_label": "Utilise ces triplets d'exemple comme guide de style/schéma :",
        "schema_fixed_label": "Utilise uniquement ce schéma/cette ontologie fixe comme guide :",
        "subject_label": "Sujet",
    },
    "it": {
        "name": "Italian",
        "intro": "Voglio costruire un knowledge graph sul tema di {topic}. Data un'entità soggetto, fornisci tutti i fatti che conosci per il soggetto in forma di lista di triple (soggetto, predicato, oggetto). Il numero di fatti può essere molto alto, da {popular_low} a {popular_high} o più, per soggetti molto popolari. Per i soggetti meno popolari, il numero di fatti può essere molto basso, come {rare_low} o {rare_high}.",
        "important_header": "Importante:",
        "rule_unknown": "Se non conosci il soggetto, fornisci una lista vuota.",
        "rule_not_named_entity": "Se il soggetto non è una named entity, fornisci una lista vuota.",
        "rule_off_topic": "Se il soggetto non appartiene al tema di {topic}, fornisci una lista vuota.",
        "rule_instance_of": 'Se il soggetto è una named entity, includi almeno una tripla dove il predicato è "istanzaDi".',
        "rule_not_wordy": "Non essere troppo prolisso.",
        "rule_separate_objects": "Separa più oggetti in più triple con un oggetto ciascuna.",
        "rule_language": "Tutte le triple devono essere in lingua italiana.",
        "rule_json_format": 'Formatta l\'output come JSON strutturato, una lista di dizionari con le tre chiavi "s", "p" e "o" e i rispettivi valori.',
        "rule_no_extra_text": "Non generare alcun altro testo oltre alle triple in formato JSON.",
        "schema_example_label": "Usa queste triple di esempio come guida di stile/schema:",
        "schema_fixed_label": "Usa esclusivamente questa guida di schema/ontologia fissa:",
        "subject_label": "Soggetto",
    },
    "pl": {
        "name": "Polish",
        "intro": "Chcę zbudować knowledge graph na temat {topic}. Dla danej encji będącej podmiotem, podaj wszystkie fakty, które znasz o tym podmiocie w formie listy trójek (podmiot, predykat, obiekt). Liczba faktów może być bardzo duża, od {popular_low} do {popular_high} lub więcej, dla bardzo popularnych podmiotów. Dla mniej popularnych podmiotów liczba faktów może być bardzo mała, na przykład {rare_low} lub {rare_high}.",
        "important_header": "Ważne:",
        "rule_unknown": "Jeśli nie znasz podmiotu, podaj pustą listę.",
        "rule_not_named_entity": "Jeśli podmiot nie jest nazwą własną, podaj pustą listę.",
        "rule_off_topic": "Jeśli podmiot nie należy do tematu {topic}, podaj pustą listę.",
        "rule_instance_of": 'Jeśli podmiot jest nazwą własną, uwzględnij przynajmniej jedną trójkę, w której predykatem jest "instancja".',
        "rule_not_wordy": "Nie bądź zbyt rozwlekły.",
        "rule_separate_objects": "Oddzielaj wiele obiektów w osobnych trójkach z jednym obiektem każda.",
        "rule_language": "Wszystkie trójki muszą być w języku polskim.",
        "rule_json_format": 'Sformatuj dane wyjściowe jako uporządkowany JSON, listę słowników z trzema kluczami "s", "p" i "o" oraz odpowiednimi wartościami.',
        "rule_no_extra_text": "Nie generuj żadnego innego tekstu poza trójkami w formacie JSON.",
        "schema_example_label": "Użyj tych przykładowych trójek jako wskazówki dotyczącej stylu/schematu:",
        "schema_fixed_label": "Użyj wyłącznie tej stałej wskazówki dotyczącej schematu/ontologii:",
        "subject_label": "Podmiot",
    },
    "pt": {
        "name": "Portuguese",
        "intro": "Quero construir um knowledge graph sobre o tema de {topic}. Dada uma entidade sujeito, forneça todos os fatos que você conhece sobre o sujeito em forma de lista de triplas (sujeito, predicado, objeto). O número de fatos pode ser muito alto, de {popular_low} a {popular_high} ou mais, para sujeitos muito populares. Para os sujeitos menos populares, o número de fatos pode ser muito baixo, como {rare_low} ou {rare_high}.",
        "important_header": "Importante:",
        "rule_unknown": "Se você não conhece o sujeito, forneça uma lista vazia.",
        "rule_not_named_entity": "Se o sujeito não é uma named entity, forneça uma lista vazia.",
        "rule_off_topic": "Se o sujeito não pertence ao tema de {topic}, forneça uma lista vazia.",
        "rule_instance_of": 'Se o sujeito é uma named entity, inclua pelo menos uma tripla onde o predicado seja "instânciaDe".',
        "rule_not_wordy": "Não seja demasiado prolixo.",
        "rule_separate_objects": "Separe vários objetos em várias triplas com um objeto cada.",
        "rule_language": "Todas as triplas devem estar em língua portuguesa.",
        "rule_json_format": 'Formate a saída como um JSON estruturado, uma lista de dicionários com as três chaves "s", "p" e "o" e os respectivos valores.',
        "rule_no_extra_text": "Não gere nenhum outro texto além das triplas em formato JSON.",
        "schema_example_label": "Use estas triplas de exemplo como orientação de estilo/esquema:",
        "schema_fixed_label": "Use apenas esta orientação fixa de esquema/ontologia:",
        "subject_label": "Sujeito",
    },
    "ru": {
        "name": "Russian",
        "intro": "Я хочу построить knowledge graph по теме {topic}. Задав субъект, верни все известные тебе факты о нём в виде списка троек (субъект, предикат, объект). Количество фактов может быть очень большим, от {popular_low} до {popular_high} и более, для очень популярных субъектов. Для менее популярных субъектов количество фактов может быть очень низким, например, {rare_low} или {rare_high}.",
        "important_header": "Важно:",
        "rule_unknown": "Если ты не знаешь субъект, верни пустой список.",
        "rule_not_named_entity": "Если субъект не является именованной сущностью, верни пустой список.",
        "rule_off_topic": "Если субъект не относится к теме {topic}, верни пустой список.",
        "rule_instance_of": 'Если субъект является именованной сущностью, включи хотя бы одну тройку, предикат которой "ЧастныйСлучайДля".',
        "rule_not_wordy": "Не будь слишком многословным.",
        "rule_separate_objects": "Разделяй несколько объектов на несколько троек с одним объектом каждая.",
        "rule_language": "Все тройки должны быть на русском языке.",
        "rule_json_format": 'Отформатируй вывод как структурированный JSON — список словарей с тремя ключами "s", "p" и "o" и соответствующими значениями.',
        "rule_no_extra_text": "Не генерируй никакого другого текста, кроме троек в формате JSON.",
        "schema_example_label": "Используй эти примеры троек как ориентир по стилю/схеме:",
        "schema_fixed_label": "Используй только эту фиксированную схему/онтологию:",
        "subject_label": "Субъект",
    },
    "sv": {
        "name": "Swedish",
        "intro": "Jag vill bygga en knowledge graph om ämnet {topic}. Givet ett subjekt som entitet, ge alla fakta du känner till om subjektet i form av en lista med tripler (subjekt, predikat, objekt). Antalet fakta kan vara mycket högt, från {popular_low} till {popular_high} eller fler, för mycket populära subjekt. För mindre populära subjekt kan antalet fakta vara mycket lågt, som {rare_low} eller {rare_high}.",
        "important_header": "Viktigt:",
        "rule_unknown": "Om du inte känner till subjektet, ge en tom lista.",
        "rule_not_named_entity": "Om subjektet inte är en namngiven entitet, ge en tom lista.",
        "rule_off_topic": "Om subjektet inte tillhör ämnet {topic}, ge en tom lista.",
        "rule_instance_of": 'Om subjektet är en namngiven entitet, inkludera minst en trippel där predikatet är "ärInstansAv".',
        "rule_not_wordy": "Var inte alltför långrandig.",
        "rule_separate_objects": "Separera flera objekt i flera tripler med ett objekt vardera.",
        "rule_language": "Alla tripler ska vara på svenska.",
        "rule_json_format": 'Formatera utdata som strukturerad JSON, en lista med dictionaries med de tre nycklarna "s", "p" och "o" och respektive värden.',
        "rule_no_extra_text": "Generera ingen annan text än triplerna i JSON-format.",
        "schema_example_label": "Använd dessa exempeltripler som stil-/schemavägledning:",
        "schema_fixed_label": "Använd endast denna fasta schema-/ontologivägledning:",
        "subject_label": "Subjekt",
    },
    "tr": {
        "name": "Turkish",
        "intro": "{topic} konusu üzerine bir knowledge graph oluşturmak istiyorum. Bir konu (özne) varlığı verildiğinde, bu konu hakkında bildiğin tüm gerçekleri (özne, yüklem, nesne) üçlüleri listesi olarak döndür. Gerçeklerin sayısı çok popüler konular için {popular_low} ile {popular_high} veya daha fazla olabilir. Daha az popüler konular için gerçek sayısı {rare_low} veya {rare_high} kadar çok düşük olabilir.",
        "important_header": "Önemli:",
        "rule_unknown": "Konuyu bilmiyorsan, boş liste döndür.",
        "rule_not_named_entity": "Konu adlandırılmış bir varlık değilse, boş liste döndür.",
        "rule_off_topic": "Konu {topic} temasına ait değilse, boş liste döndür.",
        "rule_instance_of": 'Konu adlandırılmış bir varlık ise, yüklemi "nedir" olan en az bir üçlü ekle.',
        "rule_not_wordy": "Çok ayrıntıya girme.",
        "rule_separate_objects": "Birden fazla nesneyi, her biri tek nesne içeren birden fazla üçlüye ayır.",
        "rule_language": "Tüm üçlüler Türkçe olmalıdır.",
        "rule_json_format": 'Çıktıyı yapılandırılmış JSON olarak biçimlendir: her biri "s", "p" ve "o" anahtarlarına ve ilgili değerlere sahip sözlüklerden oluşan bir liste.',
        "rule_no_extra_text": "JSON formatındaki üçlüler dışında başka hiçbir metin üretme.",
        "schema_example_label": "Bu örnek üçlüleri stil/şema rehberi olarak kullan:",
        "schema_fixed_label": "Yalnızca bu sabit şema/ontoloji rehberini kullan:",
        "subject_label": "Konu",
    },
    "cs": {
        "name": "Czech",
        "intro": "Chci vytvořit knowledge graph na téma {topic}. Pro daný podmět (entitu) uveď všechna fakta, která o tomto podmětu znáš, ve formě seznamu trojic podmět, predikát, objekt. Počet faktů může být velmi vysoký, mezi {popular_low} až {popular_high} nebo více, pro velmi populární podměty. U méně populárních podmětů může být počet faktů velmi nízký, například {rare_low} nebo {rare_high}.",
        "important_header": "Důležité:",
        "rule_unknown": "Pokud podmět neznáš, vrať prázdný seznam.",
        "rule_not_named_entity": "Pokud podmět není vlastní jméno (named entity), vrať prázdný seznam.",
        "rule_off_topic": "Pokud podmět nepatří k tématu {topic}, vrať prázdný seznam.",
        "rule_instance_of": 'Pokud je podmět vlastní jméno (named entity), zahrň alespoň jednu trojici, kde je predikátem "instance".',
        "rule_not_wordy": "Nebuď příliš zdlouhavý.",
        "rule_separate_objects": "Rozděl více objektů do více trojic, každou s jedním objektem.",
        "rule_language": "Všechny trojice musí být v češtině.",
        "rule_json_format": 'Formátuj výstup jako strukturovaný JSON, seznam slovníků se třemi klíči "s", "p" a "o" a příslušnými hodnotami.',
        "rule_no_extra_text": "Negeneruj žádný jiný text kromě trojic ve formátu JSON.",
        "schema_example_label": "Použij tyto ukázkové trojice jako vodítko pro styl/schéma:",
        "schema_fixed_label": "Použij výhradně toto pevné schéma/ontologické vodítko:",
        "subject_label": "Podmět",
    },
}


def build_elicitation_prompt_body(
    language: str,
    topic: str,
    facts_multiplier: float,
    schema_mode: str,
    schema_text: str,
) -> str:
    """Builds the elicitation instructions, excluding the trailing subject line.

    The subject line is kept separate (see append_subject_line) so a
    user-edited prompt template can still have the correct subject
    substituted in for every different entity crawled during a run.
    """
    lang = PROMPT_LANGUAGES.get(language, PROMPT_LANGUAGES["en"])

    popular_low = max(1, round(50 * facts_multiplier))
    popular_high = max(popular_low, round(100 * facts_multiplier))
    rare_low = max(1, round(5 * facts_multiplier))
    rare_high = max(rare_low, round(10 * facts_multiplier))

    intro = lang["intro"].format(
        topic=topic, popular_low=popular_low, popular_high=popular_high, rare_low=rare_low, rare_high=rare_high
    )

    rules = [
        lang["rule_unknown"],
        lang["rule_not_named_entity"],
        lang["rule_off_topic"].format(topic=topic),
        lang["rule_instance_of"],
        lang["rule_not_wordy"],
        lang["rule_separate_objects"],
    ]
    if lang.get("rule_language"):
        rules.append(lang["rule_language"])
    rules.append(lang["rule_json_format"])
    rules.append(lang["rule_no_extra_text"])
    rules_text = "\n".join(f"- {rule}" for rule in rules)

    schema_instruction = ""
    schema_text = schema_text.strip()
    if schema_mode == "example" and schema_text:
        schema_instruction = f"\n{lang['schema_example_label']}\n{schema_text}\n"
    elif schema_mode == "fixed" and schema_text:
        schema_instruction = f"\n{lang['schema_fixed_label']}\n{schema_text}\n"

    return f"{intro}\n\n{lang['important_header']} \n{rules_text}\n{schema_instruction}"


def append_subject_line(prompt_body: str, language: str, subject: str) -> str:
    lang = PROMPT_LANGUAGES.get(language, PROMPT_LANGUAGES["en"])
    body = prompt_body if prompt_body.endswith("\n") else prompt_body + "\n"
    return f'{body}{lang["subject_label"]}: "{subject}"'


def build_elicitation_prompt(
    language: str,
    topic: str,
    subject: str,
    facts_multiplier: float,
    schema_mode: str,
    schema_text: str,
) -> str:
    body = build_elicitation_prompt_body(language, topic, facts_multiplier, schema_mode, schema_text)
    return append_subject_line(body, language, subject)


@dataclass
class KBCConfig:
    llm_url: str
    api_key: str
    model: str
    topic: str
    seed_entity: str
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
    classification_batch_size: int = 30
    max_requests_per_minute: int | None = None
    reasoning_effort: str | None = None
    reasoning_enabled: bool | None = None


class RequestRateLimiter:
    """Caps the rate of outbound LLM requests, shared across all worker threads.

    Worker threads call the LLM concurrently (up to ``nthreads`` at once), so a
    per-thread limiter wouldn't bound the aggregate request rate. This enforces
    a minimum spacing between request starts across the whole runner instead.
    """

    def __init__(self, requests_per_minute: int | None) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self, stop_event: threading.Event) -> bool:
        """Blocks until a request slot is free. Returns False if stopped while waiting."""
        if self._interval <= 0:
            return not stop_event.is_set()
        while not stop_event.is_set():
            with self._lock:
                now = time.monotonic()
                if now >= self._next_slot:
                    self._next_slot = max(now, self._next_slot) + self._interval
                    return True
                wait = self._next_slot - now
            time.sleep(min(wait, 0.5))
        return False


# Cycled through in order when rendering homepage bar/donut charts.
_CHART_PALETTE: list[str] = [
    "#2563eb", "#0f766e", "#b45309", "#be123c", "#7c3aed",
    "#15803d", "#ca8a04", "#0369a1", "#c2410c", "#4d7c0f",
]


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# Third-party assets copied into each run's html/ directory so the generated
# collection stays a self-contained, offline-viewable folder (same reason
# entity pages link each other with plain relative hrefs instead of absolute
# server paths).
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

# Above this many entities, the homepage graph keeps only the most-connected
# ones -- vis-network's physics layout gets sluggish in-browser well before a
# KB's full entity count would (large runs can reach thousands of entities).
_MAX_GRAPH_NODES = 400

# Once a run is past the node cap, which entities make the "most connected"
# cut is recomputed only this often (in calls to _compute_graph_data, i.e.
# roughly every this many entities processed) rather than on every single
# call -- Counter.most_common() over the full entity/object name set is the
# one part of the graph computation that's still O(names seen so far), and
# that ranking is far more stable than the entity count is large, so most
# calls in between reuse it. Only which nodes are shown can lag by up to this
# many entities; the edges among whichever nodes are currently selected are
# still updated on every single call (see _compute_graph_data).
_GRAPH_SELECTION_REFRESH_INTERVAL = 200

# The Entity Index and All Elicited Triples sections show the first this many
# (by discovery order), frozen once that many are reached, rather than every
# entity/triple ever collected -- summary stats (counts, charts) still cover
# everything regardless of this cap; only these two browsable lists are
# capped, on the assumption nobody scrolls through tens of thousands of rows
# by hand anyway. The full data is always in knowledgeBase.db.
_MAX_DISPLAYED_ENTITIES = 3000
_MAX_DISPLAYED_TRIPLES = 5000


class LocalKBCRunner:
    def __init__(
        self,
        config: KBCConfig,
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
        client_timeout = config.request_timeout_seconds
        if self._is_openai_gpt5_request():
            client_timeout = max(client_timeout, 300.0)
        self.client = OpenAI(
            base_url=config.llm_url,
            api_key=config.api_key,
            timeout=client_timeout,
        )
        self.rate_limiter = RequestRateLimiter(config.max_requests_per_minute)
        self.usage = UsageCounter(run_dir / "usage.json")

        self.subject_events_path = run_dir / "subjectEvents.jsonl"
        self.triples_output_path = run_dir / "triples.jsonl"
        self.literals_path = run_dir / "literals.jsonl"
        self.off_domain_entities_path = run_dir / "offDomainEntities.jsonl"
        self.failed_subjects_path = run_dir / "failedSubjects.jsonl"
        self.html_dir = run_dir / "html"
        self.state_lock = threading.RLock()
        self.queue_condition = threading.Condition(self.state_lock)
        self.subject_queue: deque[str] = deque()
        self.queued_subjects: set[str] = set()
        self.processed_subjects: set[str] = set()
        self.known_objects: set[str] = set()
        self.depths: dict[str, int] = {}
        self.triples_by_subject: dict[str, list[tuple[str, str]]] = {}
        self.triples_by_object: dict[str, list[tuple[str, str]]] = {}
        self.triples_in_order: list[tuple[str, str, str]] = []
        # Maintained incrementally (updated the moment a triple/literal/off-
        # domain phrase/failure is recorded) rather than recomputed from the
        # full triple set on every homepage regeneration -- see _compute_stats
        # and _compute_graph_data, which used to rebuild these from scratch
        # on every single entity processed.
        self.predicate_counts: Counter[str] = Counter()
        self.class_counts: Counter[str] = Counter()
        self.degree_counts: Counter[str] = Counter()
        self.out_degree: Counter[str] = Counter()
        self.in_degree: Counter[str] = Counter()
        self.depth_counts: Counter[int] = Counter()
        self.literal_texts: list[str] = []
        self.off_domain_texts: list[str] = []
        self.literals_count = 0
        self.off_domain_count = 0
        self.failed_subjects_count = 0
        self.failure_records: list[dict[str, Any]] = []
        self._frozen_entity_links_html: str | None = None
        self._frozen_triples_json: str | None = None
        # See _compute_graph_data: as long as the top-connected node
        # selection hasn't changed since the last call, only triples added
        # since then need to be scanned for new edges, instead of rescanning
        # every triple ever collected on every single regeneration.
        self._graph_selected_cache: frozenset[str] = frozenset()
        self._graph_selection_refresh_countdown = 0
        self._graph_edges_cache: list[dict[str, str]] = []
        self._graph_edges_scanned_upto = 0
        # See _write_sqlite_export: a single connection kept open for the
        # run's lifetime, written to incrementally (only rows new since the
        # last call), rather than the whole database being rebuilt from
        # scratch on every entity -- the one part of this class where the
        # full data (uncapped, unlike the HTML report) genuinely has to be
        # persisted in full, so its write cost has to stay proportional to
        # what's new, not to everything collected so far.
        self._db_connection: sqlite3.Connection | None = None
        self._db_triples_scanned_upto = 0
        self._db_entities_scanned_upto = 0
        self._db_literals_scanned_upto = 0
        self._db_off_domain_scanned_upto = 0
        self._db_failures_scanned_upto = 0
        self._db_pending_entity_updates: list[str] = []
        self.active_workers = 0
        self.started_at = 0.0
        self.total_cost_usd = 0.0
        self.openrouter_pricing: dict[str, dict[str, float]] = {}

        # Resolved once (model/endpoint don't change during a run) rather
        # than on every single _prompt_llm call -- also exactly what "what
        # reasoning settings did this run actually use" means, so the HTML
        # report and database export read these same two values instead of
        # recomputing (or guessing at) them separately.
        if self._is_openrouter_endpoint():
            self.reasoning_capability = OPENROUTER_REASONING_CAPABILITY
        else:
            self.reasoning_capability = get_reasoning_capability(config.llm_url, config.model, config.api_key)
        self.resolved_reasoning_effort, self.resolved_reasoning_enabled = resolve_reasoning_settings(
            self.reasoning_capability, config.reasoning_effort, config.reasoning_enabled
        )

    def _normalized_llm_url(self) -> str:
        return self.config.llm_url.strip().rstrip("/").lower()

    def _is_openai_endpoint(self) -> bool:
        return self._normalized_llm_url() == "https://api.openai.com/v1"

    def _is_openrouter_endpoint(self) -> bool:
        return self._normalized_llm_url() == "https://openrouter.ai/api/v1"

    def _is_openai_gpt5_request(self) -> bool:
        return self._is_openai_endpoint() and self.config.model.lower().startswith("gpt-5")

    def _load_openrouter_pricing(self) -> None:
        url = self.config.llm_url.rstrip("/") + "/models"
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.config.api_key}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._log(f"Could not fetch OpenRouter model pricing: HTTP {exc.code}.", level="warning")
            return
        except Exception as exc:
            self._log(f"Could not fetch OpenRouter model pricing: {exc}", level="warning")
            return

        pricing: dict[str, dict[str, float]] = {}
        for model in data.get("data", []):
            model_id = str(model.get("id", "")).strip().lower()
            price = model.get("pricing") or {}
            if not model_id:
                continue
            try:
                prompt_per_token = float(price.get("prompt", 0) or 0)
                completion_per_token = float(price.get("completion", 0) or 0)
            except (TypeError, ValueError):
                continue
            pricing[model_id] = {"prompt_per_token": prompt_per_token, "completion_per_token": completion_per_token}
        self.openrouter_pricing = pricing

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._load_run_state()
        self.started_at = time.monotonic()
        with self.state_lock:
            self._write_sqlite_export()
            self._write_html_index()

        self._log("Knowledge extraction pipeline started.")
        if self._is_openai_endpoint():
            self._log(
                f"OpenAI pricing table last updated: {OPENAI_PRICING_LAST_UPDATED}. "
                "Verify current prices at https://developers.openai.com/api/docs/pricing before relying on cost tracking.",
                level="warning",
            )
        elif self._is_openrouter_endpoint():
            self._load_openrouter_pricing()
        self._emit_status("running")

        threads = [
            threading.Thread(target=self._worker, args=(worker_id,), daemon=True)
            for worker_id in range(self.config.nthreads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        final_status = "stopped" if self.stop_event.is_set() else "finished"
        self.usage.flush()
        self._log(f"Knowledge extraction pipeline {final_status}.")
        self._emit_status(final_status)
        with self.state_lock:
            self._write_sqlite_export(final=True)
            self._write_html_index()

    def _worker(self, worker_id: int) -> None:
        while not self.stop_event.is_set():
            self._wait_if_paused()
            subject = self._take_next_subject()
            if subject is None:
                return

            try:
                triples = self._get_triples(subject)
                if self.stop_event.is_set():
                    self._cancel_subject(subject)
                    return
                stored_count, refresh_targets = self._store_triples(triples)
                if self.stop_event.is_set():
                    self._cancel_subject(subject)
                    return
                new_subjects = self._collect_new_subjects(subject, triples)
                if self.stop_event.is_set():
                    self._cancel_subject(subject)
                    return
                self._write_entity_html(subject, new_subjects)
                for target in refresh_targets:
                    self._write_entity_html(target)
                self._finish_subject(subject, new_subjects)
                self._log(
                    f"Processed {subject}: stored {stored_count} triples, queued {len(new_subjects)} new objects."
                )
                self._emit_status("running")
            except Exception as exc:
                self._log(f"Error processing {subject}: {str(exc)[:160]}", level="error")
                if not self.stop_event.is_set():
                    self._record_failure(subject, "processing", str(exc))
                self._finish_subject(subject, [])

    def _take_next_subject(self) -> str | None:
        with self.queue_condition:
            while not self.stop_event.is_set():
                if self._should_stop():
                    self.stop_event.set()
                    self.queue_condition.notify_all()
                    return None

                if self.subject_queue:
                    subject = self.subject_queue.popleft()
                    self.queued_subjects.discard(subject)
                    self.active_workers += 1
                    self._log_subject_event("dequeue", subject)
                    return subject

                if self.active_workers == 0:
                    self.queue_condition.notify_all()
                    return None

                self.queue_condition.wait(timeout=0.5)
            return None

    def _finish_subject(self, subject: str, new_subjects: list[str]) -> None:
        with self.queue_condition:
            self.processed_subjects.add(subject)
            self._log_subject_event("finish", subject)
            self._db_pending_entity_updates.append(subject)
            for new_subject in new_subjects:
                if new_subject in self.processed_subjects or new_subject in self.queued_subjects:
                    continue
                self.subject_queue.append(new_subject)
                self.queued_subjects.add(new_subject)
                self._log_subject_event("enqueue", new_subject, self.depths.get(new_subject, 0))

            self.active_workers = max(0, self.active_workers - 1)
            self.queue_condition.notify_all()

    def _cancel_subject(self, subject: str) -> None:
        with self.queue_condition:
            if subject not in self.processed_subjects and subject not in self.queued_subjects:
                self.subject_queue.appendleft(subject)
                self.queued_subjects.add(subject)
                self._log_subject_event("enqueue", subject, self.depths.get(subject, 0))
            self.active_workers = max(0, self.active_workers - 1)
            self.queue_condition.notify_all()

    def _record_failure(self, subject: str, stage: str, error: str, detail: str | None = None) -> None:
        """Appends one entry to the unified failedSubjects.jsonl log -- the single
        place every "something went wrong for this subject" case is recorded,
        regardless of which stage of processing it happened in, so the
        generated HTML report can list them all together later.
        """
        record: dict[str, Any] = {
            "subject": subject,
            "stage": stage,
            "error": error,
            "time": time.strftime("%H:%M:%S"),
        }
        if detail is not None:
            record["detail"] = detail
        self._append_jsonl(self.failed_subjects_path, record)
        with self.state_lock:
            self.failure_records.append(record)
            self.failed_subjects_count += 1

    def _get_triples(self, subject: str) -> list[dict[str, Any]]:
        self.emit({"type": "current_subject", "subject": subject})
        self._log(f"Processing {subject}...")
        try:
            output = self._prompt_llm(self._build_elicitation_prompt(subject))
            try:
                triples = json.loads(remove_json_delimiters(output))
                if not isinstance(triples, list):
                    raise ValueError("Expected a JSON list of triples.")
                return triples
            except Exception as exc:
                self._record_failure(subject, "json_parse", str(exc), detail=output)
                self._log(f"Failed to parse JSON for {subject}.", level="warning")
                return []
        except Exception as exc:
            # _call_llm already retried this request several times with
            # backoff before giving up (see _call_llm) -- a deliberate Stop
            # surfaces here too (it's the same exception path), but that's
            # not a failed entity, so it's excluded from the failure log and
            # the subject gets re-queued by _worker's own stop check instead
            # of being abandoned.
            if not self.stop_event.is_set():
                self._record_failure(subject, "llm_request", str(exc))
                self._log(
                    f"Giving up on {subject} after repeated LLM failures ({exc}); skipping it and continuing.",
                    level="error",
                )
            return []

    def _build_elicitation_prompt(self, subject: str) -> str:
        template = self.config.prompt_template.strip()
        if template:
            return append_subject_line(template, self.config.language, subject)
        return build_elicitation_prompt(
            self.config.language,
            self.config.topic,
            subject,
            self.config.facts_multiplier,
            self.config.schema_mode,
            self.config.schema_text,
        )

    def _prompt_llm(self, message: str, role: str = "elicitation") -> str:
        request_args: dict[str, Any] = {
            "messages": [{"role": "user", "content": message}],
            "model": self.config.model,
        }

        apply_reasoning_settings(
            request_args, self.reasoning_capability, self.config.reasoning_effort, self.config.reasoning_enabled
        )

        # OpenAI's GPT-5 family on /v1/chat/completions rejects temperature=0
        # and expects the default sampling behavior instead.
        if not self._is_openai_gpt5_request():
            request_args["temperature"] = 0

        return self._call_llm(request_args, role=role)

    def _call_llm(self, request_args: dict[str, Any], max_retries: int = 5, role: str = "elicitation") -> str:
        # Retries any failure -- rate limits, timeouts, connection drops,
        # malformed/5xx responses, whatever the SDK raises -- not just rate
        # limits, since a busy shared endpoint can fail a request in more
        # ways than that. A deliberate Stop is never retried: it propagates
        # immediately so a worker doesn't sit in a backoff sleep after the
        # user asked the run to stop.
        attempt = 0
        while True:
            if not self.rate_limiter.acquire(self.stop_event):
                raise RuntimeError("Stopped while waiting for the LLM rate limiter.")
            try:
                response = self.client.chat.completions.create(**request_args)
                self._record_usage_cost(response)
                self.usage.record_call(role, response)
                return response.choices[0].message.content or ""
            except Exception as exc:
                if self.stop_event.is_set():
                    raise
                attempt += 1
                self.usage.record_retry(role, isinstance(exc, RateLimitError))
                if attempt > max_retries:
                    raise
                delay = self._retry_delay(exc, attempt)
                self._log(
                    f"LLM request failed ({type(exc).__name__}: {str(exc)[:160]}); "
                    f"retrying in {delay:.1f}s (attempt {attempt}/{max_retries}).",
                    level="warning",
                )
                if not self._sleep_or_stop(delay):
                    raise RuntimeError("Stopped while waiting to retry after a failed request.") from exc

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        if isinstance(exc, RateLimitError):
            response = getattr(exc, "response", None)
            header_value = response.headers.get("retry-after") if response is not None else None
            if header_value is not None:
                try:
                    return max(0.5, float(header_value))
                except ValueError:
                    pass
        return min(60.0, 2**attempt) + random.uniform(0, 1)

    def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleeps up to ``seconds``, waking early to check for a stop request."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                return False
            time.sleep(min(0.5, deadline - time.monotonic()))
        return not self.stop_event.is_set()

    def _record_usage_cost(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        if self._is_openai_endpoint():
            pricing = OPENAI_TEXT_PRICING.get(self.config.model.strip().lower())
            if pricing is None:
                return
            cost = (
                (input_tokens / 1_000_000) * pricing["input_per_mtok"]
                + (output_tokens / 1_000_000) * pricing["output_per_mtok"]
            )
        elif self._is_openrouter_endpoint():
            pricing = self.openrouter_pricing.get(self.config.model.strip().lower())
            if pricing is None:
                return
            cost = (
                input_tokens * pricing["prompt_per_token"]
                + output_tokens * pricing["completion_per_token"]
            )
        else:
            return

        with self.state_lock:
            self.total_cost_usd += cost

    def _collect_new_subjects(self, subject: str, triples: list[dict[str, Any]]) -> list[str]:
        source_depth = self.depths.get(subject, 0)
        candidate_depth = source_depth + 1
        if self.config.max_bfs_depth is not None and candidate_depth > self.config.max_bfs_depth:
            return []

        candidates = []
        for obj in [triple.get("o", "") for triple in triples]:
            if not obj:
                continue

            with self.state_lock:
                if obj in self.known_objects:
                    continue
                self.known_objects.add(obj)
            candidates.append(str(obj))

        new_subjects = []
        for batch in chunked(candidates, self.config.classification_batch_size):
            self._wait_if_paused()
            if self.stop_event.is_set():
                break
            classifications = self._classify_objects(batch)
            for obj in batch:
                category = classifications.get(obj, "literal")
                if category == "literal":
                    self._append_jsonl(self.literals_path, {"text": obj})
                    with self.state_lock:
                        self.literal_texts.append(obj)
                        self.literals_count += 1
                    continue
                if category == "off_domain_entity":
                    self._append_jsonl(self.off_domain_entities_path, {"text": obj})
                    with self.state_lock:
                        self.off_domain_texts.append(obj)
                        self.off_domain_count += 1
                    continue
                with self.state_lock:
                    self.depths[obj] = candidate_depth
                    self.depth_counts[candidate_depth] += 1
                new_subjects.append(obj)
        return new_subjects

    def _classify_objects(self, phrases: list[str]) -> dict[str, str]:
        if not phrases:
            return {}
        response = self._prompt_llm(
            f"I want you to classify object phrases from a knowledge graph about {self.config.topic}.\n\n"
            "Classify each phrase into exactly one of these categories:\n"
            '- "domain_entity": a named entity that is relevant to the topic and should be expanded as a future subject.\n'
            '- "off_domain_entity": a named entity, but not relevant to the topic.\n'
            '- "literal": not a named entity, such as a date, number, URL, title, common noun, description, or verbose phrase.\n\n'
            'Return only JSON in this exact shape: {"results": [{"phrase": "...", "category": "domain_entity|off_domain_entity|literal"}]}.\n'
            "Use the exact input phrase strings in the phrase field.\n\n"
            f"Phrases: {json.dumps(phrases, ensure_ascii=False)}",
            role="ner",
        )
        return self._parse_object_classifications(response, phrases)

    def _parse_object_classifications(self, response: str, phrases: list[str]) -> dict[str, str]:
        value = remove_json_delimiters(response).strip()
        result: dict[str, str] = {}
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            if len(phrases) == 1:
                return {phrases[0]: self._parse_object_category(value)}
            self._log("Could not parse batched object classifications, treating batch as literals.", level="warning")
            return {phrase: "literal" for phrase in phrases}

        rows = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            self._log("Batched object classification did not return a list, treating batch as literals.", level="warning")
            return {phrase: "literal" for phrase in phrases}

        phrase_set = set(phrases)
        for row in rows:
            if not isinstance(row, dict):
                continue
            phrase = str(row.get("phrase", ""))
            if phrase not in phrase_set:
                continue
            result[phrase] = self._normalize_object_category(str(row.get("category", "")))

        missing = [phrase for phrase in phrases if phrase not in result]
        if missing:
            self._log(f"Classification response missed {len(missing)} objects, treating them as literals.", level="warning")
            for phrase in missing:
                result[phrase] = "literal"
        return result

    def _parse_object_category(self, response: str) -> str:
        value = remove_json_delimiters(response).strip()
        try:
            data = json.loads(value)
            category = str(data.get("category", "")).strip().lower()
        except json.JSONDecodeError:
            category = value.strip('"').strip().lower()
        return self._normalize_object_category(category)

    def _normalize_object_category(self, category: str) -> str:
        category = category.strip().lower()
        aliases = {
            "domain-relevant ne": "domain_entity",
            "domain relevant ne": "domain_entity",
            "domain-relevant named entity": "domain_entity",
            "domain relevant named entity": "domain_entity",
            "named entity but not domain-relevant": "off_domain_entity",
            "named entity but not domain relevant": "off_domain_entity",
            "off-domain named entity": "off_domain_entity",
            "off domain named entity": "off_domain_entity",
        }
        category = aliases.get(category, category)
        if category not in {"domain_entity", "off_domain_entity", "literal"}:
            self._log(f"Unknown object classification '{category}', treating as literal.", level="warning")
            return "literal"
        return category

    def _store_triples(self, new_triples: list[dict[str, Any]]) -> tuple[int, list[str]]:
        unique_triples = deduplicate_triples(new_triples)
        refresh_targets: set[str] = set()
        with self.state_lock:
            with self.triples_output_path.open("a", encoding="utf-8") as file:
                for triple in unique_triples:
                    file.write(json.dumps(triple, ensure_ascii=False) + "\n")
                    s = str(triple.get("s", "")).strip()
                    p = str(triple.get("p", "")).strip()
                    o = str(triple.get("o", "")).strip()
                    if not s or not p or not o:
                        continue
                    self.triples_by_subject.setdefault(s, []).append((p, o))
                    self.triples_by_object.setdefault(o, []).append((s, p))
                    self.triples_in_order.append((s, p, o))
                    self.predicate_counts[p] += 1
                    if p.lower() == "instanceof":
                        self.class_counts[o] += 1
                    self.out_degree[s] += 1
                    self.in_degree[o] += 1
                    self.degree_counts[s] += 1
                    self.degree_counts[o] += 1
                    if o != s and o in self.processed_subjects:
                        refresh_targets.add(o)
        return len(unique_triples), sorted(refresh_targets)

    def _write_entity_html(self, subject: str, extra_linkable: list[str] | None = None) -> None:
        with self.state_lock:
            outgoing = list(self.triples_by_subject.get(subject, []))
            incoming = list(self.triples_by_object.get(subject, []))
            linkable_objects = set(self.depths) | set(self.queued_subjects) | set(self.processed_subjects)
        linkable_objects |= set(extra_linkable or [])

        outgoing_rows = []
        for predicate_value, object_value in outgoing:
            object_html = self._format_html_object(object_value, linkable_objects)
            outgoing_rows.append(
                "<tr>"
                f"<td>{escape(subject)}</td>"
                f"<td>{escape(predicate_value)}</td>"
                f"<td>{object_html}</td>"
                "</tr>"
            )
        if not outgoing_rows:
            outgoing_rows.append('<tr><td colspan="3">No outgoing triples returned for this subject.</td></tr>')

        incoming_rows = []
        for subject_value, predicate_value in incoming:
            subject_html = self._format_html_object(subject_value, linkable_objects)
            incoming_rows.append(
                "<tr>"
                f"<td>{subject_html}</td>"
                f"<td>{escape(predicate_value)}</td>"
                f"<td>{escape(subject)}</td>"
                "</tr>"
            )
        if not incoming_rows:
            incoming_rows.append('<tr><td colspan="3">No incoming triples found yet.</td></tr>')

        html = self._entity_html_document(subject, "\n".join(outgoing_rows), "\n".join(incoming_rows))
        with self.state_lock:
            self.html_dir.mkdir(parents=True, exist_ok=True)
            path = self.html_dir / self._entity_filename(subject)
            path.write_text(html, encoding="utf-8")
            self._write_sqlite_export()
            self._write_html_index()

    def _format_html_object(self, value: str, linkable_objects: set[str]) -> str:
        if value in linkable_objects:
            filename = self._entity_filename(value)
            return f'<a href="{escape(quote(filename))}">{escape(value)}</a>'
        return escape(value)

    def _entity_html_document(self, subject: str, outgoing_rows: str, incoming_rows: str) -> str:
        title = escape(subject)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        background: #f8f9fa;
        color: #212529;
        margin: 40px auto;
        max-width: 900px;
        line-height: 1.6;
    }}
    h1 {{
        text-align: center;
        color: #0077cc;
    }}
    h2 {{
        color: #0077cc;
        margin-top: 30px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    th, td {{
        border: 1px solid #dee2e6;
        padding: 10px;
        text-align: left;
        vertical-align: top;
    }}
    th {{
        background: #0077cc;
        color: white;
    }}
    tr:nth-child(even) {{
        background: #f1f3f5;
    }}
    a {{
        color: #0077cc;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    .footer {{
        margin-top: 40px;
        text-align: center;
        font-size: 0.9em;
        color: #6c757d;
    }}
    .nav {{
        margin-bottom: 20px;
        text-align: center;
    }}
</style>
</head>
<body>
<div class="nav"><a href="index.html">Home</a></div>
<h1>{title}</h1>
<h2>Outgoing</h2>
<table>
<tr><th>Subject</th><th>Predicate</th><th>Object</th></tr>
{outgoing_rows}
</table>
<h2>Incoming</h2>
<table>
<tr><th>Subject</th><th>Predicate</th><th>Object</th></tr>
{incoming_rows}
</table>
<div class="footer">miniGPTKB HTML Collection</div>
</body>
</html>
"""

    def _compute_stats(self) -> dict[str, Any]:
        """Computes KB statistics from incrementally-maintained counters.

        Callers must already hold self.state_lock (it's an RLock, but this
        method does not acquire it itself, to avoid taking two consistent
        snapshots that could disagree if state changes in between).

        Every count/chart here is sourced from state updated the moment each
        triple/literal/off-domain phrase/failure is recorded (see
        _store_triples, _collect_new_subjects, _record_failure), not
        recomputed from the full triple set on every call -- this used to be
        a handful of full O(n) passes over every triple ever collected,
        repeated on every single entity processed over the life of a run.
        self.depths is a superset of processed_subjects/queued_subjects (an
        entity is always given a depth at the moment it's discovered, before
        it's ever queued), so its length alone is the true entity count.
        """
        top_entities = [
            (name, self.out_degree.get(name, 0), self.in_degree.get(name, 0), total)
            for name, total in self.degree_counts.most_common(10)
        ]

        return {
            "total_entities": len(self.depths),
            "total_predicates": len(self.predicate_counts),
            "total_literals": self.literals_count,
            "total_triples": len(self.triples_in_order),
            "unique_classes": len(self.class_counts),
            "top_classes": self.class_counts.most_common(10),
            "top_predicates": self.predicate_counts.most_common(10),
            "top_entities": top_entities,
            "depth_counts": sorted(self.depth_counts.items()),
            "max_depth": max(self.depth_counts) if self.depth_counts else 0,
            "total_off_domain": self.off_domain_count,
            "total_failed_subjects": self.failed_subjects_count,
            "failed_subjects": self.failure_records,
            "queued_count": len(self.queued_subjects),
        }

    def _compute_graph_data(self, entity_count: int) -> dict[str, Any]:
        """Builds a capped node/edge graph for the homepage's network visualization.

        Callers must already hold self.state_lock (same convention as
        _compute_stats). Reuses the same incrementally-maintained
        self.degree_counts as _compute_stats's Central Entities table instead
        of rebuilding degree counters from the full triple set again -- and,
        since a large run is always past the node cap, node selection comes
        from Counter.most_common (a bounded operation over distinct entities,
        not one that scans every triple) rather than sorting every entity by
        degree. Takes a count rather than a set of entities so a huge run
        never has to materialize a set of everything just to measure it.
        """
        truncated = entity_count > _MAX_GRAPH_NODES
        if truncated:
            self._graph_selection_refresh_countdown -= 1
            if self._graph_selection_refresh_countdown <= 0 or not self._graph_selected_cache:
                top_nodes = {name for name, _ in self.degree_counts.most_common(_MAX_GRAPH_NODES)}
                selected = frozenset(top_nodes | {self.config.seed_entity})
                self._graph_selection_refresh_countdown = _GRAPH_SELECTION_REFRESH_INTERVAL
            else:
                selected = self._graph_selected_cache
        else:
            selected = frozenset(self.depths)

        nodes = []
        for name in selected:
            is_seed = name == self.config.seed_entity
            depth = self.depths.get(name, 0)
            node_degree = self.degree_counts.get(name, 0)
            detail = "seed entity" if is_seed else f"depth {depth}, {node_degree} connections"
            nodes.append(
                {
                    "id": name,
                    "label": name,
                    "title": f"{name} ({detail})",
                    "color": _CHART_PALETTE[depth % len(_CHART_PALETTE)],
                    "size": 24 if is_seed else min(10 + node_degree * 1.5, 40),
                    "borderWidth": 3 if is_seed else 1,
                }
            )

        # Rescanning every triple ever collected to find the edges among a
        # small selected set, on every single regeneration, is the one part
        # of this function real 100k+-scale measurement showed still costs
        # real time (tens of ms per call, times every entity processed, adds
        # up to hours over a big run). Since node selection is far more
        # stable than the triple set is large (the top-N-by-degree set
        # rarely reshuffles once a graph has any size), only a selection
        # change forces a full rescan; otherwise, only the triples added
        # since the last call -- normally a handful -- are scanned.
        if selected != self._graph_selected_cache:
            self._graph_selected_cache = selected
            self._graph_edges_cache = [
                {"from": s, "to": o, "title": f"{s} — {p} → {o}"}
                for s, p, o in self.triples_in_order
                if s != o and s in selected and o in selected
            ]
        else:
            new_triples = self.triples_in_order[self._graph_edges_scanned_upto :]
            self._graph_edges_cache.extend(
                {"from": s, "to": o, "title": f"{s} — {p} → {o}"}
                for s, p, o in new_triples
                if s != o and s in selected and o in selected
            )
        self._graph_edges_scanned_upto = len(self.triples_in_order)
        edges = self._graph_edges_cache

        return {
            "nodes": nodes,
            "edges": edges,
            "truncated": truncated,
            "total_entities": entity_count,
            "shown_entities": len(selected),
        }

    def _ensure_vendor_assets(self) -> None:
        self.html_dir.mkdir(parents=True, exist_ok=True)
        target = self.html_dir / "vis-network.min.js"
        if not target.exists():
            shutil.copyfile(_VENDOR_DIR / "vis-network.min.js", target)

    def _bar_chart_svg(self, title: str, subtitle: str, rows: list[tuple[str, int]]) -> str:
        """Renders a horizontal bar chart as inline SVG. Returns "" if rows is empty."""
        if not rows:
            return ""
        chart_w = 640
        label_w = 190
        row_h = 24
        bar_h = 14
        top = 56
        height = top + len(rows) * row_h + 12
        max_val = max(value for _, value in rows) or 1
        bar_max = chart_w - label_w - 70
        bars = []
        for i, (label, value) in enumerate(rows):
            y = top + i * row_h
            cy = y + bar_h / 2
            width = max(value / max_val * bar_max, 2.0)
            color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
            bars.append(
                f'<text x="0" y="{cy:.1f}" class="axis-label">{escape(str(label))}</text>'
                f'<rect x="{label_w}" y="{y:.1f}" width="{width:.1f}" height="{bar_h}" rx="4" fill="{color}"></rect>'
                f'<text x="{label_w + width + 8:.1f}" y="{cy:.1f}" class="value-label">{value:,}</text>'
            )
        subtitle_svg = f'<text x="0" y="42" class="chart-subtitle">{escape(subtitle)}</text>' if subtitle else ""
        return f"""<svg class="chart" viewBox="0 0 {chart_w} {height}" role="img" aria-label="{escape(title)}">
<text x="0" y="22" class="chart-title">{escape(title)}</text>
{subtitle_svg}
{"".join(bars)}
</svg>"""

    def _donut_svg(self, title: str, segments: list[tuple[str, int, str]], center_label: str) -> str:
        """Renders a 2+ segment donut chart as inline SVG. Returns "" if all values are 0."""
        total = sum(value for _, value, _ in segments)
        if total <= 0:
            return ""
        cx, cy, r, stroke_w = 130, 130, 76, 30

        def point(angle_deg: float) -> tuple[float, float]:
            rad = math.radians(angle_deg)
            return cx + r * math.sin(rad), cy - r * math.cos(rad)

        nonzero = [segment for segment in segments if segment[1] > 0]
        paths = []
        if len(nonzero) == 1:
            color = nonzero[0][2]
            x1, y1 = point(0)
            x2, y2 = point(180)
            x3, y3 = point(360)
            paths.append(
                f'<path d="M {x1:.2f} {y1:.2f} A {r} {r} 0 1 1 {x2:.2f} {y2:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke_w}" stroke-linecap="round"></path>'
            )
            paths.append(
                f'<path d="M {x2:.2f} {y2:.2f} A {r} {r} 0 1 1 {x3:.2f} {y3:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke_w}" stroke-linecap="round"></path>'
            )
        else:
            angle = 0.0
            for _, value, color in segments:
                if value <= 0:
                    continue
                span = value / total * 360
                x1, y1 = point(angle)
                x2, y2 = point(angle + span)
                large_arc = 1 if span > 180 else 0
                paths.append(
                    f'<path d="M {x1:.2f} {y1:.2f} A {r} {r} 0 {large_arc} 1 {x2:.2f} {y2:.2f}" '
                    f'fill="none" stroke="{color}" stroke-width="{stroke_w}" stroke-linecap="round"></path>'
                )
                angle += span
        legend = []
        for i, (label, value, color) in enumerate(segments):
            ly = 96 + i * 30
            legend.append(
                f'<rect x="230" y="{ly}" width="12" height="12" rx="3" fill="{color}"></rect>'
                f'<text x="250" y="{ly + 10}" class="legend-label">{escape(label)}: {value:,}</text>'
            )
        return f"""<svg class="chart donut" viewBox="0 0 380 260" role="img" aria-label="{escape(title)}">
<text x="0" y="22" class="chart-title">{escape(title)}</text>
{"".join(paths)}
<text x="{cx}" y="{cy - 4}" text-anchor="middle" class="donut-big">{total:,}</text>
<text x="{cx}" y="{cy + 18}" text-anchor="middle" class="donut-small">{escape(center_label)}</text>
{"".join(legend)}
</svg>"""

    def _panel_row(self, svgs: list[str]) -> str:
        """Wraps up to two non-empty chart SVGs in a panel band. Returns "" if none are present."""
        panels = [f'<div class="panel">{svg}</div>' for svg in svgs if svg]
        if not panels:
            return ""
        if len(panels) == 1:
            return f'<section class="band">{panels[0]}</section>'
        return f'<section class="band grid-2">{"".join(panels)}</section>'

    def _write_html_index(self) -> None:
        self._ensure_vendor_assets()
        run_id = self.run_dir.name
        # Discovery order (dicts preserve insertion order), not alphabetical --
        # entity_order[:_MAX_DISPLAYED_ENTITIES] is what actually renders in
        # the Entity Index below, and self.depths is a superset of
        # processed_subjects/queued_subjects (see _compute_stats), so this is
        # every entity ever discovered without needing to union three sets.
        entity_order = list(self.depths.keys())
        total_entity_count = len(entity_order)
        topic = escape(self.config.topic)
        seed = escape(self.config.seed_entity)
        model = escape(self.config.model)
        language = PROMPT_LANGUAGES.get(self.config.language, {}).get("name", self.config.language)
        generated_at = time.strftime("%Y-%m-%d %H:%M")
        elapsed_seconds = time.monotonic() - self.started_at
        runtime = _format_duration(elapsed_seconds)
        # Subjects actually crawled (their own elicitation call completed) per
        # minute -- the crawler's throughput. This is not the same as the
        # "Entities" count above, which also includes every entity merely
        # *discovered* as an object of another subject's triples and still
        # sitting in the queue, so it's typically much larger than
        # (subjects_per_minute * elapsed minutes).
        subjects_per_minute = (
            f"{len(self.processed_subjects) / (elapsed_seconds / 60):.1f}" if elapsed_seconds >= 1 else "0.0"
        )

        if self.config.facts_multiplier <= 0.5:
            triples_level = "Few"
        elif self.config.facts_multiplier >= 1.5:
            triples_level = "Many"
        else:
            triples_level = "Medium"

        schema_mode_labels = {"open": "Open", "example": "Example-driven", "fixed": "Fixed ontology"}
        schema_mode_label = schema_mode_labels.get(self.config.schema_mode, self.config.schema_mode)

        stopping_criteria = []
        if self.config.max_entities:
            stopping_criteria.append(f"Max objects: {self.config.max_entities:,}")
        if self.config.max_bfs_depth is not None:
            stopping_criteria.append(f"Max BFS depth: {self.config.max_bfs_depth}")
        if self.config.max_runtime_seconds:
            stopping_criteria.append(f"Max runtime: {_format_duration(self.config.max_runtime_seconds)}")
        if self.config.max_cost:
            stopping_criteria.append(f"Max cost: ${self.config.max_cost:,.2f}")
        stopping_criteria_display = ", ".join(stopping_criteria) if stopping_criteria else "None set"

        prompt_display = self.config.prompt_template.strip() or build_elicitation_prompt_body(
            self.config.language,
            self.config.topic,
            self.config.facts_multiplier,
            self.config.schema_mode,
            self.config.schema_text,
        )

        reasoning_bits = []
        if self.resolved_reasoning_effort is not None:
            reasoning_bits.append(self.resolved_reasoning_effort.capitalize())
        if self.resolved_reasoning_enabled is not None:
            reasoning_bits.append("Enabled" if self.resolved_reasoning_enabled else "Disabled")
        reasoning_display = ", ".join(reasoning_bits) if reasoning_bits else "Not applicable"

        config_items = [
            ("Language", language),
            ("Runtime", runtime),
            ("Generated", generated_at),
            ("Endpoint", self.config.llm_url),
            ("Parallel subjects", str(self.config.nthreads)),
            ("Desired triples", triples_level),
            ("Max requests/min", f"{self.config.max_requests_per_minute:,}" if self.config.max_requests_per_minute else "Unlimited"),
            ("Processed Entities / Minute", subjects_per_minute),
            ("Schema", schema_mode_label),
            ("Reasoning", reasoning_display),
            ("Stopping criteria", stopping_criteria_display),
        ]
        config_items_html = "\n".join(
            f'<div class="config-item"><span>{escape(name)}</span><strong>{escape(value)}</strong></div>'
            for name, value in config_items
        )

        schema_detail_section = ""
        if self.config.schema_mode != "open" and self.config.schema_text.strip():
            schema_detail_section = f"""<details class="reveal">
<summary>Schema Detail</summary>
<div class="reveal-body"><pre class="prompt-block">{escape(self.config.schema_text.strip())}</pre></div>
</details>"""

        run_configuration_section = f"""<section class="band">
<h2>Run Configuration</h2>
<div class="config-grid">
{config_items_html}
</div>
<details class="reveal">
<summary>Elicitation Prompt</summary>
<div class="reveal-body"><pre class="prompt-block">{escape(prompt_display)}</pre></div>
</details>
{schema_detail_section}
</section>"""

        stats = self._compute_stats()
        graph_data = self._compute_graph_data(total_entity_count)
        graph_nodes_json = json.dumps(graph_data["nodes"], ensure_ascii=False).replace("</", "<\\/")
        graph_edges_json = json.dumps(graph_data["edges"], ensure_ascii=False).replace("</", "<\\/")
        graph_note = (
            f'<p class="graph-note">Showing the {graph_data["shown_entities"]:,} most-connected of '
            f'{graph_data["total_entities"]:,} entities to keep the visualization responsive.</p>'
            if graph_data["truncated"]
            else ""
        )

        remaining_note = (
            f"{stats['queued_count']:,} entities still queued for processing."
            if stats["queued_count"]
            else "All discovered entities have been processed."
        )
        quality_bits = []
        if stats["total_off_domain"]:
            quality_bits.append(f"{stats['total_off_domain']:,} off-domain mentions filtered out")
        if stats["total_failed_subjects"]:
            quality_bits.append(f"{stats['total_failed_subjects']:,} subjects failed (see Failures below)")
        quality_note = f" {' and '.join(quality_bits)}." if quality_bits else ""

        metrics = [
            ("Triples", stats["total_triples"]),
            ("Entities", stats["total_entities"]),
            ("Predicates", stats["total_predicates"]),
            ("Literals", stats["total_literals"]),
            ("Unique classes", stats["unique_classes"]),
            ("Max BFS depth", stats["max_depth"]),
        ]
        metrics_html = "\n".join(
            f'<div class="metric"><span>{escape(name)}</span><strong>{value:,}</strong></div>'
            for name, value in metrics
        )

        top_predicates_chart = self._bar_chart_svg(
            "Top Predicate Labels", "Most frequent edge types in this run.", stats["top_predicates"]
        )
        row_predicates = self._panel_row([top_predicates_chart])

        top_classes_chart = self._bar_chart_svg(
            "Top Classes", "Most common instanceOf objects.", stats["top_classes"]
        )
        depth_chart = self._bar_chart_svg(
            "BFS Depth Reached",
            "How many entities were discovered at each hop from the seed.",
            [(f"Depth {depth}", count) for depth, count in stats["depth_counts"]],
        )
        row_classes = self._panel_row([top_classes_chart])
        row_depth = self._panel_row([depth_chart])

        node_mix_donut = self._donut_svg(
            "Entities vs Literals",
            [
                ("Entities", stats["total_entities"], "#2563eb"),
                ("Literals", stats["total_literals"], "#0f766e"),
            ],
            "nodes",
        )
        row_donuts = self._panel_row([node_mix_donut])

        failure_stage_labels = {
            "llm_request": "LLM request",
            "json_parse": "Response parsing",
            "processing": "Processing",
        }
        failures_rows = "\n".join(
            "<tr>"
            f"<td>{escape(str(record.get('subject', '')))}</td>"
            f"<td>{escape(failure_stage_labels.get(record.get('stage'), str(record.get('stage', ''))))}</td>"
            f"<td>{escape(str(record.get('error', '')))}</td>"
            f"<td>{escape(str(record.get('time', '')))}</td>"
            "</tr>"
            for record in reversed(stats["failed_subjects"])
        )
        failures_section = ""
        if failures_rows:
            failures_section = f"""<section class="band">
<details class="reveal">
<summary>Failures <span class="count-badge">{stats['total_failed_subjects']}</span></summary>
<div class="reveal-body">
<p>Subjects skipped after repeated failures, or where something else went wrong while processing them -- most recent first.</p>
<div class="table-scroll">
<table><thead><tr><th>Subject</th><th>Stage</th><th>Error</th><th>Time</th></tr></thead>
<tbody>
{failures_rows}
</tbody></table>
</div>
</div>
</details>
</section>"""

        central_entities_rows = "\n".join(
            "<tr>"
            f'<td><a href="{escape(quote(self._entity_filename(name)))}">{escape(name)}</a></td>'
            f"<td>{out_degree:,}</td><td>{in_degree:,}</td><td>{total_degree:,}</td>"
            "</tr>"
            for name, out_degree, in_degree, total_degree in stats["top_entities"]
        )
        central_entities_section = ""
        if central_entities_rows:
            central_entities_section = f"""<section class="band">
<h2>Central Entities</h2>
<p>Entities ranked by total degree (outgoing facts plus incoming references).</p>
<table><thead><tr><th>Entity</th><th>Outgoing</th><th>Incoming</th><th>Total degree</th></tr></thead>
<tbody>
{central_entities_rows}
</tbody></table>
</section>"""

        # The Entity Index and triples table only ever show the first
        # _MAX_DISPLAYED_* (by discovery order) -- once that many exist, the
        # rendered HTML/JSON for them is frozen and reused on every later
        # regeneration instead of being rebuilt from an ever-growing list, since
        # a stable prefix of an append-only list never changes once past the cap.
        if self._frozen_entity_links_html is not None:
            entity_links = self._frozen_entity_links_html
        else:
            entity_links = "\n".join(
                f'<li><a href="{escape(quote(self._entity_filename(subject)))}">{escape(subject)}</a></li>'
                for subject in entity_order[:_MAX_DISPLAYED_ENTITIES]
            )
            if total_entity_count >= _MAX_DISPLAYED_ENTITIES:
                self._frozen_entity_links_html = entity_links
        entity_index_note = (
            f'<p class="graph-note">Showing the first {min(total_entity_count, _MAX_DISPLAYED_ENTITIES):,} '
            f"of {total_entity_count:,} entities discovered, in discovery order. "
            'The complete list is in the downloadable database.</p>'
            if total_entity_count > _MAX_DISPLAYED_ENTITIES
            else ""
        )

        total_triple_count = len(self.triples_in_order)
        if self._frozen_triples_json is not None:
            triples_json = self._frozen_triples_json
        else:
            # "</" is escaped so a triple value containing the literal text
            # "</script>" can't prematurely close this data island when the
            # browser parses the HTML.
            triples_json = json.dumps(
                [list(triple) for triple in self.triples_in_order[:_MAX_DISPLAYED_TRIPLES]], ensure_ascii=False
            ).replace("</", "<\\/")
            if total_triple_count >= _MAX_DISPLAYED_TRIPLES:
                self._frozen_triples_json = triples_json
        triples_note = (
            f'<p class="graph-note">Showing the first {min(total_triple_count, _MAX_DISPLAYED_TRIPLES):,} '
            f"of {total_triple_count:,} triples collected, in collection order. "
            'The complete set is in the downloadable database.</p>'
            if total_triple_count > _MAX_DISPLAYED_TRIPLES
            else ""
        )

        index = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{seed} &mdash; miniGPTKB</title>
<style>
    :root {{
        --ink:#1f2937; --muted:#5b6472; --line:#d8dee7; --paper:#f8f9fa; --panel:#ffffff; --panel-2:#f3f6f8;
        --blue:#0077cc; --heading:#0d5f99; --chart-text:#374151;
        --note-bg:#eef7fd; --note-ink:#0b3550;
        --shadow:0 1px 2px rgba(15,23,42,.05);
    }}
    @media (prefers-color-scheme: dark) {{
        :root:not([data-theme="light"]) {{
            --ink:#e5eaef; --muted:#94a3b0; --line:#2a3744; --paper:#0d1620; --panel:#141f2b; --panel-2:#192533;
            --blue:#4db8f0; --heading:#63b8ee; --chart-text:#c7d2db;
            --note-bg:#132a3a; --note-ink:#bfe3fb;
            --shadow:0 1px 2px rgba(0,0,0,.35);
        }}
    }}
    :root[data-theme="dark"] {{
        --ink:#e5eaef; --muted:#94a3b0; --line:#2a3744; --paper:#0d1620; --panel:#141f2b; --panel-2:#192533;
        --blue:#4db8f0; --heading:#63b8ee; --chart-text:#c7d2db;
        --note-bg:#132a3a; --note-ink:#bfe3fb;
        --shadow:0 1px 2px rgba(0,0,0,.35);
    }}
    html {{ color-scheme: light dark; }}
    :root[data-theme="light"] {{ color-scheme: light; }}
    :root[data-theme="dark"] {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Arial, sans-serif; line-height:1.6; }}
    header {{ color:white; background:linear-gradient(120deg,#0b3d63 0%,#0077cc 60%,#075985 100%); padding:44px 28px 30px; }}
    .header-inner, main {{ max-width:1100px; margin:0 auto; }}
    .header-top {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }}
    .header-actions {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
    .download-button {{ background:#fff; color:#0b3d63; font-weight:800; font-size:12.5px; padding:8px 16px; border-radius:999px; text-decoration:none; white-space:nowrap; font-family:inherit; }}
    .download-button:hover {{ background:#dbeeff; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.08em; font-size:12px; opacity:.85; font-weight:700; }}
    h1.hero-title {{ margin:8px 0 10px; font-size:clamp(28px,4.5vw,46px); line-height:1.08; }}
    .lede {{ max-width:820px; font-size:16px; color:#e7f1fb; margin:0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-top:22px; }}
    .metric {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.26); border-radius:8px; padding:12px 12px 11px; min-height:78px; }}
    .metric span {{ display:block; color:#dbeeff; font-size:11px; text-transform:uppercase; letter-spacing:.07em; font-weight:700; }}
    .metric strong {{ display:block; font-size:23px; margin-top:6px; }}
    .theme-toggle {{ display:flex; gap:4px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.26); border-radius:999px; padding:3px; flex:none; }}
    .theme-toggle button {{ background:transparent; border:none; color:#dbeeff; font-size:12px; font-weight:700; padding:6px 12px; border-radius:999px; cursor:pointer; font-family:inherit; }}
    .theme-toggle button.active {{ background:rgba(255,255,255,.24); color:white; }}
    main {{ padding:22px 28px 50px; }}
    .band {{ margin:22px 0; padding:22px 0; border-top:1px solid var(--line); }}
    .band:first-child {{ border-top:none; }}
    h2 {{ font-size:21px; margin:0 0 10px; color:var(--heading); }}
    p {{ max-width:900px; color:var(--muted); }}
    .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:stretch; }}
    .grid-2 > * {{ min-width:0; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:var(--shadow); overflow:hidden; }}
    .chart {{ width:100%; height:auto; display:block; }}
    .chart-title {{ font-weight:800; fill:var(--ink); font-size:16px; }}
    .chart-subtitle {{ fill:var(--muted); font-size:12px; }}
    .axis-label {{ fill:var(--chart-text); font-size:12px; dominant-baseline:middle; }}
    .value-label {{ fill:var(--ink); font-size:12px; font-weight:800; dominant-baseline:middle; }}
    .legend-label {{ fill:var(--chart-text); font-size:12.5px; }}
    .donut-big {{ font-size:21px; font-weight:850; fill:var(--ink); }}
    .donut-small {{ font-size:11px; fill:var(--muted); text-transform:uppercase; }}
    .note {{ border-left:4px solid var(--blue); padding:12px 16px; background:var(--note-bg); color:var(--note-ink); border-radius:0 8px 8px 0; margin:14px 0; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th, td {{ text-align:left; border-bottom:1px solid var(--line); padding:9px 10px; vertical-align:top; }}
    th {{ background:var(--panel-2); font-size:11.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
    tr:last-child td {{ border-bottom:none; }}
    a {{ color:var(--blue); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .search-box {{ width:100%; max-width:420px; padding:9px 12px; border:1px solid var(--line); border-radius:6px; font-size:14px; margin-bottom:14px; background:var(--panel); color:var(--ink); }}
    .entity-grid {{ list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(3,1fr); gap:6px 16px; }}
    .entity-grid li {{ border-top:1px solid var(--line); padding-top:6px; }}
    .entity-grid a {{ display:block; padding:2px 0; }}
    .config-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px,1fr)); gap:12px; margin-bottom:14px; }}
    .config-item {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:11px 13px; }}
    .config-item span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }}
    .config-item strong {{ display:block; margin-top:5px; font-size:14px; font-weight:700; color:var(--ink); overflow-wrap:anywhere; }}
    .prompt-block {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace; font-size:12.5px; color:var(--ink); max-height:340px; overflow-y:auto; }}
    .hidden {{ display:none !important; }}
    .count-badge {{ color:var(--muted); font-weight:600; font-size:15px; }}
    details.reveal {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    details.reveal summary {{ cursor:pointer; padding:14px 16px; font-size:21px; font-weight:800; color:var(--heading); }}
    details.reveal[open] summary {{ border-bottom:1px solid var(--line); }}
    details.reveal .reveal-body {{ padding:16px; }}
    .table-scroll {{ overflow-x:auto; }}
    .triple-toolbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-bottom:12px; }}
    .triple-toolbar p {{ margin:0; }}
    .sort-toggle {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px 14px; font-size:13px; font-weight:700; color:var(--heading); cursor:pointer; white-space:nowrap; }}
    .sort-toggle:hover {{ background:var(--panel-2); }}
    .graph-canvas {{ width:100%; height:560px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    .graph-note {{ font-size:13px; color:var(--muted); margin:4px 0 12px; }}
    @media (max-width: 920px) {{ .metrics {{ grid-template-columns:repeat(3,1fr); }} .grid-2 {{ grid-template-columns:1fr; }} .entity-grid {{ grid-template-columns:repeat(2,1fr); }} }}
    @media (max-width: 640px) {{ header {{ padding:32px 18px 22px; }} main {{ padding:16px; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} .entity-grid {{ grid-template-columns:1fr; }} .graph-canvas {{ height:400px; }} }}
</style>
</head>
<body>
<header>
<div class="header-inner">
<div class="header-top">
<div class="eyebrow">miniGPTKB HTML Collection</div>
<div class="header-actions">
<a class="download-button" href="/api/runs/{run_id}/download" download>&#8681; Download results</a>
<div class="theme-toggle" id="themeToggle">
<button type="button" data-theme-choice="light">Light</button>
<button type="button" data-theme-choice="dark">Dark</button>
</div>
</div>
</div>
<h1 class="hero-title">{topic}</h1>
<p class="lede">A knowledge graph seeded from {seed}, elicited from {model}.</p>
<div class="metrics">
{metrics_html}
</div>
</div>
</header>
<main>
{run_configuration_section}
<section class="band">
<h2>Knowledge Graph</h2>
{graph_note}
<div id="graphCanvas" class="graph-canvas"></div>
</section>
<section class="band">
<h2>Run Status</h2>
<p class="note">{remaining_note}{quality_note}</p>
</section>
{failures_section}
{row_donuts}
{row_predicates}
{row_classes}
{row_depth}
{central_entities_section}
<section class="band">
<details class="reveal">
<summary>Entity Index <span class="count-badge" id="entityCount">{total_entity_count}</span></summary>
<div class="reveal-body">
{entity_index_note}
<input type="text" id="entitySearch" class="search-box" placeholder="Filter entities by name&hellip;" autocomplete="off">
<ul class="entity-grid" id="entityList">
{entity_links}
</ul>
</div>
</details>
</section>
<section class="band">
<details class="reveal" id="triplesDetails">
<summary>All Elicited Triples <span class="count-badge" id="tripleCount">{total_triple_count}</span></summary>
<div class="reveal-body">
{triples_note}
<div class="triple-toolbar">
<button type="button" id="tripleSortToggle" class="sort-toggle">Sort: Creation order</button>
</div>
<div class="table-scroll">
<table id="tripleTable">
<thead><tr><th>Subject</th><th>Predicate</th><th>Object</th></tr></thead>
<tbody id="tripleTableBody"></tbody>
</table>
</div>
</div>
</details>
</section>
</main>
<script src="vis-network.min.js"></script>
<script type="application/json" id="graph-nodes-data">{graph_nodes_json}</script>
<script type="application/json" id="graph-edges-data">{graph_edges_json}</script>
<script>
(function () {{
    var container = document.getElementById('graphCanvas');
    var nodesEl = document.getElementById('graph-nodes-data');
    var edgesEl = document.getElementById('graph-edges-data');
    if (!container || !nodesEl || !edgesEl || typeof vis === 'undefined') return;

    var nodes = new vis.DataSet(JSON.parse(nodesEl.textContent));
    var edges = new vis.DataSet(JSON.parse(edgesEl.textContent));

    // vis-network draws on <canvas>, so it can't pick up CSS variables --
    // these colors are recomputed and pushed in on every theme change below.
    function graphColors() {{
        var explicit = document.documentElement.getAttribute('data-theme');
        var dark = explicit === 'dark' || (!explicit && window.matchMedia('(prefers-color-scheme: dark)').matches);
        return dark
            ? {{ font: '#e5eaef', edge: '#3a4a58', accent: '#4db8f0' }}
            : {{ font: '#1f2937', edge: '#c7cdd6', accent: '#0077cc' }};
    }}

    var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, {{
        nodes: {{ font: {{ color: graphColors().font, size: 13 }}, borderColor: graphColors().font }},
        edges: {{
            color: {{ color: graphColors().edge, highlight: graphColors().accent, hover: graphColors().accent }},
            smooth: {{ type: 'dynamic' }},
            arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }}
        }},
        physics: {{ solver: 'barnesHut', stabilization: {{ iterations: 150 }} }},
        interaction: {{ hover: true, tooltipDelay: 100, dragView: true, zoomView: true }}
    }});
    network.once('stabilizationIterationsDone', function () {{
        network.setOptions({{ physics: false }});
    }});

    window.__miniGPTKBUpdateGraphTheme = function () {{
        var c = graphColors();
        network.setOptions({{
            nodes: {{ font: {{ color: c.font }}, borderColor: c.font }},
            edges: {{ color: {{ color: c.edge, highlight: c.accent, hover: c.accent }} }}
        }});
    }};
}})();
</script>
<script type="application/json" id="triple-data">{triples_json}</script>
<script>
(function () {{
    var input = document.getElementById('entitySearch');
    var items = Array.prototype.slice.call(document.querySelectorAll('#entityList li'));
    var countEl = document.getElementById('entityCount');
    // The badge starts showing the true total (which may be larger than
    // items.length when the list below is capped) -- searching only
    // searches what's actually rendered, so the match ratio uses that
    // smaller count, but clearing the search restores the true total
    // rather than silently shrinking it to "how many are shown".
    var trueTotal = countEl ? countEl.textContent : String(items.length);
    var shownTotal = items.length;
    if (!input) return;
    input.addEventListener('input', function () {{
        var query = input.value.trim().toLowerCase();
        var shown = 0;
        items.forEach(function (item) {{
            var match = item.textContent.toLowerCase().indexOf(query) !== -1;
            item.classList.toggle('hidden', !match);
            if (match) shown++;
        }});
        countEl.textContent = query ? (shown + ' / ' + shownTotal) : trueTotal;
    }});
}})();
</script>
<script>
(function () {{
    var details = document.getElementById('triplesDetails');
    var dataEl = document.getElementById('triple-data');
    var tbody = document.getElementById('tripleTableBody');
    var sortButton = document.getElementById('tripleSortToggle');
    if (!details || !dataEl || !tbody || !sortButton) return;

    var triples = JSON.parse(dataEl.textContent);
    var sortMode = 'creation';
    var rendered = false;

    function escapeHtml(value) {{
        return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }}

    function renderTriples() {{
        var rows = triples;
        if (sortMode === 'subject') {{
            rows = triples.slice().sort(function (a, b) {{
                return a[0].localeCompare(b[0]) || a[1].localeCompare(b[1]) || a[2].localeCompare(b[2]);
            }});
        }}
        tbody.innerHTML = rows.map(function (triple) {{
            return '<tr><td>' + escapeHtml(triple[0]) + '</td><td>' + escapeHtml(triple[1]) + '</td><td>' + escapeHtml(triple[2]) + '</td></tr>';
        }}).join('');
        rendered = true;
    }}

    // Deferred until first expanded, so a large triple list costs nothing while collapsed.
    details.addEventListener('toggle', function () {{
        if (details.open && !rendered) renderTriples();
    }});

    sortButton.addEventListener('click', function () {{
        sortMode = sortMode === 'creation' ? 'subject' : 'creation';
        sortButton.textContent = sortMode === 'creation' ? 'Sort: Creation order' : 'Sort: Subject (A-Z)';
        renderTriples();
    }});
}})();
</script>
<script>
(function () {{
    // Same storage key as the miniGPTKB control panel (same origin), so a
    // theme choice made in one place is remembered in the other too.
    var root = document.documentElement;
    var opts = document.querySelectorAll('#themeToggle button');
    if (!opts.length) return;
    var STORAGE_KEY = 'miniGPTKB.theme';

    var stored = null;
    try {{ stored = localStorage.getItem(STORAGE_KEY); }} catch (error) {{}}
    if (stored === 'light' || stored === 'dark') root.setAttribute('data-theme', stored);

    function currentChoice() {{
        var explicit = root.getAttribute('data-theme');
        if (explicit) return explicit;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }}
    function paint() {{
        var choice = currentChoice();
        opts.forEach(function (opt) {{
            opt.classList.toggle('active', opt.dataset.themeChoice === choice);
        }});
    }}
    opts.forEach(function (opt) {{
        opt.addEventListener('click', function () {{
            var choice = opt.dataset.themeChoice;
            root.setAttribute('data-theme', choice);
            try {{ localStorage.setItem(STORAGE_KEY, choice); }} catch (error) {{}}
            paint();
            if (window.__miniGPTKBUpdateGraphTheme) window.__miniGPTKBUpdateGraphTheme();
        }});
    }});
    paint();
}})();
</script>
</body>
</html>
"""
        self.html_dir.mkdir(parents=True, exist_ok=True)
        (self.html_dir / "index.html").write_text(index, encoding="utf-8")

    def _open_sqlite_export(self) -> sqlite3.Connection:
        """Creates a fresh knowledgeBase.db and schema, called once (lazily,
        from _write_sqlite_export) the first time a run writes to it -- and
        again after a resume, since resuming starts the export over rather
        than reconciling with whatever an earlier session already wrote (all
        the "scanned upto" pointers reset to 0, so the very next write below
        naturally treats all current state as new: a one-time full rebuild,
        the same cost class as the other one-time rebuilds already done at
        resume, e.g. _load_triples_index).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.run_dir / "knowledgeBase.db"
        for path in (db_path, self.run_dir / "knowledgeBase.db-wal", self.run_dir / "knowledgeBase.db-shm"):
            path.unlink(missing_ok=True)

        # check_same_thread=False: this connection is created once (usually
        # from the main thread, during startup) and then reused for the
        # life of the run by whichever worker thread next finishes an
        # entity. That's safe here specifically because every single call
        # site (all three) already runs under self.state_lock, so access is
        # already fully serialized -- sqlite3's own default thread-affinity
        # check would otherwise reject any use from a thread other than the
        # one that created the connection, which is not what this needs.
        connection = sqlite3.connect(db_path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE run (
                run_id TEXT, topic TEXT, seed_entity TEXT, model TEXT, llm_url TEXT, language TEXT,
                schema_mode TEXT, schema_text TEXT, prompt_used TEXT, facts_multiplier REAL,
                nthreads INTEGER, max_requests_per_minute INTEGER,
                max_entities INTEGER, max_bfs_depth INTEGER, max_runtime_seconds INTEGER, max_cost REAL,
                started_at TEXT, ended_at TEXT, status TEXT, cost_usd REAL,
                reasoning_effort TEXT, reasoning_enabled INTEGER
            );
            CREATE TABLE triples (
                id INTEGER PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL
            );
            CREATE INDEX idx_triples_subject ON triples(subject);
            CREATE INDEX idx_triples_object ON triples(object);
            CREATE TABLE entities (
                name TEXT PRIMARY KEY, depth INTEGER, status TEXT
            );
            CREATE TABLE literals (text TEXT);
            CREATE TABLE off_domain_entities (text TEXT);
            CREATE TABLE failures (
                id INTEGER PRIMARY KEY, subject TEXT, stage TEXT, error TEXT, detail TEXT, time TEXT
            );
            """
        )
        # Only the "how much has been written" pointers reset to 0 here --
        # not self._db_pending_entity_updates. That's a queue of status
        # changes not yet applied (not a position marker), and since this
        # method runs lazily on first use, entities may already have been
        # marked processed -- and queued for this update -- before the
        # schema above ever existed; clearing it here would silently drop
        # those updates instead of just deferring them to the write that
        # immediately follows this call.
        self._db_triples_scanned_upto = 0
        self._db_entities_scanned_upto = 0
        self._db_literals_scanned_upto = 0
        self._db_off_domain_scanned_upto = 0
        self._db_failures_scanned_upto = 0
        return connection

    def _write_sqlite_export(self, final: bool = False) -> None:
        """Keeps knowledgeBase.db continuously up to date.

        Unlike the HTML report (rebuilt wholesale each time, but cheap
        because what it displays is capped), the database holds the full,
        uncapped dataset -- so a full rebuild on every entity costs O(total
        data collected so far) per call, measured at ~1.9s/call at 100k
        entities (~54 hours cumulative over a run). Instead, one connection
        is kept open for the run's lifetime, in WAL mode -- so a concurrent
        download of the file is never corrupted by an in-progress write, it
        just sees the state as of the last checkpoint below, an ordinary
        SQLite guarantee, not a torn/invalid file -- and only rows that are
        actually new (or, for entities transitioning queued -> processed,
        whose status just changed) are written on each call.

        final=True is passed once, right after the run loop ends: it records
        the true finished/stopped status and does a full WAL checkpoint so
        the file left on disk doesn't depend on a companion -wal file to be
        complete.
        """
        if self._db_connection is None:
            self._db_connection = self._open_sqlite_export()
        connection = self._db_connection

        try:
            run_config = json.loads((self.run_dir / "runConfig.json").read_text(encoding="utf-8"))
        except Exception:
            run_config = {}

        language_name = PROMPT_LANGUAGES.get(self.config.language, {}).get("name", self.config.language)
        prompt_used = self.config.prompt_template.strip() or build_elicitation_prompt_body(
            self.config.language,
            self.config.topic,
            self.config.facts_multiplier,
            self.config.schema_mode,
            self.config.schema_text,
        )
        if final:
            status = "stopped" if self.stop_event.is_set() else "finished"
        else:
            # Best-effort: this file is written by the runner subprocess
            # itself, so mid-run it can't know its own final wall-clock end
            # time -- ended_at comes from the same runConfig.json field the
            # History table already relies on, and simply isn't set yet.
            status = "stopped" if self.stop_event.is_set() else "running"
        cost_usd = (
            round(self.total_cost_usd, 4)
            if self._is_openai_endpoint() or self._is_openrouter_endpoint()
            else None
        )

        connection.execute("DELETE FROM run")
        connection.execute(
            """
            INSERT INTO run (
                run_id, topic, seed_entity, model, llm_url, language,
                schema_mode, schema_text, prompt_used, facts_multiplier,
                nthreads, max_requests_per_minute,
                max_entities, max_bfs_depth, max_runtime_seconds, max_cost,
                started_at, ended_at, status, cost_usd,
                reasoning_effort, reasoning_enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_dir.name,
                self.config.topic,
                self.config.seed_entity,
                self.config.model,
                self.config.llm_url,
                language_name,
                self.config.schema_mode,
                self.config.schema_text,
                prompt_used,
                self.config.facts_multiplier,
                self.config.nthreads,
                self.config.max_requests_per_minute,
                self.config.max_entities,
                self.config.max_bfs_depth,
                self.config.max_runtime_seconds,
                self.config.max_cost,
                run_config.get("started_at"),
                run_config.get("ended_at"),
                status,
                cost_usd,
                self.resolved_reasoning_effort,
                self.resolved_reasoning_enabled,
            ),
        )

        new_triples = self.triples_in_order[self._db_triples_scanned_upto :]
        if new_triples:
            connection.executemany(
                "INSERT INTO triples (subject, predicate, object) VALUES (?, ?, ?)", new_triples
            )
            self._db_triples_scanned_upto = len(self.triples_in_order)

        entity_order = list(self.depths.keys())
        new_entities = entity_order[self._db_entities_scanned_upto :]
        if new_entities:
            connection.executemany(
                "INSERT INTO entities (name, depth, status) VALUES (?, ?, 'queued')",
                [(name, self.depths[name]) for name in new_entities],
            )
            self._db_entities_scanned_upto = len(entity_order)
        if self._db_pending_entity_updates:
            connection.executemany(
                "UPDATE entities SET status = 'processed' WHERE name = ?",
                [(name,) for name in self._db_pending_entity_updates],
            )
            self._db_pending_entity_updates = []

        new_literals = self.literal_texts[self._db_literals_scanned_upto :]
        if new_literals:
            connection.executemany("INSERT INTO literals (text) VALUES (?)", [(text,) for text in new_literals])
            self._db_literals_scanned_upto = len(self.literal_texts)

        new_off_domain = self.off_domain_texts[self._db_off_domain_scanned_upto :]
        if new_off_domain:
            connection.executemany(
                "INSERT INTO off_domain_entities (text) VALUES (?)", [(text,) for text in new_off_domain]
            )
            self._db_off_domain_scanned_upto = len(self.off_domain_texts)

        new_failures = self.failure_records[self._db_failures_scanned_upto :]
        if new_failures:
            connection.executemany(
                "INSERT INTO failures (subject, stage, error, detail, time) VALUES (?, ?, ?, ?, ?)",
                [
                    (record.get("subject"), record.get("stage"), record.get("error"), record.get("detail"), record.get("time"))
                    for record in new_failures
                ],
            )
            self._db_failures_scanned_upto = len(self.failure_records)

        connection.commit()
        # A passive checkpoint merges as much of the WAL back into the main
        # file as it can without blocking -- keeps the plain .db file (what
        # actually gets downloaded or zipped) reasonably current even
        # mid-run, without forcing a full flush on every single call.
        connection.execute("PRAGMA wal_checkpoint(PASSIVE)")

        if final:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
            self._db_connection = None

    def _entity_filename(self, name: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(".")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            cleaned = "unnamed"
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        }
        if cleaned.upper() in reserved:
            cleaned = f"_{cleaned}"
        if len(cleaned) > 140:
            digest = sha1(name.encode("utf-8")).hexdigest()[:10]
            cleaned = f"{cleaned[:120].rstrip()}_{digest}"
        return f"{cleaned}.html"

    def _should_stop(self) -> bool:
        if self.stop_event.is_set():
            return True
        if self.config.max_runtime_seconds and time.monotonic() - self.started_at >= self.config.max_runtime_seconds:
            self._log("Runtime limit reached.")
            return True
        if self.config.max_entities and self._get_total_object_count() >= self.config.max_entities:
            self._log("Object limit reached.")
            return True
        if self.config.max_cost is not None:
            if self._is_openai_endpoint():
                model_has_pricing = self.config.model.strip().lower() in OPENAI_TEXT_PRICING
            elif self._is_openrouter_endpoint():
                model_has_pricing = self.config.model.strip().lower() in self.openrouter_pricing
            else:
                model_has_pricing = None  # endpoint unsupported entirely

            if model_has_pricing is None:
                self._log(
                    "Cost termination is only available for the OpenAI and OpenRouter API endpoints; ignoring cost limit.",
                    level="warning",
                )
                self.config.max_cost = None
            elif not model_has_pricing:
                self._log(
                    f"No pricing data for model '{self.config.model}'; ignoring cost limit.",
                    level="warning",
                )
                self.config.max_cost = None
            else:
                with self.state_lock:
                    current_cost = self.total_cost_usd
                if current_cost >= self.config.max_cost:
                    self._log(f"Cost limit reached (${current_cost:.4f} spent, limit ${self.config.max_cost:.2f}).")
                    return True
        return False

    def _wait_if_paused(self) -> None:
        while self.pause_event.is_set() and not self.stop_event.is_set():
            self._emit_status("paused")
            time.sleep(0.5)

    def _emit_status(self, status: str) -> None:
        with self.state_lock:
            processed = len(self.processed_subjects)
            queued = len(self.subject_queue)
            active = self.active_workers
            cost_usd = (
                round(self.total_cost_usd, 4)
                if self._is_openai_endpoint() or self._is_openrouter_endpoint()
                else None
            )
        self.emit(
            {
                "type": "status",
                "status": status,
                "processed": processed,
                "queued": queued,
                "active": active,
                "triples": self._count_lines(self.triples_output_path),
                "literals": self._count_lines(self.literals_path),
                "off_domain_entities": self._count_lines(self.off_domain_entities_path),
                "total_objects": self._get_total_object_count(),
                "failed_subjects": self._count_lines(self.failed_subjects_path),
                "cost_usd": cost_usd,
            }
        )

    def _log(self, message: str, level: str = "info") -> None:
        event = {
            "type": "log",
            "level": level,
            "message": message,
            "time": time.strftime("%H:%M:%S"),
        }
        self.emit(event)

    def _get_total_object_count(self) -> int:
        # known_objects already contains every subject and object ever seen:
        # _collect_new_subjects adds each triple's object the moment that
        # triple's batch is classified (regardless of the outcome), and every
        # subject that's ever processed necessarily passed through there
        # first (as the seed, or as a previously-seen object) -- so nothing
        # in triples.jsonl can exist without already being in this set,
        # making a full re-read of that file on every call unnecessary.
        with self.state_lock:
            return len(self.known_objects)

    def _load_run_state(self) -> None:
        with self.state_lock:
            queue_order, processed_list, depths, used_event_log = load_subject_state(self.run_dir)

            if not queue_order and not processed_list:
                # Brand new run (or an event log that somehow ended up empty).
                queue_order = [self.config.seed_entity]
                depths.setdefault(self.config.seed_entity, 0)

            if not used_event_log:
                # A fresh run, or resuming one started before the event log
                # existed -- backfill it once so every mutation from here on
                # is an O(1) append instead of a full-snapshot rewrite.
                for subject in processed_list:
                    self._log_subject_event("enqueue", subject, depths.get(subject, 0))
                    self._log_subject_event("dequeue", subject)
                    self._log_subject_event("finish", subject)
                for subject in queue_order:
                    self._log_subject_event("enqueue", subject, depths.get(subject, 0))

            self.subject_queue = deque(queue_order)
            self.queued_subjects = set(queue_order)
            self.processed_subjects = set(processed_list)
            self.depths = depths
            self.known_objects = set(queue_order) | set(processed_list) | set(depths)
            self.depth_counts = Counter(depths.values())
            self._load_triples_index()
            # One-time O(n) rebuilds at startup/resume -- everything after
            # this point maintains these incrementally instead of re-deriving
            # them from disk on every homepage/database regeneration.
            self.literal_texts = [
                str(row["text"]) for row in self._read_jsonl(self.literals_path) if row.get("text") is not None
            ]
            self.off_domain_texts = [
                str(row["text"])
                for row in self._read_jsonl(self.off_domain_entities_path)
                if row.get("text") is not None
            ]
            self.literals_count = len(self.literal_texts)
            self.off_domain_count = len(self.off_domain_texts)
            self.failure_records = self._read_jsonl(self.failed_subjects_path)
            self.failed_subjects_count = len(self.failure_records)

    def _load_triples_index(self) -> None:
        """Rebuild the outgoing/incoming triple indices (and their derived
        predicate/class/degree counters) from disk on resume.

        Without this, resuming a stopped run would only see incoming/outgoing
        links discovered after the resume, silently dropping everything an
        earlier session already found when an existing page gets refreshed.
        """
        self.triples_by_subject = {}
        self.triples_by_object = {}
        self.triples_in_order = []
        self.predicate_counts = Counter()
        self.class_counts = Counter()
        self.out_degree = Counter()
        self.in_degree = Counter()
        self.degree_counts = Counter()
        if not self.triples_output_path.exists():
            return
        with self.triples_output_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    triple = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(triple, dict):
                    continue
                s = str(triple.get("s", "")).strip()
                p = str(triple.get("p", "")).strip()
                o = str(triple.get("o", "")).strip()
                if not s or not p or not o:
                    continue
                self.triples_by_subject.setdefault(s, []).append((p, o))
                self.triples_by_object.setdefault(o, []).append((s, p))
                self.triples_in_order.append((s, p, o))
                self.predicate_counts[p] += 1
                if p.lower() == "instanceof":
                    self.class_counts[o] += 1
                self.out_degree[s] += 1
                self.in_degree[o] += 1
                self.degree_counts[s] += 1
                self.degree_counts[o] += 1

    def _append_jsonl(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _log_subject_event(self, op: str, subject: str, depth: int | None = None) -> None:
        """O(1) append to subjectEvents.jsonl -- see replay_subject_events for
        how these are turned back into queue/processed/depth state. depth is
        only meaningful (and only needs to be included) on the first
        "enqueue" for a subject; omitting it on a later re-enqueue (e.g. a
        clean-stop requeue) leaves the earlier value alone.
        """
        event: dict[str, Any] = {"op": op, "subject": subject}
        if depth is not None:
            event["depth"] = depth
        self._append_jsonl(self.subject_events_path, event)

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as file:
            return sum(1 for _ in file)

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def remove_json_delimiters(input_string: str) -> str:
    value = input_string.strip()
    if value.startswith("```json"):
        value = value[len("```json") :].strip()
    if value.startswith("```"):
        value = value[len("```") :].strip()
    if value.endswith("```"):
        value = value[: -len("```")].strip()
    return value


def deduplicate_triples(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique_triples = []
    for triple in triples:
        try:
            key = (
                triple.get("s", "").strip(),
                triple.get("p", "").strip(),
                triple.get("o", "").strip(),
            )
        except AttributeError:
            continue
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)
    return unique_triples


def chunked(items: list[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def replay_subject_events(events_path: Path) -> tuple[list[str], list[str], dict[str, int]]:
    """Reconstructs (queue_order, processed_list, depths) from an append-only
    subjectEvents.jsonl log -- the durable, O(1)-per-write replacement for the
    old subjectQueue.json/processedSubjects.json/subjectDepths.json snapshots,
    which had to be rewritten in full on every single subject dequeue/finish.

    Shared between the runner (which writes the log while a run is active)
    and the server (which replays it for the History table, resume, and
    seed/topic inference), so both agree on exactly how state is derived from
    it -- in particular, that a subject "dequeue"d but never "finish"ed (the
    process died mid-subject, rather than a clean stop) is still owed a
    retry, not silently dropped. Ordered dicts (not lists) track queued/
    in-flight subjects so both membership checks and removals are O(1),
    keeping the whole replay O(n) in the number of events rather than O(n^2).
    """
    queue_order: dict[str, None] = {}
    in_flight: dict[str, None] = {}
    processed: dict[str, None] = {}
    depths: dict[str, int] = {}

    if not events_path.exists():
        return [], [], {}

    with events_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = event.get("op")
            subject = event.get("subject")
            if not subject:
                continue
            if op == "enqueue":
                if "depth" in event:
                    try:
                        depths[subject] = int(event["depth"])
                    except (TypeError, ValueError):
                        pass
                if subject not in processed and subject not in in_flight:
                    queue_order[subject] = None
            elif op == "dequeue":
                queue_order.pop(subject, None)
                in_flight[subject] = None
            elif op == "finish":
                in_flight.pop(subject, None)
                queue_order.pop(subject, None)
                processed[subject] = None

    final_queue = list(queue_order.keys()) + list(in_flight.keys())
    return final_queue, list(processed.keys()), depths


def load_subject_state(run_dir: Path) -> tuple[list[str], list[str], dict[str, int], bool]:
    """queue_order, processed_list, depths, used_event_log -- from
    subjectEvents.jsonl if a run has one, else the legacy snapshot files
    (subjectQueue.json/processedSubjects.json/subjectDepths.json) for a run
    started before the event log existed. The boolean tells a caller whether
    it's looking at a run that still needs its event log backfilled.
    """
    events_path = run_dir / "subjectEvents.jsonl"
    if events_path.exists():
        queue_order, processed_list, depths = replay_subject_events(events_path)
        return queue_order, processed_list, depths, True

    def read_json_list(path: Path) -> list[str]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [str(item) for item in data] if isinstance(data, list) else []

    def read_json_dict(path: Path) -> dict[str, int]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        result = {}
        for key, value in data.items():
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return result

    queue_order = read_json_list(run_dir / "subjectQueue.json")
    processed_list = read_json_list(run_dir / "processedSubjects.json")
    depths = read_json_dict(run_dir / "subjectDepths.json")
    return queue_order, processed_list, depths, False
