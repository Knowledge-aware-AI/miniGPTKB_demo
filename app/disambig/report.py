"""HTML report of a disambiguated miniGPTKB run: an index page with run
statistics (including how many surface labels were consolidated into how
many entities, and which labels denote several distinct entities) plus one
page per entity, named by its id (E12.html) since labels are not unique."""

from __future__ import annotations

import json
import shutil
import time
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.runner import PROMPT_LANGUAGES, _format_duration

if TYPE_CHECKING:
    from app.disambig.runner import DisambiguatedKBCRunner

_ASSETS = Path(__file__).resolve().parent / "assets"
_VIS = Path(__file__).resolve().parent.parent / "vendor" / "vis-network.min.js"
_PALETTE = ["#2563eb", "#0f766e", "#b45309", "#be123c", "#7c3aed", "#15803d", "#ca8a04", "#0369a1", "#c2410c", "#4d7c0f"]
# Knowledge-graph nodes: soft pastel fills (one per BFS depth) with a slightly
# deeper border of the same hue, and dark labels, which stay readable on every
# fill in both the light and the dark theme.
_NODE_PALETTE = [
    ("#bfe3d0", "#7cbf9c"),  # mint
    ("#c7d7f5", "#8aa6dc"),  # periwinkle
    ("#f8dcc0", "#e0aa79"),  # peach
    ("#e3d3f1", "#b39bd2"),  # lavender
    ("#f6efb9", "#d6c76f"),  # butter
    ("#c2e6ea", "#80c3cb"),  # aqua
    ("#f4cfd6", "#d996a4"),  # rose
    ("#dbe8c3", "#a9c47e"),  # sage
]
_NODE_LABEL = "#1f2a30"
MAX_GRAPH_NODES = 400
MAX_GRAPH_EDGES = 1500  # more edges make the browser's graph layout slow on large runs
MAX_INDEX_ENTITIES = 3000
MAX_INDEX_TRIPLES = 5000
DISAMBIGUATION_EXAMPLES = 3  # rows in the homonymy and synonymy example tables

_THEME_TOGGLE = (
    '<div class="theme-toggle" id="themeToggle">'
    '<button type="button" data-theme-choice="light">Light</button>'
    '<button type="button" data-theme-choice="dark">Dark</button></div>'
)


def _page(title: str, header: str, body: str, extra_scripts: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<link rel="stylesheet" href="report.css">
</head>
<body>
{header}
<main>
{body}
</main>
{extra_scripts}
<script src="report.js"></script>
</body>
</html>
"""


def _bar_chart(title: str, subtitle: str, rows: list[tuple[str, int]], slots: int | None = None) -> str:
    """A horizontal bar chart; `slots` reserves room for that many bars, so
    charts with fewer rows still come out the same size as their neighbours."""
    if not rows:
        return ""
    chart_w, label_w, row_h, bar_h, top = 760, 220, 26, 15, 56
    height = top + max(len(rows), slots or 0) * row_h + 12
    max_val = max(value for _, value in rows) or 1
    bar_max = chart_w - label_w - 70
    bars = []
    for i, (label, value) in enumerate(rows):
        y = top + i * row_h
        cy = y + bar_h / 2
        width = max(value / max_val * bar_max, 2.0)
        text = label if len(label) <= 28 else label[:27] + "…"
        bars.append(
            f'<text x="0" y="{cy:.1f}" class="axis-label"><title>{escape(label)}</title>{escape(text)}</text>'
            f'<rect x="{label_w}" y="{y:.1f}" width="{width:.1f}" height="{bar_h}" rx="4" fill="{_PALETTE[i % len(_PALETTE)]}"></rect>'
            f'<text x="{label_w + width + 8:.1f}" y="{cy:.1f}" class="value-label">{value:,}</text>'
        )
    return (
        f'<div class="panel"><svg class="chart" viewBox="0 0 {chart_w} {height}" role="img" aria-label="{escape(title)}">'
        f'<text x="0" y="22" class="chart-title">{escape(title)}</text>'
        f'<text x="0" y="42" class="chart-subtitle">{escape(subtitle)}</text>{"".join(bars)}</svg></div>'
    )


def _top_n(n: int, singular: str, plural: str) -> str:
    """'The 3 labels' / 'The label' (the example tables show at most DISAMBIGUATION_EXAMPLES rows)."""
    return f"The {singular}" if n == 1 else f"The {n} {plural}"


def _json_island(element_id: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{payload}</script>'


class ReportWriter:
    def __init__(self, runner: "DisambiguatedKBCRunner") -> None:
        self.runner = runner
        self.db = runner.db
        self.html_dir = runner.run_dir / "html"

    # ---- entry point -------------------------------------------------------

    def write(self, full: bool = False, final: bool = False) -> None:
        self.html_dir.mkdir(parents=True, exist_ok=True)
        for name in ("report.css", "report.js"):
            shutil.copyfile(_ASSETS / name, self.html_dir / name)
        if not (self.html_dir / "vis-network.min.js").exists():
            shutil.copyfile(_VIS, self.html_dir / "vis-network.min.js")

        if full:
            ids = [row[0] for row in self.db.execute("SELECT id FROM entities ORDER BY seq")]
        else:
            ids = sorted(self.runner.dirty_entities)
        self.runner.dirty_entities.clear()
        for entity_id in ids:
            self._write_entity(entity_id)
        self._write_index()
        if final:
            self._write_triples_export()

    def _scalar(self, sql: str, params: tuple = ()) -> int:
        return int(self.db.execute(sql, params).fetchone()[0] or 0)

    # ---- index ---------------------------------------------------------------

    def _stats(self) -> dict[str, Any]:
        s = self._scalar
        stats = {
            "triples": s("SELECT COUNT(*) FROM triples"),
            "entities": s("SELECT COUNT(*) FROM entities"),
            "explored": s("SELECT COUNT(*) FROM entities WHERE status = 'explored'"),
            "unexplored": s("SELECT COUNT(*) FROM entities WHERE status = 'unexplored'"),
            "failed_entities": s("SELECT COUNT(*) FROM entities WHERE status = 'failed'"),
            "empty_entities": s("SELECT COUNT(*) FROM entities WHERE status = 'explored' AND num_triples = 0"),
            "predicates": s("SELECT COUNT(*) FROM predicates"),
            "concepts": s("SELECT COUNT(*) FROM concepts"),
            "literals": s("SELECT COUNT(*) FROM triples WHERE object_type = 'literal'"),
            "max_depth": s("SELECT COALESCE(MAX(depth), 0) FROM entities"),
            "pending": s("SELECT COUNT(*) FROM triples WHERE object_status != 'finished' OR predicate_status != 'finished'"),
            "failures": s("SELECT COUNT(*) FROM failures"),
            "aliases": s("SELECT COUNT(*) FROM entity_aliases"),
        }
        resolution = dict(
            self.db.execute(
                "SELECT resolution, COUNT(*) FROM triples WHERE object_type = 'ne' GROUP BY resolution"
            ).fetchall()
        )
        stats["resolution"] = {key or "pending": value for key, value in resolution.items()}
        # Surface labels: every distinct string an entity was mentioned by
        # (its label plus each differently-labelled mention resolved to it) --
        # what a string-identity KB would have had as its entities.
        stats["surface_labels"] = s(
            "SELECT COUNT(*) FROM (SELECT label FROM entities UNION "
            "SELECT object_label FROM triples WHERE object_type = 'ne' AND object_id IS NOT NULL)"
        )
        stats["homonym_labels"] = s("SELECT COUNT(*) FROM (SELECT label FROM entities GROUP BY label HAVING COUNT(*) > 1)")
        stats["homonym_entities"] = s(
            "SELECT COALESCE(SUM(n), 0) FROM (SELECT COUNT(*) AS n FROM entities GROUP BY label HAVING COUNT(*) > 1)"
        )
        stats["relation_labels"] = s(
            "SELECT COUNT(DISTINCT predicate_label) FROM triples WHERE predicate_id IS NOT NULL"
        )
        stats["class_labels"] = s(
            "SELECT COUNT(DISTINCT concept_label) FROM instance_triples WHERE concept_id IS NOT NULL"
        )
        return stats

    def _write_index(self) -> None:
        runner = self.runner
        config = runner.config
        stats = self._stats()
        run_id = runner.run_dir.name
        elapsed = time.monotonic() - runner.started_at if runner.started_at else 0
        language = PROMPT_LANGUAGES.get(config.language, {}).get("name", config.language)

        metrics = [
            ("Triples", stats["triples"]),
            ("Entities", stats["entities"]),
            ("Relations", stats["predicates"]),
            ("Classes", stats["concepts"]),
            ("Literals", stats["literals"]),
            ("Max BFS depth", stats["max_depth"]),
        ]
        metrics_html = "".join(
            f'<div class="metric"><span>{escape(name)}</span><strong>{value:,}</strong></div>' for name, value in metrics
        )

        effort, enabled = runner.llm.resolved_reasoning("elicitation")
        reasoning = ", ".join(
            bit for bit in (
                effort.capitalize() if effort else None,
                None if enabled is None else ("Enabled" if enabled else "Disabled"),
            ) if bit
        ) or "Not applicable"
        limits = []
        if config.max_entities:
            limits.append(f"Max entities: {config.max_entities:,}")
        if config.max_bfs_depth is not None:
            limits.append(f"Max BFS depth: {config.max_bfs_depth}")
        if config.max_runtime_seconds:
            limits.append(f"Max runtime: {_format_duration(config.max_runtime_seconds)}")
        if config.max_cost:
            limits.append(f"Max cost: ${config.max_cost:,.2f}")
        cost = f"${runner.total_cost():,.4f}" if runner._cost_tracked() else "Not tracked"
        config_items = [
            ("Elicitation model", runner.llm.model_label("elicitation")),
            ("NER model", runner.llm.model_label("ner")),
            ("NED model", runner.llm.model_label("ned")),
            ("Embeddings", config.embedding_model),
            ("Endpoint", config.llm_url),
            ("Language", language),
            ("Runtime (this session)", _format_duration(elapsed)),
            ("Cost", cost),
            ("Parallel requests", str(config.nthreads)),
            ("Entity merging", f"Names ≥ {round(config.ned_similarity_floor * 100)}% similar" if config.ned_similarity_floor else "Standard"),
            ("NED candidates", str(config.ned_candidates)),
            ("Reasoning (elicitation)", reasoning),
            ("Stopping criteria", ", ".join(limits) or "None set"),
            ("Generated", time.strftime("%Y-%m-%d %H:%M")),
        ]
        config_html = "".join(
            f'<div class="config-item"><span>{escape(n)}</span><strong>{escape(v)}</strong></div>' for n, v in config_items
        )

        res = stats["resolution"]
        merged = res.get("ned1", 0) + res.get("ned2", 0)
        disamb_items = [
            ("Surface labels", stats["surface_labels"], f"consolidated into {stats['entities']:,} entities"),
            ("Merged mentions", merged + res.get("cache", 0),
             f"{res.get('ned1', 0):,} triple-context, {res.get('ned2', 0):,} description-context, {res.get('cache', 0):,} cached"),
            ("New entities", res.get("new", 0), "created by description-context NED"),
            ("Shared labels", stats["homonym_labels"], f"{stats['homonym_entities']:,} distinct entities share them"),
            ("Aliases", stats["aliases"], "alternative labels attached to entities"),
            ("Relations", stats["predicates"], f"from {stats['relation_labels']:,} relation labels"),
            ("Classes", stats["concepts"], f"from {stats['class_labels']:,} class labels"),
            ("Unsure / failed", res.get("unsure", 0) + res.get("failed", 0), "objects left without an entity id"),
        ]
        disamb_html = "".join(
            f'<div class="stat"><span>{escape(n)}</span><strong>{v:,}</strong><em>{escape(e)}</em></div>'
            for n, v, e in disamb_items
        )

        homonym_rows = self.db.execute(
            "SELECT label, COUNT(*) AS n FROM entities GROUP BY label HAVING n > 1 ORDER BY n DESC, label LIMIT ?",
            (DISAMBIGUATION_EXAMPLES,),
        ).fetchall()
        homonym_html = ""
        if homonym_rows:
            rows = []
            for label, count in homonym_rows:
                members = self.db.execute(
                    "SELECT id, description FROM entities WHERE label = ? ORDER BY seq LIMIT 6", (label,)
                ).fetchall()
                links = "<br>".join(
                    f'<a href="{escape(eid)}.html">{escape(eid)}</a> <span class="muted small">{escape(desc[:140])}</span>'
                    for eid, desc in members
                )
                more = f'<br><span class="muted small">… and {count - len(members)} more</span>' if count > len(members) else ""
                rows.append(f"<tr><td>{escape(label)}</td><td>{count}</td><td>{links}{more}</td></tr>")
            homonym_html = (
                "<h3>Labels denoting several entities (homonymy)</h3>"
                f'<p class="small">{_top_n(len(homonym_rows), "label", "labels")} shared by the most entities.</p><div class="table-scroll"><table>'
                "<thead><tr><th>Label</th><th>Entities</th><th>Distinguished as</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table></div>"
            )
        alias_rows = self.db.execute(
            "SELECT e.id, e.label, COUNT(*) AS n, GROUP_CONCAT(a.alias, ' · ') FROM entity_aliases a "
            "JOIN entities e ON e.id = a.entity_id GROUP BY e.id ORDER BY n DESC, e.seq LIMIT ?",
            (DISAMBIGUATION_EXAMPLES,),
        ).fetchall()
        alias_html = ""
        if alias_rows:
            alias_html = (
                "<h3>Entities with the most aliases (synonymy)</h3>"
                f'<p class="small">{_top_n(len(alias_rows), "entity", "entities")} with the most aliases.</p><div class="table-scroll"><table>'
                "<thead><tr><th>Entity</th><th>Aliases</th><th>Merged labels</th></tr></thead><tbody>"
                + "".join(
                    f'<tr><td><a href="{escape(eid)}.html">{escape(label)}</a><span class="idref">{escape(eid)}</span></td>'
                    f"<td>{n}</td><td>{escape((aliases or '')[:400])}</td></tr>"
                    for eid, label, n, aliases in alias_rows
                )
                + "</tbody></table></div>"
            )

        if stats["unexplored"]:
            status_note = f"{stats['unexplored']:,} entities still waiting to be explored"
        else:
            status_note = "All discovered entities have been explored"
        status_note += f", {stats['pending']:,} triples still being disambiguated." if stats["pending"] else "."
        if stats["empty_entities"]:
            status_note += (
                f" {stats['empty_entities']:,} explored entities returned no facts (unknown to the model or off-topic)."
            )
        if stats["failures"]:
            status_note += f" {stats['failures']:,} items failed repeatedly (see Failures below)."

        top_relations = self.db.execute(
            "SELECT p.label, COUNT(*) AS n FROM triples t JOIN predicates p ON p.id = t.predicate_id "
            "GROUP BY t.predicate_id ORDER BY n DESC LIMIT 10"
        ).fetchall()
        top_classes = self.db.execute(
            "SELECT c.label, COUNT(*) AS n FROM instance_triples it JOIN concepts c ON c.id = it.concept_id "
            "GROUP BY it.concept_id ORDER BY n DESC LIMIT 10"
        ).fetchall()
        depth_rows = self.db.execute("SELECT depth, COUNT(*) FROM entities GROUP BY depth ORDER BY depth").fetchall()
        # Full width, one after another, all the same size.
        slots = max(len(top_relations), len(top_classes), len(depth_rows), 1)
        charts = "".join(
            f'<section class="band">{chart}</section>'
            for chart in (
                _bar_chart("Top Relations", "Most frequent relations after disambiguation.",
                           [(l, n) for l, n in top_relations], slots),
                _bar_chart("Top Classes", "Most common classes (instanceOf objects).",
                           [(l, n) for l, n in top_classes], slots),
                _bar_chart("BFS Depth Reached", "Entities discovered at each hop from the seed.",
                           [(f"Depth {d}", n) for d, n in depth_rows], slots),
            )
            if chart
        )

        failures = self.db.execute(
            "SELECT subject, stage, error, time FROM failures ORDER BY id DESC LIMIT 500"
        ).fetchall()
        failures_html = ""
        if failures:
            failures_html = (
                f'<section class="band"><details class="reveal"><summary>Failures <span class="count-badge">'
                f'{stats["failures"]}</span></summary><div class="reveal-body"><p>Items given up on after repeated '
                f"request failures or unusable answers -- most recent first.</p><div class=\"table-scroll\"><table>"
                "<thead><tr><th>Item</th><th>Stage</th><th>Error</th><th>Time</th></tr></thead><tbody>"
                + "".join(
                    f"<tr><td>{escape(str(a))}</td><td>{escape(str(b))}</td><td>{escape(str(c)[:300])}</td><td>{escape(str(d))}</td></tr>"
                    for a, b, c, d in failures
                )
                + "</tbody></table></div></div></details></section>"
            )

        nodes, edges, truncated = self._graph()
        graph_note = (
            f'<p class="graph-note">Showing the {len(nodes):,} most-connected of {stats["entities"]:,} entities. '
            "Double-click an entity to open its page.</p>"
            if truncated else '<p class="graph-note">Double-click an entity to open its page.</p>'
        )

        entities = self.db.execute(
            "SELECT id, label, description FROM entities ORDER BY seq LIMIT ?", (MAX_INDEX_ENTITIES,)
        ).fetchall()
        entity_list = "".join(
            f'<li><a href="{escape(eid)}.html" title="{escape(desc)}">{escape(label)}</a><span class="idref">{escape(eid)}</span></li>'
            for eid, label, desc in entities
        )
        entity_note = (
            f'<p class="graph-note">Showing the first {MAX_INDEX_ENTITIES:,} of {stats["entities"]:,} entities. '
            "The complete set is in the downloadable database.</p>" if stats["entities"] > MAX_INDEX_ENTITIES else ""
        )
        triples = self.db.execute(
            "SELECT t.subject_label, t.subject_id, COALESCE(p.label, t.predicate_label), t.predicate_id, "
            "t.object_label, t.object_id, t.object_type FROM triples t LEFT JOIN predicates p ON p.id = t.predicate_id "
            "ORDER BY t.seq LIMIT ?",
            (MAX_INDEX_TRIPLES,),
        ).fetchall()
        triple_note = (
            f'<p class="graph-note">Showing the first {MAX_INDEX_TRIPLES:,} of {stats["triples"]:,} triples. '
            "The complete set is in the downloadable database.</p>" if stats["triples"] > MAX_INDEX_TRIPLES else ""
        )

        prompt = escape(runner.elicitation_prompt_body())
        header = f"""<header><div class="header-inner">
<div class="header-top">
<div class="header-actions"><a class="download-button" href="/api/runs/{escape(run_id)}/download" download>&#8681; Download results</a>{_THEME_TOGGLE}</div></div>
<h1 class="hero-title">{escape(config.topic)}</h1>
<p class="lede">A disambiguated knowledge graph seeded from <a class="seed-chip" href="E0.html" title="The seed entity: where the crawl started">E0: {escape(config.seed_entity)}</a>, elicited from {escape(config.model_for('elicitation'))}.</p>
<div class="metrics">{metrics_html}</div>
</div></header>"""
        body = f"""<section class="band"><h2>Run Configuration</h2><div class="config-grid">{config_html}</div>
<details class="reveal"><summary>Elicitation Prompt</summary><div class="reveal-body"><pre class="prompt-block">{prompt}</pre>
<p class="small">Each subject's description is appended to this prompt ("Description of subject: ..."), so that homonymous subjects are elicited as the entity they denote.</p></div></details></section>
<section class="band"><h2>Knowledge Graph</h2>{graph_note}<div id="graphCanvas" class="graph-canvas"></div></section>
<section class="band"><h2>Run Status</h2><p class="note">{escape(status_note)}</p></section>
{charts}
<section class="band"><h2>Disambiguation</h2>
<p>Entities, relations and classes have unique ids. Mentions with different labels are merged into one entity (synonymy), and one label can denote several distinct entities (homonymy).</p>
<div class="stat-grid">{disamb_html}</div>{homonym_html}{alias_html}</section>
<section class="band"><details class="reveal"><summary>Entity Index <span class="count-badge" id="entityCount">{stats['entities']}</span></summary>
<div class="reveal-body">{entity_note}<input type="text" id="entitySearch" class="search-box" placeholder="Filter entities by name or id&hellip;" autocomplete="off">
<ul class="entity-grid" id="entityList">{entity_list}</ul></div></details></section>
<section class="band"><details class="reveal" id="triplesDetails"><summary>All Triples <span class="count-badge">{stats['triples']}</span></summary>
<div class="reveal-body">{triple_note}<div class="table-scroll"><table><thead><tr><th>Subject</th><th>Relation</th><th>Object</th></tr></thead>
<tbody id="tripleTableBody"></tbody></table></div></div></details></section>
{failures_html}"""
        scripts = (
            '<script src="vis-network.min.js"></script>'
            + _json_island("graph-nodes-data", nodes)
            + _json_island("graph-edges-data", edges)
            + _json_island("triple-data", [list(t) for t in triples])
        )
        html = _page(f"{config.seed_entity} — miniGPTKB", header, body, scripts)
        (self.html_dir / "index.html").write_text(html, encoding="utf-8")

    def _graph(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        rows = self.db.execute(
            """
            SELECT e.id, e.label, e.depth,
                   (SELECT COUNT(*) FROM triples WHERE subject_id = e.id AND object_id IS NOT NULL)
                 + (SELECT COUNT(*) FROM triples WHERE object_id = e.id) AS degree
            FROM entities e ORDER BY degree DESC, e.seq LIMIT ?
            """,
            (MAX_GRAPH_NODES,),
        ).fetchall()
        total = self._scalar("SELECT COUNT(*) FROM entities")
        selected = {row[0] for row in rows}
        nodes = [
            {
                "id": eid,
                "label": label,
                "title": f"{label} ({eid}, depth {depth}, {degree} links)",
                "color": {
                    "background": _NODE_PALETTE[depth % len(_NODE_PALETTE)][0],
                    "border": _NODE_PALETTE[depth % len(_NODE_PALETTE)][1],
                    "highlight": {"background": _NODE_PALETTE[depth % len(_NODE_PALETTE)][0], "border": "#0e7c8f"},
                    "hover": {"background": _NODE_PALETTE[depth % len(_NODE_PALETTE)][0], "border": "#0e7c8f"},
                },
                "font": {"color": _NODE_LABEL},
                "size": 24 if eid == "E0" else min(10 + degree * 1.5, 40),
                "borderWidth": 3 if eid == "E0" else 1.5,
            }
            for eid, label, depth, degree in rows
        ]
        edges = []
        for subject_id, object_id, s_label, p_label, o_label in self.db.execute(
            "SELECT t.subject_id, t.object_id, t.subject_label, COALESCE(p.label, t.predicate_label), t.object_label "
            "FROM triples t LEFT JOIN predicates p ON p.id = t.predicate_id "
            "WHERE t.object_id IS NOT NULL AND t.subject_id != t.object_id ORDER BY t.seq"
        ):
            if subject_id in selected and object_id in selected:
                edges.append({"from": subject_id, "to": object_id, "title": f"{s_label} — {p_label} → {o_label}"})
                if len(edges) >= MAX_GRAPH_EDGES:
                    break
        return nodes, edges, total > len(nodes)

    # ---- entity pages ------------------------------------------------------

    def _entity_link(self, entity_id: str | None, label: str) -> str:
        if not entity_id:
            return escape(label)
        return f'<a href="{escape(entity_id)}.html">{escape(label)}</a><span class="idref">{escape(entity_id)}</span>'

    def _write_entity(self, entity_id: str) -> None:
        row = self.db.execute(
            "SELECT id, label, description, status, depth, first_appeared, num_triples FROM entities WHERE id = ?",
            (entity_id,),
        ).fetchone()
        if not row:
            return
        _, label, description, status, depth, first_appeared, num_triples = row
        aliases = [r[0] for r in self.db.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias", (entity_id,)
        )]
        classes = self.db.execute(
            "SELECT c.label, c.id FROM instance_triples it JOIN concepts c ON c.id = it.concept_id "
            "WHERE it.entity_id = ? ORDER BY c.label",
            (entity_id,),
        ).fetchall()
        homonyms = self.db.execute(
            "SELECT id, description FROM entities WHERE label = ? AND id != ? ORDER BY seq LIMIT 50", (label, entity_id)
        ).fetchall()

        outgoing = self.db.execute(
            "SELECT COALESCE(p.label, t.predicate_label), t.predicate_label, t.predicate_id, t.object_label, t.object_id, "
            "t.object_type, t.object_status, t.resolution, e.label "
            "FROM triples t LEFT JOIN predicates p ON p.id = t.predicate_id LEFT JOIN entities e ON e.id = t.object_id "
            "WHERE t.subject_id = ? ORDER BY t.seq",
            (entity_id,),
        ).fetchall()
        out_rows = []
        for p_label, raw_p, p_id, o_label, o_id, o_type, o_status, resolution, o_entity_label in outgoing:
            predicate = escape(p_label) + (f'<span class="idref">{escape(p_id)}</span>' if p_id else "")
            if raw_p != p_label:
                predicate += f'<br><span class="muted small">as "{escape(raw_p)}"</span>'
            if o_id:
                obj = self._entity_link(o_id, o_entity_label or o_label)
                if o_entity_label and o_entity_label != o_label:
                    obj += f'<br><span class="muted small">mentioned as "{escape(o_label)}"</span>'
            else:
                obj = escape(o_label)
            kind = {"ne": "entity", "literal": "literal", "class": "class"}.get(o_type, "pending")
            if o_type == "ne" and not o_id:
                kind = resolution if resolution in ("unsure", "failed") else ("pending" if o_status != "finished" else "entity")
            out_rows.append(f'<tr><td>{predicate}</td><td>{obj} <span class="tag">{escape(kind)}</span></td></tr>')

        incoming = self.db.execute(
            "SELECT t.subject_id, s.label, COALESCE(p.label, t.predicate_label), t.object_label, t.resolution "
            "FROM triples t LEFT JOIN predicates p ON p.id = t.predicate_id LEFT JOIN entities s ON s.id = t.subject_id "
            "WHERE t.object_id = ? ORDER BY t.seq LIMIT 1000",
            (entity_id,),
        ).fetchall()
        in_rows = [
            f"<tr><td>{self._entity_link(s_id, s_label or s_id)}</td><td>{escape(p_label)}</td>"
            f"<td>{escape(o_label)}{'' if o_label == label else ' <span class=\"tag\">alias</span>'}"
            f' <span class="tag">{escape(resolution or "")}</span></td></tr>'
            for s_id, s_label, p_label, o_label, resolution in incoming
        ]

        decisions = self.db.execute(
            "SELECT d.stage, d.target, d.context, d.candidates, d.outcome, t.subject_id, t.subject_label "
            "FROM decisions d LEFT JOIN triples t ON t.id = d.triple_id "
            "WHERE d.chosen_id = ? AND d.stage IN ('ned1', 'ned2') ORDER BY d.id LIMIT 100",
            (entity_id,),
        ).fetchall()
        decision_rows = []
        stage_names = {"ned1": "Triple-context NED", "ned2": "Description-context NED"}
        for stage, target, context, candidate_json, outcome, s_id, s_label in decisions:
            try:
                candidate_ids = json.loads(candidate_json or "[]")
            except json.JSONDecodeError:
                candidate_ids = []
            labels = dict(self.db.execute(
                f"SELECT id, label FROM entities WHERE id IN ({','.join('?' * len(candidate_ids))})", candidate_ids
            ).fetchall()) if candidate_ids else {}
            candidate_html = ", ".join(self._entity_link(c, labels.get(c, c)) for c in candidate_ids) or "<span class=\"muted\">none</span>"
            verdict = "created this entity" if outcome == "new" else "matched this entity"
            decision_rows.append(
                f"<tr><td>{escape(target)}<br><span class=\"muted small\">in {self._entity_link(s_id, s_label or '')}</span></td>"
                f"<td>{escape(stage_names.get(stage, stage))}<br><span class=\"muted small\">{escape(verdict)}</span></td>"
                f"<td class=\"small\">{escape(context or '')}</td><td class=\"small\">{candidate_html}</td></tr>"
            )

        first = ""
        if first_appeared and first_appeared.startswith("T"):
            src = self.db.execute(
                "SELECT subject_id, subject_label, predicate_label FROM triples WHERE id = ?", (first_appeared,)
            ).fetchone()
            if src:
                first = f"First mentioned as the object of {self._entity_link(src[0], src[1])} &mdash; {escape(src[2])}."
        elif entity_id == "E0":
            first = "The seed entity, where the exploration starts."

        status_text = {
            "explored": f"Explored ({num_triples or 0} triples elicited)" if num_triples is not None else "Explored",
            "unexplored": "Not explored yet",
            "failed": "Elicitation failed",
        }.get(status, status)
        facts = [
            ("Status", escape(status_text)),
            ("BFS depth", str(depth)),
            ("Classes", " ".join(f'<span class="tag">{escape(c)}</span>' for c, _ in classes) or "&mdash;"),
            ("Aliases", " ".join(f'<span class="tag">{escape(a)}</span>' for a in aliases) or "&mdash;"),
        ]
        facts_html = "".join(f'<div class="config-item"><span>{n}</span><strong>{v}</strong></div>' for n, v in facts)
        homonym_html = ""
        if homonyms:
            homonym_html = (
                f'<section class="band"><h2>Other entities labelled "{escape(label)}"</h2>'
                "<p>Distinct entities sharing this label, kept apart by disambiguation.</p><table><tbody>"
                + "".join(
                    f'<tr><td>{self._entity_link(hid, label)}</td><td class="small">{escape(hdesc)}</td></tr>'
                    for hid, hdesc in homonyms
                )
                + "</tbody></table></section>"
            )
        decisions_html = ""
        if decision_rows:
            decisions_html = (
                '<section class="band"><h2>Disambiguation Decisions</h2>'
                "<p>Mentions resolved to this entity by the LLM, with the context and candidates it was shown.</p>"
                '<div class="table-scroll"><table><thead><tr><th>Mention</th><th>Step</th><th>Context</th><th>Candidates</th></tr></thead><tbody>'
                + "".join(decision_rows) + "</tbody></table></div></section>"
            )

        header = f"""<header class="compact"><div class="header-inner">
<div class="header-top"><a class="header-link" href="index.html">&larr; {escape(self.runner.config.topic)}</a>{_THEME_TOGGLE}</div>
<h1 class="hero-title">{escape(label)}<span class="entity-id">{escape(entity_id)}</span></h1>
<p class="description">{escape(description)}</p>
</div></header>"""
        body = f"""<section class="band"><div class="config-grid">{facts_html}</div><p class="small">{first}</p></section>
<section class="band"><h2>Outgoing</h2><div class="table-scroll"><table><thead><tr><th>Relation</th><th>Object</th></tr></thead><tbody>
{''.join(out_rows) or '<tr><td colspan="2">No facts elicited for this entity yet.</td></tr>'}</tbody></table></div></section>
<section class="band"><h2>Incoming</h2><div class="table-scroll"><table><thead><tr><th>Subject</th><th>Relation</th><th>Mentioned as</th></tr></thead><tbody>
{''.join(in_rows) or '<tr><td colspan="3">No incoming triples yet.</td></tr>'}</tbody></table></div></section>
{decisions_html}
{homonym_html}"""
        html = _page(f"{label} ({entity_id}) — miniGPTKB", header, body)
        (self.html_dir / f"{entity_id}.html").write_text(html, encoding="utf-8")

    # ---- flat export ---------------------------------------------------------

    def _write_triples_export(self) -> None:
        """triples.jsonl: every triple with labels and ids, for use outside the database."""
        path = self.runner.run_dir / "triples.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in self.db.execute(
                "SELECT t.subject_id, t.subject_label, t.predicate_id, COALESCE(p.label, t.predicate_label), "
                "t.object_id, t.object_label, t.object_type, it.concept_id "
                "FROM triples t LEFT JOIN predicates p ON p.id = t.predicate_id "
                "LEFT JOIN instance_triples it ON it.triple_id = t.id ORDER BY t.seq"
            ):
                s_id, s, p_id, p, o_id, o, o_type, concept_id = row
                file.write(json.dumps(
                    {"s_id": s_id, "s": s, "p_id": p_id, "p": p, "o_id": o_id or concept_id, "o": o, "o_type": o_type},
                    ensure_ascii=False,
                ) + "\n")
