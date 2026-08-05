"""Render the customer-facing comparison report.

Two outputs, because they get used differently: a self-contained HTML page the
SA walks the customer through, and a markdown summary that can be pasted into
an email or a deal thread without an attachment.

The report is written to be read by a sceptic. Sample sizes, confidence
intervals, and the provenance of the ground truth are stated up front rather
than buried, because the first question a serious prospect asks is "how do you
know?" — and the honest answer is more persuasive than a bare win rate.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from parse_bench.analysis.metric_definitions import display_name as metric_display_name
from parse_bench.customer.comparison.scores import CategoryScores
from parse_bench.customer.comparison.stats import (
    ComparisonRow,
    PipelineSummary,
    compare_to_baseline,
)
from parse_bench.customer.project import CustomerProjectConfig, ProjectPaths

REPORT_HTML_FILENAME = "comparison_report.html"
REPORT_MARKDOWN_FILENAME = "comparison_report.md"
REPORT_JSON_FILENAME = "comparison_report.json"


def _fmt_score(value: float) -> str:
    return "—" if value is None or math.isnan(value) else f"{value * 100:.1f}"


def _fmt_delta(value: float) -> str:
    return "—" if value is None or math.isnan(value) else f"{value * 100:+.1f}"


def _fmt_p(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    return "<0.001"


def _category_label(category: str) -> str:
    return category.replace("_", " ").title()


def build_report_data(
    config: CustomerProjectConfig,
    scores: dict[str, CategoryScores],
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble everything the renderers need, as plain data.

    Kept separate from rendering so the numbers can be tested, and so the JSON
    export is exactly what the HTML shows.
    """
    baseline = config.resolved_baseline()
    challengers = config.challengers()

    categories: list[dict[str, Any]] = []
    for name in sorted(scores):
        category_scores = scores[name]
        summaries, rows = compare_to_baseline(category_scores, baseline or "", challengers)
        categories.append(
            {
                "category": name,
                "label": _category_label(name),
                "metric": category_scores.metric,
                "metric_label": metric_display_name(category_scores.metric),
                "summaries": [asdict(s) for s in summaries],
                "comparisons": [asdict(r) for r in rows],
                "verdicts": {r.challenger: r.verdict() for r in rows},
            }
        )

    overall = _overall_standings(scores, config.pipelines)

    return {
        "customer": config.name,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "baseline": baseline,
        "pipelines": config.pipelines,
        "notes": config.notes,
        "ground_truth": ground_truth or {},
        "categories": categories,
        "overall": overall,
    }


def _overall_standings(
    scores: dict[str, CategoryScores],
    pipelines: list[str],
) -> list[dict[str, Any]]:
    """Macro-average across categories, so no single dimension dominates.

    Averaging category means (rather than pooling documents) keeps a category
    with 200 text pages from drowning out one with 20 tables.
    """
    standings: list[dict[str, Any]] = []
    for pipeline in pipelines:
        per_category: dict[str, float] = {}
        for name, category_scores in scores.items():
            values = category_scores.by_pipeline.get(pipeline)
            if values:
                per_category[name] = sum(values.values()) / len(values)
        if not per_category:
            continue
        standings.append(
            {
                "pipeline": pipeline,
                "overall": sum(per_category.values()) / len(per_category),
                "per_category": per_category,
                "categories_scored": len(per_category),
            }
        )
    standings.sort(key=lambda s: -s["overall"])
    return standings


# ── Markdown ─────────────────────────────────────────────────────────────────


def render_markdown(data: dict[str, Any]) -> str:
    """Render the summary an SA can paste into an email."""
    lines: list[str] = []
    lines.append(f"# Document parsing evaluation — {data['customer']}")
    lines.append("")
    lines.append(f"Generated {data['generated_at']} · baseline: `{data['baseline']}`")
    lines.append("")

    gt = data.get("ground_truth") or {}
    if gt:
        lines.append(
            f"Ground truth: {gt.get('documents', 0)} document(s), "
            f"{gt.get('rules', 0)} rule(s), "
            f"{gt.get('verified_pct', 0):.0f}% human-verified"
            + (f", bootstrapped with `{gt['model']}`" if gt.get("model") else "")
        )
        lines.append("")

    if data["overall"]:
        lines.append("## Overall (macro-average across dimensions)")
        lines.append("")
        lines.append("| Pipeline | Overall | Dimensions |")
        lines.append("|---|---:|---:|")
        for row in data["overall"]:
            lines.append(f"| {row['pipeline']} | {_fmt_score(row['overall'])} | {row['categories_scored']} |")
        lines.append("")

    for category in data["categories"]:
        lines.append(f"## {category['label']} — {category['metric_label']}")
        lines.append("")
        lines.append("| Pipeline | Score | 95% CI | Docs |")
        lines.append("|---|---:|---:|---:|")
        for summary in category["summaries"]:
            ci = f"{_fmt_score(summary['ci_low'])}–{_fmt_score(summary['ci_high'])}"
            lines.append(f"| {summary['pipeline']} | {_fmt_score(summary['mean'])} | {ci} | {summary['n']} |")
        lines.append("")

        if category["comparisons"]:
            lines.append(f"Versus `{data['baseline']}`:")
            lines.append("")
            for row in category["comparisons"]:
                lines.append(f"- **{row['challenger']}** — {category['verdicts'][row['challenger']]}")
                lines.append(f"  - won {row['wins']}, lost {row['losses']}, tied {row['ties']} of {row['n']} documents")
            lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.extend(f"- {line}" for line in METHODOLOGY_NOTES)
    if data.get("notes"):
        lines.append("")
        lines.append(f"Notes: {data['notes']}")
    lines.append("")
    return "\n".join(lines)


METHODOLOGY_NOTES: tuple[str, ...] = (
    "Every pipeline is scored on the same documents, so comparisons are paired: "
    "each document contributes one score per pipeline and the difference is taken per document.",
    "Confidence intervals are 95% percentile bootstrap (10,000 resamples, fixed seed — "
    "re-running the report gives identical numbers).",
    "P-values are two-sided Wilcoxon signed-rank, Holm-corrected across the challengers "
    "in each dimension. Parse scores are bounded and tie-heavy, which is why a "
    "non-parametric test is used rather than a t-test.",
    "Scoring is fully deterministic and rule-based. No language model judges any output.",
    "A dimension with fewer than 10 documents is flagged as underpowered: the numbers are "
    "shown, but they should not settle an argument.",
)


# ── HTML ─────────────────────────────────────────────────────────────────────

# The palette matches the other ParseBench reports. Web fonts are deliberately
# not linked: these reports are opened offline inside customer environments,
# and a page that renders identically without a network is worth more than an
# exact typeface match.
_CSS = """
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg: #f8f7f4; --fg: #1c1917; --card: #ffffff; --border: #e7e5e4;
    --muted: #78716c; --muted-light: #a8a29e; --cream: #faf9f6;
    --emerald: #059669; --emerald-bg: #ecfdf5;
    --amber: #d97706; --amber-bg: #fffbeb;
    --red: #dc2626; --red-bg: #fef2f2;
    --blue: #2563eb; --blue-bg: #eff6ff;
    --font-heading: 'Newsreader', Georgia, 'Times New Roman', serif;
    --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
    --radius: 12px;
}
html { font-size: 15px; }
body {
    font-family: var(--font-body); background: var(--bg); color: var(--fg);
    line-height: 1.6; padding: 2.5rem 1.5rem;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-family: var(--font-heading); font-size: 2.1rem; font-weight: 600; letter-spacing: -0.01em; }
h2 { font-family: var(--font-heading); font-size: 1.45rem; font-weight: 600; margin: 2.5rem 0 0.35rem; }
h3 { font-size: 0.95rem; font-weight: 600; margin: 1.4rem 0 0.5rem; }
.sub { color: var(--muted); font-size: 0.9rem; margin-top: 0.35rem; }
.card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.25rem 1.4rem; margin-top: 1rem;
}
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { padding: 0.55rem 0.7rem; text-align: left; border-bottom: 1px solid var(--border); }
th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
tr.baseline td { background: var(--cream); }
.pill {
    display: inline-block; padding: 0.12rem 0.5rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
}
.pill.win { background: var(--emerald-bg); color: var(--emerald); }
.pill.loss { background: var(--red-bg); color: var(--red); }
.pill.flat { background: var(--cream); color: var(--muted); }
.pill.weak { background: var(--amber-bg); color: var(--amber); }
.pill.base { background: var(--blue-bg); color: var(--blue); }
.verdict { font-size: 0.9rem; margin: 0.15rem 0 0.9rem; }
.verdict .name { font-weight: 600; }
.wl { font-family: var(--font-mono); font-size: 0.8rem; color: var(--muted); }
.method {
    background: var(--cream); border-left: 3px solid var(--border);
    padding: 1rem 1.2rem; border-radius: 0 var(--radius) var(--radius) 0;
}
.method li { margin: 0.4rem 0 0.4rem 1rem; font-size: 0.87rem; color: var(--muted); }
.meta { display: flex; flex-wrap: wrap; gap: 1.75rem; margin-top: 1rem; }
.meta div { font-size: 0.85rem; }
.meta .k { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
.meta .v { font-family: var(--font-mono); font-size: 0.95rem; }
code { font-family: var(--font-mono); font-size: 0.85em; }
footer { margin-top: 3rem; color: var(--muted-light); font-size: 0.8rem; }
"""


def _pill_for(row: dict[str, Any]) -> str:
    if row["underpowered"]:
        return '<span class="pill weak">underpowered</span>'
    if row["p_value_adjusted"] >= 0.05:
        return '<span class="pill flat">no sig. difference</span>'
    if row["mean_difference"] > 0:
        return '<span class="pill win">challenger better</span>'
    return '<span class="pill loss">baseline better</span>'


def _render_summary_table(category: dict[str, Any], baseline: str | None) -> str:
    parts = [
        "<table><thead><tr>",
        "<th>Pipeline</th>",
        '<th class="num">Score</th>',
        '<th class="num">95% CI</th>',
        '<th class="num">Docs</th>',
        "</tr></thead><tbody>",
    ]
    for summary in category["summaries"]:
        is_baseline = summary["pipeline"] == baseline
        parts.append(f'<tr class="{"baseline" if is_baseline else ""}">')
        label = html.escape(summary["pipeline"])
        if is_baseline:
            label += ' <span class="pill base">baseline</span>'
        parts.append(f"<td>{label}</td>")
        parts.append(f'<td class="num">{_fmt_score(summary["mean"])}</td>')
        parts.append(f'<td class="num">{_fmt_score(summary["ci_low"])}–{_fmt_score(summary["ci_high"])}</td>')
        parts.append(f'<td class="num">{summary["n"]}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _render_comparisons(category: dict[str, Any]) -> str:
    if not category["comparisons"]:
        return '<p class="sub">No challenger produced scores in this dimension.</p>'

    parts = [
        "<table><thead><tr>",
        "<th>Challenger</th>",
        '<th class="num">Δ vs baseline</th>',
        '<th class="num">95% CI</th>',
        '<th class="num">p (Holm)</th>',
        '<th class="num">Effect</th>',
        '<th class="num">W / L / T</th>',
        "<th>Result</th>",
        "</tr></thead><tbody>",
    ]
    for row in category["comparisons"]:
        parts.append("<tr>")
        parts.append(f"<td>{html.escape(row['challenger'])}</td>")
        parts.append(f'<td class="num">{_fmt_delta(row["mean_difference"])}</td>')
        parts.append(f'<td class="num">{_fmt_delta(row["ci_low"])} / {_fmt_delta(row["ci_high"])}</td>')
        parts.append(f'<td class="num">{_fmt_p(row["p_value_adjusted"])}</td>')
        parts.append(f'<td class="num">{row["effect_size"]:.2f}</td>')
        parts.append(f'<td class="num wl">{row["wins"]} / {row["losses"]} / {row["ties"]}</td>')
        parts.append(f"<td>{_pill_for(row)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    for row in category["comparisons"]:
        verdict = html.escape(category["verdicts"][row["challenger"]])
        parts.append(f'<p class="verdict"><span class="name">{html.escape(row["challenger"])}</span> — {verdict}</p>')
    return "".join(parts)


def render_html(data: dict[str, Any]) -> str:
    """Render the self-contained HTML report."""
    gt = data.get("ground_truth") or {}
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"<title>Parsing evaluation — {html.escape(data['customer'])}</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        "<h1>Document parsing evaluation</h1>",
        f'<p class="sub">{html.escape(data["customer"])} · {data["generated_at"]} · '
        f"baseline <code>{html.escape(str(data['baseline']))}</code></p>",
    ]

    parts.append('<div class="card"><div class="meta">')
    parts.append(f'<div><div class="k">Documents</div><div class="v">{gt.get("documents", 0)}</div></div>')
    parts.append(f'<div><div class="k">Rules</div><div class="v">{gt.get("rules", 0)}</div></div>')
    parts.append(f'<div><div class="k">Human-verified</div><div class="v">{gt.get("verified_pct", 0):.0f}%</div></div>')
    if gt.get("model"):
        parts.append(
            f'<div><div class="k">Ground truth from</div><div class="v">{html.escape(str(gt["model"]))}</div></div>'
        )
    parts.append(f'<div><div class="k">Pipelines</div><div class="v">{len(data["pipelines"])}</div></div>')
    parts.append("</div></div>")

    if data["overall"]:
        parts.append("<h2>Overall</h2>")
        parts.append('<p class="sub">Macro-average across dimensions — each dimension weighted equally.</p>')
        parts.append('<div class="card"><table><thead><tr><th>Pipeline</th>')
        parts.append('<th class="num">Overall</th><th class="num">Dimensions</th></tr></thead><tbody>')
        for row in data["overall"]:
            is_baseline = row["pipeline"] == data["baseline"]
            label = html.escape(row["pipeline"])
            if is_baseline:
                label += ' <span class="pill base">baseline</span>'
            parts.append(f'<tr class="{"baseline" if is_baseline else ""}"><td>{label}</td>')
            parts.append(f'<td class="num">{_fmt_score(row["overall"])}</td>')
            parts.append(f'<td class="num">{row["categories_scored"]}</td></tr>')
        parts.append("</tbody></table></div>")

    for category in data["categories"]:
        parts.append(f"<h2>{html.escape(category['label'])}</h2>")
        parts.append(
            f'<p class="sub">Scored on <code>{html.escape(category["metric"])}</code> — '
            f"{html.escape(category['metric_label'])}</p>"
        )
        parts.append('<div class="card">')
        parts.append(_render_summary_table(category, data["baseline"]))
        parts.append("<h3>Compared with the baseline</h3>")
        parts.append(_render_comparisons(category))
        parts.append("</div>")

    parts.append("<h2>How to read this</h2>")
    parts.append('<div class="method"><ul>')
    for note in METHODOLOGY_NOTES:
        parts.append(f"<li>{html.escape(note)}</li>")
    parts.append("</ul></div>")

    if data.get("notes"):
        parts.append(f'<p class="sub">{html.escape(data["notes"])}</p>')

    parts.append(
        "<footer>Generated by ParseBench · every number on this page can be "
        "recomputed from the evaluation output in this project directory.</footer>"
    )
    parts.append("</div></body></html>")
    return "\n".join(parts)


def write_reports(
    paths: ProjectPaths,
    data: dict[str, Any],
) -> dict[str, Path]:
    """Write HTML, markdown, and JSON reports into the project's reports/ dir."""
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    html_path = paths.reports_dir / REPORT_HTML_FILENAME
    md_path = paths.reports_dir / REPORT_MARKDOWN_FILENAME
    json_path = paths.reports_dir / REPORT_JSON_FILENAME

    html_path.write_text(render_html(data), encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

    return {"html": html_path, "markdown": md_path, "json": json_path}


__all__ = [
    "ComparisonRow",
    "PipelineSummary",
    "build_report_data",
    "render_html",
    "render_markdown",
    "write_reports",
]
