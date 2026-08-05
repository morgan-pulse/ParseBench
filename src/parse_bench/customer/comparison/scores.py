"""Load per-document scores from evaluation output.

Aggregate scores can't support a paired comparison — for that you need each
pipeline's score on each individual document, keyed by test id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from parse_bench.analysis.aggregation_report import _DEFAULT_METRICS
from parse_bench.customer.project import ProjectPaths
from parse_bench.schemas.evaluation import EvaluationSummary

# Fallback when a category has no configured default and no rule pass rate.
_GENERIC_FALLBACK_METRICS = ("rule_pass_rate", "normalized_text_score")


@dataclass
class CategoryScores:
    """Per-document scores for one category, across pipelines."""

    category: str
    metric: str
    # pipeline -> test_id -> score
    by_pipeline: dict[str, dict[str, float]] = field(default_factory=dict)
    # pipeline -> stat name -> aggregate value (cost, latency, ...)
    stats_by_pipeline: dict[str, dict[str, float]] = field(default_factory=dict)

    def pipelines(self) -> list[str]:
        return list(self.by_pipeline)

    def common_test_ids(self, pipeline_a: str, pipeline_b: str) -> list[str]:
        """Documents both pipelines produced a score for, in stable order."""
        a = self.by_pipeline.get(pipeline_a, {})
        b = self.by_pipeline.get(pipeline_b, {})
        return sorted(set(a) & set(b))

    def paired(self, pipeline_a: str, pipeline_b: str) -> tuple[list[float], list[float], list[str]]:
        """Aligned score vectors for two pipelines over their common documents."""
        test_ids = self.common_test_ids(pipeline_a, pipeline_b)
        a = self.by_pipeline[pipeline_a]
        b = self.by_pipeline[pipeline_b]
        return [a[t] for t in test_ids], [b[t] for t in test_ids], test_ids


def _load_summary(path: Path) -> EvaluationSummary | None:
    if not path.exists():
        return None
    try:
        return EvaluationSummary.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _aggregate_stats(summary: EvaluationSummary) -> dict[str, float]:
    """Flatten operational stats to the averages worth putting in a report."""
    stats: dict[str, float] = {}
    for name, values in summary.aggregate_stats.items():
        if not isinstance(values, dict):
            continue
        for key in ("avg", "total"):
            value = values.get(key)
            if isinstance(value, (int, float)):
                stats[f"{name}_{key}"] = float(value)
    return stats


def load_scores(
    paths: ProjectPaths,
    pipelines: list[str],
    categories: list[str] | None = None,
) -> dict[str, CategoryScores]:
    """Load per-document scores for every pipeline that has been evaluated.

    :param pipelines: Pipeline names to load, in report order.
    :param categories: Restrict to these categories; None discovers them.
    :return: Category name -> CategoryScores. Categories with no data are omitted.
    """
    # Pass 1: collect every summary, so the metric is chosen from what all
    # pipelines report rather than from whichever one happened to load first.
    summaries: dict[str, dict[str, EvaluationSummary]] = {}
    for pipeline in pipelines:
        pipeline_dir = paths.pipeline_output_dir(pipeline)
        if not pipeline_dir.exists():
            continue
        for category_dir in sorted(pipeline_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            if categories is not None and category not in categories:
                continue
            summary = _load_summary(category_dir / "_evaluation_report.json")
            if summary is not None:
                summaries.setdefault(category, {})[pipeline] = summary

    # Pass 2: pick one metric per category, then pull per-document scores.
    discovered: dict[str, CategoryScores] = {}
    for category, per_pipeline in summaries.items():
        metric = _pick_shared_metric(per_pipeline, category)
        if metric is None:
            continue
        entry = CategoryScores(category=category, metric=metric)

        for pipeline, summary in per_pipeline.items():
            scores: dict[str, float] = {}
            for result in summary.per_example_results:
                if not result.success:
                    continue
                for m in result.metrics:
                    if m.metric_name == metric:
                        scores[result.test_id] = float(m.value)
                        break
            if scores:
                entry.by_pipeline[pipeline] = scores
                entry.stats_by_pipeline[pipeline] = _aggregate_stats(summary)

        if entry.by_pipeline:
            discovered[category] = entry

    return discovered


def _pick_shared_metric(
    per_pipeline: dict[str, EvaluationSummary],
    category: str,
) -> str | None:
    """Pick a metric every pipeline in the category reports, if one exists.

    Comparing pipelines on different metrics would be meaningless, so a metric
    common to all of them wins over the nominal default.
    """
    metric_sets: list[set[str]] = []
    for summary in per_pipeline.values():
        available = {m.metric_name for result in summary.per_example_results for m in result.metrics}
        if available:
            metric_sets.append(available)
    if not metric_sets:
        return None

    shared = set.intersection(*metric_sets)
    candidates = shared or set().union(*metric_sets)

    preferred = _DEFAULT_METRICS.get(category)
    if preferred and preferred in candidates:
        return preferred
    for fallback in _GENERIC_FALLBACK_METRICS:
        if fallback in candidates:
            return fallback
    return sorted(candidates)[0]
