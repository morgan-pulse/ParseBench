"""Paired significance testing for pipeline comparisons.

Documents are the unit of analysis, and every pipeline sees the same documents,
so the comparison is paired throughout. A paired test is what makes "we beat
them by 4 points" a claim you can defend: it controls for the fact that some
documents are simply harder than others.

Three numbers are reported per comparison, and all three matter:

* **mean difference with a bootstrap CI** — the size of the effect, on the
  metric's own scale. This is what a customer actually cares about.
* **Wilcoxon signed-rank p-value** — whether the difference survives the
  sample size. Non-parametric because parse scores are bounded, skewed, and
  full of ties at 0.0 and 1.0, which breaks the t-test's assumptions.
* **win / loss / tie counts** — the honest picture when a mean hides a
  bimodal split (great on half the documents, terrible on the other half).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from parse_bench.customer.comparison.scores import CategoryScores

# Fixed so a report is reproducible: an SA re-running it in front of a customer
# must get the same intervals.
BOOTSTRAP_SEED = 20260101
BOOTSTRAP_RESAMPLES = 10_000

# Below this, a paired test says almost nothing. Results are still reported —
# suppressing them would be worse — but they are flagged as underpowered.
MIN_DOCUMENTS_FOR_POWER = 10

# Differences smaller than this count as a tie in the win/loss tally: parse
# metrics wobble in the third decimal for reasons no customer cares about.
TIE_THRESHOLD = 1e-6


@dataclass
class PipelineSummary:
    """A single pipeline's standing on one category."""

    pipeline: str
    n: int
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonRow:
    """One challenger measured against the baseline on one category."""

    category: str
    metric: str
    baseline: str
    challenger: str
    n: int
    baseline_mean: float
    challenger_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_adjusted: float
    effect_size: float
    wins: int
    losses: int
    ties: int
    underpowered: bool

    @property
    def significant(self) -> bool:
        """Significant at alpha=0.05 after multiple-comparison correction."""
        return self.p_value_adjusted < 0.05 and not self.underpowered

    @property
    def direction(self) -> str:
        if not self.significant:
            return "inconclusive"
        return "challenger better" if self.mean_difference > 0 else "baseline better"

    def verdict(self) -> str:
        """A sentence an SA can read aloud without over-claiming."""
        delta = abs(self.mean_difference) * 100
        if self.underpowered:
            return f"Only {self.n} document(s) — too few to draw a conclusion. Observed gap: {delta:.1f} points."
        if not self.significant:
            return f"No significant difference ({delta:.1f} point gap, p={self.p_value_adjusted:.3f}, n={self.n})."
        better = self.challenger if self.mean_difference > 0 else self.baseline
        return (
            f"{better} is better by {delta:.1f} points "
            f"(95% CI {self.ci_low * 100:+.1f} to {self.ci_high * 100:+.1f}, "
            f"p={self.p_value_adjusted:.4f}, n={self.n})."
        )


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Returns ``(nan, nan)`` for empty input and a degenerate interval for a
    single observation, since neither supports an interval estimate.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return (math.nan, math.nan)
    if array.size == 1:
        return (float(array[0]), float(array[0]))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(low), float(high))


def wilcoxon_p_value(differences: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired differences.

    Returns 1.0 when every pair is tied (nothing to detect) or when SciPy
    cannot compute a statistic, which is the conservative answer.
    """
    array = np.asarray(differences, dtype=float)
    nonzero = array[np.abs(array) > TIE_THRESHOLD]
    if nonzero.size == 0:
        return 1.0

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return 1.0

    try:
        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return 1.0
    p = float(result.pvalue)
    return 1.0 if math.isnan(p) else p


def cohens_dz(differences: Sequence[float]) -> float:
    """Standardized effect size for paired differences.

    Zero when the differences have no spread, which is the sensible reading:
    a constant offset has no variance to standardize against.
    """
    array = np.asarray(differences, dtype=float)
    if array.size < 2:
        return 0.0
    std = float(array.std(ddof=1))
    if std < TIE_THRESHOLD:
        return 0.0
    return float(array.mean() / std)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Comparing several challengers against one baseline is several tests, and
    one of them clearing p<0.05 by luck is exactly the sort of result a
    prospect's data-science team will pull apart. Holm controls that while
    losing less power than plain Bonferroni.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def summarize_pipeline(
    pipeline: str,
    scores: dict[str, float],
    stats: dict[str, float] | None = None,
) -> PipelineSummary:
    """Mean, median, spread, and bootstrap CI for one pipeline's scores."""
    values = list(scores.values())
    array = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(values)
    return PipelineSummary(
        pipeline=pipeline,
        n=len(values),
        mean=float(array.mean()) if array.size else math.nan,
        median=float(np.median(array)) if array.size else math.nan,
        std=float(array.std(ddof=1)) if array.size > 1 else 0.0,
        ci_low=low,
        ci_high=high,
        stats=stats or {},
    )


def compare_pair(
    category: str,
    metric: str,
    baseline: str,
    challenger: str,
    baseline_scores: Sequence[float],
    challenger_scores: Sequence[float],
) -> ComparisonRow:
    """Run the paired comparison for one challenger against the baseline.

    Score vectors must already be aligned document by document.
    """
    baseline_array = np.asarray(baseline_scores, dtype=float)
    challenger_array = np.asarray(challenger_scores, dtype=float)
    if baseline_array.shape != challenger_array.shape:
        raise ValueError("Paired comparison requires equal-length score vectors")

    differences = challenger_array - baseline_array
    n = int(differences.size)

    if n == 0:
        return ComparisonRow(
            category=category,
            metric=metric,
            baseline=baseline,
            challenger=challenger,
            n=0,
            baseline_mean=math.nan,
            challenger_mean=math.nan,
            mean_difference=math.nan,
            ci_low=math.nan,
            ci_high=math.nan,
            p_value=1.0,
            p_value_adjusted=1.0,
            effect_size=0.0,
            wins=0,
            losses=0,
            ties=0,
            underpowered=True,
        )

    ci_low, ci_high = bootstrap_ci(differences.tolist())
    p_value = wilcoxon_p_value(differences.tolist())

    return ComparisonRow(
        category=category,
        metric=metric,
        baseline=baseline,
        challenger=challenger,
        n=n,
        baseline_mean=float(baseline_array.mean()),
        challenger_mean=float(challenger_array.mean()),
        mean_difference=float(differences.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        # Overwritten by compare_to_baseline once the family is known.
        p_value_adjusted=p_value,
        effect_size=cohens_dz(differences.tolist()),
        wins=int((differences > TIE_THRESHOLD).sum()),
        losses=int((differences < -TIE_THRESHOLD).sum()),
        ties=int((np.abs(differences) <= TIE_THRESHOLD).sum()),
        underpowered=n < MIN_DOCUMENTS_FOR_POWER,
    )


def compare_to_baseline(
    category_scores: CategoryScores,
    baseline: str,
    challengers: list[str],
) -> tuple[list[PipelineSummary], list[ComparisonRow]]:
    """Compare every challenger against the baseline within one category.

    P-values are Holm-corrected across the challengers, which is the family of
    tests a reader of this category's table will be looking at.

    :return: (per-pipeline summaries, comparison rows). Pipelines with no
             scores in this category are skipped.
    """
    summaries = [
        summarize_pipeline(pipeline, scores, category_scores.stats_by_pipeline.get(pipeline))
        for pipeline, scores in category_scores.by_pipeline.items()
    ]
    # Best first, with unscored pipelines at the bottom rather than sorted as 0.
    summaries.sort(key=lambda s: (math.isnan(s.mean), -s.mean if not math.isnan(s.mean) else 0.0))

    if baseline not in category_scores.by_pipeline:
        return summaries, []

    rows: list[ComparisonRow] = []
    for challenger in challengers:
        if challenger not in category_scores.by_pipeline:
            continue
        baseline_values, challenger_values, _ = category_scores.paired(baseline, challenger)
        rows.append(
            compare_pair(
                category=category_scores.category,
                metric=category_scores.metric,
                baseline=baseline,
                challenger=challenger,
                baseline_scores=baseline_values,
                challenger_scores=challenger_values,
            )
        )

    for row, adjusted in zip(rows, holm_bonferroni([r.p_value for r in rows]), strict=True):
        row.p_value_adjusted = adjusted

    return summaries, rows
