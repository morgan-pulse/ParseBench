"""Paired statistics behind the customer comparison report."""

from __future__ import annotations

import math

import pytest

from parse_bench.customer.comparison.scores import CategoryScores
from parse_bench.customer.comparison.stats import (
    MIN_DOCUMENTS_FOR_POWER,
    bootstrap_ci,
    cohens_dz,
    compare_pair,
    compare_to_baseline,
    holm_bonferroni,
    summarize_pipeline,
    wilcoxon_p_value,
)


class TestBootstrapCI:
    def test_is_reproducible(self) -> None:
        # An SA re-running the report in front of a customer must get the same
        # interval, so the seed is fixed rather than drawn from the clock.
        values = [0.1, 0.4, 0.55, 0.9, 0.3, 0.7]
        assert bootstrap_ci(values) == bootstrap_ci(values)

    def test_brackets_the_mean(self) -> None:
        values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        low, high = bootstrap_ci(values)
        mean = sum(values) / len(values)
        assert low <= mean <= high

    def test_empty_input_is_undefined_not_zero(self) -> None:
        low, high = bootstrap_ci([])
        assert math.isnan(low) and math.isnan(high)

    def test_single_observation_gives_a_point(self) -> None:
        assert bootstrap_ci([0.42]) == (0.42, 0.42)

    def test_constant_values_give_a_zero_width_interval(self) -> None:
        low, high = bootstrap_ci([0.5] * 20)
        assert low == pytest.approx(0.5)
        assert high == pytest.approx(0.5)


class TestWilcoxon:
    def test_all_ties_are_not_significant(self) -> None:
        assert wilcoxon_p_value([0.0] * 20) == 1.0

    def test_consistent_difference_is_significant(self) -> None:
        assert wilcoxon_p_value([0.1] * 20) < 0.05

    def test_noise_is_not_significant(self) -> None:
        differences = [0.05, -0.04, 0.03, -0.06, 0.02, -0.01, 0.04, -0.05, 0.01, -0.02]
        assert wilcoxon_p_value(differences) > 0.05


class TestEffectSize:
    def test_zero_when_differences_do_not_vary(self) -> None:
        assert cohens_dz([0.2] * 10) == 0.0

    def test_sign_follows_the_difference(self) -> None:
        assert cohens_dz([0.3, 0.2, 0.4, 0.25]) > 0
        assert cohens_dz([-0.3, -0.2, -0.4, -0.25]) < 0

    def test_too_few_points_to_standardize(self) -> None:
        assert cohens_dz([0.5]) == 0.0


class TestHolmBonferroni:
    def test_empty_family(self) -> None:
        assert holm_bonferroni([]) == []

    def test_single_test_is_unchanged(self) -> None:
        assert holm_bonferroni([0.03]) == [0.03]

    def test_smallest_p_gets_the_largest_multiplier(self) -> None:
        adjusted = holm_bonferroni([0.01, 0.04, 0.03])
        assert adjusted[0] == pytest.approx(0.03)  # 0.01 * 3

    def test_adjustment_is_monotonic_in_rank(self) -> None:
        raw = [0.001, 0.02, 0.04]
        adjusted = holm_bonferroni(raw)
        ordered = sorted(zip(raw, adjusted, strict=True))
        values = [a for _, a in ordered]
        assert values == sorted(values)

    def test_never_exceeds_one(self) -> None:
        assert all(a <= 1.0 for a in holm_bonferroni([0.5, 0.6, 0.9]))


class TestComparePair:
    def _row(self, baseline: list[float], challenger: list[float]):
        return compare_pair("table", "grits", "base", "chal", baseline, challenger)

    def test_counts_wins_losses_and_ties(self) -> None:
        row = self._row([0.5, 0.5, 0.5], [0.6, 0.4, 0.5])
        assert (row.wins, row.losses, row.ties) == (1, 1, 1)

    def test_mean_difference_direction(self) -> None:
        row = self._row([0.5] * 12, [0.7] * 12)
        assert row.mean_difference == pytest.approx(0.2)
        assert row.significant
        assert row.direction == "challenger better"

    def test_baseline_winning_is_reported_as_such(self) -> None:
        row = self._row([0.9] * 12, [0.6] * 12)
        assert row.direction == "baseline better"
        assert "base is better" in row.verdict()

    def test_small_sample_is_flagged_underpowered(self) -> None:
        n = MIN_DOCUMENTS_FOR_POWER - 1
        row = self._row([0.2] * n, [0.9] * n)
        assert row.underpowered
        assert not row.significant
        assert "too few" in row.verdict()

    def test_empty_comparison_does_not_crash(self) -> None:
        row = self._row([], [])
        assert row.n == 0
        assert row.underpowered
        assert math.isnan(row.mean_difference)

    def test_mismatched_lengths_are_rejected(self) -> None:
        # Silently truncating would pair the wrong documents together.
        with pytest.raises(ValueError, match="equal-length"):
            self._row([0.1, 0.2], [0.1])

    def test_identical_scores_are_inconclusive(self) -> None:
        row = self._row([0.5] * 15, [0.5] * 15)
        assert row.ties == 15
        assert not row.significant
        assert row.direction == "inconclusive"


class TestSummarize:
    def test_reports_n_mean_and_median(self) -> None:
        summary = summarize_pipeline("p", {"a": 0.2, "b": 0.4, "c": 0.9})
        assert summary.n == 3
        assert summary.mean == pytest.approx(0.5)
        assert summary.median == pytest.approx(0.4)


class TestCompareToBaseline:
    def _scores(self) -> CategoryScores:
        return CategoryScores(
            category="table",
            metric="grits_trm_composite",
            by_pipeline={
                "base": {f"table/doc{i}": 0.5 for i in range(12)},
                "better": {f"table/doc{i}": 0.7 for i in range(12)},
                "worse": {f"table/doc{i}": 0.3 for i in range(12)},
            },
        )

    def test_produces_one_row_per_challenger(self) -> None:
        summaries, rows = compare_to_baseline(self._scores(), "base", ["better", "worse"])
        assert len(summaries) == 3
        assert [r.challenger for r in rows] == ["better", "worse"]

    def test_summaries_are_best_first(self) -> None:
        summaries, _ = compare_to_baseline(self._scores(), "base", ["better", "worse"])
        assert [s.pipeline for s in summaries] == ["better", "base", "worse"]

    def test_p_values_are_corrected_across_the_family(self) -> None:
        _, rows = compare_to_baseline(self._scores(), "base", ["better", "worse"])
        assert all(r.p_value_adjusted >= r.p_value for r in rows)

    def test_missing_challenger_is_skipped_not_zero_scored(self) -> None:
        # A pipeline that failed to run must be absent from the comparison,
        # not silently recorded as scoring zero.
        _, rows = compare_to_baseline(self._scores(), "base", ["better", "never_ran"])
        assert [r.challenger for r in rows] == ["better"]

    def test_missing_baseline_yields_no_comparisons(self) -> None:
        summaries, rows = compare_to_baseline(self._scores(), "absent_baseline", ["better"])
        assert rows == []
        assert len(summaries) == 3

    def test_pairs_only_documents_both_pipelines_scored(self) -> None:
        scores = CategoryScores(
            category="table",
            metric="grits",
            by_pipeline={
                "base": {"table/a": 0.4, "table/b": 0.5, "table/c": 0.6},
                "chal": {"table/a": 0.5, "table/b": 0.6},
            },
        )
        _, rows = compare_to_baseline(scores, "base", ["chal"])
        assert rows[0].n == 2
