"""Report assembly and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from parse_bench.customer.comparison.report import (
    _overall_standings,
    build_report_data,
    render_html,
    render_markdown,
    write_reports,
)
from parse_bench.customer.comparison.scores import CategoryScores
from parse_bench.customer.project import ProjectPaths, new_config


def _scores(**per_pipeline: float) -> dict[str, CategoryScores]:
    return {
        "table": CategoryScores(
            category="table",
            metric="grits_trm_composite",
            by_pipeline={
                pipeline: {f"table/doc{i}": value for i in range(12)} for pipeline, value in per_pipeline.items()
            },
        )
    }


class TestOverallStandings:
    def test_macro_averages_across_categories(self) -> None:
        scores = {
            "table": CategoryScores("table", "m", {"a": {"table/x": 0.8}}),
            "chart": CategoryScores("chart", "m", {"a": {"chart/y": 0.4}}),
        }
        standings = _overall_standings(scores, ["a"])
        assert standings[0]["overall"] == pytest.approx(0.6)
        assert standings[0]["categories_scored"] == 2

    def test_a_large_category_does_not_dominate(self) -> None:
        # 100 text documents must not outvote 2 table documents; each
        # dimension carries equal weight.
        scores = {
            "table": CategoryScores("table", "m", {"a": {f"table/{i}": 0.0 for i in range(2)}}),
            "text_content": CategoryScores("text_content", "m", {"a": {f"text/{i}": 1.0 for i in range(100)}}),
        }
        assert _overall_standings(scores, ["a"])[0]["overall"] == pytest.approx(0.5)

    def test_best_first(self) -> None:
        standings = _overall_standings(_scores(a=0.3, b=0.9), ["a", "b"])
        assert [s["pipeline"] for s in standings] == ["b", "a"]

    def test_pipeline_with_no_scores_is_omitted(self) -> None:
        standings = _overall_standings(_scores(a=0.5), ["a", "never_ran"])
        assert [s["pipeline"] for s in standings] == ["a"]


class TestBuildReportData:
    def test_includes_baseline_and_verdicts(self) -> None:
        config = new_config("Acme", ["base", "chal"], baseline="base")
        data = build_report_data(config, _scores(base=0.5, chal=0.8))

        assert data["baseline"] == "base"
        category = data["categories"][0]
        assert category["category"] == "table"
        assert "chal" in category["verdicts"]
        assert "chal is better" in category["verdicts"]["chal"]

    def test_ground_truth_provenance_is_carried(self) -> None:
        config = new_config("Acme", ["base"])
        data = build_report_data(config, _scores(base=0.5), ground_truth={"documents": 12, "rules": 400})
        assert data["ground_truth"]["documents"] == 12

    def test_no_scores_produces_an_empty_but_valid_report(self) -> None:
        config = new_config("Acme", ["base"])
        data = build_report_data(config, {})
        assert data["categories"] == []
        assert data["overall"] == []


class TestRenderers:
    def _data(self) -> dict:
        config = new_config("Acme Insurance", ["base", "chal"], baseline="base")
        return build_report_data(
            config,
            _scores(base=0.5, chal=0.8),
            ground_truth={"documents": 12, "rules": 400, "verified_pct": 0.0, "model": "google/gemini-3-pro"},
        )

    def test_markdown_states_provenance_and_verification(self) -> None:
        markdown = render_markdown(self._data())
        assert "Acme Insurance" in markdown
        # A prospect must not be able to mistake bootstrapped labels for
        # human-verified ones.
        assert "0% human-verified" in markdown
        assert "google/gemini-3-pro" in markdown

    def test_markdown_includes_methodology(self) -> None:
        assert "Wilcoxon" in render_markdown(self._data())

    def test_html_is_self_contained(self) -> None:
        html = render_html(self._data())
        assert html.startswith("<!DOCTYPE html>")
        # Customer environments are often offline; no external asset may be
        # required for the page to render.
        assert "http://" not in html
        assert "https://" not in html

    def test_html_escapes_customer_supplied_text(self) -> None:
        config = new_config("<script>alert(1)</script>", ["base"])
        html = render_html(build_report_data(config, _scores(base=0.5)))
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_write_reports_emits_all_three_formats(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        written = write_reports(paths, self._data())
        assert set(written) == {"html", "markdown", "json"}
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0


class TestHeaderProvenance:
    """The header must describe the ground truth that is actually on disk.

    Naming the configured model on a report whose labels a human wrote credits
    a model for someone else's work. On a customer-facing artifact that is
    worse than a missing field — it is a false claim a reader can catch.
    """

    def _data(self, **gt) -> dict:
        config = new_config("Acme", ["base", "chal"], baseline="base")
        return build_report_data(config, _scores(base=0.5, chal=0.8), ground_truth=gt)

    def test_customer_supplied_labels_name_no_model(self) -> None:
        data = self._data(documents=26, rules=265, verified_pct=100.0, source="supplied by you", model=None)
        html_out = render_html(data)
        md_out = render_markdown(data)
        assert "supplied by you" in html_out
        assert "gemini" not in html_out.lower()
        assert "gemini" not in md_out.lower()

    def test_bootstrapped_labels_name_the_model(self) -> None:
        data = self._data(documents=12, rules=400, verified_pct=0.0, source="bootstrapped", model="google/gemini-3-pro")
        assert "google/gemini-3-pro" in render_html(data)
        assert "google/gemini-3-pro" in render_markdown(data)

    def test_document_count_is_shown(self) -> None:
        assert ">26<" in render_html(self._data(documents=26, rules=265, source="supplied by you"))


class TestSingleDimensionReport:
    def test_overall_block_is_suppressed_for_one_dimension(self) -> None:
        # With one dimension the macro-average restates the table below it, and
        # a "Dimensions: 1" column invites the reader to ask what the other
        # four were.
        config = new_config("Acme", ["base", "chal"], baseline="base")
        data = build_report_data(config, _scores(base=0.5, chal=0.8))
        assert len(data["categories"]) == 1
        assert "Macro-average across dimensions" not in render_html(data)
        assert "## Overall" not in render_markdown(data)

    def test_overall_block_is_kept_for_several_dimensions(self) -> None:
        config = new_config("Acme", ["base", "chal"], baseline="base")
        scores = _scores(base=0.5, chal=0.8)
        scores["chart"] = CategoryScores(
            category="chart",
            metric="rule_pass_rate",
            by_pipeline={
                "base": {f"chart/doc{i}": 0.4 for i in range(12)},
                "chal": {f"chart/doc{i}": 0.6 for i in range(12)},
            },
        )
        data = build_report_data(config, scores)
        assert "Macro-average across dimensions" in render_html(data)
