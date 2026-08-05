"""Loading per-document scores out of evaluation output."""

from __future__ import annotations

import json
from pathlib import Path

from parse_bench.customer.comparison.scores import load_scores
from parse_bench.customer.project import ProjectPaths


def _write_report(
    paths: ProjectPaths,
    pipeline: str,
    category: str,
    scores: dict[str, dict[str, float]],
    aggregate_stats: dict | None = None,
) -> None:
    """Write a minimal but schema-valid _evaluation_report.json."""
    results = [
        {
            "test_id": test_id,
            "example_id": test_id,
            "pipeline_name": pipeline,
            "product_type": "parse",
            "success": True,
            "metrics": [{"metric_name": name, "value": value} for name, value in metrics.items()],
        }
        for test_id, metrics in scores.items()
    ]
    report_dir = paths.pipeline_output_dir(pipeline) / category
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "_evaluation_report.json").write_text(
        json.dumps(
            {
                "total_examples": len(results),
                "successful": len(results),
                "failed": 0,
                "skipped": 0,
                "aggregate_metrics": {},
                "per_example_results": results,
                "aggregate_stats": aggregate_stats or {},
            }
        ),
        encoding="utf-8",
    )


class TestLoadScores:
    def test_loads_per_document_scores(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(paths, "a", "table", {"table/x": {"grits_trm_composite": 0.8}})
        _write_report(paths, "b", "table", {"table/x": {"grits_trm_composite": 0.6}})

        scores = load_scores(paths, ["a", "b"])
        assert scores["table"].metric == "grits_trm_composite"
        assert scores["table"].by_pipeline["a"]["table/x"] == 0.8
        assert scores["table"].by_pipeline["b"]["table/x"] == 0.6

    def test_prefers_the_categorys_default_metric(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(
            paths,
            "a",
            "text_content",
            {"text/x": {"content_faithfulness": 0.9, "rule_pass_rate": 0.5}},
        )
        assert load_scores(paths, ["a"])["text_content"].metric == "content_faithfulness"

    def test_falls_back_when_the_default_is_absent(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(paths, "a", "table", {"table/x": {"rule_pass_rate": 0.5}})
        assert load_scores(paths, ["a"])["table"].metric == "rule_pass_rate"

    def test_chooses_a_metric_all_pipelines_report(self, tmp_path: Path) -> None:
        # Comparing pipelines on different metrics would be meaningless, so a
        # shared metric beats a preferred one only one pipeline emits.
        paths = ProjectPaths(tmp_path)
        _write_report(paths, "a", "table", {"table/x": {"grits_trm_composite": 0.8, "rule_pass_rate": 0.7}})
        _write_report(paths, "b", "table", {"table/x": {"rule_pass_rate": 0.6}})

        scores = load_scores(paths, ["a", "b"])
        assert scores["table"].metric == "rule_pass_rate"
        assert set(scores["table"].by_pipeline) == {"a", "b"}

    def test_failed_examples_are_excluded(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        report_dir = paths.pipeline_output_dir("a") / "table"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "_evaluation_report.json").write_text(
            json.dumps(
                {
                    "total_examples": 2,
                    "successful": 1,
                    "failed": 1,
                    "skipped": 0,
                    "aggregate_metrics": {},
                    "per_example_results": [
                        {
                            "test_id": "table/ok",
                            "example_id": "table/ok",
                            "pipeline_name": "a",
                            "product_type": "parse",
                            "success": True,
                            "metrics": [{"metric_name": "rule_pass_rate", "value": 0.9}],
                        },
                        {
                            "test_id": "table/broken",
                            "example_id": "table/broken",
                            "pipeline_name": "a",
                            "product_type": "parse",
                            "success": False,
                            "metrics": [{"metric_name": "rule_pass_rate", "value": 0.0}],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        # A crashed evaluation is missing data, not a zero — scoring it as zero
        # would silently punish whichever pipeline hit an infrastructure error.
        assert set(load_scores(paths, ["a"])["table"].by_pipeline["a"]) == {"table/ok"}

    def test_missing_pipeline_directory_is_skipped(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(paths, "a", "table", {"table/x": {"rule_pass_rate": 0.5}})
        scores = load_scores(paths, ["a", "never_ran"])
        assert set(scores["table"].by_pipeline) == {"a"}

    def test_category_filter(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(paths, "a", "table", {"table/x": {"rule_pass_rate": 0.5}})
        _write_report(paths, "a", "chart", {"chart/y": {"rule_pass_rate": 0.4}})
        assert set(load_scores(paths, ["a"], categories=["chart"])) == {"chart"}

    def test_corrupt_report_does_not_break_the_run(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(paths, "a", "table", {"table/x": {"rule_pass_rate": 0.5}})
        bad_dir = paths.pipeline_output_dir("b") / "table"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "_evaluation_report.json").write_text("{ truncated", encoding="utf-8")

        scores = load_scores(paths, ["a", "b"])
        assert set(scores["table"].by_pipeline) == {"a"}

    def test_operational_stats_are_carried_through(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(
            paths,
            "a",
            "table",
            {"table/x": {"rule_pass_rate": 0.5}},
            aggregate_stats={"latency_ms": {"avg": 1200.0, "total": 2400.0, "unit": "ms"}},
        )
        stats = load_scores(paths, ["a"])["table"].stats_by_pipeline["a"]
        assert stats["latency_ms_avg"] == 1200.0

    def test_no_output_yields_nothing(self, tmp_path: Path) -> None:
        assert load_scores(ProjectPaths(tmp_path), ["a"]) == {}


class TestPairing:
    def test_pairs_only_common_documents(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(paths, "a", "table", {"table/x": {"rule_pass_rate": 0.5}, "table/y": {"rule_pass_rate": 0.7}})
        _write_report(paths, "b", "table", {"table/x": {"rule_pass_rate": 0.6}})

        category = load_scores(paths, ["a", "b"])["table"]
        left, right, test_ids = category.paired("a", "b")
        assert test_ids == ["table/x"]
        assert left == [0.5]
        assert right == [0.6]

    def test_pairs_are_aligned_by_test_id(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        _write_report(
            paths,
            "a",
            "table",
            {"table/b": {"rule_pass_rate": 0.2}, "table/a": {"rule_pass_rate": 0.8}},
        )
        _write_report(
            paths,
            "b",
            "table",
            {"table/a": {"rule_pass_rate": 0.9}, "table/b": {"rule_pass_rate": 0.1}},
        )

        category = load_scores(paths, ["a", "b"])["table"]
        left, right, test_ids = category.paired("a", "b")
        # Same document must land at the same index on both sides, whatever
        # order the reports happened to list them in.
        assert test_ids == ["table/a", "table/b"]
        assert left == [0.8, 0.2]
        assert right == [0.9, 0.1]
