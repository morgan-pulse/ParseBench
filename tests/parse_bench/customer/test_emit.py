"""Emitted ground truth must load through the stock ParseBench loader.

This is the integration seam that matters most: if the generated dataset
doesn't load with no special-casing, the customer workflow has quietly forked
the benchmark and the numbers stop being comparable to anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from parse_bench.customer.groundtruth.derive import derive_rules
from parse_bench.customer.groundtruth.emit import (
    BOOTSTRAP_TAG,
    DocumentGroundTruth,
    dataset_summary,
    read_reference,
    write_dataset,
    write_reference,
)
from parse_bench.customer.project import ProjectPaths
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.loader import load_test_cases

REFERENCE = """\
# Claim Form

The claimant is **Jane Doe** of 14 Bridge Street.

The claim reference is CLM-2026-0041 and was filed on 3 March 2026.

<table><tr><td>Item</td><td>Amount</td></tr><tr><td>Repairs</td><td>820</td></tr></table>
"""


def _project(tmp_path: Path, groups: dict[str, list[str]]) -> ProjectPaths:
    """Build a project with empty placeholder PDFs for the named documents."""
    paths = ProjectPaths(tmp_path)
    paths.ensure_dirs()
    for group, stems in groups.items():
        group_dir = paths.group_pdfs_dir(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            (group_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
    return paths


def _document(group: str, stem: str, categories: list[str]) -> DocumentGroundTruth:
    derived = derive_rules(stem, REFERENCE, [], categories)
    return DocumentGroundTruth(
        pdf_rel=f"pdfs/{group}/{stem}.pdf",
        group=group,
        stem=stem,
        markdown=REFERENCE,
        rules_by_category=derived.by_category,
        pages=1,
    )


class TestWriteDataset:
    def test_writes_one_jsonl_per_category(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01"]})
        doc = _document("text", "claim_01", ["text_content", "text_formatting"])
        counts = write_dataset(paths, [doc])

        assert set(counts) == {"text_content", "text_formatting"}
        assert paths.category_jsonl("text_content").exists()
        assert paths.category_jsonl("text_formatting").exists()

    def test_rows_carry_provenance(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01"]})
        write_dataset(paths, [_document("text", "claim_01", ["text_content"])])

        rows = [
            json.loads(line)
            for line in paths.category_jsonl("text_content").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows
        for row in rows:
            # Bootstrapped ground truth must never claim to be verified.
            assert row["verified"] is False
            assert BOOTSTRAP_TAG in row["tags"]
            assert row["pdf"] == "pdfs/text/claim_01.pdf"

    def test_expected_markdown_map_is_written(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01"]})
        write_dataset(paths, [_document("text", "claim_01", ["text_content"])])

        mapping = json.loads((paths.data_dir / "expected_markdown.json").read_text(encoding="utf-8"))
        assert mapping["pdfs/text/claim_01.pdf"] == REFERENCE

    def test_table_category_emits_a_pointer_row(self, tmp_path: Path) -> None:
        # Tables are scored from the reference markdown, but the category still
        # needs a row or the loader builds no test case for it at all.
        paths = _project(tmp_path, {"table": ["invoice_01"]})
        write_dataset(paths, [_document("table", "invoice_01", ["table"])])

        rows = [
            json.loads(line)
            for line in paths.category_jsonl("table").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["type"] == "expected_markdown"

    def test_rewrites_rather_than_appends(self, tmp_path: Path) -> None:
        # Re-running generation after removing a document must not leave the
        # removed document's rules behind.
        paths = _project(tmp_path, {"text": ["a", "b"]})
        write_dataset(paths, [_document("text", "a", ["text_content"]), _document("text", "b", ["text_content"])])
        first = dataset_summary(paths)["text_content"]["documents"]

        write_dataset(paths, [_document("text", "a", ["text_content"])])
        second = dataset_summary(paths)["text_content"]["documents"]

        assert first == 2
        assert second == 1

    def test_document_with_no_reference_is_omitted(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["blank"]})
        blank = DocumentGroundTruth(pdf_rel="pdfs/text/blank.pdf", group="text", stem="blank", markdown="   ")
        counts = write_dataset(paths, [blank])
        assert counts == {}


class TestLoaderRoundTrip:
    def test_generated_dataset_loads_as_parse_test_cases(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01", "claim_02"]})
        write_dataset(
            paths,
            [
                _document("text", "claim_01", ["text_content", "text_formatting"]),
                _document("text", "claim_02", ["text_content", "text_formatting"]),
            ],
        )

        cases = load_test_cases(paths.data_dir, product_type=ProductType.PARSE.value)
        assert cases

        groups = {case.group for case in cases}
        assert groups == {"text_content", "text_formatting"}

        for case in cases:
            assert case.test_rules, f"{case.test_id} loaded with no rules"
            assert case.expected_markdown == REFERENCE
            # Shared inference group: both categories parse the same PDF once.
            assert case.test_id.startswith("text/")

    def test_table_case_loads_with_expected_markdown_only(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"table": ["invoice_01"]})
        write_dataset(paths, [_document("table", "invoice_01", ["table"])])

        cases = load_test_cases(paths.data_dir, product_type=ProductType.PARSE.value)
        assert len(cases) == 1
        assert cases[0].group == "table"
        assert "<table>" in (cases[0].expected_markdown or "")

    def test_file_paths_resolve_to_staged_pdfs(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01"]})
        write_dataset(paths, [_document("text", "claim_01", ["text_content"])])

        cases = load_test_cases(paths.data_dir, product_type=ProductType.PARSE.value)
        assert cases[0].file_path.exists()


class TestReference:
    def test_reference_roundtrips(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01"]})
        doc = _document("text", "claim_01", ["text_content"])
        write_reference(paths, doc)

        assert read_reference(paths, "text", "claim_01") == REFERENCE
        assert read_reference(paths, "text", "missing") is None

    def test_metadata_records_rule_counts(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01"]})
        doc = _document("text", "claim_01", ["text_content"])
        write_reference(paths, doc)

        meta = json.loads((paths.data_dir / "_groundtruth" / "text" / "claim_01.json").read_text(encoding="utf-8"))
        assert meta["verified"] is False
        assert meta["rule_counts"]["text_content"] > 0


class TestDatasetSummary:
    def test_empty_project(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        paths.ensure_dirs()
        assert dataset_summary(paths) == {}

    def test_counts_rules_documents_and_verification(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["a", "b"]})
        write_dataset(paths, [_document("text", "a", ["text_content"]), _document("text", "b", ["text_content"])])

        summary = dataset_summary(paths)["text_content"]
        assert summary["documents"] == 2
        assert summary["rules"] > 0
        assert summary["verified"] == 0


class TestSidecarGroundTruth:
    """Customer-supplied labels often arrive as sidecar .test.json files.

    The stock loader reads that layout, so the customer workflow must
    recognise it too — otherwise it tells a customer to regenerate (and pay
    for) ground truth they already have.
    """

    def _sidecar(self, paths: ProjectPaths, group: str, stem: str, rules: int) -> None:
        group_dir = paths.data_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
        (group_dir / f"{stem}.test.json").write_text(
            json.dumps({"test_rules": [{"type": "present", "text": f"line {i}"} for i in range(rules)]}),
            encoding="utf-8",
        )

    def test_summary_counts_sidecar_rules(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        paths.ensure_dirs()
        self._sidecar(paths, "energy", "report_a", 5)
        self._sidecar(paths, "energy", "report_b", 3)

        summary = dataset_summary(paths)["energy"]
        assert summary["documents"] == 2
        assert summary["rules"] == 8
        # Hand-built labels count as verified; the report must not describe a
        # customer's own ground truth as 0% verified.
        assert summary["verified"] == 8

    def test_run_accepts_sidecar_ground_truth(self, tmp_path: Path) -> None:
        from parse_bench.customer.cli import has_ground_truth

        paths = ProjectPaths(tmp_path)
        paths.ensure_dirs()
        assert not has_ground_truth(paths)

        self._sidecar(paths, "energy", "report_a", 2)
        assert has_ground_truth(paths)

    def test_run_accepts_generated_jsonl(self, tmp_path: Path) -> None:
        from parse_bench.customer.cli import has_ground_truth

        paths = _project(tmp_path, {"text": ["claim_01"]})
        assert not has_ground_truth(paths)
        write_dataset(paths, [_document("text", "claim_01", ["text_content"])])
        assert has_ground_truth(paths)

    def test_sidecar_dataset_loads_through_the_stock_loader(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        paths.ensure_dirs()
        self._sidecar(paths, "energy", "report_a", 4)

        cases = load_test_cases(paths.data_dir, product_type=ProductType.PARSE.value)
        assert [c.group for c in cases] == ["energy"]
        assert len(cases[0].test_rules) == 4


class TestBootstrapProvenanceCounts:
    def test_generated_rules_are_counted_as_bootstrapped(self, tmp_path: Path) -> None:
        paths = _project(tmp_path, {"text": ["claim_01"]})
        write_dataset(paths, [_document("text", "claim_01", ["text_content"])])
        summary = dataset_summary(paths)["text_content"]
        assert summary["bootstrapped"] == summary["rules"]

    def test_sidecar_rules_are_not_counted_as_bootstrapped(self, tmp_path: Path) -> None:
        # Hand-built labels must never be attributed to the ground-truth model.
        paths = ProjectPaths(tmp_path)
        paths.ensure_dirs()
        group_dir = paths.data_dir / "energy"
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "a.pdf").write_bytes(b"%PDF-1.4\n")
        (group_dir / "a.test.json").write_text(
            json.dumps({"test_rules": [{"type": "present", "text": "x"}]}), encoding="utf-8"
        )
        assert dataset_summary(paths)["energy"]["bootstrapped"] == 0
