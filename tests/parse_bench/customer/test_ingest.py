"""Staging customer documents into the dataset layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from parse_bench.customer.ingest import (
    discover_documents,
    ingest,
    read_manifest,
    staged_documents,
)
from parse_bench.customer.project import ProjectPaths

PDF_BYTES = b"%PDF-1.4\n%stub\n"


def _project(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths(tmp_path)
    paths.ensure_dirs()
    return paths


def _drop(paths: ProjectPaths, relative: str) -> Path:
    path = paths.docs_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PDF_BYTES)
    return path


class TestDiscovery:
    def test_finds_documents_recursively(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "a.pdf")
        _drop(paths, "table/b.pdf")
        assert len(discover_documents(paths.docs_dir)) == 2

    def test_ignores_unsupported_and_hidden_files(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "keep.pdf")
        (paths.docs_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
        (paths.docs_dir / ".DS_Store").write_bytes(b"")
        assert [p.name for p in discover_documents(paths.docs_dir)] == ["keep.pdf"]

    def test_missing_docs_dir_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert discover_documents(tmp_path / "nope") == []


class TestGrouping:
    def test_subdirectory_sets_the_group(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "table/invoice.pdf")
        _drop(paths, "chart/deck.pdf")
        result = ingest(paths)
        assert {d.group for d in result.docs} == {"table", "chart"}

    def test_loose_files_default_to_text(self, tmp_path: Path) -> None:
        # Every document has text; not every document has tables. Text is the
        # safe default for an unsorted drop.
        paths = _project(tmp_path)
        _drop(paths, "loose.pdf")
        assert ingest(paths).docs[0].group == "text"

    def test_unknown_subdirectory_falls_back_to_text(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "quarterly/report.pdf")
        assert ingest(paths).docs[0].group == "text"

    def test_group_override_wins(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "text/invoice.pdf")
        assert ingest(paths, group_override="table").docs[0].group == "table"


class TestStaging:
    def test_copies_into_the_dataset_layout(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        source = _drop(paths, "table/invoice.pdf")
        ingest(paths)

        staged = paths.group_pdfs_dir("table") / "invoice.pdf"
        assert staged.exists()
        # Originals must be left untouched.
        assert source.exists()

    def test_duplicate_basenames_are_reported_not_overwritten(self, tmp_path: Path) -> None:
        # Two files with the same name would collapse into one test id and one
        # of the customer's documents would vanish without a word.
        paths = _project(tmp_path)
        _drop(paths, "text/report.pdf")
        _drop(paths, "text/2025/report.pdf")

        result = ingest(paths)
        assert len(result.docs) == 1
        assert len(result.skipped) == 1
        assert "duplicate filename" in result.skipped[0][1]

    def test_rerunning_is_idempotent(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "text/a.pdf")
        first = ingest(paths)
        second = ingest(paths)
        assert len(first.docs) == len(second.docs) == 1


class TestManifest:
    def test_records_counts_and_skips(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "text/a.pdf")
        ingest(paths)

        manifest = read_manifest(paths)
        assert manifest is not None
        assert manifest["total_documents"] == 1
        assert manifest["documents"][0]["dest_rel"] == "pdfs/text/a.pdf"

    def test_absent_manifest_reads_as_none(self, tmp_path: Path) -> None:
        assert read_manifest(_project(tmp_path)) is None


class TestStagedDocuments:
    def test_reads_state_from_disk(self, tmp_path: Path) -> None:
        paths = _project(tmp_path)
        _drop(paths, "chart/deck.pdf")
        ingest(paths)

        docs = staged_documents(paths)
        assert [(d.group, d.dest.name) for d in docs] == [("chart", "deck.pdf")]

    def test_empty_project(self, tmp_path: Path) -> None:
        assert staged_documents(_project(tmp_path)) == []


class TestTruncation:
    @pytest.fixture
    def long_pdf(self, tmp_path: Path) -> Path:
        fitz = pytest.importorskip("fitz", reason="PyMuPDF needed to build a multi-page PDF")
        path = tmp_path / "long.pdf"
        doc = fitz.open()
        for i in range(6):
            doc.new_page().insert_text((72, 72), f"page {i}")
        doc.save(path)
        doc.close()
        return path

    def test_documents_are_capped_to_the_ground_truth_scope(self, tmp_path: Path, long_pdf: Path) -> None:
        # Ground truth only covers the first N pages, so the evaluated document
        # must cover the same pages or parsers get punished for pages the
        # ground truth never described.
        paths = _project(tmp_path / "project")
        target = paths.docs_dir / "text" / "long.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(long_pdf.read_bytes())

        result = ingest(paths, max_pages=3)
        assert len(result.truncated) == 1
        assert result.docs[0].pages == 3
        assert "__pages1-3" in result.docs[0].dest.name

    def test_short_documents_are_left_alone(self, tmp_path: Path, long_pdf: Path) -> None:
        paths = _project(tmp_path / "project")
        target = paths.docs_dir / "text" / "long.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(long_pdf.read_bytes())

        result = ingest(paths, max_pages=20)
        assert result.truncated == []
        assert result.docs[0].dest.name == "long.pdf"
