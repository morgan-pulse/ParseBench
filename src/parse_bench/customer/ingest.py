"""Ingest customer documents into a ParseBench dataset layout.

Customers drop files into ``docs/`` — optionally sorted into ``docs/table/``,
``docs/chart/``, ``docs/text/`` — and this module mirrors them into
``data/pdfs/<group>/`` where the loader expects them. Documents are copied, not
moved, so the customer's originals stay untouched.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from parse_bench.customer.project import (
    DEFAULT_DOC_GROUP,
    DOC_GROUPS,
    ProjectPaths,
)
from parse_bench.test_cases.loader import SUPPORTED_EXTENSIONS

MANIFEST_FILENAME = "_ingest_manifest.json"


@dataclass
class IngestedDoc:
    """One document staged into the dataset."""

    source: Path
    group: str
    dest: Path
    pages: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "group": self.group,
            "dest_rel": f"pdfs/{self.group}/{self.dest.name}",
            "pages": self.pages,
        }


@dataclass
class IngestResult:
    """Outcome of an ingest pass."""

    docs: list[IngestedDoc] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    truncated: list[tuple[Path, int, int]] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return sum(d.pages or 1 for d in self.docs)

    def by_group(self) -> dict[str, list[IngestedDoc]]:
        grouped: dict[str, list[IngestedDoc]] = {}
        for doc in self.docs:
            grouped.setdefault(doc.group, []).append(doc)
        return grouped


def count_pages(path: Path) -> int | None:
    """Page count for a document, or None when it can't be determined.

    Images count as one page. PDFs need PyMuPDF; without it, page counts are
    unknown and cost estimates fall back to per-document counts.
    """
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    try:
        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception:
        return None


def _resolve_group(path: Path, docs_dir: Path) -> str:
    """Infer a document's group from its subdirectory under docs/."""
    try:
        relative = path.relative_to(docs_dir)
    except ValueError:
        return DEFAULT_DOC_GROUP
    parts = relative.parts
    if len(parts) > 1 and parts[0] in DOC_GROUPS:
        return parts[0]
    return DEFAULT_DOC_GROUP


def discover_documents(docs_dir: Path) -> list[Path]:
    """Find every supported document under docs/, recursively."""
    if not docs_dir.exists():
        return []
    found = [
        p
        for p in sorted(docs_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
    ]
    return found


def _stage_document(source: Path, dest: Path, max_pages: int | None) -> tuple[Path, int | None]:
    """Copy a document into the dataset, truncating it if it exceeds *max_pages*.

    Truncation matters for scoring, not just cost: ground truth is generated
    for the first N pages, so a parser handed the full document would be
    penalised for faithfully transcribing pages the ground truth never
    described. Both sides must cover the same pages.

    :return: (staged path, original page count).
    """
    pages = count_pages(source)
    if max_pages is None or pages is None or pages <= max_pages or source.suffix.lower() != ".pdf":
        shutil.copy2(source, dest)
        return dest, pages

    from parse_bench.customer.groundtruth.render import truncate_pdf

    truncated_dest = dest.with_name(f"{dest.stem}__pages1-{max_pages}{dest.suffix}")
    if truncate_pdf(source, truncated_dest, max_pages):
        return truncated_dest, pages

    # No PyMuPDF: stage the whole document rather than silently dropping it.
    shutil.copy2(source, dest)
    return dest, pages


def ingest(
    paths: ProjectPaths,
    *,
    group_override: str | None = None,
    force: bool = False,
    max_pages: int | None = None,
) -> IngestResult:
    """Stage documents from docs/ into data/pdfs/<group>/.

    :param paths: Project paths.
    :param group_override: Force every document into this group.
    :param force: Re-copy documents already staged.
    :param max_pages: Truncate documents longer than this many pages.
    """
    result = IngestResult()
    documents = discover_documents(paths.docs_dir)
    if not documents:
        return result

    seen_names: dict[str, Path] = {}
    for source in documents:
        group = group_override or _resolve_group(source, paths.docs_dir)
        dest_dir = paths.group_pdfs_dir(group)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name

        # Two source files with the same basename would collide into one test
        # id and silently drop one of the customer's documents.
        key = f"{group}/{source.name}"
        if key in seen_names:
            result.skipped.append((source, f"duplicate filename, already staged from {seen_names[key]}"))
            continue
        seen_names[key] = source

        # A previous run may have staged this as a truncated copy.
        existing = next(
            (p for p in (dest, *dest.parent.glob(f"{dest.stem}__pages1-*{dest.suffix}")) if p.exists()),
            None,
        )
        if existing is not None and not force:
            result.docs.append(IngestedDoc(source=source, group=group, dest=existing, pages=count_pages(existing)))
            continue

        staged, original_pages = _stage_document(source, dest, max_pages)
        if staged != dest and original_pages is not None:
            result.truncated.append((source, original_pages, max_pages or original_pages))
        result.docs.append(IngestedDoc(source=source, group=group, dest=staged, pages=count_pages(staged)))

    write_manifest(paths, result)
    return result


def write_manifest(paths: ProjectPaths, result: IngestResult) -> Path:
    """Record what was ingested, for `customer status` and the report header."""
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.data_dir / MANIFEST_FILENAME
    payload = {
        "documents": [d.to_dict() for d in result.docs],
        "total_documents": len(result.docs),
        "total_pages": result.total_pages,
        "skipped": [{"source": str(p), "reason": r} for p, r in result.skipped],
        "truncated": [
            {"source": str(p), "original_pages": original, "kept_pages": kept} for p, original, kept in result.truncated
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def read_manifest(paths: ProjectPaths) -> dict[str, object] | None:
    """Load the ingest manifest, or None if nothing has been ingested."""
    manifest_path = paths.data_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def staged_documents(paths: ProjectPaths) -> list[IngestedDoc]:
    """List documents currently staged under data/pdfs/, straight from disk."""
    docs: list[IngestedDoc] = []
    if not paths.pdfs_dir.exists():
        return docs
    for group_dir in sorted(paths.pdfs_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        for f in sorted(group_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.append(
                    IngestedDoc(source=f, group=group_dir.name, dest=f, pages=count_pages(f)),
                )
    return docs
