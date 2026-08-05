"""Write derived ground truth in the ParseBench dataset format.

Produces the same layout the public dataset uses, so the customer's evaluation
runs through the stock loader with no special-casing::

    data/
      {category}.jsonl        # one row per rule
      expected_markdown.json  # {pdf_rel: reference markdown}
      pdfs/{group}/*.pdf
      _groundtruth/{group}/{stem}.md    # reference, for human review
      _groundtruth/{group}/{stem}.json  # charts, notes, token usage
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parse_bench.customer.project import ProjectPaths

GROUNDTRUTH_SUBDIR = "_groundtruth"
EXPECTED_MARKDOWN_FILENAME = "expected_markdown.json"

# Marks every bootstrapped rule. Reports read this to state plainly how much of
# the ground truth a human has actually confirmed.
BOOTSTRAP_TAG = "bootstrap"


@dataclass
class DocumentGroundTruth:
    """Everything generated for a single document."""

    pdf_rel: str
    group: str
    stem: str
    markdown: str
    rules_by_category: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    charts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pages: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def write_reference(paths: ProjectPaths, doc: DocumentGroundTruth) -> Path:
    """Save the reference transcription and metadata for human review."""
    ref_dir = paths.data_dir / GROUNDTRUTH_SUBDIR / doc.group
    ref_dir.mkdir(parents=True, exist_ok=True)

    md_path = ref_dir / f"{doc.stem}.md"
    md_path.write_text(doc.markdown, encoding="utf-8")

    meta_path = ref_dir / f"{doc.stem}.json"
    meta_path.write_text(
        json.dumps(
            {
                "pdf": doc.pdf_rel,
                "group": doc.group,
                "pages": doc.pages,
                "charts": doc.charts,
                "notes": doc.notes,
                "rule_counts": {k: len(v) for k, v in doc.rules_by_category.items()},
                "prompt_tokens": doc.prompt_tokens,
                "completion_tokens": doc.completion_tokens,
                "verified": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return md_path


def read_reference(paths: ProjectPaths, group: str, stem: str) -> str | None:
    """Load a previously generated reference, if one exists."""
    md_path = paths.data_dir / GROUNDTRUTH_SUBDIR / group / f"{stem}.md"
    if not md_path.exists():
        return None
    return md_path.read_text(encoding="utf-8")


def _rule_row(doc: DocumentGroundTruth, category: str, rule: dict[str, Any]) -> dict[str, Any]:
    """Build one JSONL row from a derived rule."""
    payload = {k: v for k, v in rule.items() if k not in ("type", "id", "page")}
    row: dict[str, Any] = {
        "pdf": doc.pdf_rel,
        "category": category,
        "id": rule.get("id"),
        "type": rule.get("type"),
        "verified": False,
        "tags": [BOOTSTRAP_TAG],
        "rule": payload,
    }
    if rule.get("page") is not None:
        row["page"] = rule["page"]
    return row


def _pointer_row(doc: DocumentGroundTruth, category: str) -> dict[str, Any]:
    """A rule-free row so a category exists for documents scored from markdown."""
    return {
        "pdf": doc.pdf_rel,
        "category": category,
        "id": f"{doc.stem}::expected_markdown",
        "type": "expected_markdown",
        "verified": False,
        "tags": [BOOTSTRAP_TAG],
        "rule": {},
    }


def write_dataset(paths: ProjectPaths, docs: list[DocumentGroundTruth]) -> dict[str, int]:
    """Write all JSONL files and the expected-markdown map.

    Rewrites each category file from scratch so re-running generation cannot
    leave stale rules for documents that were removed.

    :return: Rule-row count per category.
    """
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    expected_markdown: dict[str, str] = {}

    for doc in docs:
        if doc.markdown.strip():
            expected_markdown[doc.pdf_rel] = doc.markdown
        for category, rules in doc.rules_by_category.items():
            bucket = rows_by_category.setdefault(category, [])
            if rules:
                bucket.extend(_rule_row(doc, category, rule) for rule in rules)
            elif doc.markdown.strip():
                # Categories scored straight from the reference (e.g. tables)
                # still need a row so the loader builds a test case.
                bucket.append(_pointer_row(doc, category))

    counts: dict[str, int] = {}
    for category, rows in sorted(rows_by_category.items()):
        path = paths.category_jsonl(category)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[category] = len(rows)

    if expected_markdown:
        (paths.data_dir / EXPECTED_MARKDOWN_FILENAME).write_text(
            json.dumps(expected_markdown, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return counts


def dataset_summary(paths: ProjectPaths) -> dict[str, dict[str, int]]:
    """Count rules and documents per category from the JSONL files on disk."""
    summary: dict[str, dict[str, int]] = {}
    if not paths.data_dir.exists():
        return summary
    for jsonl_path in sorted(paths.data_dir.glob("*.jsonl")):
        rules = 0
        documents: set[str] = set()
        verified = 0
        bootstrapped = 0
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rules += 1
                documents.add(str(row.get("pdf", "")))
                if row.get("verified"):
                    verified += 1
                if BOOTSTRAP_TAG in (row.get("tags") or []):
                    bootstrapped += 1
        summary[jsonl_path.stem] = {
            "rules": rules,
            "documents": len(documents),
            "verified": verified,
            "bootstrapped": bootstrapped,
        }

    summary.update(_sidecar_summary(paths))
    return summary


def _sidecar_summary(paths: ProjectPaths) -> dict[str, dict[str, int]]:
    """Count ground truth supplied as sidecar ``.test.json`` files.

    Customer-supplied labels often arrive in this layout rather than JSONL.
    Reporting them as absent would tell the customer to regenerate ground
    truth they already have — and pay for it.
    """
    summary: dict[str, dict[str, int]] = {}
    for test_path in sorted(paths.data_dir.rglob("*.test.json")):
        # Sidecar group is the containing directory, matching the loader.
        category = test_path.parent.name
        try:
            config = json.loads(test_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rules = config.get("test_rules") or []
        if not isinstance(rules, list):
            continue
        entry = summary.setdefault(category, {"rules": 0, "documents": 0, "verified": 0, "bootstrapped": 0})
        entry["rules"] += len(rules)
        entry["documents"] += 1
        # Sidecar datasets are hand-built, so their rules count as verified
        # unless a rule explicitly says otherwise.
        entry["verified"] += sum(1 for r in rules if isinstance(r, dict) and r.get("verified", True))
    return summary
