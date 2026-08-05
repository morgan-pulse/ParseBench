"""Orchestrate ground-truth generation for a customer project."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parse_bench.customer.groundtruth.client import GroundTruthModelError, VisionModelClient
from parse_bench.customer.groundtruth.derive import derive_rules
from parse_bench.customer.groundtruth.emit import (
    DocumentGroundTruth,
    read_reference,
    write_dataset,
    write_reference,
)
from parse_bench.customer.groundtruth.prompts import (
    CHART_SYSTEM_PROMPT,
    TRANSCRIPTION_SYSTEM_PROMPT,
    chart_user_prompt,
    transcription_user_prompt,
)
from parse_bench.customer.groundtruth.render import RenderError, render_pages, to_data_url
from parse_bench.customer.ingest import IngestedDoc, staged_documents
from parse_bench.customer.project import (
    GROUP_TO_CATEGORIES,
    CustomerProjectConfig,
    ProjectPaths,
)
from parse_bench.test_cases.parse_rule_schemas import coerce_parse_rule_list

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """What a generation run is expected to cost before it runs."""

    documents: int
    pages: int
    cost_per_page_usd: float
    chart_pages: int = 0

    @property
    def model_calls(self) -> int:
        return self.pages + self.chart_pages

    @property
    def estimated_usd(self) -> float:
        return self.model_calls * self.cost_per_page_usd


@dataclass
class GenerationResult:
    """Outcome of a ground-truth generation run."""

    documents: list[DocumentGroundTruth] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    rule_counts: dict[str, int] = field(default_factory=dict)
    dropped_rules: list[tuple[str, str]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_rules(self) -> int:
        return sum(self.rule_counts.values())


def _categories_for_group(group: str, requested: list[str]) -> list[str]:
    """Categories to derive for a document, intersected with project config."""
    group_categories = GROUP_TO_CATEGORIES.get(group, ())
    return [c for c in group_categories if c in requested]


def estimate_cost(
    paths: ProjectPaths,
    config: CustomerProjectConfig,
) -> CostEstimate:
    """Estimate the model cost of generating ground truth for staged documents."""
    docs = staged_documents(paths)
    cap = config.groundtruth.max_pages_per_doc
    pages = 0
    chart_pages = 0
    for doc in docs:
        doc_pages = min(doc.pages or 1, cap)
        pages += doc_pages
        if "chart" in _categories_for_group(doc.group, config.categories):
            chart_pages += doc_pages
    return CostEstimate(
        documents=len(docs),
        pages=pages,
        chart_pages=chart_pages,
        cost_per_page_usd=config.groundtruth.estimated_cost_per_page_usd,
    )


def _validate_rules(
    doc_stem: str,
    category: str,
    rules: list[dict[str, Any]],
    dropped: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Drop rules that fail the evaluator's own schema validation.

    A malformed rule would otherwise blow up the whole evaluation run in front
    of a customer. Better to lose one rule and say so.
    """
    valid: list[dict[str, Any]] = []
    for rule in rules:
        try:
            coerce_parse_rule_list([dict(rule)])
        except Exception as e:
            dropped.append((f"{doc_stem}/{category}/{rule.get('type')}", str(e)[:200]))
            continue
        valid.append(rule)
    return valid


def _transcribe_document(
    client: VisionModelClient,
    doc: IngestedDoc,
    config: CustomerProjectConfig,
    categories: list[str],
) -> DocumentGroundTruth:
    """Render, transcribe, and (when charts are in scope) read chart data."""
    gt_config = config.groundtruth
    images = render_pages(doc.dest, dpi=gt_config.dpi, max_pages=gt_config.max_pages_per_doc)
    if not images:
        raise RenderError(f"No renderable pages in {doc.dest}")

    total = len(images)
    filename = doc.dest.name
    page_markdown: list[str] = []
    charts: list[dict[str, Any]] = []
    notes: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    want_charts = "chart" in categories

    for index, image in enumerate(images, start=1):
        data_url = to_data_url(image)

        response = client.complete(
            TRANSCRIPTION_SYSTEM_PROMPT,
            transcription_user_prompt(filename, index, total),
            images=[data_url],
        )
        prompt_tokens += response.prompt_tokens
        completion_tokens += response.completion_tokens
        payload = response.as_json()

        markdown = str(payload.get("markdown") or "").strip()
        if markdown:
            page_markdown.append(markdown)
        note = str(payload.get("notes") or "").strip()
        if note:
            notes.append(f"page {index}: {note}")

        if want_charts:
            chart_response = client.complete(
                CHART_SYSTEM_PROMPT,
                chart_user_prompt(filename, index, total),
                images=[data_url],
            )
            prompt_tokens += chart_response.prompt_tokens
            completion_tokens += chart_response.completion_tokens
            for chart in chart_response.as_json().get("charts") or []:
                if isinstance(chart, dict) and chart.get("points"):
                    chart.setdefault("page", index)
                    charts.append(chart)

    return DocumentGroundTruth(
        pdf_rel=f"pdfs/{doc.group}/{doc.dest.name}",
        group=doc.group,
        stem=doc.dest.stem,
        markdown="\n\n".join(page_markdown),
        charts=charts,
        notes=notes,
        pages=total,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def generate_ground_truth(
    paths: ProjectPaths,
    config: CustomerProjectConfig,
    *,
    client: VisionModelClient | None = None,
    force: bool = False,
    max_concurrent: int = 4,
    progress: bool = True,
) -> GenerationResult:
    """Generate ground truth for every staged document.

    Documents with an existing reference are reused unless *force* is set, so an
    interrupted run resumes instead of paying for the same pages twice.

    :param client: Override the model client (used by tests).
    :param force: Regenerate references that already exist.
    :param max_concurrent: Documents transcribed in parallel.
    """
    result = GenerationResult()
    docs = staged_documents(paths)
    if not docs:
        return result

    if client is None:
        gt = config.groundtruth
        client = VisionModelClient(model=gt.model, base_url=gt.base_url, api_key_env=gt.api_key_env)

    pending: list[IngestedDoc] = []
    for doc in docs:
        categories = _categories_for_group(doc.group, config.categories)
        if not categories:
            continue
        cached = None if force else read_reference(paths, doc.group, doc.dest.stem)
        if cached is not None:
            result.documents.append(
                DocumentGroundTruth(
                    pdf_rel=f"pdfs/{doc.group}/{doc.dest.name}",
                    group=doc.group,
                    stem=doc.dest.stem,
                    markdown=cached,
                    pages=doc.pages or 1,
                )
            )
            if progress:
                print(f"  reusing reference: {doc.group}/{doc.dest.name}")
            continue
        pending.append(doc)

    if pending:
        client.require_key()

    with ThreadPoolExecutor(max_workers=max(1, max_concurrent)) as pool:
        futures = {
            pool.submit(
                _transcribe_document,
                client,
                doc,
                config,
                _categories_for_group(doc.group, config.categories),
            ): doc
            for doc in pending
        }
        for future in as_completed(futures):
            doc = futures[future]
            try:
                generated = future.result()
            except (GroundTruthModelError, RenderError) as e:
                result.failures.append((str(doc.dest), str(e)))
                if progress:
                    print(f"  FAILED {doc.group}/{doc.dest.name}: {e}")
                continue
            except Exception as e:  # noqa: BLE001 - one bad document must not kill the run
                result.failures.append((str(doc.dest), f"{type(e).__name__}: {e}"))
                if progress:
                    print(f"  FAILED {doc.group}/{doc.dest.name}: {e}")
                continue

            result.documents.append(generated)
            result.prompt_tokens += generated.prompt_tokens
            result.completion_tokens += generated.completion_tokens
            if progress:
                print(f"  transcribed {generated.group}/{generated.stem} ({generated.pages} page(s))")

    # Derive rules from every reference, cached or fresh, so a resumed run
    # produces the same dataset as an uninterrupted one.
    for generated in result.documents:
        categories = _categories_for_group(generated.group, config.categories)
        derived = derive_rules(
            doc_stem=generated.stem,
            markdown=generated.markdown,
            charts=generated.charts,
            categories=categories,
        )
        for category, rules in derived.by_category.items():
            generated.rules_by_category[category] = _validate_rules(
                generated.stem, category, rules, result.dropped_rules
            )
        write_reference(paths, generated)

    result.rule_counts = write_dataset(paths, result.documents)
    return result


def reference_paths(paths: ProjectPaths) -> list[Path]:
    """Every reference transcription currently on disk."""
    root = paths.data_dir / "_groundtruth"
    if not root.exists():
        return []
    return sorted(root.rglob("*.md"))
