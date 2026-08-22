"""Page-wise execution adapter for image-backed parse providers.

Many vision providers accept exactly one raster image per request.  This module
lets those providers keep their existing single-image implementation while
giving PDF inputs document semantics: every page is rendered, submitted in
order, and normalized into one :class:`ParseOutput`.

The adapter is deliberately opt-in.  Providers that natively accept PDFs or
already implement page-wise inference should not use it.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from PIL import Image

from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderRetryExhaustedError,
    ProviderTransientError,
)
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

_MULTIPAGE_KEY = "_parse_bench_multipage"
_PAGE_MAX_ATTEMPTS = 3
_PAGE_INITIAL_BACKOFF_S = 2.0


def run_page_with_retries[PageResultT](
    call: Callable[[], PageResultT],
    *,
    provider_name: str,
    page_number: int,
) -> PageResultT:
    """Run one billable page with the sole retry budget for that page."""

    for attempt in range(_PAGE_MAX_ATTEMPTS):
        try:
            return call()
        except (ProviderTransientError, ProviderRateLimitError) as exc:
            if attempt == _PAGE_MAX_ATTEMPTS - 1:
                raise ProviderRetryExhaustedError(
                    f"{provider_name} page {page_number} failed after {_PAGE_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc
            time.sleep(_PAGE_INITIAL_BACKOFF_S * (2**attempt))

    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ImageBackedPdfProviderSpec:
    """Registered parse provider whose PDF path rasterizes pages locally."""

    provider_name: str
    module_name: str
    class_name: str
    dpi: int
    execution: str


@dataclass(frozen=True)
class ParseProviderPdfClassification:
    """Explicit PDF execution classification for one registered parse provider."""

    provider_name: str
    module_name: str
    class_name: str
    pdf_handling: str
    dpi: int | None = None
    execution: str | None = None


# Authoritative inventory for every registered parse provider. A provider is
# either locally page-rasterized (and must declare its bounded execution path)
# or explicitly classified as not locally page-rasterized. Coverage tests compare
# this inventory with both the authoritative parse-module manifest and the runtime
# registry, independent of decorator source syntax.
PARSE_PROVIDER_PDF_CLASSIFICATIONS = (
    ParseProviderPdfClassification(
        "amazon_nova", "amazon_nova", "AmazonNovaProvider", "local-page-raster", 150, "direct"
    ),
    ParseProviderPdfClassification("anthropic", "anthropic", "AnthropicProvider", "local-page-raster", 144, "direct"),
    ParseProviderPdfClassification(
        "azure_document_intelligence",
        "azure_document_intelligence",
        "AzureDocumentIntelligenceProvider",
        "no-local-page-raster",
    ),
    ParseProviderPdfClassification("chandra2", "chandra2", "Chandra2Provider", "local-page-raster", 144, "adapter"),
    ParseProviderPdfClassification("chunkr", "chunkr", "ChunkrProvider", "no-local-page-raster"),
    ParseProviderPdfClassification(
        "databricks_ai_parse", "databricks_ai_parse", "DatabricksAiParseProvider", "no-local-page-raster"
    ),
    ParseProviderPdfClassification("datalab", "datalab", "DatalabProvider", "no-local-page-raster"),
    ParseProviderPdfClassification(
        "deepseekocr2", "deepseekocr2", "DeepSeekOCR2Provider", "local-page-raster", 144, "adapter"
    ),
    ParseProviderPdfClassification("docling_parse", "docling", "DoclingParseProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("docling_serve", "docling_serve", "DoclingServeProvider", "no-local-page-raster"),
    ParseProviderPdfClassification(
        "dots_ocr_parse", "dots_ocr", "DotsOcrParseProvider", "local-page-raster", 144, "direct"
    ),
    ParseProviderPdfClassification("extend_parse", "extend_parse", "ExtendParseProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("falconocr", "falconocr", "FalconOcrProvider", "local-page-raster", 144, "adapter"),
    ParseProviderPdfClassification("gemma4", "gemma4", "Gemma4Provider", "local-page-raster", 144, "adapter"),
    ParseProviderPdfClassification("google", "google", "GoogleProvider", "local-page-raster", 144, "direct"),
    ParseProviderPdfClassification("google_docai", "google_docai", "GoogleDocAIProvider", "no-local-page-raster"),
    ParseProviderPdfClassification(
        "granite_vision", "granite_vision", "GraniteVisionProvider", "local-page-raster", 144, "adapter"
    ),
    ParseProviderPdfClassification(
        "infinity_parser2", "infinity_parser2", "InfinityParser2Provider", "local-page-raster", 300, "adapter"
    ),
    ParseProviderPdfClassification(
        "kdl_frontier_nano", "kdl_frontier_nano", "KdlFrontierNanoProvider", "local-page-raster", 144, "kdl"
    ),
    ParseProviderPdfClassification("landingai", "landingai", "LandingAIParseProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("liteparse", "liteparse", "LiteParseProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("llamaparse", "llamaparse", "LlamaParseProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("markitdown", "markitdown", "MarkItDownProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("mineru25", "mineru25", "MinerU25Provider", "local-page-raster", 144, "adapter"),
    ParseProviderPdfClassification(
        "mineru2605pro", "mineru2605pro", "MinerU2605ProProvider", "local-page-raster", 144, "adapter"
    ),
    ParseProviderPdfClassification(
        "mineru_diffusion", "mineru_diffusion", "MinerUDiffusionProvider", "local-page-raster", 144, "adapter"
    ),
    ParseProviderPdfClassification("mistral_ocr", "mistral_ocr", "MistralOCRProvider", "no-local-page-raster"),
    ParseProviderPdfClassification(
        "nemotron_omni", "nemotron_omni", "NemotronOmniProvider", "local-page-raster", 144, "adapter"
    ),
    ParseProviderPdfClassification("oi_parser", "oi_parser", "OIParserProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("openai", "openai", "OpenAIProvider", "local-page-raster", 144, "direct"),
    ParseProviderPdfClassification(
        "opendataloader", "opendataloader", "OpenDataLoaderProvider", "no-local-page-raster"
    ),
    ParseProviderPdfClassification("paddleocr", "paddleocr", "PaddleOCRProvider", "local-page-raster", 144, "adapter"),
    ParseProviderPdfClassification("pdf_inspector", "pdf_inspector", "PdfInspectorProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("pulse", "pulse", "PulseProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("pymupdf", "pymupdf", "PyMuPDFProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("pymupdf4llm", "pymupdf4llm", "PyMuPDF4LLMProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("pypdf", "pypdf", "PyPDFProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("qwen3_5", "qwen3_5", "Qwen35Provider", "local-page-raster", 144, "adapter"),
    ParseProviderPdfClassification("reducto", "reducto", "ReductoProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("surya2", "surya2", "Surya2Provider", "local-page-raster", 144, "adapter"),
    ParseProviderPdfClassification("tesseract", "tesseract", "TesseractProvider", "local-page-raster", 144, "direct"),
    ParseProviderPdfClassification("textract", "textract", "TextractProvider", "local-page-raster", 300, "direct"),
    ParseProviderPdfClassification(
        "unlimitedocr", "unlimitedocr", "UnlimitedOCRProvider", "local-page-raster", 144, "adapter"
    ),
    ParseProviderPdfClassification("unstructured", "unstructured", "UnstructuredProvider", "no-local-page-raster"),
    ParseProviderPdfClassification("warp_ingest", "warp_ingest", "WarpIngestProvider", "no-local-page-raster"),
)


def _image_backed_provider_specs() -> tuple[ImageBackedPdfProviderSpec, ...]:
    specs: list[ImageBackedPdfProviderSpec] = []
    for classification in PARSE_PROVIDER_PDF_CLASSIFICATIONS:
        if classification.pdf_handling != "local-page-raster":
            continue
        if classification.dpi is None or classification.execution is None:
            raise ValueError(f"Incomplete local raster classification for {classification.provider_name}")
        specs.append(
            ImageBackedPdfProviderSpec(
                classification.provider_name,
                classification.module_name,
                classification.class_name,
                classification.dpi,
                classification.execution,
            )
        )
    return tuple(specs)


IMAGE_BACKED_PDF_PROVIDERS = _image_backed_provider_specs()

# Public operational-stat fields consumed by evaluation/stats.py. Totals are
# additive across requests; per-page values are arithmetic means.
_ADDITIVE_STAT_FIELDS = (
    "credits_used",
    "cost_usd",
    "input_cost_usd",
    "tool_use_prompt_cost_usd",
    "cached_input_cost_usd",
    "output_and_thinking_cost_usd",
    "cache_storage_cost_usd",
    "input_tokens",
    "tool_use_prompt_tokens",
    "cached_content_tokens",
    "output_tokens",
    "total_tokens",
    "thinking_tokens",
    "num_api_calls",
)
_PER_PAGE_STAT_FIELDS = (
    "cost_per_page_usd",
    "input_tokens_per_page",
    "tool_use_prompt_tokens_per_page",
    "cached_content_tokens_per_page",
    "output_tokens_per_page",
)


@contextmanager
def close_derived_images(
    original: Image.Image,
) -> Iterator[Callable[[Image.Image], Image.Image]]:
    """Track and close PIL images derived from a caller-owned image.

    Providers may resize or convert a page several times before encoding it.
    Every tracked derivative is closed on success and failure, while the
    original page remains owned by the caller.
    """

    with ExitStack() as stack:
        tracked: set[int] = set()

        def track(image: Image.Image) -> Image.Image:
            image_id = id(image)
            if image is not original and image_id not in tracked:
                stack.callback(image.close)
                tracked.add(image_id)
            return image

        yield track


class PageImages:
    """One-shot, bounded-memory collection of document page images."""

    def __init__(self, page_count: int, images: Iterator[Image.Image]) -> None:
        self._page_count = page_count
        self._images = images

    def __len__(self) -> int:
        return self._page_count

    def __iter__(self) -> Iterator[Image.Image]:
        return self._images


@contextmanager
def open_document_page_images(source_path: str | Path, *, dpi: int) -> Iterator[PageImages]:
    """Open an image document or incrementally rasterize a PDF.

    PDF pages are inspected up front but rendered one at a time.  The current
    image is closed before the next page is rendered and also when inference
    exits early with an exception.
    """

    path = Path(source_path)
    if path.suffix.lower() != ".pdf":
        with Image.open(path) as image:
            yield PageImages(1, iter((image,)))
        return

    page_count = _pdf_page_count(path)
    images = _iter_pdf_page_images(path, dpi=dpi, page_count=page_count)
    try:
        yield PageImages(page_count, images)
    finally:
        images.close()


def _pdf_page_count(source_path: Path) -> int:
    try:
        from pdf2image import pdfinfo_from_path
    except ImportError as exc:
        raise ProviderConfigError("pdf2image is required to process PDF inputs") from exc

    try:
        page_count = pdfinfo_from_path(str(source_path)).get("Pages")
    except Exception as exc:
        raise ProviderPermanentError(f"Failed to inspect PDF: {exc}") from exc

    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count < 1:
        raise ProviderPermanentError(f"No pages found in PDF: {source_path}")
    return page_count


def _iter_pdf_page_images(source_path: Path, *, dpi: int, page_count: int) -> Generator[Image.Image]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ProviderConfigError("pdf2image is required to process PDF inputs") from exc

    for page_number in range(1, page_count + 1):
        try:
            rendered = convert_from_path(
                str(source_path),
                dpi=dpi,
                first_page=page_number,
                last_page=page_number,
            )
        except Exception as exc:
            raise ProviderPermanentError(f"Failed to render PDF page {page_number}: {exc}") from exc

        if len(rendered) != 1:
            for image in rendered:
                image.close()
            raise ProviderPermanentError(f"Expected one image for PDF page {page_number}, got {len(rendered)}")

        image = rendered[0]
        try:
            yield image
        finally:
            image.close()


def run_pdf_pages(
    pipeline: PipelineSpec,
    request: InferenceRequest,
    *,
    dpi: int,
    run_single_image: Callable[[PipelineSpec, InferenceRequest], RawInferenceResult],
) -> RawInferenceResult | None:
    """Run a single-image provider once per PDF page.

    ``None`` means the source is not a PDF and the provider should continue down
    its normal single-image path.  Page results remain JSON-serializable so raw
    benchmark artifacts can be checkpointed and normalized again later.
    """

    source_path = Path(request.source_file_path)
    if request.product_type != ProductType.PARSE or source_path.suffix.lower() != ".pdf":
        return None

    started_at = datetime.now()
    page_results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="parse-bench-pages-") as temp_dir:
        with open_document_page_images(source_path, dpi=dpi) as images:
            for page_index, image in enumerate(images):
                page_path = Path(temp_dir) / f"page-{page_index + 1:06d}.png"
                _save_png(image, page_path)
                page_request = request.model_copy(update={"source_file_path": str(page_path)})
                page_result = run_page_with_retries(
                    partial(run_single_image, pipeline, page_request),
                    provider_name=pipeline.provider_name,
                    page_number=page_index + 1,
                )
                page_results.append(
                    {
                        "page_index": page_index,
                        "raw_output": page_result.raw_output,
                        "latency_in_ms": page_result.latency_in_ms,
                    }
                )

    completed_at = datetime.now()
    aggregate = _aggregate_page_raw_outputs(page_results)
    return RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=request.product_type,
        raw_output={
            **aggregate,
            _MULTIPAGE_KEY: {
                "version": 1,
                "num_pages": len(page_results),
                "pages": page_results,
            },
        },
        started_at=started_at,
        completed_at=completed_at,
        latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
    )


def _aggregate_page_raw_outputs(page_results: list[dict[str, object]]) -> dict[str, int | float]:
    """Project complete page-level operational metadata to the top level.

    A field is omitted when any page is missing it, uses a non-numeric value,
    or contains NaN/infinity. This avoids presenting partial totals as accurate
    document totals while preserving heterogeneous provider payloads verbatim
    inside the multipage envelope.
    """

    aggregate: dict[str, int | float] = {"num_pages": len(page_results)}
    raw_outputs = [record.get("raw_output") for record in page_results]

    for field in _ADDITIVE_STAT_FIELDS:
        values = _complete_numeric_values(raw_outputs, field)
        if values is not None:
            aggregate[field] = sum(values)

    for field in _PER_PAGE_STAT_FIELDS:
        values = _complete_numeric_values(raw_outputs, field)
        if values is not None:
            aggregate[field] = sum(values) / len(values)

    return aggregate


def _complete_numeric_values(raw_outputs: list[object], field: str) -> list[int | float] | None:
    values: list[int | float] = []
    for raw_output in raw_outputs:
        if not isinstance(raw_output, dict):
            return None
        value = raw_output.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return None
        values.append(value)
    return values or None


def normalize_pdf_pages(
    raw_result: RawInferenceResult,
    *,
    normalize_single_image: Callable[[RawInferenceResult], InferenceResult],
) -> InferenceResult | None:
    """Normalize and combine a result produced by :func:`run_pdf_pages`."""

    envelope = raw_result.raw_output.get(_MULTIPAGE_KEY)
    if not isinstance(envelope, dict):
        return None

    if envelope.get("version") != 1:
        raise ProviderPermanentError("Invalid multipage raw output: unsupported version")

    num_pages = envelope.get("num_pages")
    if not isinstance(num_pages, int) or isinstance(num_pages, bool) or num_pages < 1:
        raise ProviderPermanentError("Invalid multipage raw output: 'num_pages' must be a positive integer")

    page_records = envelope.get("pages")
    if not isinstance(page_records, list):
        raise ProviderPermanentError("Invalid multipage raw output: 'pages' must be a list")
    if len(page_records) != num_pages:
        raise ProviderPermanentError("Invalid multipage raw output: 'num_pages' does not match 'pages'")

    pages: list[PageIR] = []
    layout_pages: list[ParseLayoutPageIR] = []
    for expected_index, record in enumerate(page_records):
        if not isinstance(record, dict) or not isinstance(record.get("raw_output"), dict):
            raise ProviderPermanentError(f"Invalid multipage raw output for page {expected_index + 1}")

        page_index = record.get("page_index")
        if not isinstance(page_index, int) or isinstance(page_index, bool) or page_index != expected_index:
            raise ProviderPermanentError("Invalid multipage raw output: pages must be contiguous and in document order")

        single_raw = raw_result.model_copy(update={"raw_output": record["raw_output"]})
        single_result = normalize_single_image(single_raw)
        single_output = single_result.output
        if not isinstance(single_output, ParseOutput):
            raise ProviderPermanentError("Multipage image adapter only supports parse outputs")

        pages.append(PageIR(page_index=expected_index, markdown=single_output.markdown))
        layout_pages.extend(
            page.model_copy(update={"page_number": expected_index + 1}) for page in single_output.layout_pages
        )

    markdown = "\n\n".join(page.markdown for page in pages)
    output = ParseOutput(
        example_id=raw_result.request.example_id,
        pipeline_name=raw_result.pipeline_name,
        pages=pages,
        layout_pages=layout_pages,
        markdown=markdown,
    )
    return InferenceResult(
        request=raw_result.request,
        pipeline_name=raw_result.pipeline_name,
        product_type=raw_result.product_type,
        raw_output=raw_result.raw_output,
        output=output,
        started_at=raw_result.started_at,
        completed_at=raw_result.completed_at,
        latency_in_ms=raw_result.latency_in_ms,
    )


def _save_png(image: Image.Image, destination: Path) -> None:
    """Persist a rendered page without leaking an open PDF image handle."""

    if image.mode in ("RGB", "RGBA"):
        image.save(destination, format="PNG")
        return

    with image.convert("RGB") as converted:
        converted.save(destination, format="PNG")
