"""Page-wise execution adapter for image-backed parse providers.

Many vision providers accept exactly one raster image per request.  This module
lets those providers keep their existing single-image implementation while
giving PDF inputs document semantics: every page is rendered, submitted in
order, and normalized into one :class:`ParseOutput`.

The adapter is deliberately opt-in.  Providers that natively accept PDFs or
already implement page-wise inference should not use it.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from PIL import Image

from parse_bench.inference.providers.base import ProviderConfigError, ProviderPermanentError
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

_MULTIPAGE_KEY = "_parse_bench_multipage"


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
                page_result = run_single_image(pipeline, page_request)
                page_results.append(
                    {
                        "page_index": page_index,
                        "raw_output": page_result.raw_output,
                        "latency_in_ms": page_result.latency_in_ms,
                    }
                )

    completed_at = datetime.now()
    return RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=request.product_type,
        raw_output={
            _MULTIPAGE_KEY: {
                "version": 1,
                "num_pages": len(page_results),
                "pages": page_results,
            }
        },
        started_at=started_at,
        completed_at=completed_at,
        latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
    )


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
