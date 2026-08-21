from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from parse_bench.inference.providers.parse._multipage_image import (
    normalize_pdf_pages,
    run_pdf_pages,
)
from parse_bench.schemas.parse_output import ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="test_image_provider",
        provider_name="test",
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def test_pdf_pages_are_processed_and_combined_in_document_order(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered_pages = [Image.new("RGB", (20 + page_index, 30), (page_index, 0, 0)) for page_index in (1, 2, 3)]
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi: rendered_pages,
    )

    observed_pages: list[tuple[str, int]] = []

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        image_path = Path(request.source_file_path)
        with Image.open(image_path) as image:
            page_number = image.getpixel((0, 0))[0]
        observed_pages.append((image_path.name, page_number))
        now = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"markdown": f"page {page_number}"},
            started_at=now,
            completed_at=now,
            latency_in_ms=page_number,
        )

    raw_result = run_pdf_pages(
        _pipeline(),
        _request(source),
        dpi=144,
        run_single_image=run_single_image,
    )

    assert raw_result is not None
    assert observed_pages == [
        ("page-000001.png", 1),
        ("page-000002.png", 2),
        ("page-000003.png", 3),
    ]
    # Raw artifacts must remain checkpoint-safe after temporary images disappear.
    json.dumps(raw_result.model_dump(mode="json"))

    def normalize_single_image(raw: RawInferenceResult) -> InferenceResult:
        markdown = raw.raw_output["markdown"]
        output = ParseOutput(
            example_id=raw.request.example_id,
            pipeline_name=raw.pipeline_name,
            markdown=markdown,
            layout_pages=[ParseLayoutPageIR(page_number=1, md=markdown)],
        )
        return InferenceResult(
            request=raw.request,
            pipeline_name=raw.pipeline_name,
            product_type=raw.product_type,
            raw_output=raw.raw_output,
            output=output,
            started_at=raw.started_at,
            completed_at=raw.completed_at,
            latency_in_ms=raw.latency_in_ms,
        )

    result = normalize_pdf_pages(raw_result, normalize_single_image=normalize_single_image)

    assert result is not None
    assert result.output.markdown == "page 1\n\npage 2\n\npage 3"
    assert [(page.page_index, page.markdown) for page in result.output.pages] == [
        (0, "page 1"),
        (1, "page 2"),
        (2, "page 3"),
    ]
    assert [page.page_number for page in result.output.layout_pages] == [1, 2, 3]


def test_single_image_input_stays_on_the_provider_path(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(source)
    called = False

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        nonlocal called
        called = True
        raise AssertionError("the adapter must not intercept a single image")

    result = run_pdf_pages(
        _pipeline(),
        _request(source),
        dpi=144,
        run_single_image=run_single_image,
    )

    assert result is None
    assert called is False
