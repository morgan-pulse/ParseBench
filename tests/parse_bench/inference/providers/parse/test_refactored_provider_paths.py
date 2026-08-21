from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

LEGACY_PROVIDERS = [
    ("amazon_nova", "AmazonNovaProvider", 150),
    ("anthropic", "AnthropicProvider", 144),
    ("dots_ocr", "DotsOcrParseProvider", 144),
    ("google", "GoogleProvider", 144),
    ("openai", "OpenAIProvider", 144),
    ("tesseract", "TesseractProvider", 144),
    ("textract", "TextractProvider", 300),
]


def _pipeline(module_name: str) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=f"{module_name}_test",
        provider_name=module_name,
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


class _TextractClient:
    def __init__(self, failure: Exception | None = None, fail_on_call: int = 1) -> None:
        self.calls = 0
        self.failure = failure
        self.fail_on_call = fail_on_call

    def analyze_document(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        if self.failure is not None and self.calls == self.fail_on_call:
            raise self.failure
        return {
            "Blocks": [
                {
                    "Id": f"line-{self.calls}",
                    "BlockType": "LINE",
                    "Text": f"page {self.calls}",
                }
            ]
        }


def _provider(module_name: str, class_name: str, dpi: int) -> Any:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    provider = object.__new__(getattr(module, class_name))
    defaults = {
        "_dpi": dpi,
        "_mode": "image",
        "_bbox_scale": 1000,
        "_model": "test-model",
        "_max_tokens": 100,
        "_reasoning_effort": None,
        "_thinking_level": None,
        "_temperature": None,
        "_top_p": None,
        "_region": "test-region",
        "_prompt_mode": "parse",
        "_is_layout_mode": False,
        "_timeout": 30,
        "_lang": "eng",
        "_config": "",
        "_output_type": "text",
        "_detect_tables": False,
        "_detect_forms": False,
        "_output_tables_as_html": True,
    }
    for name, value in defaults.items():
        setattr(provider, name, value)
    if module_name in {"amazon_nova", "anthropic", "google", "openai"}:
        provider._get_pricing = lambda: (1.0, 2.0)
    if module_name == "textract":
        provider._textract_client = _TextractClient()

        def convert_to_markdown(response: dict[str, object]) -> dict[str, object]:
            page_count = response.get("DocumentMetadata", {}).get("Pages", 1)
            pages = [{"page_index": page - 1, "markdown": f"page {page}"} for page in range(1, int(page_count) + 1)]
            return {"pages": pages, "markdown": "\n\n".join(page["markdown"] for page in pages)}

        provider._convert_to_markdown = convert_to_markdown
    return provider


def _install_boundary(
    provider: Any,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: Exception | None = None,
    fail_on_call: int = 1,
) -> list[int]:
    calls: list[int] = []

    def next_page() -> int:
        page_number = len(calls) + 1
        calls.append(page_number)
        if failure is not None and page_number == fail_on_call:
            raise failure
        return page_number

    def usage(page_number: int) -> dict[str, int]:
        return {
            "input_tokens": page_number * 10,
            "output_tokens": page_number,
            "thinking_tokens": 0,
            "total_tokens": page_number * 11,
        }

    if module_name == "amazon_nova":
        provider._parse_image_with_layout = lambda image: (
            [{"label": "Text", "text": f"page {(page := next_page())}", "bbox": [0, 0, 8, 8]}],
            f"page {page}",
            usage(page),
            "end_turn",
        )
    elif module_name in {"anthropic", "google", "openai"}:

        def parse_image(image: Image.Image) -> tuple[str, dict[str, int]]:
            page = next_page()
            return f"page {page}", usage(page)

        provider._parse_image = parse_image
    elif module_name == "dots_ocr":
        provider._call_endpoint = lambda image: f"page {next_page()}"
    elif module_name == "tesseract":
        import pytesseract

        monkeypatch.setattr(pytesseract, "image_to_string", lambda *args, **kwargs: f"page {next_page()}")
    elif module_name == "textract":
        provider._textract_client = _TextractClient(failure=failure, fail_on_call=fail_on_call)
    return calls


def _mock_pdf(source: Path, monkeypatch: pytest.MonkeyPatch) -> list[Image.Image]:
    rendered: list[Image.Image] = []
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        image = Image.new("RGB", (8 + first_page, 8), "white")
        rendered.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render)
    return rendered


@pytest.mark.parametrize(("module_name", "class_name", "dpi"), LEGACY_PROVIDERS)
def test_refactored_provider_runs_and_normalizes_ordered_pdf_pages(
    module_name: str,
    class_name: str,
    dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, dpi)
    calls = _install_boundary(provider, module_name, monkeypatch)

    raw_result = provider.run_inference(_pipeline(module_name), _request(source))
    result = provider.normalize(raw_result)

    boundary_calls = provider._textract_client.calls if module_name == "textract" else calls
    assert boundary_calls == 2 if module_name == "textract" else [1, 2]
    assert (
        raw_result.raw_output.get("num_pages") == 2
        or raw_result.raw_output.get("textract_response", {}).get("DocumentMetadata", {}).get("Pages") == 2
    )
    assert isinstance(result.output, ParseOutput)
    assert result.output.markdown == "page 1\n\npage 2"
    assert [(page.page_index, page.markdown) for page in result.output.pages] == [
        (0, "page 1"),
        (1, "page 2"),
    ]
    if module_name in {"amazon_nova", "anthropic", "google", "openai"}:
        assert raw_result.raw_output["input_tokens"] == 30
        assert raw_result.raw_output["output_tokens"] == 3
        assert raw_result.raw_output["cost_per_page_usd"] == pytest.approx(18 / 1_000_000)
    if module_name == "amazon_nova":
        assert [page.page_number for page in result.output.layout_pages] == [1, 2]
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(("module_name", "class_name", "dpi"), LEGACY_PROVIDERS)
def test_refactored_provider_preserves_single_image_behavior(
    module_name: str,
    class_name: str,
    dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(source)
    provider = _provider(module_name, class_name, dpi)
    calls = _install_boundary(provider, module_name, monkeypatch)
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda *args, **kwargs: pytest.fail("single images must not use PDF rasterization"),
    )

    raw_result = provider.run_inference(_pipeline(module_name), _request(source))
    result = provider.normalize(raw_result)

    boundary_calls = provider._textract_client.calls if module_name == "textract" else calls
    assert boundary_calls == 1 if module_name == "textract" else [1]
    assert isinstance(result.output, ParseOutput)
    assert result.output.markdown == "page 1"


@pytest.mark.parametrize(("module_name", "class_name", "dpi"), LEGACY_PROVIDERS)
def test_refactored_provider_propagates_permanent_failure_without_partial_document(
    module_name: str,
    class_name: str,
    dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, dpi)
    calls = _install_boundary(
        provider,
        module_name,
        monkeypatch,
        failure=ProviderPermanentError("provider boundary failed"),
        fail_on_call=2,
    )
    delays: list[int] = []
    monkeypatch.setattr("time.sleep", delays.append)

    with pytest.raises(ProviderPermanentError, match="provider boundary failed"):
        provider.run_inference(_pipeline(module_name), _request(source))

    boundary_calls = provider._textract_client.calls if module_name == "textract" else calls
    assert boundary_calls == 2 if module_name == "textract" else [1, 2]
    assert delays == []
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(
    ("module_name", "class_name", "dpi"),
    [provider for provider in LEGACY_PROVIDERS if provider[0] != "dots_ocr"],
)
def test_refactored_provider_propagates_transient_page_failure_for_document_retry(
    module_name: str,
    class_name: str,
    dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, dpi)
    calls = _install_boundary(
        provider,
        module_name,
        monkeypatch,
        failure=ProviderTransientError("timeout on page two"),
        fail_on_call=2,
    )

    with pytest.raises(ProviderTransientError, match="timeout on page two"):
        provider.run_inference(_pipeline(module_name), _request(source))

    boundary_calls = provider._textract_client.calls if module_name == "textract" else calls
    assert boundary_calls == 2 if module_name == "textract" else [1, 2]
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


def test_dots_ocr_retries_the_document_after_transient_page_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    _mock_pdf(source, monkeypatch)
    provider = _provider("dots_ocr", "DotsOcrParseProvider", 144)
    page_widths: list[int] = []

    def call_endpoint(image: Image.Image) -> str:
        page_widths.append(image.width)
        if len(page_widths) == 2:
            raise ProviderTransientError("retry me")
        return f"page {image.width - 8}"

    provider._call_endpoint = call_endpoint
    delays: list[int] = []
    monkeypatch.setattr("time.sleep", delays.append)

    raw_result = provider.run_inference(_pipeline("dots_ocr"), _request(source))
    result = provider.normalize(raw_result)

    assert page_widths == [9, 10, 9, 10]
    assert delays == [15]
    assert result.output.markdown == "page 1\n\npage 2"


def test_dots_ocr_propagates_transient_failure_after_document_retries_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    _mock_pdf(source, monkeypatch)
    provider = _provider("dots_ocr", "DotsOcrParseProvider", 144)
    page_widths: list[int] = []

    def call_endpoint(image: Image.Image) -> str:
        page_widths.append(image.width)
        raise ProviderTransientError("retry me")

    provider._call_endpoint = call_endpoint
    delays: list[int] = []
    monkeypatch.setattr("time.sleep", delays.append)

    with pytest.raises(ProviderTransientError, match="retry me"):
        provider.run_inference(_pipeline("dots_ocr"), _request(source))

    assert page_widths == [9, 9, 9]
    assert delays == [15, 30]
