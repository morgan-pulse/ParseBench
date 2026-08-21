from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderTransientError
from parse_bench.inference.providers.parse.google import GoogleProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="google_test",
        provider_name="google",
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def _response(text: str | None) -> SimpleNamespace:
    candidates = []
    if text is not None:
        candidates = [
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                finish_reason=None,
            )
        ]
    return SimpleNamespace(
        candidates=candidates,
        prompt_feedback=SimpleNamespace(block_reason=None),
        usage_metadata=None,
    )


class _Models:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return next(self._responses)


def _provider(mode: str, responses: list[SimpleNamespace]) -> tuple[GoogleProvider, _Models]:
    provider = object.__new__(GoogleProvider)
    models = _Models(responses)
    provider._client = SimpleNamespace(models=models)
    provider._types = SimpleNamespace(
        Part=SimpleNamespace(
            from_bytes=lambda **kwargs: kwargs,
            from_text=lambda **kwargs: kwargs,
        ),
        GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Content=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    provider._model = "gemini-3-flash"
    provider._dpi = 144
    provider._max_tokens = 1024
    provider._thinking_level = None
    provider._mode = mode
    provider._bbox_scale = 1000
    provider._layout_system_prompt = "layout system prompt"
    provider._layout_user_prompt = "layout user prompt"
    return provider, models


@pytest.mark.parametrize(
    ("mode", "page_one", "message"),
    [
        ("image", "page one", "returned no text after 2 attempts"),
        (
            "parse_with_layout",
            '<div data-bbox="[0,0,1000,1000]" data-label="Text">page one</div>',
            "returned no layout text after 2 attempts",
        ),
    ],
)
def test_google_page_two_empty_responses_abort_without_partial_payload(
    mode: str,
    page_one: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered: list[Image.Image] = []
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert first_page == last_page
        image = Image.new("RGB", (8 + first_page, 8), "white")
        rendered.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)
    provider, models = _provider(mode, [_response(page_one), _response(None), _response(None)])
    successful_result = None

    with pytest.raises(ProviderTransientError, match=message):
        successful_result = provider.run_inference(_pipeline(), _request(source))

    assert successful_result is None
    assert models.calls == 3
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))
