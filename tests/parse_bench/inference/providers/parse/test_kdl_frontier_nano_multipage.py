from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse import kdl_frontier_nano as kdl
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="kdl_frontier_nano_test",
        provider_name="kdl_frontier_nano",
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def _provider() -> kdl.KdlFrontierNanoProvider:
    provider = object.__new__(kdl.KdlFrontierNanoProvider)
    provider._dpi = 144
    provider._endpoint_url = "http://provider.invalid/v1"
    provider._model = "test-model"
    provider._max_concurrent = 1
    provider._timeout = 30
    provider._max_pages = 10
    return provider


def _png_bytes(page_number: int) -> bytes:
    with Image.new("RGBA", (8 + page_number, 8), (page_number, 0, 0, 255)) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class _FakePixmap:
    def __init__(self, page_number: int) -> None:
        self._page_number = page_number

    def tobytes(self, output: str) -> bytes:
        assert output == "png"
        return _png_bytes(self._page_number)


class _FakePage:
    def __init__(self, page_number: int, render: Any) -> None:
        self._page_number = page_number
        self._render = render

    def get_pixmap(self, **kwargs: object) -> _FakePixmap:
        self._render(self._page_number)
        return _FakePixmap(self._page_number)


class _FakeDocument:
    def __init__(self, page_count: int, render: Any, events: list[tuple[str, int]]) -> None:
        self.page_count = page_count
        self._render = render
        self._events = events

    def __enter__(self) -> _FakeDocument:
        return self

    def __exit__(self, *args: object) -> None:
        self._events.append(("close_document", 0))

    def __iter__(self):
        return iter(_FakePage(page_number, self._render) for page_number in range(1, self.page_count + 1))


def _track_opened_and_normalized_images(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Image.Image], list[Image.Image]]:
    opened: list[Image.Image] = []
    normalized: list[Image.Image] = []
    real_open = Image.open
    real_normalize = kdl.normalize_image_mode

    def tracked_open(*args: object, **kwargs: object) -> Image.Image:
        image = real_open(*args, **kwargs)
        opened.append(image)
        return image

    def tracked_normalize(image: Image.Image, target_mode: str = "RGB") -> Image.Image:
        result = real_normalize(image, target_mode)
        if result is not image:
            normalized.append(result)
        return result

    monkeypatch.setattr(kdl.Image, "open", tracked_open)
    monkeypatch.setattr(kdl, "normalize_image_mode", tracked_normalize)
    return opened, normalized


def _assert_closed(image: Image.Image) -> None:
    with pytest.raises(ValueError, match="Operation on closed image"):
        image.getpixel((0, 0))


def test_kdl_streams_pdf_pages_in_order_and_closes_owned_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    events: list[tuple[str, int]] = []
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)

    def render(page_number: int) -> None:
        if opened:
            _assert_closed(opened[-1])
            _assert_closed(normalized[-1])
        events.append(("render", page_number))

    monkeypatch.setattr(
        "fitz.open",
        lambda path: _FakeDocument(3, render, events),
    )

    async def parse_page(
        self: object,
        client: object,
        semaphore: object,
        image: Image.Image,
        page_number: int,
    ) -> list[dict[str, object]]:
        assert image.mode == "RGB"
        events.append(("infer", page_number))
        return [
            {
                "category": "Text",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "content": f"page {page_number}",
                "layout_order": 0,
                "page_number": page_number,
            }
        ]

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)

    raw_result = _provider().run_inference(_pipeline(), _request(source))

    assert raw_result.raw_output["markdown"] == "page 1\n\n---\n\n**Page 2**\n\npage 2\n\n---\n\n**Page 3**\n\npage 3"
    assert events == [
        ("render", 1),
        ("infer", 1),
        ("render", 2),
        ("infer", 2),
        ("render", 3),
        ("infer", 3),
        ("close_document", 0),
    ]
    assert len(opened) == len(normalized) == 3
    for image in [*opened, *normalized]:
        _assert_closed(image)


@pytest.mark.parametrize(
    "failure",
    [
        ProviderPermanentError("invalid page"),
        ProviderTransientError("page timed out"),
    ],
    ids=["permanent", "transient"],
)
def test_kdl_page_two_failure_aborts_and_closes_images(
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    events: list[tuple[str, int]] = []
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)
    monkeypatch.setattr(
        "fitz.open",
        lambda path: _FakeDocument(3, lambda page: events.append(("render", page)), events),
    )

    async def parse_page(
        self: object,
        client: object,
        semaphore: object,
        image: Image.Image,
        page_number: int,
    ) -> list[dict[str, object]]:
        events.append(("infer", page_number))
        if page_number == 2:
            raise failure
        return []

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)
    successful_result = None

    with pytest.raises(type(failure), match=str(failure)):
        successful_result = _provider().run_inference(_pipeline(), _request(source))

    assert successful_result is None
    assert events == [
        ("render", 1),
        ("infer", 1),
        ("render", 2),
        ("infer", 2),
        ("close_document", 0),
    ]
    assert len(opened) == len(normalized) == 2
    for image in [*opened, *normalized]:
        _assert_closed(image)


def test_kdl_render_failure_closes_prior_page_and_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    events: list[tuple[str, int]] = []
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)

    def render(page_number: int) -> None:
        events.append(("render", page_number))
        if page_number == 2:
            raise RuntimeError("renderer failed")

    monkeypatch.setattr("fitz.open", lambda path: _FakeDocument(3, render, events))

    async def parse_page(*args: object, **kwargs: object) -> list[dict[str, object]]:
        events.append(("infer", len(opened)))
        return []

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)

    with pytest.raises(ProviderPermanentError, match="Failed to render document page 2: renderer failed"):
        _provider().run_inference(_pipeline(), _request(source))

    assert events == [
        ("render", 1),
        ("infer", 1),
        ("render", 2),
        ("close_document", 0),
    ]
    assert len(opened) == len(normalized) == 1
    _assert_closed(opened[0])
    _assert_closed(normalized[0])


def test_kdl_single_image_path_preserves_one_page_behavior_and_closes_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    with Image.new("RGBA", (8, 8), "white") as image:
        image.save(source)
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)
    page_numbers: list[int] = []
    monkeypatch.setattr("fitz.open", lambda path: pytest.fail("single images must not open as PDFs"))

    async def parse_page(
        self: object,
        client: object,
        semaphore: object,
        image: Image.Image,
        page_number: int,
    ) -> list[dict[str, object]]:
        page_numbers.append(page_number)
        return []

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)

    raw_result = _provider().run_inference(_pipeline(), _request(source))

    assert raw_result.raw_output["markdown"] == ""
    assert page_numbers == [1]
    assert len(opened) == len(normalized) == 1
    _assert_closed(opened[0])
    _assert_closed(normalized[0])
