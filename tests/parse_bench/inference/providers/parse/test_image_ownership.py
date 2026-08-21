from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse.amazon_nova import AmazonNovaProvider
from parse_bench.inference.providers.parse.anthropic import AnthropicProvider
from parse_bench.inference.providers.parse.google import GoogleProvider
from parse_bench.inference.providers.parse.openai import OpenAIProvider
from parse_bench.inference.providers.parse.tesseract import TesseractProvider
from parse_bench.inference.providers.parse.textract import TextractProvider

ENCODERS = [
    (AmazonNovaProvider, "_image_to_jpeg_bytes"),
    (AnthropicProvider, "_image_to_base64"),
    (GoogleProvider, "_image_to_bytes"),
    (OpenAIProvider, "_image_to_base64"),
]


@pytest.mark.parametrize(("provider_class", "method_name"), ENCODERS)
@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_vision_encoders_close_derived_images_but_not_caller_image(
    provider_class: type[Any],
    method_name: str,
    fail: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = object.__new__(provider_class)
    provider.MAX_IMAGE_DIMENSION = 4
    provider.MAX_IMAGE_SIZE_BYTES = 1024 * 1024
    original = Image.new("RGBA", (8, 8), "white")
    derived: list[Image.Image] = []
    real_resize = Image.Image.resize
    real_convert = Image.Image.convert
    real_save = Image.Image.save
    inside_pillow_operation = False

    def track(image: Image.Image) -> Image.Image:
        image.close = Mock(wraps=image.close)
        derived.append(image)
        return image

    def resize(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        nonlocal inside_pillow_operation
        if inside_pillow_operation:
            return real_resize(image, *args, **kwargs)
        inside_pillow_operation = True
        try:
            resized = real_resize(image, *args, **kwargs)
        finally:
            inside_pillow_operation = False
        return track(resized)

    def convert(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        nonlocal inside_pillow_operation
        if inside_pillow_operation:
            return real_convert(image, *args, **kwargs)
        inside_pillow_operation = True
        try:
            converted = real_convert(image, *args, **kwargs)
        finally:
            inside_pillow_operation = False
        return track(converted)

    def save(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        if fail:
            raise RuntimeError("encoding failed")
        real_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    monkeypatch.setattr(Image.Image, "convert", convert)
    monkeypatch.setattr(Image.Image, "save", save)

    if fail:
        with pytest.raises(RuntimeError, match="encoding failed"):
            getattr(provider, method_name)(original)
    else:
        assert getattr(provider, method_name)(original)

    assert len(derived) == 2
    assert all(isinstance(image.close, Mock) and image.close.call_count == 1 for image in derived)
    assert original.getpixel((0, 0)) == (255, 255, 255, 255)
    original.close()


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_textract_closes_all_resizes_but_not_caller_image(fail: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object.__new__(TextractProvider)
    provider._MAX_DIMENSION = 4
    provider._TARGET_BYTES = 0
    original = Image.new("RGB", (8, 8), "white")
    derived: list[Image.Image] = []
    real_resize = Image.Image.resize
    real_save = Image.Image.save

    def resize(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        resized = real_resize(image, *args, **kwargs)
        resized.close = Mock(wraps=resized.close)
        derived.append(resized)
        return resized

    save_calls = 0

    def save(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        nonlocal save_calls
        save_calls += 1
        if fail and save_calls == 2:
            raise RuntimeError("encoding failed")
        real_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    monkeypatch.setattr(Image.Image, "save", save)

    if fail:
        with pytest.raises(RuntimeError, match="encoding failed"):
            provider._resize_image_for_textract(original)
    else:
        assert provider._resize_image_for_textract(original)

    assert derived
    assert all(isinstance(image.close, Mock) and image.close.call_count == 1 for image in derived)
    assert original.getpixel((0, 0)) == (255, 255, 255)
    original.close()


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_tesseract_single_image_is_closed_on_success_and_failure(
    tmp_path: Path, fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (8, 6), "white").save(source)
    provider = object.__new__(TesseractProvider)
    provider._output_type = "text"
    provider._lang = "eng"
    provider._config = ""
    real_open = Image.open
    opened: list[Image.Image] = []

    class TrackedImageContext:
        def __init__(self, path: str | Path) -> None:
            self.image = real_open(path)
            self.image.close = Mock(wraps=self.image.close)
            opened.append(self.image)

        def __enter__(self) -> Image.Image:
            return self.image

        def __exit__(self, *args: object) -> None:
            self.image.close()

    def image_to_string(*args: Any, **kwargs: Any) -> str:
        if fail:
            raise RuntimeError("ocr failed")
        return "page text"

    import pytesseract

    monkeypatch.setattr(Image, "open", TrackedImageContext)
    monkeypatch.setattr(pytesseract, "image_to_string", image_to_string)
    monkeypatch.setattr(pytesseract, "Output", SimpleNamespace(DICT="dict"))

    if fail:
        with pytest.raises(ProviderPermanentError, match="Error during OCR: ocr failed"):
            provider._ocr_image(str(source))
    else:
        result = provider._ocr_image(str(source))
        assert result["pages"][0]["text"] == "page text"
        assert result["pages"][0]["width"] == 8

    assert len(opened) == 1
    assert isinstance(opened[0].close, Mock)
    assert opened[0].close.call_count == 1
