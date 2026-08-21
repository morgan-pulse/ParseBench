from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

ADAPTER_PROVIDERS = [
    ("chandra2", "Chandra2Provider", 144),
    ("deepseekocr2", "DeepSeekOCR2Provider", 144),
    ("falconocr", "FalconOcrProvider", 144),
    ("gemma4", "Gemma4Provider", 144),
    ("granite_vision", "GraniteVisionProvider", 144),
    ("infinity_parser2", "InfinityParser2Provider", 300),
    ("mineru25", "MinerU25Provider", 144),
    ("mineru2605pro", "MinerU2605ProProvider", 144),
    ("mineru_diffusion", "MinerUDiffusionProvider", 144),
    ("nemotron_omni", "NemotronOmniProvider", 144),
    ("paddleocr", "PaddleOCRProvider", 144),
    ("qwen3_5", "Qwen35Provider", 144),
    ("surya2", "Surya2Provider", 144),
    ("unlimitedocr", "UnlimitedOCRProvider", 144),
]


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
def test_provider_run_inference_uses_multipage_adapter(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    provider_class = getattr(module, class_name)
    provider = object.__new__(provider_class)
    if expected_dpi != 300 or module_name != "infinity_parser2":
        provider._dpi = expected_dpi

    sentinel = object()
    pipeline = object()
    request = object()

    def fake_run_pdf_pages(
        actual_pipeline: object,
        actual_request: object,
        *,
        dpi: int,
        run_single_image: Any,
    ) -> object:
        assert actual_pipeline is pipeline
        assert actual_request is request
        assert dpi == expected_dpi
        assert run_single_image.__self__ is provider
        assert run_single_image.__func__ is provider_class.run_inference
        return sentinel

    monkeypatch.setattr(module, "run_pdf_pages", fake_run_pdf_pages)

    assert provider.run_inference(pipeline, request) is sentinel


@pytest.mark.parametrize(("module_name", "class_name", "_expected_dpi"), ADAPTER_PROVIDERS)
def test_provider_normalize_uses_multipage_adapter(
    module_name: str,
    class_name: str,
    _expected_dpi: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    provider_class = getattr(module, class_name)
    provider = object.__new__(provider_class)
    sentinel = object()
    raw_result = object()

    def fake_normalize_pdf_pages(actual_raw_result: object, *, normalize_single_image: Any) -> object:
        assert actual_raw_result is raw_result
        assert normalize_single_image.__self__ is provider
        assert normalize_single_image.__func__ is provider_class.normalize
        return sentinel

    monkeypatch.setattr(module, "normalize_pdf_pages", fake_normalize_pdf_pages)

    assert provider.normalize(raw_result) is sentinel


@pytest.mark.parametrize(
    "module_name",
    ["amazon_nova", "dots_ocr", "anthropic", "google", "openai", "tesseract", "textract"],
)
def test_refactored_providers_do_not_call_pdf2image_eagerly(module_name: str) -> None:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    tree = ast.parse(inspect.getsource(module))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "convert_from_path"
    ]

    assert calls == []


def test_shared_rasterizer_bounds_every_convert_from_path_call() -> None:
    module = importlib.import_module("parse_bench.inference.providers.parse._multipage_image")
    source_path = Path(inspect.getsourcefile(module) or "")
    tree = ast.parse(source_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "convert_from_path"
    ]

    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords} >= {"first_page", "last_page"}
