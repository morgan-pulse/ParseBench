from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse._multipage_image import (
    IMAGE_BACKED_PDF_PROVIDERS,
    PARSE_PROVIDER_PDF_CLASSIFICATIONS,
)
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

ADAPTER_PROVIDERS = [
    (spec.module_name, spec.class_name, spec.dpi) for spec in IMAGE_BACKED_PDF_PROVIDERS if spec.execution == "adapter"
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


def _provider(module_name: str, class_name: str, dpi: int) -> Any:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    provider = object.__new__(getattr(module, class_name))
    defaults = {
        "_dpi": dpi,
        "_timeout": 30,
        "_server_url": "http://provider.invalid",
        "_api_format": "openai",
        "_task": "parse",
        "_model": "test-model",
        "_served_model_name": "test-model",
        "_prompt_mode": "parse",
    }
    for name, value in defaults.items():
        setattr(provider, name, value)
    return provider


def _install_local_boundary(
    provider: Any,
    module_name: str,
    *,
    failure: Exception | None = None,
    fail_on_call: int = 1,
) -> list[int]:
    calls: list[int] = []

    def raw_output(page_number: int) -> dict[str, object]:
        if module_name == "infinity_parser2":
            return {
                "result": json.dumps(
                    [
                        {
                            "page": 1,
                            "category": "text",
                            "text": f"page {page_number}",
                            "bbox": [0, 0, 8, 8],
                        }
                    ]
                ),
                "_config": {"page_width": 8, "page_height": 8},
                "input_tokens": page_number * 10,
                "cost_usd": page_number / 10,
                "num_api_calls": 1,
            }
        return {
            "markdown": f"page {page_number}",
            "input_tokens": page_number * 10,
            "cost_usd": page_number / 10,
            "num_api_calls": 1,
        }

    if module_name == "infinity_parser2":

        def parse_document(path: str) -> dict[str, object]:
            calls.append(len(calls) + 1)
            if failure is not None and calls[-1] == fail_on_call:
                raise failure
            return raw_output(calls[-1])

        provider._parse_document = parse_document
    else:

        async def run_inference_async(*args: object, **kwargs: object) -> dict[str, object]:
            calls.append(len(calls) + 1)
            if failure is not None and calls[-1] == fail_on_call:
                raise failure
            return raw_output(calls[-1])

        provider._run_inference_async = run_inference_async

    return calls


def _mock_two_page_pdf(source: Path, monkeypatch: pytest.MonkeyPatch) -> list[Image.Image]:
    rendered: list[Image.Image] = []
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert first_page == last_page
        image = Image.new("RGB", (8 + first_page, 8), "white")
        rendered.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)
    return rendered


def _qualified_name(expression: ast.expr, bindings: dict[str, str]) -> str | None:
    if isinstance(expression, ast.Name):
        return bindings.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        parent = _qualified_name(expression.value, bindings)
        return f"{parent}.{expression.attr}" if parent is not None else None
    return None


def _registered_provider_classes(tree: ast.Module) -> list[tuple[str, str]]:
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                bindings[local_name] = alias.name if alias.asname else local_name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    registrations: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            target = _qualified_name(decorator.func, bindings)
            if target not in {
                "parse_bench.inference.register_provider",
                "parse_bench.inference.providers.register_provider",
                "parse_bench.inference.providers.registry.register_provider",
            }:
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            provider_name = decorator.args[0].value
            if isinstance(provider_name, str):
                registrations.append((provider_name, node.name))
    return registrations


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
def test_adapter_provider_runs_and_normalizes_actual_pdf_pages(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, expected_dpi)
    boundary_calls = _install_local_boundary(provider, module_name)

    raw_result = provider.run_inference(_pipeline(module_name), _request(source))
    result = provider.normalize(raw_result)

    assert boundary_calls == [1, 2]
    assert raw_result.raw_output["num_pages"] == 2
    assert raw_result.raw_output["input_tokens"] == 30
    assert raw_result.raw_output["cost_usd"] == pytest.approx(0.3)
    assert raw_result.raw_output["num_api_calls"] == 2
    envelope = raw_result.raw_output["_parse_bench_multipage"]
    assert isinstance(envelope, dict)
    assert [page["page_index"] for page in envelope["pages"]] == [0, 1]
    assert isinstance(result.output, ParseOutput)
    assert result.output.markdown == "page 1\n\npage 2"
    assert [(page.page_index, page.markdown) for page in result.output.pages] == [
        (0, "page 1"),
        (1, "page 2"),
    ]
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
def test_adapter_provider_preserves_single_image_path(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(source)
    provider = _provider(module_name, class_name, expected_dpi)
    boundary_calls = _install_local_boundary(provider, module_name)
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda *args, **kwargs: pytest.fail("single-image inference must not rasterize a PDF"),
    )

    raw_result = provider.run_inference(_pipeline(module_name), _request(source))
    result = provider.normalize(raw_result)

    assert boundary_calls == [1]
    assert "_parse_bench_multipage" not in raw_result.raw_output
    assert isinstance(result.output, ParseOutput)
    assert result.output.markdown == "page 1"


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
def test_adapter_provider_closes_rendered_page_when_persistence_fails(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, expected_dpi)
    _install_local_boundary(provider, module_name)
    multipage_module = importlib.import_module("parse_bench.inference.providers.parse._multipage_image")

    def fail_save(image: Image.Image, destination: Path) -> None:
        raise ProviderPermanentError("save failed")

    monkeypatch.setattr(multipage_module, "_save_png", fail_save)

    with pytest.raises(ProviderPermanentError, match="save failed"):
        provider.run_inference(_pipeline(module_name), _request(source))

    assert len(rendered) == 1
    with pytest.raises(ValueError, match="Operation on closed image"):
        rendered[0].getpixel((0, 0))


@pytest.mark.parametrize(
    "failure",
    [
        ProviderPermanentError("page two is invalid"),
        ProviderTransientError("page two timed out"),
    ],
    ids=["permanent", "transient"],
)
def test_paddleocr_page_two_failure_aborts_without_partial_result(
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider("paddleocr", "PaddleOCRProvider", 144)
    calls = 0
    successful_result = None

    async def run_inference_async(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return {"markdown": "page 1"}

    provider._run_inference_async = run_inference_async

    with pytest.raises(type(failure), match=str(failure)):
        successful_result = provider.run_inference(_pipeline("paddleocr"), _request(source))

    assert successful_result is None
    assert calls == 2
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
@pytest.mark.parametrize(
    "failure",
    [
        ProviderPermanentError("page two is invalid"),
        ProviderTransientError("page two timed out"),
    ],
    ids=["permanent", "transient"],
)
def test_adapter_provider_page_failure_aborts_without_partial_result(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, expected_dpi)
    calls = _install_local_boundary(provider, module_name, failure=failure, fail_on_call=2)
    successful_result = None

    with pytest.raises(type(failure), match=str(failure)):
        successful_result = provider.run_inference(_pipeline(module_name), _request(source))

    assert successful_result is None
    assert calls == [1, 2]
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(
    "module_name",
    [spec.module_name for spec in IMAGE_BACKED_PDF_PROVIDERS if spec.execution == "direct"],
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


@pytest.mark.parametrize(
    "source",
    [
        "from parse_bench.inference.providers.registry import register_provider\n"
        "@register_provider('sample')\nclass Sample: pass\n",
        "from parse_bench.inference.providers.registry import register_provider as register\n"
        "@register('sample')\nclass Sample: pass\n",
        "import parse_bench.inference.providers.registry as registry\n"
        "@registry.register_provider('sample')\nclass Sample: pass\n",
        "from parse_bench.inference.providers import registry as provider_registry\n"
        "@provider_registry.register_provider('sample')\nclass Sample: pass\n",
    ],
)
def test_registered_provider_discovery_resolves_import_aliases(source: str) -> None:
    assert _registered_provider_classes(ast.parse(source)) == [("sample", "Sample")]


def test_every_registered_parse_provider_has_explicit_pdf_classification() -> None:
    provider_dir = Path(__file__).parents[5] / "src/parse_bench/inference/providers/parse"
    registered: dict[str, tuple[str, str]] = {}

    for source_path in provider_dir.glob("*.py"):
        tree = ast.parse(source_path.read_text())
        for provider_name, class_name in _registered_provider_classes(tree):
            assert provider_name not in registered
            registered[provider_name] = (source_path.stem, class_name)

    classified = {spec.provider_name: spec for spec in PARSE_PROVIDER_PDF_CLASSIFICATIONS}
    assert len(classified) == len(PARSE_PROVIDER_PDF_CLASSIFICATIONS)
    assert set(classified) == set(registered)
    assert {
        provider_name: (spec.module_name, spec.class_name) for provider_name, spec in classified.items()
    } == registered
    assert {spec.pdf_handling for spec in classified.values()} == {
        "local-page-raster",
        "no-local-page-raster",
    }
    for spec in classified.values():
        if spec.pdf_handling == "local-page-raster":
            assert spec.dpi is not None
            assert spec.execution in {"adapter", "direct", "kdl"}
        else:
            assert spec.dpi is None
            assert spec.execution is None

    local_raster = {
        provider_name: (spec.module_name, spec.class_name)
        for provider_name, spec in classified.items()
        if spec.pdf_handling == "local-page-raster"
    }
    assert {
        spec.provider_name: (spec.module_name, spec.class_name) for spec in IMAGE_BACKED_PDF_PROVIDERS
    } == local_raster
    assert {spec.execution for spec in IMAGE_BACKED_PDF_PROVIDERS} == {"adapter", "direct", "kdl"}
