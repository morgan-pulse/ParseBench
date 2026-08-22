from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from PIL import Image

from parse_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.dots_ocr import DotsOcrParseProvider


def _provider_raising(exc: Exception) -> DotsOcrParseProvider:
    provider = object.__new__(DotsOcrParseProvider)

    def create(**kwargs: object) -> None:
        raise exc

    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider._model = "dots-ocr"
    provider._prompt = "parse"
    provider._max_tokens = 100
    provider._temperature = 0.1
    provider._top_p = 0.9
    return provider


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://dots.invalid/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    if status_code == 429:
        return RateLimitError("rate limited", response=response, body=None)
    return APIStatusError("request failed", response=response, body=None)


@pytest.mark.parametrize(
    "exc",
    [
        APITimeoutError(request=httpx.Request("POST", "https://dots.invalid")),
        APIConnectionError(request=httpx.Request("POST", "https://dots.invalid")),
        _status_error(408),
        _status_error(500),
        _status_error(503),
    ],
)
def test_dots_real_transport_and_transient_status_errors_are_retryable(exc: Exception) -> None:
    provider = _provider_raising(exc)

    with pytest.raises(ProviderTransientError):
        provider._call_endpoint(Image.new("RGB", (1, 1)))


def test_dots_real_429_is_rate_limited() -> None:
    provider = _provider_raising(_status_error(429))

    with pytest.raises(ProviderRateLimitError):
        provider._call_endpoint(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_dots_other_real_4xx_statuses_are_permanent(status_code: int) -> None:
    provider = _provider_raising(_status_error(status_code))

    with pytest.raises(ProviderPermanentError):
        provider._call_endpoint(Image.new("RGB", (1, 1)))
