"""Minimal OpenAI-compatible vision client for ground-truth generation.

Deliberately dependency-light (httpx only) so it works against OpenRouter,
Azure OpenAI, a customer's own gateway, or anything else speaking
``/chat/completions``. Customers in regulated environments routinely need to
point this at an internal endpoint; a hardcoded vendor SDK would block that.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# Retried: transient server and rate-limit responses.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class GroundTruthModelError(Exception):
    """Raised when the ground-truth model cannot be called or returns junk."""


@dataclass
class ModelResponse:
    """One completion, with the usage numbers needed for cost reporting."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_json(self) -> dict[str, Any]:
        """Parse the completion as a JSON object, tolerating code fences."""
        text = _JSON_FENCE.sub("", self.content).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            # Models occasionally prepend prose. Recover the outermost object.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise GroundTruthModelError(f"Model did not return JSON: {self.content[:300]!r}") from e
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise GroundTruthModelError(f"Model did not return JSON: {self.content[:300]!r}") from e
        if not isinstance(parsed, dict):
            raise GroundTruthModelError(f"Expected a JSON object, got {type(parsed).__name__}")
        return parsed


class VisionModelClient:
    """Chat-completions client for image + text prompts."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        *,
        timeout: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key = os.getenv(api_key_env)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def require_key(self) -> None:
        """Fail early with an actionable message when the key is absent."""
        if not self._api_key:
            raise GroundTruthModelError(
                f"{self.api_key_env} is not set. Add it to your project's .env file, "
                f"or pass --api_key. Ground-truth generation is the only step that "
                f"needs it — you can skip it entirely by supplying your own ground truth."
            )

    def set_api_key(self, api_key: str) -> None:
        self._api_key = api_key

    def complete(
        self,
        system_prompt: str,
        user_text: str,
        images: list[str] | None = None,
        *,
        json_object: bool = True,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Run one completion.

        :param images: Data URLs, in page order.
        :param json_object: Request a JSON object response.
        :raises GroundTruthModelError: On configuration errors or exhausted retries.
        """
        try:
            import httpx
        except ImportError as e:
            raise GroundTruthModelError(
                "httpx is required for ground-truth generation. Install it with: uv sync --extra runners"
            ) from e

        self.require_key()

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_url in images or []:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers; harmless on other gateways.
            "HTTP-Referer": "https://parsebench.ai",
            "X-Title": "ParseBench customer evaluation",
        }

        last_error: str = ""
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                if response.status_code in _RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    time.sleep(min(2**attempt, 30))
                    continue
                if response.status_code >= 400:
                    raise GroundTruthModelError(f"HTTP {response.status_code}: {response.text[:500]}")
                body = response.json()
            except GroundTruthModelError:
                raise
            except Exception as e:  # network-level failure
                last_error = str(e)
                time.sleep(min(2**attempt, 30))
                continue

            choices = body.get("choices") or []
            if not choices:
                last_error = f"No choices in response: {str(body)[:200]}"
                time.sleep(min(2**attempt, 30))
                continue

            message = choices[0].get("message") or {}
            usage = body.get("usage") or {}
            return ModelResponse(
                content=message.get("content") or "",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            )

        raise GroundTruthModelError(f"Ground-truth model failed after {self.max_retries} attempts: {last_error}")
