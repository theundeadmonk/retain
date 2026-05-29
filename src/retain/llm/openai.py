"""OpenAI-compatible LLM provider.

Works with any OpenAI-compatible API (OpenAI, DeepSeek, Grok, etc.).
Uses httpx for async HTTP calls — no openai SDK dependency.
"""

from __future__ import annotations

from typing import Any

import httpx

from retain.errors import RetainLLMError
from retain.llm.base import LLMProvider

__all__ = [
    "OpenAIProvider",
]

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TIMEOUT = 30.0


class OpenAIProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible chat completions endpoint.

    Args:
        api_key: API key for the provider.
        base_url: API base URL (e.g. ``https://api.openai.com/v1``).
        model: Default model to use when ``complete()`` is called without one.
        timeout: Timeout in seconds for each request.
        client: Optional pre-configured httpx AsyncClient. If omitted one is
            created with the given timeout.

    The provider can be used as an async context manager to ensure the
    underlying HTTP client is properly closed::

        async with OpenAIProvider(api_key="...") as llm:
            result = await llm.complete([...])
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = _DEFAULT_MODEL,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def __aenter__(self) -> OpenAIProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model or self._model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs,
                },
            )
        except httpx.HTTPError as exc:
            raise RetainLLMError(str(exc)) from exc

        if response.status_code != 200:
            body = response.text
            raise RetainLLMError(
                f"API returned {response.status_code}: {body}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RetainLLMError(f"Invalid JSON response: {exc}") from exc

        choices = data.get("choices")
        if not choices:
            raise RetainLLMError("API response missing choices")

        content = choices[0].get("message", {}).get("content", "")
        return content
