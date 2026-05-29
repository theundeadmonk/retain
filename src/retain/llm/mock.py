"""Mock LLM provider for testing."""

from typing import Any

from retain.llm.base import LLMProvider

__all__ = [
    "MockProvider",
]


class MockProvider(LLMProvider):
    """Deterministic mock that records every call.

    Args:
        response: The string(s) to return. If a single string, every
            call returns it. If a list, the *n*-th call returns the
            *n*-th string, repeating the last element when exhausted.

    The :attr:`calls` list records each invocation so tests can
    inspect what was sent to the provider.
    """

    def __init__(self, response: str | list[str] = "") -> None:
        if isinstance(response, str):
            self._responses = [response]
        else:
            self._responses = list(response)
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        self.calls.append({
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "kwargs": kwargs,
        })
        idx = min(self._call_count, len(self._responses) - 1)
        result = self._responses[idx]
        self._call_count += 1
        return result
