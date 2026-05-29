"""Mock LLM provider for testing."""

from typing import Any

from retain.llm.base import LLMProvider

__all__ = [
    "MockProvider",
]


class MockProvider(LLMProvider):
    """Deterministic mock that records every call.

    Args:
        response: The string to return on every call.

    The :attr:`calls` list records each invocation so tests can
    inspect what was sent to the provider.
    """

    def __init__(self, response: str = "") -> None:
        self.response = response
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
        return self.response
