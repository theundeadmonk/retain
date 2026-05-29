"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "LLMProvider",
]


class LLMProvider(ABC):
    """Interface for calling an LLM.

    Implementations must be async and thread-safe. Providers should
    reuse a single HTTP client across all calls.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return the response content.

        Args:
            messages: OpenAI-format message list (role/content pairs).
            model: Model identifier. Falls back to provider default if None.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            **kwargs: Additional provider-specific parameters.

        Returns:
            The text content of the model's response.

        Raises:
            RetainLLMError: On API errors, timeouts, or unexpected responses.
        """
        ...
