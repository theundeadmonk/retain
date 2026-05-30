"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod

__all__ = ["EmbeddingProvider"]


class EmbeddingProvider(ABC):
    """Interface for text → vector embedding.

    Implementations must be async-capable and thread-safe. Providers
    should cache the loaded model and reuse it across all calls.
    """

    @abstractmethod
    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode one or more texts into embedding vectors.

        Returns a list of vectors, one per input text, in the same order.
        Each vector is a list of floats whose length matches the provider's
        dimensionality.
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the vectors produced by this provider."""
        ...

    async def encode_query(self, text: str) -> list[float]:
        """Encode a single search query (defaults to encode)."""
        result = await self.encode([text])
        return result[0]

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode documents for storage (defaults to encode)."""
        return await self.encode(texts)

    def encode_query_sync(self, text: str) -> list[float]:
        """Synchronous convenience for database callers."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._encode_sync_impl([text])[0]

        future = asyncio.run_coroutine_threadsafe(
            self.encode_query(text), loop
        )
        return future.result()

    def _encode_sync_impl(self, texts: list[str]) -> list[list[float]]:
        """Override in subclasses to provide a true synchronous path."""
        raise NotImplementedError(
            "Synchronous encoding is not supported by this provider"
        )
