"""Tests for embedding providers."""

import pytest

from retain.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Fake provider that returns fixed-size vectors."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_sync_impl(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]


class TestEmbeddingProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_concrete_implementation(self):
        provider = MockEmbeddingProvider(dim=256)
        assert provider.dim == 256

    async def test_encode_returns_correct_shape(self):
        provider = MockEmbeddingProvider(dim=128)
        result = await provider.encode(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 128

    async def test_encode_query_returns_single_vector(self):
        provider = MockEmbeddingProvider(dim=384)
        vec = await provider.encode_query("test")
        assert len(vec) == 384

    async def test_encode_documents_returns_list(self):
        provider = MockEmbeddingProvider(dim=256)
        vecs = await provider.encode_documents(["doc1", "doc2", "doc3"])
        assert len(vecs) == 3
        assert all(len(v) == 256 for v in vecs)

    def test_encode_query_sync(self):
        provider = MockEmbeddingProvider(dim=64)
        vec = provider.encode_query_sync("test")
        assert len(vec) == 64
