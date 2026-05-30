"""Embedding providers for semantic search."""

from retain.embeddings.base import EmbeddingProvider
from retain.embeddings.local import FastEmbedProvider, SparseEmbedProvider

__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
    "SparseEmbedProvider",
]
