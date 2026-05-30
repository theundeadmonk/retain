"""Embedding providers for semantic search."""

from retain.embeddings.base import EmbeddingProvider
from retain.embeddings.local import FastEmbedProvider

__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
]
