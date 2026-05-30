"""Local embedding provider using fastembed (ONNX Runtime).

No PyTorch dependency. ~3x faster inference on CPU vs. sentence-transformers.
"""


import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from retain.embeddings.base import EmbeddingProvider

__all__ = ["FastEmbedProvider"]


class FastEmbedProvider(EmbeddingProvider):
    """Embedding provider backed by fastembed (ONNX Runtime).

    Runs synchronous model.embed() in a thread pool to avoid blocking
    the asyncio event loop. Caller should access ``.dim`` or ``.encode()``
    eagerly during startup to preload the ONNX model; otherwise loading
    is deferred to first use.
    No PyTorch — ONNX-only, ~500 MB Docker overhead, 1-2s cold start.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        *,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._dim: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=self._model_name,
            batch_size=self._batch_size,
        )
        self._dim = self._model._get_model_description(self._model_name)["dim"]

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._encode_sync,
            texts,
        )

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.encode(texts)

    async def encode_query(self, text: str) -> list[float]:
        result = await self.encode([text])
        return result[0]

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        assert self._model is not None
        embeddings = list(self._model.embed(texts))
        result: list[list[float]] = []
        for emb in embeddings:
            vec = np.asarray(emb, dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) or 1.0)
            result.append(vec.tolist())
        return result

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def _encode_sync_impl(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return self._encode_sync(texts)
