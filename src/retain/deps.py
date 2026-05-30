"""FastAPI dependencies."""

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.embeddings.base import EmbeddingProvider
from retain.llm.base import LLMProvider


async def get_engine(request: Request) -> AsyncEngine:
    return request.app.state.engine


async def get_llm(request: Request) -> LLMProvider | None:
    return request.app.state.llm


async def get_embedding(request: Request) -> EmbeddingProvider | None:
    return getattr(request.app.state, "embedding_provider", None)
