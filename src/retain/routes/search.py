"""Warm-path route — POST /v1/search.

Semantic search over transcript chunks using locally-embedded vectors.
~70ms end-to-end (embedding + pgvector HNSW). Safe to call mid-call;
results return before the LLM response completes.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.deps import get_embedding, get_engine
from retain.embeddings.base import EmbeddingProvider
from retain.search import search as search_chunks

router = APIRouter(tags=["search"])


@router.post("/search")
async def search_route(
    query: str = Query(..., description="Natural language search query"),
    entity_type: str | None = Query(None),
    entity_id: str | None = Query(None),
    limit: int = Query(5, ge=1, le=50),
    engine: AsyncEngine = Depends(get_engine),
    provider: EmbeddingProvider | None = Depends(get_embedding),
) -> dict[str, Any]:
    """Semantic search over stored transcript chunks.

    Returns chunks ranked by cosine similarity, optionally filtered
    to a specific entity.
    """
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding provider not configured — search requires embeddings",
        )

    return await search_chunks(
        engine,
        provider,
        query,
        entity_type=entity_type,
        entity_id=entity_id,
        limit=limit,
    )
