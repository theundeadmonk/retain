"""Warm-path route — POST /v1/search.

Hybrid dense+sparse search with Reciprocal Rank Fusion (RRF).
~40-60ms end-to-end. Safe to call mid-call.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.deps import get_embedding, get_engine, get_sparse
from retain.embeddings.base import EmbeddingProvider
from retain.embeddings.local import SparseEmbedProvider
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
    sparse_provider: SparseEmbedProvider | None = Depends(get_sparse),
) -> dict[str, Any]:
    """Hybrid search over stored transcript chunks.

    Combines dense (semantic) and sparse (keyword) retrieval via RRF.
    Optionally filtered to a specific entity.
    """
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding provider not configured",
        )
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="query must not be empty",
        )

    return await search_chunks(
        engine,
        provider,
        query,
        entity_type=entity_type,
        entity_id=entity_id,
        limit=limit,
        sparse_provider=sparse_provider,
    )
