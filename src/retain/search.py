"""Semantic search over transcript chunks — warm path.

~30-40ms end-to-end. Safe to call mid-call.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.embeddings.base import EmbeddingProvider
from retain.models import TranscriptChunk

__all__ = ["search"]


async def search(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    query: str,
    *,
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search transcript chunks by semantic similarity.

    Returns chunks ranked by cosine similarity (1 = perfect match).
    """
    query_vector = provider.encode_query_sync(query)

    distance = TranscriptChunk.embedding.cosine_distance(query_vector)
    score = 1.0 - distance

    stmt = select(
        TranscriptChunk.id,
        TranscriptChunk.transcript_id,
        TranscriptChunk.entity_type,
        TranscriptChunk.entity_id,
        TranscriptChunk.chunk_index,
        TranscriptChunk.chunk_text,
        score.label("score"),
    ).where(TranscriptChunk.embedding.isnot(None))

    if entity_type is not None:
        stmt = stmt.where(TranscriptChunk.entity_type == entity_type)
    if entity_id is not None:
        stmt = stmt.where(TranscriptChunk.entity_id == entity_id)

    stmt = stmt.order_by(distance).limit(limit)

    async with engine.begin() as conn:
        result = await conn.execute(stmt)
        rows = result.fetchall()

    return {
        "chunks": [
            {
                "id": row.id,
                "transcript_id": row.transcript_id,
                "entity_type": row.entity_type,
                "entity_id": row.entity_id,
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "score": round(float(row.score), 4),
            }
            for row in rows
        ]
    }
