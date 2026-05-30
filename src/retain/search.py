"""Hybrid search over transcript chunks — warm path.

Dense + sparse retrieval with Reciprocal Rank Fusion (RRF).
Query-adaptive weighting + SPLADE tiebreaking.
~40-60ms end-to-end. Safe to call mid-call.

Techniques:
- Hybrid Dense+Sparse Retrieval (2025)
  Why: Dense vectors excel at semantic similarity.
       SPLADE sparse vectors excel at keyword matching.
       RRF fuses both rankings for higher recall than either alone.

- Query-Adaptive Hybrid Weighting
  Why: Queries containing order numbers, ticket IDs, or other identifiers
       benefit from higher sparse weight. Free — zero latency cost.

- SPLADE Score Tiebreaking
  Why: Two rows with similar RRF scores can be broken by raw SPLADE
       distance. Raw distances are already computed during ranking.
       Zero added latency — reuse pre-computed sparse distances.
"""

import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.embeddings.base import EmbeddingProvider
from retain.embeddings.local import SparseEmbedProvider
from retain.models import TranscriptChunk

__all__ = ["search"]

_HNSW_SETUP = [
    "SET LOCAL hnsw.ef_search = 100",
    "SET LOCAL hnsw.iterative_scan = 'relaxed_order'",
    "SET LOCAL hnsw.max_scan_tuples = 20000",
]

_IDENTIFIER_DIGITS_RE = re.compile(r"\d{3,}")


def _query_is_identifier_heavy(query: str) -> bool:
    """Queries with 3+ digit sequences (order numbers, ticket IDs, phone
    numbers, booking refs) benefit from higher sparse weight for exact
    term matching."""
    return bool(_IDENTIFIER_DIGITS_RE.search(query))


async def search(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    query: str,
    *,
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 5,
    sparse_provider: SparseEmbedProvider | None = None,
) -> dict[str, Any]:
    """Hybrid search: dense + sparse with Reciprocal Rank Fusion.

    Returns chunks ranked by combined relevance score.
    Falls back to dense-only if no sparse provider is available.
    """
    if not query.strip():
        return {"chunks": []}

    query_vector = provider.encode_query_sync(query)

    if sparse_provider is not None:
        query_sparse = sparse_provider.encode_query_sync(query)
        identifier_heavy = _query_is_identifier_heavy(query)
        return await _hybrid_search(
            engine,
            _format_vector(query_vector),
            query_sparse,
            entity_type=entity_type,
            entity_id=entity_id,
            limit=limit,
            identifier_heavy=identifier_heavy,
        )

    return await _dense_search(
        engine,
        query_vector,
        entity_type=entity_type,
        entity_id=entity_id,
        limit=limit,
    )


async def _dense_search(
    engine: AsyncEngine,
    query_vector: list[float],
    *,
    entity_type: str | None,
    entity_id: str | None,
    limit: int,
) -> dict[str, Any]:
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
        for setup in _HNSW_SETUP:
            await conn.execute(text(setup))
        result = await conn.execute(stmt)
        return _rows_to_response(result.fetchall())


async def _hybrid_search(
    engine: AsyncEngine,
    query_vector: str,
    query_sparse: dict[str, object],
    *,
    entity_type: str | None,
    entity_id: str | None,
    limit: int,
    identifier_heavy: bool = False,
) -> dict[str, Any]:
    # text() is required here — pgvector's sparsevec SQLAlchemy type
    # does not expose l2_distance() / cosine_distance() operators.
    # Using op("<->") with Core literal() adds ::VARCHAR to the bind
    # parameter, which breaks PostgreSQL's implicit text→sparsevec cast.
    # The raw text() path lets asyncpg bind without type coercion.
    sparse_str = _format_sparse(query_sparse)
    filters = _entity_filter(entity_type, entity_id)

    if identifier_heavy:
        dense_weight, sparse_weight = 0.4, 0.6
    else:
        dense_weight, sparse_weight = 1.0, 1.0

    params: dict[str, Any] = {
        "query_vec": query_vector,
        "query_sparse": sparse_str,
        "limit": limit,
    }
    _bind_entity(params, entity_type, entity_id)

    stmt = text(f"""
        WITH dense AS (
            SELECT tc.id, tc.transcript_id, tc.entity_type, tc.entity_id,
                   tc.chunk_index, tc.chunk_text,
                   ROW_NUMBER() OVER (
                       ORDER BY tc.embedding <=> :query_vec
                   ) AS rank
            FROM transcript_chunks tc
            WHERE tc.embedding IS NOT NULL{filters}
        ),
        sparse AS (
            SELECT tc.id, tc.transcript_id, tc.entity_type, tc.entity_id,
                   tc.chunk_index, tc.chunk_text,
                   ROW_NUMBER() OVER (
                       ORDER BY tc.sparse_embedding <-> :query_sparse
                   ) AS rank,
                   (tc.sparse_embedding <-> :query_sparse) AS raw_distance
            FROM transcript_chunks tc
            WHERE tc.sparse_embedding IS NOT NULL{filters}
        )
        SELECT COALESCE(d.id, s.id) AS id,
               COALESCE(d.transcript_id, s.transcript_id) AS transcript_id,
               COALESCE(d.entity_type, s.entity_type) AS entity_type,
               COALESCE(d.entity_id, s.entity_id) AS entity_id,
               COALESCE(d.chunk_index, s.chunk_index) AS chunk_index,
               COALESCE(d.chunk_text, s.chunk_text) AS chunk_text,
               {dense_weight}/(60 + COALESCE(d.rank, 1000))
               + {sparse_weight}/(60 + COALESCE(s.rank, 1000))
               + CASE WHEN s.raw_distance IS NOT NULL
                      THEN 1.0e-4 * (1.0 - LEAST(s.raw_distance, 2.0) / 2.0)
                      ELSE 0 END AS score
        FROM dense d
        FULL OUTER JOIN sparse s ON d.id = s.id
        ORDER BY score DESC
        LIMIT :limit
    """)
    async with engine.begin() as conn:
        for setup in _HNSW_SETUP:
            await conn.execute(text(setup))
        result = await conn.execute(stmt, params)
        return _rows_to_response(result.fetchall())


def _entity_filter(
    entity_type: str | None, entity_id: str | None
) -> str:
    parts: list[str] = []
    if entity_type is not None:
        parts.append("tc.entity_type = :entity_type")
    if entity_id is not None:
        parts.append("tc.entity_id = :entity_id")
    if parts:
        return " AND " + " AND ".join(parts)
    return ""


def _bind_entity(
    params: dict[str, Any],
    entity_type: str | None,
    entity_id: str | None,
) -> None:
    if entity_type is not None:
        params["entity_type"] = entity_type
    if entity_id is not None:
        params["entity_id"] = entity_id


def _format_vector(values: list[float]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def _format_sparse(sparse: dict[str, object]) -> str:
    indices: list[int] = sparse.get("indices", [])  # type: ignore[assignment]
    values: list[float] = sparse.get("values", [])  # type: ignore[assignment]
    if len(indices) != len(values):
        raise ValueError(
            f"SPLADE indices/values length mismatch: "
            f"{len(indices)} vs {len(values)}"
        )
    pairs = ",".join(f"{i}:{v}" for i, v in zip(indices, values))
    return "{" + pairs + "}/30522"


def _rows_to_response(rows: Sequence[Any]) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "id": row.id,
                "transcript_id": row.transcript_id or "",
                "entity_type": row.entity_type or "",
                "entity_id": row.entity_id or "",
                "chunk_index": row.chunk_index if row.chunk_index is not None else -1,
                "chunk_text": row.chunk_text or "",
                "score": round(float(row.score), 4),
            }
            for row in rows
        ]
    }
