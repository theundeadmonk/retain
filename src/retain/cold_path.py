"""Cold-path operations — async LLM extraction, dedup, synthesis.

Functions may take 2-5 seconds total (extract, synthesize profile).
Use process() to submit work and event_status() to poll completion.
"""


import asyncio
import logging

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.chunking import chunk_transcript
from retain.embeddings.base import EmbeddingProvider
from retain.extraction import deduplicate, extract, synthesize_profile
from retain.hot_path import (
    _new_id,
    _row_to_memory,
    _utcnow,
    remember,
)
from retain.llm.base import LLMProvider
from retain.models import Entity, TranscriptChunk
from retain.models import Event as EventModel
from retain.models import Memory as MemoryModel
from retain.models import Transcript as TranscriptModel
from retain.types import MemoryRecord, ProcessRequest

__all__ = [
    "event_status",
    "process",
]

logger = logging.getLogger("retain")


async def process(
    engine: AsyncEngine,
    llm: LLMProvider,
    request: ProcessRequest,
    *,
    extraction_timeout: float = 30.0,
    embedding_provider: EmbeddingProvider | None = None,
) -> str:
    """Submit a transcript for async extraction. Returns event_id."""
    event_id = _new_id()
    transcript_id = _new_id()
    now = _utcnow()

    async with engine.begin() as conn:
        await conn.execute(
            insert(TranscriptModel).values(
                id=transcript_id,
                entities={"entities": [
                    {"entity_type": e.entity_type, "entity_id": e.entity_id}
                    for e in request.entities
                ]},
                content={"messages": request.transcript},
                extra=request.metadata,
                status="processing",
                created_at=now,
            )
        )
        await conn.execute(
            insert(EventModel).values(
                id=event_id,
                event_type="extraction",
                status="pending",
                payload={
                    "transcript_id": transcript_id,
                    "entities": [
                        {"entity_type": e.entity_type, "entity_id": e.entity_id}
                        for e in request.entities
                    ],
                    "extraction_timeout": extraction_timeout,
                },
                created_at=now,
            )
        )

    _ = asyncio.create_task(
        _process_background(
            engine, llm, event_id, request, extraction_timeout, embedding_provider,
        )
    )
    return event_id


async def _process_background(
    engine: AsyncEngine,
    llm: LLMProvider,
    event_id: str,
    request: ProcessRequest,
    extraction_timeout: float,
    embedding_provider: EmbeddingProvider | None = None,
) -> None:
    """Async pipeline: extract → dedup → store → synthesize → embed."""
    transcript_id = await _get_transcript_id_from_event(engine, event_id)

    try:
        await asyncio.wait_for(
            _run_extraction(engine, llm, event_id, request),
            timeout=extraction_timeout,
        )
    except TimeoutError:
        await _fail_event(engine, event_id, f"extraction timed out after {extraction_timeout}s")
        return
    except Exception as exc:
        logger.exception("Background extraction %s failed", event_id)
        await _fail_event(engine, event_id, str(exc))
        return

    if embedding_provider is not None and transcript_id is not None:
        try:
            await _embed_chunks(
                engine, embedding_provider, transcript_id, request
            )
        except Exception:
            logger.exception(
                "Chunk embedding failed for %s — extraction already complete",
                event_id,
            )


async def _run_extraction(
    engine: AsyncEngine,
    llm: LLMProvider,
    event_id: str,
    request: ProcessRequest,
) -> None:
    """Core extraction pipeline."""
    now = _utcnow()

    async with engine.begin() as conn:
        await conn.execute(
            update(EventModel)
            .where(EventModel.id == event_id)
            .values(status="processing", updated_at=now)
        )

    facts = await extract(
        llm,
        request.transcript,
        request.entities,
        instructions=request.instructions,
    )

    entity_facts: dict[tuple[str, str], list[MemoryRecord]] = {}
    for f in facts:
        key = (f.entity_type, f.entity_id)
        entity_facts.setdefault(key, []).append(f)

    for (entity_type, entity_id), new_facts in entity_facts.items():
        async with engine.begin() as conn:
            result = await conn.execute(
                select(MemoryModel).where(
                    MemoryModel.entity_type == entity_type,
                    MemoryModel.entity_id == entity_id,
                )
            )
            existing = [_row_to_memory(r) for r in result.all()]

        novel = deduplicate(new_facts, existing)

        for fact in novel:
            await remember(
                engine,
                fact.entity_type,
                fact.entity_id,
                fact.memory_type,
                fact.value,
                source=fact.source,
            )

        all_facts = existing + novel
        profile = await synthesize_profile(
            llm, entity_type, entity_id, all_facts,
        )

        async with engine.begin() as conn:
            await conn.execute(
                update(Entity)
                .where(
                    Entity.entity_type == entity_type,
                    Entity.entity_id == entity_id,
                )
                .values(profile_blob=profile, updated_at=_utcnow())
            )

    async with engine.begin() as conn:
        await conn.execute(
            update(EventModel)
            .where(EventModel.id == event_id)
            .values(
                status="completed",
                result={
                    "facts_extracted": len(facts),
                    "entities_processed": len(entity_facts),
                },
                completed_at=_utcnow(),
                updated_at=_utcnow(),
            )
        )


async def _fail_event(engine: AsyncEngine, event_id: str, error: str) -> None:
    """Mark an event as ``"failed"`` with the given error string."""
    try:
        async with engine.begin() as conn:
            await conn.execute(
                update(EventModel)
                .where(EventModel.id == event_id)
                .values(
                    status="failed",
                    result={"error": error},
                    completed_at=_utcnow(),
                    updated_at=_utcnow(),
                )
            )
    except Exception:
        logger.exception("Failed to mark event %s as failed", event_id)


async def event_status(
    engine: AsyncEngine, event_id: str
) -> dict[str, object]:
    """Poll the status of an async extraction event."""
    async with engine.begin() as conn:
        result = await conn.execute(
            select(EventModel).where(EventModel.id == event_id)
        )
        row = result.fetchone()
        if row is None:
            return {"status": "not_found"}

    return {
        "event_id": row.id,
        "event_type": row.event_type,
        "status": row.status,
        "payload": row.payload or {},
        "result": row.result or {},
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "completed_at": row.completed_at,
    }


async def _get_transcript_id_from_event(
    engine: AsyncEngine, event_id: str
) -> str | None:
    async with engine.begin() as conn:
        result = await conn.execute(
            select(EventModel.payload).where(EventModel.id == event_id)
        )
        row = result.fetchone()
        if row is None:
            return None
        return row.payload.get("transcript_id") if row.payload else None


async def _embed_chunks(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    transcript_id: str,
    request: ProcessRequest,
) -> None:
    """Chunk the transcript, embed, and store in transcript_chunks."""
    chunks = chunk_transcript(request.transcript)
    if not chunks:
        return

    texts: list[str] = []
    rows: list[dict[str, object]] = []
    for i, chunk_text in enumerate(chunks):
        entity_type = request.entities[0].entity_type if request.entities else "unknown"
        entity_id = request.entities[0].entity_id if request.entities else "unknown"
        texts.append(chunk_text)
        rows.append({
            "id": _new_id(),
            "transcript_id": transcript_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "chunk_index": i,
            "chunk_text": chunk_text,
        })

    embeddings = await provider.encode_documents(texts)

    async with engine.begin() as conn:
        for row_data, embedding in zip(rows, embeddings):
            await conn.execute(
                insert(TranscriptChunk).values(
                    id=row_data["id"],
                    transcript_id=row_data["transcript_id"],
                    entity_type=row_data["entity_type"],
                    entity_id=row_data["entity_id"],
                    chunk_index=row_data["chunk_index"],
                    chunk_text=row_data["chunk_text"],
                    embedding=embedding,
                    created_at=_utcnow(),
                    updated_at=_utcnow(),
                )
            )
