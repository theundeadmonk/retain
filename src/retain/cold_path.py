"""Cold-path operations — async LLM extraction, dedup, synthesis.

Functions may take 2-5 seconds total (extract, synthesize profile).
Use process() to submit work and event_status() to poll completion.
"""

import asyncio
import logging

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.extraction import deduplicate, extract, synthesize_profile
from retain.hot_path import (
    _new_id,
    _row_to_memory,
    _utcnow,
    remember,
)
from retain.llm.base import LLMProvider
from retain.models import Entity
from retain.models import Event as EventModel
from retain.models import Memory as MemoryModel
from retain.models import Transcript as TranscriptModel
from retain.types import MemoryRecord, ProcessRequest

__all__ = [
    "event_status",
    "process",
    "search",
]

logger = logging.getLogger("retain")


async def process(
    engine: AsyncEngine,
    llm: LLMProvider,
    request: ProcessRequest,
    *,
    extraction_timeout: float = 30.0,
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
        _process_background(engine, llm, event_id, request, extraction_timeout)
    )
    return event_id


async def _process_background(
    engine: AsyncEngine,
    llm: LLMProvider,
    event_id: str,
    request: ProcessRequest,
    extraction_timeout: float,
) -> None:
    """Async pipeline: extract → dedup → store → synthesize."""
    try:
        await asyncio.wait_for(
            _run_extraction(engine, llm, event_id, request),
            timeout=extraction_timeout,
        )
    except TimeoutError:
        await _fail_event(
            engine,
            event_id,
            f"extraction timed out after {extraction_timeout}s",
        )
    except Exception as exc:
        logger.exception("Background extraction %s failed", event_id)
        await _fail_event(engine, event_id, str(exc))


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


async def search(
    engine: AsyncEngine,
    entity_type: str,
    entity_id: str,
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    """Semantic search across conversation history (optional)."""
    raise NotImplementedError("search is not yet implemented")
