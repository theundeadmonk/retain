"""Hot-path operations — real-time reads and writes.

Functions execute in <50ms (reads) and <5ms (writes). No LLM calls.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Row, case, insert, select, update
from sqlalchemy.dialects.postgresql import insert as _pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from retain.models import Entity
from retain.models import Memory as MemoryModel
from retain.models import Task as TaskModel
from retain.types import Context, MemoryRecord, TaskRecord

__all__ = [
    "_row_to_memory",
    "_row_to_task",
    "complete_task",
    "context",
    "create_task",
    "list_tasks",
    "remember",
]


def _new_id() -> str:
    return uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _row_to_memory(row: Row[Any]) -> MemoryRecord:
    return MemoryRecord(
        id=row.id,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        memory_type=row.memory_type,
        value=row.value,
        metadata=row.extra or {},
        source=row.source,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _row_to_task(row: Row[Any]) -> TaskRecord:
    return TaskRecord(
        id=row.id,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        task_type=row.task_type,
        status=row.status,
        description=row.description,
        metadata=row.extra or {},
        created_at=row.created_at,
        updated_at=row.updated_at,
        resolved_at=row.resolved_at,
    )


async def _get_or_create_entity(
    conn: AsyncConnection, entity_type: str, entity_id: str
) -> Row[Any]:
    eid = _new_id()
    now = _utcnow()
    result = await conn.execute(
        _pg_insert(Entity)
        .values(
            id=eid,
            entity_type=entity_type,
            entity_id=entity_id,
            profile_blob=None,
            extra={},
            memory_count=0,
            task_count_open=0,
            first_seen=now,
            last_seen=now,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_update(
            index_elements=["entity_type", "entity_id"],
            set_={"entity_type": entity_type},
        )
        .returning(
            Entity.id,
            Entity.profile_blob,
            Entity.memory_count,
            Entity.task_count_open,
            Entity.first_seen,
            Entity.last_seen,
        )
    )
    row = result.fetchone()
    assert row is not None, "Entity must exist after upsert"
    return row


# ── hot path ──────────────────────────────────────────────


async def context(
    engine: AsyncEngine, entity_type: str, entity_id: str
) -> Context:
    """Return everything an agent needs at call start. <50ms."""
    async with engine.begin() as conn:
        entity = await _get_or_create_entity(conn, entity_type, entity_id)

        recent_result = await conn.execute(
            select(MemoryModel)
            .where(
                MemoryModel.entity_type == entity_type,
                MemoryModel.entity_id == entity_id,
            )
            .order_by(MemoryModel.created_at.desc())
            .limit(10)
        )
        recent_memories = [_row_to_memory(r) for r in recent_result.all()]

        open_tasks_result = await conn.execute(
            select(TaskModel)
            .where(
                TaskModel.entity_type == entity_type,
                TaskModel.entity_id == entity_id,
                TaskModel.status == "open",
            )
            .order_by(TaskModel.created_at.desc())
        )
        open_tasks = [_row_to_task(r) for r in open_tasks_result.all()]

    return Context(
        entity_type=entity_type,
        entity_id=entity_id,
        profile_blob=entity.profile_blob,
        recent_memories=recent_memories,
        open_tasks=open_tasks,
        memory_count=entity.memory_count,
        task_count_open=entity.task_count_open,
        first_seen=entity.first_seen,
        last_seen=entity.last_seen,
    )


async def remember(
    engine: AsyncEngine,
    entity_type: str,
    entity_id: str,
    memory_type: str,
    value: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    source: str = "agent",
) -> str:
    """Store a typed fact. Fire-and-forget. <5ms."""
    memory_id = _new_id()
    now = _utcnow()

    async with engine.begin() as conn:
        entity = await _get_or_create_entity(conn, entity_type, entity_id)

        await conn.execute(
            insert(MemoryModel).values(
                id=memory_id,
                entity_type=entity_type,
                entity_id=entity_id,
                memory_type=memory_type,
                value=value,
                extra=metadata or {},
                source=source,
                created_at=now,
            )
        )

        await conn.execute(
            update(Entity)
            .where(Entity.id == entity.id)
            .values(
                memory_count=Entity.memory_count + 1,
                last_seen=now,
                updated_at=now,
            )
        )

    return memory_id


async def create_task(
    engine: AsyncEngine,
    entity_type: str,
    entity_id: str,
    task_type: str,
    description: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> TaskRecord:
    """Create a tracked task. Fire-and-forget. <5ms."""
    task_id = _new_id()
    now = _utcnow()

    async with engine.begin() as conn:
        entity = await _get_or_create_entity(conn, entity_type, entity_id)

        await conn.execute(
            insert(TaskModel).values(
                id=task_id,
                entity_type=entity_type,
                entity_id=entity_id,
                task_type=task_type,
                status="open",
                description=description,
                extra=metadata or {},
                created_at=now,
                updated_at=now,
            )
        )

        await conn.execute(
            update(Entity)
            .where(Entity.id == entity.id)
            .values(
                task_count_open=Entity.task_count_open + 1,
                last_seen=now,
                updated_at=now,
            )
        )

    return TaskRecord(
        id=task_id,
        entity_type=entity_type,
        entity_id=entity_id,
        task_type=task_type,
        status="open",
        description=description,
        metadata=metadata or {},
        created_at=now,
        updated_at=now,
    )


async def complete_task(engine: AsyncEngine, task_id: str) -> TaskRecord | None:
    """Mark a task as resolved. <5ms."""
    now = _utcnow()

    async with engine.begin() as conn:
        result = await conn.execute(
            update(TaskModel)
            .where(TaskModel.id == task_id)
            .values(status="resolved", resolved_at=now, updated_at=now)
            .returning(
                TaskModel.id,
                TaskModel.entity_type,
                TaskModel.entity_id,
                TaskModel.task_type,
                TaskModel.description,
                TaskModel.extra,
                TaskModel.created_at,
            )
        )
        task_row = result.fetchone()
        if task_row is None:
            return None

        await conn.execute(
            update(Entity)
            .where(
                Entity.entity_type == task_row.entity_type,
                Entity.entity_id == task_row.entity_id,
            )
            .values(
                task_count_open=case(
                    (Entity.task_count_open > 0, Entity.task_count_open - 1),
                    else_=0,
                ),
                last_seen=now,
                updated_at=now,
            )
        )

    return TaskRecord(
        id=task_row.id,
        entity_type=task_row.entity_type,
        entity_id=task_row.entity_id,
        task_type=task_row.task_type,
        status="resolved",
        description=task_row.description,
        metadata=task_row.extra or {},
        created_at=task_row.created_at,
        updated_at=now,
        resolved_at=now,
    )


async def list_tasks(
    engine: AsyncEngine,
    entity_type: str,
    entity_id: str,
    status: str | None = None,
) -> list[TaskRecord]:
    """List tasks, optionally filtered by status. <50ms."""
    stmt = select(TaskModel).where(
        TaskModel.entity_type == entity_type,
        TaskModel.entity_id == entity_id,
    )
    if status is not None:
        stmt = stmt.where(TaskModel.status == status)
    stmt = stmt.order_by(TaskModel.created_at.desc())

    async with engine.begin() as conn:
        result = await conn.execute(stmt)
        tasks = [_row_to_task(r) for r in result.all()]

    return tasks
