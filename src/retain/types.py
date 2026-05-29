"""Core types for retain."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

__all__ = [
    "Context",
    "EntityRef",
    "EventRecord",
    "MemoryRecord",
    "ProcessRequest",
    "SearchResult",
    "TaskRecord",
]


class EntityRef(BaseModel):
    """Identifies an entity by type and ID."""

    entity_type: str
    entity_id: str


class MemoryRecord(BaseModel):
    """A typed fact about an entity.

    ``id`` is ``None`` for candidate facts returned by
    :func:`~retain.extraction.extract` and is assigned when the fact
    is stored via :meth:`~retain.memory.Memory.remember`.
    """

    id: str | None = None
    entity_type: str
    entity_id: str
    memory_type: str
    value: dict[str, Any]
    metadata: dict[str, Any] = {}
    source: str = "agent"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TaskRecord(BaseModel):
    """A tracked task with a status lifecycle."""

    id: str
    entity_type: str
    entity_id: str
    task_type: str
    status: str  # open | in_progress | resolved | cancelled
    description: str
    metadata: dict[str, Any] = {}
    created_at: datetime | None = None
    updated_at: datetime | None = None
    resolved_at: datetime | None = None


class EventRecord(BaseModel):
    """Tracks async processing status."""

    id: str
    event_type: str
    status: str  # pending | processing | completed | failed
    payload: dict[str, Any] = {}
    result: dict[str, Any] = {}
    created_at: datetime | None = None
    completed_at: datetime | None = None


class Context(BaseModel):
    """Returned by context() — everything an agent needs at call start."""

    entity_type: str
    entity_id: str
    profile_blob: str | None = None
    recent_memories: list[MemoryRecord] = []
    open_tasks: list[TaskRecord] = []
    memory_count: int = 0
    task_count_open: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None


class ProcessRequest(BaseModel):
    """Input for process() — a transcript to extract memories from."""

    entities: list[EntityRef]
    transcript: list[dict[str, Any]]
    instructions: str = ""
    metadata: dict[str, Any] = {}


class SearchResult(BaseModel):
    """A result from semantic search."""

    chunk_text: str
    score: float
    metadata: dict[str, Any] = {}
