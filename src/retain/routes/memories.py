"""Hot-path route — POST /v1/memories.

<5ms fire-and-forget insert. No response body beyond the new memory ID.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.deps import get_engine
from retain.hot_path import remember
from retain.types import CreateMemoryRequest

router = APIRouter(tags=["memories"])


@router.post("/memories")
async def create_memory(
    req: CreateMemoryRequest, engine: AsyncEngine = Depends(get_engine)
):
    memory_id = await remember(
        engine,
        req.entity_type,
        req.entity_id,
        req.memory_type,
        req.value,
        metadata=req.metadata or None,
        source=req.source,
    )
    return {"id": memory_id}
