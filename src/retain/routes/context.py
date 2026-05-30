"""Hot-path route — POST /v1/context.

<50ms entity lookup returning profile_blob + recent memories + open tasks.
Pure PostgreSQL read; no LLM, no embeddings.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.deps import get_engine
from retain.hot_path import context
from retain.types import ContextRequest

router = APIRouter(tags=["context"])


@router.post("/context")
async def get_context(
    req: ContextRequest, engine: AsyncEngine = Depends(get_engine)
):
    return await context(engine, req.entity_type, req.entity_id)
