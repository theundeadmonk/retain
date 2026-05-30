"""POST /v1/context — entity lookup."""

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
