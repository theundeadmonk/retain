"""Task routes — create, complete, list."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.deps import get_engine
from retain.hot_path import complete_task, create_task, list_tasks
from retain.types import CreateTaskRequest

router = APIRouter(tags=["tasks"])


@router.post("/tasks")
async def create_task_route(
    req: CreateTaskRequest, engine: AsyncEngine = Depends(get_engine)
):
    return await create_task(
        engine,
        req.entity_type,
        req.entity_id,
        req.task_type,
        req.description,
        metadata=req.metadata or None,
    )


@router.patch("/tasks/{task_id}")
async def complete_task_route(
    task_id: str, engine: AsyncEngine = Depends(get_engine)
):
    task = await complete_task(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/tasks")
async def list_tasks_route(
    entity_type: str = Query(...),
    entity_id: str = Query(...),
    status: str | None = Query(None),
    engine: AsyncEngine = Depends(get_engine),
):
    return await list_tasks(engine, entity_type, entity_id, status=status)
