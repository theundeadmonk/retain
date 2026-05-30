"""API route aggregation."""

from fastapi import APIRouter

from retain.routes.context import router as context_router
from retain.routes.memories import router as memories_router
from retain.routes.tasks import router as tasks_router

router = APIRouter()
router.include_router(context_router)
router.include_router(memories_router)
router.include_router(tasks_router)
