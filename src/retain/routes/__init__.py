"""API route aggregation — hot, warm, and cold path routes."""

from fastapi import APIRouter

from retain.routes.context import router as context_router
from retain.routes.events import router as events_router
from retain.routes.memories import router as memories_router
from retain.routes.search import router as search_router
from retain.routes.tasks import router as tasks_router

router = APIRouter()

# hot path (<50ms, pure PostgreSQL)
router.include_router(context_router)
router.include_router(memories_router)
router.include_router(tasks_router)

# warm path (<200ms, local embedding + pgvector)
router.include_router(search_router)

# cold path (2-5s+, fire-and-forget async)
router.include_router(events_router)
