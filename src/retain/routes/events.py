"""Cold-path routes — process transcripts, poll events."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.cold_path import event_status, process
from retain.deps import get_engine, get_llm
from retain.llm.base import LLMProvider
from retain.types import ProcessRequest

router = APIRouter(tags=["events"])


@router.post("/process", status_code=http_status.HTTP_202_ACCEPTED)
async def process_transcript(
    req: ProcessRequest,
    engine: AsyncEngine = Depends(get_engine),
    llm: LLMProvider | None = Depends(get_llm),
):
    if llm is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider not configured — extraction requires an LLM",
        )
    event_id = await process(engine, llm, req)
    return {"event_id": event_id}


@router.get("/events/{event_id}")
async def get_event_status(
    event_id: str,
    engine: AsyncEngine = Depends(get_engine),
):
    status_info = await event_status(engine, event_id)
    if status_info.get("status") == "not_found":
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id!r} not found",
        )
    return status_info
