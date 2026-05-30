"""FastAPI dependencies."""

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.llm.base import LLMProvider


async def get_engine(request: Request) -> AsyncEngine:
    return request.app.state.engine


async def get_llm(request: Request) -> LLMProvider | None:
    return request.app.state.llm
