"""Retain FastAPI server."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from retain.llm.openai import OpenAIProvider
from retain.routes import router as v1_router
from retain.settings import settings
from retain.storage import create_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(settings.database_url)

    app.state.engine = engine
    app.state.llm = None
    if settings.llm_api_key and settings.llm_base_url:
        app.state.llm = OpenAIProvider(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
    yield
    await engine.dispose()


app = FastAPI(
    title="Retain",
    description="The memory layer built for real-time agents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router, prefix="/v1")
