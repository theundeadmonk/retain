import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from retain.models import Base
from retain.settings import settings


class _MockEmbeddingProvider:
    """Lightweight embedding provider for integration tests.

    Produces non-random vectors so the same text always maps to the same
    vector, making search results reproducible.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    @property
    def dim(self) -> int:
        return self._dim

    async def encode_query(self, text: str) -> list[float]:
        return self._text_to_vec(text)

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    def encode_query_sync(self, text: str) -> list[float]:
        return self._text_to_vec(text)

    def _encode_sync_impl(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    def _text_to_vec(self, text: str) -> list[float]:
        text = text.lower()
        base = [0.0] * self._dim
        for i, ch in enumerate(text):
            base[i % self._dim] += ord(ch) / 256.0
        norm = sum(v * v for v in base) ** 0.5 or 1.0
        return [v / norm for v in base]


@pytest.fixture
async def engine():
    engine = create_async_engine(settings.test_database_url, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def embedding_provider() -> _MockEmbeddingProvider:
    return _MockEmbeddingProvider(dim=1024)


@pytest.fixture
async def app(engine, embedding_provider):
    from fastapi import FastAPI

    from retain.routes import router as v1_router

    app = FastAPI(
        title="Retain-Test",
        description="Test server",
        version="0.0.0",
    )
    app.state.engine = engine
    app.state.llm = None
    app.state.embedding_provider = embedding_provider
    app.state.sparse_provider = None
    app.include_router(v1_router, prefix="/v1")
    return app


@pytest.fixture
async def client(app):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        yield ac
