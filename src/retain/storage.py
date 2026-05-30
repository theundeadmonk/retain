"""Async storage engine using SQLAlchemy Core + Alembic migrations.

Primary backend is PostgreSQL + pgvector. Migrations run via CLI before
server start: ``alembic upgrade head``.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from retain.models import Base

__all__ = [
    "create_engine",
    "drop_db",
    "init_db",
]

_DEFAULT_URL = "postgresql+asyncpg://retain:retain@localhost:5432/retain"


def create_engine(url: str = _DEFAULT_URL) -> AsyncEngine:
    """Create an async engine for the given database URL."""
    return create_async_engine(url, echo=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables via metadata.create_all (for tests only)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    if engine.dialect.name == "postgresql":
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def drop_db(engine: AsyncEngine) -> None:
    """Drop all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
