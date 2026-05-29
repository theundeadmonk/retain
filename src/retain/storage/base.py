"""Async storage engine using SQLAlchemy Core.

Uses SQLAlchemy ORM for model definitions but Core for all queries.
Supports SQLite (default) and PostgreSQL.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from retain.storage.models import Base

__all__ = [
    "create_engine",
    "init_db",
    "drop_db",
]


def create_engine(url: str = "sqlite+aiosqlite:///retain.db") -> AsyncEngine:
    """Create an async engine for the given database URL.

    Args:
        url: SQLAlchemy async database URL.
             Defaults to SQLite (``sqlite+aiosqlite:///retain.db``).
             For PostgreSQL: ``postgresql+asyncpg://user:pass@host/db``

    Returns:
        An async SQLAlchemy engine.
    """
    engine = create_async_engine(url, echo=False)
    return engine


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db(engine: AsyncEngine) -> None:
    """Drop all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
