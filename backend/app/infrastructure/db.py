"""
Async SQLAlchemy engine and session factory for PostgreSQL.

Provides the async engine singleton, a session maker, and a FastAPI
dependency that yields a scoped ``AsyncSession`` per request.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from ..config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine():
    """Return the global async engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Set the DATABASE_URL environment variable to a valid "
            "postgresql+asyncpg:// connection string."
        )

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        # SSL is configured via ?ssl=require in the DATABASE_URL
    )
    logger.info("Async SQLAlchemy engine created")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return (and cache) the global async session factory."""
    global _async_session_factory
    if _async_session_factory is not None:
        return _async_session_factory

    _async_session_factory = async_sessionmaker(
        bind=_get_engine(),
        expire_on_commit=False,
    )
    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a scoped async session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables. Call once at application startup."""
    from ..models.metadata import FileMetadata  # noqa: F401
    from ..models.db_models import JobRecord, JobChunk  # noqa: F401

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


async def close_db() -> None:
    """Dispose of the engine connection pool."""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database engine disposed")
