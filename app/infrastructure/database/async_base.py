"""Async SQLAlchemy base configuration for TimescaleDB.

Provides an async engine and session factory to avoid blocking the event loop
on database I/O, used by hot write paths.
"""

import logging
from typing import AsyncIterator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text

from app.core.config import settings

logger = logging.getLogger(__name__)


def _build_async_db_url() -> str:
    if settings.DATABASE_URL:
        # normalize scheme to asyncpg
        if "+" in settings.DATABASE_URL:
            # already has a driver
            return settings.DATABASE_URL.replace("postgresql+psycopg2", "postgresql+asyncpg").replace(
                "postgresql+pg8000", "postgresql+asyncpg"
            ).replace("postgresql://", "postgresql+asyncpg://")
        return settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    return (
        f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


_ASYNC_ENGINE = None
_ASYNC_SESSION_LOCAL = None

def get_async_engine():
    """Lazily create async engine if DB is enabled."""
    global _ASYNC_ENGINE
    if not settings.DB_ENABLED:
        logger.info("Async DB disabled by configuration (DB_ENABLED=false).")
        return None
    if _ASYNC_ENGINE is None:
        _ASYNC_ENGINE = create_async_engine(
            _build_async_db_url(),
            pool_pre_ping=bool(getattr(settings, 'DB_POOL_PRE_PING', True)),
            pool_size=int(getattr(settings, 'DB_POOL_SIZE', 20)),
            max_overflow=int(getattr(settings, 'DB_MAX_OVERFLOW', 30)),
            echo=settings.DEBUG,
        )
    return _ASYNC_ENGINE

def get_async_session_factory():
    global _ASYNC_SESSION_LOCAL
    engine = get_async_engine()
    if engine is None:
        return None
    if _ASYNC_SESSION_LOCAL is None:
        _ASYNC_SESSION_LOCAL = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    return _ASYNC_SESSION_LOCAL


@asynccontextmanager
async def get_async_db() -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession for use with async repositories."""
    session: Optional[AsyncSession] = None
    try:
        SessionFactory = get_async_session_factory()
        if SessionFactory is None:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database disabled")
        session = SessionFactory()
        yield session
    finally:
        if session is not None:
            await session.close()


async def check_async_database_connection() -> bool:
    try:
        engine = get_async_engine()
        if engine is None:
            return False
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.scalar()
            return row == 1
    except Exception as e:
        logger.error(f"Async DB connection check failed: {e}")
        return False
