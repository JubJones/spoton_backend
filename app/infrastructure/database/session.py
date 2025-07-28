"""Database session management utilities."""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy.orm import Session
from app.infrastructure.database.base import SessionLocal
import logging

logger = logging.getLogger(__name__)

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all database tables."""
    from app.infrastructure.database.base import engine, Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")

def drop_tables():
    """Drop all database tables."""
    from app.infrastructure.database.base import engine, Base
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped successfully")