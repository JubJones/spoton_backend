"""SQLAlchemy base configuration for TimescaleDB."""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings

logger = logging.getLogger(__name__)

_ENGINE = None
_SESSION_LOCAL = None

def _build_sync_db_url() -> str:
    return settings.DATABASE_URL or (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

def get_engine():
    """Lazily create and return the SQLAlchemy engine if DB is enabled."""
    global _ENGINE
    if not settings.DB_ENABLED:
        logger.info("Database disabled by configuration (DB_ENABLED=false).")
        return None
    if _ENGINE is None:
        _ENGINE = create_engine(
            _build_sync_db_url(),
            poolclass=QueuePool,
            pool_size=int(getattr(settings, 'DB_POOL_SIZE', 20)),
            max_overflow=int(getattr(settings, 'DB_MAX_OVERFLOW', 30)),
            pool_pre_ping=bool(getattr(settings, 'DB_POOL_PRE_PING', True)),
            pool_recycle=int(getattr(settings, 'DB_POOL_RECYCLE', 3600)),
            echo=settings.DEBUG,
            connect_args={
                "options": "-c timezone=utc",
                "application_name": "spoton_backend"
            }
        )
    return _ENGINE

def get_session_factory():
    """Lazily create session factory if DB enabled."""
    global _SESSION_LOCAL
    engine = get_engine()
    if engine is None:
        return None
    if _SESSION_LOCAL is None:
        _SESSION_LOCAL = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SESSION_LOCAL

# Create declarative base
Base = declarative_base()

def get_db():
    """Dependency to get database session (raises if DB disabled)."""
    SessionLocal = get_session_factory()
    if SessionLocal is None:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database disabled")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def create_tables():
    """Create all database tables."""
    try:
        # Import models to register them
        from app.infrastructure.database.models.tracking_models import (
            TrackingEvent, DetectionEvent, PersonTrajectory, 
            PersonIdentity, AnalyticsAggregation, SessionRecord,
            AnalyticsTotals, AnalyticsUptimeDaily, GeometricMetricsEvent
        )
        
        # Create tables
        engine = get_engine()
        if engine is None:
            logger.info("DB setup skipped (DB disabled)")
            return
        Base.metadata.create_all(bind=engine)
        
        # Create TimescaleDB hypertables
        await create_hypertables()
        
        logger.info("Database tables created successfully")
        
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {e}")
        raise

async def create_hypertables():
    """Create TimescaleDB hypertables for time-series data."""
    try:
        engine = get_engine()
        if engine is None:
            logger.info("Hypertables creation skipped (DB disabled)")
            return
        with engine.connect() as conn:
            # Create hypertables for time-series tables
            hypertable_queries = [
                "SELECT create_hypertable('tracking_events', 'timestamp', if_not_exists => TRUE);",
                "SELECT create_hypertable('detection_events', 'timestamp', if_not_exists => TRUE);",
                "SELECT create_hypertable('person_trajectories', 'timestamp', if_not_exists => TRUE);",
                "SELECT create_hypertable('analytics_aggregations', 'time_bucket', if_not_exists => TRUE);",
                "SELECT create_hypertable('analytics_totals', 'bucket_start', if_not_exists => TRUE);"
            ]
            
            for query in hypertable_queries:
                try:
                    conn.execute(text(query))
                    logger.debug(f"Executed hypertable query: {query}")
                except SQLAlchemyError as e:
                    # Hypertable might already exist
                    if "already exists" not in str(e):
                        logger.warning(f"Error creating hypertable: {e}")
            
            conn.commit()
            logger.info("TimescaleDB hypertables created successfully")
            
    except SQLAlchemyError as e:
        logger.error(f"Error creating hypertables: {e}")
        raise

async def create_indexes():
    """Create additional database indexes for performance."""
    try:
        engine = get_engine()
        if engine is None:
            logger.info("Index creation skipped (DB disabled)")
            return
        # Use AUTOCOMMIT for concurrent index creation
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            # Additional indexes for performance
            index_queries = [
                # Composite indexes for common queries
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracking_events_person_camera_time ON tracking_events(global_person_id, camera_id, timestamp);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_events_camera_confidence ON detection_events(camera_id, confidence, timestamp);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_person_trajectories_person_seq ON person_trajectories(global_person_id, sequence_number, timestamp);",
                
                # GIN indexes for JSON columns
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracking_events_metadata_gin ON tracking_events USING GIN(event_metadata);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_events_metadata_gin ON detection_events USING GIN(detection_metadata);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_person_identities_embedding_gin ON person_identities USING GIN(primary_embedding);",
                
                # Performance indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_person_identities_last_seen_active ON person_identities(last_seen_at, is_active);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_session_records_environment_status ON session_records(environment_id, status, start_time);"
            ]
            
            for query in index_queries:
                try:
                    conn.execute(text(query))
                    logger.debug(f"Created index: {query}")
                except SQLAlchemyError as e:
                    # Index might already exist
                    if "already exists" not in str(e):
                        logger.warning(f"Error creating index: {e}")
            
            conn.commit()
            logger.info("Additional database indexes created successfully")
            
    except SQLAlchemyError as e:
        logger.error(f"Error creating indexes: {e}")
        raise

async def setup_database():
    """Complete database setup including tables, hypertables, and indexes."""
    try:
        await create_tables()
        await create_indexes()
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise

def check_database_connection():
    """Check database connection health."""
    try:
        engine = get_engine()
        if engine is None:
            return False
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except SQLAlchemyError as e:
        logger.error(f"Database connection check failed: {e}")
        return False
