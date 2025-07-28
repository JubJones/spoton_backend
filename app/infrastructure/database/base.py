"""SQLAlchemy base configuration for TimescaleDB."""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create database engine with optimized settings for TimescaleDB
engine = create_engine(
    settings.DATABASE_URL or f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG,
    # TimescaleDB optimizations
    connect_args={
        "options": "-c timezone=utc",
        "application_name": "spoton_backend"
    }
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

def get_db():
    """Dependency to get database session."""
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
            PersonIdentity, AnalyticsAggregation, SessionRecord
        )
        
        # Create tables
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
        with engine.connect() as conn:
            # Create hypertables for time-series tables
            hypertable_queries = [
                "SELECT create_hypertable('tracking_events', 'timestamp', if_not_exists => TRUE);",
                "SELECT create_hypertable('detection_events', 'timestamp', if_not_exists => TRUE);",
                "SELECT create_hypertable('person_trajectories', 'timestamp', if_not_exists => TRUE);",
                "SELECT create_hypertable('analytics_aggregations', 'time_bucket', if_not_exists => TRUE);"
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
        with engine.connect() as conn:
            # Additional indexes for performance
            index_queries = [
                # Composite indexes for common queries
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracking_events_person_camera_time ON tracking_events(global_person_id, camera_id, timestamp);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_events_camera_confidence ON detection_events(camera_id, confidence, timestamp);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_person_trajectories_person_seq ON person_trajectories(global_person_id, sequence_number, timestamp);",
                
                # GIN indexes for JSON columns
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracking_events_metadata_gin ON tracking_events USING GIN(metadata);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_events_metadata_gin ON detection_events USING GIN(metadata);",
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
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except SQLAlchemyError as e:
        logger.error(f"Database connection check failed: {e}")
        return False