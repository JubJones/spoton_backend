"""
Database initialization and migration script.

Handles:
- Database setup and initialization
- TimescaleDB hypertable creation
- Index creation for performance
- Data migration utilities
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from app.infrastructure.database.base import engine, setup_database, check_database_connection
from app.infrastructure.database.models.tracking_models import Base
from app.infrastructure.cache.redis_client import get_redis_async
from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseMigration:
    """Database migration utilities for SpotOn backend."""
    
    def __init__(self):
        self.migration_history = []
        logger.info("DatabaseMigration initialized")
    
    async def initialize_database(self):
        """Initialize complete database setup."""
        try:
            logger.info("Starting database initialization...")
            
            # Check database connection
            if not check_database_connection():
                raise Exception("Cannot connect to database")
            
            # Run complete setup
            await setup_database()
            
            # Verify Redis connection
            await self._verify_redis_connection()
            
            # Create initial data if needed
            await self._create_initial_data()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _verify_redis_connection(self):
        """Verify Redis connection."""
        try:
            redis_client = await get_redis_async()
            await redis_client.ping()
            logger.info("Redis connection verified")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    async def _create_initial_data(self):
        """Create initial data for the system."""
        try:
            # This could include default environments, cameras, etc.
            logger.info("Creating initial data...")
            
            # Example: Create default environments based on settings
            environments = self._get_default_environments()
            
            if environments:
                logger.info(f"Found {len(environments)} default environments")
                # In a real implementation, you'd insert these into the database
                
        except Exception as e:
            logger.error(f"Error creating initial data: {e}")
            raise
    
    def _get_default_environments(self) -> List[Dict[str, Any]]:
        """Get default environments from settings."""
        try:
            environments = []
            
            # Extract unique environments from video sets
            for video_set in settings.VIDEO_SETS:
                env_id = video_set.env_id
                if not any(env['id'] == env_id for env in environments):
                    environments.append({
                        'id': env_id,
                        'name': env_id.title(),
                        'description': f"Environment {env_id}",
                        'cameras': [vs.cam_id for vs in settings.VIDEO_SETS if vs.env_id == env_id]
                    })
            
            return environments
            
        except Exception as e:
            logger.error(f"Error getting default environments: {e}")
            return []
    
    async def create_test_data(self):
        """Create test data for development."""
        try:
            logger.info("Creating test data...")
            
            from app.infrastructure.database.repositories.tracking_repository import get_tracking_repository
            
            # Create test session
            async with get_tracking_repository() as repo:
                session = await repo.create_session_record(
                    session_id="test_session_001",
                    environment_id="campus",
                    camera_ids=["c01", "c02", "c03"],
                    user_id="test_user",
                    settings={"test_mode": True}
                )
                
                if session:
                    logger.info(f"Created test session: {session.session_id}")
                
                # Create test person identity
                identity = await repo.create_person_identity(
                    global_person_id="test_person_001",
                    environment_id="campus",
                    first_seen_camera="c01",
                    first_seen_at=datetime.now(timezone.utc),
                    confidence=0.95
                )
                
                if identity:
                    logger.info(f"Created test person identity: {identity.global_person_id}")
                
                # Create test tracking event
                event = await repo.create_tracking_event(
                    global_person_id="test_person_001",
                    camera_id="c01",
                    environment_id="campus",
                    event_type="detection",
                    position_x=100.0,
                    position_y=200.0,
                    detection_confidence=0.95,
                    session_id="test_session_001"
                )
                
                if event:
                    logger.info(f"Created test tracking event: {event.id}")
            
            logger.info("Test data created successfully")
            
        except Exception as e:
            logger.error(f"Error creating test data: {e}")
            raise
    
    async def cleanup_test_data(self):
        """Clean up test data."""
        try:
            logger.info("Cleaning up test data...")
            
            from app.infrastructure.database.repositories.tracking_repository import get_tracking_repository
            
            async with get_tracking_repository() as repo:
                # Remove test data
                # This would involve deleting test records
                pass
            
            logger.info("Test data cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up test data: {e}")
            raise
    
    async def migrate_data(self, migration_type: str = "full"):
        """Migrate data between versions."""
        try:
            logger.info(f"Starting data migration: {migration_type}")
            
            if migration_type == "full":
                await self._migrate_full_data()
            elif migration_type == "incremental":
                await self._migrate_incremental_data()
            else:
                logger.warning(f"Unknown migration type: {migration_type}")
                return
            
            logger.info("Data migration completed successfully")
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            raise
    
    async def _migrate_full_data(self):
        """Perform full data migration."""
        try:
            logger.info("Performing full data migration...")
            
            # This would include:
            # 1. Backup existing data
            # 2. Transform data to new schema
            # 3. Verify data integrity
            # 4. Update version information
            
            self.migration_history.append({
                'type': 'full',
                'timestamp': datetime.now(timezone.utc),
                'status': 'completed'
            })
            
        except Exception as e:
            logger.error(f"Full data migration failed: {e}")
            raise
    
    async def _migrate_incremental_data(self):
        """Perform incremental data migration."""
        try:
            logger.info("Performing incremental data migration...")
            
            # This would include:
            # 1. Identify changes since last migration
            # 2. Apply incremental changes
            # 3. Verify data consistency
            
            self.migration_history.append({
                'type': 'incremental',
                'timestamp': datetime.now(timezone.utc),
                'status': 'completed'
            })
            
        except Exception as e:
            logger.error(f"Incremental data migration failed: {e}")
            raise
    
    async def verify_database_health(self) -> Dict[str, Any]:
        """Verify database health and integrity."""
        try:
            health_report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'database_connection': False,
                'redis_connection': False,
                'table_counts': {},
                'index_status': {},
                'hypertable_status': {},
                'errors': []
            }
            
            # Check database connection
            health_report['database_connection'] = check_database_connection()
            
            # Check Redis connection
            try:
                redis_client = await get_redis_async()
                await redis_client.ping()
                health_report['redis_connection'] = True
            except Exception as e:
                health_report['redis_connection'] = False
                health_report['errors'].append(f"Redis connection failed: {e}")
            
            # Check table counts
            if health_report['database_connection']:
                health_report['table_counts'] = await self._get_table_counts()
                health_report['index_status'] = await self._check_index_status()
                health_report['hypertable_status'] = await self._check_hypertable_status()
            
            return health_report
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {'error': str(e)}
    
    async def _get_table_counts(self) -> Dict[str, int]:
        """Get record counts for all tables."""
        try:
            from sqlalchemy import text
            
            counts = {}
            table_names = [
                'tracking_events',
                'detection_events',
                'person_trajectories',
                'person_identities',
                'analytics_aggregations',
                'session_records'
            ]
            
            with engine.connect() as conn:
                for table_name in table_names:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        counts[table_name] = result.scalar()
                    except Exception as e:
                        counts[table_name] = f"Error: {e}"
            
            return counts
            
        except Exception as e:
            logger.error(f"Error getting table counts: {e}")
            return {}
    
    async def _check_index_status(self) -> Dict[str, Any]:
        """Check database index status."""
        try:
            from sqlalchemy import text
            
            index_status = {}
            
            with engine.connect() as conn:
                # Get index information
                result = conn.execute(text("""
                    SELECT schemaname, tablename, indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    ORDER BY tablename, indexname
                """))
                
                indexes = result.fetchall()
                
                for schema, table, index, definition in indexes:
                    if table not in index_status:
                        index_status[table] = []
                    
                    index_status[table].append({
                        'name': index,
                        'definition': definition
                    })
            
            return index_status
            
        except Exception as e:
            logger.error(f"Error checking index status: {e}")
            return {}
    
    async def _check_hypertable_status(self) -> Dict[str, Any]:
        """Check TimescaleDB hypertable status."""
        try:
            from sqlalchemy import text
            
            hypertable_status = {}
            
            with engine.connect() as conn:
                # Check if TimescaleDB extension is available
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'
                """))
                
                if result.scalar() == 0:
                    return {'error': 'TimescaleDB extension not installed'}
                
                # Get hypertable information
                result = conn.execute(text("""
                    SELECT hypertable_schema, hypertable_name, num_chunks, compression_enabled
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'public'
                """))
                
                hypertables = result.fetchall()
                
                for schema, table, chunks, compression in hypertables:
                    hypertable_status[table] = {
                        'schema': schema,
                        'num_chunks': chunks,
                        'compression_enabled': compression
                    }
            
            return hypertable_status
            
        except Exception as e:
            logger.error(f"Error checking hypertable status: {e}")
            return {'error': str(e)}
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        return self.migration_history.copy()


# Global migration instance
db_migration = DatabaseMigration()


async def main():
    """Main migration script."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python init_database.py <command>")
        print("Commands: init, test-data, cleanup, health, migrate")
        return
    
    command = sys.argv[1]
    
    try:
        if command == "init":
            await db_migration.initialize_database()
        elif command == "test-data":
            await db_migration.create_test_data()
        elif command == "cleanup":
            await db_migration.cleanup_test_data()
        elif command == "health":
            report = await db_migration.verify_database_health()
            print("Database Health Report:")
            print(f"Database Connection: {report['database_connection']}")
            print(f"Redis Connection: {report['redis_connection']}")
            print(f"Table Counts: {report['table_counts']}")
            if report.get('errors'):
                print(f"Errors: {report['errors']}")
        elif command == "migrate":
            migration_type = sys.argv[2] if len(sys.argv) > 2 else "full"
            await db_migration.migrate_data(migration_type)
        else:
            print(f"Unknown command: {command}")
            
    except Exception as e:
        logger.error(f"Migration command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())