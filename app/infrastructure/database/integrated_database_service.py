"""
Integrated database service combining Redis and TimescaleDB.

Handles:
- Real-time data flow from Redis to TimescaleDB
- Unified data access layer
- Performance optimization
- Data consistency management
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

from app.infrastructure.cache.tracking_cache import tracking_cache, CachedPersonState
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.infrastructure.database.base import get_db, check_database_connection
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.reid.entities.feature_vector import FeatureVector
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.mapping.entities.trajectory import Trajectory
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


@dataclass
class DataSyncStats:
    """Statistics for data synchronization."""
    redis_writes: int = 0
    db_writes: int = 0
    sync_operations: int = 0
    sync_failures: int = 0
    last_sync_time: Optional[datetime] = None


class IntegratedDatabaseService:
    """
    Integrated database service for Redis and TimescaleDB.
    
    Features:
    - Real-time caching with Redis
    - Persistent storage with TimescaleDB
    - Automatic data synchronization
    - Performance optimization
    """
    
    def __init__(self):
        self.sync_stats = DataSyncStats()
        self.sync_enabled = True
        self.sync_interval = 60  # seconds
        self.sync_task: Optional[asyncio.Task] = None
        self.batch_size = 100
        
        # Performance settings
        self.cache_ttl = 300  # 5 minutes
        self.sync_retry_attempts = 3
        self.sync_retry_delay = 5  # seconds
        
        logger.info("IntegratedDatabaseService initialized")
    
    async def initialize(self):
        """Initialize the integrated database service."""
        try:
            # Initialize Redis cache
            await tracking_cache.initialize()
            
            # Check database connection
            if not check_database_connection():
                raise Exception("Database connection failed")
            
            # Start sync task
            if self.sync_enabled:
                await self.start_sync_task()
            
            logger.info("IntegratedDatabaseService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing IntegratedDatabaseService: {e}")
            raise
    
    async def cleanup(self):
        """Clean up the integrated database service."""
        try:
            # Stop sync task
            if self.sync_task:
                self.sync_task.cancel()
                try:
                    await self.sync_task
                except asyncio.CancelledError:
                    pass
                self.sync_task = None
            
            # Clean up Redis cache
            await tracking_cache.cleanup()
            
            logger.info("IntegratedDatabaseService cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up IntegratedDatabaseService: {e}")
    
    @asynccontextmanager
    async def get_repository(self):
        """Get database repository with context management."""
        db = next(get_db())
        try:
            yield TrackingRepository(db)
        finally:
            db.close()
    
    # Person State Management
    async def store_person_state(
        self,
        person_identity: PersonIdentity,
        camera_id: CameraID,
        position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """Store person state in both Redis and TimescaleDB."""
        try:
            # Store in Redis for real-time access
            cache_success = await tracking_cache.cache_person_state(
                person_identity, camera_id, position, trajectory
            )
            
            # Store in TimescaleDB for persistence
            db_success = await self._store_person_state_db(
                person_identity, camera_id, position, trajectory, session_id
            )
            
            if cache_success:
                self.sync_stats.redis_writes += 1
            if db_success:
                self.sync_stats.db_writes += 1
            
            return cache_success and db_success
            
        except Exception as e:
            logger.error(f"Error storing person state: {e}")
            return False
    
    async def get_person_state(
        self,
        global_person_id: str,
        prefer_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get person state from cache or database."""
        try:
            if prefer_cache:
                # Try cache first
                cached_state = await tracking_cache.get_person_state(global_person_id)
                if cached_state:
                    return {
                        'global_id': cached_state.global_id,
                        'last_seen_camera': cached_state.last_seen_camera,
                        'last_seen_time': cached_state.last_seen_time,
                        'current_position': cached_state.current_position,
                        'confidence': cached_state.confidence,
                        'track_id': cached_state.track_id,
                        'is_active': cached_state.is_active,
                        'source': 'cache'
                    }
            
            # Fall back to database
            async with self.get_repository() as repo:
                identity = await repo.get_person_identity(global_person_id)
                if identity:
                    return {
                        'global_id': identity.global_person_id,
                        'last_seen_camera': identity.last_seen_camera,
                        'last_seen_time': identity.last_seen_at,
                        'current_position': None,  # Not stored in identity table
                        'confidence': identity.confidence,
                        'track_id': None,  # Not stored in identity table
                        'is_active': identity.is_active,
                        'source': 'database'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting person state: {e}")
            return None
    
    async def get_active_persons(
        self,
        camera_id: Optional[CameraID] = None,
        environment_id: Optional[str] = None,
        prefer_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get active persons from cache or database."""
        try:
            if prefer_cache:
                # Try cache first
                cached_persons = await tracking_cache.get_active_persons(camera_id)
                if cached_persons:
                    return [
                        {
                            'global_id': person.global_id,
                            'last_seen_camera': person.last_seen_camera,
                            'last_seen_time': person.last_seen_time,
                            'current_position': person.current_position,
                            'confidence': person.confidence,
                            'track_id': person.track_id,
                            'is_active': person.is_active,
                            'source': 'cache'
                        }
                        for person in cached_persons
                    ]
            
            # Fall back to database
            async with self.get_repository() as repo:
                persons = await repo.get_active_persons(
                    environment_id=environment_id,
                    camera_id=str(camera_id) if camera_id else None,
                    since=datetime.now(timezone.utc) - timedelta(minutes=5)
                )
                
                return [
                    {
                        'global_id': person.global_person_id,
                        'last_seen_camera': person.last_seen_camera,
                        'last_seen_time': person.last_seen_at,
                        'current_position': None,
                        'confidence': person.confidence,
                        'track_id': None,
                        'is_active': person.is_active,
                        'source': 'database'
                    }
                    for person in persons
                ]
            
        except Exception as e:
            logger.error(f"Error getting active persons: {e}")
            return []
    
    # Detection and Tracking Events
    async def store_detection_event(
        self,
        camera_id: str,
        environment_id: str,
        bbox_data: Dict[str, float],
        confidence: float,
        frame_number: Optional[int] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store detection event in database."""
        try:
            async with self.get_repository() as repo:
                event = await repo.create_detection_event(
                    camera_id=camera_id,
                    environment_id=environment_id,
                    bbox_x1=bbox_data['x1'],
                    bbox_y1=bbox_data['y1'],
                    bbox_x2=bbox_data['x2'],
                    bbox_y2=bbox_data['y2'],
                    confidence=confidence,
                    frame_number=frame_number,
                    session_id=session_id,
                    metadata=metadata
                )
                
                self.sync_stats.db_writes += 1
                return event is not None
                
        except Exception as e:
            logger.error(f"Error storing detection event: {e}")
            return False
    
    async def store_tracking_event(
        self,
        global_person_id: str,
        camera_id: str,
        environment_id: str,
        event_type: str,
        position: Optional[Coordinate] = None,
        bbox_data: Optional[Dict[str, float]] = None,
        detection_confidence: Optional[float] = None,
        reid_confidence: Optional[float] = None,
        track_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store tracking event in database."""
        try:
            async with self.get_repository() as repo:
                event = await repo.create_tracking_event(
                    global_person_id=global_person_id,
                    camera_id=camera_id,
                    environment_id=environment_id,
                    event_type=event_type,
                    position_x=position.x if position else None,
                    position_y=position.y if position else None,
                    world_x=position.world_x if position else None,
                    world_y=position.world_y if position else None,
                    bbox_data=bbox_data,
                    detection_confidence=detection_confidence,
                    reid_confidence=reid_confidence,
                    track_id=track_id,
                    session_id=session_id,
                    metadata=metadata
                )
                
                self.sync_stats.db_writes += 1
                return event is not None
                
        except Exception as e:
            logger.error(f"Error storing tracking event: {e}")
            return False
    
    # Trajectory Management
    async def store_trajectory_point(
        self,
        global_person_id: str,
        camera_id: str,
        environment_id: str,
        sequence_number: int,
        position: Coordinate,
        velocity: Optional[Tuple[float, float]] = None,
        acceleration: Optional[Tuple[float, float]] = None,
        confidence: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store trajectory point in database."""
        try:
            async with self.get_repository() as repo:
                point = await repo.create_trajectory_point(
                    global_person_id=global_person_id,
                    camera_id=camera_id,
                    environment_id=environment_id,
                    sequence_number=sequence_number,
                    position_x=position.x,
                    position_y=position.y,
                    world_x=position.world_x,
                    world_y=position.world_y,
                    velocity_x=velocity[0] if velocity else None,
                    velocity_y=velocity[1] if velocity else None,
                    acceleration_x=acceleration[0] if acceleration else None,
                    acceleration_y=acceleration[1] if acceleration else None,
                    confidence=confidence,
                    session_id=session_id,
                    metadata=metadata
                )
                
                self.sync_stats.db_writes += 1
                return point is not None
                
        except Exception as e:
            logger.error(f"Error storing trajectory point: {e}")
            return False
    
    async def get_person_trajectory(
        self,
        global_person_id: str,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get person trajectory from database."""
        try:
            async with self.get_repository() as repo:
                trajectory_points = await repo.get_person_trajectory(
                    global_person_id=global_person_id,
                    camera_id=camera_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                
                return [
                    {
                        'sequence_number': point.sequence_number,
                        'position_x': point.position_x,
                        'position_y': point.position_y,
                        'world_x': point.world_x,
                        'world_y': point.world_y,
                        'velocity_x': point.velocity_x,
                        'velocity_y': point.velocity_y,
                        'acceleration_x': point.acceleration_x,
                        'acceleration_y': point.acceleration_y,
                        'confidence': point.confidence,
                        'timestamp': point.timestamp,
                        'camera_id': point.camera_id,
                        'smoothed': point.smoothed
                    }
                    for point in trajectory_points
                ]
                
        except Exception as e:
            logger.error(f"Error getting person trajectory: {e}")
            return []
    
    # Session Management
    async def create_session(
        self,
        session_id: str,
        environment_id: str,
        camera_ids: List[CameraID],
        user_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new session in both cache and database."""
        try:
            # Create in cache
            cache_success = await tracking_cache.create_session(
                session_id=session_id,
                camera_ids=camera_ids,
                user_id=user_id,
                settings=settings
            )
            
            # Create in database
            async with self.get_repository() as repo:
                session = await repo.create_session_record(
                    session_id=session_id,
                    environment_id=environment_id,
                    camera_ids=[str(cam_id) for cam_id in camera_ids],
                    user_id=user_id,
                    settings=settings
                )
                db_success = session is not None
            
            if cache_success:
                self.sync_stats.redis_writes += 1
            if db_success:
                self.sync_stats.db_writes += 1
            
            return cache_success and db_success
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    async def end_session(
        self,
        session_id: str,
        total_persons_tracked: Optional[int] = None,
        total_detections: Optional[int] = None,
        total_events: Optional[int] = None
    ) -> bool:
        """End session in both cache and database."""
        try:
            # End in cache
            cache_success = await tracking_cache.end_session(session_id)
            
            # Update in database
            async with self.get_repository() as repo:
                db_success = await repo.update_session_record(
                    session_id=session_id,
                    end_time=datetime.now(timezone.utc),
                    status='completed',
                    total_persons_tracked=total_persons_tracked,
                    total_detections=total_detections,
                    total_events=total_events
                )
            
            return cache_success and db_success
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False
    
    # Analytics and Statistics
    async def get_detection_statistics(
        self,
        environment_id: str,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detection statistics from database."""
        try:
            async with self.get_repository() as repo:
                return await repo.get_detection_statistics(
                    environment_id=environment_id,
                    camera_id=camera_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
        except Exception as e:
            logger.error(f"Error getting detection statistics: {e}")
            return {}
    
    async def get_person_statistics(
        self,
        environment_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get person statistics from database."""
        try:
            async with self.get_repository() as repo:
                return await repo.get_person_statistics(
                    environment_id=environment_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
        except Exception as e:
            logger.error(f"Error getting person statistics: {e}")
            return {}
    
    # Data Synchronization
    async def start_sync_task(self):
        """Start the data synchronization task."""
        try:
            if self.sync_task is None:
                self.sync_task = asyncio.create_task(self._sync_loop())
                logger.info("Data synchronization task started")
        except Exception as e:
            logger.error(f"Error starting sync task: {e}")
    
    async def _sync_loop(self):
        """Main synchronization loop."""
        try:
            while True:
                await self._sync_cache_to_database()
                await asyncio.sleep(self.sync_interval)
                
        except asyncio.CancelledError:
            logger.info("Sync loop cancelled")
        except Exception as e:
            logger.error(f"Error in sync loop: {e}")
    
    async def _sync_cache_to_database(self):
        """Sync cache data to database."""
        try:
            # This is a simplified sync - in production, you'd want more sophisticated logic
            # to track which data needs to be synced
            
            # Clean up expired entries
            await tracking_cache.cleanup_expired_entries()
            
            self.sync_stats.sync_operations += 1
            self.sync_stats.last_sync_time = datetime.now(timezone.utc)
            
            logger.debug("Cache to database sync completed")
            
        except Exception as e:
            logger.error(f"Error syncing cache to database: {e}")
            self.sync_stats.sync_failures += 1
    
    async def _store_person_state_db(
        self,
        person_identity: PersonIdentity,
        camera_id: CameraID,
        position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """Store person state in database."""
        try:
            async with self.get_repository() as repo:
                # Check if person identity exists
                existing_identity = await repo.get_person_identity(person_identity.global_id)
                
                if existing_identity:
                    # Update existing identity
                    success = await repo.update_person_identity(
                        global_person_id=person_identity.global_id,
                        last_seen_camera=str(camera_id),
                        last_seen_at=datetime.now(timezone.utc),
                        new_embedding=person_identity.feature_vector.vector.tolist() if person_identity.feature_vector else None,
                        confidence=person_identity.confidence
                    )
                else:
                    # Create new identity
                    identity = await repo.create_person_identity(
                        global_person_id=person_identity.global_id,
                        environment_id="default",  # You'd get this from context
                        first_seen_camera=str(camera_id),
                        first_seen_at=datetime.now(timezone.utc),
                        primary_embedding=person_identity.feature_vector.vector.tolist() if person_identity.feature_vector else None,
                        confidence=person_identity.confidence
                    )
                    success = identity is not None
                
                # Store tracking event
                if success:
                    await repo.create_tracking_event(
                        global_person_id=person_identity.global_id,
                        camera_id=str(camera_id),
                        environment_id="default",
                        event_type="detection",
                        position_x=position.x if position else None,
                        position_y=position.y if position else None,
                        world_x=position.world_x if position else None,
                        world_y=position.world_y if position else None,
                        detection_confidence=person_identity.confidence,
                        reid_confidence=person_identity.feature_vector.confidence if person_identity.feature_vector else None,
                        track_id=person_identity.track_id,
                        session_id=session_id
                    )
                
                return success
                
        except Exception as e:
            logger.error(f"Error storing person state in database: {e}")
            return False
    
    # Performance and Maintenance
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from both cache and database."""
        try:
            # Clean up cache
            await tracking_cache.cleanup_expired_entries()
            
            # Clean up database
            async with self.get_repository() as repo:
                cleanup_result = await repo.cleanup_old_data(days_to_keep)
                
                logger.info(f"Cleaned up old data: {cleanup_result}")
                return cleanup_result
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {}
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        try:
            # Get cache statistics
            cache_stats = await tracking_cache.get_cache_stats()
            
            # Get database connection status
            db_connected = check_database_connection()
            
            return {
                'cache_stats': cache_stats,
                'sync_stats': {
                    'redis_writes': self.sync_stats.redis_writes,
                    'db_writes': self.sync_stats.db_writes,
                    'sync_operations': self.sync_stats.sync_operations,
                    'sync_failures': self.sync_stats.sync_failures,
                    'last_sync_time': self.sync_stats.last_sync_time.isoformat() if self.sync_stats.last_sync_time else None
                },
                'database_status': {
                    'connected': db_connected,
                    'sync_enabled': self.sync_enabled,
                    'sync_interval': self.sync_interval
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting service statistics: {e}")
            return {'error': str(e)}
    
    def enable_sync(self, enabled: bool):
        """Enable or disable data synchronization."""
        self.sync_enabled = enabled
        if enabled and self.sync_task is None:
            asyncio.create_task(self.start_sync_task())
        elif not enabled and self.sync_task:
            self.sync_task.cancel()
            self.sync_task = None
        
        logger.info(f"Data synchronization {'enabled' if enabled else 'disabled'}")
    
    def set_sync_interval(self, interval: int):
        """Set data synchronization interval."""
        self.sync_interval = interval
        logger.info(f"Sync interval set to {interval} seconds")


# Global integrated database service instance
integrated_db_service = IntegratedDatabaseService()