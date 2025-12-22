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
from app.infrastructure.database.repositories.analytics_totals_repository import (
    AnalyticsTotalsRepository,
    DEFAULT_BUCKET_SECONDS,
)
from app.infrastructure.database.repositories.geometric_metrics_repository import GeometricMetricsRepository
from app.infrastructure.database.base import get_db, get_session_factory, check_database_connection
from typing import Any
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

    @asynccontextmanager
    async def get_analytics_repository(self):
        """Context manager yielding analytics totals repository when DB enabled."""
        session_factory = get_session_factory()
        if session_factory is None:
            yield None
            return

        db = session_factory()
        try:
            yield AnalyticsTotalsRepository(db)
        finally:
            db.close()
    
    @asynccontextmanager
    async def get_geometric_metrics_repository(self):
        """Context manager yielding geometric metrics repository when DB enabled."""
        session_factory = get_session_factory()
        if session_factory is None:
            yield None
            return

        db = session_factory()
        try:
            yield GeometricMetricsRepository(db)
        finally:
            db.close()
    
    # Person State Management
    async def store_person_state(
        self,
        person_identity: Any,
        camera_id: CameraID,
        position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Store person state in both Redis and TimescaleDB."""
        try:
            # Store in Redis for real-time access
            cache_success = await tracking_cache.cache_person_state(person_identity, camera_id, position, trajectory)
            
            # Store in TimescaleDB for persistence
            db_success = await self._store_person_state_db(person_identity, camera_id, position, trajectory, session_id)
            
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
                if event is not None:
                    await self._aggregate_detection_event(
                        environment_id=environment_id,
                        camera_id=camera_id,
                        confidence=confidence,
                        event_timestamp=datetime.now(timezone.utc),
                        detections=1,
                        unique_entities=1,
                    )
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

    async def _aggregate_detection_event(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        confidence: Optional[float],
        event_timestamp: Optional[datetime],
        detections: int,
        unique_entities: int,
        bucket_size_seconds: int = DEFAULT_BUCKET_SECONDS,
    ) -> None:
        """Persist aggregated detection metrics for dashboard consumption."""
        if detections <= 0:
            return

        timestamp = event_timestamp or datetime.now(timezone.utc)
        confidence_value = float(confidence) if confidence is not None else 0.0
        confidence_samples = detections if confidence is not None else 0

        try:
            async with self.get_analytics_repository() as repo:
                if repo is None:
                    return

                repo.increment_detection_totals(
                    environment_id=environment_id,
                    camera_id=camera_id,
                    event_time=timestamp,
                    detections=detections,
                    unique_entities=unique_entities,
                    confidence_sum=confidence_value * detections,
                    confidence_samples=confidence_samples,
                    bucket_size_seconds=bucket_size_seconds,
                )

                repo.increment_detection_totals(
                    environment_id=environment_id,
                    camera_id=None,
                    event_time=timestamp,
                    detections=detections,
                    unique_entities=unique_entities,
                    confidence_sum=confidence_value * detections,
                    confidence_samples=confidence_samples,
                    bucket_size_seconds=bucket_size_seconds,
                )

                # logger.debug(
                #     "Aggregated detection totals",
                #     extra={
                #         "environment_id": environment_id,
                #         "camera_id": camera_id,
                #         "detections": detections,
                #         "unique_entities": unique_entities,
                #         "avg_confidence": confidence,
                #         "bucket_size_seconds": bucket_size_seconds,
                #         "event_timestamp": timestamp.isoformat(),
                #     },
                # )

        except Exception as exc:
            logger.warning(f"Failed to aggregate detection metrics: {exc}")

    async def record_uptime_snapshot(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        uptime_percent: float,
        snapshot_day: Optional[datetime] = None,
        samples: int = 1,
    ) -> None:
        """Record uptime statistics for later dashboard retrieval."""
        day = (snapshot_day or datetime.now(timezone.utc)).date()
        try:
            async with self.get_analytics_repository() as repo:
                if repo is None:
                    return
                repo.upsert_uptime_snapshot(
                    environment_id=environment_id,
                    camera_id=camera_id,
                    day=day,
                    uptime_percent=uptime_percent,
                    samples=samples,
                )
        except Exception as exc:
            logger.warning(f"Failed to record uptime snapshot: {exc}")

    async def record_detection_batch(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        detections: int,
        unique_entities: int,
        average_confidence: Optional[float],
        event_timestamp: Optional[datetime] = None,
        bucket_size_seconds: int = DEFAULT_BUCKET_SECONDS,
    ) -> None:
        """Public helper to aggregate detection batches without persisting raw events."""
        await self._aggregate_detection_event(
            environment_id=environment_id,
            camera_id=camera_id,
            confidence=average_confidence,
            event_timestamp=event_timestamp,
            detections=detections,
            unique_entities=unique_entities,
            bucket_size_seconds=bucket_size_seconds,
        )

    async def record_geometric_metrics(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        extraction_total_attempts: int,
        extraction_validation_failures: int,
        extraction_success_rate: float,
        transformation_total_attempts: Optional[int],
        transformation_validation_failures: Optional[int],
        transformation_success_rate: Optional[float],
        roi_total_created: int,
        event_timestamp: datetime,
    ) -> None:
        """Persist geometric pipeline metrics snapshot for analytics."""
        try:
            async with self.get_geometric_metrics_repository() as repo:
                if repo is None:
                    return
                repo.insert_metric(
                    event_timestamp=event_timestamp,
                    environment_id=environment_id,
                    camera_id=camera_id,
                    extraction_total_attempts=extraction_total_attempts,
                    extraction_validation_failures=extraction_validation_failures,
                    extraction_success_rate=extraction_success_rate,
                    transformation_total_attempts=transformation_total_attempts,
                    transformation_validation_failures=transformation_validation_failures,
                    transformation_success_rate=transformation_success_rate,
                    roi_total_created=roi_total_created,
                )
        except Exception as exc:
            logger.warning(f"Failed to record geometric metrics snapshot: {exc}")

    async def get_dashboard_snapshot(
        self,
        *,
        environment_id: str,
        window_hours: int = 24,
        bucket_size_seconds: int = DEFAULT_BUCKET_SECONDS,
        uptime_history_days: int = 7,
    ) -> Dict[str, Any]:
        """Return aggregated analytics snapshot for dashboard endpoint."""
        window_hours = max(window_hours, 1)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=window_hours)

        snapshot = {
            "generated_at": end_time.isoformat(),
            "summary": {
                "total_detections": 0,
                "average_confidence_percent": 0.0,
                "system_uptime_percent": 0.0,
                "uptime_delta_percent": 0.0,
            },
            "cameras": [],
            "charts": {
                "detections_per_bucket": [],
                "average_confidence_trend": [],
                "uptime_trend": [],
            },
            "notes": ["Analytics values are aggregated from persisted detection and uptime events."],
        }

        try:
            async with self.get_analytics_repository() as repo:
                if repo is None:
                    return snapshot

                totals = repo.fetch_window_totals(
                    environment_id=environment_id,
                    start=start_time,
                    end=end_time,
                    bucket_size_seconds=bucket_size_seconds,
                )

                camera_totals = repo.fetch_camera_breakdown(
                    environment_id=environment_id,
                    start=start_time,
                    end=end_time,
                    bucket_size_seconds=bucket_size_seconds,
                )

                time_buckets = repo.fetch_bucketed_series(
                    environment_id=environment_id,
                    start=start_time,
                    end=end_time,
                    bucket_size_seconds=bucket_size_seconds,
                )

                uptime_series = repo.fetch_uptime_series(
                    environment_id=environment_id,
                    days=uptime_history_days,
                )

                snapshot["summary"]["total_detections"] = totals.detections
                snapshot["summary"]["average_confidence_percent"] = round(totals.average_confidence * 100, 2)

                if uptime_series:
                    latest_uptime = uptime_series[-1].uptime_percent
                    snapshot["summary"]["system_uptime_percent"] = round(latest_uptime, 2)
                    if len(uptime_series) > 1:
                        previous = uptime_series[-2].uptime_percent
                        snapshot["summary"]["uptime_delta_percent"] = round(latest_uptime - previous, 2)

                for totals_per_camera in camera_totals:
                    camera_snapshot = {
                        "camera_id": totals_per_camera.camera_id,
                        "detections": totals_per_camera.detections,
                        "unique_entities": totals_per_camera.unique_entities,
                        "average_confidence_percent": round(totals_per_camera.average_confidence * 100, 2),
                        "uptime_percent": 0.0,
                    }

                    camera_uptime = repo.fetch_uptime_series(
                        environment_id=environment_id,
                        days=1,
                        camera_id=totals_per_camera.camera_id,
                    )
                    if camera_uptime:
                        camera_snapshot["uptime_percent"] = round(camera_uptime[-1].uptime_percent, 2)

                    snapshot["cameras"].append(camera_snapshot)

                snapshot["charts"]["detections_per_bucket"] = [
                    {
                        "timestamp": bucket.bucket_start.isoformat(),
                        "detections": bucket.detections,
                    }
                    for bucket in time_buckets
                ]

                snapshot["charts"]["average_confidence_trend"] = [
                    {
                        "timestamp": bucket.bucket_start.isoformat(),
                        "confidence_percent": round(bucket.average_confidence * 100, 2),
                    }
                    for bucket in time_buckets
                ]

                snapshot["charts"]["uptime_trend"] = [
                    {
                        "date": str(item.day),
                        "uptime_percent": round(item.uptime_percent, 2),
                    }
                    for item in uptime_series
                ]

        except Exception as exc:
            logger.error(f"Failed to build dashboard snapshot: {exc}")

        return snapshot
    
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
            
            # logger.debug("Cache to database sync completed")
            
        except Exception as e:
            logger.error(f"Error syncing cache to database: {e}")
            self.sync_stats.sync_failures += 1
    
    async def _store_person_state_db(
        self,
        person_identity: Any,
        camera_id: CameraID,
        position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Store person state in database."""
        try:
            async with self.get_repository() as repo:
                # Check if person identity exists
                gid = getattr(person_identity, 'global_id', None)
                if gid is None and isinstance(person_identity, dict):
                    gid = person_identity.get('global_id')
                existing_identity = await repo.get_person_identity(gid)
                
                if existing_identity:
                    # Update existing identity
                    success = await repo.update_person_identity(
                        global_person_id=gid,
                        last_seen_camera=str(camera_id),
                        last_seen_at=datetime.now(timezone.utc),
                        new_embedding=(
                            (getattr(getattr(person_identity, 'feature_vector', None), 'vector', None) or (person_identity.get('feature_vector', {}) if isinstance(person_identity, dict) else {})).get('vector').tolist()
                            if isinstance(getattr(person_identity, 'feature_vector', None), dict) and getattr(person_identity['feature_vector'], 'vector', None) is not None else (
                                getattr(getattr(person_identity, 'feature_vector', None), 'vector', None).tolist()
                                if getattr(getattr(person_identity, 'feature_vector', None), 'vector', None) is not None else None
                            )
                        ),
                        confidence=float(getattr(person_identity, 'confidence', 0.0)) if not isinstance(person_identity, dict) else float(person_identity.get('confidence', 0.0)),
                    )
                else:
                    # Create new identity
                    identity = await repo.create_person_identity(
                        global_person_id=gid,
                        environment_id="default",  # You'd get this from context
                        first_seen_camera=str(camera_id),
                        first_seen_at=datetime.now(timezone.utc),
                        primary_embedding=(
                            (getattr(getattr(person_identity, 'feature_vector', None), 'vector', None) or (person_identity.get('feature_vector', {}) if isinstance(person_identity, dict) else {})).get('vector').tolist()
                            if isinstance(getattr(person_identity, 'feature_vector', None), dict) and getattr(person_identity['feature_vector'], 'vector', None) is not None else (
                                getattr(getattr(person_identity, 'feature_vector', None), 'vector', None).tolist()
                                if getattr(getattr(person_identity, 'feature_vector', None), 'vector', None) is not None else None
                            )
                        ),
                        confidence=float(getattr(person_identity, 'confidence', 0.0)) if not isinstance(person_identity, dict) else float(person_identity.get('confidence', 0.0)),
                    )
                    success = identity is not None
                
                # Store tracking event
                if success:
                    await repo.create_tracking_event(
                        global_person_id=gid,
                        camera_id=str(camera_id),
                        environment_id="default",
                        event_type="detection",
                        position_x=position.x if position else None,
                        position_y=position.y if position else None,
                        world_x=position.world_x if position else None,
                        world_y=position.world_y if position else None,
                        detection_confidence=(float(getattr(person_identity, 'confidence', 0.0)) if not isinstance(person_identity, dict) else float(person_identity.get('confidence', 0.0))),
                        reid_confidence=None,
                        track_id=(getattr(person_identity, 'track_id', None) if not isinstance(person_identity, dict) else person_identity.get('track_id')),
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
