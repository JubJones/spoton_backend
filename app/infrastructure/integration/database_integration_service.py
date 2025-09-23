"""
Database integration service for domain services.

Handles:
- Integration between domain services and database layer
- Data transformation and validation
- Performance optimization
- Error handling and recovery
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.infrastructure.cache.tracking_cache import tracking_cache
from app.domains.detection.entities.detection import Detection, BoundingBox
from typing import Any
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.mapping.entities.trajectory import Trajectory
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class DatabaseIntegrationService:
    """
    Database integration service for domain services.
    
    Provides a unified interface for domain services to interact with
    the database layer while maintaining domain boundaries.
    """
    
    def __init__(self):
        self.service_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("DatabaseIntegrationService initialized")
    
    async def initialize(self):
        """Initialize the database integration service."""
        try:
            await integrated_db_service.initialize()
            logger.info("DatabaseIntegrationService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DatabaseIntegrationService: {e}")
            raise
    
    async def cleanup(self):
        """Clean up the database integration service."""
        try:
            await integrated_db_service.cleanup()
            logger.info("DatabaseIntegrationService cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up DatabaseIntegrationService: {e}")
    
    # Detection Domain Integration
    async def store_detection(
        self,
        detection: Detection,
        camera_id: CameraID,
        environment_id: str,
        frame_number: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """Store detection data in the database."""
        try:
            self.service_stats['total_operations'] += 1
            
            # Convert detection to database format
            bbox_data = {
                'x1': detection.bbox.x,
                'y1': detection.bbox.y,
                'x2': detection.bbox.x2,
                'y2': detection.bbox.y2
            }
            
            # Store detection event
            success = await integrated_db_service.store_detection_event(
                camera_id=str(camera_id),
                environment_id=environment_id,
                bbox_data=bbox_data,
                confidence=detection.confidence,
                frame_number=frame_number,
                session_id=session_id,
                metadata={
                    'detection_id': detection.id,
                    'object_class': detection.class_id.name if detection.class_id else 'UNKNOWN',
                    'timestamp': detection.timestamp.isoformat() if detection.timestamp else None
                }
            )
            
            if success:
                self.service_stats['successful_operations'] += 1
                logger.debug(f"Stored detection for camera {camera_id}")
            else:
                self.service_stats['failed_operations'] += 1
                logger.warning(f"Failed to store detection for camera {camera_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing detection: {e}")
            self.service_stats['failed_operations'] += 1
            return False
    
    async def get_detection_history(
        self,
        camera_id: Optional[CameraID] = None,
        environment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get detection history from the database."""
        try:
            self.service_stats['total_operations'] += 1
            
            # Get detection statistics
            stats = await integrated_db_service.get_detection_statistics(
                environment_id=environment_id or "default",
                camera_id=str(camera_id) if camera_id else None,
                start_time=start_time,
                end_time=end_time
            )
            
            self.service_stats['successful_operations'] += 1
            return [stats]  # Return as list for consistency
            
        except Exception as e:
            logger.error(f"Error getting detection history: {e}")
            self.service_stats['failed_operations'] += 1
            return []
    
    # Identity Domain Integration
    async def store_person_identity(
        self,
        person_identity: Any,
        camera_id: CameraID,
        position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Store person identity data in the database."""
        try:
            self.service_stats['total_operations'] += 1
            
            # Store person state
            success = await integrated_db_service.store_person_state(
                person_identity=person_identity,
                camera_id=camera_id,
                position=position,
                trajectory=trajectory,
                session_id=session_id
            )
            
            # Cache embedding if available
            fv = getattr(person_identity, 'feature_vector', None)
            if success and fv:
                await tracking_cache.cache_embedding(
                    person_id=getattr(person_identity, 'global_id', None) if not isinstance(person_identity, dict) else person_identity.get('global_id'),
                    feature_vector=fv,
                    camera_id=camera_id
                )
            
            if success:
                self.service_stats['successful_operations'] += 1
                logger.debug(f"Stored person identity {getattr(person_identity, 'global_id', None)}")
            else:
                self.service_stats['failed_operations'] += 1
                logger.warning(f"Failed to store person identity {getattr(person_identity, 'global_id', None)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing person identity: {e}")
            self.service_stats['failed_operations'] += 1
            return False
    
    async def get_person_identity(
        self,
        global_person_id: str,
        prefer_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get person identity from cache or database."""
        try:
            self.service_stats['total_operations'] += 1
            
            # Get person state
            person_state = await integrated_db_service.get_person_state(
                global_person_id=global_person_id,
                prefer_cache=prefer_cache
            )
            
            if person_state:
                if person_state.get('source') == 'cache':
                    self.service_stats['cache_hits'] += 1
                else:
                    self.service_stats['cache_misses'] += 1
                
                self.service_stats['successful_operations'] += 1
                return person_state
            else:
                self.service_stats['failed_operations'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting person identity: {e}")
            self.service_stats['failed_operations'] += 1
            return None
    
    async def get_active_persons(
        self,
        camera_id: Optional[CameraID] = None,
        environment_id: Optional[str] = None,
        prefer_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get active persons from cache or database."""
        try:
            self.service_stats['total_operations'] += 1
            
            # Get active persons
            persons = await integrated_db_service.get_active_persons(
                camera_id=camera_id,
                environment_id=environment_id,
                prefer_cache=prefer_cache
            )
            
            # Update cache statistics
            if persons:
                if persons[0].get('source') == 'cache':
                    self.service_stats['cache_hits'] += 1
                else:
                    self.service_stats['cache_misses'] += 1
            
            self.service_stats['successful_operations'] += 1
            return persons
            
        except Exception as e:
            logger.error(f"Error getting active persons: {e}")
            self.service_stats['failed_operations'] += 1
            return []
    
    async def get_person_embeddings(
        self,
        person_id: str,
        camera_id: Optional[CameraID] = None
    ) -> List[Dict[str, Any]]:
        """Get cached embeddings for a person."""
        try:
            self.service_stats['total_operations'] += 1
            
            if camera_id:
                # Get embeddings from specific camera
                embeddings = await tracking_cache.get_camera_embeddings(camera_id)
                # Filter by person ID
                embeddings = [emb for emb in embeddings if emb.get('person_id') == person_id]
            else:
                # Get all embeddings for person
                embeddings = await tracking_cache.get_person_embeddings(person_id)
            
            self.service_stats['successful_operations'] += 1
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting person embeddings: {e}")
            self.service_stats['failed_operations'] += 1
            return []
    
    # Mapping Domain Integration
    async def store_trajectory(
        self,
        person_id: str,
        camera_id: CameraID,
        environment_id: str,
        trajectory: Trajectory,
        session_id: Optional[str] = None
    ) -> bool:
        """Store trajectory data in the database."""
        try:
            self.service_stats['total_operations'] += 1
            
            # Store trajectory in cache
            cache_success = await tracking_cache.cache_trajectory(
                person_id=person_id,
                trajectory=trajectory,
                camera_id=camera_id
            )
            
            # Store trajectory points in database
            db_success = True
            trajectory_dict = trajectory.to_dict()
            
            for i, point in enumerate(trajectory_dict.get('points', [])):
                coordinate = Coordinate(
                    x=point['x'],
                    y=point['y'],
                    world_x=point.get('world_x'),
                    world_y=point.get('world_y')
                )
                
                point_success = await integrated_db_service.store_trajectory_point(
                    global_person_id=person_id,
                    camera_id=str(camera_id),
                    environment_id=environment_id,
                    sequence_number=i,
                    position=coordinate,
                    confidence=point.get('confidence'),
                    session_id=session_id
                )
                
                if not point_success:
                    db_success = False
                    break
            
            success = cache_success and db_success
            
            if success:
                self.service_stats['successful_operations'] += 1
                logger.debug(f"Stored trajectory for person {person_id}")
            else:
                self.service_stats['failed_operations'] += 1
                logger.warning(f"Failed to store trajectory for person {person_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing trajectory: {e}")
            self.service_stats['failed_operations'] += 1
            return False
    
    async def get_person_trajectory(
        self,
        person_id: str,
        camera_id: Optional[CameraID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        prefer_cache: bool = True,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get person trajectory from cache or database."""
        try:
            self.service_stats['total_operations'] += 1
            
            if prefer_cache:
                # Try cache first
                cached_trajectories = await tracking_cache.get_person_trajectories(person_id)
                if cached_trajectories:
                    self.service_stats['cache_hits'] += 1
                    self.service_stats['successful_operations'] += 1
                    return cached_trajectories
            
            # Fall back to database
            trajectory_points = await integrated_db_service.get_person_trajectory(
                global_person_id=person_id,
                camera_id=str(camera_id) if camera_id else None,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            self.service_stats['cache_misses'] += 1
            self.service_stats['successful_operations'] += 1
            return trajectory_points
            
        except Exception as e:
            logger.error(f"Error getting person trajectory: {e}")
            self.service_stats['failed_operations'] += 1
            return []
    
    # Session Management Integration
    async def create_tracking_session(
        self,
        session_id: str,
        environment_id: str,
        camera_ids: List[CameraID],
        user_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new tracking session."""
        try:
            self.service_stats['total_operations'] += 1
            
            success = await integrated_db_service.create_session(
                session_id=session_id,
                environment_id=environment_id,
                camera_ids=camera_ids,
                user_id=user_id,
                settings=settings
            )
            
            if success:
                self.service_stats['successful_operations'] += 1
                logger.info(f"Created tracking session {session_id}")
            else:
                self.service_stats['failed_operations'] += 1
                logger.warning(f"Failed to create tracking session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating tracking session: {e}")
            self.service_stats['failed_operations'] += 1
            return False
    
    async def end_tracking_session(
        self,
        session_id: str,
        session_stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """End a tracking session."""
        try:
            self.service_stats['total_operations'] += 1
            
            success = await integrated_db_service.end_session(
                session_id=session_id,
                total_persons_tracked=session_stats.get('total_persons') if session_stats else None,
                total_detections=session_stats.get('total_detections') if session_stats else None,
                total_events=session_stats.get('total_events') if session_stats else None
            )
            
            if success:
                self.service_stats['successful_operations'] += 1
                logger.info(f"Ended tracking session {session_id}")
            else:
                self.service_stats['failed_operations'] += 1
                logger.warning(f"Failed to end tracking session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error ending tracking session: {e}")
            self.service_stats['failed_operations'] += 1
            return False
    
    # Analytics Integration
    async def get_analytics_summary(
        self,
        environment_id: str,
        camera_id: Optional[CameraID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics summary for environment."""
        try:
            self.service_stats['total_operations'] += 1
            
            # Get detection statistics
            detection_stats = await integrated_db_service.get_detection_statistics(
                environment_id=environment_id,
                camera_id=str(camera_id) if camera_id else None,
                start_time=start_time,
                end_time=end_time
            )
            
            # Get person statistics
            person_stats = await integrated_db_service.get_person_statistics(
                environment_id=environment_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Combine statistics
            analytics_summary = {
                'environment_id': environment_id,
                'camera_id': str(camera_id) if camera_id else None,
                'time_range': {
                    'start_time': start_time.isoformat() if start_time else None,
                    'end_time': end_time.isoformat() if end_time else None
                },
                'detection_statistics': detection_stats,
                'person_statistics': person_stats,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.service_stats['successful_operations'] += 1
            return analytics_summary
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            self.service_stats['failed_operations'] += 1
            return {}
    
    # Maintenance and Utilities
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up old data from the database."""
        try:
            self.service_stats['total_operations'] += 1
            
            cleanup_result = await integrated_db_service.cleanup_old_data(days_to_keep)
            
            self.service_stats['successful_operations'] += 1
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            self.service_stats['failed_operations'] += 1
            return {}
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            # Get integrated service statistics
            service_stats = await integrated_db_service.get_service_statistics()
            
            # Add integration layer statistics
            health_status = {
                'integration_layer': {
                    'total_operations': self.service_stats['total_operations'],
                    'successful_operations': self.service_stats['successful_operations'],
                    'failed_operations': self.service_stats['failed_operations'],
                    'cache_hits': self.service_stats['cache_hits'],
                    'cache_misses': self.service_stats['cache_misses'],
                    'success_rate': (
                        self.service_stats['successful_operations'] / 
                        max(1, self.service_stats['total_operations'])
                    ) * 100,
                    'cache_hit_rate': (
                        self.service_stats['cache_hits'] / 
                        max(1, self.service_stats['cache_hits'] + self.service_stats['cache_misses'])
                    ) * 100
                },
                'integrated_service': service_stats
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting service health: {e}")
            return {'error': str(e)}
    
    def reset_statistics(self):
        """Reset service statistics."""
        self.service_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        logger.info("Service statistics reset")


# Global database integration service instance
database_integration_service = DatabaseIntegrationService()
