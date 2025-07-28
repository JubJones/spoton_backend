"""
Repository layer for tracking data operations.

Handles:
- CRUD operations for tracking data
- Time-series queries
- Analytics aggregations
- Performance optimization
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy import func, and_, or_, desc, asc, text
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.database.models.tracking_models import (
    TrackingEvent, DetectionEvent, PersonTrajectory, PersonIdentity,
    AnalyticsAggregation, SessionRecord
)
from app.infrastructure.database.base import get_db
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class TrackingRepository:
    """
    Repository for tracking data operations.
    
    Provides optimized data access for time-series tracking data.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    # Tracking Events
    async def create_tracking_event(
        self,
        global_person_id: str,
        camera_id: str,
        environment_id: str,
        event_type: str,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
        world_x: Optional[float] = None,
        world_y: Optional[float] = None,
        bbox_data: Optional[Dict[str, float]] = None,
        detection_confidence: Optional[float] = None,
        reid_confidence: Optional[float] = None,
        track_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrackingEvent:
        """Create a new tracking event."""
        try:
            event = TrackingEvent(
                global_person_id=global_person_id,
                camera_id=camera_id,
                environment_id=environment_id,
                event_type=event_type,
                position_x=position_x,
                position_y=position_y,
                world_x=world_x,
                world_y=world_y,
                detection_confidence=detection_confidence,
                reid_confidence=reid_confidence,
                track_id=track_id,
                session_id=session_id,
                metadata=metadata
            )
            
            # Add bounding box data if provided
            if bbox_data:
                event.bbox_x1 = bbox_data.get('x1')
                event.bbox_y1 = bbox_data.get('y1')
                event.bbox_x2 = bbox_data.get('x2')
                event.bbox_y2 = bbox_data.get('y2')
            
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            
            logger.debug(f"Created tracking event for person {global_person_id}")
            return event
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating tracking event: {e}")
            self.db.rollback()
            raise
    
    async def get_tracking_events(
        self,
        global_person_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[TrackingEvent]:
        """Get tracking events with filtering."""
        try:
            query = self.db.query(TrackingEvent)
            
            # Apply filters
            if global_person_id:
                query = query.filter(TrackingEvent.global_person_id == global_person_id)
            if camera_id:
                query = query.filter(TrackingEvent.camera_id == camera_id)
            if environment_id:
                query = query.filter(TrackingEvent.environment_id == environment_id)
            if session_id:
                query = query.filter(TrackingEvent.session_id == session_id)
            if event_type:
                query = query.filter(TrackingEvent.event_type == event_type)
            if start_time:
                query = query.filter(TrackingEvent.timestamp >= start_time)
            if end_time:
                query = query.filter(TrackingEvent.timestamp <= end_time)
            
            # Order by timestamp and limit
            query = query.order_by(desc(TrackingEvent.timestamp)).limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting tracking events: {e}")
            raise
    
    async def get_person_timeline(
        self,
        global_person_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TrackingEvent]:
        """Get complete timeline for a person."""
        try:
            query = self.db.query(TrackingEvent).filter(
                TrackingEvent.global_person_id == global_person_id
            )
            
            if start_time:
                query = query.filter(TrackingEvent.timestamp >= start_time)
            if end_time:
                query = query.filter(TrackingEvent.timestamp <= end_time)
            
            return query.order_by(asc(TrackingEvent.timestamp)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting person timeline: {e}")
            raise
    
    # Detection Events
    async def create_detection_event(
        self,
        camera_id: str,
        environment_id: str,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        confidence: float,
        frame_number: Optional[int] = None,
        object_class: str = 'person',
        tracking_event_id: Optional[UUID] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionEvent:
        """Create a new detection event."""
        try:
            event = DetectionEvent(
                camera_id=camera_id,
                environment_id=environment_id,
                bbox_x1=bbox_x1,
                bbox_y1=bbox_y1,
                bbox_x2=bbox_x2,
                bbox_y2=bbox_y2,
                confidence=confidence,
                frame_number=frame_number,
                object_class=object_class,
                tracking_event_id=tracking_event_id,
                session_id=session_id,
                metadata=metadata
            )
            
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            
            logger.debug(f"Created detection event for camera {camera_id}")
            return event
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating detection event: {e}")
            self.db.rollback()
            raise
    
    async def get_detection_events(
        self,
        camera_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        session_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[DetectionEvent]:
        """Get detection events with filtering."""
        try:
            query = self.db.query(DetectionEvent)
            
            # Apply filters
            if camera_id:
                query = query.filter(DetectionEvent.camera_id == camera_id)
            if environment_id:
                query = query.filter(DetectionEvent.environment_id == environment_id)
            if session_id:
                query = query.filter(DetectionEvent.session_id == session_id)
            if min_confidence:
                query = query.filter(DetectionEvent.confidence >= min_confidence)
            if start_time:
                query = query.filter(DetectionEvent.timestamp >= start_time)
            if end_time:
                query = query.filter(DetectionEvent.timestamp <= end_time)
            
            # Order by timestamp and limit
            query = query.order_by(desc(DetectionEvent.timestamp)).limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting detection events: {e}")
            raise
    
    # Person Trajectories
    async def create_trajectory_point(
        self,
        global_person_id: str,
        camera_id: str,
        environment_id: str,
        sequence_number: int,
        position_x: float,
        position_y: float,
        world_x: Optional[float] = None,
        world_y: Optional[float] = None,
        velocity_x: Optional[float] = None,
        velocity_y: Optional[float] = None,
        acceleration_x: Optional[float] = None,
        acceleration_y: Optional[float] = None,
        confidence: Optional[float] = None,
        smoothed: bool = False,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PersonTrajectory:
        """Create a new trajectory point."""
        try:
            trajectory_point = PersonTrajectory(
                global_person_id=global_person_id,
                camera_id=camera_id,
                environment_id=environment_id,
                sequence_number=sequence_number,
                position_x=position_x,
                position_y=position_y,
                world_x=world_x,
                world_y=world_y,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                acceleration_x=acceleration_x,
                acceleration_y=acceleration_y,
                confidence=confidence,
                smoothed=smoothed,
                session_id=session_id,
                metadata=metadata
            )
            
            self.db.add(trajectory_point)
            self.db.commit()
            self.db.refresh(trajectory_point)
            
            logger.debug(f"Created trajectory point for person {global_person_id}")
            return trajectory_point
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating trajectory point: {e}")
            self.db.rollback()
            raise
    
    async def get_person_trajectory(
        self,
        global_person_id: str,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000
    ) -> List[PersonTrajectory]:
        """Get trajectory points for a person."""
        try:
            query = self.db.query(PersonTrajectory).filter(
                PersonTrajectory.global_person_id == global_person_id
            )
            
            if camera_id:
                query = query.filter(PersonTrajectory.camera_id == camera_id)
            if start_time:
                query = query.filter(PersonTrajectory.timestamp >= start_time)
            if end_time:
                query = query.filter(PersonTrajectory.timestamp <= end_time)
            
            return query.order_by(asc(PersonTrajectory.sequence_number)).limit(limit).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting person trajectory: {e}")
            raise
    
    # Person Identity Management
    async def create_person_identity(
        self,
        global_person_id: str,
        environment_id: str,
        first_seen_camera: str,
        first_seen_at: datetime,
        primary_embedding: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PersonIdentity:
        """Create a new person identity."""
        try:
            identity = PersonIdentity(
                global_person_id=global_person_id,
                environment_id=environment_id,
                first_seen_camera=first_seen_camera,
                last_seen_camera=first_seen_camera,
                first_seen_at=first_seen_at,
                last_seen_at=first_seen_at,
                cameras_seen=[first_seen_camera],
                primary_embedding=primary_embedding,
                confidence=confidence,
                metadata=metadata
            )
            
            self.db.add(identity)
            self.db.commit()
            self.db.refresh(identity)
            
            logger.debug(f"Created person identity {global_person_id}")
            return identity
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating person identity: {e}")
            self.db.rollback()
            raise
    
    async def get_person_identity(self, global_person_id: str) -> Optional[PersonIdentity]:
        """Get person identity by global ID."""
        try:
            return self.db.query(PersonIdentity).filter(
                PersonIdentity.global_person_id == global_person_id
            ).first()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting person identity: {e}")
            raise
    
    async def update_person_identity(
        self,
        global_person_id: str,
        last_seen_camera: str,
        last_seen_at: datetime,
        new_embedding: Optional[List[float]] = None,
        confidence: Optional[float] = None
    ) -> bool:
        """Update person identity with new sighting."""
        try:
            identity = await self.get_person_identity(global_person_id)
            if not identity:
                return False
            
            # Update basic fields
            identity.last_seen_camera = last_seen_camera
            identity.last_seen_at = last_seen_at
            identity.updated_at = datetime.now(timezone.utc)
            
            # Update camera list
            if isinstance(identity.cameras_seen, list):
                if last_seen_camera not in identity.cameras_seen:
                    identity.cameras_seen.append(last_seen_camera)
            else:
                identity.cameras_seen = [last_seen_camera]
            
            # Update embedding if provided
            if new_embedding:
                identity.primary_embedding = new_embedding
            
            # Update confidence if provided
            if confidence:
                identity.confidence = confidence
            
            # Increment detection count
            identity.total_detections += 1
            
            self.db.commit()
            
            logger.debug(f"Updated person identity {global_person_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error updating person identity: {e}")
            self.db.rollback()
            raise
    
    async def get_active_persons(
        self,
        environment_id: str,
        camera_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[PersonIdentity]:
        """Get active persons in environment."""
        try:
            query = self.db.query(PersonIdentity).filter(
                and_(
                    PersonIdentity.environment_id == environment_id,
                    PersonIdentity.is_active == True
                )
            )
            
            if camera_id:
                query = query.filter(PersonIdentity.last_seen_camera == camera_id)
            
            if since:
                query = query.filter(PersonIdentity.last_seen_at >= since)
            
            return query.order_by(desc(PersonIdentity.last_seen_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting active persons: {e}")
            raise
    
    # Session Management
    async def create_session_record(
        self,
        session_id: str,
        environment_id: str,
        camera_ids: List[str],
        user_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> SessionRecord:
        """Create a new session record."""
        try:
            session = SessionRecord(
                session_id=session_id,
                environment_id=environment_id,
                camera_ids=camera_ids,
                user_id=user_id,
                start_time=datetime.now(timezone.utc),
                settings=settings
            )
            
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            
            logger.info(f"Created session record {session_id}")
            return session
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating session record: {e}")
            self.db.rollback()
            raise
    
    async def get_session_record(self, session_id: str) -> Optional[SessionRecord]:
        """Get session record by ID."""
        try:
            return self.db.query(SessionRecord).filter(
                SessionRecord.session_id == session_id
            ).first()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting session record: {e}")
            raise
    
    async def update_session_record(
        self,
        session_id: str,
        end_time: Optional[datetime] = None,
        status: Optional[str] = None,
        total_persons_tracked: Optional[int] = None,
        total_detections: Optional[int] = None,
        total_events: Optional[int] = None,
        avg_detection_confidence: Optional[float] = None,
        avg_reid_confidence: Optional[float] = None
    ) -> bool:
        """Update session record statistics."""
        try:
            session = await self.get_session_record(session_id)
            if not session:
                return False
            
            # Update fields
            if end_time:
                session.end_time = end_time
                if session.start_time:
                    session.duration = (end_time - session.start_time).total_seconds()
            
            if status:
                session.status = status
            
            if total_persons_tracked is not None:
                session.total_persons_tracked = total_persons_tracked
            
            if total_detections is not None:
                session.total_detections = total_detections
            
            if total_events is not None:
                session.total_events = total_events
            
            if avg_detection_confidence is not None:
                session.avg_detection_confidence = avg_detection_confidence
            
            if avg_reid_confidence is not None:
                session.avg_reid_confidence = avg_reid_confidence
            
            session.updated_at = datetime.now(timezone.utc)
            
            self.db.commit()
            
            logger.debug(f"Updated session record {session_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error updating session record: {e}")
            self.db.rollback()
            raise
    
    # Analytics and Aggregations
    async def get_detection_statistics(
        self,
        environment_id: str,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detection statistics for analysis."""
        try:
            query = self.db.query(DetectionEvent).filter(
                DetectionEvent.environment_id == environment_id
            )
            
            if camera_id:
                query = query.filter(DetectionEvent.camera_id == camera_id)
            if start_time:
                query = query.filter(DetectionEvent.timestamp >= start_time)
            if end_time:
                query = query.filter(DetectionEvent.timestamp <= end_time)
            
            # Get aggregated statistics
            stats = query.with_entities(
                func.count(DetectionEvent.id).label('total_detections'),
                func.avg(DetectionEvent.confidence).label('avg_confidence'),
                func.min(DetectionEvent.confidence).label('min_confidence'),
                func.max(DetectionEvent.confidence).label('max_confidence'),
                func.min(DetectionEvent.timestamp).label('first_detection'),
                func.max(DetectionEvent.timestamp).label('last_detection')
            ).first()
            
            return {
                'total_detections': stats.total_detections or 0,
                'avg_confidence': float(stats.avg_confidence) if stats.avg_confidence else 0.0,
                'min_confidence': float(stats.min_confidence) if stats.min_confidence else 0.0,
                'max_confidence': float(stats.max_confidence) if stats.max_confidence else 0.0,
                'first_detection': stats.first_detection,
                'last_detection': stats.last_detection
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting detection statistics: {e}")
            raise
    
    async def get_person_statistics(
        self,
        environment_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get person tracking statistics."""
        try:
            query = self.db.query(PersonIdentity).filter(
                PersonIdentity.environment_id == environment_id
            )
            
            if start_time:
                query = query.filter(PersonIdentity.first_seen_at >= start_time)
            if end_time:
                query = query.filter(PersonIdentity.last_seen_at <= end_time)
            
            # Get aggregated statistics
            stats = query.with_entities(
                func.count(PersonIdentity.id).label('total_persons'),
                func.count(PersonIdentity.id).filter(PersonIdentity.is_active == True).label('active_persons'),
                func.avg(PersonIdentity.total_detections).label('avg_detections_per_person'),
                func.avg(PersonIdentity.total_tracking_time).label('avg_tracking_time'),
                func.avg(PersonIdentity.confidence).label('avg_confidence')
            ).first()
            
            return {
                'total_persons': stats.total_persons or 0,
                'active_persons': stats.active_persons or 0,
                'avg_detections_per_person': float(stats.avg_detections_per_person) if stats.avg_detections_per_person else 0.0,
                'avg_tracking_time': float(stats.avg_tracking_time) if stats.avg_tracking_time else 0.0,
                'avg_confidence': float(stats.avg_confidence) if stats.avg_confidence else 0.0
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting person statistics: {e}")
            raise
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old tracking data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            # Delete old tracking events
            deleted_tracking = self.db.query(TrackingEvent).filter(
                TrackingEvent.timestamp < cutoff_date
            ).delete()
            
            # Delete old detection events
            deleted_detection = self.db.query(DetectionEvent).filter(
                DetectionEvent.timestamp < cutoff_date
            ).delete()
            
            # Delete old trajectory points
            deleted_trajectory = self.db.query(PersonTrajectory).filter(
                PersonTrajectory.timestamp < cutoff_date
            ).delete()
            
            self.db.commit()
            
            logger.info(f"Cleaned up old data: {deleted_tracking} tracking events, "
                       f"{deleted_detection} detection events, {deleted_trajectory} trajectory points")
            
            return {
                'deleted_tracking_events': deleted_tracking,
                'deleted_detection_events': deleted_detection,
                'deleted_trajectory_points': deleted_trajectory
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error cleaning up old data: {e}")
            self.db.rollback()
            raise


def get_tracking_repository(db: Session = None) -> TrackingRepository:
    """Get tracking repository instance."""
    if db is None:
        db = next(get_db())
    return TrackingRepository(db)