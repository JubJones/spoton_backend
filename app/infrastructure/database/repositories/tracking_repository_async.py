"""
Async repository layer for tracking data operations.

Provides non-blocking database accessors for hot write paths and selected reads.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models.tracking_models import (
    TrackingEvent, DetectionEvent, PersonTrajectory, PersonIdentity,
)

logger = logging.getLogger(__name__)


class TrackingRepositoryAsync:
    def __init__(self, session: AsyncSession):
        self.db = session

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrackingEvent:
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
            metadata=metadata,
        )
        if bbox_data:
            event.bbox_x1 = bbox_data.get("x1")
            event.bbox_y1 = bbox_data.get("y1")
            event.bbox_x2 = bbox_data.get("x2")
            event.bbox_y2 = bbox_data.get("y2")
        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)
        return event

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
        object_class: str = "person",
        tracking_event_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DetectionEvent:
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
            metadata=metadata,
        )
        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)
        return event

    # Person Trajectories
    async def create_trajectory_points_bulk(
        self,
        points: List[Dict[str, Any]],
    ) -> int:
        """Bulk insert multiple trajectory points in a single transaction.

        Each point dict must include the fields required by PersonTrajectory.
        Returns number of rows inserted.
        """
        if not points:
            return 0
        objs = [
            PersonTrajectory(
                global_person_id=p["global_person_id"],
                camera_id=p["camera_id"],
                environment_id=p["environment_id"],
                sequence_number=p["sequence_number"],
                position_x=p["position_x"],
                position_y=p["position_y"],
                world_x=p.get("world_x"),
                world_y=p.get("world_y"),
                velocity_x=p.get("velocity_x"),
                velocity_y=p.get("velocity_y"),
                acceleration_x=p.get("acceleration_x"),
                acceleration_y=p.get("acceleration_y"),
                confidence=p.get("confidence"),
                smoothed=bool(p.get("smoothed", False)),
                session_id=p.get("session_id"),
                metadata=p.get("metadata"),
            )
            for p in points
        ]
        self.db.add_all(objs)
        await self.db.commit()
        return len(objs)

    # Person Identity Management
    async def get_person_identity(self, global_person_id: str) -> Optional[PersonIdentity]:
        result = await self.db.execute(
            select(PersonIdentity).where(PersonIdentity.global_person_id == global_person_id)
        )
        return result.scalars().first()

    async def create_person_identity(
        self,
        global_person_id: str,
        environment_id: str,
        first_seen_camera: str,
        first_seen_at: datetime,
        primary_embedding: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PersonIdentity:
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
            metadata=metadata,
        )
        self.db.add(identity)
        await self.db.commit()
        await self.db.refresh(identity)
        return identity

    async def update_person_identity(
        self,
        global_person_id: str,
        last_seen_camera: str,
        last_seen_at: datetime,
        new_embedding: Optional[List[float]] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        identity = await self.get_person_identity(global_person_id)
        if not identity:
            return False
        identity.last_seen_camera = last_seen_camera
        identity.last_seen_at = last_seen_at
        identity.updated_at = datetime.now(timezone.utc)
        # Update cameras_seen list
        try:
            cams = list(identity.cameras_seen or [])
            if last_seen_camera not in cams:
                cams.append(last_seen_camera)
            identity.cameras_seen = cams
        except Exception:
            identity.cameras_seen = [last_seen_camera]
        if new_embedding:
            identity.primary_embedding = new_embedding
        if confidence is not None:
            identity.confidence = confidence
        identity.total_detections += 1
        await self.db.commit()
        return True
