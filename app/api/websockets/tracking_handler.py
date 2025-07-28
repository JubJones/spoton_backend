"""
Tracking handler for WebSocket tracking update messages.

Handles:
- Tracking update messages
- Person tracking data
- Camera transitions
- Real-time person movement
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from app.api.websockets.connection_manager import binary_websocket_manager, MessageType
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.mapping.entities.trajectory import Trajectory
from app.shared.types import CameraID, TrackID, GlobalID

logger = logging.getLogger(__name__)


@dataclass
class CameraTransition:
    """Represents a camera transition event."""
    source_camera: CameraID
    target_camera: CameraID
    transition_time: datetime
    confidence: float
    coordinates: Optional[Coordinate] = None


@dataclass
class TrackingUpdate:
    """Represents a tracking update message."""
    person_id: int
    global_id: GlobalID
    camera_transitions: List[CameraTransition]
    current_position: Optional[Coordinate]
    trajectory_path: List[Coordinate]
    last_seen: datetime
    confidence: float
    cameras_seen: List[CameraID]


class TrackingHandler:
    """
    Handles tracking update messages for WebSocket transmission.
    
    Features:
    - Real-time person tracking updates
    - Camera transition detection
    - Trajectory path management
    - Performance monitoring
    """
    
    def __init__(self):
        # Tracking state
        self.active_persons: Dict[str, TrackingUpdate] = {}
        self.camera_transitions: Dict[str, List[CameraTransition]] = {}
        
        # Performance monitoring
        self.tracking_stats = {
            "total_updates_sent": 0,
            "total_persons_tracked": 0,
            "total_camera_transitions": 0,
            "average_update_latency": 0.0,
            "failed_updates": 0
        }
        
        # Update throttling
        self.update_throttle_ms = 100  # Minimum time between updates
        self.last_update_times: Dict[str, float] = {}
        
        logger.info("TrackingHandler initialized")
    
    async def send_tracking_update(
        self, 
        task_id: str, 
        person_identity: PersonIdentity,
        current_position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None
    ) -> bool:
        """
        Send tracking update for a person.
        
        Args:
            task_id: Task identifier
            person_identity: Person identity object
            current_position: Current position coordinate
            trajectory: Person trajectory
            
        Returns:
            True if sent successfully
        """
        try:
            update_start = time.time()
            
            # Check throttling
            person_key = f"{task_id}_{person_identity.global_id}"
            if not self._should_send_update(person_key):
                return True  # Skip this update
            
            # Create tracking update
            tracking_update = self._create_tracking_update(
                person_identity, current_position, trajectory
            )
            
            # Create message
            message = {
                "type": MessageType.TRACKING_UPDATE.value,
                "task_id": task_id,
                "person_id": int(person_identity.global_id.replace("person_", "")),
                "global_id": person_identity.global_id,
                "camera_transitions": [
                    {
                        "source_camera": trans.source_camera,
                        "target_camera": trans.target_camera,
                        "transition_time": trans.transition_time.isoformat(),
                        "confidence": trans.confidence,
                        "coordinates": {
                            "x": trans.coordinates.x,
                            "y": trans.coordinates.y,
                            "coordinate_system": trans.coordinates.coordinate_system.value
                        } if trans.coordinates else None
                    }
                    for trans in tracking_update.camera_transitions
                ],
                "current_position": {
                    "x": tracking_update.current_position.x,
                    "y": tracking_update.current_position.y,
                    "coordinate_system": tracking_update.current_position.coordinate_system.value,
                    "timestamp": tracking_update.current_position.timestamp.isoformat(),
                    "confidence": tracking_update.current_position.confidence
                } if tracking_update.current_position else None,
                "trajectory_path": [
                    {
                        "x": coord.x,
                        "y": coord.y,
                        "coordinate_system": coord.coordinate_system.value,
                        "timestamp": coord.timestamp.isoformat(),
                        "confidence": coord.confidence
                    }
                    for coord in tracking_update.trajectory_path
                ],
                "last_seen": tracking_update.last_seen.isoformat(),
                "confidence": tracking_update.confidence,
                "cameras_seen": tracking_update.cameras_seen,
                "metadata": {
                    "tracks_count": len(person_identity.track_ids_by_camera),
                    "first_seen": person_identity.first_seen.isoformat() if person_identity.first_seen else None,
                    "identity_confidence": person_identity.identity_confidence
                }
            }
            
            # Send message
            success = await binary_websocket_manager.send_json_message(
                task_id, message, MessageType.TRACKING_UPDATE
            )
            
            if success:
                # Update tracking state
                self.active_persons[person_key] = tracking_update
                self.last_update_times[person_key] = time.time()
                
                # Update statistics
                update_latency = time.time() - update_start
                self._update_tracking_stats(update_latency)
                
                logger.debug(f"Sent tracking update for person {person_identity.global_id}")
            else:
                self.tracking_stats["failed_updates"] += 1
                logger.warning(f"Failed to send tracking update for person {person_identity.global_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending tracking update: {e}")
            self.tracking_stats["failed_updates"] += 1
            return False
    
    async def send_batch_tracking_updates(
        self, 
        task_id: str, 
        person_updates: List[Dict[str, Any]]
    ) -> bool:
        """
        Send batch tracking updates.
        
        Args:
            task_id: Task identifier
            person_updates: List of person update data
            
        Returns:
            True if sent successfully
        """
        try:
            # Create batch message
            batch_message = {
                "type": "batch_tracking_update",
                "task_id": task_id,
                "update_count": len(person_updates),
                "persons": person_updates,
                "batch_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Send batch
            success = await binary_websocket_manager.send_json_message(
                task_id, batch_message, MessageType.TRACKING_UPDATE
            )
            
            if success:
                self.tracking_stats["total_updates_sent"] += len(person_updates)
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending batch tracking updates: {e}")
            return False
    
    async def send_camera_transition(
        self, 
        task_id: str, 
        person_id: str,
        transition: CameraTransition
    ) -> bool:
        """
        Send camera transition notification.
        
        Args:
            task_id: Task identifier
            person_id: Person identifier
            transition: Camera transition data
            
        Returns:
            True if sent successfully
        """
        try:
            # Create transition message
            message = {
                "type": "camera_transition",
                "task_id": task_id,
                "person_id": person_id,
                "transition": {
                    "source_camera": transition.source_camera,
                    "target_camera": transition.target_camera,
                    "transition_time": transition.transition_time.isoformat(),
                    "confidence": transition.confidence,
                    "coordinates": {
                        "x": transition.coordinates.x,
                        "y": transition.coordinates.y,
                        "coordinate_system": transition.coordinates.coordinate_system.value
                    } if transition.coordinates else None
                }
            }
            
            # Send message
            success = await binary_websocket_manager.send_json_message(
                task_id, message, MessageType.TRACKING_UPDATE
            )
            
            if success:
                # Store transition
                if person_id not in self.camera_transitions:
                    self.camera_transitions[person_id] = []
                self.camera_transitions[person_id].append(transition)
                
                self.tracking_stats["total_camera_transitions"] += 1
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending camera transition: {e}")
            return False
    
    def _create_tracking_update(
        self, 
        person_identity: PersonIdentity,
        current_position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None
    ) -> TrackingUpdate:
        """Create tracking update from person identity."""
        try:
            # Extract camera transitions
            camera_transitions = []
            if hasattr(person_identity, 'camera_transitions'):
                camera_transitions = person_identity.camera_transitions
            
            # Extract trajectory path
            trajectory_path = []
            if trajectory and hasattr(trajectory, 'path_points'):
                trajectory_path = trajectory.path_points
            
            # Create tracking update
            return TrackingUpdate(
                person_id=int(person_identity.global_id.replace("person_", "")),
                global_id=person_identity.global_id,
                camera_transitions=camera_transitions,
                current_position=current_position,
                trajectory_path=trajectory_path,
                last_seen=person_identity.last_seen or datetime.now(timezone.utc),
                confidence=person_identity.identity_confidence,
                cameras_seen=list(person_identity.cameras_seen)
            )
            
        except Exception as e:
            logger.error(f"Error creating tracking update: {e}")
            return TrackingUpdate(
                person_id=0,
                global_id=person_identity.global_id,
                camera_transitions=[],
                current_position=current_position,
                trajectory_path=[],
                last_seen=datetime.now(timezone.utc),
                confidence=0.0,
                cameras_seen=[]
            )
    
    def _should_send_update(self, person_key: str) -> bool:
        """Check if update should be sent based on throttling."""
        try:
            current_time = time.time()
            
            if person_key in self.last_update_times:
                time_since_last = (current_time - self.last_update_times[person_key]) * 1000
                return time_since_last >= self.update_throttle_ms
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking update throttling: {e}")
            return True
    
    def _update_tracking_stats(self, update_latency: float):
        """Update tracking statistics."""
        try:
            self.tracking_stats["total_updates_sent"] += 1
            
            # Update average latency
            current_avg = self.tracking_stats["average_update_latency"]
            total_updates = self.tracking_stats["total_updates_sent"]
            
            self.tracking_stats["average_update_latency"] = (
                (current_avg * (total_updates - 1) + update_latency) / total_updates
            )
            
            # Update unique persons count
            self.tracking_stats["total_persons_tracked"] = len(self.active_persons)
            
        except Exception as e:
            logger.error(f"Error updating tracking stats: {e}")
    
    def get_active_persons(self, task_id: str) -> List[Dict[str, Any]]:
        """Get active persons for a task."""
        try:
            active_persons = []
            
            for person_key, tracking_update in self.active_persons.items():
                if person_key.startswith(f"{task_id}_"):
                    active_persons.append({
                        "person_id": tracking_update.person_id,
                        "global_id": tracking_update.global_id,
                        "last_seen": tracking_update.last_seen.isoformat(),
                        "confidence": tracking_update.confidence,
                        "cameras_seen": tracking_update.cameras_seen,
                        "current_position": {
                            "x": tracking_update.current_position.x,
                            "y": tracking_update.current_position.y,
                            "coordinate_system": tracking_update.current_position.coordinate_system.value
                        } if tracking_update.current_position else None
                    })
            
            return active_persons
            
        except Exception as e:
            logger.error(f"Error getting active persons: {e}")
            return []
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            **self.tracking_stats,
            "active_persons_count": len(self.active_persons),
            "camera_transitions_count": sum(len(transitions) for transitions in self.camera_transitions.values()),
            "throttle_settings": {
                "update_throttle_ms": self.update_throttle_ms
            }
        }
    
    def reset_stats(self):
        """Reset tracking statistics."""
        self.tracking_stats = {
            "total_updates_sent": 0,
            "total_persons_tracked": 0,
            "total_camera_transitions": 0,
            "average_update_latency": 0.0,
            "failed_updates": 0
        }
        
        self.active_persons.clear()
        self.camera_transitions.clear()
        self.last_update_times.clear()
        
        logger.info("TrackingHandler statistics reset")


# Global tracking handler instance
tracking_handler = TrackingHandler()