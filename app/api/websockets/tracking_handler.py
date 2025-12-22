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
from app.core.config import settings
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
        # Pruning controls to avoid unbounded growth
        self.max_persons_kept = 1000
        self.prune_after_seconds = 60.0
        
        logger.info("TrackingHandler initialized")
    
    async def send_tracking_update(
        self,
        task_id: str,
        person_identity: Any,
        current_position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
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
            gid = getattr(person_identity, 'global_id', None)
            if gid is None and isinstance(person_identity, dict):
                gid = person_identity.get('global_id')
            person_key = f"{task_id}_{gid}"
            if not self._should_send_update(person_key):
                return True  # Skip this update
            
            # Create tracking update
            tracking_update = self._create_tracking_update(person_identity, current_position, trajectory)
            
            # Cap trajectory path length for payload size
            traj_limit = int(getattr(settings, 'WS_TRACKING_TRAJECTORY_POINTS_LIMIT', 50))
            capped_traj = tracking_update.trajectory_path[-traj_limit:] if tracking_update.trajectory_path else []

            # Create message
            message = {
                "type": MessageType.TRACKING_UPDATE.value,
                "task_id": task_id,
                "person_id": int(str(gid).replace("person_", "")) if gid is not None else 0,
                "global_id": gid,
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
                    for coord in capped_traj
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
                
                # logger.debug(f"Sent tracking update for person {gid}")
            else:
                self.tracking_stats["failed_updates"] += 1
                logger.warning(f"Failed to send tracking update for person {gid}")
            
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
        Send batch tracking updates using per-camera messaging to avoid WebSocket size limits.
        
        Args:
            task_id: Task identifier
            person_updates: List of person update data
            
        Returns:
            True if all camera messages sent successfully
        """
        try:
            # Group person updates by camera to avoid oversized WebSocket messages
            camera_updates = {}
            for person_update in person_updates:
                camera_id = person_update.get("camera_id")
                if camera_id:
                    if camera_id not in camera_updates:
                        camera_updates[camera_id] = []
                    camera_updates[camera_id].append(person_update)
            
            # Send individual messages per camera to avoid oversized WebSocket messages
            all_success = True
            total_sent = 0
            batch_timestamp = datetime.now(timezone.utc).isoformat()
            
            for camera_id, camera_persons in camera_updates.items():
                camera_message = {
                    "type": "tracking_update",
                    "task_id": task_id,
                    "camera_id": camera_id,
                    "update_count": len(camera_persons),
                    "persons": camera_persons,
                    "batch_timestamp": batch_timestamp
                }
                
                # Send individual camera message
                camera_success = await binary_websocket_manager.send_json_message(
                    task_id, camera_message, MessageType.TRACKING_UPDATE
                )
                
                if camera_success:
                    total_sent += len(camera_persons)
                    # logger.debug(f"Sent tracking updates for camera {camera_id}: {len(camera_persons)} persons")
                else:
                    all_success = False
                    logger.warning(f"Failed to send tracking updates for camera {camera_id}")
            
            if all_success:
                self.tracking_stats["total_updates_sent"] += total_sent
                # logger.debug(f"All camera tracking updates sent successfully: {total_sent} total persons")
            
            return all_success
            
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
        person_identity: Any,
        current_position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
    ) -> TrackingUpdate:
        """Create tracking update from person identity."""
        try:
            # Extract camera transitions
            camera_transitions = []
            if hasattr(person_identity, 'camera_transitions'):
                camera_transitions = getattr(person_identity, 'camera_transitions')
            elif isinstance(person_identity, dict):
                camera_transitions = person_identity.get('camera_transitions', [])
            
            # Extract trajectory path
            trajectory_path = []
            if trajectory and hasattr(trajectory, 'path_points'):
                trajectory_path = trajectory.path_points
            
            # Create tracking update
            gid = getattr(person_identity, 'global_id', None)
            if gid is None and isinstance(person_identity, dict):
                gid = person_identity.get('global_id')
            return TrackingUpdate(
                person_id=int(str(gid).replace("person_", "")) if gid is not None else 0,
                global_id=gid,
                camera_transitions=camera_transitions,
                current_position=current_position,
                trajectory_path=trajectory_path,
                last_seen=(getattr(person_identity, 'last_seen', None) or datetime.now(timezone.utc)),
                confidence=float(getattr(person_identity, 'identity_confidence', 0.0)) if not isinstance(person_identity, dict) else float(person_identity.get('identity_confidence', 0.0)),
                cameras_seen=list(getattr(person_identity, 'cameras_seen', [])) if not isinstance(person_identity, dict) else list(person_identity.get('cameras_seen', []))
            )
            
        except Exception as e:
            logger.error(f"Error creating tracking update: {e}")
            return TrackingUpdate(
                person_id=0,
                global_id=gid,
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
        """Get tracking statistics (prunes stale state)."""
        try:
            self._prune_stale_state()
        except Exception:
            pass
        return {
            **self.tracking_stats,
            "active_persons_count": len(self.active_persons),
            "camera_transitions_count": sum(len(transitions) for transitions in self.camera_transitions.values()),
            "throttle_settings": {
                "update_throttle_ms": self.update_throttle_ms
            }
        }

    def _prune_stale_state(self):
        """Remove persons with no updates for a while and cap total size."""
        now = time.time()
        # Remove entries older than threshold
        stale_keys = [k for k, ts in self.last_update_times.items() if (now - ts) > self.prune_after_seconds]
        for k in stale_keys:
            self.last_update_times.pop(k, None)
            self.active_persons.pop(k, None)
        # Cap total size by removing oldest
        if len(self.active_persons) > self.max_persons_kept:
            ordered = sorted(self.last_update_times.items(), key=lambda kv: kv[1])
            excess = len(self.active_persons) - self.max_persons_kept
            for k, _ in ordered[:excess]:
                self.last_update_times.pop(k, None)
                self.active_persons.pop(k, None)
    
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
