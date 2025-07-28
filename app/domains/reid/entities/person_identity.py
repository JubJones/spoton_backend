"""
Person identity entity for cross-camera tracking.

Contains the core identity aggregate that represents a person
across multiple camera views.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum

from app.shared.types import CameraID

class IdentityStatus(Enum):
    """Person identity status."""
    ACTIVE = "active"
    LOST = "lost"
    MERGED = "merged"
    ARCHIVED = "archived"

@dataclass
class PersonIdentity:
    """Person identity aggregate representing a person across cameras."""
    
    global_id: str
    track_ids_by_camera: Dict[CameraID, int] = field(default_factory=dict)
    cameras_seen: Set[CameraID] = field(default_factory=set)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    status: IdentityStatus = IdentityStatus.ACTIVE
    
    # Confidence and quality metrics
    identity_confidence: float = 0.0
    feature_quality: float = 0.0
    track_stability: float = 0.0
    
    # Metadata
    total_detections: int = 0
    total_cameras: int = 0
    merge_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate person identity."""
        if not self.global_id:
            raise ValueError("Global ID cannot be empty")
        
        if not 0 <= self.identity_confidence <= 1:
            raise ValueError("Identity confidence must be between 0 and 1")
        
        if not 0 <= self.feature_quality <= 1:
            raise ValueError("Feature quality must be between 0 and 1")
        
        if not 0 <= self.track_stability <= 1:
            raise ValueError("Track stability must be between 0 and 1")
    
    def add_camera_track(self, camera_id: CameraID, track_id: int) -> 'PersonIdentity':
        """Add or update camera track for this identity."""
        new_track_ids = self.track_ids_by_camera.copy()
        new_track_ids[camera_id] = track_id
        
        new_cameras_seen = self.cameras_seen.copy()
        new_cameras_seen.add(camera_id)
        
        return PersonIdentity(
            global_id=self.global_id,
            track_ids_by_camera=new_track_ids,
            cameras_seen=new_cameras_seen,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            status=self.status,
            identity_confidence=self.identity_confidence,
            feature_quality=self.feature_quality,
            track_stability=self.track_stability,
            total_detections=self.total_detections,
            total_cameras=len(new_cameras_seen),
            merge_history=self.merge_history.copy()
        )
    
    def remove_camera_track(self, camera_id: CameraID) -> 'PersonIdentity':
        """Remove camera track from this identity."""
        new_track_ids = self.track_ids_by_camera.copy()
        if camera_id in new_track_ids:
            del new_track_ids[camera_id]
        
        new_cameras_seen = self.cameras_seen.copy()
        new_cameras_seen.discard(camera_id)
        
        return PersonIdentity(
            global_id=self.global_id,
            track_ids_by_camera=new_track_ids,
            cameras_seen=new_cameras_seen,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            status=self.status,
            identity_confidence=self.identity_confidence,
            feature_quality=self.feature_quality,
            track_stability=self.track_stability,
            total_detections=self.total_detections,
            total_cameras=len(new_cameras_seen),
            merge_history=self.merge_history.copy()
        )
    
    def update_last_seen(self, timestamp: datetime) -> 'PersonIdentity':
        """Update last seen timestamp."""
        first_seen = self.first_seen or timestamp
        
        return PersonIdentity(
            global_id=self.global_id,
            track_ids_by_camera=self.track_ids_by_camera.copy(),
            cameras_seen=self.cameras_seen.copy(),
            first_seen=first_seen,
            last_seen=timestamp,
            status=self.status,
            identity_confidence=self.identity_confidence,
            feature_quality=self.feature_quality,
            track_stability=self.track_stability,
            total_detections=self.total_detections,
            total_cameras=self.total_cameras,
            merge_history=self.merge_history.copy()
        )
    
    def update_confidence(self, confidence: float) -> 'PersonIdentity':
        """Update identity confidence."""
        return PersonIdentity(
            global_id=self.global_id,
            track_ids_by_camera=self.track_ids_by_camera.copy(),
            cameras_seen=self.cameras_seen.copy(),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            status=self.status,
            identity_confidence=confidence,
            feature_quality=self.feature_quality,
            track_stability=self.track_stability,
            total_detections=self.total_detections,
            total_cameras=self.total_cameras,
            merge_history=self.merge_history.copy()
        )
    
    def update_status(self, status: IdentityStatus) -> 'PersonIdentity':
        """Update identity status."""
        return PersonIdentity(
            global_id=self.global_id,
            track_ids_by_camera=self.track_ids_by_camera.copy(),
            cameras_seen=self.cameras_seen.copy(),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            status=status,
            identity_confidence=self.identity_confidence,
            feature_quality=self.feature_quality,
            track_stability=self.track_stability,
            total_detections=self.total_detections,
            total_cameras=self.total_cameras,
            merge_history=self.merge_history.copy()
        )
    
    def increment_detections(self, count: int = 1) -> 'PersonIdentity':
        """Increment total detection count."""
        return PersonIdentity(
            global_id=self.global_id,
            track_ids_by_camera=self.track_ids_by_camera.copy(),
            cameras_seen=self.cameras_seen.copy(),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            status=self.status,
            identity_confidence=self.identity_confidence,
            feature_quality=self.feature_quality,
            track_stability=self.track_stability,
            total_detections=self.total_detections + count,
            total_cameras=self.total_cameras,
            merge_history=self.merge_history.copy()
        )
    
    def merge_with(self, other: 'PersonIdentity') -> 'PersonIdentity':
        """Merge this identity with another identity."""
        # Combine track IDs
        combined_track_ids = self.track_ids_by_camera.copy()
        combined_track_ids.update(other.track_ids_by_camera)
        
        # Combine cameras seen
        combined_cameras = self.cameras_seen.union(other.cameras_seen)
        
        # Update timestamps
        first_seen = min(
            self.first_seen or datetime.max,
            other.first_seen or datetime.max
        )
        if first_seen == datetime.max:
            first_seen = None
        
        last_seen = max(
            self.last_seen or datetime.min,
            other.last_seen or datetime.min
        )
        if last_seen == datetime.min:
            last_seen = None
        
        # Update merge history
        merge_history = self.merge_history.copy()
        merge_history.append(other.global_id)
        
        return PersonIdentity(
            global_id=self.global_id,  # Keep this identity's global ID
            track_ids_by_camera=combined_track_ids,
            cameras_seen=combined_cameras,
            first_seen=first_seen,
            last_seen=last_seen,
            status=self.status,
            identity_confidence=max(self.identity_confidence, other.identity_confidence),
            feature_quality=max(self.feature_quality, other.feature_quality),
            track_stability=max(self.track_stability, other.track_stability),
            total_detections=self.total_detections + other.total_detections,
            total_cameras=len(combined_cameras),
            merge_history=merge_history
        )
    
    @property
    def is_active(self) -> bool:
        """Check if identity is active."""
        return self.status == IdentityStatus.ACTIVE
    
    @property
    def is_lost(self) -> bool:
        """Check if identity is lost."""
        return self.status == IdentityStatus.LOST
    
    @property
    def is_merged(self) -> bool:
        """Check if identity has been merged."""
        return self.status == IdentityStatus.MERGED
    
    @property
    def camera_count(self) -> int:
        """Get number of cameras this identity has been seen in."""
        return len(self.cameras_seen)
    
    @property
    def track_count(self) -> int:
        """Get number of active tracks for this identity."""
        return len(self.track_ids_by_camera)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds between first and last seen."""
        if not self.first_seen or not self.last_seen:
            return None
        
        return (self.last_seen - self.first_seen).total_seconds()
    
    @property
    def has_been_merged(self) -> bool:
        """Check if this identity has merged with others."""
        return len(self.merge_history) > 0
    
    def get_track_id_for_camera(self, camera_id: CameraID) -> Optional[int]:
        """Get track ID for specific camera."""
        return self.track_ids_by_camera.get(camera_id)
    
    def has_camera_track(self, camera_id: CameraID) -> bool:
        """Check if identity has track in specific camera."""
        return camera_id in self.track_ids_by_camera
    
    def get_active_cameras(self) -> List[CameraID]:
        """Get list of cameras with active tracks."""
        return list(self.track_ids_by_camera.keys())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "global_id": self.global_id,
            "track_ids_by_camera": dict(self.track_ids_by_camera),
            "cameras_seen": list(self.cameras_seen),
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "status": self.status.value,
            "identity_confidence": self.identity_confidence,
            "feature_quality": self.feature_quality,
            "track_stability": self.track_stability,
            "total_detections": self.total_detections,
            "total_cameras": self.total_cameras,
            "merge_history": self.merge_history.copy(),
            "camera_count": self.camera_count,
            "track_count": self.track_count,
            "duration_seconds": self.duration_seconds,
            "has_been_merged": self.has_been_merged
        }