"""
Track ID value object for identifying tracked objects within cameras.

Provides type-safe tracking identification with camera-specific scoping.
"""
from dataclasses import dataclass
from typing import Optional

from app.domain.shared.value_objects.base_value_object import BaseValueObject
from app.domain.shared.value_objects.camera_id import CameraID


@dataclass(frozen=True)
class TrackID(BaseValueObject):
    """
    Track identifier value object.
    
    Represents a tracking ID that is unique within a camera's scope.
    Multiple cameras can have the same track ID for different objects.
    """
    
    id: int
    camera_id: CameraID
    
    def _validate(self) -> None:
        """Validate track ID."""
        if self.id < 0:
            raise ValueError("Track ID must be non-negative")
    
    @classmethod
    def create(cls, track_id: int, camera_id: CameraID) -> 'TrackID':
        """
        Create TrackID with validation.
        
        Args:
            track_id: Numeric track identifier
            camera_id: Camera where tracking occurs
            
        Returns:
            TrackID instance
        """
        return cls(id=track_id, camera_id=camera_id)
    
    @classmethod
    def from_legacy(cls, track_id: int, camera_id: str) -> 'TrackID':
        """
        Create TrackID from legacy format.
        
        Args:
            track_id: Legacy track ID
            camera_id: Legacy camera identifier string
            
        Returns:
            TrackID instance
        """
        return cls(
            id=track_id,
            camera_id=CameraID.from_legacy(camera_id)
        )
    
    @property
    def unique_key(self) -> str:
        """Get unique key combining camera and track ID."""
        return f"{self.camera_id.id}_{self.id}"
    
    @property
    def is_valid(self) -> bool:
        """Check if track ID is valid."""
        return self.id >= 0 and self.camera_id.is_valid_camera()
    
    def matches_camera(self, camera_id: CameraID) -> bool:
        """Check if track ID belongs to specific camera."""
        return self.camera_id == camera_id
    
    def to_legacy_format(self) -> int:
        """Convert to legacy track ID format (just the numeric ID)."""
        return self.id
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.camera_id.id}_{self.id}"


@dataclass(frozen=True)
class GlobalTrackID(BaseValueObject):
    """
    Global track identifier for cross-camera tracking.
    
    Represents a globally unique tracking ID that spans multiple cameras
    for the same person or object.
    """
    
    id: str
    
    def _validate(self) -> None:
        """Validate global track ID."""
        if not self.id.strip():
            raise ValueError("Global track ID cannot be empty")
    
    @classmethod
    def generate(cls, prefix: str = "global") -> 'GlobalTrackID':
        """
        Generate new global track ID.
        
        Args:
            prefix: Prefix for the ID
            
        Returns:
            New GlobalTrackID instance
        """
        from uuid import uuid4
        return cls(id=f"{prefix}_{uuid4().hex[:8]}")
    
    @classmethod
    def from_string(cls, global_id: str) -> 'GlobalTrackID':
        """
        Create GlobalTrackID from string.
        
        Args:
            global_id: Global identifier string
            
        Returns:
            GlobalTrackID instance
        """
        return cls(id=global_id.strip())
    
    @property
    def short_id(self) -> str:
        """Get shortened ID for display."""
        return self.id[:12] if len(self.id) > 12 else self.id
    
    def __str__(self) -> str:
        """String representation."""
        return self.id