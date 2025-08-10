"""
Track entity representing a tracked object within a camera view.

Replaces legacy TrackedObject with proper domain-driven design.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np

from app.domain.shared.entities.base_entity import BaseEntity
from app.domain.shared.value_objects.bounding_box import BoundingBox
from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.detection.entities.detection import Detection
from app.domain.tracking.value_objects.track_id import TrackID, GlobalTrackID
from app.domain.tracking.value_objects.velocity import Velocity


class TrackState(Enum):
    """Track state enumeration."""
    ACTIVE = "active"
    TENTATIVE = "tentative"
    LOST = "lost"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class Track(BaseEntity):
    """
    Track entity representing a tracked object within a camera.
    
    Maintains state and history for a single tracked object,
    providing rich domain behavior for tracking operations.
    """
    
    # Core tracking data
    track_id: TrackID
    camera_id: CameraID
    state: TrackState
    
    # Current detection data
    current_bbox: BoundingBox
    current_confidence: float
    last_detection: Optional[Detection] = None
    
    # Tracking metadata
    age: int = 0  # Number of frames tracked
    hits: int = 0  # Number of successful detections
    hit_streak: int = 0  # Consecutive detection count
    time_since_update: int = 0  # Frames since last update
    
    # Movement data
    velocity: Optional[Velocity] = None
    previous_bbox: Optional[BoundingBox] = None
    
    # Cross-camera tracking
    global_id: Optional[GlobalTrackID] = None
    
    # Re-identification features
    feature_vector: Optional[np.ndarray] = None
    feature_history: List[np.ndarray] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Additional metadata
    class_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize entity after dataclass creation."""
        super().__init__()  # Initialize BaseEntity
        self._validate_track()
    
    def _validate_track(self) -> None:
        """Validate track data consistency."""
        if self.age < 0:
            raise ValueError("Track age cannot be negative")
        if self.hits < 0:
            raise ValueError("Track hits cannot be negative")
        if self.hit_streak < 0:
            raise ValueError("Track hit streak cannot be negative")
        if self.time_since_update < 0:
            raise ValueError("Time since update cannot be negative")
        if not (0.0 <= self.current_confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @classmethod
    def create(
        cls,
        track_id: TrackID,
        camera_id: CameraID,
        initial_detection: Detection,
        state: TrackState = TrackState.TENTATIVE
    ) -> 'Track':
        """
        Create new track from initial detection.
        
        Args:
            track_id: Track identifier
            camera_id: Camera identifier  
            initial_detection: Initial detection
            state: Initial track state
            
        Returns:
            New Track instance
        """
        return cls(
            track_id=track_id,
            camera_id=camera_id,
            state=state,
            current_bbox=initial_detection.bbox,
            current_confidence=float(initial_detection.confidence),
            last_detection=initial_detection,
            age=1,
            hits=1,
            hit_streak=1,
            time_since_update=0,
            class_id=initial_detection.detection_class.to_legacy_format(),
            feature_vector=initial_detection.feature_vector.copy() if initial_detection.feature_vector is not None else None
        )
    
    @classmethod
    def from_legacy_tracked_object(
        cls,
        legacy_tracked: Any,  # LegacyTrackedObject type
        camera_id: CameraID
    ) -> 'Track':
        """
        Create Track from legacy TrackedObject.
        
        Args:
            legacy_tracked: Legacy TrackedObject instance
            camera_id: Camera identifier
            
        Returns:
            New Track instance
        """
        from app.core.model_migration import ModelMigrationHelper
        
        # Convert legacy bounding box
        unified_bbox = ModelMigrationHelper.migrate_legacy_bbox_to_unified(legacy_tracked.bbox)
        
        # Create track ID
        track_id = TrackID.create(legacy_tracked.track_id, camera_id)
        
        # Determine state
        state_mapping = {
            'active': TrackState.ACTIVE,
            'lost': TrackState.LOST,
            'tentative': TrackState.TENTATIVE
        }
        state = state_mapping.get(getattr(legacy_tracked, 'state', None), TrackState.ACTIVE)
        
        return cls(
            track_id=track_id,
            camera_id=camera_id,
            state=state,
            current_bbox=unified_bbox,
            current_confidence=legacy_tracked.confidence or 0.5,
            age=getattr(legacy_tracked, 'age', 1),
            hits=1,
            hit_streak=1,
            time_since_update=0,
            global_id=GlobalTrackID.from_string(str(legacy_tracked.global_id)) if legacy_tracked.global_id else None,
            feature_vector=legacy_tracked.feature_embedding.copy() if legacy_tracked.feature_embedding is not None else None,
            class_id=legacy_tracked.class_id
        )
    
    # Properties
    @property
    def is_active(self) -> bool:
        """Check if track is active."""
        return self.state == TrackState.ACTIVE
    
    @property
    def is_tentative(self) -> bool:
        """Check if track is tentative."""
        return self.state == TrackState.TENTATIVE
    
    @property
    def is_lost(self) -> bool:
        """Check if track is lost."""
        return self.state == TrackState.LOST
    
    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED
    
    @property
    def is_deleted(self) -> bool:
        """Check if track is deleted."""
        return self.state == TrackState.DELETED
    
    @property
    def has_global_id(self) -> bool:
        """Check if track has global ID."""
        return self.global_id is not None
    
    @property
    def has_features(self) -> bool:
        """Check if track has feature vector."""
        return self.feature_vector is not None
    
    @property
    def bbox_center(self) -> tuple[float, float]:
        """Get current bounding box center."""
        return (self.current_bbox.center_x, self.current_bbox.center_y)
    
    @property
    def bbox_area(self) -> float:
        """Get current bounding box area."""
        return self.current_bbox.area
    
    @property
    def is_moving(self, threshold: float = 0.1) -> bool:
        """Check if track is moving."""
        return self.velocity is not None and self.velocity.is_moving(threshold)
    
    @property
    def age_seconds(self) -> float:
        """Get track age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def time_since_update_seconds(self) -> float:
        """Get time since last update in seconds."""
        return (datetime.utcnow() - self.last_updated_at).total_seconds()
    
    # State transition methods
    def update_with_detection(self, detection: Detection) -> 'Track':
        """
        Update track with new detection.
        
        Args:
            detection: New detection to update with
            
        Returns:
            Updated Track instance
        """
        # Calculate velocity if we have previous bbox
        new_velocity = None
        if self.previous_bbox is not None:
            try:
                new_velocity = Velocity.from_points(
                    point1=self.bbox_center,
                    point2=(detection.bbox.center_x, detection.bbox.center_y),
                    time_delta=1.0,  # Assume 1 frame delta
                    units="pixels/frame"
                )
            except ValueError:
                new_velocity = self.velocity  # Keep previous velocity
        
        # Update track
        new_track = Track(
            track_id=self.track_id,
            camera_id=self.camera_id,
            state=TrackState.ACTIVE if self.state == TrackState.TENTATIVE else self.state,
            current_bbox=detection.bbox,
            current_confidence=float(detection.confidence),
            last_detection=detection,
            age=self.age + 1,
            hits=self.hits + 1,
            hit_streak=self.hit_streak + 1,
            time_since_update=0,
            velocity=new_velocity,
            previous_bbox=self.current_bbox,
            global_id=self.global_id,
            feature_vector=detection.feature_vector.copy() if detection.feature_vector is not None else self.feature_vector,
            feature_history=self._update_feature_history(detection.feature_vector),
            created_at=self.created_at,
            last_updated_at=datetime.utcnow(),
            class_id=detection.detection_class.to_legacy_format(),
            metadata=self.metadata.copy()
        )
        
        new_track._id = self._id  # Keep same entity ID
        new_track.increment_version()
        return new_track
    
    def predict_next_position(self) -> BoundingBox:
        """
        Predict next bounding box position based on velocity.
        
        Returns:
            Predicted bounding box
        """
        if self.velocity is None or self.velocity.is_stationary():
            return self.current_bbox  # Return current position if stationary
        
        # Predict center position
        current_center = self.bbox_center
        predicted_center = self.velocity.predict_position(current_center, 1.0)
        
        # Create predicted bounding box with same size
        predicted_bbox = BoundingBox.from_center(
            cx=predicted_center[0],
            cy=predicted_center[1],
            w=self.current_bbox.width,
            h=self.current_bbox.height,
            normalized=self.current_bbox.normalized
        )
        
        return predicted_bbox
    
    def mark_missed(self) -> 'Track':
        """
        Mark track as missed (no detection in current frame).
        
        Returns:
            Updated Track instance
        """
        new_state = self.state
        if self.time_since_update >= 5:  # Threshold for marking as lost
            new_state = TrackState.LOST
        
        new_track = Track(
            track_id=self.track_id,
            camera_id=self.camera_id,
            state=new_state,
            current_bbox=self.current_bbox,
            current_confidence=self.current_confidence,
            last_detection=self.last_detection,
            age=self.age + 1,
            hits=self.hits,
            hit_streak=0,  # Reset hit streak
            time_since_update=self.time_since_update + 1,
            velocity=self.velocity,
            previous_bbox=self.previous_bbox,
            global_id=self.global_id,
            feature_vector=self.feature_vector,
            feature_history=self.feature_history.copy(),
            created_at=self.created_at,
            last_updated_at=datetime.utcnow(),
            class_id=self.class_id,
            metadata=self.metadata.copy()
        )
        
        new_track._id = self._id  # Keep same entity ID
        new_track.increment_version()
        return new_track
    
    def assign_global_id(self, global_id: GlobalTrackID) -> 'Track':
        """
        Assign global ID for cross-camera tracking.
        
        Args:
            global_id: Global track identifier
            
        Returns:
            Updated Track instance
        """
        new_track = Track(
            track_id=self.track_id,
            camera_id=self.camera_id,
            state=self.state,
            current_bbox=self.current_bbox,
            current_confidence=self.current_confidence,
            last_detection=self.last_detection,
            age=self.age,
            hits=self.hits,
            hit_streak=self.hit_streak,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
            previous_bbox=self.previous_bbox,
            global_id=global_id,
            feature_vector=self.feature_vector,
            feature_history=self.feature_history.copy(),
            created_at=self.created_at,
            last_updated_at=datetime.utcnow(),
            class_id=self.class_id,
            metadata=self.metadata.copy()
        )
        
        new_track._id = self._id  # Keep same entity ID
        new_track.increment_version()
        return new_track
    
    def update_features(self, new_features: np.ndarray) -> 'Track':
        """
        Update track features for re-identification.
        
        Args:
            new_features: New feature vector
            
        Returns:
            Updated Track instance
        """
        new_track = Track(
            track_id=self.track_id,
            camera_id=self.camera_id,
            state=self.state,
            current_bbox=self.current_bbox,
            current_confidence=self.current_confidence,
            last_detection=self.last_detection,
            age=self.age,
            hits=self.hits,
            hit_streak=self.hit_streak,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
            previous_bbox=self.previous_bbox,
            global_id=self.global_id,
            feature_vector=new_features.copy(),
            feature_history=self._update_feature_history(new_features),
            created_at=self.created_at,
            last_updated_at=datetime.utcnow(),
            class_id=self.class_id,
            metadata=self.metadata.copy()
        )
        
        new_track._id = self._id  # Keep same entity ID
        new_track.increment_version()
        return new_track
    
    def mark_deleted(self) -> 'Track':
        """
        Mark track as deleted.
        
        Returns:
            Updated Track instance
        """
        new_track = Track(
            track_id=self.track_id,
            camera_id=self.camera_id,
            state=TrackState.DELETED,
            current_bbox=self.current_bbox,
            current_confidence=self.current_confidence,
            last_detection=self.last_detection,
            age=self.age,
            hits=self.hits,
            hit_streak=self.hit_streak,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
            previous_bbox=self.previous_bbox,
            global_id=self.global_id,
            feature_vector=self.feature_vector,
            feature_history=self.feature_history.copy(),
            created_at=self.created_at,
            last_updated_at=datetime.utcnow(),
            class_id=self.class_id,
            metadata=self.metadata.copy()
        )
        
        new_track._id = self._id  # Keep same entity ID
        new_track.increment_version()
        return new_track
    
    def _update_feature_history(self, new_features: Optional[np.ndarray]) -> List[np.ndarray]:
        """Update feature history with new features."""
        if new_features is None:
            return self.feature_history.copy()
        
        new_history = self.feature_history.copy()
        new_history.append(new_features.copy())
        
        # Keep only last N features for efficiency
        max_history = 10
        if len(new_history) > max_history:
            new_history = new_history[-max_history:]
        
        return new_history
    
    def get_average_features(self) -> Optional[np.ndarray]:
        """
        Get average feature vector from history.
        
        Returns:
            Average feature vector or None if no features
        """
        if not self.feature_history:
            return self.feature_vector
        
        if self.feature_vector is not None:
            all_features = self.feature_history + [self.feature_vector]
        else:
            all_features = self.feature_history
        
        if not all_features:
            return None
        
        return np.mean(all_features, axis=0)
    
    # Legacy compatibility methods
    def to_legacy_tracked_object(self) -> Any:
        """
        Convert to legacy TrackedObject format.
        
        Returns:
            Legacy TrackedObject instance
        """
        from app.core.model_migration import ModelMigrationHelper
        
        return ModelMigrationHelper.migrate_tracked_object_to_legacy(
            unified_bbox=self.current_bbox,
            track_id=self.track_id.to_legacy_format(),
            confidence=self.current_confidence,
            class_id=self.class_id,
            global_id=int(self.global_id.id) if self.global_id else None,
            feature_embedding=self.feature_vector,
            state=self.state.value,
            age=self.age
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert track to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': str(self.id),
            'track_id': str(self.track_id),
            'camera_id': str(self.camera_id),
            'state': self.state.value,
            'current_bbox': self.current_bbox.to_dict(),
            'current_confidence': self.current_confidence,
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update,
            'velocity': self.velocity.to_dict() if self.velocity else None,
            'global_id': str(self.global_id) if self.global_id else None,
            'has_features': self.has_features,
            'feature_history_length': len(self.feature_history),
            'created_at': self.created_at.isoformat(),
            'last_updated_at': self.last_updated_at.isoformat(),
            'age_seconds': self.age_seconds,
            'is_moving': self.is_moving,
            'version': self.version,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        global_id_str = f", global={self.global_id.short_id}" if self.global_id else ""
        return (f"Track({self.track_id}, {self.state.value}, "
                f"age={self.age}, hits={self.hits}{global_id_str})")