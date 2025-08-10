"""
Unified Detection entity for the detection domain.

Replaces both legacy Detection and the existing domain Detection models
with a single, well-designed domain entity.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Any, Dict
from uuid import UUID, uuid4
import numpy as np

from app.domain.shared.entities.base_entity import BaseEntity
from app.domain.shared.value_objects.bounding_box import BoundingBox
from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.frame_id import FrameID
from app.domain.detection.value_objects.confidence import Confidence
from app.domain.detection.value_objects.detection_class import DetectionClass


@dataclass
class Detection(BaseEntity):
    """
    Unified Detection entity representing a detected object.
    
    Combines the best aspects of legacy and domain models while providing
    proper domain-driven design structure with value objects.
    """
    
    # Core detection data
    camera_id: CameraID
    frame_id: FrameID
    bbox: BoundingBox
    confidence: Confidence
    detection_class: DetectionClass
    timestamp: datetime
    
    # Optional tracking data
    track_id: Optional[int] = None
    global_id: Optional[str] = None
    
    # Optional metadata
    feature_vector: Optional[np.ndarray] = None
    raw_detection_data: Optional[Dict[str, Any]] = None
    processing_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize entity after dataclass creation."""
        super().__init__()  # Initialize BaseEntity
        self._validate_detection()
    
    def _validate_detection(self) -> None:
        """Validate detection data consistency."""
        if self.timestamp > datetime.utcnow():
            raise ValueError("Detection timestamp cannot be in the future")
        
        # Validate that frame timestamp is consistent with detection timestamp
        # (allowing for some processing delay)
        max_delay_seconds = 60
        if abs((self.timestamp - datetime.utcnow()).total_seconds()) > max_delay_seconds:
            pass  # Log warning instead of raising error for flexibility
    
    @classmethod
    def create(
        cls,
        camera_id: CameraID,
        frame_id: FrameID,
        bbox: BoundingBox,
        confidence: Confidence,
        detection_class: DetectionClass,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> 'Detection':
        """
        Factory method to create a new Detection.
        
        Args:
            camera_id: Camera that produced detection
            frame_id: Frame where detection occurred
            bbox: Bounding box of detected object
            confidence: Detection confidence score
            detection_class: Class of detected object
            timestamp: Detection timestamp (defaults to now)
            **kwargs: Additional optional parameters
            
        Returns:
            New Detection instance
        """
        return cls(
            camera_id=camera_id,
            frame_id=frame_id,
            bbox=bbox,
            confidence=confidence,
            detection_class=detection_class,
            timestamp=timestamp or datetime.utcnow(),
            **kwargs
        )
    
    @classmethod
    def from_legacy_detection(
        cls,
        legacy_detection: Any,  # LegacyDetection type
        camera_id: CameraID,
        frame_id: FrameID,
        timestamp: Optional[datetime] = None
    ) -> 'Detection':
        """
        Create Detection from legacy Detection model.
        
        Args:
            legacy_detection: Legacy detection instance
            camera_id: Camera identifier
            frame_id: Frame identifier
            timestamp: Detection timestamp
            
        Returns:
            New Detection instance
        """
        from app.core.model_migration import ModelMigrationHelper
        
        # Convert legacy bounding box
        unified_bbox = ModelMigrationHelper.migrate_legacy_bbox_to_unified(legacy_detection.bbox)
        
        # Convert confidence
        confidence = Confidence.from_float(legacy_detection.confidence)
        
        # Convert detection class
        detection_class = DetectionClass.from_legacy_id(legacy_detection.class_id)
        
        return cls.create(
            camera_id=camera_id,
            frame_id=frame_id,
            bbox=unified_bbox,
            confidence=confidence,
            detection_class=detection_class,
            timestamp=timestamp,
            raw_detection_data={
                'legacy_class_name': getattr(legacy_detection, 'class_name', None)
            }
        )
    
    # Properties for easy access
    @property
    def is_person(self) -> bool:
        """Check if detection is a person."""
        return self.detection_class.is_person
    
    @property
    def is_vehicle(self) -> bool:
        """Check if detection is a vehicle."""
        return self.detection_class.is_vehicle
    
    @property
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if detection has high confidence."""
        return self.confidence.above_threshold(threshold)
    
    @property
    def bbox_area(self) -> float:
        """Get bounding box area."""
        return self.bbox.area
    
    @property
    def bbox_center(self) -> tuple[float, float]:
        """Get bounding box center coordinates."""
        return (self.bbox.center_x, self.bbox.center_y)
    
    @property
    def has_tracking_id(self) -> bool:
        """Check if detection has tracking ID assigned."""
        return self.track_id is not None
    
    @property
    def has_global_id(self) -> bool:
        """Check if detection has global ID assigned."""
        return self.global_id is not None
    
    @property
    def has_features(self) -> bool:
        """Check if detection has feature vector."""
        return self.feature_vector is not None
    
    # Modification methods (return new instances for immutability)
    def with_track_id(self, track_id: int) -> 'Detection':
        """
        Create new detection with assigned track ID.
        
        Args:
            track_id: Track identifier
            
        Returns:
            New Detection instance with track ID
        """
        new_detection = Detection(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            bbox=self.bbox,
            confidence=self.confidence,
            detection_class=self.detection_class,
            timestamp=self.timestamp,
            track_id=track_id,
            global_id=self.global_id,
            feature_vector=self.feature_vector,
            raw_detection_data=self.raw_detection_data,
            processing_metadata=self.processing_metadata.copy()
        )
        new_detection._id = self._id  # Keep same entity ID
        new_detection.increment_version()
        return new_detection
    
    def with_global_id(self, global_id: str) -> 'Detection':
        """
        Create new detection with assigned global ID.
        
        Args:
            global_id: Global identifier for cross-camera tracking
            
        Returns:
            New Detection instance with global ID
        """
        new_detection = Detection(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            bbox=self.bbox,
            confidence=self.confidence,
            detection_class=self.detection_class,
            timestamp=self.timestamp,
            track_id=self.track_id,
            global_id=global_id,
            feature_vector=self.feature_vector,
            raw_detection_data=self.raw_detection_data,
            processing_metadata=self.processing_metadata.copy()
        )
        new_detection._id = self._id  # Keep same entity ID
        new_detection.increment_version()
        return new_detection
    
    def with_features(self, feature_vector: np.ndarray) -> 'Detection':
        """
        Create new detection with feature vector.
        
        Args:
            feature_vector: Feature vector for re-identification
            
        Returns:
            New Detection instance with features
        """
        new_detection = Detection(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            bbox=self.bbox,
            confidence=self.confidence,
            detection_class=self.detection_class,
            timestamp=self.timestamp,
            track_id=self.track_id,
            global_id=self.global_id,
            feature_vector=feature_vector.copy() if feature_vector is not None else None,
            raw_detection_data=self.raw_detection_data,
            processing_metadata=self.processing_metadata.copy()
        )
        new_detection._id = self._id  # Keep same entity ID
        new_detection.increment_version()
        return new_detection
    
    def update_confidence(self, new_confidence: Confidence) -> 'Detection':
        """
        Create new detection with updated confidence.
        
        Args:
            new_confidence: New confidence score
            
        Returns:
            New Detection instance with updated confidence
        """
        new_detection = Detection(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            bbox=self.bbox,
            confidence=new_confidence,
            detection_class=self.detection_class,
            timestamp=self.timestamp,
            track_id=self.track_id,
            global_id=self.global_id,
            feature_vector=self.feature_vector,
            raw_detection_data=self.raw_detection_data,
            processing_metadata=self.processing_metadata.copy()
        )
        new_detection._id = self._id  # Keep same entity ID
        new_detection.increment_version()
        return new_detection
    
    def add_metadata(self, key: str, value: Any) -> 'Detection':
        """
        Add processing metadata to detection.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            New Detection instance with added metadata
        """
        new_metadata = self.processing_metadata.copy()
        new_metadata[key] = value
        
        new_detection = Detection(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            bbox=self.bbox,
            confidence=self.confidence,
            detection_class=self.detection_class,
            timestamp=self.timestamp,
            track_id=self.track_id,
            global_id=self.global_id,
            feature_vector=self.feature_vector,
            raw_detection_data=self.raw_detection_data,
            processing_metadata=new_metadata
        )
        new_detection._id = self._id  # Keep same entity ID
        new_detection.increment_version()
        return new_detection
    
    # Legacy compatibility methods
    def to_legacy_detection(self) -> Any:
        """
        Convert to legacy Detection format.
        
        Returns:
            Legacy Detection instance
        """
        from app.core.model_migration import ModelMigrationHelper
        
        return ModelMigrationHelper.migrate_detection_to_legacy(
            unified_bbox=self.bbox,
            confidence=float(self.confidence),
            class_id=self.detection_class.to_legacy_format(),
            class_name=self.detection_class.class_name
        )
    
    def to_tracker_format(self) -> np.ndarray:
        """
        Convert to tracker format [x1, y1, x2, y2, conf, cls_id].
        
        Returns:
            Numpy array in tracker format
        """
        x1, y1, x2, y2 = self.bbox.to_coordinates()
        return np.array([
            x1, y1, x2, y2,
            float(self.confidence),
            self.detection_class.to_legacy_format()
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert detection to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': str(self.id),
            'camera_id': str(self.camera_id),
            'frame_id': str(self.frame_id),
            'bbox': self.bbox.to_dict(),
            'confidence': float(self.confidence),
            'detection_class': {
                'type': self.detection_class.class_type.value,
                'id': self.detection_class.class_id,
                'name': self.detection_class.class_name
            },
            'timestamp': self.timestamp.isoformat(),
            'track_id': self.track_id,
            'global_id': self.global_id,
            'has_features': self.has_features,
            'version': self.version,
            'processing_metadata': self.processing_metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"Detection({self.detection_class.display_name}, "
                f"conf={self.confidence}, cam={self.camera_id}, "
                f"track={self.track_id}, global={self.global_id})")


@dataclass
class DetectionBatch:
    """
    Batch of detections for efficient processing.
    
    Represents multiple detections from the same frame or processing cycle.
    """
    
    detections: List[Detection]
    batch_id: UUID = field(default_factory=uuid4)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate batch after creation."""
        if self.processing_time_ms < 0:
            raise ValueError("Processing time must be non-negative")
    
    @classmethod
    def create(cls, detections: List[Detection], **kwargs) -> 'DetectionBatch':
        """
        Create detection batch.
        
        Args:
            detections: List of Detection instances
            **kwargs: Additional batch metadata
            
        Returns:
            DetectionBatch instance
        """
        return cls(detections=detections, **kwargs)
    
    @property
    def detection_count(self) -> int:
        """Get number of detections in batch."""
        return len(self.detections)
    
    @property
    def person_detections(self) -> List[Detection]:
        """Get only person detections."""
        return [d for d in self.detections if d.is_person]
    
    @property
    def person_count(self) -> int:
        """Get number of person detections."""
        return len(self.person_detections)
    
    @property
    def high_confidence_detections(self, threshold: float = 0.7) -> List[Detection]:
        """Get high confidence detections."""
        return [d for d in self.detections if d.is_high_confidence(threshold)]
    
    @property
    def cameras(self) -> List[CameraID]:
        """Get unique cameras in batch."""
        return list(set(d.camera_id for d in self.detections))
    
    def get_detections_by_camera(self, camera_id: CameraID) -> List[Detection]:
        """Get detections for specific camera."""
        return [d for d in self.detections if d.camera_id == camera_id]
    
    def get_detections_by_class(self, detection_class: DetectionClass) -> List[Detection]:
        """Get detections for specific class."""
        return [d for d in self.detections if d.detection_class == detection_class]
    
    def filter_by_confidence(self, min_confidence: Confidence) -> 'DetectionBatch':
        """
        Create new batch with detections above confidence threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered DetectionBatch
        """
        filtered_detections = [
            d for d in self.detections 
            if d.confidence.meets_threshold(min_confidence)
        ]
        
        return DetectionBatch.create(
            detections=filtered_detections,
            processing_time_ms=self.processing_time_ms,
            metadata={
                **self.metadata,
                'filtered_from': str(self.batch_id),
                'original_count': self.detection_count,
                'confidence_threshold': float(min_confidence)
            }
        )
    
    def to_legacy_format(self) -> List[Any]:
        """Convert all detections to legacy format."""
        return [d.to_legacy_detection() for d in self.detections]