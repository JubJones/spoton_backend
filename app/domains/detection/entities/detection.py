"""
Detection domain entities.

Contains core detection domain objects and value objects.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from enum import Enum

from app.shared.types import CameraID

class DetectionClass(Enum):
    """Detection class enumeration."""
    PERSON = 1
    VEHICLE = 2
    OBJECT = 3

@dataclass(frozen=True)
class BoundingBox:
    """Bounding box value object."""
    x: float
    y: float
    width: float
    height: float
    normalized: bool = False
    
    def __post_init__(self):
        """Validate bounding box parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding box width and height must be positive")
        
        if self.normalized:
            if not (0 <= self.x <= 1 and 0 <= self.y <= 1):
                raise ValueError("Normalized coordinates must be between 0 and 1")
            if not (0 <= self.width <= 1 and 0 <= self.height <= 1):
                raise ValueError("Normalized dimensions must be between 0 and 1")
    
    @property
    def x2(self) -> float:
        """Right edge of bounding box."""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge of bounding box."""
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        """Center X coordinate."""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Center Y coordinate."""
        return self.y + self.height / 2
    
    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height
    
    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x2, self.y2)
    
    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        return (self.x, self.y, self.width, self.height)
    
    def scale(self, scale_x: float, scale_y: float) -> 'BoundingBox':
        """Scale bounding box by given factors."""
        return BoundingBox(
            x=self.x * scale_x,
            y=self.y * scale_y,
            width=self.width * scale_x,
            height=self.height * scale_y,
            normalized=False
        )
    
    def normalize(self, image_width: int, image_height: int) -> 'BoundingBox':
        """Normalize bounding box to [0, 1] range."""
        if self.normalized:
            return self
        
        return BoundingBox(
            x=self.x / image_width,
            y=self.y / image_height,
            width=self.width / image_width,
            height=self.height / image_height,
            normalized=True
        )
    
    def denormalize(self, image_width: int, image_height: int) -> 'BoundingBox':
        """Denormalize bounding box to pixel coordinates."""
        if not self.normalized:
            return self
        
        return BoundingBox(
            x=self.x * image_width,
            y=self.y * image_height,
            width=self.width * image_width,
            height=self.height * image_height,
            normalized=False
        )

@dataclass
class Detection:
    """Detection entity representing a detected object."""
    
    id: str
    camera_id: CameraID
    bbox: BoundingBox
    confidence: float
    class_id: DetectionClass
    timestamp: datetime
    frame_index: int
    
    # Optional metadata
    track_id: Optional[int] = None
    global_id: Optional[str] = None
    feature_vector: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate detection parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.frame_index < 0:
            raise ValueError("Frame index must be non-negative")
    
    @property
    def is_person(self) -> bool:
        """Check if detection is a person."""
        return self.class_id == DetectionClass.PERSON
    
    @property
    def bbox_area(self) -> float:
        """Get bounding box area."""
        return self.bbox.area
    
    def update_confidence(self, new_confidence: float) -> 'Detection':
        """Create new detection with updated confidence."""
        return Detection(
            id=self.id,
            camera_id=self.camera_id,
            bbox=self.bbox,
            confidence=new_confidence,
            class_id=self.class_id,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            track_id=self.track_id,
            global_id=self.global_id,
            feature_vector=self.feature_vector
        )
    
    def assign_track_id(self, track_id: int) -> 'Detection':
        """Create new detection with assigned track ID."""
        return Detection(
            id=self.id,
            camera_id=self.camera_id,
            bbox=self.bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            track_id=track_id,
            global_id=self.global_id,
            feature_vector=self.feature_vector
        )
    
    def assign_global_id(self, global_id: str) -> 'Detection':
        """Create new detection with assigned global ID."""
        return Detection(
            id=self.id,
            camera_id=self.camera_id,
            bbox=self.bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            track_id=self.track_id,
            global_id=global_id,
            feature_vector=self.feature_vector
        )

@dataclass
class FrameMetadata:
    """Frame metadata for detection context."""
    
    frame_index: int
    timestamp: datetime
    camera_id: CameraID
    width: int
    height: int
    
    # Optional metadata
    fps: Optional[float] = None
    format: Optional[str] = None
    encoding: Optional[str] = None
    
    def __post_init__(self):
        """Validate frame metadata."""
        if self.frame_index < 0:
            raise ValueError("Frame index must be non-negative")
        
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame dimensions must be positive")
    
    @property
    def resolution(self) -> tuple[int, int]:
        """Get frame resolution as (width, height)."""
        return (self.width, self.height)
    
    @property
    def aspect_ratio(self) -> float:
        """Get frame aspect ratio."""
        return self.width / self.height

@dataclass
class DetectionBatch:
    """Batch of detections from multiple cameras."""
    
    detections: List[Detection]
    frame_metadata: FrameMetadata
    processing_time: float
    
    def __post_init__(self):
        """Validate detection batch."""
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")
    
    @property
    def detection_count(self) -> int:
        """Get number of detections."""
        return len(self.detections)
    
    @property
    def person_detections(self) -> List[Detection]:
        """Get only person detections."""
        return [d for d in self.detections if d.is_person]
    
    @property
    def person_count(self) -> int:
        """Get number of person detections."""
        return len(self.person_detections)
    
    def get_detections_by_camera(self, camera_id: CameraID) -> List[Detection]:
        """Get detections for specific camera."""
        return [d for d in self.detections if d.camera_id == camera_id]
    
    def get_high_confidence_detections(self, threshold: float = 0.5) -> List[Detection]:
        """Get detections above confidence threshold."""
        return [d for d in self.detections if d.confidence >= threshold]