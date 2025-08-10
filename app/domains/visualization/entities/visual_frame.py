"""
Visual Frame Entity

Represents a processed frame with overlays and visual enhancements.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import base64

from app.domains.detection.entities.detection import Detection
from .overlay_config import OverlayConfig
from .cropped_image import CroppedImage


@dataclass
class VisualFrame:
    """Represents a processed camera frame with overlays and enhancements."""
    
    # Required fields
    camera_id: str
    frame_index: int
    timestamp: datetime
    original_frame_data: bytes
    original_width: int
    original_height: int
    processed_frame_data: bytes
    detections: List[Detection]
    
    # Optional fields with defaults
    processed_format: str = "jpeg"
    processed_quality: int = 95
    focused_person_id: Optional[str] = None
    cropped_persons: Optional[Dict[str, CroppedImage]] = None
    overlay_config: Optional[OverlayConfig] = None
    processing_time_ms: float = 0.0
    total_persons: int = 0
    overlay_elements_count: int = 0
    cache_key: Optional[str] = None
    cached_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.cropped_persons is None:
            self.cropped_persons = {}
        if self.overlay_config is None:
            self.overlay_config = OverlayConfig()
        if self.cache_key is None:
            self.cache_key = self._generate_cache_key()
        
        # Update counts
        self.total_persons = len(self.detections)
        self.overlay_elements_count = self._count_overlay_elements()
    
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key for this visual frame."""
        return f"visual_frame_{self.camera_id}_{self.frame_index}_{int(self.timestamp.timestamp())}"
    
    def _count_overlay_elements(self) -> int:
        """Count the number of overlay elements (bboxes, text, etc.)."""
        count = 0
        for detection in self.detections:
            count += 1  # bounding box
            if self.overlay_config.show_person_id:
                count += 1  # person ID text
            if self.overlay_config.show_confidence:
                count += 1  # confidence text
            if self.overlay_config.show_tracking_duration:
                count += 1  # duration text
        return count
    
    def to_base64(self, use_processed: bool = True) -> str:
        """Convert frame data to base64 string."""
        frame_data = self.processed_frame_data if use_processed else self.original_frame_data
        return base64.b64encode(frame_data).decode('utf-8')
    
    def to_data_uri(self, use_processed: bool = True) -> str:
        """Convert frame to data URI format."""
        format_type = self.processed_format if use_processed else "jpeg"
        mime_type = f"image/{format_type}"
        b64_data = self.to_base64(use_processed)
        return f"data:{mime_type};base64,{b64_data}"
    
    def get_frame_size_kb(self, use_processed: bool = True) -> float:
        """Get frame size in kilobytes."""
        frame_data = self.processed_frame_data if use_processed else self.original_frame_data
        return len(frame_data) / 1024
    
    def get_cropped_persons_dict(self) -> Dict[str, str]:
        """Get cropped person images as data URIs."""
        return {
            person_id: cropped_img.to_data_uri()
            for person_id, cropped_img in self.cropped_persons.items()
        }
    
    def get_detection_by_person_id(self, global_person_id: str) -> Optional[Detection]:
        """Get detection for a specific person ID."""
        for detection in self.detections:
            if hasattr(detection, 'global_person_id') and detection.global_person_id == global_person_id:
                return detection
        return None
    
    def is_person_focused(self, global_person_id: str) -> bool:
        """Check if a person is currently focused."""
        return self.focused_person_id == global_person_id
    
    def get_focused_detection(self) -> Optional[Detection]:
        """Get the detection for the focused person."""
        if self.focused_person_id:
            return self.get_detection_by_person_id(self.focused_person_id)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert visual frame to dictionary."""
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat(),
            "original_width": self.original_width,
            "original_height": self.original_height,
            "processed_format": self.processed_format,
            "processed_quality": self.processed_quality,
            "detections": [detection.to_dict() if hasattr(detection, 'to_dict') else str(detection) for detection in self.detections],
            "focused_person_id": self.focused_person_id,
            "cropped_persons": {
                person_id: cropped_img.to_dict() 
                for person_id, cropped_img in self.cropped_persons.items()
            },
            "overlay_config": self.overlay_config.to_dict(),
            "processing_time_ms": self.processing_time_ms,
            "total_persons": self.total_persons,
            "overlay_elements_count": self.overlay_elements_count,
            "cache_key": self.cache_key,
            "cached_at": self.cached_at.isoformat() if self.cached_at else None,
            "created_at": self.created_at.isoformat(),
            "frame_size_kb": self.get_frame_size_kb(),
            "original_frame_size_kb": self.get_frame_size_kb(use_processed=False)
        }
    
    def get_tracking_update_payload(self) -> Dict[str, Any]:
        """Get payload for tracking update WebSocket message."""
        return {
            "image_source": f"{self.frame_index:06d}.jpg",
            "frame_image_base64": self.to_base64(),
            "cropped_persons": self.get_cropped_persons_dict(),
            "tracks": [
                {
                    "track_id": getattr(detection, 'track_id', 0),
                    "global_id": getattr(detection, 'global_person_id', f"person_{i}"),
                    "bbox_xyxy": detection.bbox,
                    "confidence": detection.confidence,
                    "map_coords": getattr(detection, 'map_coordinates', [0.0, 0.0]),
                    "is_focused": self.is_person_focused(getattr(detection, 'global_person_id', f"person_{i}")),
                    "detection_time": detection.timestamp.isoformat(),
                    "tracking_duration": 0.0  # This would be calculated by tracking service
                }
                for i, detection in enumerate(self.detections)
            ]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this frame processing."""
        return {
            "processing_time_ms": self.processing_time_ms,
            "total_persons": self.total_persons,
            "overlay_elements": self.overlay_elements_count,
            "frame_size_kb": self.get_frame_size_kb(),
            "compression_ratio": (
                self.get_frame_size_kb(use_processed=False) / self.get_frame_size_kb()
                if self.get_frame_size_kb() > 0 else 1.0
            ),
            "cropped_images_count": len(self.cropped_persons),
            "cropped_images_total_kb": sum(
                img.get_size_kb() for img in self.cropped_persons.values()
            )
        }