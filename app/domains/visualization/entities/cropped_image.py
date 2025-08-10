"""
Cropped Image Entity

Represents a cropped person image with metadata.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import base64


@dataclass
class CroppedImage:
    """Represents a cropped person image with metadata."""
    
    # Required fields
    global_person_id: str
    camera_id: str
    frame_index: int
    timestamp: datetime
    image_data: bytes
    original_bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    width: int
    height: int
    confidence: float
    
    # Optional fields with defaults
    image_format: str = "jpeg"  # jpeg, png
    image_quality: int = 85
    detection_quality: str = "high"  # high, medium, low
    cache_key: Optional[str] = None
    cached_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.cache_key is None:
            self.cache_key = self._generate_cache_key()
    
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key for this cropped image."""
        return f"cropped_{self.global_person_id}_{self.camera_id}_{self.frame_index}_{int(self.timestamp.timestamp())}"
    
    def to_base64(self) -> str:
        """Convert image data to base64 string."""
        return base64.b64encode(self.image_data).decode('utf-8')
    
    def to_data_uri(self) -> str:
        """Convert image to data URI format."""
        mime_type = f"image/{self.image_format}"
        b64_data = self.to_base64()
        return f"data:{mime_type};base64,{b64_data}"
    
    def get_size_kb(self) -> float:
        """Get image size in kilobytes."""
        return len(self.image_data) / 1024
    
    def is_expired(self) -> bool:
        """Check if the cached image is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cropped image to dictionary."""
        return {
            "global_person_id": self.global_person_id,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat(),
            "image_format": self.image_format,
            "image_quality": self.image_quality,
            "original_bbox": list(self.original_bbox),
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "detection_quality": self.detection_quality,
            "cache_key": self.cache_key,
            "cached_at": self.cached_at.isoformat() if self.cached_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "size_kb": self.get_size_kb()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], image_data: bytes) -> "CroppedImage":
        """Create cropped image from dictionary and image data."""
        # Convert string timestamps back to datetime
        timestamp = datetime.fromisoformat(data["timestamp"])
        cached_at = datetime.fromisoformat(data["cached_at"]) if data.get("cached_at") else None
        expires_at = datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        created_at = datetime.fromisoformat(data["created_at"])
        
        return cls(
            global_person_id=data["global_person_id"],
            camera_id=data["camera_id"],
            frame_index=data["frame_index"],
            timestamp=timestamp,
            image_data=image_data,
            image_format=data.get("image_format", "jpeg"),
            image_quality=data.get("image_quality", 85),
            original_bbox=tuple(data["original_bbox"]),
            width=data["width"],
            height=data["height"],
            confidence=data["confidence"],
            detection_quality=data.get("detection_quality", "high"),
            cache_key=data.get("cache_key"),
            cached_at=cached_at,
            expires_at=expires_at,
            created_at=created_at
        )
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get a summary of image metadata for logging."""
        return {
            "person_id": self.global_person_id,
            "camera": self.camera_id,
            "frame": self.frame_index,
            "size_kb": round(self.get_size_kb(), 2),
            "dimensions": f"{self.width}x{self.height}",
            "confidence": round(self.confidence, 3),
            "quality": self.detection_quality,
            "format": self.image_format
        }