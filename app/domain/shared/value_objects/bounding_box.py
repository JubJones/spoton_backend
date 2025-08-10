"""
Unified BoundingBox value object for the entire application.

This resolves the critical model duplication issue between legacy and domain models.
Supports both coordinate formats with proper conversion methods.
"""
from dataclasses import dataclass
from typing import Tuple, Union, List
import numpy as np

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class BoundingBox(BaseValueObject):
    """
    Unified bounding box value object.
    
    Internally stores coordinates in (x, y, width, height) format
    but provides conversion methods for (x1, y1, x2, y2) format
    to maintain compatibility with legacy code.
    """
    
    x: float
    y: float
    width: float
    height: float
    normalized: bool = False
    
    def _validate(self) -> None:
        """Validate bounding box parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding box width and height must be positive")
        
        if self.normalized:
            if not (0 <= self.x <= 1 and 0 <= self.y <= 1):
                raise ValueError("Normalized coordinates must be between 0 and 1")
            if not (0 <= self.width <= 1 and 0 <= self.height <= 1):
                raise ValueError("Normalized dimensions must be between 0 and 1")
    
    @classmethod
    def from_coordinates(cls, x1: float, y1: float, x2: float, y2: float, normalized: bool = False) -> 'BoundingBox':
        """
        Create BoundingBox from corner coordinates (legacy format).
        
        Args:
            x1: Left edge
            y1: Top edge  
            x2: Right edge
            y2: Bottom edge
            normalized: Whether coordinates are normalized
            
        Returns:
            BoundingBox instance
        """
        return cls(
            x=x1,
            y=y1, 
            width=x2 - x1,
            height=y2 - y1,
            normalized=normalized
        )
    
    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float, normalized: bool = False) -> 'BoundingBox':
        """
        Create BoundingBox from (x, y, width, height) format.
        
        Args:
            x: Left edge
            y: Top edge
            w: Width
            h: Height
            normalized: Whether coordinates are normalized
            
        Returns:
            BoundingBox instance
        """
        return cls(x=x, y=y, width=w, height=h, normalized=normalized)
    
    @classmethod
    def from_center(cls, cx: float, cy: float, w: float, h: float, normalized: bool = False) -> 'BoundingBox':
        """
        Create BoundingBox from center coordinates.
        
        Args:
            cx: Center X
            cy: Center Y
            w: Width
            h: Height
            normalized: Whether coordinates are normalized
            
        Returns:
            BoundingBox instance
        """
        return cls(
            x=cx - w / 2,
            y=cy - h / 2,
            width=w,
            height=h,
            normalized=normalized
        )
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, normalized: bool = False) -> 'BoundingBox':
        """
        Create BoundingBox from numpy array.
        
        Args:
            array: Array in [x1, y1, x2, y2] or [x, y, w, h] format
            normalized: Whether coordinates are normalized
            
        Returns:
            BoundingBox instance
        """
        if len(array) != 4:
            raise ValueError("Array must have 4 elements")
        
        # Assume it's in [x1, y1, x2, y2] format (legacy compatibility)
        x1, y1, x2, y2 = array
        return cls.from_coordinates(x1, y1, x2, y2, normalized)
    
    # Core properties
    @property
    def x1(self) -> float:
        """Left edge coordinate (legacy compatibility)."""
        return self.x
    
    @property
    def y1(self) -> float:
        """Top edge coordinate (legacy compatibility)."""
        return self.y
    
    @property
    def x2(self) -> float:
        """Right edge coordinate."""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge coordinate."""
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
    
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height)."""
        return self.width / self.height
    
    # Format conversion methods
    def to_coordinates(self) -> Tuple[float, float, float, float]:
        """Convert to corner coordinates (x1, y1, x2, y2) - legacy format."""
        return (self.x, self.y, self.x2, self.y2)
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return self.to_coordinates()
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        return (self.x, self.y, self.width, self.height)
    
    def to_center_format(self) -> Tuple[float, float, float, float]:
        """Convert to (center_x, center_y, width, height) format."""
        return (self.center_x, self.center_y, self.width, self.height)
    
    def to_list(self) -> List[float]:
        """Convert to list in [x1, y1, x2, y2] format (legacy compatibility)."""
        return [self.x, self.y, self.x2, self.y2]
    
    def to_numpy(self, format: str = "xyxy") -> np.ndarray:
        """
        Convert to numpy array.
        
        Args:
            format: Either "xyxy" for [x1, y1, x2, y2] or "xywh" for [x, y, w, h]
            
        Returns:
            Numpy array with bounding box coordinates
        """
        if format == "xyxy":
            return np.array([self.x, self.y, self.x2, self.y2], dtype=np.float32)
        elif format == "xywh":
            return np.array([self.x, self.y, self.width, self.height], dtype=np.float32)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xyxy' or 'xywh'")
    
    # Geometric operations
    def scale(self, scale_x: float, scale_y: float) -> 'BoundingBox':
        """
        Scale bounding box by given factors.
        
        Args:
            scale_x: X scaling factor
            scale_y: Y scaling factor
            
        Returns:
            Scaled BoundingBox
        """
        return BoundingBox(
            x=self.x * scale_x,
            y=self.y * scale_y,
            width=self.width * scale_x,
            height=self.height * scale_y,
            normalized=False  # Scaling denormalizes
        )
    
    def normalize(self, image_width: float, image_height: float) -> 'BoundingBox':
        """
        Normalize bounding box to [0, 1] range.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Normalized BoundingBox
        """
        if self.normalized:
            return self
        
        return BoundingBox(
            x=self.x / image_width,
            y=self.y / image_height,
            width=self.width / image_width,
            height=self.height / image_height,
            normalized=True
        )
    
    def denormalize(self, image_width: float, image_height: float) -> 'BoundingBox':
        """
        Denormalize bounding box to pixel coordinates.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Denormalized BoundingBox
        """
        if not self.normalized:
            return self
        
        return BoundingBox(
            x=self.x * image_width,
            y=self.y * image_height,
            width=self.width * image_width,
            height=self.height * image_height,
            normalized=False
        )
    
    def crop(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        Crop this bounding box to fit within another bounding box.
        
        Args:
            other: Bounding box to crop within
            
        Returns:
            Cropped BoundingBox
        """
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        # Ensure valid crop
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding boxes do not overlap")
        
        return BoundingBox.from_coordinates(x1, y1, x2, y2, self.normalized)
    
    def expand(self, margin: Union[float, Tuple[float, float]]) -> 'BoundingBox':
        """
        Expand bounding box by given margin.
        
        Args:
            margin: Margin to expand by. If float, same margin for x and y.
                   If tuple, (margin_x, margin_y)
                   
        Returns:
            Expanded BoundingBox
        """
        if isinstance(margin, (int, float)):
            margin_x = margin_y = margin
        else:
            margin_x, margin_y = margin
        
        return BoundingBox(
            x=self.x - margin_x,
            y=self.y - margin_y,
            width=self.width + 2 * margin_x,
            height=self.height + 2 * margin_y,
            normalized=self.normalized
        )
    
    def iou(self, other: 'BoundingBox') -> float:
        """
        Calculate Intersection over Union (IoU) with another bounding box.
        
        Args:
            other: Other bounding box
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        # No intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def overlaps(self, other: 'BoundingBox', threshold: float = 0.0) -> bool:
        """
        Check if this bounding box overlaps with another.
        
        Args:
            other: Other bounding box
            threshold: Minimum IoU threshold for overlap
            
        Returns:
            True if bounding boxes overlap above threshold
        """
        return self.iou(other) > threshold
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside the bounding box.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside
        """
        return self.x <= x <= self.x2 and self.y <= y <= self.y2
    
    def contains(self, other: 'BoundingBox') -> bool:
        """
        Check if this bounding box completely contains another.
        
        Args:
            other: Other bounding box
            
        Returns:
            True if other is completely inside this one
        """
        return (
            self.x <= other.x and
            self.y <= other.y and
            self.x2 >= other.x2 and
            self.y2 >= other.y2
        )