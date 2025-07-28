"""
Coordinate entity for spatial mapping.

Contains coordinate system objects and spatial transformations.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import math

class CoordinateSystem(Enum):
    """Coordinate system types."""
    IMAGE = "image"          # Pixel coordinates in image
    NORMALIZED = "normalized" # Normalized [0,1] coordinates
    MAP = "map"              # World/map coordinates
    CAMERA = "camera"        # Camera coordinate system

@dataclass(frozen=True)
class Coordinate:
    """Coordinate value object."""
    
    x: float
    y: float
    coordinate_system: CoordinateSystem
    timestamp: datetime
    confidence: float = 1.0
    
    # Optional metadata
    z: Optional[float] = None
    camera_id: Optional[str] = None
    frame_index: Optional[int] = None
    
    def __post_init__(self):
        """Validate coordinate parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.coordinate_system == CoordinateSystem.NORMALIZED:
            if not (0 <= self.x <= 1 and 0 <= self.y <= 1):
                raise ValueError("Normalized coordinates must be between 0 and 1")
        
        if self.coordinate_system == CoordinateSystem.IMAGE:
            if self.x < 0 or self.y < 0:
                raise ValueError("Image coordinates must be non-negative")
    
    @property
    def is_2d(self) -> bool:
        """Check if coordinate is 2D."""
        return self.z is None
    
    @property
    def is_3d(self) -> bool:
        """Check if coordinate is 3D."""
        return self.z is not None
    
    def distance_to(self, other: 'Coordinate') -> float:
        """Calculate Euclidean distance to another coordinate."""
        if self.coordinate_system != other.coordinate_system:
            raise ValueError("Cannot calculate distance between different coordinate systems")
        
        dx = self.x - other.x
        dy = self.y - other.y
        
        if self.is_3d and other.is_3d:
            dz = (self.z or 0) - (other.z or 0)
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            return math.sqrt(dx*dx + dy*dy)
    
    def manhattan_distance_to(self, other: 'Coordinate') -> float:
        """Calculate Manhattan distance to another coordinate."""
        if self.coordinate_system != other.coordinate_system:
            raise ValueError("Cannot calculate distance between different coordinate systems")
        
        distance = abs(self.x - other.x) + abs(self.y - other.y)
        
        if self.is_3d and other.is_3d:
            distance += abs((self.z or 0) - (other.z or 0))
        
        return distance
    
    def translate(self, dx: float, dy: float, dz: Optional[float] = None) -> 'Coordinate':
        """Translate coordinate by given offsets."""
        new_z = None
        if self.is_3d:
            new_z = (self.z or 0) + (dz or 0)
        elif dz is not None:
            new_z = dz
        
        return Coordinate(
            x=self.x + dx,
            y=self.y + dy,
            z=new_z,
            coordinate_system=self.coordinate_system,
            timestamp=self.timestamp,
            confidence=self.confidence,
            camera_id=self.camera_id,
            frame_index=self.frame_index
        )
    
    def scale(self, scale_x: float, scale_y: float, scale_z: Optional[float] = None) -> 'Coordinate':
        """Scale coordinate by given factors."""
        new_z = None
        if self.is_3d:
            new_z = (self.z or 0) * (scale_z or 1.0)
        
        return Coordinate(
            x=self.x * scale_x,
            y=self.y * scale_y,
            z=new_z,
            coordinate_system=self.coordinate_system,
            timestamp=self.timestamp,
            confidence=self.confidence,
            camera_id=self.camera_id,
            frame_index=self.frame_index
        )
    
    def to_normalized(self, image_width: int, image_height: int) -> 'Coordinate':
        """Convert image coordinates to normalized coordinates."""
        if self.coordinate_system != CoordinateSystem.IMAGE:
            raise ValueError("Can only normalize image coordinates")
        
        return Coordinate(
            x=self.x / image_width,
            y=self.y / image_height,
            coordinate_system=CoordinateSystem.NORMALIZED,
            timestamp=self.timestamp,
            confidence=self.confidence,
            camera_id=self.camera_id,
            frame_index=self.frame_index
        )
    
    def to_image(self, image_width: int, image_height: int) -> 'Coordinate':
        """Convert normalized coordinates to image coordinates."""
        if self.coordinate_system != CoordinateSystem.NORMALIZED:
            raise ValueError("Can only convert normalized coordinates to image")
        
        return Coordinate(
            x=self.x * image_width,
            y=self.y * image_height,
            coordinate_system=CoordinateSystem.IMAGE,
            timestamp=self.timestamp,
            confidence=self.confidence,
            camera_id=self.camera_id,
            frame_index=self.frame_index
        )
    
    def to_tuple(self) -> Tuple[float, float] | Tuple[float, float, float]:
        """Convert to tuple representation."""
        if self.is_3d:
            return (self.x, self.y, self.z or 0)
        else:
            return (self.x, self.y)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "coordinate_system": self.coordinate_system.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "is_2d": self.is_2d,
            "is_3d": self.is_3d
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Coordinate':
        """Create coordinate from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            z=data.get("z"),
            coordinate_system=CoordinateSystem(data["coordinate_system"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            confidence=data.get("confidence", 1.0),
            camera_id=data.get("camera_id"),
            frame_index=data.get("frame_index")
        )
    
    @classmethod
    def from_tuple(
        cls,
        coords: Tuple[float, float] | Tuple[float, float, float],
        coordinate_system: CoordinateSystem,
        timestamp: datetime,
        **kwargs
    ) -> 'Coordinate':
        """Create coordinate from tuple."""
        if len(coords) == 2:
            x, y = coords
            z = None
        elif len(coords) == 3:
            x, y, z = coords
        else:
            raise ValueError("Coordinate tuple must have 2 or 3 elements")
        
        return cls(
            x=x,
            y=y,
            z=z,
            coordinate_system=coordinate_system,
            timestamp=timestamp,
            **kwargs
        )

@dataclass
class CoordinateTransformation:
    """Coordinate transformation between different systems."""
    
    source_system: CoordinateSystem
    target_system: CoordinateSystem
    transformation_matrix: List[List[float]]
    
    # Optional metadata
    camera_id: Optional[str] = None
    calibration_date: Optional[datetime] = None
    accuracy: Optional[float] = None
    
    def __post_init__(self):
        """Validate transformation matrix."""
        if not self.transformation_matrix:
            raise ValueError("Transformation matrix cannot be empty")
        
        # Check matrix dimensions
        rows = len(self.transformation_matrix)
        if rows < 2:
            raise ValueError("Transformation matrix must have at least 2 rows")
        
        cols = len(self.transformation_matrix[0]) if self.transformation_matrix else 0
        if cols < 2:
            raise ValueError("Transformation matrix must have at least 2 columns")
        
        # Check matrix consistency
        for i, row in enumerate(self.transformation_matrix):
            if len(row) != cols:
                raise ValueError(f"Inconsistent matrix row length at row {i}")
    
    def transform_coordinate(self, coord: Coordinate) -> Coordinate:
        """Transform coordinate using this transformation."""
        if coord.coordinate_system != self.source_system:
            raise ValueError(f"Coordinate system mismatch: expected {self.source_system}, got {coord.coordinate_system}")
        
        # Apply transformation matrix
        if coord.is_3d:
            input_vector = [coord.x, coord.y, coord.z or 0, 1]
        else:
            input_vector = [coord.x, coord.y, 1]
        
        # Matrix multiplication
        result = []
        for row in self.transformation_matrix:
            value = sum(row[i] * input_vector[i] for i in range(min(len(row), len(input_vector))))
            result.append(value)
        
        # Create transformed coordinate
        new_x = result[0] if len(result) > 0 else coord.x
        new_y = result[1] if len(result) > 1 else coord.y
        new_z = result[2] if len(result) > 2 and coord.is_3d else None
        
        return Coordinate(
            x=new_x,
            y=new_y,
            z=new_z,
            coordinate_system=self.target_system,
            timestamp=coord.timestamp,
            confidence=coord.confidence * (self.accuracy or 1.0),
            camera_id=coord.camera_id,
            frame_index=coord.frame_index
        )
    
    def transform_batch(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        """Transform multiple coordinates."""
        return [self.transform_coordinate(coord) for coord in coordinates]
    
    def is_valid(self) -> bool:
        """Check if transformation is valid."""
        try:
            # Check matrix properties
            if not self.transformation_matrix:
                return False
            
            # Check for reasonable values (not all zeros)
            has_non_zero = any(
                any(abs(val) > 1e-10 for val in row)
                for row in self.transformation_matrix
            )
            
            return has_non_zero
            
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_system": self.source_system.value,
            "target_system": self.target_system.value,
            "transformation_matrix": self.transformation_matrix,
            "camera_id": self.camera_id,
            "calibration_date": self.calibration_date.isoformat() if self.calibration_date else None,
            "accuracy": self.accuracy
        }

@dataclass
class BoundingRegion:
    """Bounding region in a coordinate system."""
    
    min_coord: Coordinate
    max_coord: Coordinate
    
    def __post_init__(self):
        """Validate bounding region."""
        if self.min_coord.coordinate_system != self.max_coord.coordinate_system:
            raise ValueError("Bounding region coordinates must be in same coordinate system")
        
        if self.min_coord.x >= self.max_coord.x or self.min_coord.y >= self.max_coord.y:
            raise ValueError("Invalid bounding region: min coordinates must be less than max coordinates")
    
    @property
    def width(self) -> float:
        """Get region width."""
        return self.max_coord.x - self.min_coord.x
    
    @property
    def height(self) -> float:
        """Get region height."""
        return self.max_coord.y - self.min_coord.y
    
    @property
    def area(self) -> float:
        """Get region area."""
        return self.width * self.height
    
    @property
    def center(self) -> Coordinate:
        """Get region center."""
        center_x = (self.min_coord.x + self.max_coord.x) / 2
        center_y = (self.min_coord.y + self.max_coord.y) / 2
        
        return Coordinate(
            x=center_x,
            y=center_y,
            coordinate_system=self.min_coord.coordinate_system,
            timestamp=self.min_coord.timestamp,
            confidence=min(self.min_coord.confidence, self.max_coord.confidence)
        )
    
    def contains(self, coord: Coordinate) -> bool:
        """Check if coordinate is within region."""
        if coord.coordinate_system != self.min_coord.coordinate_system:
            return False
        
        return (self.min_coord.x <= coord.x <= self.max_coord.x and
                self.min_coord.y <= coord.y <= self.max_coord.y)
    
    def intersects(self, other: 'BoundingRegion') -> bool:
        """Check if this region intersects with another."""
        if self.min_coord.coordinate_system != other.min_coord.coordinate_system:
            return False
        
        return not (self.max_coord.x < other.min_coord.x or
                   self.min_coord.x > other.max_coord.x or
                   self.max_coord.y < other.min_coord.y or
                   self.min_coord.y > other.max_coord.y)
    
    def expand(self, margin: float) -> 'BoundingRegion':
        """Expand region by given margin."""
        return BoundingRegion(
            min_coord=self.min_coord.translate(-margin, -margin),
            max_coord=self.max_coord.translate(margin, margin)
        )