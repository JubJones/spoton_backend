"""
Coordinate value objects for spatial mapping.

Provides type-safe coordinate representation for different coordinate systems.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import math

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class ImageCoordinates(BaseValueObject):
    """
    Image coordinates value object.
    
    Represents pixel coordinates in image space (2D).
    Origin (0,0) is typically at top-left corner.
    """
    
    x: float  # X coordinate in pixels
    y: float  # Y coordinate in pixels
    image_width: Optional[int] = None  # Image width for validation
    image_height: Optional[int] = None  # Image height for validation
    
    def _validate(self) -> None:
        """Validate image coordinates."""
        if self.image_width is not None and self.image_width <= 0:
            raise ValueError("Image width must be positive")
        if self.image_height is not None and self.image_height <= 0:
            raise ValueError("Image height must be positive")
        
        if self.image_width is not None:
            if not (0 <= self.x <= self.image_width):
                raise ValueError(f"X coordinate {self.x} outside image bounds [0, {self.image_width}]")
        
        if self.image_height is not None:
            if not (0 <= self.y <= self.image_height):
                raise ValueError(f"Y coordinate {self.y} outside image bounds [0, {self.image_height}]")
    
    @classmethod
    def create(cls, x: float, y: float, image_width: Optional[int] = None, image_height: Optional[int] = None) -> 'ImageCoordinates':
        """
        Create ImageCoordinates with validation.
        
        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            image_width: Optional image width for bounds checking
            image_height: Optional image height for bounds checking
            
        Returns:
            ImageCoordinates instance
        """
        return cls(x=x, y=y, image_width=image_width, image_height=image_height)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float], **kwargs) -> 'ImageCoordinates':
        """Create from tuple (x, y)."""
        return cls(x=coords[0], y=coords[1], **kwargs)
    
    @property
    def is_normalized(self) -> bool:
        """Check if coordinates appear to be normalized (0-1 range)."""
        return 0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0
    
    @property
    def is_within_bounds(self) -> bool:
        """Check if coordinates are within image bounds."""
        if self.image_width is None or self.image_height is None:
            return True  # Cannot validate without bounds
        
        return (0 <= self.x <= self.image_width and 
                0 <= self.y <= self.image_height)
    
    def normalize(self) -> 'ImageCoordinates':
        """
        Normalize coordinates to [0, 1] range.
        
        Returns:
            Normalized ImageCoordinates
            
        Raises:
            ValueError: If image dimensions not available
        """
        if self.image_width is None or self.image_height is None:
            raise ValueError("Cannot normalize without image dimensions")
        
        return ImageCoordinates(
            x=self.x / self.image_width,
            y=self.y / self.image_height,
            image_width=None,  # Normalized coords don't have pixel bounds
            image_height=None
        )
    
    def denormalize(self, image_width: int, image_height: int) -> 'ImageCoordinates':
        """
        Denormalize coordinates from [0, 1] range to pixel coordinates.
        
        Args:
            image_width: Target image width
            image_height: Target image height
            
        Returns:
            Denormalized ImageCoordinates
        """
        return ImageCoordinates(
            x=self.x * image_width,
            y=self.y * image_height,
            image_width=image_width,
            image_height=image_height
        )
    
    def distance_to(self, other: 'ImageCoordinates') -> float:
        """
        Calculate Euclidean distance to another point.
        
        Args:
            other: Other image coordinates
            
        Returns:
            Euclidean distance in pixels
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (x, y) tuple."""
        return (self.x, self.y)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Image({self.x:.1f}, {self.y:.1f})"


@dataclass(frozen=True)
class WorldCoordinates(BaseValueObject):
    """
    World coordinates value object.
    
    Represents real-world coordinates in physical space.
    Typically in meters from a reference origin.
    """
    
    x: float  # X coordinate in world units (typically meters)
    y: float  # Y coordinate in world units (typically meters)
    z: float = 0.0  # Z coordinate (height), default to ground level
    units: str = "meters"  # Units of measurement
    
    def _validate(self) -> None:
        """Validate world coordinates."""
        valid_units = {"meters", "feet", "centimeters", "millimeters"}
        if self.units not in valid_units:
            raise ValueError(f"Invalid units. Must be one of: {valid_units}")
    
    @classmethod
    def create(cls, x: float, y: float, z: float = 0.0, units: str = "meters") -> 'WorldCoordinates':
        """
        Create WorldCoordinates with validation.
        
        Args:
            x: X coordinate in world units
            y: Y coordinate in world units
            z: Z coordinate (height)
            units: Units of measurement
            
        Returns:
            WorldCoordinates instance
        """
        return cls(x=x, y=y, z=z, units=units)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float], z: float = 0.0, units: str = "meters") -> 'WorldCoordinates':
        """Create from tuple (x, y)."""
        return cls(x=coords[0], y=coords[1], z=z, units=units)
    
    @classmethod
    def origin(cls, units: str = "meters") -> 'WorldCoordinates':
        """Create origin coordinates (0, 0, 0)."""
        return cls(x=0.0, y=0.0, z=0.0, units=units)
    
    @property
    def is_2d(self) -> bool:
        """Check if coordinates are 2D (z = 0)."""
        return abs(self.z) < 1e-6
    
    @property
    def is_origin(self) -> bool:
        """Check if coordinates are at origin."""
        return abs(self.x) < 1e-6 and abs(self.y) < 1e-6 and abs(self.z) < 1e-6
    
    def distance_to(self, other: 'WorldCoordinates') -> float:
        """
        Calculate 3D Euclidean distance to another point.
        
        Args:
            other: Other world coordinates
            
        Returns:
            3D Euclidean distance
            
        Raises:
            ValueError: If units don't match
        """
        if self.units != other.units:
            raise ValueError(f"Cannot calculate distance between different units: {self.units} vs {other.units}")
        
        return math.sqrt(
            (self.x - other.x) ** 2 + 
            (self.y - other.y) ** 2 + 
            (self.z - other.z) ** 2
        )
    
    def distance_2d(self, other: 'WorldCoordinates') -> float:
        """
        Calculate 2D distance (ignoring Z coordinate).
        
        Args:
            other: Other world coordinates
            
        Returns:
            2D Euclidean distance
        """
        if self.units != other.units:
            raise ValueError(f"Cannot calculate distance between different units: {self.units} vs {other.units}")
        
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def convert_units(self, target_units: str) -> 'WorldCoordinates':
        """
        Convert to different units.
        
        Args:
            target_units: Target units
            
        Returns:
            WorldCoordinates in target units
        """
        if self.units == target_units:
            return self
        
        # Conversion factors to meters
        to_meters = {
            "meters": 1.0,
            "centimeters": 0.01,
            "millimeters": 0.001,
            "feet": 0.3048
        }
        
        # Conversion factors from meters
        from_meters = {
            "meters": 1.0,
            "centimeters": 100.0,
            "millimeters": 1000.0,
            "feet": 3.28084
        }
        
        if self.units not in to_meters or target_units not in from_meters:
            raise ValueError(f"Unsupported unit conversion: {self.units} to {target_units}")
        
        # Convert to meters first, then to target units
        meters_x = self.x * to_meters[self.units]
        meters_y = self.y * to_meters[self.units]
        meters_z = self.z * to_meters[self.units]
        
        target_x = meters_x * from_meters[target_units]
        target_y = meters_y * from_meters[target_units]
        target_z = meters_z * from_meters[target_units]
        
        return WorldCoordinates(x=target_x, y=target_y, z=target_z, units=target_units)
    
    def translate(self, dx: float, dy: float, dz: float = 0.0) -> 'WorldCoordinates':
        """
        Translate coordinates by given offsets.
        
        Args:
            dx: X offset
            dy: Y offset  
            dz: Z offset
            
        Returns:
            Translated WorldCoordinates
        """
        return WorldCoordinates(
            x=self.x + dx,
            y=self.y + dy,
            z=self.z + dz,
            units=self.units
        )
    
    def to_2d(self) -> 'WorldCoordinates':
        """Convert to 2D coordinates (z = 0)."""
        return WorldCoordinates(x=self.x, y=self.y, z=0.0, units=self.units)
    
    def to_tuple_2d(self) -> Tuple[float, float]:
        """Convert to 2D tuple (x, y)."""
        return (self.x, self.y)
    
    def to_tuple_3d(self) -> Tuple[float, float, float]:
        """Convert to 3D tuple (x, y, z)."""
        return (self.x, self.y, self.z)
    
    def __str__(self) -> str:
        """String representation."""
        if self.is_2d:
            return f"World({self.x:.2f}, {self.y:.2f} {self.units})"
        else:
            return f"World({self.x:.2f}, {self.y:.2f}, {self.z:.2f} {self.units})"


@dataclass(frozen=True)
class CoordinateTransformation:
    """
    Coordinate transformation value object.
    
    Represents a transformation from one coordinate system to another.
    """
    
    from_coords: ImageCoordinates
    to_coords: WorldCoordinates
    accuracy: float = 1.0  # Transformation accuracy/confidence (0-1)
    transformation_method: str = "homography"  # Method used for transformation
    
    def _validate(self) -> None:
        """Validate coordinate transformation."""
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0.0 and 1.0")
        
        valid_methods = {"homography", "affine", "perspective", "interpolation"}
        if self.transformation_method not in valid_methods:
            raise ValueError(f"Invalid transformation method. Must be one of: {valid_methods}")
    
    @classmethod
    def create(
        cls,
        from_coords: ImageCoordinates,
        to_coords: WorldCoordinates,
        accuracy: float = 1.0,
        method: str = "homography"
    ) -> 'CoordinateTransformation':
        """
        Create coordinate transformation.
        
        Args:
            from_coords: Source image coordinates
            to_coords: Target world coordinates
            accuracy: Transformation accuracy
            method: Transformation method
            
        Returns:
            CoordinateTransformation instance
        """
        return cls(
            from_coords=from_coords,
            to_coords=to_coords,
            accuracy=accuracy,
            transformation_method=method
        )
    
    @property
    def is_high_accuracy(self, threshold: float = 0.8) -> bool:
        """Check if transformation has high accuracy."""
        return self.accuracy >= threshold
    
    @property
    def error_estimate(self) -> float:
        """Get estimated error (1 - accuracy)."""
        return 1.0 - self.accuracy
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'from_coords': self.from_coords.to_dict(),
            'to_coords': self.to_coords.to_dict(),
            'accuracy': self.accuracy,
            'transformation_method': self.transformation_method,
            'error_estimate': self.error_estimate
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"Transform({self.from_coords} -> {self.to_coords}, "
                f"acc={self.accuracy:.2f}, method={self.transformation_method})")