"""
Velocity value object for tracking movement speed and direction.

Provides type-safe velocity representation with magnitude and direction.
"""
from dataclasses import dataclass
from typing import Tuple
import math

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class Velocity(BaseValueObject):
    """
    Velocity value object representing movement speed and direction.
    
    Stores velocity as (vx, vy) components in pixels per frame or meters per second.
    """
    
    vx: float  # X-component of velocity
    vy: float  # Y-component of velocity
    units: str = "pixels/frame"  # Units of measurement
    
    def _validate(self) -> None:
        """Validate velocity components."""
        if not isinstance(self.vx, (int, float)) or not isinstance(self.vy, (int, float)):
            raise ValueError("Velocity components must be numeric")
        
        valid_units = {"pixels/frame", "pixels/second", "m/s", "km/h"}
        if self.units not in valid_units:
            raise ValueError(f"Invalid units. Must be one of: {valid_units}")
    
    @classmethod
    def zero(cls, units: str = "pixels/frame") -> 'Velocity':
        """Create zero velocity."""
        return cls(vx=0.0, vy=0.0, units=units)
    
    @classmethod
    def from_displacement(
        cls, 
        displacement: Tuple[float, float], 
        time_delta: float,
        units: str = "pixels/frame"
    ) -> 'Velocity':
        """
        Create velocity from displacement and time delta.
        
        Args:
            displacement: (dx, dy) displacement tuple
            time_delta: Time difference
            units: Velocity units
            
        Returns:
            Velocity instance
        """
        if time_delta <= 0:
            raise ValueError("Time delta must be positive")
        
        dx, dy = displacement
        return cls(
            vx=dx / time_delta,
            vy=dy / time_delta,
            units=units
        )
    
    @classmethod
    def from_points(
        cls,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        time_delta: float,
        units: str = "pixels/frame"
    ) -> 'Velocity':
        """
        Create velocity from two points and time delta.
        
        Args:
            point1: Starting point (x1, y1)
            point2: Ending point (x2, y2)
            time_delta: Time difference
            units: Velocity units
            
        Returns:
            Velocity instance
        """
        x1, y1 = point1
        x2, y2 = point2
        displacement = (x2 - x1, y2 - y1)
        return cls.from_displacement(displacement, time_delta, units)
    
    @property
    def magnitude(self) -> float:
        """Get velocity magnitude (speed)."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2)
    
    @property
    def speed(self) -> float:
        """Get speed (alias for magnitude)."""
        return self.magnitude
    
    @property
    def direction_radians(self) -> float:
        """Get direction in radians."""
        return math.atan2(self.vy, self.vx)
    
    @property
    def direction_degrees(self) -> float:
        """Get direction in degrees."""
        return math.degrees(self.direction_radians)
    
    @property
    def is_stationary(self, threshold: float = 0.1) -> bool:
        """Check if velocity indicates stationary object."""
        return self.magnitude <= threshold
    
    @property
    def is_moving(self, threshold: float = 0.1) -> bool:
        """Check if velocity indicates moving object."""
        return self.magnitude > threshold
    
    @property
    def is_moving_right(self) -> bool:
        """Check if moving to the right (positive X)."""
        return self.vx > 0
    
    @property
    def is_moving_left(self) -> bool:
        """Check if moving to the left (negative X)."""
        return self.vx < 0
    
    @property
    def is_moving_up(self) -> bool:
        """Check if moving up (negative Y in image coordinates)."""
        return self.vy < 0
    
    @property
    def is_moving_down(self) -> bool:
        """Check if moving down (positive Y in image coordinates)."""
        return self.vy > 0
    
    def add(self, other: 'Velocity') -> 'Velocity':
        """
        Add two velocities (vector addition).
        
        Args:
            other: Other velocity to add
            
        Returns:
            Sum of velocities
        """
        if self.units != other.units:
            raise ValueError(f"Cannot add velocities with different units: {self.units} vs {other.units}")
        
        return Velocity(
            vx=self.vx + other.vx,
            vy=self.vy + other.vy,
            units=self.units
        )
    
    def subtract(self, other: 'Velocity') -> 'Velocity':
        """
        Subtract two velocities.
        
        Args:
            other: Velocity to subtract
            
        Returns:
            Difference of velocities
        """
        if self.units != other.units:
            raise ValueError(f"Cannot subtract velocities with different units: {self.units} vs {other.units}")
        
        return Velocity(
            vx=self.vx - other.vx,
            vy=self.vy - other.vy,
            units=self.units
        )
    
    def scale(self, factor: float) -> 'Velocity':
        """
        Scale velocity by a factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            Scaled velocity
        """
        return Velocity(
            vx=self.vx * factor,
            vy=self.vy * factor,
            units=self.units
        )
    
    def normalize(self) -> 'Velocity':
        """
        Get normalized velocity (unit vector).
        
        Returns:
            Normalized velocity with magnitude 1
        """
        magnitude = self.magnitude
        if magnitude == 0:
            return self  # Return zero vector as-is
        
        return Velocity(
            vx=self.vx / magnitude,
            vy=self.vy / magnitude,
            units="normalized"
        )
    
    def dot_product(self, other: 'Velocity') -> float:
        """
        Calculate dot product with another velocity.
        
        Args:
            other: Other velocity
            
        Returns:
            Dot product result
        """
        return self.vx * other.vx + self.vy * other.vy
    
    def angle_between(self, other: 'Velocity') -> float:
        """
        Calculate angle between two velocities in radians.
        
        Args:
            other: Other velocity
            
        Returns:
            Angle in radians
        """
        dot = self.dot_product(other)
        magnitudes = self.magnitude * other.magnitude
        
        if magnitudes == 0:
            return 0.0
        
        cos_angle = dot / magnitudes
        # Clamp to [-1, 1] to handle floating point errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        return math.acos(cos_angle)
    
    def is_similar_direction(self, other: 'Velocity', angle_threshold: float = math.pi / 4) -> bool:
        """
        Check if two velocities have similar direction.
        
        Args:
            other: Other velocity
            angle_threshold: Maximum angle difference in radians (default: 45 degrees)
            
        Returns:
            True if directions are similar
        """
        if self.is_stationary() or other.is_stationary():
            return False  # Stationary objects have no meaningful direction
        
        return self.angle_between(other) <= angle_threshold
    
    def predict_position(
        self, 
        current_position: Tuple[float, float], 
        time_steps: float
    ) -> Tuple[float, float]:
        """
        Predict future position based on current velocity.
        
        Args:
            current_position: Current (x, y) position
            time_steps: Number of time steps to predict
            
        Returns:
            Predicted (x, y) position
        """
        x, y = current_position
        return (
            x + self.vx * time_steps,
            y + self.vy * time_steps
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (vx, vy) tuple."""
        return (self.vx, self.vy)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Velocity({self.vx:.2f}, {self.vy:.2f} {self.units})"