"""
Confidence value object for detection confidence scores.

Provides type-safe confidence handling with validation and comparison.
"""
from dataclasses import dataclass
from typing import Union

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class Confidence(BaseValueObject):
    """
    Detection confidence value object.
    
    Represents detection confidence scores with validation and comparison methods.
    """
    
    value: float
    
    def _validate(self) -> None:
        """Validate confidence value."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.value}")
    
    @classmethod
    def from_float(cls, confidence: float) -> 'Confidence':
        """
        Create Confidence from float value.
        
        Args:
            confidence: Confidence score between 0.0 and 1.0
            
        Returns:
            Confidence instance
        """
        return cls(value=float(confidence))
    
    @classmethod
    def from_percentage(cls, percentage: float) -> 'Confidence':
        """
        Create Confidence from percentage (0-100).
        
        Args:
            percentage: Confidence as percentage (0-100)
            
        Returns:
            Confidence instance
        """
        return cls(value=percentage / 100.0)
    
    @classmethod
    def zero(cls) -> 'Confidence':
        """Create zero confidence."""
        return cls(value=0.0)
    
    @classmethod
    def max(cls) -> 'Confidence':
        """Create maximum confidence."""
        return cls(value=1.0)
    
    @property
    def percentage(self) -> float:
        """Get confidence as percentage."""
        return self.value * 100.0
    
    @property
    def is_high(self, threshold: float = 0.7) -> bool:
        """Check if confidence is high."""
        return self.value >= threshold
    
    @property
    def is_low(self, threshold: float = 0.3) -> bool:
        """Check if confidence is low."""
        return self.value <= threshold
    
    @property
    def is_medium(self, low_threshold: float = 0.3, high_threshold: float = 0.7) -> bool:
        """Check if confidence is medium."""
        return low_threshold < self.value < high_threshold
    
    def above_threshold(self, threshold: Union[float, 'Confidence']) -> bool:
        """
        Check if confidence is above threshold.
        
        Args:
            threshold: Threshold value or Confidence instance
            
        Returns:
            True if above threshold
        """
        threshold_value = threshold.value if isinstance(threshold, Confidence) else threshold
        return self.value > threshold_value
    
    def meets_threshold(self, threshold: Union[float, 'Confidence']) -> bool:
        """
        Check if confidence meets or exceeds threshold.
        
        Args:
            threshold: Threshold value or Confidence instance
            
        Returns:
            True if meets threshold
        """
        threshold_value = threshold.value if isinstance(threshold, Confidence) else threshold
        return self.value >= threshold_value
    
    def combine_with(self, other: 'Confidence', method: str = 'multiply') -> 'Confidence':
        """
        Combine with another confidence score.
        
        Args:
            other: Other confidence score
            method: Combination method ('multiply', 'average', 'min', 'max')
            
        Returns:
            Combined confidence
        """
        if method == 'multiply':
            return Confidence(value=self.value * other.value)
        elif method == 'average':
            return Confidence(value=(self.value + other.value) / 2.0)
        elif method == 'min':
            return Confidence(value=min(self.value, other.value))
        elif method == 'max':
            return Confidence(value=max(self.value, other.value))
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def __float__(self) -> float:
        """Convert to float."""
        return self.value
    
    def __lt__(self, other: Union['Confidence', float]) -> bool:
        """Less than comparison."""
        other_value = other.value if isinstance(other, Confidence) else other
        return self.value < other_value
    
    def __le__(self, other: Union['Confidence', float]) -> bool:
        """Less than or equal comparison."""
        other_value = other.value if isinstance(other, Confidence) else other
        return self.value <= other_value
    
    def __gt__(self, other: Union['Confidence', float]) -> bool:
        """Greater than comparison."""
        other_value = other.value if isinstance(other, Confidence) else other
        return self.value > other_value
    
    def __ge__(self, other: Union['Confidence', float]) -> bool:
        """Greater than or equal comparison."""
        other_value = other.value if isinstance(other, Confidence) else other
        return self.value >= other_value
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.percentage:.1f}%"