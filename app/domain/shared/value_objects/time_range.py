"""
Time range value object for temporal operations.

Immutable value object representing a time range with start and end times.
"""
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class TimeRange(BaseValueObject):
    """
    Time range value object.
    
    Represents an immutable time range with start and end times.
    """
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self):
        """Validate time range after initialization."""
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.duration_seconds / 60.0
    
    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return self.duration_seconds / 3600.0
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this time range."""
        return self.start_time <= timestamp <= self.end_time
    
    def overlaps(self, other: 'TimeRange') -> bool:
        """Check if this time range overlaps with another."""
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds
        }