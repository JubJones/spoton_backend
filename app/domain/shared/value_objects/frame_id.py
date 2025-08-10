"""
Frame ID value object for identifying video frames.

Provides type-safe frame identification with validation.
"""
from dataclasses import dataclass

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True) 
class FrameID(BaseValueObject):
    """
    Frame identifier value object.
    
    Identifies specific frames within video sequences with validation.
    """
    
    frame_index: int
    sequence_id: str
    
    def _validate(self) -> None:
        """Validate frame ID."""
        if self.frame_index < 0:
            raise ValueError("Frame index must be non-negative")
        if not self.sequence_id.strip():
            raise ValueError("Sequence ID cannot be empty or whitespace")
    
    @classmethod
    def create(cls, frame_index: int, sequence_id: str) -> 'FrameID':
        """
        Create FrameID with validation.
        
        Args:
            frame_index: Zero-based frame index
            sequence_id: Video sequence identifier
            
        Returns:
            FrameID instance
        """
        return cls(frame_index=frame_index, sequence_id=sequence_id.strip())
    
    @property
    def is_first_frame(self) -> bool:
        """Check if this is the first frame."""
        return self.frame_index == 0
    
    def next_frame(self) -> 'FrameID':
        """Get next frame ID in sequence."""
        return FrameID(frame_index=self.frame_index + 1, sequence_id=self.sequence_id)
    
    def previous_frame(self) -> 'FrameID':
        """
        Get previous frame ID in sequence.
        
        Returns:
            Previous FrameID
            
        Raises:
            ValueError: If current frame is first frame
        """
        if self.is_first_frame:
            raise ValueError("Cannot get previous frame of first frame")
        return FrameID(frame_index=self.frame_index - 1, sequence_id=self.sequence_id)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.sequence_id}#{self.frame_index}"