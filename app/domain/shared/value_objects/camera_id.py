"""
Camera ID value object for identifying camera sources.

Provides type-safe camera identification across the domain.
"""
from dataclasses import dataclass
from typing import Optional

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class CameraID(BaseValueObject):
    """
    Camera identifier value object.
    
    Provides type-safe camera identification with validation
    and conversion capabilities for different naming schemes.
    """
    
    id: str
    name: Optional[str] = None
    location: Optional[str] = None
    
    def _validate(self) -> None:
        """Validate camera ID."""
        if not self.id:
            raise ValueError("Camera ID cannot be empty")
        if not self.id.strip():
            raise ValueError("Camera ID cannot be only whitespace")
    
    @classmethod
    def from_string(cls, camera_id: str) -> 'CameraID':
        """
        Create CameraID from string.
        
        Args:
            camera_id: Camera identifier string
            
        Returns:
            CameraID instance
        """
        return cls(id=camera_id.strip())
    
    @classmethod
    def from_legacy(cls, camera_id: str) -> 'CameraID':
        """
        Create CameraID from legacy camera identifier.
        
        Maps legacy camera names to standardized format:
        - c09 -> camera1
        - c12 -> camera2  
        - c13 -> camera3
        - c16 -> camera4
        
        Args:
            camera_id: Legacy camera identifier
            
        Returns:
            CameraID instance
        """
        legacy_mapping = {
            'c09': 'camera1',
            'c12': 'camera2', 
            'c13': 'camera3',
            'c16': 'camera4'
        }
        
        standardized_id = legacy_mapping.get(camera_id.lower(), camera_id)
        return cls(id=standardized_id, name=camera_id if camera_id != standardized_id else None)
    
    def to_legacy_format(self) -> str:
        """
        Convert to legacy camera format.
        
        Returns:
            Legacy camera identifier
        """
        reverse_mapping = {
            'camera1': 'c09',
            'camera2': 'c12',
            'camera3': 'c13', 
            'camera4': 'c16'
        }
        
        return reverse_mapping.get(self.id.lower(), self.id)
    
    def is_valid_camera(self) -> bool:
        """Check if this is a valid camera identifier."""
        valid_cameras = {'camera1', 'camera2', 'camera3', 'camera4', 'c09', 'c12', 'c13', 'c16'}
        return self.id.lower() in valid_cameras
    
    def get_display_name(self) -> str:
        """Get human-readable display name."""
        if self.name:
            return f"{self.name} ({self.id})"
        return self.id
    
    def __str__(self) -> str:
        """String representation."""
        return self.id