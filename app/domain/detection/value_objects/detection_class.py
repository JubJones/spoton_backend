"""
Detection class value object for object classification.

Provides type-safe detection class handling with predefined categories.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.domain.shared.value_objects.base_value_object import BaseValueObject


class DetectionClassType(Enum):
    """Enumeration of detection class types."""
    PERSON = "person"
    VEHICLE = "vehicle" 
    BICYCLE = "bicycle"
    ANIMAL = "animal"
    OBJECT = "object"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectionClass(BaseValueObject):
    """
    Detection class value object.
    
    Represents object class with type safety and validation.
    """
    
    class_type: DetectionClassType
    class_id: int
    class_name: Optional[str] = None
    
    def _validate(self) -> None:
        """Validate detection class."""
        if self.class_id < 0:
            raise ValueError("Class ID must be non-negative")
    
    @classmethod
    def person(cls, class_id: int = 0) -> 'DetectionClass':
        """Create person detection class."""
        return cls(
            class_type=DetectionClassType.PERSON,
            class_id=class_id,
            class_name="person"
        )
    
    @classmethod
    def vehicle(cls, class_id: int = 1, vehicle_type: Optional[str] = None) -> 'DetectionClass':
        """Create vehicle detection class."""
        return cls(
            class_type=DetectionClassType.VEHICLE,
            class_id=class_id,
            class_name=vehicle_type or "vehicle"
        )
    
    @classmethod
    def from_legacy_id(cls, class_id: int) -> 'DetectionClass':
        """
        Create DetectionClass from legacy class ID.
        
        Maps legacy class IDs to detection types:
        - 0: Person
        - 1: Vehicle
        - Others: Unknown
        
        Args:
            class_id: Legacy class identifier
            
        Returns:
            DetectionClass instance
        """
        if class_id == 0:
            return cls.person(class_id)
        elif class_id == 1:
            return cls.vehicle(class_id)
        else:
            return cls(
                class_type=DetectionClassType.UNKNOWN,
                class_id=class_id,
                class_name=f"class_{class_id}"
            )
    
    @classmethod
    def from_coco_class(cls, class_id: int, class_name: str) -> 'DetectionClass':
        """
        Create DetectionClass from COCO dataset class.
        
        Args:
            class_id: COCO class ID
            class_name: COCO class name
            
        Returns:
            DetectionClass instance
        """
        # Map COCO classes to our types
        coco_to_type = {
            'person': DetectionClassType.PERSON,
            'car': DetectionClassType.VEHICLE,
            'truck': DetectionClassType.VEHICLE,
            'bus': DetectionClassType.VEHICLE,
            'bicycle': DetectionClassType.BICYCLE,
            'motorcycle': DetectionClassType.VEHICLE,
            'dog': DetectionClassType.ANIMAL,
            'cat': DetectionClassType.ANIMAL,
        }
        
        detection_type = coco_to_type.get(class_name.lower(), DetectionClassType.OBJECT)
        
        return cls(
            class_type=detection_type,
            class_id=class_id,
            class_name=class_name
        )
    
    @property
    def is_person(self) -> bool:
        """Check if detection is a person."""
        return self.class_type == DetectionClassType.PERSON
    
    @property
    def is_vehicle(self) -> bool:
        """Check if detection is a vehicle."""
        return self.class_type == DetectionClassType.VEHICLE
    
    @property
    def is_moving_object(self) -> bool:
        """Check if detection represents a moving object."""
        return self.class_type in {DetectionClassType.PERSON, DetectionClassType.VEHICLE, DetectionClassType.BICYCLE}
    
    @property
    def display_name(self) -> str:
        """Get display-friendly class name."""
        return self.class_name or self.class_type.value
    
    def matches_type(self, other_type: DetectionClassType) -> bool:
        """Check if class matches given type."""
        return self.class_type == other_type
    
    def to_legacy_format(self) -> int:
        """Convert to legacy class ID format."""
        legacy_mapping = {
            DetectionClassType.PERSON: 0,
            DetectionClassType.VEHICLE: 1,
            DetectionClassType.BICYCLE: 2,
            DetectionClassType.ANIMAL: 3,
            DetectionClassType.OBJECT: 4,
            DetectionClassType.UNKNOWN: 99
        }
        return legacy_mapping.get(self.class_type, self.class_id)
    
    def __str__(self) -> str:
        """String representation."""
        return self.display_name