"""
User ID value object for user identification.

Immutable value object representing a user identifier.
"""
from dataclasses import dataclass

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class UserID(BaseValueObject):
    """
    User ID value object.
    
    Represents an immutable user identifier with validation.
    """
    value: str
    
    def __post_init__(self):
        """Validate user ID after initialization."""
        if not self.value or not isinstance(self.value, str):
            raise ValueError("User ID must be a non-empty string")
        
        if len(self.value.strip()) == 0:
            raise ValueError("User ID cannot be empty or whitespace")
    
    def __str__(self) -> str:
        """String representation of user ID."""
        return self.value
    
    def __repr__(self) -> str:
        """Representation of user ID."""
        return f"UserID('{self.value}')"