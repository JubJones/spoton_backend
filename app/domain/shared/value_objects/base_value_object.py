"""
Base value object implementation for the domain layer.

Value objects are immutable objects that are defined by their attributes.
They have no conceptual identity and are compared by their values.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BaseValueObject(ABC):
    """
    Base class for all value objects in the domain.
    
    Value objects must be:
    - Immutable (frozen=True)
    - Compared by value equality
    - Have no conceptual identity
    """
    
    def __post_init__(self) -> None:
        """Validate value object after creation."""
        self._validate()
    
    def _validate(self) -> None:
        """Override in subclasses to add validation logic."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert value object to dictionary representation."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def __str__(self) -> str:
        """String representation of the value object."""
        fields = ", ".join(
            f"{k}={v}" for k, v in self.to_dict().items()
        )
        return f"{self.__class__.__name__}({fields})"