"""
Base entity implementation for the domain layer.

Entities are objects with conceptual identity that persists over time.
They are compared by their identity, not their attributes.
"""
from abc import ABC
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


class BaseEntity(ABC):
    """
    Base class for all entities in the domain.
    
    Entities have:
    - Conceptual identity (ID)
    - Mutable attributes
    - Identity-based equality
    """
    
    def __init__(self, entity_id: Optional[UUID] = None):
        """
        Initialize entity with ID.
        
        Args:
            entity_id: Unique identifier for the entity.
                      If None, a new UUID will be generated.
        """
        self._id = entity_id or uuid4()
        self._version = 0
    
    @property
    def id(self) -> UUID:
        """Get the entity's unique identifier."""
        return self._id
    
    @property
    def version(self) -> int:
        """Get the entity's version for optimistic concurrency control."""
        return self._version
    
    def increment_version(self) -> None:
        """Increment version number (for optimistic locking)."""
        self._version += 1
    
    def __eq__(self, other: Any) -> bool:
        """Entities are equal if they have the same ID and type."""
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self.id)
    
    def __str__(self) -> str:
        """String representation of the entity."""
        return f"{self.__class__.__name__}(id={self.id})"
    
    def __repr__(self) -> str:
        """Detailed representation of the entity."""
        return f"{self.__class__.__name__}(id={self.id}, version={self.version})"