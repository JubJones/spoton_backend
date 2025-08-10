"""
Base domain event implementation.

Domain events represent something important that happened in the domain.
They are used for loose coupling between bounded contexts.
"""
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
from uuid import UUID, uuid4


@dataclass(frozen=True)
class BaseDomainEvent(ABC):
    """
    Base class for all domain events.
    
    Domain events are:
    - Immutable (frozen=True)
    - Have a unique identifier
    - Include timestamp when they occurred
    - Contain relevant domain data
    """
    
    event_id: UUID
    occurred_at: datetime
    aggregate_id: UUID
    event_version: int = 1
    
    def __post_init__(self) -> None:
        """Validate event after creation."""
        if self.occurred_at > datetime.utcnow():
            raise ValueError("Event cannot occur in the future")
    
    @classmethod
    def create(
        cls, 
        aggregate_id: UUID, 
        **kwargs: Any
    ) -> "BaseDomainEvent":
        """
        Factory method to create a domain event.
        
        Args:
            aggregate_id: ID of the aggregate that produced this event
            **kwargs: Additional event-specific data
        """
        return cls(
            event_id=uuid4(),
            occurred_at=datetime.utcnow(),
            aggregate_id=aggregate_id,
            **kwargs
        )
    
    @property
    def event_type(self) -> str:
        """Get the event type name."""
        return self.__class__.__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = {}
        for field_name, field in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, datetime):
                result[field_name] = value.isoformat()
            elif isinstance(value, UUID):
                result[field_name] = str(value)
            else:
                result[field_name] = value
        result["event_type"] = self.event_type
        return result
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"{self.event_type}(id={self.event_id}, aggregate={self.aggregate_id})"