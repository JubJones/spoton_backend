"""
Session ID value object for identifying processing sessions.

Provides type-safe session identification with generation capabilities.
"""
from dataclasses import dataclass
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class SessionID(BaseValueObject):
    """
    Session identifier value object.
    
    Identifies processing sessions with UUID-based unique identification.
    """
    
    id: UUID
    created_at: datetime
    name: Optional[str] = None
    
    def _validate(self) -> None:
        """Validate session ID."""
        if self.created_at > datetime.utcnow():
            raise ValueError("Session creation time cannot be in the future")
    
    @classmethod
    def generate(cls, name: Optional[str] = None) -> 'SessionID':
        """
        Generate new session ID.
        
        Args:
            name: Optional human-readable session name
            
        Returns:
            New SessionID instance
        """
        return cls(id=uuid4(), created_at=datetime.utcnow(), name=name)
    
    @classmethod
    def from_string(cls, session_id_str: str, name: Optional[str] = None) -> 'SessionID':
        """
        Create SessionID from string representation.
        
        Args:
            session_id_str: UUID string
            name: Optional session name
            
        Returns:
            SessionID instance
        """
        try:
            session_uuid = UUID(session_id_str)
            return cls(id=session_uuid, created_at=datetime.utcnow(), name=name)
        except ValueError as e:
            raise ValueError(f"Invalid session ID format: {e}")
    
    @property
    def short_id(self) -> str:
        """Get shortened ID for display."""
        return str(self.id)[:8]
    
    @property
    def age_seconds(self) -> float:
        """Get session age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def get_display_name(self) -> str:
        """Get human-readable display name."""
        if self.name:
            return f"{self.name} ({self.short_id})"
        return self.short_id
    
    def __str__(self) -> str:
        """String representation."""
        return str(self.id)