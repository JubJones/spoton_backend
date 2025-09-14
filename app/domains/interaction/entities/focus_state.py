"""
Simple focus state entities for WebSocket focus tracking.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PersonDetails:
    """Person details for focus tracking."""
    person_id: str
    confidence: float = 0.0
    location: Optional[Dict[str, Any]] = None


@dataclass
class FocusState:
    """Focus state for person tracking."""
    task_id: str
    focused_person_id: Optional[str] = None
    person_details: Optional[PersonDetails] = None
    is_active: bool = False