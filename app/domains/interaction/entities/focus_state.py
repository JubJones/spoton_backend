"""
Focus State Entity

Represents the current focus tracking state for a task.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.domains.reid.entities.person_identity import PersonIdentity


@dataclass
class PersonDetails:
    """Detailed information about a focused person."""
    global_id: str
    first_detected: datetime
    tracking_duration: float
    current_camera: str
    position_history: List[Dict[str, Any]] = field(default_factory=list)
    movement_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "global_id": self.global_id,
            "first_detected": self.first_detected.isoformat(),
            "tracking_duration": self.tracking_duration,
            "current_camera": self.current_camera,
            "position_history": self.position_history,
            "movement_metrics": self.movement_metrics
        }


@dataclass
class FocusState:
    """Represents the current focus tracking state."""
    
    task_id: str
    focused_person_id: Optional[str] = None
    focus_start_time: Optional[datetime] = None
    cross_camera_sync: bool = True
    
    # Person details
    person_details: Optional[PersonDetails] = None
    
    # Highlight settings
    highlight_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking state
    active_cameras: List[str] = field(default_factory=list)
    last_seen_camera: Optional[str] = None
    focus_lost: bool = False
    focus_lost_time: Optional[datetime] = None
    
    # Session information
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.highlight_settings:
            self.highlight_settings = {
                "enabled": True,
                "intensity": 0.3,
                "border_thickness": 4,
                "border_color": [255, 0, 0],  # Red
                "glow_effect": True,
                "darken_background": True
            }
    
    def set_focus_person(self, person_id: str, person_details: Optional[PersonDetails] = None):
        """Set the focused person."""
        self.focused_person_id = person_id
        self.person_details = person_details
        self.focus_start_time = datetime.utcnow()
        self.focus_lost = False
        self.focus_lost_time = None
        self.updated_at = datetime.utcnow()
    
    def clear_focus(self):
        """Clear the current focus."""
        self.focused_person_id = None
        self.person_details = None
        self.focus_start_time = None
        self.focus_lost = False
        self.focus_lost_time = None
        self.updated_at = datetime.utcnow()
    
    def mark_focus_lost(self):
        """Mark that the focused person has been lost."""
        self.focus_lost = True
        self.focus_lost_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_person_location(self, camera_id: str, position: Dict[str, Any]):
        """Update the current location of the focused person."""
        if self.person_details and self.focused_person_id:
            self.last_seen_camera = camera_id
            self.person_details.current_camera = camera_id
            
            # Add to position history
            position_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "camera": camera_id,
                **position
            }
            self.person_details.position_history.append(position_record)
            
            # Keep only recent history (last 100 positions)
            if len(self.person_details.position_history) > 100:
                self.person_details.position_history = self.person_details.position_history[-100:]
            
            # Update tracking duration
            if self.focus_start_time:
                self.person_details.tracking_duration = (
                    datetime.utcnow() - self.focus_start_time
                ).total_seconds()
            
            self.focus_lost = False
            self.updated_at = datetime.utcnow()
    
    def get_focus_duration(self) -> float:
        """Get the duration of current focus in seconds."""
        if not self.focus_start_time:
            return 0.0
        return (datetime.utcnow() - self.focus_start_time).total_seconds()
    
    def is_person_focused(self, person_id: str) -> bool:
        """Check if a specific person is currently focused."""
        return self.focused_person_id == person_id and not self.focus_lost
    
    def has_active_focus(self) -> bool:
        """Check if there's an active focus."""
        return self.focused_person_id is not None and not self.focus_lost
    
    def update_highlight_settings(self, settings: Dict[str, Any]):
        """Update highlight settings."""
        self.highlight_settings.update(settings)
        self.updated_at = datetime.utcnow()
    
    def add_active_camera(self, camera_id: str):
        """Add a camera to the active cameras list."""
        if camera_id not in self.active_cameras:
            self.active_cameras.append(camera_id)
            self.updated_at = datetime.utcnow()
    
    def remove_active_camera(self, camera_id: str):
        """Remove a camera from the active cameras list."""
        if camera_id in self.active_cameras:
            self.active_cameras.remove(camera_id)
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert focus state to dictionary."""
        return {
            "task_id": self.task_id,
            "focused_person_id": self.focused_person_id,
            "focus_start_time": self.focus_start_time.isoformat() if self.focus_start_time else None,
            "cross_camera_sync": self.cross_camera_sync,
            "person_details": self.person_details.to_dict() if self.person_details else None,
            "highlight_settings": self.highlight_settings,
            "active_cameras": self.active_cameras,
            "last_seen_camera": self.last_seen_camera,
            "focus_lost": self.focus_lost,
            "focus_lost_time": self.focus_lost_time.isoformat() if self.focus_lost_time else None,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "focus_duration_seconds": self.get_focus_duration(),
            "has_active_focus": self.has_active_focus()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FocusState":
        """Create focus state from dictionary."""
        # Parse datetime fields
        focus_start_time = None
        if data.get("focus_start_time"):
            focus_start_time = datetime.fromisoformat(data["focus_start_time"])
        
        focus_lost_time = None
        if data.get("focus_lost_time"):
            focus_lost_time = datetime.fromisoformat(data["focus_lost_time"])
        
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])
        
        # Parse person details
        person_details = None
        if data.get("person_details"):
            pd_data = data["person_details"]
            person_details = PersonDetails(
                global_id=pd_data["global_id"],
                first_detected=datetime.fromisoformat(pd_data["first_detected"]),
                tracking_duration=pd_data["tracking_duration"],
                current_camera=pd_data["current_camera"],
                position_history=pd_data.get("position_history", []),
                movement_metrics=pd_data.get("movement_metrics", {})
            )
        
        return cls(
            task_id=data["task_id"],
            focused_person_id=data.get("focused_person_id"),
            focus_start_time=focus_start_time,
            cross_camera_sync=data.get("cross_camera_sync", True),
            person_details=person_details,
            highlight_settings=data.get("highlight_settings", {}),
            active_cameras=data.get("active_cameras", []),
            last_seen_camera=data.get("last_seen_camera"),
            focus_lost=data.get("focus_lost", False),
            focus_lost_time=focus_lost_time,
            user_id=data.get("user_id"),
            created_at=created_at,
            updated_at=updated_at
        )
    
    def get_focus_update_payload(self) -> Dict[str, Any]:
        """Get payload for focus update WebSocket message."""
        return {
            "focused_person_id": self.focused_person_id,
            "person_details": self.person_details.to_dict() if self.person_details else None,
            "focus_duration": self.get_focus_duration(),
            "active_cameras": self.active_cameras,
            "last_seen_camera": self.last_seen_camera,
            "focus_lost": self.focus_lost,
            "highlight_settings": self.highlight_settings
        }