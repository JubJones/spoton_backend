"""
Overlay Configuration Entity

Defines configuration for visual overlays on camera frames.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from datetime import datetime


@dataclass
class OverlayConfig:
    """Configuration for visual overlays on camera frames."""
    
    # Bounding box settings
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # RGB color
    bbox_thickness: int = 2
    bbox_opacity: float = 1.0
    
    # Text settings
    text_color: Tuple[int, int, int] = (255, 255, 255)  # RGB color
    text_font_scale: float = 0.7
    text_thickness: int = 2
    text_background_color: Optional[Tuple[int, int, int]] = (0, 0, 0)
    text_background_opacity: float = 0.7
    
    # Focus highlighting
    focus_color: Tuple[int, int, int] = (255, 0, 0)  # Red for focused person
    focus_thickness: int = 4
    focus_opacity: float = 1.0
    
    # Person ID display
    show_person_id: bool = True
    show_confidence: bool = False
    show_tracking_duration: bool = False
    
    # Visual effects
    enable_glow_effect: bool = False
    glow_radius: int = 5
    
    # Quality settings
    overlay_quality: int = 95  # JPEG quality for overlaid frames
    adaptive_quality: bool = True
    
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert overlay config to dictionary."""
        return {
            "bbox_color": self.bbox_color,
            "bbox_thickness": self.bbox_thickness,
            "bbox_opacity": self.bbox_opacity,
            "text_color": self.text_color,
            "text_font_scale": self.text_font_scale,
            "text_thickness": self.text_thickness,
            "text_background_color": self.text_background_color,
            "text_background_opacity": self.text_background_opacity,
            "focus_color": self.focus_color,
            "focus_thickness": self.focus_thickness,
            "focus_opacity": self.focus_opacity,
            "show_person_id": self.show_person_id,
            "show_confidence": self.show_confidence,
            "show_tracking_duration": self.show_tracking_duration,
            "enable_glow_effect": self.enable_glow_effect,
            "glow_radius": self.glow_radius,
            "overlay_quality": self.overlay_quality,
            "adaptive_quality": self.adaptive_quality,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverlayConfig":
        """Create overlay config from dictionary."""
        # Convert string timestamps back to datetime
        created_at = None
        updated_at = None
        
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])
        
        return cls(
            bbox_color=tuple(data.get("bbox_color", (0, 255, 0))),
            bbox_thickness=data.get("bbox_thickness", 2),
            bbox_opacity=data.get("bbox_opacity", 1.0),
            text_color=tuple(data.get("text_color", (255, 255, 255))),
            text_font_scale=data.get("text_font_scale", 0.7),
            text_thickness=data.get("text_thickness", 2),
            text_background_color=tuple(data["text_background_color"]) if data.get("text_background_color") else None,
            text_background_opacity=data.get("text_background_opacity", 0.7),
            focus_color=tuple(data.get("focus_color", (255, 0, 0))),
            focus_thickness=data.get("focus_thickness", 4),
            focus_opacity=data.get("focus_opacity", 1.0),
            show_person_id=data.get("show_person_id", True),
            show_confidence=data.get("show_confidence", False),
            show_tracking_duration=data.get("show_tracking_duration", False),
            enable_glow_effect=data.get("enable_glow_effect", False),
            glow_radius=data.get("glow_radius", 5),
            overlay_quality=data.get("overlay_quality", 95),
            adaptive_quality=data.get("adaptive_quality", True),
            created_at=created_at,
            updated_at=updated_at
        )
    
    def get_bbox_style(self, is_focused: bool = False) -> Dict[str, Any]:
        """Get bounding box style configuration."""
        if is_focused:
            return {
                "color": self.focus_color,
                "thickness": self.focus_thickness,
                "opacity": self.focus_opacity
            }
        return {
            "color": self.bbox_color,
            "thickness": self.bbox_thickness,
            "opacity": self.bbox_opacity
        }
    
    def get_text_style(self) -> Dict[str, Any]:
        """Get text style configuration."""
        return {
            "color": self.text_color,
            "font_scale": self.text_font_scale,
            "thickness": self.text_thickness,
            "background_color": self.text_background_color,
            "background_opacity": self.text_background_opacity
        }