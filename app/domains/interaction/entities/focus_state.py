"""Focus state entities and helpers for tracking a focused person."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PersonDetails:
    """Detailed metadata about a focused person."""

    global_id: str
    first_detected: datetime
    tracking_duration: float = 0.0
    current_camera: Optional[str] = None
    position_history: List[Dict[str, Any]] = field(default_factory=list)
    movement_metrics: Dict[str, Any] = field(default_factory=dict)
    camera_id: Optional[str] = None
    track_id: Optional[int] = None
    detection_id: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None

    def update_observation(
        self,
        *,
        camera_id: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        detection_id: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> None:
        if camera_id is not None:
            self.camera_id = camera_id
            self.current_camera = camera_id

        if bbox is not None:
            try:
                x1, y1, x2, y2 = bbox
                self.bbox = {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            except (TypeError, ValueError):
                pass

        if confidence is not None:
            self.confidence = float(confidence)

        if detection_id is not None:
            self.detection_id = detection_id

        if track_id is not None:
            self.track_id = track_id


@dataclass
class FocusState:
    """In-memory representation of the current focus selection."""

    task_id: str
    focused_person_id: Optional[str] = None
    person_details: Optional[PersonDetails] = None
    is_active: bool = False
    highlight_settings: Dict[str, Any] = field(default_factory=dict)
    focus_started_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None

    def set_focus_person(self, person_id: str, details: PersonDetails) -> None:
        self.focused_person_id = person_id
        self.person_details = details
        self.is_active = True
        now = _utcnow()
        self.focus_started_at = now
        self.last_updated_at = now

    def clear_focus(self) -> None:
        self.focused_person_id = None
        self.person_details = None
        self.is_active = False
        self.last_updated_at = _utcnow()

    def update_person_location(self, camera_id: str, position: Dict[str, Any]) -> None:
        if not self.person_details:
            return

        bbox = None
        if position:
            bbox = [
                position.get("x1"),
                position.get("y1"),
                position.get("x2"),
                position.get("y2"),
            ]

        self.person_details.update_observation(
            camera_id=camera_id,
            bbox=bbox,
            confidence=position.get("confidence"),
            detection_id=position.get("detection_id"),
            track_id=position.get("track_id"),
        )
        self.last_updated_at = _utcnow()

    def update_highlight_settings(self, settings: Dict[str, Any]) -> None:
        if not settings:
            return
        self.highlight_settings.update(settings)
        self.last_updated_at = _utcnow()

    def has_active_focus(self) -> bool:
        return self.is_active and self.focused_person_id is not None

    def get_focus_duration(self) -> float:
        if not self.focus_started_at:
            return 0.0
        return (_utcnow() - self.focus_started_at).total_seconds()

    def record_observation(
        self,
        *,
        camera_id: str,
        bbox: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        detection_id: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> None:
        if not self.person_details and self.focused_person_id:
            self.person_details = PersonDetails(
                global_id=self.focused_person_id,
                first_detected=self.focus_started_at or _utcnow(),
            )

        if not self.person_details:
            return

        self.person_details.update_observation(
            camera_id=camera_id,
            bbox=bbox,
            confidence=confidence,
            detection_id=detection_id,
            track_id=track_id,
        )
        self.last_updated_at = _utcnow()

    def get_focus_update_payload(self) -> Dict[str, Any]:
        details: Optional[Dict[str, Any]] = None
        if self.person_details:
            details = {
                "global_id": self.person_details.global_id,
                "current_camera": self.person_details.current_camera or self.person_details.camera_id,
                "camera_id": self.person_details.camera_id,
                "track_id": self.person_details.track_id,
                "detection_id": self.person_details.detection_id,
                "bbox": self.person_details.bbox,
                "confidence": self.person_details.confidence,
                "first_detected": self.person_details.first_detected.isoformat(),
                "tracking_duration": self.person_details.tracking_duration,
                "position_history": self.person_details.position_history,
                "movement_metrics": self.person_details.movement_metrics,
            }

        return {
            "task_id": self.task_id,
            "focused_person_id": self.focused_person_id,
            "is_active": self.has_active_focus(),
            "focus_started_at": self.focus_started_at.isoformat() if self.focus_started_at else None,
            "last_updated_at": self.last_updated_at.isoformat() if self.last_updated_at else None,
            "highlight_settings": self.highlight_settings,
            "person_details": details,
            "focus_duration": self.get_focus_duration(),
        }

    def get_focus_target_summary(self) -> Dict[str, Any]:
        if not self.has_active_focus() or not self.person_details:
            return {}

        return {
            "focused_person_id": self.focused_person_id,
            "camera_id": self.person_details.camera_id or self.person_details.current_camera,
            "track_id": self.person_details.track_id,
            "detection_id": self.person_details.detection_id,
            "bbox": self.person_details.bbox,
            "confidence": self.person_details.confidence,
        }
