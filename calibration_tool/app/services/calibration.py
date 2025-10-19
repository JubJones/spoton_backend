"""Core calibration state management and math helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class CalibrationSession:
    camera_id: str
    source_points: List[Point] = field(default_factory=list)
    destination_points: List[Point] = field(default_factory=list)
    homography: Optional[np.ndarray] = None
    test_points: List[Point] = field(default_factory=list)
    frame_path: Optional[str] = None
    frame_image: Optional[np.ndarray] = None

    def reset_points(self) -> None:
        self.source_points.clear()
        self.test_points.clear()
        self.homography = None

    def to_serializable(self) -> Dict[str, object]:
        homography_list = self.homography.tolist() if self.homography is not None else None
        return {
            "id": self.camera_id,
            "source_points": [[float(x), float(y)] for x, y in self.source_points],
            "destination_points": [[float(x), float(y)] for x, y in self.destination_points],
            "homography": homography_list,
            "frame_path": self.frame_path,
        }


class CalibrationManager:
    """Tracks calibration sessions for multiple cameras."""

    def __init__(self) -> None:
        self._sessions: Dict[str, CalibrationSession] = {}
        self._shared_destination_points: List[Point] = []

    # Session helpers -------------------------------------------------
    def get_session(self, camera_id: str) -> CalibrationSession:
        if camera_id not in self._sessions:
            self._sessions[camera_id] = CalibrationSession(camera_id=camera_id)
        session = self._sessions[camera_id]
        session.destination_points = self._shared_destination_points
        return self._sessions[camera_id]

    def set_frame(self, camera_id: str, frame_path: str, frame_image: np.ndarray) -> None:
        session = self.get_session(camera_id)
        session.frame_path = frame_path
        session.frame_image = frame_image

    # Point operations ------------------------------------------------
    def add_source_point(self, camera_id: str, point: Point) -> int:
        session = self.get_session(camera_id)
        session.source_points.append(point)
        session.homography = None
        return len(session.source_points)

    def set_source_point(self, camera_id: str, index: int, point: Point) -> None:
        session = self.get_session(camera_id)
        session.source_points[index] = point
        session.homography = None

    def delete_source_point(self, camera_id: str, index: int) -> None:
        session = self.get_session(camera_id)
        session.source_points.pop(index)
        session.homography = None

    def add_destination_point(self, point: Point) -> int:
        self._shared_destination_points.append(point)
        self._invalidate_homographies()
        return len(self._shared_destination_points)

    def set_destination_point(self, index: int, point: Point) -> None:
        self._shared_destination_points[index] = point
        self._invalidate_homographies()

    def delete_destination_point(self, index: int) -> None:
        self._shared_destination_points.pop(index)
        for session in self._sessions.values():
            if len(session.source_points) > index:
                session.source_points.pop(index)
            session.homography = None
            session.test_points.clear()

    @property
    def destination_points(self) -> List[Point]:
        return self._shared_destination_points

    def undo_last_point(self, camera_id: str) -> None:
        session = self.get_session(camera_id)
        if session.source_points:
            session.source_points.pop()
        session.homography = None

    def reset_camera(self, camera_id: str, clear_frame: bool = False) -> None:
        session = self.get_session(camera_id)
        session.reset_points()
        if clear_frame:
            session.frame_path = None
            session.frame_image = None

    # Homography ------------------------------------------------------
    def can_compute_homography(self, camera_id: str) -> bool:
        session = self.get_session(camera_id)
        dest_count = len(self._shared_destination_points)
        return len(session.source_points) >= 4 and len(session.source_points) == dest_count

    def compute_homography(self, camera_id: str) -> np.ndarray:
        session = self.get_session(camera_id)
        if not self.can_compute_homography(camera_id):
            raise ValueError("Need at least 4 matching point pairs before computing homography")

        src = np.array(session.source_points, dtype=np.float32)
        dst = np.array(self._shared_destination_points, dtype=np.float32)
        matrix, mask = cv2.findHomography(src, dst)
        if matrix is None:
            raise RuntimeError("OpenCV could not compute a homography matrix with the provided points")
        session.homography = matrix
        return matrix

    def transform_point(self, camera_id: str, point: Point) -> Point:
        session = self.get_session(camera_id)
        if session.homography is None:
            raise ValueError("No homography available for this camera")
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, session.homography)
        x, y = transformed[0][0]
        session.test_points.append((float(x), float(y)))
        return float(x), float(y)

    # Summary ---------------------------------------------------------
    def sessions(self) -> Sequence[CalibrationSession]:
        return tuple(self._sessions.values())

    def _invalidate_homographies(self) -> None:
        for session in self._sessions.values():
            session.homography = None
            session.test_points.clear()

    def cameras_with_homography(self) -> List[str]:
        return [session.camera_id for session in self._sessions.values() if session.homography is not None]

    def export_mapping(self) -> Dict[str, object]:
        return {
            "cameras": [session.to_serializable() for session in self._sessions.values() if session.homography is not None],
        }

    def has_any_homography(self) -> bool:
        return any(session.homography is not None for session in self._sessions.values())
