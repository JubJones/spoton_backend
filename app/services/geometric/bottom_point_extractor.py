"""
Bottom-center point extraction service for geometric cross-camera matching.

Implements Phase 1 of the space-based re-identification pipeline by providing
utilities to derive the most geometrically stable point (feet contact point)
from tracked bounding boxes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

from app.shared.types import CameraID, TrackID

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImagePoint:
    """Point in image coordinate space anchored at the ground contact location."""

    x: float
    y: float
    camera_id: CameraID
    person_id: Optional[TrackID]
    frame_number: Optional[int]
    timestamp: Optional[float]


class BottomPointExtractor:
    """
    Extract bottom-center points from bounding boxes.

    This point represents the person's ground contact location, which has
    minimal parallax error across different camera angles.
    """

    def __init__(self, validation_enabled: bool = True) -> None:
        """
        Initialize extractor.

        Args:
            validation_enabled: Enable validation checks on extracted points.
        """
        self.validation_enabled = validation_enabled
        self.extraction_count = 0
        self.validation_failures = 0

    def extract_point(
        self,
        bbox_x: float,
        bbox_y: float,
        bbox_width: float,
        bbox_height: float,
        camera_id: CameraID,
        person_id: Optional[TrackID],
        frame_number: Optional[int],
        timestamp: Optional[float],
        frame_width: int,
        frame_height: int,
    ) -> ImagePoint:
        """
        Extract bottom-center point from a single bounding box.

        Args:
            bbox_x: Top-left X coordinate.
            bbox_y: Top-left Y coordinate.
            bbox_width: Bounding box width.
            bbox_height: Bounding box height.
            camera_id: Source camera identifier.
            person_id: Person/track identifier.
            frame_number: Frame number.
            timestamp: Frame timestamp.
            frame_width: Frame width for validation.
            frame_height: Frame height for validation.

        Returns:
            ImagePoint representing ground contact location.

        Raises:
            ValueError: If validation fails and validation_enabled=True.
        """
        point_x = bbox_x + (bbox_width / 2.0)
        point_y = bbox_y + bbox_height

        self.extraction_count += 1

        if self.validation_enabled:
            try:
                self._validate_point(point_x, point_y, frame_width, frame_height, camera_id)
            except ValueError as exc:
                self.validation_failures += 1
                raise exc

        return ImagePoint(
            x=point_x,
            y=point_y,
            camera_id=camera_id,
            person_id=person_id,
            frame_number=frame_number,
            timestamp=timestamp,
        )

    def extract_batch(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        camera_id: CameraID,
        person_ids: List[Optional[TrackID]],
        frame_number: Optional[int],
        timestamp: Optional[float],
        frame_width: int,
        frame_height: int,
    ) -> List[ImagePoint]:
        """
        Extract bottom-center points for a batch of bounding boxes.

        Args:
            bboxes: List of (x, y, width, height) tuples.
            camera_id: Source camera identifier.
            person_ids: List of person/track IDs corresponding to bboxes.
            frame_number: Frame number.
            timestamp: Frame timestamp.
            frame_width: Frame width for validation.
            frame_height: Frame height for validation.

        Returns:
            List of ImagePoint objects.
        """
        if len(bboxes) != len(person_ids):
            raise ValueError("Number of bboxes must match number of person_ids")

        points: List[ImagePoint] = []
        for (x, y, w, h), person_id in zip(bboxes, person_ids):
            try:
                point = self.extract_point(
                    bbox_x=x,
                    bbox_y=y,
                    bbox_width=w,
                    bbox_height=h,
                    camera_id=camera_id,
                    person_id=person_id,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                points.append(point)
            except ValueError as exc:
                logger.debug(
                    "Validation failed for bottom point extraction in camera %s: %s",
                    camera_id,
                    exc,
                )
        return points

    def _validate_point(
        self,
        point_x: float,
        point_y: float,
        frame_width: int,
        frame_height: int,
        camera_id: CameraID,
    ) -> None:
        """
        Validate that extracted point is within reasonable bounds.

        Args:
            point_x: X coordinate to validate.
            point_y: Y coordinate to validate.
            frame_width: Frame width.
            frame_height: Frame height.
            camera_id: Camera identifier for error reporting.

        Raises:
            ValueError: If point is outside frame bounds.
        """
        if point_x < 0 or point_x > frame_width:
            raise ValueError(
                f"Point X {point_x} outside frame bounds [0, {frame_width}] for camera {camera_id}"
            )

        if point_y < 0 or point_y > frame_height:
            raise ValueError(
                f"Point Y {point_y} outside frame bounds [0, {frame_height}] for camera {camera_id}"
            )

    def get_statistics(self) -> dict:
        """Get extraction statistics."""
        success_count = max(self.extraction_count - self.validation_failures, 0)
        success_rate = (
            success_count / self.extraction_count if self.extraction_count > 0 else 0.0
        )
        return {
            "total_attempts": self.extraction_count,
            "validation_failures": self.validation_failures,
            "success_rate": success_rate,
        }
