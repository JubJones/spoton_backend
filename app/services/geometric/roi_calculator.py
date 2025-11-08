"""
ROI calculator for search regions on the world plane (Phase 3).

Generates adaptive regions of interest around predicted world coordinates based on
time elapsed, transformation quality, and configurable uncertainty parameters.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class ROIShape(Enum):
    """Supported ROI shapes."""

    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"


@dataclass(frozen=True)
class SearchROI:
    """
    Region of Interest for spatial matching.

    Defines a search region on the world plane where we expect
    to find a matching person from another camera.
    """

    center: Tuple[float, float]
    radius: float
    width: Optional[float] = None
    height: Optional[float] = None
    shape: ROIShape = ROIShape.CIRCULAR

    # Metadata
    source_camera: Optional[str] = None
    dest_camera: Optional[str] = None
    person_id: Optional[int] = None
    timestamp: Optional[float] = None

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a world-plane point falls within this ROI.
        """
        if self.shape == ROIShape.CIRCULAR:
            return self._contains_circular(point)
        return self._contains_rectangular(point)

    def _contains_circular(self, point: Tuple[float, float]) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= self.radius

    def _contains_rectangular(self, point: Tuple[float, float]) -> bool:
        if self.width is None or self.height is None:
            raise ValueError("Rectangular ROI requires width and height to be set.")

        half_w = self.width / 2.0
        half_h = self.height / 2.0

        return (
            self.center[0] - half_w <= point[0] <= self.center[0] + half_w
            and self.center[1] - half_h <= point[1] <= self.center[1] + half_h
        )

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Euclidean distance from ROI center to a world-plane point."""
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return math.sqrt(dx**2 + dy**2)

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Serialize ROI to a dictionary suitable for JSON outputs."""
        return {
            "center_x": self.center[0],
            "center_y": self.center[1],
            "radius": self.radius,
            "width": self.width,
            "height": self.height,
            "shape": self.shape.value,
            "source_camera": self.source_camera,
            "dest_camera": self.dest_camera,
            "person_id": self.person_id,
            "timestamp": self.timestamp,
        }


class ROICalculator:
    """
    Calculate search regions for cross-camera geometric matching.

    The ROI size adapts based on:
    - Time elapsed (longer time = larger uncertainty)
    - Homography quality (poor calibration = larger uncertainty)
    - Expected camera handoff characteristics
    """

    def __init__(
        self,
        base_radius: float = 1.5,
        max_walking_speed: float = 1.5,
        min_radius: float = 0.5,
        max_radius: float = 10.0,
    ) -> None:
        self.base_radius = base_radius
        self.max_walking_speed = max_walking_speed
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.logger = logging.getLogger(__name__)

        self.roi_count = 0

    def calculate_roi(
        self,
        predicted_location: Tuple[float, float],
        *,
        time_elapsed: float = 0.0,
        transformation_quality: Optional[float] = 1.0,
        shape: ROIShape = ROIShape.CIRCULAR,
        source_camera: Optional[str] = None,
        dest_camera: Optional[str] = None,
        person_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> SearchROI:
        """
        Calculate search ROI for a predicted world-plane location.
        """
        radius = self.base_radius

        time_uncertainty = max(0.0, time_elapsed) * self.max_walking_speed
        radius += time_uncertainty

        quality = transformation_quality if transformation_quality is not None else 1.0
        quality = max(0.0, min(1.0, quality))
        quality_factor = 1.0 + (1.0 - quality)
        radius *= quality_factor

        radius = max(self.min_radius, min(radius, self.max_radius))

        self.roi_count += 1

        self.logger.debug(
            "ROI generated: center=(%.2f, %.2f) radius=%.2f (time=%.2fs, quality=%.2f)",
            predicted_location[0],
            predicted_location[1],
            radius,
            time_elapsed,
            quality,
        )

        return SearchROI(
            center=predicted_location,
            radius=radius,
            shape=shape,
            source_camera=source_camera,
            dest_camera=dest_camera,
            person_id=person_id,
            timestamp=timestamp,
        )

    def calculate_rectangular_roi(
        self,
        predicted_location: Tuple[float, float],
        *,
        time_elapsed: float = 0.0,
        transformation_quality: Optional[float] = 1.0,
        corridor_direction: str = "horizontal",
        source_camera: Optional[str] = None,
        dest_camera: Optional[str] = None,
        person_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> SearchROI:
        """
        Calculate rectangular ROI for corridor-like scenarios.
        """
        base_dimension = self.base_radius * 2.0
        time_expansion = max(0.0, time_elapsed) * self.max_walking_speed * 2.0
        quality = transformation_quality if transformation_quality is not None else 1.0
        quality = max(0.0, min(1.0, quality))
        quality_factor = 1.0 + (1.0 - quality)

        if corridor_direction == "horizontal":
            width = (base_dimension + time_expansion) * quality_factor
            height = base_dimension * quality_factor
        else:
            width = base_dimension * quality_factor
            height = (base_dimension + time_expansion) * quality_factor

        width = max(self.min_radius * 2, min(width, self.max_radius * 2))
        height = max(self.min_radius * 2, min(height, self.max_radius * 2))

        self.roi_count += 1

        return SearchROI(
            center=predicted_location,
            radius=0.0,
            width=width,
            height=height,
            shape=ROIShape.RECTANGULAR,
            source_camera=source_camera,
            dest_camera=dest_camera,
            person_id=person_id,
            timestamp=timestamp,
        )

    def get_statistics(self) -> Dict[str, float]:
        """Get ROI calculation statistics."""
        return {
            "total_rois_created": self.roi_count,
            "base_radius": self.base_radius,
            "max_walking_speed": self.max_walking_speed,
        }
