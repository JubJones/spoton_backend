"""
World-plane transformer service for pure geometric matching (Phase 2).

Loads per-camera homography matrices and projects image-space points onto a
shared world coordinate frame. Provides statistics to support observability.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.services.geometric.bottom_point_extractor import ImagePoint
from app.shared.types import CameraID, TrackID

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorldPoint:
    """Point in world coordinate space."""

    x: float
    y: float
    camera_id: CameraID
    person_id: Optional[TrackID]
    original_image_point: Tuple[float, float]
    frame_number: Optional[int]
    timestamp: Optional[float]
    transformation_quality: float  # 0..1
    global_id: Optional[str] = None
    is_matched: bool = False


class WorldPlaneTransformer:
    """
    Transform image coordinates to world-plane coordinates using homography matrices.

    When world bounds validation is enabled, transformations that fall outside of the
    configured bounds degrade the confidence score and are tracked as validation failures.
    """

    def __init__(
        self,
        homography_file_path: str,
        *,
        enable_bounds_validation: bool = True,
        world_bounds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.homography_file_path = Path(homography_file_path)
        if not self.homography_file_path.exists():
            raise FileNotFoundError(f"Homography file not found: {homography_file_path}")

        self.homography_matrices = self._load_homography_data()
        self.transformation_count = 0
        self.validation_failures = 0
        self.enable_bounds_validation = enable_bounds_validation
        self.world_bounds = world_bounds or {
            "x_min": settings.WORLD_BOUNDS_X_MIN,
            "x_max": settings.WORLD_BOUNDS_X_MAX,
            "y_min": settings.WORLD_BOUNDS_Y_MIN,
            "y_max": settings.WORLD_BOUNDS_Y_MAX,
        }

        self.logger.info(
            "WorldPlaneTransformer loaded %d homography matrices from %s",
            len(self.homography_matrices),
            homography_file_path,
        )

    @classmethod
    def from_settings(cls) -> Optional["WorldPlaneTransformer"]:
        """Factory helper using global settings; returns None if configuration missing."""
        homography_path = getattr(settings, "HOMOGRAPHY_FILE_PATH", None)
        if not homography_path:
            logger.warning("HOMOGRAPHY_FILE_PATH not configured; world-plane transformer disabled.")
            return None
        try:
            transformer = cls(
                homography_file_path=homography_path,
                enable_bounds_validation=getattr(settings, "ENABLE_BOUNDS_VALIDATION", True),
                world_bounds={
                    "x_min": settings.WORLD_BOUNDS_X_MIN,
                    "x_max": settings.WORLD_BOUNDS_X_MAX,
                    "y_min": settings.WORLD_BOUNDS_Y_MIN,
                    "y_max": settings.WORLD_BOUNDS_Y_MAX,
                },
            )
            return transformer
        except FileNotFoundError as exc:
            logger.warning("World-plane transformer configuration file missing: %s", exc)
        except ValueError as exc:
            logger.warning("World-plane transformer configuration invalid: %s", exc)
        return None

    def _load_homography_data(self) -> Dict[CameraID, np.ndarray]:
        """Load homography matrices from JSON file(s)."""
        data = {"cameras": []}
        files_to_load = []

        if self.homography_file_path.is_dir():
            files_to_load = sorted(list(self.homography_file_path.glob("*.json")))
            if not files_to_load:
                self.logger.warning("No .json files found in %s", self.homography_file_path)
        else:
            files_to_load = [self.homography_file_path]

        for file_path in files_to_load:
            try:
                with file_path.open("r", encoding="utf-8") as fh:
                    file_data = json.load(fh)
                    
                    if isinstance(file_data, dict) and isinstance(file_data.get("cameras"), list):
                        data["cameras"].extend(file_data["cameras"])
                    elif isinstance(file_data, dict):
                        # Handle legacy top-level H_c01_to_world format by mixing it in
                        for k, v in file_data.items():
                            if k not in data and not k == "cameras":
                                data[k] = v
            except json.JSONDecodeError as exc:
                self.logger.error("Invalid JSON in homography file %s: %s", file_path, exc)
            except Exception as e:
                self.logger.error("Error loading homography file %s: %s", file_path, e)

        matrices: Dict[CameraID, np.ndarray] = {}
        if isinstance(data, dict) and isinstance(data.get("cameras"), list):
            for entry in data["cameras"]:
                if not isinstance(entry, dict):
                    continue

                try:
                    camera_id = self._resolve_camera_id(entry)
                except ValueError as exc:
                    self.logger.warning("Skipping homography entry without camera id: %s", exc)
                    continue

                matrix_list = entry.get("homography")
                matrix: Optional[np.ndarray] = None

                if matrix_list is not None:
                    try:
                        matrix = np.array(matrix_list, dtype=np.float32)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid matrix data for camera {camera_id}: {exc}") from exc

                if matrix is None:
                    source_points = entry.get("source_points") or entry.get("image_points")
                    dest_points = entry.get("destination_points") or entry.get("map_points")
                    if source_points and dest_points and len(source_points) >= 4 and len(dest_points) >= 4:
                        try:
                            matrix, _ = cv2.findHomography(
                                np.array(source_points, dtype=np.float32),
                                np.array(dest_points, dtype=np.float32),
                                method=cv2.RANSAC,
                                ransacReprojThreshold=5.0,
                            )
                        except cv2.error as exc:
                            raise ValueError(
                                f"Failed to compute homography for camera {camera_id}: {exc}"
                            ) from exc

                if matrix is None:
                    raise ValueError(f"Homography matrix unavailable for camera {camera_id}")

                if matrix.shape != (3, 3):
                    raise ValueError(
                        f"Matrix for camera {camera_id} has invalid shape {matrix.shape}, expected 3x3"
                    )

                if matrix.dtype != np.float32:
                    matrix = matrix.astype(np.float32)

                det = float(np.linalg.det(matrix))
                if abs(det) < 1e-6:
                    self.logger.warning("Homography matrix for camera %s may be singular (det=%s)", camera_id, det)

                matrices[camera_id] = matrix
        else:
            for key, matrix_list in data.items():
                if not (isinstance(key, str) and key.startswith("H_") and key.endswith("_to_world")):
                    continue
                camera_id = CameraID(key[2:-9])
                try:
                    matrix = np.array(matrix_list, dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid matrix data for camera {camera_id}: {exc}") from exc

                if matrix.shape != (3, 3):
                    raise ValueError(f"Matrix for camera {camera_id} has invalid shape {matrix.shape}, expected 3x3")

                if matrix.dtype != np.float32:
                    matrix = matrix.astype(np.float32)

                det = float(np.linalg.det(matrix))
                if abs(det) < 1e-6:
                    self.logger.warning("Homography matrix for camera %s may be singular (det=%s)", camera_id, det)

                matrices[camera_id] = matrix

        if not matrices:
            raise ValueError(f"No valid homography matrices found in {self.homography_file_path}")

        return matrices

    @staticmethod
    def _resolve_camera_id(entry: Dict[str, object]) -> CameraID:
        """Resolve a CameraID from a consolidated homography entry."""
        candidate = entry.get("camera_id") or entry.get("id")
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate.startswith("c") and candidate[1:].isdigit():
                return CameraID(candidate)

        frame_path = entry.get("frame_path")
        if isinstance(frame_path, str):
            match = re.search(r"/(c\d{2})/", frame_path)
            if match:
                return CameraID(match.group(1))

        if isinstance(candidate, str):
            digits = re.findall(r"\d+", candidate)
            if digits:
                return CameraID(f"c{int(digits[0]):02d}")

        raise ValueError("Unable to determine camera identifier")

    def get_available_cameras(self) -> List[CameraID]:
        """Return camera IDs with available homography matrices."""
        return list(self.homography_matrices.keys())

    def has_camera(self, camera_id: CameraID) -> bool:
        """Check if a camera has an associated homography matrix."""
        return camera_id in self.homography_matrices

    def get_homography_matrix(self, camera_id: CameraID) -> np.ndarray:
        """Return homography matrix for a camera."""
        try:
            return self.homography_matrices[camera_id]
        except KeyError as exc:
            raise KeyError(
                f"No homography matrix found for camera '{camera_id}'. "
                f"Available cameras: {self.get_available_cameras()}"
            ) from exc

    def transform_point(self, image_point: ImagePoint) -> WorldPoint:
        """
        Transform image point to world coordinates.

        Raises:
            KeyError: if camera matrix does not exist.
            ValueError: if transformation yields non-finite values.
        """
        matrix = self.get_homography_matrix(image_point.camera_id)
        input_point = np.array([[[image_point.x, image_point.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(input_point, matrix)

        world_x = float(transformed[0, 0, 0])
        world_y = float(transformed[0, 0, 1])

        if not np.isfinite(world_x) or not np.isfinite(world_y):
            self.validation_failures += 1
            raise ValueError(
                f"Non-finite world coordinates for camera {image_point.camera_id}: ({world_x}, {world_y})"
            )

        transformation_quality = 1.0
        if self.enable_bounds_validation and not self._is_within_world_bounds(world_x, world_y):
            transformation_quality = 0.5
            self.validation_failures += 1
            pass # self.logger.debug(
            #     "World point (%s, %s) outside bounds for camera %s",
            #     f"{world_x:.3f}",
            #     f"{world_y:.3f}",
            #     image_point.camera_id,
            # )

        self.transformation_count += 1

        return WorldPoint(
            x=world_x,
            y=world_y,
            camera_id=image_point.camera_id,
            person_id=image_point.person_id,
            original_image_point=(image_point.x, image_point.y),
            frame_number=image_point.frame_number,
            timestamp=image_point.timestamp,
            transformation_quality=transformation_quality,
        )

    def transform_batch(
        self,
        image_points: List[ImagePoint],
    ) -> List[WorldPoint]:
        """Transform a batch of image points using camera-grouped homography matrices."""
        if not image_points:
            return []

        grouped_indices: Dict[CameraID, List[Tuple[int, ImagePoint]]] = {}
        for idx, point in enumerate(image_points):
            grouped_indices.setdefault(point.camera_id, []).append((idx, point))

        result: List[Optional[WorldPoint]] = [None] * len(image_points)
        for camera_id, points in grouped_indices.items():
            if camera_id not in self.homography_matrices:
                pass # self.logger.debug("Skipping camera %s with no homography matrix", camera_id)
                self.validation_failures += len(points)
                continue

            matrix = self.homography_matrices[camera_id]
            input_array = np.array(
                [[[p.x, p.y]] for _, p in points],
                dtype=np.float32,
            )

            transformed = cv2.perspectiveTransform(input_array, matrix)
            for (original_index, point), transformed_point in zip(points, transformed):
                world_x = float(transformed_point[0, 0])
                world_y = float(transformed_point[0, 1])

                if not np.isfinite(world_x) or not np.isfinite(world_y):
                    self.validation_failures += 1
                    continue

                quality = 1.0
                if self.enable_bounds_validation and not self._is_within_world_bounds(world_x, world_y):
                    quality = 0.5
                    self.validation_failures += 1

                self.transformation_count += 1
                result[original_index] = WorldPoint(
                    x=world_x,
                    y=world_y,
                    camera_id=point.camera_id,
                    person_id=point.person_id,
                    original_image_point=(point.x, point.y),
                    frame_number=point.frame_number,
                    timestamp=point.timestamp,
                    transformation_quality=quality,
                )

        return [rp for rp in result if rp is not None]

    def set_world_bounds(self, *, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """Override world bounds at runtime."""
        self.world_bounds = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }
        self.logger.info("Updated world bounds for transformer: %s", self.world_bounds)

    def _is_within_world_bounds(self, world_x: float, world_y: float) -> bool:
        return (
            self.world_bounds["x_min"] <= world_x <= self.world_bounds["x_max"]
            and self.world_bounds["y_min"] <= world_y <= self.world_bounds["y_max"]
        )

    def get_statistics(self) -> Dict[str, float]:
        """Return aggregate transformation statistics."""
        success_count = max(self.transformation_count - self.validation_failures, 0)
        success_rate = (
            success_count / self.transformation_count if self.transformation_count > 0 else 0.0
        )
        return {
            "total_transformations": self.transformation_count,
            "validation_failures": self.validation_failures,
            "success_rate": success_rate,
            "available_cameras": len(self.homography_matrices),
        }
