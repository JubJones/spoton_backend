"""
Inverse homography projector used for reprojection debugging (Phase 2b).

Given a world-plane coordinate, this module projects the point back into a
camera's image plane so we can visualize where a person is expected to appear.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from app.shared.types import CameraID
from .world_plane_transformer import WorldPlaneTransformer, WorldPoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectedImagePoint:
    """Predicted location of a person in image coordinates."""

    x: float
    y: float
    camera_id: str
    person_id: Optional[int]
    source_camera_id: str
    world_point: Tuple[float, float]
    frame_number: Optional[int]
    timestamp: Optional[float]
    global_id: Optional[str] = None
    reprojection_error_px: Optional[float] = None
    is_matched: bool = False


class InverseHomographyProjector:
    """
    Convert world-plane points back to image-plane coordinates using inverse
    homography matrices.
    """

    def __init__(
        self,
        *,
        world_plane_transformer: Optional[WorldPlaneTransformer] = None,
        homography_matrices: Optional[Dict[Union[str, CameraID], np.ndarray]] = None,
        homography_file_path: Optional[str] = None,
    ) -> None:
        if world_plane_transformer is not None:
            base_matrices = world_plane_transformer.homography_matrices
        elif homography_matrices is not None:
            base_matrices = homography_matrices
        elif homography_file_path is not None:
            transformer = WorldPlaneTransformer(
                homography_file_path=homography_file_path,
                enable_bounds_validation=False,
            )
            base_matrices = transformer.homography_matrices
        else:
            raise ValueError(
                "InverseHomographyProjector requires either a world-plane transformer, "
                "homography matrices, or a homography file path."
            )

        self.logger = logging.getLogger(__name__)
        self._inverse_matrices: Dict[str, np.ndarray] = {}

        for cam_id_raw, matrix in base_matrices.items():
            camera_id = str(cam_id_raw)
            try:
                inverse_matrix = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                self.logger.warning("Homography matrix for camera %s is singular; skipping.", camera_id)
                continue

            self._inverse_matrices[camera_id] = inverse_matrix.astype(np.float32)

        if not self._inverse_matrices:
            raise ValueError("No valid inverse homography matrices available for reprojection.")

        self.logger.info(
            "InverseHomographyProjector initialized with %d inverse matrices.",
            len(self._inverse_matrices),
        )

    def project(self, world_point: WorldPoint, target_camera: Union[str, CameraID]) -> Optional[ProjectedImagePoint]:
        """
        Project a world coordinate back into a camera image plane.

        Args:
            world_point: Point in world coordinates.
            target_camera: Camera identifier to project into.

        Returns:
            ProjectedImagePoint or None if projection fails.
        """
        camera_key = str(target_camera)
        matrix = self._inverse_matrices.get(camera_key)
        if matrix is None:
            pass # self.logger.debug("No inverse homography found for camera %s; skipping reprojection.", camera_key)
            return None

        input_world = np.array([[[world_point.x, world_point.y]]], dtype=np.float32)
        try:
            projected = cv2.perspectiveTransform(input_world, matrix)
        except cv2.error as exc:
            pass # self.logger.debug("cv2 perspectiveTransform failed for camera %s: %s", camera_key, exc)
            return None

        x_px = float(projected[0, 0, 0])
        y_px = float(projected[0, 0, 1])
        if not np.isfinite(x_px) or not np.isfinite(y_px):
            pass # self.logger.debug(
            #     "Reprojection produced non-finite result for camera %s (%.2f, %.2f).",
            #     camera_key,
            #     x_px,
            #     y_px,
            # )
            return None

        return ProjectedImagePoint(
            x=x_px,
            y=y_px,
            camera_id=camera_key,
            person_id=world_point.person_id,
            source_camera_id=str(world_point.camera_id),
            world_point=(world_point.x, world_point.y),
            frame_number=world_point.frame_number,
            timestamp=world_point.timestamp,
            global_id=getattr(world_point, "global_id", None),
            is_matched=getattr(world_point, "is_matched", False),
        )
