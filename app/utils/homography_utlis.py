"""
Utilities for loading homography matrices and projecting points.
Adapted from reid_poc.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.shared.types import CameraID # Relative import

logger = logging.getLogger(__name__)

def load_homography_matrix_from_points_file(
    camera_id: CameraID,
    scene_id: str, # Or environment_id, depending on naming convention
    homography_points_dir: Path
) -> Optional[np.ndarray]:
    """
    Loads homography points from an .npz file and computes the homography matrix.

    Args:
        camera_id: The ID of the camera.
        scene_id: The ID of the scene/environment.
        homography_points_dir: Path to the directory containing .npz files.

    Returns:
        A NumPy array representing the homography matrix, or None if loading/computation fails.
    """
    filename = homography_points_dir / f"homography_points_{str(camera_id)}_scene_{scene_id}.npz"
    logger.debug(f"Attempting to load homography points from: {filename}")
    if not filename.is_file():
        logger.warning(f"Homography points file not found for Cam {camera_id}, Scene {scene_id}: {filename}")
        return None

    try:
        data = np.load(str(filename))
        image_points = data['image_points']
        map_points = data['map_points']

        logger.debug(f"[{camera_id}] Loaded points from {filename.name}. Image points shape: {image_points.shape}, Map points shape: {map_points.shape}")

        if len(image_points) < 4 or len(map_points) < 4:
            logger.warning(f"Insufficient points (<4) found in {filename} for Cam {camera_id}.")
            return None
        if len(image_points) != len(map_points):
            logger.error(f"Mismatch in point counts in {filename} for Cam {camera_id}.")
            return None

        # Use findHomography which is robust (RANSAC) and handles > 4 points
        homography_matrix, mask = cv2.findHomography(image_points, map_points, cv2.RANSAC, 5.0)

        if homography_matrix is None:
            logger.error(f"Homography calculation failed (cv2.findHomography returned None) for Cam {camera_id} using {filename}.")
            return None

        logger.info(f"Successfully loaded and computed homography matrix for Cam {camera_id}, Scene {scene_id} from {filename.name}")
        logger.debug(f"[{camera_id}] Calculated Homography Matrix (shape: {homography_matrix.shape}, type: {homography_matrix.dtype})")
        return homography_matrix

    except Exception as e:
        logger.error(f"Error loading or computing homography for Cam {camera_id} from {filename}: {e}", exc_info=True)
        return None


def project_point_to_map(
    image_point_xy: Tuple[float, float],
    homography_matrix: np.ndarray
) -> Optional[Tuple[float, float]]:
    """
    Projects a single image point (x, y) to map coordinates (X, Y) using the homography matrix.

    Args:
        image_point_xy: A tuple (x, y) representing the point in image coordinates.
        homography_matrix: The 3x3 homography matrix.

    Returns:
        A tuple (X, Y) representing the projected map coordinates, or None if projection fails.
    """
    if homography_matrix is None:
        logger.debug(f"Projection skipped: Homography matrix is None for point {image_point_xy}")
        return None

    # logger.debug(f"Projecting image point {image_point_xy} using H matrix (shape {homography_matrix.shape})")

    try:
        # Input point needs to be in shape (1, 1, 2) for perspectiveTransform
        img_pt_np = np.array([[image_point_xy]], dtype=np.float32)

        # Apply perspective transformation
        map_pt_np = cv2.perspectiveTransform(img_pt_np, homography_matrix)

        if map_pt_np is not None and map_pt_np.shape == (1, 1, 2):
            map_x = float(map_pt_np[0, 0, 0])
            map_y = float(map_pt_np[0, 0, 1])
            # logger.debug(f"  -> Projection successful: Map point ({map_x:.2f}, {map_y:.2f})")
            return (map_x, map_y)
        else:
            logger.warning(f"Perspective transform returned unexpected shape or None for point {image_point_xy}. Result: {map_pt_np}")
            return None

    except Exception as e:
        logger.debug(f"Error projecting point {image_point_xy}: {e}", exc_info=False)
        return None