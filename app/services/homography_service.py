"""
Homography Service for coordinate transformations between image and map coordinates.

Enhanced version for Phase 4: Spatial Intelligence that supports both the existing
preload system and new Phase 4 features including JSON configuration files,
coordinate projection, and WebSocket data formatting.

Features:
- Load homography matrices from JSON configuration files (Phase 4)
- Compute homography matrices from calibration points using RANSAC (Phase 4) 
- Project image coordinates to map coordinates (Phase 4)
- Support for multiple cameras with individual calibration data (Phase 4)
- Backwards compatibility with existing preload system
"""
import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2  # For cv2.findHomography

from app.core.config import Settings  # Use the main Settings class
from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class HomographyService:
    """
    Manages pre-loading and providing homography matrices.
    Homography matrices are computed from point files at application startup.
    """

    def __init__(self, settings: Settings, homography_dir: str = "homography_data/"):
        """Initialize the homography service using a single consolidated JSON file."""
        self._settings = settings
        self.homography_dir = Path(homography_dir)

        # Phase 4: JSON-driven calibration storage
        self.json_homography_matrices: Dict[str, np.ndarray] = {}
        self.calibration_points: Dict[str, Dict[str, Any]] = {}
        self.ransac_threshold = 5.0  # Maximum allowed reprojection error
        self._warned_cameras: set = set()  # Track warned cameras to avoid spam

        # Default bounds (Campus)
        self.x_min = settings.WORLD_BOUNDS_X_MIN
        self.x_max = settings.WORLD_BOUNDS_X_MAX
        self.y_min = settings.WORLD_BOUNDS_Y_MIN
        self.y_max = settings.WORLD_BOUNDS_Y_MAX

        # Configuration
        self.homography_file_path = self._resolve_homography_file_path()
        self._loaded = False

        # Load matrices immediately so the service is ready for use.
        self.load_json_homography_data()

        logger.info(
            "HomographyService initialized. Loaded %d camera matrices from %s",
            len(self.json_homography_matrices),
            self.homography_file_path,
        )

    def _resolve_homography_file_path(self) -> Optional[Path]:
        """Resolve the configured homography JSON file path."""
        configured = getattr(self._settings, "HOMOGRAPHY_FILE_PATH", None)
        if not configured:
            return None

        path = Path(configured)
        if not path.is_absolute():
            # Resolve relative to the project root (current working directory)
            path = (Path.cwd() / path).resolve()

        return path

    def _ingest_consolidated_payload(self, payload: Dict[str, Any]) -> int:
        """Populate local caches from the consolidated homography JSON payload."""
        cameras = payload.get("cameras")
        loaded_count = 0

        if isinstance(cameras, list):
            for entry in cameras:
                if not isinstance(entry, dict):
                    continue

                try:
                    camera_id = self._resolve_camera_id(entry)
                except ValueError as exc:
                    logger.warning("Skipping homography entry without camera id: %s", exc)
                    continue

                matrix_list = entry.get("homography")
                if matrix_list is not None:
                    matrix = np.array(matrix_list, dtype=np.float64)
                    if matrix.shape == (3, 3):
                        self.json_homography_matrices[camera_id] = matrix
                    else:
                        logger.warning(
                            "Invalid homography shape for %s: expected (3, 3), got %s",
                            camera_id,
                            matrix.shape,
                        )

                image_points = entry.get("source_points") or entry.get("image_points")
                map_points = entry.get("destination_points") or entry.get("map_points")
                if image_points and map_points and len(image_points) >= 4 and len(map_points) >= 4:
                    self.calibration_points[camera_id] = {
                        "image_points": image_points,
                        "map_points": map_points,
                    }

                loaded_count += 1

        else:
            # Fallback to legacy format with top-level H_<cam>_to_world keys
            for key, matrix_list in payload.items():
                if not (isinstance(key, str) and key.startswith("H_") and key.endswith("_to_world")):
                    continue
                camera_id = key[2:-9]
                try:
                    matrix = np.array(matrix_list, dtype=np.float64)
                except (TypeError, ValueError):
                    logger.warning("Invalid matrix data for key %s", key)
                    continue

                if matrix.shape != (3, 3):
                    logger.warning(
                        "Matrix for camera %s has invalid shape %s (expected 3x3)",
                        camera_id,
                        matrix.shape,
                    )
                    continue

                self.json_homography_matrices[camera_id] = matrix
                loaded_count += 1

        return loaded_count

    @staticmethod
    def _resolve_camera_id(entry: Dict[str, Any]) -> str:
        """Attempt to resolve a camera identifier from a JSON entry."""
        # Preferred: explicit camera_id field
        candidate = entry.get("camera_id") or entry.get("id")
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate.startswith("c") and candidate[1:].isdigit():
                return candidate

        # Try to parse from frame path (e.g., .../c03/....png)
        frame_path = entry.get("frame_path")
        if isinstance(frame_path, str):
            match = re.search(r"/(c\d{2})/", frame_path)
            if match:
                return match.group(1)

        # Fall back to numeric suffix in id (e.g., "Cam 3" -> c03)
        if isinstance(candidate, str):
            digits = re.findall(r"\d+", candidate)
            if digits:
                return f"c{int(digits[0]):02d}"

        raise ValueError("Unable to determine camera identifier")

    def _ensure_loaded(self) -> None:
        """Reload the homography data if it has not been loaded yet."""
        if not self._loaded:
            self.load_json_homography_data()

    async def preload_all_homography_matrices(self) -> None:
        """Ensure homography matrices are available before processing begins."""
        if self._loaded:
            logger.info(
                "Homography matrices already loaded from %s", self.homography_file_path
            )
            return

        self.load_json_homography_data()
        logger.info(
            "Homography matrices preloaded from %s (%d cameras)",
            self.homography_file_path,
            len(self.json_homography_matrices),
        )


    def get_homography_matrix(self, environment_id: str, camera_id: CameraID) -> Optional[np.ndarray]:
        """Return the homography matrix for the requested camera."""
        del environment_id  # Environment-scoped matrices are replaced by a global JSON.

        self._ensure_loaded()

        camera_key = str(camera_id)
        matrix = self.json_homography_matrices.get(camera_key)
        if matrix is not None:
            return matrix

        # Attempt lazy computation from calibration points if a matrix was not provided
        if camera_key not in self.calibration_points:
            if camera_key not in self._warned_cameras:
                logger.warning("No homography data available for camera %s", camera_key)
                self._warned_cameras.add(camera_key)
            return None

        return self.compute_homography(camera_key)
    
    # Phase 4: Spatial Intelligence Methods
    
    def load_json_homography_data(self) -> None:
        """Load homography matrices and calibration points from JSON file(s)."""
        self.json_homography_matrices.clear()
        self.calibration_points.clear()

        path = self.homography_file_path
        if path is None:
            logger.warning("HOMOGRAPHY_FILE_PATH is not configured; unable to load homography data.")
            self._loaded = False
            return

        if not path.exists():
            logger.warning("Homography path not found: %s", path)
            self._loaded = False
            return

        files_to_load = []
        if path.is_dir():
            # Load all .json files in the directory
            files_to_load = sorted(list(path.glob("*.json")))
            if not files_to_load:
                logger.warning("No .json files found in homography directory: %s", path)
        else:
            files_to_load = [path]

        total_loaded = 0
        for file_path in files_to_load:
            try:
                with file_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                    count = self._ingest_consolidated_payload(payload)
                    total_loaded += count
                    logger.info("Loaded %d cameras from %s", count, file_path.name)
            except json.JSONDecodeError as exc:
                logger.error("Invalid homography JSON %s: %s", file_path, exc)
            except Exception as e:
                logger.error("Error loading homography file %s: %s", file_path, e)

        self._loaded = total_loaded > 0

        if self._loaded:
            logger.info("Total loaded homography data for %d cameras from %s", total_loaded, path)
        else:
            logger.warning("No valid homography matrices were loaded from %s", path)
    
    def compute_homography(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Compute homography matrix from calibration points using RANSAC (Phase 4).
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            3x3 homography matrix or None if computation fails
        """
        self._ensure_loaded()

        camera_key = str(camera_id)

        if camera_key not in self.calibration_points:
            # Only warn once per camera to avoid log spam
            if camera_key not in self._warned_cameras:
                logger.warning("No calibration points available for camera %s", camera_key)
                self._warned_cameras.add(camera_key)
            return None

        calib_data = self.calibration_points[camera_key]
        
        try:
            # Convert points to numpy arrays
            image_points = np.array(calib_data['image_points'], dtype=np.float32)
            map_points = np.array(calib_data['map_points'], dtype=np.float32)
            
            # Validate minimum number of points
            if len(image_points) < 4 or len(map_points) < 4:
                logger.error("Insufficient calibration points for camera %s: %d", camera_key, len(image_points))
                return None

            if len(image_points) != len(map_points):
                logger.error(
                    "Mismatched calibration points for camera %s: %d vs %d",
                    camera_key,
                    len(image_points),
                    len(map_points),
                )
                return None
            
            # Compute homography using RANSAC for robust estimation
            matrix, mask = cv2.findHomography(
                image_points, 
                map_points, 
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold
            )

            if matrix is not None and matrix.shape == (3, 3):
                # Store computed matrix for future use
                self.json_homography_matrices[camera_key] = matrix

                # Log computation details
                inliers = np.sum(mask) if mask is not None else len(image_points)
                logger.info(
                    "Computed homography for camera %s: %s/%s inliers",
                    camera_key,
                    inliers,
                    len(image_points),
                )

                return matrix
            else:
                logger.error("Failed to compute valid homography matrix for camera %s", camera_key)
                return None
                
        except Exception as e:
            logger.error("Error computing homography for camera %s: %s", camera_key, e)
            return None
    
    def project_to_map(self, camera_id: str, image_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Project image coordinates to map coordinates using homography (Phase 4).

        Args:
            camera_id: Camera identifier
            image_point: (x, y) coordinates in image space

        Returns:
            (map_x, map_y) coordinates in map space or None if projection fails
        """
        self._ensure_loaded()

        camera_key = str(camera_id)

        # Get or compute homography matrix from JSON data
        if camera_key not in self.json_homography_matrices:
            pass # logger.debug("No cached JSON matrix for camera %s, attempting to compute", camera_key)
            matrix = self.compute_homography(camera_key)
            if matrix is None:
                logger.info(f"ProjectToMap: No matrix found for {camera_key}")
                return None
        else:
            matrix = self.json_homography_matrices[camera_key]

        try:
            # Prepare point for perspective transformation
            # OpenCV expects shape (1, 1, 2) for single point transformation
            point = np.array([[image_point[0], image_point[1]]], dtype=np.float32)
            point = point.reshape(-1, 1, 2)
            
            # Apply homography transformation
            transformed = cv2.perspectiveTransform(point, matrix)
            
            # Extract transformed coordinates
            map_x, map_y = transformed[0, 0]
            # logger.debug(
            #     "Homography projection for camera %s: image_point=%s -> map=(%.4f, %.4f)",
            #     camera_key,
            #     image_point,
            #     map_x,
            #     map_y,
            # )

            # Validate transformed coordinates
            if np.isfinite(map_x) and np.isfinite(map_y):
                return float(map_x), float(map_y)
            else:
                logger.warning("Invalid projection result for camera %s: (%.4f, %.4f)", camera_key, map_x, map_y)
                return None

        except Exception as e:
            logger.error("Error projecting point for camera %s: %s", camera_key, e)
            return None
    
    def get_homography_data(self, camera_id: str) -> Dict:
        """
        Get comprehensive homography data for WebSocket response (Phase 4).
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary containing matrix availability, matrix data, and calibration points
        """
        self._ensure_loaded()

        camera_key = str(camera_id)
        data = {
            "matrix_available": camera_key in self.json_homography_matrices,
            "matrix": None,
            "calibration_points": None
        }
        
        # Include matrix data if available
        if camera_key in self.json_homography_matrices:
            data["matrix"] = self.json_homography_matrices[camera_key].tolist()
        
        # Include calibration points if available
        if camera_key in self.calibration_points:
            data["calibration_points"] = self.calibration_points[camera_key]
        
        return data

    # ====== Phase 4: Coordinate Validation & Bounds Helpers ======
    def get_map_bounds(self, camera_id: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Compute expected map bounds from calibration points for a camera.
        Returns (min_x, max_x, min_y, max_y) or None if unavailable.
        """
        try:
            self._ensure_loaded()

            camera_key = str(camera_id)

            if camera_key not in self.calibration_points:
                return None
            map_points = self.calibration_points[camera_key].get("map_points")
            if not map_points:
                return None
            arr = np.array(map_points, dtype=np.float64)
            min_x = float(np.min(arr[:, 0]))
            max_x = float(np.max(arr[:, 0]))
            min_y = float(np.min(arr[:, 1]))
            max_y = float(np.max(arr[:, 1]))
            # Add 10% padding
            pad_x = max(1.0, 0.1 * max(abs(min_x), abs(max_x)))
            pad_y = max(1.0, 0.1 * max(abs(min_y), abs(max_y)))
            return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)
        except Exception as e:
            pass # logger.debug("Failed computing map bounds for %s: %s", camera_id, e)
            return None

    def validate_map_coordinate(self, camera_id: str, map_x: float, map_y: float) -> bool:
        """
        Validate that map coordinate is finite and within configured WORLD BOUNDS.
        Uses instance-level bounds that may be environment-specific.
        """
        if not (np.isfinite(map_x) and np.isfinite(map_y)):
            return False

        if not settings.ENABLE_BOUNDS_VALIDATION:
            return True

        # Strict check against configured world bounds (default or factory)
        if (map_x < self.x_min or map_x > self.x_max or
            map_y < self.y_min or map_y > self.y_max):
            return False

        return True
