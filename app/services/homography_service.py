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
from typing import Dict, Optional, Tuple
from pathlib import Path
import asyncio
import json

import numpy as np
import cv2 # For cv2.findHomography

from app.core.config import Settings # Use the main Settings class
from app.shared.types import CameraID, CameraHandoffDetailConfig

logger = logging.getLogger(__name__)

class HomographyService:
    """
    Manages pre-loading and providing homography matrices.
    Homography matrices are computed from point files at application startup.
    """

    def __init__(self, settings: Settings, homography_dir: str = "homography_data/"):
        """
        Initializes the HomographyService with both existing and Phase 4 capabilities.

        Args:
            settings: The application settings instance.
            homography_dir: Directory containing Phase 4 JSON homography files
        """
        self._settings = settings
        self._homography_matrices: Dict[Tuple[str, CameraID], Optional[np.ndarray]] = {}
        self._preloaded = False
        
        # Phase 4: Additional storage for JSON-based homography data
        self.homography_dir = Path(homography_dir)
        self.json_homography_matrices: Dict[str, np.ndarray] = {}
        self.calibration_points: Dict[str, Dict] = {}
        self.ransac_threshold = 5.0  # Maximum allowed reprojection error
        self._warned_cameras: set = set()  # Track warned cameras to avoid spam
        
        # Load Phase 4 JSON homography data
        self.load_json_homography_data()
        
        # Load NPZ homography data (legacy format)
        self.load_npz_homography_data()
        
        logger.info("HomographyService initialized with Phase 4 enhancements.")

    async def preload_all_homography_matrices(self):
        """
        Loads all homography point files specified in settings,
        computes the matrices, and caches them.
        This method is intended to be called once at application startup.
        """
        if self._preloaded:
            logger.info("Homography matrices already preloaded.")
            return

        logger.info("Preloading all homography matrices...")
        base_homography_path = self._settings.resolved_homography_base_path
        
        # Create tasks for each homography calculation
        # Note: cv2.findHomography is CPU-bound, so asyncio.to_thread is appropriate
        # But loading files and preparing data can be concurrent.
        
        load_tasks = []
        for (env_id, cam_id_str), cam_config in self._settings.CAMERA_HANDOFF_DETAILS.items():
            cam_id = CameraID(cam_id_str)
            if cam_config.homography_matrix_path:
                file_path = base_homography_path / cam_config.homography_matrix_path
                # Prepare a coroutine for each computation
                load_tasks.append(self._compute_and_cache_matrix(env_id, cam_id, file_path))
            else:
                # Cache as None if no path is configured
                self._homography_matrices[(env_id, cam_id)] = None
                logger.debug(f"No homography path for env '{env_id}', cam '{cam_id}'. Cached as None.")

        # Execute all computations concurrently
        results = await asyncio.gather(*load_tasks, return_exceptions=True)

        num_successful = 0
        for i, result in enumerate(results):
            # Extract env_id, cam_id from the original list based on task order, if needed for logging errors
            # For simplicity, errors are logged within _compute_and_cache_matrix
            if isinstance(result, np.ndarray): # Assuming _compute_and_cache_matrix returns matrix on success
                num_successful += 1
            elif result is None and not isinstance(result, Exception): # Successfully processed as "no matrix"
                pass # Already handled by _compute_and_cache_matrix logging or was intentional None
            elif isinstance(result, Exception):
                logger.error(f"Error during homography preloading task {i}: {result}")


        self._preloaded = True
        logger.info(f"Homography matrices preloading complete. {num_successful} matrices computed and cached.")


    async def _compute_and_cache_matrix(
        self, env_id: str, cam_id: CameraID, file_path: Path
    ) -> Optional[np.ndarray]:
        """
        Computes a single homography matrix from a file and caches it.
        Helper for preload_all_homography_matrices.
        """
        matrix: Optional[np.ndarray] = None
        if not file_path.is_file():
            logger.warning(f"Homography file not found for env '{env_id}', cam '{cam_id}': {file_path}")
            self._homography_matrices[(env_id, cam_id)] = None
            return None
        
        try:
            # np.load is blocking, run in thread
            data = await asyncio.to_thread(np.load, str(file_path))
            image_points = data.get('image_points')
            map_points = data.get('map_points')

            if image_points is not None and map_points is not None and \
               len(image_points) >= 4 and len(map_points) >= 4 and \
               len(image_points) == len(map_points):
                
                # cv2.findHomography is CPU-bound
                h_matrix, _ = await asyncio.to_thread(
                    cv2.findHomography, image_points, map_points, cv2.RANSAC, 5.0
                )
                if h_matrix is not None:
                    matrix = h_matrix
                    logger.info(f"Successfully computed homography for env '{env_id}', cam '{cam_id}'.")
                else:
                    logger.warning(f"Homography calculation failed for env '{env_id}', cam '{cam_id}'.")
            else:
                logger.warning(f"Insufficient/mismatched points in homography file for env '{env_id}', cam '{cam_id}': {file_path}")
        except Exception as e:
            logger.error(f"Error computing homography for env '{env_id}', cam '{cam_id}': {e}", exc_info=True)
        
        self._homography_matrices[(env_id, cam_id)] = matrix
        return matrix


    def get_homography_matrix(self, environment_id: str, camera_id: CameraID) -> Optional[np.ndarray]:
        """
        Retrieves a pre-computed homography matrix.

        Args:
            environment_id: The environment ID.
            camera_id: The camera ID.

        Returns:
            The pre-computed NumPy array for the homography matrix, or None if not available.
        
        Raises:
            RuntimeError: If matrices have not been preloaded.
        """
        if not self._preloaded:
            # This case should ideally not be hit if startup logic is correct.
            # Fallback to on-demand loading could be an option here, but goal is preloading.
            logger.error("Attempted to get homography matrix before preloading. This indicates a startup logic issue.")
            # For robustness, you could try to load it on-demand here, but it defeats the purpose.
            # For now, raise error or return None.
            # await self.preload_all_homography_matrices() # NOT ideal to call here, should be done at startup.
            raise RuntimeError("HomographyService: Matrices not preloaded. Call preload_all_homography_matrices() at startup.")

        return self._homography_matrices.get((environment_id, camera_id))
    
    # Phase 4: Spatial Intelligence Methods
    
    def load_json_homography_data(self):
        """
        Load homography matrices and calibration points from JSON files (Phase 4).
        
        Expected file format: {camera_id}_homography.json
        JSON structure:
        {
            "matrix": [[...], [...], [...]],  # 3x3 homography matrix (optional)
            "calibration_points": {
                "image_points": [[x1, y1], [x2, y2], ...],
                "map_points": [[mx1, my1], [mx2, my2], ...]
            }
        }
        """
        if not self.homography_dir.exists():
            logger.warning(f"Phase 4 homography directory {self.homography_dir} does not exist")
            return
        
        loaded_count = 0
        for camera_file in self.homography_dir.glob("*_homography.json"):
            camera_id = camera_file.stem.replace("_homography", "")
            
            try:
                with open(camera_file, 'r') as f:
                    data = json.load(f)
                
                # Load pre-computed matrix if available
                if 'matrix' in data and data['matrix']:
                    matrix_data = np.array(data['matrix'], dtype=np.float64)
                    if matrix_data.shape == (3, 3):
                        self.json_homography_matrices[camera_id] = matrix_data
                        logger.debug(f"Loaded JSON homography matrix for camera {camera_id}")
                    else:
                        logger.warning(f"Invalid matrix shape for camera {camera_id}: {matrix_data.shape}")
                
                # Load calibration points for on-demand computation
                if 'calibration_points' in data and data['calibration_points']:
                    calib_data = data['calibration_points']
                    if ('image_points' in calib_data and 'map_points' in calib_data and
                        len(calib_data['image_points']) >= 4 and 
                        len(calib_data['map_points']) >= 4):
                        self.calibration_points[camera_id] = calib_data
                        logger.debug(f"Loaded calibration points for camera {camera_id}")
                    else:
                        logger.warning(f"Invalid calibration points for camera {camera_id}")
                
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Error loading JSON homography data for {camera_id}: {e}")
        
        logger.info(f"Loaded Phase 4 JSON homography data for {loaded_count} cameras")
    
    def load_npz_homography_data(self):
        """
        Load homography calibration points from NPZ files (legacy format).
        
        Expected file format: homography_points_{camera_id}_scene_{scene_id}.npz
        NPZ structure:
        - 'image_points': array of image coordinates
        - 'map_points': array of map coordinates
        """
        if not self.homography_dir.exists():
            logger.warning(f"Homography directory {self.homography_dir} does not exist")
            return
        
        loaded_count = 0
        for npz_file in self.homography_dir.glob("homography_points_*.npz"):
            # Extract camera_id from filename like "homography_points_c02_scene_s47.npz"
            filename_parts = npz_file.stem.split('_')
            if len(filename_parts) >= 3:
                camera_id = filename_parts[2]  # c02, c09, etc.
                
                try:
                    # Load NPZ file
                    npz_data = np.load(npz_file)
                    
                    if 'image_points' in npz_data and 'map_points' in npz_data:
                        image_points = npz_data['image_points']
                        map_points = npz_data['map_points']
                        
                        # Validate minimum number of points
                        if len(image_points) >= 4 and len(map_points) >= 4:
                            # Convert to list format expected by calibration_points
                            calib_data = {
                                'image_points': image_points.tolist(),
                                'map_points': map_points.tolist()
                            }
                            
                            # Only add if not already loaded from JSON
                            if camera_id not in self.calibration_points:
                                self.calibration_points[camera_id] = calib_data
                                logger.debug(f"Loaded NPZ calibration points for camera {camera_id}")
                                loaded_count += 1
                            else:
                                logger.debug(f"Skipping NPZ data for camera {camera_id} - JSON data already loaded")
                        else:
                            logger.warning(f"Insufficient calibration points in {npz_file}: image={len(image_points)}, map={len(map_points)}")
                    else:
                        logger.warning(f"Missing required arrays in {npz_file}")
                        
                except Exception as e:
                    logger.error(f"Error loading NPZ homography data from {npz_file}: {e}")
        
        logger.info(f"Loaded {loaded_count} NPZ homography configurations")
    
    def compute_homography(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Compute homography matrix from calibration points using RANSAC (Phase 4).
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            3x3 homography matrix or None if computation fails
        """
        if camera_id not in self.calibration_points:
            # Only warn once per camera to avoid log spam
            if camera_id not in self._warned_cameras:
                logger.warning(f"No calibration points available for camera {camera_id}")
                self._warned_cameras.add(camera_id)
            return None
        
        calib_data = self.calibration_points[camera_id]
        
        try:
            # Convert points to numpy arrays
            image_points = np.array(calib_data['image_points'], dtype=np.float32)
            map_points = np.array(calib_data['map_points'], dtype=np.float32)
            
            # Validate minimum number of points
            if len(image_points) < 4 or len(map_points) < 4:
                logger.error(f"Insufficient calibration points for camera {camera_id}: {len(image_points)}")
                return None
            
            if len(image_points) != len(map_points):
                logger.error(f"Mismatched calibration points for camera {camera_id}: {len(image_points)} vs {len(map_points)}")
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
                self.json_homography_matrices[camera_id] = matrix
                
                # Log computation details
                inliers = np.sum(mask) if mask is not None else len(image_points)
                logger.info(f"Computed homography for camera {camera_id}: {inliers}/{len(image_points)} inliers")
                
                return matrix
            else:
                logger.error(f"Failed to compute valid homography matrix for camera {camera_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error computing homography for camera {camera_id}: {e}")
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
        # Get or compute homography matrix from JSON data
        if camera_id not in self.json_homography_matrices:
            logger.debug(f"No cached JSON matrix for camera {camera_id}, attempting to compute")
            matrix = self.compute_homography(camera_id)
            if matrix is None:
                return None
        else:
            matrix = self.json_homography_matrices[camera_id]
        
        try:
            # Prepare point for perspective transformation
            # OpenCV expects shape (1, 1, 2) for single point transformation
            point = np.array([[image_point[0], image_point[1]]], dtype=np.float32)
            point = point.reshape(-1, 1, 2)
            
            # Apply homography transformation
            transformed = cv2.perspectiveTransform(point, matrix)
            
            # Extract transformed coordinates
            map_x, map_y = transformed[0, 0]
            
            # Validate transformed coordinates
            if np.isfinite(map_x) and np.isfinite(map_y):
                return float(map_x), float(map_y)
            else:
                logger.warning(f"Invalid projection result for camera {camera_id}: ({map_x}, {map_y})")
                return None
                
        except Exception as e:
            logger.error(f"Error projecting point for camera {camera_id}: {e}")
            return None
    
    def get_homography_data(self, camera_id: str) -> Dict:
        """
        Get comprehensive homography data for WebSocket response (Phase 4).
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary containing matrix availability, matrix data, and calibration points
        """
        data = {
            "matrix_available": camera_id in self.json_homography_matrices,
            "matrix": None,
            "calibration_points": None
        }
        
        # Include matrix data if available
        if camera_id in self.json_homography_matrices:
            data["matrix"] = self.json_homography_matrices[camera_id].tolist()
        
        # Include calibration points if available
        if camera_id in self.calibration_points:
            data["calibration_points"] = self.calibration_points[camera_id]
        
        return data