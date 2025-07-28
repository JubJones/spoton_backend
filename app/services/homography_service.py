"""
Service for managing and providing pre-computed homography matrices.
"""
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import asyncio

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

    def __init__(self, settings: Settings):
        """
        Initializes the HomographyService.

        Args:
            settings: The application settings instance.
        """
        self._settings = settings
        self._homography_matrices: Dict[Tuple[str, CameraID], Optional[np.ndarray]] = {}
        self._preloaded = False
        logger.info("HomographyService initialized.")

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