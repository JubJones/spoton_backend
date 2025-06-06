import logging
from typing import Optional, List, Dict, Tuple # Added Tuple
import numpy as np
import torch
import asyncio
from pathlib import Path

try:
    # Import the specific tracker class directly
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.basetracker import BaseTracker as BoxMOTBaseTracker
    BOXMOT_AVAILABLE = True
except ImportError as e:
    logging.critical(f"Failed to import BoxMOT components. Tracking unavailable. Error: {e}")
    BOXMOT_AVAILABLE = False
    BotSort = None # Define as None if import fails
    BoxMOTBaseTracker = type(None)

from .base_models import AbstractTracker
from app.core.config import settings
from app.utils.device_utils import get_boxmot_device_string

logger = logging.getLogger(__name__)

class BotSortTracker(AbstractTracker):
    """
    Implementation of the BotSort tracker using the BoxMOT library.
    """
    def __init__(self):
        if not BOXMOT_AVAILABLE or BotSort is None: # Check if BotSort itself was imported
            raise ImportError("BoxMOT library or BotSort class is required but not available.")

        self.tracker_instance: Optional[BotSort] = None # Type hint with BotSort
        self.device: Optional[torch.device] = None
        self.reid_model_path: Path = settings.resolved_reid_weights_path
        self.use_half: bool = settings.TRACKER_HALF_PRECISION
        self.per_class: bool = settings.TRACKER_PER_CLASS
        self._model_loaded_flag: bool = False # Added flag

        logger.info(f"BotSortTracker configured. ReID Weights: {self.reid_model_path}, Half: {self.use_half}, PerClass: {self.per_class}")

    async def load_model(self):
        """
        Loads and initializes the BotSort tracker, including its ReID model.
        """
        if self._model_loaded_flag and self.tracker_instance is not None: # Check flag
            logger.info("BotSort tracker already loaded.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BotSort tracker on device: {self.device}...")

        boxmot_device_str = get_boxmot_device_string(self.device)
        effective_half = self.use_half and self.device.type == 'cuda'

        try:
            self.tracker_instance = await asyncio.to_thread(
                BotSort,
                reid_weights=self.reid_model_path,
                device=boxmot_device_str,
                half=effective_half,
                per_class=self.per_class
            )
            self._model_loaded_flag = True # Set flag
            logger.info(f"BotSort tracker instance created with ReID model '{self.reid_model_path}'.")
        except TypeError as te:
            logger.exception(f"TypeError loading BotSort tracker. Check constructor arguments: {te}")
            self.tracker_instance = None
            self._model_loaded_flag = False
            raise
        except Exception as e:
            logger.exception(f"Error loading BotSort tracker: {e}")
            self.tracker_instance = None
            self._model_loaded_flag = False
            raise

    async def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """
        Warms up the tracker by performing a dummy update.
        Should be called after load_model().
        """
        if not self._model_loaded_flag or not self.tracker_instance:
            logger.warning("BotSort tracker not loaded. Cannot perform warmup.")
            return
        
        logger.info(f"Warming up BotSortTracker on device {self.device}...")
        try:
            dummy_frame = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            dummy_dets = np.empty((0, 6), dtype=np.float32) # [x1, y1, x2, y2, conf, cls]
            # The update method expects detections and the image
            _ = await self.update(dummy_dets, dummy_frame) # Use existing update for warmup
            logger.info("BotSortTracker warmup successful.")
        except Exception as e:
            logger.error(f"BotSortTracker warmup failed: {e}", exc_info=True)


    async def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Updates tracks with new detections for the current frame.

        Args:
            detections: A NumPy array of detections in [x1, y1, x2, y2, conf, cls_id] format.
            image: The current frame as a NumPy array (BGR).

        Returns:
            A NumPy array representing tracked objects, typically in a format like
            [x1, y1, x2, y2, track_id, conf, cls_id, global_id (optional), ...].
        """
        if not self._model_loaded_flag or self.tracker_instance is None: # Check flag
            raise RuntimeError("BotSort tracker not loaded. Call load_model() first.")

        detections_for_update = detections
        if detections.ndim != 2 or detections.shape[1] != 6:
            if detections.size == 0:
                 detections_for_update = np.empty((0, 6))
            else:
                logger.error(f"Invalid detections shape: {detections.shape}. Expected (N, 6) for tracker update.")
                return np.empty((0, 8))

        if image is None or image.size == 0:
            logger.error("Invalid image provided to tracker update.")
            return np.empty((0, 8))

        try:
            tracked_output_np = await asyncio.to_thread(
                self.tracker_instance.update,
                detections_for_update,
                image
            )

            if tracked_output_np is None or not isinstance(tracked_output_np, np.ndarray):
                return np.empty((0, 8))
            if tracked_output_np.size == 0:
                return np.empty((0, 8))
            if tracked_output_np.ndim != 2 or tracked_output_np.shape[1] < 5:
                 logger.warning(f"Tracker output has an unexpected shape: {tracked_output_np.shape}. Expected at least 5 columns.")
                 return np.empty((0, 8))

            return tracked_output_np
        except Exception as e:
            logger.exception(f"Error during BotSort tracker update: {e}")
            return np.empty((0, 8))

    async def reset(self):
        """Resets the tracker's state (e.g., for a new video sequence)."""
        if self.tracker_instance and hasattr(self.tracker_instance, 'reset'):
            try:
                await asyncio.to_thread(self.tracker_instance.reset)
                logger.info("BotSort tracker state reset.")
            except Exception as e:
                logger.error(f"Error resetting BotSort tracker: {e}")
        elif self.tracker_instance:
            logger.warning("BotSort tracker instance does not have a 'reset' method. Re-initializing for reset.")
            self._model_loaded_flag = False # Reset flag
            self.tracker_instance = None
            await self.load_model() # Reload to get a fresh state
            # Consider if warmup is needed again after reset/reload.
        else:
            logger.warning("BotSort tracker instance not available to reset.")