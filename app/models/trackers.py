import logging
from typing import Optional, List, Dict
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

        self.tracker_instance = None 
        self.device: Optional[torch.device] = None
        self.reid_model_path: Path = settings.resolved_reid_weights_path
        self.use_half: bool = settings.TRACKER_HALF_PRECISION
        self.per_class: bool = settings.TRACKER_PER_CLASS

        logger.info(f"BotSortTracker configured. ReID Weights: {self.reid_model_path}, Half: {self.use_half}, PerClass: {self.per_class}")

    async def load_model(self):
        """
        Loads and initializes the BotSort tracker, including its ReID model.
        """
        if self.tracker_instance is not None:
            logger.info("BotSort tracker already loaded.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"BotSortTracker: Attempting to load BotSort tracker on device: {self.device}...")

        boxmot_device_str = get_boxmot_device_string(self.device)
        effective_half = self.use_half and self.device.type == 'cuda'

        try:
            logger.info(f"BotSortTracker: Instantiating BotSort with reid_weights='{self.reid_model_path}', device='{boxmot_device_str}', half={effective_half}, per_class={self.per_class}")
            self.tracker_instance = await asyncio.to_thread(
                BotSort,
                reid_weights=self.reid_model_path, 
                device=boxmot_device_str,
                half=effective_half,
                per_class=self.per_class
            )
            logger.info(f"BotSortTracker: BotSort tracker instance CREATED successfully. Type: {type(self.tracker_instance)}")
            
            logger.info("BotSortTracker: Attempting warmup...")
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            dummy_dets = np.empty((0, 6)) 
            await self.update(dummy_dets, dummy_frame) # This already calls tracker_instance.update
            logger.info("BotSortTracker: BotSort tracker warmup update complete.")
        except TypeError as te:
            logger.exception(f"BotSortTracker: TypeError loading BotSort tracker. Check constructor arguments: {te}")
            self.tracker_instance = None
            raise
        except Exception as e:
            logger.exception(f"BotSortTracker: CRITICAL ERROR loading BotSort tracker: {e}") # Log full exception
            self.tracker_instance = None
            raise

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
        if self.tracker_instance is None:
            logger.error("BotSortTracker: Tracker not loaded. Call load_model() first.")
            raise RuntimeError("BotSort tracker not loaded. Call load_model() first.")

        detections_for_update = detections
        if detections.ndim != 2 or detections.shape[1] != 6:
            if detections.size == 0: 
                 detections_for_update = np.empty((0, 6))
            else:
                logger.error(f"BotSortTracker: Invalid detections shape: {detections.shape}. Expected (N, 6) for tracker update.")
                return np.empty((0, 8)) 

        if image is None or image.size == 0:
            logger.error("BotSortTracker: Invalid image provided to tracker update.")
            return np.empty((0, 8))

        try:
            # logger.debug(f"BotSortTracker: Calling tracker_instance.update with dets shape {detections_for_update.shape}, image shape {image.shape}")
            tracked_output_np = await asyncio.to_thread(
                self.tracker_instance.update,
                detections_for_update, 
                image
            )
            # logger.debug(f"BotSortTracker: tracker_instance.update returned output shape: {tracked_output_np.shape if tracked_output_np is not None else 'None'}")


            if tracked_output_np is None or not isinstance(tracked_output_np, np.ndarray):
                return np.empty((0, 8))
            if tracked_output_np.size == 0:
                return np.empty((0, 8))
            if tracked_output_np.ndim != 2 or tracked_output_np.shape[1] < 5:
                 logger.warning(f"BotSortTracker: Tracker output has an unexpected shape: {tracked_output_np.shape}. Expected at least 5 columns.")
                 return np.empty((0, 8)) 

            return tracked_output_np
        except Exception as e:
            logger.exception(f"BotSortTracker: Error during BotSort tracker update: {e}")
            return np.empty((0, 8)) 

    async def reset(self):
        """Resets the tracker's state (e.g., for a new video sequence)."""
        if self.tracker_instance and hasattr(self.tracker_instance, 'reset'):
            try:
                logger.info("BotSortTracker: Calling tracker_instance.reset()")
                await asyncio.to_thread(self.tracker_instance.reset)
                logger.info("BotSortTracker: BotSort tracker state reset.")
            except Exception as e:
                logger.error(f"BotSortTracker: Error resetting BotSort tracker: {e}")
        elif self.tracker_instance:
            logger.warning("BotSortTracker: Tracker instance does not have a 'reset' method. Re-initializing for reset.")
            self.tracker_instance = None 
            await self.load_model() 
        else:
            logger.warning("BotSortTracker: Tracker instance not available to reset.")