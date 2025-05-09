import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import asyncio
from pathlib import Path

try:      
    from boxmot import create_tracker
    from boxmot.trackers.basetracker import BaseTracker as BoxMOTBaseTracker
    BOXMOT_AVAILABLE = True
except ImportError as e:
    logging.critical(f"Failed to import BoxMOT components. Tracking unavailable. Error: {e}")
    BOXMOT_AVAILABLE = False
    BoxMOTBaseTracker = type(None)
    create_tracker = None

from .base_models import AbstractTracker, BoundingBox, TrackedObject, Detection
from app.core.config import settings
from app.utils.reid_device_utils import get_reid_device_specifier_string

logger = logging.getLogger(__name__)

class BotSortTracker(AbstractTracker):
    """
    Implementation of the BotSort tracker using the BoxMOT library.
    Integrates Re-ID using configured weights (e.g., CLIP).
    """
    def __init__(self):
        if not BOXMOT_AVAILABLE:
            raise ImportError("BoxMOT library is required for BotSortTracker but not available.")

        self.tracker_instance: Optional[BoxMOTBaseTracker] = None
        self.device: Optional[torch.device] = None # Set during load_model
        self.reid_model_path: Optional[Path] = settings.resolved_reid_weights_path
        self.use_half: bool = settings.TRACKER_HALF_PRECISION
        self.per_class: bool = settings.TRACKER_PER_CLASS

        logger.info(f"BotSortTracker configured. ReID Weights: {self.reid_model_path}, Half: {self.use_half}, PerClass: {self.per_class}")

    async def load_model(self):
        """Loads the BotSort tracker instance with integrated Re-ID."""
        if self.tracker_instance is not None:
            logger.info("BotSort tracker already loaded.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BotSort tracker on device: {self.device}...")

        if create_tracker is None:
             raise RuntimeError("BoxMOT create_tracker function not available.")

        reid_device_specifier = get_reid_device_specifier_string(self.device)
        effective_half = self.use_half and self.device.type == 'cuda' # Half only works on CUDA

        try:
            # Use asyncio.to_thread for potentially blocking BoxMOT initialization
            self.tracker_instance = await asyncio.to_thread(
                create_tracker,
                tracker_type='botsort', # Explicitly botsort
                # Pass the path object or string identifier for weights
                weights=self.reid_model_path, # 'weights' is the arg name used by create_tracker
                device=reid_device_specifier,
                half=effective_half,
                per_class=self.per_class
            )
            logger.info(f"BotSort tracker instance created with ReID model '{self.reid_model_path}'.")
            # Perform a dummy update for potential warmup (optional)
            logger.info("Performing dummy tracker update for warmup...")
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            dummy_dets = np.empty((0, 6))
            await self.update(dummy_dets, dummy_frame) # Call async update
            logger.info("BotSort tracker warmup update complete.")

        except Exception as e:
            logger.exception(f"Error loading BotSort tracker: {e}")
            self.tracker_instance = None
            raise # Re-raise the exception

    async def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Updates the BotSort tracker with new detections.

        Args:
            detections: A NumPy array of detections in [x1, y1, x2, y2, conf, cls_id] format.
            image: The current frame as a NumPy array (BGR).

        Returns:
            A NumPy array representing tracked objects. The exact columns depend on
            BoxMOT's BotSort output format when ReID is enabled.
            Expected format might be: [x1, y1, x2, y2, track_id, conf, cls_id, global_id, ...]
        """
        if self.tracker_instance is None:
            raise RuntimeError("BotSort tracker not loaded. Call load_model() first.")

        # Input validation
        if detections.ndim != 2 or detections.shape[1] != 6:
            if detections.size == 0: # Handle empty detections case gracefully
                 logger.debug("No detections provided to tracker update.")
                 detections_for_update = np.empty((0, 6))
            else:
                logger.error(f"Invalid detections shape for tracker update: {detections.shape}. Expected (N, 6).")
                # Return empty array to signify no tracks could be updated/returned
                return np.empty((0, 8)) # Assume 8 columns for potential output format
        else:
            detections_for_update = detections

        if image is None or image.size == 0:
            logger.error("Invalid image provided to tracker update.")
            return np.empty((0, 8))

        try:
            # BoxMOT's update is likely CPU/GPU bound, run in thread pool
            tracked_output_np = await asyncio.to_thread(
                self.tracker_instance.update,
                detections_for_update,
                image
            )

            # Basic validation of the output format
            if tracked_output_np is None or not isinstance(tracked_output_np, np.ndarray):
                logger.warning("Tracker update returned None or non-numpy array. Returning empty.")
                return np.empty((0, 8))
            if tracked_output_np.size == 0:
                # logger.debug("Tracker update returned no tracks.")
                return np.empty((0, 8)) # Return empty with expected shape if no tracks
            if tracked_output_np.ndim != 2:
                 logger.warning(f"Tracker update returned array with invalid dimensions: {tracked_output_np.ndim}. Expected 2.")
                 return np.empty((0, 8))
            # Check for minimum expected columns (x1, y1, x2, y2, track_id)
            if tracked_output_np.shape[1] < 5:
                 logger.warning(f"Tracker update returned array with too few columns: {tracked_output_np.shape[1]}. Expected >= 5.")
                 return np.empty((0, 8))

            return tracked_output_np

        except Exception as e:
            logger.exception(f"Error during BotSort tracker update: {e}")
            return np.empty((0, 8)) # Return empty array on error


    async def reset(self):
        """Resets the tracker's internal state."""
        if self.tracker_instance and hasattr(self.tracker_instance, 'reset'):
            try:
                await asyncio.to_thread(self.tracker_instance.reset)
                logger.info("BotSort tracker state reset.")
            except Exception as e:
                logger.error(f"Error resetting BotSort tracker: {e}")
        else:
            logger.warning("BotSort tracker instance not available or does not support reset().")