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

from .base_models import AbstractTracker
from app.core.config import settings
# CORRECTED: Import from existing device_utils
from app.utils.device_utils import get_boxmot_device_string

logger = logging.getLogger(__name__)

class BotSortTracker(AbstractTracker):
    """
    Implementation of the BotSort tracker using the BoxMOT library.
    """
    def __init__(self):
        if not BOXMOT_AVAILABLE:
            raise ImportError("BoxMOT library is required for BotSortTracker but not available.")

        self.tracker_instance: Optional[BoxMOTBaseTracker] = None
        self.device: Optional[torch.device] = None 
        self.reid_model_path: Path = settings.resolved_reid_weights_path
        self.use_half: bool = settings.TRACKER_HALF_PRECISION
        self.per_class: bool = settings.TRACKER_PER_CLASS

        logger.info(f"BotSortTracker configured. ReID Weights: {self.reid_model_path}, Half: {self.use_half}, PerClass: {self.per_class}")

    async def load_model(self):
        if self.tracker_instance is not None:
            logger.info("BotSort tracker already loaded.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BotSort tracker on device: {self.device}...")

        if create_tracker is None:
             raise RuntimeError("BoxMOT create_tracker function not available.")

        # CORRECTED: Use get_boxmot_device_string
        boxmot_device_str = get_boxmot_device_string(self.device)
        effective_half = self.use_half and self.device.type == 'cuda'

        try:
            self.tracker_instance = await asyncio.to_thread(
                create_tracker,
                tracker_type='botsort',
                weights=self.reid_model_path, 
                device=boxmot_device_str,
                half=effective_half,
                per_class=self.per_class
            )
            logger.info(f"BotSort tracker instance created with ReID model '{self.reid_model_path}'.")
            # Warmup
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            dummy_dets = np.empty((0, 6))
            await self.update(dummy_dets, dummy_frame)
            logger.info("BotSort tracker warmup update complete.")
        except Exception as e:
            logger.exception(f"Error loading BotSort tracker: {e}")
            self.tracker_instance = None
            raise

    async def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        if self.tracker_instance is None:
            raise RuntimeError("BotSort tracker not loaded. Call load_model() first.")

        detections_for_update = detections
        if detections.ndim != 2 or detections.shape[1] != 6:
            if detections.size == 0:
                 detections_for_update = np.empty((0, 6))
            else:
                logger.error(f"Invalid detections shape: {detections.shape}. Expected (N, 6).")
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
                 logger.warning(f"Tracker output invalid shape: {tracked_output_np.shape}")
                 return np.empty((0, 8))
            return tracked_output_np
        except Exception as e:
            logger.exception(f"Error during BotSort tracker update: {e}")
            return np.empty((0, 8))

    async def reset(self):
        if self.tracker_instance and hasattr(self.tracker_instance, 'reset'):
            try:
                await asyncio.to_thread(self.tracker_instance.reset)
                logger.info("BotSort tracker state reset.")
            except Exception as e:
                logger.error(f"Error resetting BotSort tracker: {e}")
        else:
            logger.warning("BotSort tracker instance not available or does not support reset().")