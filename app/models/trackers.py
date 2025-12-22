import logging
from typing import Optional, List, Dict, Tuple # Added Tuple
import numpy as np
import torch
import asyncio
 

try:
    # Import specific tracker class directly (ByteTrack)
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
    from boxmot.trackers.basetracker import BaseTracker as BoxMOTBaseTracker
    BOXMOT_AVAILABLE = True
except ImportError as e:
    logging.critical(f"Failed to import BoxMOT components. Tracking unavailable. Error: {e}")
    BOXMOT_AVAILABLE = False
    ByteTrack = None  # Define as None if import fails
    BoxMOTBaseTracker = type(None)

from .base_models import AbstractTracker
from app.core.config import settings

logger = logging.getLogger(__name__)

"""
ByteTrack tracker implementation (BoxMOT).
"""


class ByteTrackTracker(AbstractTracker):
    """
    Implementation of the ByteTrack tracker using the BoxMOT library.
    """
    def __init__(self):
        if not BOXMOT_AVAILABLE or ByteTrack is None:  # Check if ByteTrack itself was imported
            raise ImportError("BoxMOT library or ByteTrack class is required but not available.")

        self.tracker_instance: Optional[ByteTrack] = None  # Type hint with ByteTrack
        self.device: Optional[torch.device] = None
        # ByteTrack is motion-only; no identity model weights. Keep per-class behavior from settings.
        self.per_class: bool = settings.TRACKER_PER_CLASS
        # Optional: use app target FPS for ByteTrack frame_rate parameter
        self.frame_rate: int = getattr(settings, "TARGET_FPS", 30) or 30
        # Other ByteTrack params use library defaults (min_conf, track_thresh, match_thresh, track_buffer)
        self._model_loaded_flag: bool = False

        self._model_loaded_flag: bool = False

        # logger.info(
        #     f"ByteTrackTracker configured. PerClass: {self.per_class}, FrameRate: {self.frame_rate}"
        # )

    async def load_model(self):
        """
        Loads and initializes the ByteTrack tracker.
        """
        if self._model_loaded_flag and self.tracker_instance is not None:
            # logger.info("ByteTrack tracker already loaded.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logger.info(f"Loading ByteTrack tracker on device: {self.device}...")

        try:
            # Instantiate with per_class and frame_rate; others use defaults
            self.tracker_instance = await asyncio.to_thread(
                ByteTrack,
                per_class=self.per_class,
                frame_rate=int(self.frame_rate),
            )
            self._model_loaded_flag = True
            # logger.info("ByteTrack tracker instance created.")
        except TypeError as te:
            logger.exception(f"TypeError loading ByteTrack tracker. Check constructor arguments: {te}")
            self.tracker_instance = None
            self._model_loaded_flag = False
            raise
        except Exception as e:
            logger.exception(f"Error loading ByteTrack tracker: {e}")
            self.tracker_instance = None
            self._model_loaded_flag = False
            raise

    async def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """
        Warms up the tracker by performing a dummy update.
        Should be called after load_model().
        """
        if not self._model_loaded_flag or not self.tracker_instance:
            logger.warning("ByteTrack tracker not loaded. Cannot perform warmup.")
            return

        logger.info(f"Warming up ByteTrackTracker on device {self.device}...")
        try:
            dummy_frame = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            dummy_dets = np.empty((0, 6), dtype=np.float32)  # [x1, y1, x2, y2, conf, cls]
            _ = await self.update(dummy_dets, dummy_frame)
            logger.info("ByteTrackTracker warmup successful.")
        except Exception as e:
            logger.error(f"ByteTrackTracker warmup failed: {e}", exc_info=True)

    async def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Updates tracks with new detections for the current frame.

        Args:
            detections: A NumPy array of detections in [x1, y1, x2, y2, conf, cls_id] format.
            image: The current frame as a NumPy array (BGR).

        Returns:
            A NumPy array representing tracked objects, typically in a format like
            [x1, y1, x2, y2, track_id, conf, cls_id, det_ind].
        """
        if not self._model_loaded_flag or self.tracker_instance is None:
            raise RuntimeError("ByteTrack tracker not loaded. Call load_model() first.")

        detections_for_update = detections
        if detections.ndim != 2 or detections.shape[1] != 6:
            if detections.size == 0:
                detections_for_update = np.empty((0, 6))
            else:
                logger.error(
                    f"Invalid detections shape: {detections.shape}. Expected (N, 6) for tracker update."
                )
                return np.empty((0, 8))

        if image is None or image.size == 0:
            logger.error("Invalid image provided to tracker update.")
            return np.empty((0, 8))

        try:
            tracked_output_np = await asyncio.to_thread(
                self.tracker_instance.update,
                detections_for_update,
                image,
            )

            if tracked_output_np is None or not isinstance(tracked_output_np, np.ndarray):
                return np.empty((0, 8))
            if tracked_output_np.size == 0:
                return np.empty((0, 8))
            if tracked_output_np.ndim != 2 or tracked_output_np.shape[1] < 5:
                logger.warning(
                    f"Tracker output has an unexpected shape: {tracked_output_np.shape}. Expected at least 5 columns."
                )
                return np.empty((0, 8))

            return tracked_output_np
        except Exception as e:
            logger.exception(f"Error during ByteTrack tracker update: {e}")
            return np.empty((0, 8))

    async def reset(self):
        """Resets the tracker's state (e.g., for a new video sequence)."""
        if self.tracker_instance and hasattr(self.tracker_instance, "reset"):
            try:
                await asyncio.to_thread(self.tracker_instance.reset)
                logger.info("ByteTrack tracker state reset.")
            except Exception as e:
                logger.error(f"Error resetting ByteTrack tracker: {e}")
        elif self.tracker_instance:
            logger.warning(
                "ByteTrack tracker instance does not have a 'reset' method. Re-initializing for reset."
            )
            self._model_loaded_flag = False
            self.tracker_instance = None
            await self.load_model()
        else:
            logger.warning("ByteTrack tracker instance not available to reset.")
