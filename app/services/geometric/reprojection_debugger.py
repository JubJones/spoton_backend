"""
Persist reprojection overlays for debugging geometric predictions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .debug_overlay import DebugOverlay
from .inverse_projector import ProjectedImagePoint

logger = logging.getLogger(__name__)


FrameProvider = Callable[[str, int], Optional[np.ndarray]]


class ReprojectionDebugger:
    """
    Persist debug overlays that show predicted vs actual positions.

    A pluggable frame_provider fetches raw frames (e.g., from an RTSP buffer
    or recorded video). Sampling parameters limit disk usage.
    """

    def __init__(
        self,
        frame_provider: FrameProvider,
        output_dir: str,
        sampling_rate: int = 1,
        max_frames_per_camera: int = 500,
    ) -> None:
        self.frame_provider = frame_provider
        self.output_dir = Path(output_dir)
        self.sampling_rate = max(1, sampling_rate)
        self.max_frames_per_camera = max(1, max_frames_per_camera)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_counts = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "ReprojectionDebugger writing overlays to %s (sampling=%d, max_per_camera=%d)",
            self.output_dir,
            self.sampling_rate,
            self.max_frames_per_camera,
        )

    def emit(
        self,
        camera_id: str,
        frame_number: int,
        overlay: DebugOverlay,
        predicted_point: ProjectedImagePoint,
        actual_point: Optional[Tuple[float, float]],
    ) -> None:
        """Render and persist an overlay for a given camera frame."""
        if frame_number % self.sampling_rate != 0:
            return

        count = self.frame_counts.get(camera_id, 0)

        camera_dir = self.output_dir / camera_id
        camera_dir.mkdir(exist_ok=True)
        output_path = camera_dir / f"{frame_number:06d}.png"
        is_existing_output = output_path.exists()

        if count >= self.max_frames_per_camera and not is_existing_output:
            return

        frame: Optional[np.ndarray] = None
        if is_existing_output:
            frame = cv2.imread(str(output_path))

        if frame is None:
            frame = self.frame_provider(camera_id, frame_number)
            if frame is None:
                logger.debug(
                    "ReprojectionDebugger: no frame available for camera %s (frame %d)",
                    camera_id,
                    frame_number,
                )
                return

        overlay.draw_prediction(frame=frame, predicted_point=predicted_point, actual_point=actual_point)

        if not cv2.imwrite(str(output_path), frame):
            logger.debug("Failed to write reprojection debug frame to %s", output_path)
            return

        if not is_existing_output:
            self.frame_counts[camera_id] = count + 1
