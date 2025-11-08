"""
Utility for drawing reprojection debug overlays on frames.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2

from .inverse_projector import ProjectedImagePoint


class DebugOverlay:
    """Draw predicted vs actual positions on frames for visual inspection."""

    def __init__(
        self,
        radius_px: int = 6,
        color_pred: Tuple[int, int, int] = (0, 255, 255),
        color_actual: Tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        self.radius_px = max(1, radius_px)
        self.color_pred = color_pred
        self.color_actual = color_actual

    def draw_prediction(
        self,
        frame,
        predicted_point: ProjectedImagePoint,
        actual_point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Overlay predicted (and optional actual) positions on the frame."""
        if frame is None:
            return

        pred_center = (int(round(predicted_point.x)), int(round(predicted_point.y)))
        cv2.circle(
            frame,
            center=pred_center,
            radius=self.radius_px,
            color=self.color_pred,
            thickness=2,
        )

        label = f"{predicted_point.person_id or '?'} {predicted_point.source_camera_id}→{predicted_point.camera_id}"
        cv2.putText(
            frame,
            label,
            (pred_center[0] + 8, pred_center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.color_pred,
            1,
            cv2.LINE_AA,
        )

        if actual_point is not None:
            actual_center = (int(round(actual_point[0])), int(round(actual_point[1])))
            cv2.circle(
                frame,
                center=actual_center,
                radius=self.radius_px,
                color=self.color_actual,
                thickness=2,
            )
            cv2.line(frame, pred_center, actual_center, (255, 255, 255), 1, cv2.LINE_AA)
