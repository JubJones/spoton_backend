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
        radius_px: int = 8,  # Increased from 6
        color_pred: Tuple[int, int, int] = (0, 255, 255),
        color_actual: Tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        self.radius_px = max(1, radius_px)
        self.color_pred = color_pred
        self.color_actual = color_actual

    def _id_to_color(self, id_str: str) -> Tuple[int, int, int]:
        """Generate a consistent unique color from an ID string."""
        if not id_str or id_str == "?":
            return self.color_pred  # Default yellow for unknown
        
        # Simple hash to RGB
        hash_val = hash(id_str)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        
        # Ensure high visibility (bright colors)
        # Verify it's not too dark: if sum is low, adding offset
        if r + g + b < 200:
            r = (r + 100) % 255
            g = (g + 100) % 255
            b = (b + 100) % 255

        return (int(b), int(g), int(r)) # OpenCV uses BGR

    def draw_prediction(
        self,
        frame,
        predicted_point: ProjectedImagePoint,
        actual_point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Overlay predicted (and optional actual) positions on the frame."""
        if frame is None:
            return

        # Prefer global_id if available
        display_id = getattr(predicted_point, "global_id", None) or predicted_point.person_id or "?"
        
        # Check if matched across cameras
        is_matched = getattr(predicted_point, "is_matched", False)

        if not is_matched:
             # Default GREEN for unmatched tracks
             color = (0, 255, 0)
        else:
            # Consistent hashed color for matched (shared) IDs
            color = self._id_to_color(str(display_id))

        pred_center = (int(round(predicted_point.x)), int(round(predicted_point.y)))
        
        # Draw target circle
        cv2.circle(
            frame,
            center=pred_center,
            radius=self.radius_px,
            color=color,
            thickness=2,
        )
        # Add a white rim for contrast
        cv2.circle(
            frame,
            center=pred_center,
            radius=self.radius_px + 2,
            color=(255, 255, 255),
            thickness=1,
        )

        label = f"{display_id} {predicted_point.source_camera_id}->{predicted_point.camera_id}"
        
        # Calculate text size for background box
        font_scale = 0.8  # Increased from 0.4
        thickness = 2     # Increased from 1
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw background rectangle for text readability
        text_origin = (pred_center[0] + 10, pred_center[1])
        cv2.rectangle(
            frame, 
            (text_origin[0], text_origin[1] - text_h - baseline), 
            (text_origin[0] + text_w, text_origin[1] + baseline), 
            (0, 0, 0), 
            cv2.FILLED
        )
        
        cv2.putText(
            frame,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
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
