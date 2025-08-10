"""
Overlay Renderer

Handles rendering of visual overlays on camera frames including bounding boxes,
person IDs, and visual effects.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import math

from app.domains.detection.entities.detection import Detection
from app.domains.visualization.entities.overlay_config import OverlayConfig

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Renders visual overlays on camera frames."""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.min_text_size = 12
        self.max_text_size = 32
    
    def render_frame_overlays(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        config: OverlayConfig,
        focused_person_id: Optional[str] = None
    ) -> np.ndarray:
        """Render all overlays on a frame."""
        try:
            # Create a copy to avoid modifying original
            overlay_frame = frame.copy()
            
            # Sort detections by confidence (render higher confidence on top)
            sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
            
            for detection in sorted_detections:
                global_person_id = getattr(detection, 'global_person_id', None)
                is_focused = focused_person_id == global_person_id
                
                # Render bounding box
                overlay_frame = self._render_bounding_box(
                    overlay_frame, detection, config, is_focused
                )
                
                # Render person information text
                overlay_frame = self._render_person_info(
                    overlay_frame, detection, config, is_focused
                )
                
                # Render glow effect if enabled and person is focused
                if is_focused and config.enable_glow_effect:
                    overlay_frame = self._render_glow_effect(
                        overlay_frame, detection, config
                    )
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Error rendering frame overlays: {e}")
            return frame
    
    def _render_bounding_box(
        self,
        frame: np.ndarray,
        detection: Detection,
        config: OverlayConfig,
        is_focused: bool = False
    ) -> np.ndarray:
        """Render bounding box for a detection."""
        try:
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Get style configuration
            bbox_style = config.get_bbox_style(is_focused)
            color = bbox_style["color"]
            thickness = bbox_style["thickness"]
            opacity = bbox_style["opacity"]
            
            if opacity < 1.0:
                # Create overlay for transparency
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
            else:
                # Draw solid rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add corner markers for focused persons
            if is_focused:
                corner_length = 20
                corner_thickness = max(2, thickness // 2)
                self._draw_corner_markers(
                    frame, (x1, y1), (x2, y2), color, corner_length, corner_thickness
                )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering bounding box: {e}")
            return frame
    
    def _render_person_info(
        self,
        frame: np.ndarray,
        detection: Detection,
        config: OverlayConfig,
        is_focused: bool = False
    ) -> np.ndarray:
        """Render person information text."""
        try:
            # Get text style
            text_style = config.get_text_style()
            
            # Collect text lines to display
            text_lines = []
            
            if config.show_person_id:
                global_person_id = getattr(detection, 'global_person_id', 'Unknown')
                # Show short version of ID for display
                display_id = global_person_id.split('-')[-1] if '-' in global_person_id else global_person_id
                text_lines.append(f"ID: {display_id}")
            
            if config.show_confidence:
                text_lines.append(f"Conf: {detection.confidence:.2f}")
            
            if config.show_tracking_duration:
                tracking_duration = getattr(detection, 'tracking_duration', 0.0)
                text_lines.append(f"Time: {tracking_duration:.1f}s")
            
            if not text_lines:
                return frame
            
            # Calculate text position (above bounding box)
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            text_x = x1
            text_y = y1 - 5
            
            # Render each text line
            line_height = int(20 * text_style["font_scale"])
            
            for i, text_line in enumerate(text_lines):
                current_y = text_y - (i * line_height)
                
                # Make sure text stays within frame bounds
                if current_y < line_height:
                    current_y = y2 + line_height * (i + 1)
                
                frame = self._render_text_with_background(
                    frame, text_line, (text_x, current_y), text_style, is_focused
                )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering person info: {e}")
            return frame
    
    def _render_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        text_style: Dict[str, Any],
        is_focused: bool = False
    ) -> np.ndarray:
        """Render text with optional background."""
        try:
            x, y = position
            font_scale = text_style["font_scale"]
            if is_focused:
                font_scale *= 1.1  # Slightly larger for focused
            
            thickness = text_style["thickness"]
            color = text_style["color"]
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, font_scale, thickness
            )
            
            # Render background if configured
            if text_style["background_color"]:
                bg_color = text_style["background_color"]
                bg_opacity = text_style["background_opacity"]
                
                # Background rectangle coordinates
                bg_x1 = x - 2
                bg_y1 = y - text_height - 2
                bg_x2 = x + text_width + 2
                bg_y2 = y + baseline + 2
                
                if bg_opacity < 1.0:
                    # Transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
                    frame = cv2.addWeighted(overlay, bg_opacity, frame, 1 - bg_opacity, 0)
                else:
                    # Solid background
                    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            
            # Render text
            cv2.putText(frame, text, (x, y), self.font, font_scale, color, thickness)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering text: {e}")
            return frame
    
    def _render_glow_effect(
        self,
        frame: np.ndarray,
        detection: Detection,
        config: OverlayConfig
    ) -> np.ndarray:
        """Render glow effect around focused person."""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Create glow by drawing multiple rectangles with decreasing opacity
            glow_radius = config.glow_radius
            glow_color = config.focus_color
            
            for i in range(glow_radius, 0, -1):
                # Calculate opacity that decreases with radius
                opacity = 0.3 * (i / glow_radius)
                
                # Expand rectangle
                glow_x1 = max(0, x1 - i)
                glow_y1 = max(0, y1 - i)
                glow_x2 = min(frame.shape[1], x2 + i)
                glow_y2 = min(frame.shape[0], y2 + i)
                
                # Draw glow layer
                overlay = frame.copy()
                cv2.rectangle(overlay, (glow_x1, glow_y1), (glow_x2, glow_y2), 
                            glow_color, 2)
                frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering glow effect: {e}")
            return frame
    
    def _draw_corner_markers(
        self,
        frame: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        color: Tuple[int, int, int],
        length: int,
        thickness: int
    ) -> None:
        """Draw corner markers for focused bounding box."""
        try:
            x1, y1 = top_left
            x2, y2 = bottom_right
            
            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
            
            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
            
            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
            
            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)
            
        except Exception as e:
            logger.error(f"Error drawing corner markers: {e}")
    
    def create_multi_camera_view(
        self,
        frames: Dict[str, np.ndarray],
        layout: str = "grid",
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Create a multi-camera view composition."""
        try:
            if not frames:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            camera_ids = list(frames.keys())
            num_cameras = len(camera_ids)
            
            if target_size is None:
                target_size = (1280, 720)
            
            total_width, total_height = target_size
            
            if layout == "grid":
                # Calculate grid dimensions
                grid_cols = math.ceil(math.sqrt(num_cameras))
                grid_rows = math.ceil(num_cameras / grid_cols)
                
                cell_width = total_width // grid_cols
                cell_height = total_height // grid_rows
                
                # Create composite image
                composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)
                
                for i, camera_id in enumerate(camera_ids):
                    frame = frames[camera_id]
                    
                    # Calculate position
                    row = i // grid_cols
                    col = i % grid_cols
                    
                    start_x = col * cell_width
                    start_y = row * cell_height
                    end_x = start_x + cell_width
                    end_y = start_y + cell_height
                    
                    # Resize frame to fit cell
                    resized_frame = cv2.resize(frame, (cell_width, cell_height))
                    composite[start_y:end_y, start_x:end_x] = resized_frame
                    
                    # Add camera label
                    cv2.putText(composite, camera_id, (start_x + 10, start_y + 30),
                              self.font, 0.7, (255, 255, 255), 2)
                
                return composite
            
            elif layout == "horizontal":
                # Arrange cameras horizontally
                cell_width = total_width // num_cameras
                composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)
                
                for i, camera_id in enumerate(camera_ids):
                    frame = frames[camera_id]
                    start_x = i * cell_width
                    end_x = start_x + cell_width
                    
                    # Resize frame
                    resized_frame = cv2.resize(frame, (cell_width, total_height))
                    composite[:, start_x:end_x] = resized_frame
                    
                    # Add camera label
                    cv2.putText(composite, camera_id, (start_x + 10, 30),
                              self.font, 0.7, (255, 255, 255), 2)
                
                return composite
            
            else:
                # Default: return first frame
                return list(frames.values())[0]
                
        except Exception as e:
            logger.error(f"Error creating multi-camera view: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def apply_focus_highlight(
        self,
        frame: np.ndarray,
        detection: Detection,
        highlight_intensity: float = 0.3
    ) -> np.ndarray:
        """Apply highlight effect to focused person."""
        try:
            # Create a copy
            highlighted = frame.copy()
            
            # Darken the entire frame
            darkened = cv2.multiply(highlighted, np.array([1.0 - highlight_intensity]))
            
            # Create mask for the person
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            # Apply mask to keep person area bright
            mask_3d = cv2.merge([mask, mask, mask]) / 255.0
            result = darkened * (1 - mask_3d) + highlighted * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error applying focus highlight: {e}")
            return frame
    
    def get_optimal_text_size(self, text: str, max_width: int, max_height: int) -> float:
        """Calculate optimal text size for given constraints."""
        try:
            # Start with default scale
            font_scale = 0.7
            thickness = 2
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                text, self.font, font_scale, thickness
            )
            
            # Adjust scale to fit constraints
            if text_width > max_width:
                font_scale *= max_width / text_width
            if text_height > max_height:
                font_scale *= max_height / text_height
            
            # Clamp to reasonable bounds
            font_scale = max(0.3, min(2.0, font_scale))
            
            return font_scale
            
        except Exception as e:
            logger.error(f"Error calculating text size: {e}")
            return 0.7