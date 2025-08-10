"""
Frame Composition Service

Handles composition and processing of camera frames with overlays.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from app.domains.detection.entities.detection import Detection
from app.domains.visualization.entities.visual_frame import VisualFrame
from app.domains.visualization.entities.overlay_config import OverlayConfig
from app.domains.visualization.entities.cropped_image import CroppedImage
from app.domains.visualization.models.image_processor import ImageProcessor
from app.domains.visualization.models.overlay_renderer import OverlayRenderer

logger = logging.getLogger(__name__)


class FrameCompositionService:
    """Service for composing camera frames with overlays."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.overlay_renderer = OverlayRenderer()
        
        # Performance tracking
        self.processing_times = []
        self.max_history = 100
        
        logger.info("FrameCompositionService initialized")
    
    def compose_visual_frame(
        self,
        camera_id: str,
        frame_index: int,
        frame_data: bytes,
        detections: List[Detection],
        overlay_config: Optional[OverlayConfig] = None,
        focused_person_id: Optional[str] = None
    ) -> VisualFrame:
        """Compose a visual frame with overlays and cropped persons."""
        start_time = time.time()
        
        try:
            if overlay_config is None:
                overlay_config = OverlayConfig()
            
            # Decode original frame
            original_frame = self.image_processor.decode_image(frame_data)
            original_height, original_width = original_frame.shape[:2]
            
            # Create cropped person images
            cropped_persons = self.image_processor.batch_crop_persons(
                frame=original_frame,
                detections=detections,
                camera_id=camera_id,
                frame_index=frame_index,
                image_quality=overlay_config.overlay_quality
            )
            
            # Render overlays
            processed_frame = self.overlay_renderer.render_frame_overlays(
                frame=original_frame,
                detections=detections,
                config=overlay_config,
                focused_person_id=focused_person_id
            )
            
            # Apply focus highlight if needed
            if focused_person_id:
                focused_detection = self._get_detection_by_person_id(detections, focused_person_id)
                if focused_detection:
                    processed_frame = self.overlay_renderer.apply_focus_highlight(
                        processed_frame, focused_detection
                    )
            
            # Encode processed frame
            processed_frame_data = self.image_processor.encode_image(
                processed_frame, 
                format="jpeg", 
                quality=overlay_config.overlay_quality
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            self._track_processing_time(processing_time_ms)
            
            # Create visual frame entity
            visual_frame = VisualFrame(
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp=datetime.utcnow(),
                original_frame_data=frame_data,
                original_width=original_width,
                original_height=original_height,
                processed_frame_data=processed_frame_data,
                processed_format="jpeg",
                processed_quality=overlay_config.overlay_quality,
                detections=detections,
                focused_person_id=focused_person_id,
                cropped_persons=cropped_persons,
                overlay_config=overlay_config,
                processing_time_ms=processing_time_ms
            )
            
            logger.debug(
                f"Composed visual frame for {camera_id}:{frame_index} - "
                f"{len(detections)} detections, {len(cropped_persons)} crops, "
                f"{processing_time_ms:.1f}ms"
            )
            
            return visual_frame
            
        except Exception as e:
            logger.error(f"Error composing visual frame: {e}")
            raise
    
    def compose_multi_camera_frame(
        self,
        frames: Dict[str, VisualFrame],
        layout: str = "grid",
        target_size: Optional[Tuple[int, int]] = None
    ) -> bytes:
        """Compose multiple camera frames into a single view."""
        try:
            # Extract processed frames
            frame_images = {}
            for camera_id, visual_frame in frames.items():
                frame_array = self.image_processor.decode_image(visual_frame.processed_frame_data)
                frame_images[camera_id] = frame_array
            
            # Create multi-camera composition
            composite_frame = self.overlay_renderer.create_multi_camera_view(
                frames=frame_images,
                layout=layout,
                target_size=target_size
            )
            
            # Encode composite frame
            composite_data = self.image_processor.encode_image(
                composite_frame, format="jpeg", quality=85
            )
            
            logger.debug(f"Composed multi-camera frame with {len(frames)} cameras")
            return composite_data
            
        except Exception as e:
            logger.error(f"Error composing multi-camera frame: {e}")
            raise
    
    def update_overlay_config(
        self,
        visual_frame: VisualFrame,
        new_config: OverlayConfig
    ) -> VisualFrame:
        """Update overlay configuration and reprocess frame."""
        try:
            # Decode original frame
            original_frame = self.image_processor.decode_image(visual_frame.original_frame_data)
            
            # Re-render with new config
            processed_frame = self.overlay_renderer.render_frame_overlays(
                frame=original_frame,
                detections=visual_frame.detections,
                config=new_config,
                focused_person_id=visual_frame.focused_person_id
            )
            
            # Encode updated frame
            processed_frame_data = self.image_processor.encode_image(
                processed_frame,
                format="jpeg",
                quality=new_config.overlay_quality
            )
            
            # Update visual frame
            visual_frame.processed_frame_data = processed_frame_data
            visual_frame.overlay_config = new_config
            visual_frame.processed_quality = new_config.overlay_quality
            
            logger.debug(f"Updated overlay config for {visual_frame.camera_id}:{visual_frame.frame_index}")
            return visual_frame
            
        except Exception as e:
            logger.error(f"Error updating overlay config: {e}")
            return visual_frame
    
    def set_focus_person(
        self,
        visual_frame: VisualFrame,
        person_id: Optional[str]
    ) -> VisualFrame:
        """Update focus person and reprocess frame."""
        try:
            if visual_frame.focused_person_id == person_id:
                return visual_frame  # No change needed
            
            # Update focus
            visual_frame.focused_person_id = person_id
            
            # Decode original frame
            original_frame = self.image_processor.decode_image(visual_frame.original_frame_data)
            
            # Re-render with focus
            processed_frame = self.overlay_renderer.render_frame_overlays(
                frame=original_frame,
                detections=visual_frame.detections,
                config=visual_frame.overlay_config,
                focused_person_id=person_id
            )
            
            # Apply focus highlight if needed
            if person_id:
                focused_detection = self._get_detection_by_person_id(visual_frame.detections, person_id)
                if focused_detection:
                    processed_frame = self.overlay_renderer.apply_focus_highlight(
                        processed_frame, focused_detection
                    )
            
            # Encode updated frame
            processed_frame_data = self.image_processor.encode_image(
                processed_frame,
                format=visual_frame.processed_format,
                quality=visual_frame.processed_quality
            )
            
            # Update frame data
            visual_frame.processed_frame_data = processed_frame_data
            
            logger.debug(f"Set focus person {person_id} for {visual_frame.camera_id}:{visual_frame.frame_index}")
            return visual_frame
            
        except Exception as e:
            logger.error(f"Error setting focus person: {e}")
            return visual_frame
    
    def get_cropped_person_image(
        self,
        visual_frame: VisualFrame,
        global_person_id: str
    ) -> Optional[CroppedImage]:
        """Get cropped image for a specific person."""
        return visual_frame.cropped_persons.get(global_person_id)
    
    def regenerate_cropped_images(
        self,
        visual_frame: VisualFrame,
        image_quality: Optional[int] = None
    ) -> VisualFrame:
        """Regenerate cropped person images with different quality."""
        try:
            if image_quality is None:
                image_quality = visual_frame.overlay_config.overlay_quality
            
            # Decode original frame
            original_frame = self.image_processor.decode_image(visual_frame.original_frame_data)
            
            # Regenerate cropped images
            cropped_persons = self.image_processor.batch_crop_persons(
                frame=original_frame,
                detections=visual_frame.detections,
                camera_id=visual_frame.camera_id,
                frame_index=visual_frame.frame_index,
                image_quality=image_quality
            )
            
            # Update visual frame
            visual_frame.cropped_persons = cropped_persons
            
            logger.debug(f"Regenerated {len(cropped_persons)} cropped images for {visual_frame.camera_id}:{visual_frame.frame_index}")
            return visual_frame
            
        except Exception as e:
            logger.error(f"Error regenerating cropped images: {e}")
            return visual_frame
    
    def _get_detection_by_person_id(
        self,
        detections: List[Detection],
        global_person_id: str
    ) -> Optional[Detection]:
        """Find detection by global person ID."""
        for detection in detections:
            if hasattr(detection, 'global_person_id') and detection.global_person_id == global_person_id:
                return detection
        return None
    
    def _track_processing_time(self, processing_time_ms: float):
        """Track processing time for performance monitoring."""
        self.processing_times.append(processing_time_ms)
        if len(self.processing_times) > self.max_history:
            self.processing_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {
                "avg_processing_time_ms": 0.0,
                "min_processing_time_ms": 0.0,
                "max_processing_time_ms": 0.0,
                "samples_count": 0
            }
        
        return {
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time_ms": min(self.processing_times),
            "max_processing_time_ms": max(self.processing_times),
            "samples_count": len(self.processing_times),
            "recent_avg_ms": sum(self.processing_times[-10:]) / min(10, len(self.processing_times))
        }
    
    def create_thumbnail(
        self,
        visual_frame: VisualFrame,
        size: Tuple[int, int] = (320, 240)
    ) -> bytes:
        """Create thumbnail version of the frame."""
        try:
            # Decode processed frame
            frame = self.image_processor.decode_image(visual_frame.processed_frame_data)
            
            # Resize to thumbnail size
            thumbnail = self.image_processor.resize_frame(
                frame, size[0], size[1], maintain_aspect_ratio=True
            )
            
            # Encode with reduced quality
            thumbnail_data = self.image_processor.encode_image(
                thumbnail, format="jpeg", quality=70
            )
            
            return thumbnail_data
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            # Return original if thumbnail creation fails
            return visual_frame.processed_frame_data