"""
Detection Annotator for Phase 2: Core Detection Pipeline.

Provides bounding box visualization, confidence scoring, and frame encoding
for real-time detection streaming as outlined in DETECTION.md Phase 2.
"""

import cv2
import numpy as np
import base64
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnnotationStyle:
    """Configuration for detection annotation appearance."""
    box_color: Tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    text_color: Tuple[int, int, int] = (0, 0, 0)   # Black text
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    label_padding: int = 10


class DetectionAnnotator:
    """
    Annotates detection frames with bounding boxes and confidence scores.
    
    Features:
    - Bounding box visualization with confidence scores
    - Base64 frame encoding for WebSocket transmission
    - Configurable annotation styles
    - Error handling for invalid frames/detections
    """
    
    def __init__(self, style: Optional[AnnotationStyle] = None):
        """Initialize annotator with optional custom styling."""
        self.style = style or AnnotationStyle()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        logger.info("DetectionAnnotator initialized for Phase 2 detection pipeline")
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Annotate frame with detection bounding boxes and confidence scores.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            detections: List of detection dictionaries with bbox and confidence
            
        Returns:
            Annotated frame as numpy array
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided for annotation")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            annotated_frame = frame.copy()
            
            if not detections:
                logger.debug("No detections to annotate")
                return annotated_frame
            
            for i, detection in enumerate(detections):
                try:
                    self._draw_detection_box(annotated_frame, detection, i)
                except Exception as e:
                    logger.warning(f"Error annotating detection {i}: {e}")
                    continue
            
            logger.debug(f"Annotated frame with {len(detections)} detections")
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error in frame annotation: {e}")
            return frame.copy()  # Return original frame on error
    
    def _draw_detection_box(self, frame: np.ndarray, detection: Dict, detection_idx: int):
        """Draw individual detection bounding box with label."""
        bbox = detection.get("bbox", {})
        confidence = detection.get("confidence", 0.0)
        class_name = detection.get("class_name", "person")
        detection_id = detection.get("detection_id", f"det_{detection_idx:03d}")
        
        # Extract bounding box coordinates
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        
        # Validate coordinates
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bounding box coordinates: ({x1}, {y1}, {x2}, {y2})")
            return
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.style.box_color, self.style.box_thickness)
        
        # Draw confidence label
        label = f"{class_name}: {confidence:.2f}"
        if detection_id and detection_id != f"det_{detection_idx:03d}":
            label = f"{detection_id} | {label}"
        
        self._draw_label(frame, label, x1, y1)
    
    def _draw_label(self, frame: np.ndarray, label: str, x: int, y: int):
        """Draw text label with background rectangle."""
        # Calculate label size
        label_size, baseline = cv2.getTextSize(
            label, self.font, self.style.font_scale, self.style.font_thickness
        )
        
        # Ensure label is within frame bounds
        h, w = frame.shape[:2]
        label_y = max(label_size[1] + self.style.label_padding, y)
        label_x = min(x, w - label_size[0])
        
        # Draw label background rectangle
        cv2.rectangle(
            frame,
            (label_x, label_y - label_size[1] - self.style.label_padding),
            (label_x + label_size[0], label_y + baseline),
            self.style.box_color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, label, (label_x, label_y - 5),
            self.font, self.style.font_scale, self.style.text_color, self.style.font_thickness
        )
    
    def frame_to_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """
        Convert frame to base64 string for WebSocket transmission.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            quality: JPEG quality (1-100, higher is better quality)
            
        Returns:
            Base64 encoded JPEG string
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided for base64 encoding")
            return ""
        
        try:
            # Validate quality parameter
            quality = max(1, min(100, quality))
            
            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                logger.error("Failed to encode frame as JPEG")
                return ""
            
            # Convert to base64
            base64_string = base64.b64encode(buffer).decode('utf-8')
            
            logger.debug(f"Encoded frame to base64: {len(base64_string)} characters, quality={quality}")
            return base64_string
            
        except Exception as e:
            logger.error(f"Error encoding frame to base64: {e}")
            return ""
    
    def create_detection_overlay(self, frame: np.ndarray, detections: List[Dict]) -> Dict[str, str]:
        """
        Create both original and annotated frame encodings for WebSocket streaming.
        
        Args:
            frame: Input frame as numpy array
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with 'original_b64' and 'annotated_b64' keys
        """
        try:
            # Encode original frame
            original_b64 = self.frame_to_base64(frame)
            
            # Create and encode annotated frame
            annotated_frame = self.annotate_frame(frame, detections)
            annotated_b64 = self.frame_to_base64(annotated_frame)
            
            return {
                "original_b64": original_b64,
                "annotated_b64": annotated_b64,
                "width": frame.shape[1] if frame is not None else 0,
                "height": frame.shape[0] if frame is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Error creating detection overlay: {e}")
            return {
                "original_b64": "",
                "annotated_b64": "",
                "width": 0,
                "height": 0
            }
    
    def validate_detection_format(self, detection: Dict) -> bool:
        """
        Validate detection dictionary format for annotation.
        
        Args:
            detection: Detection dictionary to validate
            
        Returns:
            True if detection format is valid, False otherwise
        """
        required_fields = ["bbox", "confidence"]
        bbox_fields = ["x1", "y1", "x2", "y2"]
        
        try:
            # Check required top-level fields
            for field in required_fields:
                if field not in detection:
                    logger.warning(f"Missing required field '{field}' in detection")
                    return False
            
            # Check bbox fields
            bbox = detection["bbox"]
            for field in bbox_fields:
                if field not in bbox:
                    logger.warning(f"Missing required bbox field '{field}' in detection")
                    return False
                
                # Ensure numeric values
                if not isinstance(bbox[field], (int, float)):
                    logger.warning(f"Invalid bbox field '{field}': not numeric")
                    return False
            
            # Check confidence is numeric
            if not isinstance(detection["confidence"], (int, float)):
                logger.warning("Invalid confidence: not numeric")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating detection format: {e}")
            return False
    
    def get_annotation_stats(self, detections: List[Dict]) -> Dict[str, any]:
        """
        Get statistics about detections for debugging and monitoring.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with annotation statistics
        """
        if not detections:
            return {
                "total_detections": 0,
                "valid_detections": 0,
                "invalid_detections": 0,
                "average_confidence": 0.0,
                "confidence_range": {"min": 0.0, "max": 0.0}
            }
        
        valid_count = 0
        confidences = []
        
        for detection in detections:
            if self.validate_detection_format(detection):
                valid_count += 1
                confidences.append(detection["confidence"])
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_detections": len(detections),
            "valid_detections": valid_count,
            "invalid_detections": len(detections) - valid_count,
            "average_confidence": avg_confidence,
            "confidence_range": {
                "min": min(confidences) if confidences else 0.0,
                "max": max(confidences) if confidences else 0.0
            }
        }