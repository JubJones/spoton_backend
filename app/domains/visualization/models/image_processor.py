"""
Image Processor

Handles image processing operations including cropping, resizing, and encoding.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64

from app.domains.detection.entities.detection import Detection
from app.domains.visualization.entities.cropped_image import CroppedImage

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations for visualization."""
    
    def __init__(self):
        self.supported_formats = ["jpeg", "jpg", "png", "bmp"]
        self.default_quality = 85
        self.default_crop_padding = 10  # pixels
        
    def decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode image bytes to numpy array."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image data")
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise
    
    def encode_image(self, image: np.ndarray, format: str = "jpeg", quality: int = 85) -> bytes:
        """Encode numpy array to image bytes."""
        try:
            format = format.lower()
            if format in ["jpg", "jpeg"]:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                success, encoded_img = cv2.imencode('.jpg', image, encode_param)
            elif format == "png":
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                success, encoded_img = cv2.imencode('.png', image, encode_param)
            else:
                # Default to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                success, encoded_img = cv2.imencode('.jpg', image, encode_param)
            
            if not success:
                raise ValueError(f"Failed to encode image as {format}")
            
            return encoded_img.tobytes()
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def crop_person_from_frame(
        self, 
        frame: np.ndarray, 
        detection: Detection,
        padding: int = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Crop person from frame based on detection bounding box."""
        try:
            if padding is None:
                padding = self.default_crop_padding
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = detection.bbox
            
            # Add padding
            x1_padded = max(0, int(x1 - padding))
            y1_padded = max(0, int(y1 - padding))
            x2_padded = min(frame_width, int(x2 + padding))
            y2_padded = min(frame_height, int(y2 + padding))
            
            # Crop the image
            cropped = frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if cropped.size == 0:
                logger.warning(f"Empty crop for detection: {detection.bbox}")
                return np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Resize if target size specified
            if target_size is not None:
                cropped = cv2.resize(cropped, target_size)
            
            return cropped
            
        except Exception as e:
            logger.error(f"Error cropping person from frame: {e}")
            # Return a placeholder image
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def create_cropped_image(
        self,
        frame: np.ndarray,
        detection: Detection,
        camera_id: str,
        frame_index: int,
        global_person_id: str,
        image_quality: int = None
    ) -> CroppedImage:
        """Create a CroppedImage entity from a detection."""
        try:
            if image_quality is None:
                image_quality = self.default_quality
            
            # Crop the person from frame
            cropped_array = self.crop_person_from_frame(frame, detection)
            
            # Encode as bytes
            cropped_bytes = self.encode_image(cropped_array, "jpeg", image_quality)
            
            # Get dimensions
            height, width = cropped_array.shape[:2]
            
            # Create CroppedImage entity
            cropped_image = CroppedImage(
                global_person_id=global_person_id,
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp=detection.timestamp,
                image_data=cropped_bytes,
                image_format="jpeg",
                image_quality=image_quality,
                original_bbox=tuple(detection.bbox),
                width=width,
                height=height,
                confidence=detection.confidence,
                detection_quality=self._assess_detection_quality(detection)
            )
            
            logger.debug(f"Created cropped image for person {global_person_id}: {width}x{height}, {len(cropped_bytes)} bytes")
            return cropped_image
            
        except Exception as e:
            logger.error(f"Error creating cropped image: {e}")
            raise
    
    def batch_crop_persons(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        camera_id: str,
        frame_index: int,
        image_quality: int = None
    ) -> Dict[str, CroppedImage]:
        """Create cropped images for multiple detections."""
        cropped_images = {}
        
        for i, detection in enumerate(detections):
            try:
                # Use global_person_id if available, otherwise generate one
                global_person_id = getattr(detection, 'global_person_id', f"person_{camera_id}_{frame_index}_{i}")
                
                cropped_image = self.create_cropped_image(
                    frame=frame,
                    detection=detection,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    global_person_id=global_person_id,
                    image_quality=image_quality
                )
                
                cropped_images[global_person_id] = cropped_image
                
            except Exception as e:
                logger.error(f"Error creating cropped image for detection {i}: {e}")
                continue
        
        logger.debug(f"Created {len(cropped_images)} cropped images from {len(detections)} detections")
        return cropped_images
    
    def resize_frame(
        self, 
        frame: np.ndarray, 
        target_width: int, 
        target_height: int,
        maintain_aspect_ratio: bool = True
    ) -> np.ndarray:
        """Resize frame to target dimensions."""
        try:
            if maintain_aspect_ratio:
                # Calculate aspect ratios
                frame_h, frame_w = frame.shape[:2]
                frame_aspect = frame_w / frame_h
                target_aspect = target_width / target_height
                
                if frame_aspect > target_aspect:
                    # Frame is wider, fit to width
                    new_width = target_width
                    new_height = int(target_width / frame_aspect)
                else:
                    # Frame is taller, fit to height
                    new_height = target_height
                    new_width = int(target_height * frame_aspect)
                
                resized = cv2.resize(frame, (new_width, new_height))
                
                # Pad to target size if needed
                if new_width != target_width or new_height != target_height:
                    # Create black canvas
                    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    
                    # Center the resized image
                    start_y = (target_height - new_height) // 2
                    start_x = (target_width - new_width) // 2
                    canvas[start_y:start_y+new_height, start_x:start_x+new_width] = resized
                    
                    return canvas
                else:
                    return resized
            else:
                return cv2.resize(frame, (target_width, target_height))
                
        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame
    
    def apply_quality_settings(
        self, 
        frame: np.ndarray, 
        quality: int,
        format: str = "jpeg"
    ) -> bytes:
        """Apply quality settings and encode frame."""
        return self.encode_image(frame, format, quality)
    
    def _assess_detection_quality(self, detection: Detection) -> str:
        """Assess the quality of a detection based on confidence and bbox size."""
        confidence = detection.confidence
        
        # Calculate bounding box area (rough quality indicator)
        x1, y1, x2, y2 = detection.bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if confidence >= 0.8 and bbox_area >= 5000:  # High confidence, decent size
            return "high"
        elif confidence >= 0.6 and bbox_area >= 2000:  # Medium confidence/size
            return "medium"
        else:
            return "low"
    
    def convert_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def create_data_uri(self, image_data: bytes, format: str = "jpeg") -> str:
        """Create data URI from image bytes."""
        mime_type = f"image/{format}"
        b64_data = self.convert_to_base64(image_data)
        return f"data:{mime_type};base64,{b64_data}"
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get information about an image."""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes,
            "aspect_ratio": width / height
        }