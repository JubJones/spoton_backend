"""
RT-DETR detector implementation for the legacy models system.
"""

import logging
import asyncio
from typing import List, Optional, Tuple
import numpy as np

try:
    import torch
    from ultralytics import RTDETR
    ULTRALYTICS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    TORCH_AVAILABLE = False

from .base_models import AbstractDetector, Detection, BoundingBox

logger = logging.getLogger(__name__)


class RTDETRDetector(AbstractDetector):
    """
    RT-DETR detector implementation using Ultralytics.
    Optimized for person detection with real-time performance.
    """
    
    def __init__(self, model_name: str = "rtdetr-l.pt", confidence_threshold: float = 0.5):
        """
        Initialize RT-DETR detector.
        
        Args:
            model_name: Model weights filename
            confidence_threshold: Minimum confidence for detections
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install PyTorch.")
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Please install ultralytics.")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model: Optional[RTDETR] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_loaded_flag = False
        
        # Performance optimization settings (configurable via env vars)
        import os
        self.imgsz = int(os.getenv("DETECTION_IMGSZ", "640"))
        # Only use half precision if on CUDA
        self.use_half = self.device.type == 'cuda'
        
        # Person class ID in COCO (0-indexed)
        self.person_class_id = 0
            
    async def load_model(self):
        """
        Load the RT-DETR model.
        """
        if self._model_loaded_flag and self.model is not None:
            return
        
        try:
            logger.info(f"Loading RT-DETR model: {self.model_name} on device: {self.device}...")
            
            # CRITICAL CHECK: Ensure model file exists before attempting to load
            import os
            import sys
            if not os.path.exists(self.model_name):
                logger.critical(f"❌ FATAL: Model file not found at: {self.model_name}")
                sys.exit(1)

            # Initialize RTDETR model
            self.model = RTDETR(self.model_name)
            
            # Note: We don't need to explicitly move to device as Ultralytics handles it,
            # but for TensorRT engines it's automatic. For PyTorch models, we can hint.
            # self.model.to(self.device)
            
            self._model_loaded_flag = True
            logger.info(f"✅ RT-DETR model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load RT-DETR model {self.model_name}: {e}")
            self.model = None
            raise

    async def warmup(self):
        """
        Perform a warmup run to compile shaders/engines.
        """
        if not self._model_loaded_flag or not self.model:
            await self.load_model()
            
        try:
            logger.info("Starting RT-DETR warmup...")
            # Create dummy image
            dummy_image_shape = (self.imgsz, self.imgsz, 3)
            # Run inference on dummy thread
            import asyncio
            
            # Using random data for warmup
            dummy_np_image = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            _ = await self.detect(dummy_np_image)
            pass # logger.info("RT-DETR detector warmup successful.")
        except Exception as e:
            logger.error(f"RT-DETR detector warmup failed: {e}", exc_info=True)
    
    async def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Perform person detection on an image.
        
        Args:
            image: A NumPy array representing the image (expected in BGR format from OpenCV).
            
        Returns:
            A list of Detection objects for persons found.
            
        Raises:
            RuntimeError: If the detector model has not been loaded.
        """
        if not self._model_loaded_flag or not self.model:
            raise RuntimeError("Detector model not loaded. Call load_model() first.")
        
        try:
            # Run inference without tracking gradients and offload to thread to avoid blocking event loop
            import asyncio
            import torch

            def _infer():
                with torch.inference_mode():
                    return self.model(image, conf=self.confidence_threshold, imgsz=self.imgsz, half=self.use_half, verbose=False)

            results = await asyncio.to_thread(_infer)
            
            # Process results
            detections_result: List[Detection] = []
            
            if len(results) > 0:
                result = results[0]  # Get first (and only) result
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    if boxes.xyxy is not None:
                        box_coords = boxes.xyxy.cpu().numpy()
                        confidences = boxes.conf.cpu().numpy()
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                        
                        original_h, original_w = image.shape[:2]
                        
                        for box, score_val, label_id in zip(box_coords, confidences, class_ids):
                            # Filter for person class only (class_id = 0 in COCO)
                            if int(label_id) == self.person_class_id and score_val >= self.confidence_threshold:
                                x1, y1, x2, y2 = box
                                
                                # Clip coordinates to image bounds
                                x1_clipped = max(0.0, float(x1))
                                y1_clipped = max(0.0, float(y1))
                                x2_clipped = min(float(original_w), float(x2))
                                y2_clipped = min(float(original_h), float(y2))
                                
                                # Ensure box has positive area after clipping
                                if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                                    class_name = "person"
                                    
                                    detections_result.append(
                                        Detection(
                                            bbox=BoundingBox(x1=x1_clipped, y1=y1_clipped, x2=x2_clipped, y2=y2_clipped),
                                            confidence=float(score_val),
                                            class_id=int(label_id),
                                            class_name=class_name
                                        )
                                    )
            
            return detections_result
            
        except Exception as e:
            logger.error(f"Error during RT-DETR detection: {e}")
            return []

    async def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Perform batch person detection on multiple images in a single GPU call.
        
        This is more efficient than calling detect() multiple times when processing
        multiple cameras, as it runs a single GPU inference for all images.
        
        Args:
            images: List of NumPy arrays representing images (BGR format from OpenCV).
            
        Returns:
            List of detection lists, one per input image.
            
        Raises:
            RuntimeError: If the detector model has not been loaded.
        """
        if not self._model_loaded_flag or not self.model:
            raise RuntimeError("Detector model not loaded. Call load_model() first.")
        
        if not images:
            return []
        
        # Single image fallback to regular detect
        if len(images) == 1:
            result = await self.detect(images[0])
            return [result]
        
        try:
            import torch

            def _batch_infer():
                with torch.inference_mode():
                    # Ultralytics supports list of images for batch inference
                    return self.model(images, conf=self.confidence_threshold, imgsz=self.imgsz, half=self.use_half, verbose=False)

            results = await asyncio.to_thread(_batch_infer)
            
            # Parse each result
            batch_detections: List[List[Detection]] = []
            for i, (result, image) in enumerate(zip(results, images)):
                try:
                    detections = self._parse_single_result(result, image)
                    batch_detections.append(detections)
                except Exception as e:
                    logger.error(f"Error parsing batch result {i}: {e}")
                    batch_detections.append([])
            
            return batch_detections
            
        except Exception as e:
            logger.error(f"Error during RT-DETR batch detection: {e}")
            return [[] for _ in images]

    def _parse_single_result(self, result, image: np.ndarray) -> List[Detection]:
        """
        Parse a single inference result into Detection objects.
        
        Args:
            result: Ultralytics inference result object
            image: Original image for dimension reference
            
        Returns:
            List of Detection objects for persons found
        """
        detections_result: List[Detection] = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            if boxes.xyxy is not None:
                box_coords = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                
                original_h, original_w = image.shape[:2]
                
                for box, score_val, label_id in zip(box_coords, confidences, class_ids):
                    # Filter for person class only (class_id = 0 in COCO)
                    if int(label_id) == self.person_class_id and score_val >= self.confidence_threshold:
                        x1, y1, x2, y2 = box
                        
                        # Clip coordinates to image bounds
                        x1_clipped = max(0.0, float(x1))
                        y1_clipped = max(0.0, float(y1))
                        x2_clipped = min(float(original_w), float(x2))
                        y2_clipped = min(float(original_h), float(y2))
                        
                        # Ensure box has positive area after clipping
                        if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                            class_name = "person"
                            
                            detections_result.append(
                                Detection(
                                    bbox=BoundingBox(x1=x1_clipped, y1=y1_clipped, x2=x2_clipped, y2=y2_clipped),
                                    confidence=float(score_val),
                                    class_id=int(label_id),
                                    class_name=class_name
                                )
                            )
        
        return detections_result
