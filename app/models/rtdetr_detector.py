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
        
        # COCO class names (RT-DETR is trained on COCO)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Person class ID in COCO (0-indexed)
        self.person_class_id = 0
        
        logger.info(f"RTDETRDetector configured for Person Class ID: {self.person_class_id}, Threshold: {self.confidence_threshold}")
    
    async def load_model(self):
        """
        Load the RT-DETR model.
        """
        if self._model_loaded_flag and self.model is not None:
            logger.info("RT-DETR model already loaded.")
            return
        
        logger.info(f"Loading RT-DETR model on device: {self.device}...")
        
        try:
            # Load RT-DETR model
            logger.info("RT-DETR weights path: %s", self.model_name)
            self.model = RTDETR(self.model_name)
            
            # Set device
            if self.device.type == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            else:
                self.model.to('cpu')
            
            self._model_loaded_flag = True
            logger.info("RT-DETR model loaded and configured successfully.")
            
        except Exception as e:
            logger.exception(f"Error loading RT-DETR model: {e}")
            self.model = None
            self._model_loaded_flag = False
            raise
    
    async def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """
        Warm up the model by performing a dummy inference.
        """
        if not self._model_loaded_flag or not self.model:
            logger.warning("Detector model not loaded. Cannot perform warmup.")
            return
        
        logger.info(f"Warming up RT-DETR detector on device {self.device}...")
        try:
            dummy_np_image = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            _ = await self.detect(dummy_np_image)
            logger.info("RT-DETR detector warmup successful.")
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
                    return self.model(image, conf=self.confidence_threshold, verbose=False)

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
