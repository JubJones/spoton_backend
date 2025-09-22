"""
YOLO detector implementation with GPU acceleration.
High-speed person detection using ultralytics YOLOv8/YOLOv5.
"""
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .base_detector import AbstractDetector, DetectionResult, DetectorFactory
from app.infrastructure.gpu import get_gpu_manager

logger = logging.getLogger(__name__)


class YOLODetector(AbstractDetector):
    """
    YOLO detector with GPU acceleration.
    High-speed person detection with real-time performance.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
        batch_size: int = 8,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: Model name (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')
            confidence_threshold: Minimum confidence for detections
            device: Device for inference (auto-detected if None)
            weights_path: Path to custom weights (optional)
            nms_threshold: NMS threshold for duplicate removal
            max_detections: Maximum number of detections per image
            batch_size: Batch size for inference
            input_size: Input image size (width, height)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install PyTorch.")
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Please install ultralytics.")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.weights_path = weights_path or model_name
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.batch_size = batch_size
        self.input_size = input_size
        
        # Model and device setup
        self.model: Optional[YOLO] = None
        self.device = device or get_gpu_manager().get_optimal_device()
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.total_detections = 0
        self.total_images = 0
        
        # COCO class names (YOLO is trained on COCO)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Person class ID in COCO (0-indexed)
        self.person_class_id = 0
        
        logger.info(f"YOLODetector initialized with device: {self.device}")
    
    async def load_model(self) -> None:
        """Load the YOLO model."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            # Load YOLO model
            self.model = YOLO(self.weights_path)
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model parameters
            self.model.overrides['conf'] = self.confidence_threshold
            self.model.overrides['iou'] = self.nms_threshold
            self.model.overrides['max_det'] = self.max_detections
            self.model.overrides['imgsz'] = self.input_size
            
            logger.info(f"YOLO model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    async def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a single image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of detections
        """
        if self.model is None:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            # Preprocess image
            preprocessed = await self.preprocess_image(image)
            
            # Run inference
            results = self.model.predict(
                source=preprocessed,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False
            )
            
            # Postprocess results
            detections = await self.postprocess_detections(
                results[0], 
                image.shape[:2]
            )
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += len(detections)
            self.total_images += 1
            
            # Keep only recent inference times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    async def batch_detect(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in multiple images simultaneously.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of detection lists, one per input image
        """
        if self.model is None:
            await self.load_model()
        
        if not images:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_results = await self._process_batch(batch_images)
            results.extend(batch_results)
        
        return results
    
    async def _process_batch(self, batch_images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Process a batch of images."""
        start_time = time.time()
        
        try:
            # Preprocess all images in batch
            preprocessed_batch = []
            original_shapes = []
            
            for image in batch_images:
                preprocessed = await self.preprocess_image(image)
                preprocessed_batch.append(preprocessed)
                original_shapes.append(image.shape[:2])
            
            # Run batch inference
            results = self.model.predict(
                source=preprocessed_batch,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False
            )
            
            # Postprocess all results
            batch_results = []
            for result, shape in zip(results, original_shapes):
                detections = await self.postprocess_detections(result, shape)
                batch_results.append(detections)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += sum(len(dets) for dets in batch_results)
            self.total_images += len(batch_images)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch YOLO detection: {e}")
            return [[] for _ in batch_images]
    
    async def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # YOLO preprocessing is handled internally by ultralytics
        # We just need to ensure the image is in the correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            if image.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        return image_rgb
    
    async def postprocess_detections(
        self, 
        raw_output: Any, 
        image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Postprocess raw model output to detections.
        
        Args:
            raw_output: Raw YOLO results
            image_shape: Original image shape (H, W)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Extract detection data
            boxes = raw_output.boxes
            
            if boxes is None or len(boxes) == 0:
                return detections
            
            # Get detection data
            xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            conf = boxes.conf.cpu().numpy()  # confidence
            cls = boxes.cls.cpu().numpy()    # class
            
            height, width = image_shape
            
            for i, (box, score, label) in enumerate(zip(xyxy, conf, cls)):
                # Filter by confidence threshold
                if score < self.confidence_threshold:
                    continue
                
                # Filter for person class only (class_id = 0 in COCO)
                if int(label) != self.person_class_id:
                    continue
                
                # Convert box coordinates
                x1, y1, x2, y2 = box
                
                # Clamp coordinates to image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Create detection dictionary
                detection = {
                    'bbox': {
                        'x': float(x1),
                        'y': float(y1),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    },
                    'confidence': float(score),
                    'class_id': int(label),
                    'class_name': self.coco_classes[int(label)] if int(label) < len(self.coco_classes) else 'unknown'
                }
                
                detections.append(detection)
                
                # Limit number of detections
                if len(detections) >= self.max_detections:
                    break
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return []
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': 'YOLO',
            'architecture': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'batch_size': self.batch_size,
            'input_size': self.input_size,
            'weights_path': self.weights_path,
            'loaded': self.is_loaded()
        }
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported detection classes."""
        return ['person']  # We only care about person detection
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for detections."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        
        # Update model parameters if loaded
        if self.model is not None:
            self.model.overrides['conf'] = self.confidence_threshold
            
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self.confidence_threshold
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'fps': 0.0,
                'total_detections': 0,
                'total_images': 0,
                'avg_detections_per_image': 0.0
            }
        
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        avg_detections_per_image = self.total_detections / self.total_images if self.total_images > 0 else 0.0
        
        return {
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'total_detections': self.total_detections,
            'total_images': self.total_images,
            'avg_detections_per_image': avg_detections_per_image,
            'recent_inference_times': self.inference_times[-10:]  # Last 10 times
        }
    
    async def warm_up(self) -> None:
        """Warm up the model with dummy inference."""
        if self.model is None:
            await self.load_model()
        
        logger.info("Warming up YOLO model...")
        
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run a few warm-up inferences
        for _ in range(3):
            await self.detect(dummy_image)
        
        logger.info("YOLO model warmed up successfully")
    
    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            self.model = None
            
        # Clear CUDA cache if using GPU
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("YOLO detector cleaned up")


# Register the detector
DetectorFactory.register_detector('yolo', YOLODetector)
DetectorFactory.register_detector('yolov8', YOLODetector)
DetectorFactory.register_detector('yolov8n', YOLODetector)
DetectorFactory.register_detector('yolov8s', YOLODetector)
DetectorFactory.register_detector('yolov8m', YOLODetector)
DetectorFactory.register_detector('yolov8l', YOLODetector)
DetectorFactory.register_detector('yolov8x', YOLODetector)