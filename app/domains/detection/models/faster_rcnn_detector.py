"""
Faster R-CNN detector implementation with GPU acceleration.
High-precision person detection using torchvision's pre-trained models.
"""
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import detection
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .base_detector import AbstractDetector, DetectionResult, DetectorFactory
from app.infrastructure.gpu import get_gpu_manager

logger = logging.getLogger(__name__)


class FasterRCNNDetector(AbstractDetector):
    """
    Faster R-CNN detector with GPU acceleration.
    Optimized for person detection with high accuracy.
    """
    
    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn_v2",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        batch_size: int = 4
    ):
        """
        Initialize Faster R-CNN detector.
        
        Args:
            model_name: Model architecture name
            confidence_threshold: Minimum confidence for detections
            device: Device for inference (auto-detected if None)
            weights_path: Path to custom weights (optional)
            nms_threshold: NMS threshold for duplicate removal
            max_detections: Maximum number of detections per image
            batch_size: Batch size for inference
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install PyTorch.")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.weights_path = weights_path
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.batch_size = batch_size
        
        # Model and device setup
        self.model: Optional[nn.Module] = None
        self.device = device or get_gpu_manager().get_optimal_device()
        self.torch_device = torch.device(self.device)
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.total_detections = 0
        self.total_images = 0
        
        # COCO class names (Faster R-CNN is trained on COCO)
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        
        # Person class ID in COCO
        self.person_class_id = 1
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        logger.info(f"FasterRCNNDetector initialized with device: {self.device}")
    
    async def load_model(self) -> None:
        """Load the Faster R-CNN model."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            # Load pre-trained model
            if self.model_name == "fasterrcnn_resnet50_fpn_v2":
                self.model = detection.fasterrcnn_resnet50_fpn_v2(
                    weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                )
            elif self.model_name == "fasterrcnn_resnet50_fpn":
                self.model = detection.fasterrcnn_resnet50_fpn(
                    weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                )
            elif self.model_name == "fasterrcnn_mobilenet_v3_large_fpn":
                self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(
                    weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Load custom weights if provided
            if self.weights_path:
                checkpoint = torch.load(self.weights_path, map_location=self.torch_device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded custom weights from {self.weights_path}")
            
            # Move model to device and set eval mode
            self.model.to(self.torch_device)
            self.model.eval()
            
            # Configure NMS threshold
            self.model.roi_heads.nms_thresh = self.nms_threshold
            
            logger.info(f"Faster R-CNN model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Faster R-CNN model: {e}")
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
            with torch.no_grad():
                predictions = self.model([preprocessed])
            
            # Postprocess results
            detections = await self.postprocess_detections(
                predictions[0], 
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
            logger.error(f"Error in Faster R-CNN detection: {e}")
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
            with torch.no_grad():
                predictions = self.model(preprocessed_batch)
            
            # Postprocess all results
            batch_results = []
            for pred, shape in zip(predictions, original_shapes):
                detections = await self.postprocess_detections(pred, shape)
                batch_results.append(detections)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += sum(len(dets) for dets in batch_results)
            self.total_images += len(batch_images)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch Faster R-CNN detection: {e}")
            return [[] for _ in batch_images]
    
    async def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for Faster R-CNN.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            if image.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # Apply transforms
        tensor = self.transform(image_rgb).to(self.torch_device)
        
        return tensor
    
    async def postprocess_detections(
        self, 
        raw_output: Dict[str, torch.Tensor], 
        image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Postprocess raw model output to detections.
        
        Args:
            raw_output: Raw model predictions
            image_shape: Original image shape (H, W)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            boxes = raw_output['boxes'].cpu().numpy()
            scores = raw_output['scores'].cpu().numpy()
            labels = raw_output['labels'].cpu().numpy()
            
            height, width = image_shape
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                # Filter by confidence threshold
                if score < self.confidence_threshold:
                    continue
                
                # Filter for person class only (class_id = 1 in COCO)
                if label != self.person_class_id:
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
                    'class_name': self.coco_classes[label] if label < len(self.coco_classes) else 'unknown'
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
            'name': 'Faster R-CNN',
            'architecture': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'batch_size': self.batch_size,
            'weights_path': self.weights_path,
            'loaded': self.is_loaded()
        }
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported detection classes."""
        return ['person']  # We only care about person detection
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for detections."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
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
        
        logger.info("Warming up Faster R-CNN model...")
        
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run a few warm-up inferences
        for _ in range(3):
            await self.detect(dummy_image)
        
        logger.info("Faster R-CNN model warmed up successfully")
    
    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            self.model = None
            
        # Clear CUDA cache if using GPU
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Faster R-CNN detector cleaned up")


# Register the detector
DetectorFactory.register_detector('fasterrcnn', FasterRCNNDetector)
DetectorFactory.register_detector('faster_rcnn', FasterRCNNDetector)
DetectorFactory.register_detector('fasterrcnn_resnet50_fpn_v2', FasterRCNNDetector)