"""
YOLO detector implementation for the legacy models system.
Migrated from RT-DETR to YOLO11-L for improved performance.
"""

import logging
import asyncio
from typing import List, Optional, Tuple
import numpy as np

try:
    import torch
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    TORCH_AVAILABLE = False

from .base_models import AbstractDetector, Detection, BoundingBox

logger = logging.getLogger(__name__)

# Dedicated file logger for [SPEED_DEBUG] timing logs ONLY
from pathlib import Path

class SpeedDebugFilter(logging.Filter):
    """Filter to only allow [SPEED_DEBUG] messages"""
    def filter(self, record):
        return '[SPEED_DEBUG]' in record.getMessage()

speed_debug_logger = logging.getLogger("speed_debug_yolo")
speed_debug_logger.setLevel(logging.INFO)
_speed_log_path = Path("speed_debug.log")
_speed_file_handler = logging.FileHandler(_speed_log_path, mode='a', encoding='utf-8')
_speed_file_handler.setLevel(logging.INFO)
_speed_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
_speed_file_handler.addFilter(SpeedDebugFilter())  # Only [SPEED_DEBUG] messages
speed_debug_logger.addHandler(_speed_file_handler)
speed_debug_logger.propagate = True  # Also show in console



def fix_yolo_serialization():
    """
    Workaround for 'AttributeError: Can't get attribute 'PatchedC3k2' on <module '__main__'...'.
    This happens when loading certain YOLO checkpoint versions (like YOLO11/26).
    We map the missing class to the standard C3k2 block in ultralytics.
    """
    try:
        import sys
        from ultralytics.nn.modules import block
        
        # Check if C3k2 exists (YOLO11 specific)
        if hasattr(block, 'C3k2'):
            # Map __main__.PatchedC3k2 to real C3k2
            # Use 'app.main' if running via uvicorn/fastapi, or '__main__' for scripts
            # The error log showed <module '__mp_main__' ...> or similar, often mapping to main scope.
            # We'll try to set it on sys.modules['__main__'] and maybe others if needed.
            
            # Define the mapping of patched names to real classes
            # We add PatchedSPPF based on the latest error report
            mapping = {
                'PatchedC3k2': 'C3k2',
                'PatchedSPPF': 'SPPF',
                'PatchedConv': 'Conv', # Pre-emptive add: common in some exports
                'PatchedBottleneck': 'Bottleneck', # Pre-emptive add
            }

            target_modules = ['__main__', 'app.main']
            patched = False
            
            for mod_name in target_modules:
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    for patched_name, real_name in mapping.items():
                        if not hasattr(mod, patched_name) and hasattr(block, real_name):
                            setattr(mod, patched_name, getattr(block, real_name))
                            patched = True
            
            if patched:
                logger.debug("🔧 Applied serialization fix: Mapped patched classes to ultralytics.nn.modules.block")
                
    except Exception as e:
        logger.warning(f"⚠️ Could not apply serialization fix: {e}")


class YOLODetector(AbstractDetector):
    """
    YOLO detector implementation using Ultralytics.
    Uses YOLO11-L model for person detection with real-time performance.
    """
    
    def __init__(self, model_name: str = "yolo26m.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: Model weights filename (default: yolo11l.pt)
            confidence_threshold: Minimum confidence for detections
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install PyTorch.")
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Please install ultralytics.")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_loaded_flag = False
        
        # Detect if this is a TensorRT engine or ONNX model
        self.is_tensorrt = model_name.endswith('.engine')
        self.is_onnx = model_name.endswith('.onnx')
        
        # Performance optimization settings (configurable via env vars)
        import os
        self.imgsz = int(os.environ.get("DETECTION_IMGSZ", "640"))
        # TensorRT engines have precision baked in, so half is not needed
        # For PyTorch models, use half precision on CUDA for speed
        if self.is_tensorrt or self.is_onnx:
            self.use_half = False  # TensorRT/ONNX engine has precision already set/handled
        else:
            self.use_half = os.environ.get("DETECTION_HALF", "true").lower() == "true" and self.device.type == "cuda"
        
        if self.is_tensorrt:
            model_type = "TensorRT"
        elif self.is_onnx:
            model_type = "ONNX"
        else:
            model_type = "PyTorch"
            
        logger.info(f"YOLODetector configured: type={model_type}, imgsz={self.imgsz}, half={self.use_half}, device={self.device}")
        
        # COCO class names (YOLO is trained on COCO)
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
    
    async def load_model(self):
        """
        Load the YOLO model (PyTorch .pt or TensorRT .engine).
        """
        if self._model_loaded_flag and self.model is not None:
            return
        
        try:
            if self.is_tensorrt:
                model_type = "TensorRT engine"
            elif self.is_onnx:
                model_type = "ONNX model"
            else:
                model_type = "PyTorch model"
            logger.info(f"Loading YOLO {model_type}: {self.model_name} on device: {self.device}...")
            
            # Ultralytics YOLO automatically handles .engine files
            # For TensorRT engines, explicitly specify task='detect' to avoid auto-detection warning
            
            # CRITICAL CHECK: Ensure model file exists before attempting to load
            import os
            import sys
            if not os.path.exists(self.model_name):
                logger.critical(f"❌ FATAL: Model file not found at: {self.model_name}")
                logger.critical("   The container cannot start without the required model weights.")
                logger.critical("   Please ensure 'yolo26m.pt' (or configured model) is in the weights directory.")
                sys.exit(1)

            fix_yolo_serialization()
            if self.is_tensorrt or self.is_onnx:
                self.model = YOLO(self.model_name, task='detect')
            else:
                self.model = YOLO(self.model_name)
            
            # For PyTorch models, explicitly set device
            # TensorRT engines are already device-optimized during export
            if not self.is_tensorrt:
                if self.device.type == 'cuda' and torch.cuda.is_available():
                    self.model.to('cuda')
                else:
                    self.model.to('cpu')
            
            self._model_loaded_flag = True
            logger.info(f"✅ YOLO {model_type} loaded successfully!")
            if self.is_tensorrt:
                logger.info("🚀 TensorRT optimization active - expect 3-5x faster inference")
            elif self.is_onnx:
                logger.info("🚀 ONNX optimization active - optimized for CPU inference")
            
        except Exception as e:
            logger.exception(f"Error loading YOLO model: {e}")
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
        
        try:
            dummy_np_image = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            _ = await self.detect(dummy_np_image)
            logger.info("YOLO detector warmup successful.")
        except Exception as e:
            logger.error(f"YOLO detector warmup failed: {e}", exc_info=True)
    
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
        
        import time as _time
        _total_start = _time.perf_counter()
        
        try:
            # Run inference without tracking gradients and offload to thread to avoid blocking event loop
            import asyncio
            import torch

            _thread_start = _time.perf_counter()
            
            def _infer():
                _infer_start = _time.perf_counter()
                with torch.inference_mode():
                    result = self.model(image, conf=self.confidence_threshold, imgsz=self.imgsz, half=self.use_half, verbose=False)
                _infer_time = (_time.perf_counter() - _infer_start) * 1000
                return result, _infer_time

            results, _gpu_infer_time = await asyncio.to_thread(_infer)
            _thread_time = (_time.perf_counter() - _thread_start) * 1000
            
            # Process results
            _parse_start = _time.perf_counter()
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
            
            _parse_time = (_time.perf_counter() - _parse_start) * 1000
            _total_time = (_time.perf_counter() - _total_start) * 1000
            
            # Log detailed timing to file and console
            speed_debug_logger.info(
                "[SPEED_DEBUG] YOLO.detect | Total=%.1fms | Thread=%.1fms | Inference=%.1fms | Parse=%.1fms | Dets=%d | ImgShape=%s",
                _total_time, _thread_time, _gpu_infer_time, _parse_time, len(detections_result), image.shape[:2]
            )
            
            return detections_result
            
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
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
            import time as _time
            
            _batch_total_start = _time.perf_counter()

            def _batch_infer():
                _infer_start = _time.perf_counter()
                with torch.inference_mode():
                    # Ultralytics supports list of images for batch inference
                    result = self.model(images, conf=self.confidence_threshold, imgsz=self.imgsz, half=self.use_half, verbose=False)
                _infer_time = (_time.perf_counter() - _infer_start) * 1000
                return result, _infer_time

            _thread_start = _time.perf_counter()
            results, _gpu_infer_time = await asyncio.to_thread(_batch_infer)
            _thread_time = (_time.perf_counter() - _thread_start) * 1000
            
            # Parse each result
            _parse_start = _time.perf_counter()
            batch_detections: List[List[Detection]] = []
            for i, (result, image) in enumerate(zip(results, images)):
                try:
                    detections = self._parse_single_result(result, image)
                    batch_detections.append(detections)
                except Exception as e:
                    logger.error(f"Error parsing batch result {i}: {e}")
                    batch_detections.append([])
            _parse_time = (_time.perf_counter() - _parse_start) * 1000
            
            _batch_total_time = (_time.perf_counter() - _batch_total_start) * 1000
            
            # Detailed timing log to file and console
            total_dets = sum(len(d) for d in batch_detections)
            model_type = "TRT" if self.is_tensorrt else ("ONNX" if self.is_onnx else "PT")
            speed_debug_logger.info(
                "[SPEED_DEBUG] YOLO.detect_batch | Type=%s | BatchSize=%d | Total=%.1fms | Thread=%.1fms | Inference=%.1fms | Parse=%.1fms | TotalDets=%d",
                model_type, len(images), _batch_total_time, _thread_time, _gpu_infer_time, _parse_time, total_dets
            )
            
            return batch_detections
            
        except Exception as e:
            logger.error(f"Error during YOLO batch detection: {e}")
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


# Backward compatibility alias
RTDETRDetector = YOLODetector
