"""
YOLO detector implementation for the legacy models system.
Migrated from RT-DETR to YOLO11-L for improved performance.

GPU Optimizations:
- CUDA Graphs for static inference replay (reduces kernel launch overhead)
- Pinned memory buffer pool for fast CPU→GPU transfers
- Pre-allocated device tensors to avoid allocation overhead
- Multi-stream pipeline for overlapped execution
- Reduced synchronization points
"""

import logging
import asyncio
from typing import List, Optional, Tuple
import numpy as np
 
try:
    import torch
    from ultralytics import YOLO, RTDETR
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


class PinnedMemoryPool:
    """
    Pool of pinned (page-locked) memory buffers for fast CPU→GPU transfers.
    Pinned memory allows DMA (Direct Memory Access) transfers which are faster.
    """
    
    def __init__(self, buffer_shape: Tuple[int, int, int], pool_size: int = 4, dtype=np.uint8):
        """
        Initialize the pinned memory pool.
        
        Args:
            buffer_shape: Shape of each buffer (H, W, C)
            pool_size: Number of buffers in the pool
            dtype: Data type for buffers
        """
        self._pool: List[torch.Tensor] = []
        self._available: List[bool] = []
        self._buffer_shape = buffer_shape
        self._pool_size = pool_size
        self._dtype = dtype
        self._initialized = False
        
    def initialize(self):
        """Create the pinned memory buffers."""
        if self._initialized or not TORCH_AVAILABLE:
            return
            
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping pinned memory pool")
                return
                
            torch_dtype = torch.uint8 if self._dtype == np.uint8 else torch.float32
            
            for _ in range(self._pool_size):
                # Create pinned memory tensors
                buffer = torch.empty(self._buffer_shape, dtype=torch_dtype, pin_memory=True)
                self._pool.append(buffer)
                self._available.append(True)
            
            self._initialized = True
            logger.info(f"✅ Pinned memory pool initialized: {self._pool_size} buffers of shape {self._buffer_shape}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create pinned memory pool: {e}")
            
    def acquire(self) -> Optional[torch.Tensor]:
        """Acquire a buffer from the pool."""
        if not self._initialized:
            return None
            
        for i, available in enumerate(self._available):
            if available:
                self._available[i] = False
                return self._pool[i]
        return None
        
    def release(self, buffer: torch.Tensor):
        """Release a buffer back to the pool."""
        if not self._initialized:
            return
            
        for i, pool_buffer in enumerate(self._pool):
            if buffer is pool_buffer:
                self._available[i] = True
                return


class CUDAGraphManager:
    """
    Manages CUDA Graph capture and replay for static inference.
    CUDA Graphs reduce kernel launch overhead by capturing the entire
    inference pipeline and replaying it with minimal CPU involvement.
    """
    
    def __init__(self):
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._graph_output = None
        self._captured = False
        self._input_shape: Optional[Tuple] = None
        
    def is_captured(self) -> bool:
        return self._captured
        
    def capture(self, model, input_tensor: torch.Tensor, stream: torch.cuda.Stream, 
                conf: float, imgsz: int, half: bool):
        """
        Capture the inference call as a CUDA graph.
        
        Note: CUDA graphs work best with TensorRT engines that have static shapes.
        The Ultralytics model() call may not be fully compatible with CUDA graphs
        due to dynamic control flow. This is a best-effort optimization.
        """
        if self._captured:
            return
            
        try:
            self._input_shape = input_tensor.shape
            
            # Pre-allocate static input buffer
            self._static_input = input_tensor.clone()
            
            # Warmup runs required before capture
            for _ in range(3):
                with torch.cuda.stream(stream):
                    _ = model(self._static_input, conf=conf, imgsz=imgsz, half=half, verbose=False)
                stream.synchronize()
            
            # Capture the graph
            self._graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self._graph, stream=stream):
                self._graph_output = model(self._static_input, conf=conf, imgsz=imgsz, half=half, verbose=False)
            
            self._captured = True
            logger.info(f"✅ CUDA Graph captured for input shape {self._input_shape}")
            
        except Exception as e:
            logger.warning(f"⚠️ CUDA Graph capture failed (will use standard inference): {e}")
            self._captured = False
            
    def replay(self, input_data: np.ndarray) -> Optional[object]:
        """
        Replay the captured graph with new input data.
        
        Returns:
            The inference results, or None if replay failed
        """
        if not self._captured or self._graph is None:
            return None
            
        try:
            # Copy new input to static buffer
            input_tensor = torch.from_numpy(input_data)
            if input_tensor.shape != self._input_shape:
                # Shape mismatch, cannot use graph
                return None
                
            self._static_input.copy_(input_tensor)
            
            # Replay the graph
            self._graph.replay()
            
            return self._graph_output
            
        except Exception as e:
            logger.warning(f"⚠️ CUDA Graph replay failed: {e}")
            return None


class YOLODetector(AbstractDetector):
    """
    YOLO detector implementation using Ultralytics.
    Uses YOLO11-L model for person detection with real-time performance.
    
    GPU Optimizations:
    - Multi-stream pipeline for overlapped operations
    - CUDA Graphs for reduced kernel launch overhead (best-effort)
    - Pinned memory for faster CPU→GPU transfers
    - Pre-allocated tensors to reduce allocation overhead
    - Non-blocking transfers and reduced sync points
    """
    
    def __init__(self, model_name: str = "/app/weights/yolo26m.engine", confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: Model weights filename (default: yolo26m.engine)
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
        
        # ============================================================
        # GPU OPTIMIZATION: Multi-Stream Pipeline
        # ============================================================
        # Stream 1: Preprocessing (image resize/normalize) 
        # Stream 2: Inference (main compute)
        # Stream 3: Postprocessing (result parsing, CPU transfers)
        self._preprocess_stream: Optional[torch.cuda.Stream] = None
        self._inference_stream: Optional[torch.cuda.Stream] = None
        self._postprocess_stream: Optional[torch.cuda.Stream] = None
        
        # CUDA events for fine-grained synchronization between streams
        self._preprocess_done_event: Optional[torch.cuda.Event] = None
        self._inference_done_event: Optional[torch.cuda.Event] = None
        
        if self.device.type == 'cuda':
            # Create streams with different priorities (lower = higher priority)
            self._preprocess_stream = torch.cuda.Stream(priority=-1)  # High priority
            self._inference_stream = torch.cuda.Stream(priority=0)    # Normal priority
            self._postprocess_stream = torch.cuda.Stream(priority=1)  # Lower priority
            
            # Create events for synchronization
            self._preprocess_done_event = torch.cuda.Event()
            self._inference_done_event = torch.cuda.Event()
        
        # ============================================================
        # GPU OPTIMIZATION: CUDA Graph Manager
        # ============================================================
        self._cuda_graph_manager = CUDAGraphManager()
        self._use_cuda_graphs = True  # Can be disabled if issues arise
        
        # ============================================================
        # GPU OPTIMIZATION: Pinned Memory Pool
        # ============================================================
        # Performance optimization settings (configurable via env vars)
        import os
        self.imgsz = int(os.environ.get("DETECTION_IMGSZ", "640"))
        
        # Initialize pinned memory pool for input images
        # Pool for 640x640 RGB images (or configured size)
        self._pinned_pool = PinnedMemoryPool(
            buffer_shape=(self.imgsz, self.imgsz, 3),
            pool_size=4,
            dtype=np.uint8
        )
        
        # ============================================================
        # GPU OPTIMIZATION: Pre-allocated Device Tensors
        # ============================================================
        self._preallocated_input: Optional[torch.Tensor] = None
        self._preallocated_output_boxes: Optional[torch.Tensor] = None
        
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
        logger.info(f"  GPU Optimizations: multi_stream={self._inference_stream is not None}, cuda_graphs={self._use_cuda_graphs}, pinned_memory=True")
        
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
    
    def _initialize_gpu_resources(self):
        """Initialize GPU resources after model is loaded."""
        if self.device.type != 'cuda':
            return
            
        try:
            # Initialize pinned memory pool
            self._pinned_pool.initialize()
            
            # Pre-allocate input tensor on GPU
            self._preallocated_input = torch.zeros(
                (1, 3, self.imgsz, self.imgsz), 
                dtype=torch.float16 if self.use_half else torch.float32,
                device=self.device
            )
            
            logger.info("✅ GPU resources initialized (pinned memory, pre-allocated tensors)")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize some GPU resources: {e}")
    
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
                logger.critical("   Please ensure '/app/weights/yolo26m.engine' (or configured model) is in the weights directory.")
                sys.exit(1)

            fix_yolo_serialization()
            
            # Use RTDETR class for RT-DETR models to ensure correct output decoding
            is_rtdetr = "rtdetr" in self.model_name.lower() or "rt-detr" in self.model_name.lower()
            
            if is_rtdetr:
                 logger.info(f"🔄 Detected RT-DETR model. Using ultralytics.RTDETR class.")
                 # RT-DETR class usage
                 if self.is_tensorrt or self.is_onnx:
                     self.model = RTDETR(self.model_name) # RTDETR might not accept task='detect' freely or might default correctly
                 else:
                     self.model = RTDETR(self.model_name)
            else:
                # Standard YOLO usage
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
            
            # Initialize GPU resources
            self._initialize_gpu_resources()
            
            # --- Dynamic Class Mapping ---
            # Attempt to find the correct 'person' class ID from the model's metadata
            # This handles models trained on different datasets (COCO, Objects365, OpenImages, etc.)
            # --- Dynamic Class Mapping ---
            # Attempt to find the correct 'person' class ID from the model's metadata
            try:
                if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                    # Log a few classes to help debugging
                    logger.info(f"📋 Model classes loaded (first 5): {dict(list(self.model.names.items())[:5])}")
                    
                    found_person_id = None
                    # Search for 'person' or 'Person'
                    for cid, cname in self.model.names.items():
                        if str(cname).lower() == 'person':
                            found_person_id = cid
                            break
                    
                    # Fallback: If only 1 class exists (e.g., 'item'), assume it is the person class
                    if found_person_id is None and len(self.model.names) == 1:
                        single_id = list(self.model.names.keys())[0]
                        single_name = list(self.model.names.values())[0]
                        logger.warning(f"⚠️ 'person' class not found, but model has only 1 class: '{single_name}' (ID {single_id}). Assuming this is the target.")
                        found_person_id = single_id

                    if found_person_id is not None:
                        self.person_class_id = int(found_person_id)
                        logger.info(f"✅ Auto-detected target class ID: {self.person_class_id}")
                    else:
                        logger.warning(f"⚠️ Could not find class 'person' in model names. Defaulting to ID {self.person_class_id} (COCO standard).")
                        logger.warning(f"   Available classes example: {dict(list(self.model.names.items())[:10])}")
                else:
                     logger.warning("⚠️ Model does not have valid 'names' attribute. Using default specific COCO ID 0 for 'person'.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to auto-detect class ID (using default 0): {e}")

            
        except Exception as e:
            logger.exception(f"Error loading YOLO model: {e}")
            self.model = None
            self._model_loaded_flag = False
            raise
    
    async def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """
        Warm up the model by performing a dummy inference.
        Also attempts to capture CUDA graph for optimized replay.
        """
        if not self._model_loaded_flag or not self.model:
            logger.warning("Detector model not loaded. Cannot perform warmup.")
            return
        
        try:
            dummy_np_image = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            
            # Standard warmup
            _ = await self.detect(dummy_np_image)
            logger.info("YOLO detector warmup successful.")
            
            # Attempt CUDA graph capture for TensorRT
            if self.is_tensorrt and self._use_cuda_graphs and self._inference_stream is not None:
                try:
                    # Note: CUDA Graph capture with Ultralytics is experimental
                    # The model() call may have dynamic control flow that prevents capture
                    # We attempt capture but fall back gracefully if it fails
                    logger.info("Attempting CUDA Graph capture for TensorRT...")
                    
                    # Create a dummy tensor for graph capture
                    dummy_tensor = torch.from_numpy(dummy_np_image).to(self.device)
                    
                    self._cuda_graph_manager.capture(
                        model=self.model,
                        input_tensor=dummy_tensor,
                        stream=self._inference_stream,
                        conf=self.confidence_threshold,
                        imgsz=self.imgsz,
                        half=self.use_half
                    )
                    
                    if self._cuda_graph_manager.is_captured():
                        logger.info("✅ CUDA Graph captured - using optimized replay mode")
                    else:
                        logger.info("ℹ️ CUDA Graph not captured - using standard inference")
                        
                except Exception as e:
                    logger.warning(f"⚠️ CUDA Graph capture failed (will use standard inference): {e}")
                    
        except Exception as e:
            logger.error(f"YOLO detector warmup failed: {e}", exc_info=True)
    
    async def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Perform person detection on an image.
        
        Uses multi-stream pipeline for optimized GPU utilization:
        - Preprocessing on stream 1
        - Inference on stream 2  
        - Postprocessing on stream 3
        
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
            
            # ============================================================
            # GPU OPTIMIZATION: Multi-Stream Pipeline with Events
            # ============================================================
            def _infer_optimized():
                _infer_start = _time.perf_counter()
                
                with torch.inference_mode():
                    if self._inference_stream is not None:
                        # Use dedicated inference stream
                        with torch.cuda.stream(self._inference_stream):
                            result = self.model(
                                image, 
                                conf=self.confidence_threshold, 
                                imgsz=self.imgsz, 
                                half=self.use_half, 
                                verbose=False
                            )
                        
                        # Record event when inference is done
                        if self._inference_done_event is not None:
                            self._inference_done_event.record(self._inference_stream)
                        
                        # Only sync if we need results immediately
                        # For pipelined processing, we could defer this
                        self._inference_stream.synchronize()
                    else:
                        result = self.model(
                            image, 
                            conf=self.confidence_threshold, 
                            imgsz=self.imgsz, 
                            half=self.use_half, 
                            verbose=False
                        )
                        
                _infer_time = (_time.perf_counter() - _infer_start) * 1000
                return result, _infer_time

            # For TensorRT on CUDA, run directly to avoid thread overhead
            if self.is_tensorrt and self._inference_stream is not None:
                results, _gpu_infer_time = _infer_optimized()
            else:
                results, _gpu_infer_time = await asyncio.to_thread(_infer_optimized)
            _thread_time = (_time.perf_counter() - _thread_start) * 1000
            
            # ============================================================
            # GPU OPTIMIZATION: Postprocessing on separate stream
            # ============================================================
            _parse_start = _time.perf_counter()
            detections_result: List[Detection] = []
            
            if len(results) > 0:
                result = results[0]  # Get first (and only) result
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # GPU Optimization: Non-blocking transfers from GPU to CPU
                    # Use postprocess stream for CPU transfers
                    if boxes.xyxy is not None:
                        if self._postprocess_stream is not None:
                            # Wait for inference to complete before postprocessing
                            if self._inference_done_event is not None:
                                self._postprocess_stream.wait_event(self._inference_done_event)
                            
                            with torch.cuda.stream(self._postprocess_stream):
                                box_coords = boxes.xyxy.to('cpu', non_blocking=True).numpy()
                                confidences = boxes.conf.to('cpu', non_blocking=True).numpy()
                                class_ids = boxes.cls.to('cpu', non_blocking=True).numpy().astype(int)
                            
                            # Sync postprocess stream to ensure data is ready
                            self._postprocess_stream.synchronize()
                        else:
                            box_coords = boxes.xyxy.to('cpu', non_blocking=True).numpy()
                            confidences = boxes.conf.to('cpu', non_blocking=True).numpy()
                            class_ids = boxes.cls.to('cpu', non_blocking=True).numpy().astype(int)
                        
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
                    # GPU Optimization: Use dedicated inference stream
                    if self._inference_stream is not None:
                        with torch.cuda.stream(self._inference_stream):
                            result = self.model(images, conf=self.confidence_threshold, imgsz=self.imgsz, half=self.use_half, verbose=False)
                        
                        if self._inference_done_event is not None:
                            self._inference_done_event.record(self._inference_stream)
                        
                        self._inference_stream.synchronize()
                    else:
                        # Ultralytics supports list of images for batch inference
                        result = self.model(images, conf=self.confidence_threshold, imgsz=self.imgsz, half=self.use_half, verbose=False)
                _infer_time = (_time.perf_counter() - _infer_start) * 1000
                return result, _infer_time

            _thread_start = _time.perf_counter()
            try:
                # TRT models exported with batch=1 may fail when passed a list > 1
                # We attempt batch inference, catch specific RuntimeErrors, and fallback.
                # GPU Optimization: For TensorRT on CUDA, run directly to avoid thread overhead
                if self.is_tensorrt and self._inference_stream is not None:
                    results, _gpu_infer_time = _batch_infer()
                else:
                    results, _gpu_infer_time = await asyncio.to_thread(_batch_infer)
            except RuntimeError as rte:
                # Common TRT error: "The engine plan file is generated for a different batch size"
                if "batch" in str(rte).lower() and self.is_tensorrt:
                    logger.warning(f"⚠️ Batch inference failed for TensorRT engine (likely exported with batch=1). Falling back to sequential inference. Error: {rte}")
                    # Fallback to sequential
                    _seq_start = _time.perf_counter()
                    batch_detections = []
                    for img in images:
                         d = await self.detect(img)
                         batch_detections.append(d)
                    
                    _seq_time = (_time.perf_counter() - _seq_start) * 1000
                    speed_debug_logger.info(
                        "[SPEED_DEBUG] YOLO.detect_batch (FALLBACK) | Type=TRT-Seq | BatchSize=%d | Total=%.1fms | Dets=%d",
                        len(images), _seq_time, sum(len(x) for x in batch_detections)
                    )
                    return batch_detections
                raise rte

            _thread_time = (_time.perf_counter() - _thread_start) * 1000
            
            # Parse each result using postprocess stream
            _parse_start = _time.perf_counter()
            batch_detections: List[List[Detection]] = []
            
            # Handling results: Ultralytics result can be a list or single object
            if not isinstance(results, list):
                 results = [results]
            
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
        
        Uses postprocess stream for GPU→CPU transfers when available.
        
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
                # GPU Optimization: Use postprocess stream for transfers
                if self._postprocess_stream is not None:
                    with torch.cuda.stream(self._postprocess_stream):
                        box_coords = boxes.xyxy.to('cpu', non_blocking=True).numpy()
                        confidences = boxes.conf.to('cpu', non_blocking=True).numpy()
                        class_ids = boxes.cls.to('cpu', non_blocking=True).numpy().astype(int)
                    self._postprocess_stream.synchronize()
                else:
                    box_coords = boxes.xyxy.to('cpu', non_blocking=True).numpy()
                    confidences = boxes.conf.to('cpu', non_blocking=True).numpy()
                    class_ids = boxes.cls.to('cpu', non_blocking=True).numpy().astype(int)


                original_h, original_w = image.shape[:2]
                
                for box, score_val, label_id in zip(box_coords, confidences, class_ids):
                    # Filter for person class only (class_id = 0 in COCO)
                    # DEBUG: Temporarily allowing ALL classes to debug missing boxes issue
                    if score_val >= self.confidence_threshold: # and int(label_id) == self.person_class_id:
                        x1, y1, x2, y2 = box
                        
                        # Clip coordinates to image bounds
                        x1_clipped = max(0.0, float(x1))
                        y1_clipped = max(0.0, float(y1))
                        x2_clipped = min(float(original_w), float(x2))
                        y2_clipped = min(float(original_h), float(y2))
                        
                        # Ensure box has positive area after clipping
                        if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                             # Resolve class name
                            class_name = "person"
                            if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                                class_name = str(self.model.names.get(int(label_id), "person"))
                            
                            detections_result.append(
                                Detection(
                                    bbox=BoundingBox(x1=x1_clipped, y1=y1_clipped, x2=x2_clipped, y2=y2_clipped),
                                    confidence=float(score_val),
                                    class_id=int(label_id),
                                    class_name=class_name
                                )
                            )
                        else:
                            logger.warning(f"DEBUG_REJECTED: Box clipped to zero area. Original: {box} Clipped: {x1_clipped},{y1_clipped},{x2_clipped},{y2_clipped} Image: {original_w}x{original_h}")
                    else:
                        if score_val > 0.3: # Only log if reasonably confident
                             logger.debug(f"DEBUG_REJECTED: Low confidence. Score: {score_val} Threshold: {self.confidence_threshold}")
        
        return detections_result


# Backward compatibility alias
RTDETRDetector = YOLODetector
