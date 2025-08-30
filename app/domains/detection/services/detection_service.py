"""
Detection service for multi-view person detection.

Provides business logic for:
- Person detection orchestration
- Multi-camera detection coordination
- Detection validation and filtering
- Performance optimization
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timezone
import time
import numpy as np

from app.domains.detection.entities.detection import Detection, DetectionBatch, FrameMetadata, BoundingBox, DetectionClass
from app.domains.detection.models.base_detector import AbstractDetector
from app.domains.detection.models import DetectorFactory
from app.infrastructure.gpu import get_gpu_manager
from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class DetectionService:
    """
    Enhanced service for person detection operations.
    
    Features:
    - GPU-accelerated detection with automatic device selection
    - Batch processing for improved performance
    - Multiple detector support (Faster R-CNN, YOLO)
    - Advanced detection filtering and validation
    - Performance monitoring and optimization
    """
    
    def __init__(
        self, 
        detector: Optional[AbstractDetector] = None,
        detector_type: str = "yolo",
        batch_size: int = 8,
        enable_gpu: bool = True
    ):
        """
        Initialize detection service.
        
        Args:
            detector: Pre-configured detector instance
            detector_type: Type of detector to create if detector is None
            batch_size: Batch size for processing multiple frames
            enable_gpu: Whether to use GPU acceleration
        """
        self.batch_size = batch_size
        self.enable_gpu = enable_gpu
        
        # GPU manager for resource allocation (must be initialized before detector creation)
        self.gpu_manager = get_gpu_manager()
        
        # Initialize detector
        if detector is None:
            self.detector = self._create_detector(detector_type)
        else:
            self.detector = detector
        
        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "person_detections": 0,
            "processing_times": [],
            "batch_processing_times": [],
            "error_count": 0,
            "gpu_enabled": enable_gpu,
            "batch_size": batch_size
        }
        
        logger.info(f"DetectionService initialized with detector: {detector_type}, GPU: {enable_gpu}")
    
    def _create_detector(self, detector_type: str) -> AbstractDetector:
        """Create and configure a detector instance."""
        try:
            # Get optimal device
            device = self.gpu_manager.get_optimal_device() if self.enable_gpu else "cpu"
            
            # Create detector with appropriate configuration
            if detector_type.lower() in ["yolo", "yolov8"]:
                detector = DetectorFactory.create_detector(
                    "yolo",
                    device=device,
                    batch_size=self.batch_size,
                    confidence_threshold=0.5
                )
            elif detector_type.lower() in ["fasterrcnn", "faster_rcnn"]:
                detector = DetectorFactory.create_detector(
                    "fasterrcnn",
                    device=device,
                    batch_size=self.batch_size,
                    confidence_threshold=0.5
                )
            else:
                logger.warning(f"Unknown detector type: {detector_type}, defaulting to YOLO")
                detector = DetectorFactory.create_detector(
                    "yolo",
                    device=device,
                    batch_size=self.batch_size,
                    confidence_threshold=0.5
                )
            
            return detector
            
        except Exception as e:
            logger.error(f"Error creating detector: {e}")
            raise
    
    async def initialize_detector(self):
        """Initialize and warm up the detector."""
        try:
            await self.detector.load_model()
            await self.detector.warm_up()
            logger.info("Detector initialized and warmed up successfully")
        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            raise
    
    async def detect_persons_in_frame(
        self,
        frame_data: Dict[str, Any],
        camera_id: CameraID,
        frame_index: int,
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Detect persons in a single frame.
        
        Args:
            frame_data: Frame image data
            camera_id: Camera identifier
            frame_index: Frame sequence number
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of person detections
        """
        start_time = time.time()
        
        try:
            # Extract frame metadata
            frame_metadata = self._extract_frame_metadata(frame_data, camera_id, frame_index)
            
            # Run detection
            raw_detections = await self.detector.detect(frame_data["image"])
            
            # Filter and convert to domain objects
            detections = self._process_raw_detections(
                raw_detections,
                camera_id,
                frame_index,
                frame_metadata,
                confidence_threshold
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(detections, processing_time)
            
            logger.debug(f"Detected {len(detections)} persons in frame {frame_index} from camera {camera_id}")
            return detections
            
        except Exception as e:
            self.detection_stats["error_count"] += 1
            logger.error(f"Error detecting persons in frame {frame_index} from camera {camera_id}: {e}")
            raise
    
    async def detect_persons_in_batch(
        self,
        frame_batch: Dict[CameraID, Dict[str, Any]],
        frame_index: int,
        confidence_threshold: float = 0.5,
        use_gpu_batch: bool = True
    ) -> DetectionBatch:
        """
        Detect persons in a batch of frames from multiple cameras.
        
        Args:
            frame_batch: Dictionary of frame data by camera ID
            frame_index: Frame sequence number
            confidence_threshold: Minimum confidence for detections
            use_gpu_batch: Whether to use GPU batch processing
            
        Returns:
            Detection batch with all detections
        """
        start_time = time.time()
        
        try:
            if use_gpu_batch and len(frame_batch) > 1:
                # Use GPU batch processing for better performance
                all_detections = await self._gpu_batch_detect(
                    frame_batch, 
                    frame_index, 
                    confidence_threshold
                )
            else:
                # Fall back to individual frame processing
                detection_tasks = [
                    self.detect_persons_in_frame(
                        frame_data,
                        camera_id,
                        frame_index,
                        confidence_threshold
                    )
                    for camera_id, frame_data in frame_batch.items()
                ]
                
                camera_detections = await asyncio.gather(*detection_tasks)
                
                # Flatten detections from all cameras
                all_detections = []
                for detections in camera_detections:
                    all_detections.extend(detections)
            
            # Create batch metadata (use first camera's metadata as reference)
            first_camera_id = next(iter(frame_batch.keys()))
            first_frame_data = frame_batch[first_camera_id]
            frame_metadata = self._extract_frame_metadata(
                first_frame_data,
                first_camera_id,
                frame_index
            )
            
            processing_time = time.time() - start_time
            
            # Update batch processing statistics
            self.detection_stats["batch_processing_times"].append(processing_time)
            if len(self.detection_stats["batch_processing_times"]) > 100:
                self.detection_stats["batch_processing_times"] = self.detection_stats["batch_processing_times"][-100:]
            
            detection_batch = DetectionBatch(
                detections=all_detections,
                frame_metadata=frame_metadata,
                processing_time=processing_time
            )
            
            logger.info(
                f"Batch detection completed: {len(all_detections)} detections "
                f"from {len(frame_batch)} cameras in {processing_time:.3f}s"
            )
            
            return detection_batch
            
        except Exception as e:
            self.detection_stats["error_count"] += 1
            logger.error(f"Error in batch detection for frame {frame_index}: {e}")
            raise
    
    async def _gpu_batch_detect(
        self,
        frame_batch: Dict[CameraID, Dict[str, Any]],
        frame_index: int,
        confidence_threshold: float
    ) -> List[Detection]:
        """
        Perform GPU-accelerated batch detection.
        
        Args:
            frame_batch: Dictionary of frame data by camera ID
            frame_index: Frame sequence number
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of all detections from all cameras
        """
        try:
            # Prepare batch data
            images = []
            camera_ids = []
            frame_metadatas = []
            
            for camera_id, frame_data in frame_batch.items():
                images.append(frame_data["image"])
                camera_ids.append(camera_id)
                frame_metadatas.append(
                    self._extract_frame_metadata(frame_data, camera_id, frame_index)
                )
            
            # Run batch detection
            batch_results = await self.detector.batch_detect(images)
            
            # Process results for each camera
            all_detections = []
            
            for i, (camera_id, raw_detections, frame_metadata) in enumerate(
                zip(camera_ids, batch_results, frame_metadatas)
            ):
                detections = self._process_raw_detections(
                    raw_detections,
                    camera_id,
                    frame_index,
                    frame_metadata,
                    confidence_threshold
                )
                all_detections.extend(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Error in GPU batch detection: {e}")
            # Fall back to individual processing
            return await self._fallback_individual_detection(
                frame_batch, 
                frame_index, 
                confidence_threshold
            )
    
    async def _fallback_individual_detection(
        self,
        frame_batch: Dict[CameraID, Dict[str, Any]],
        frame_index: int,
        confidence_threshold: float
    ) -> List[Detection]:
        """Fallback to individual frame processing."""
        all_detections = []
        
        for camera_id, frame_data in frame_batch.items():
            try:
                detections = await self.detect_persons_in_frame(
                    frame_data,
                    camera_id,
                    frame_index,
                    confidence_threshold
                )
                all_detections.extend(detections)
            except Exception as e:
                logger.error(f"Error in fallback detection for camera {camera_id}: {e}")
                continue
        
        return all_detections
    
    def _extract_frame_metadata(
        self,
        frame_data: Dict[str, Any],
        camera_id: CameraID,
        frame_index: int
    ) -> FrameMetadata:
        """Extract frame metadata from frame data."""
        return FrameMetadata(
            frame_index=frame_index,
            timestamp=datetime.now(timezone.utc),
            camera_id=camera_id,
            width=frame_data.get("width", 1920),
            height=frame_data.get("height", 1080),
            fps=frame_data.get("fps"),
            format=frame_data.get("format"),
            encoding=frame_data.get("encoding")
        )
    
    def _process_raw_detections(
        self,
        raw_detections: List[Dict[str, Any]],
        camera_id: CameraID,
        frame_index: int,
        frame_metadata: FrameMetadata,
        confidence_threshold: float
    ) -> List[Detection]:
        """Process raw detector output into domain objects."""
        detections = []
        
        for i, raw_detection in enumerate(raw_detections):
            try:
                # Filter by confidence
                confidence = raw_detection.get("confidence", 0.0)
                if confidence < confidence_threshold:
                    continue
                
                # Filter by class (only persons)
                class_id = raw_detection.get("class_id", 0)
                if class_id != DetectionClass.PERSON.value:
                    continue
                
                # Create bounding box
                bbox_data = raw_detection.get("bbox", {})
                bbox = BoundingBox(
                    x=bbox_data.get("x", 0),
                    y=bbox_data.get("y", 0),
                    width=bbox_data.get("width", 0),
                    height=bbox_data.get("height", 0),
                    normalized=bbox_data.get("normalized", False)
                )
                
                # Create detection
                detection = Detection(
                    id=f"{camera_id}_{frame_index}_{i}",
                    camera_id=camera_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_id=DetectionClass.PERSON,
                    timestamp=frame_metadata.timestamp,
                    frame_index=frame_index
                )
                
                detections.append(detection)
                
            except Exception as e:
                logger.warning(f"Error processing detection {i}: {e}")
                continue
        
        return detections
    
    def _update_stats(self, detections: List[Detection], processing_time: float):
        """Update detection statistics."""
        self.detection_stats["total_detections"] += len(detections)
        self.detection_stats["person_detections"] += len([d for d in detections if d.is_person])
        self.detection_stats["processing_times"].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.detection_stats["processing_times"]) > 100:
            self.detection_stats["processing_times"] = self.detection_stats["processing_times"][-100:]
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get current detection statistics."""
        processing_times = self.detection_stats["processing_times"]
        batch_processing_times = self.detection_stats["batch_processing_times"]
        
        stats = {
            "total_detections": self.detection_stats["total_detections"],
            "person_detections": self.detection_stats["person_detections"],
            "error_count": self.detection_stats["error_count"],
            "gpu_enabled": self.detection_stats["gpu_enabled"],
            "batch_size": self.detection_stats["batch_size"],
            "total_frames_processed": len(processing_times),
            "total_batches_processed": len(batch_processing_times)
        }
        
        # Single frame processing stats
        if processing_times:
            stats.update({
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "fps": len(processing_times) / sum(processing_times) if sum(processing_times) > 0 else 0
            })
        else:
            stats.update({
                "avg_processing_time": 0,
                "min_processing_time": 0,
                "max_processing_time": 0,
                "fps": 0
            })
        
        # Batch processing stats
        if batch_processing_times:
            stats.update({
                "avg_batch_processing_time": sum(batch_processing_times) / len(batch_processing_times),
                "min_batch_processing_time": min(batch_processing_times),
                "max_batch_processing_time": max(batch_processing_times),
                "batch_fps": len(batch_processing_times) / sum(batch_processing_times) if sum(batch_processing_times) > 0 else 0
            })
        else:
            stats.update({
                "avg_batch_processing_time": 0,
                "min_batch_processing_time": 0,
                "max_batch_processing_time": 0,
                "batch_fps": 0
            })
        
        # GPU stats if available
        if self.enable_gpu:
            try:
                gpu_stats = self.gpu_manager.get_device_stats()
                stats["gpu_info"] = gpu_stats
            except Exception as e:
                logger.warning(f"Error getting GPU stats: {e}")
        
        return stats
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.detection_stats = {
            "total_detections": 0,
            "person_detections": 0,
            "processing_times": [],
            "batch_processing_times": [],
            "error_count": 0,
            "gpu_enabled": self.enable_gpu,
            "batch_size": self.batch_size
        }
        logger.info("Detection statistics reset")
    
    async def validate_detection_quality(
        self,
        detection: Detection,
        frame_data: Dict[str, Any]
    ) -> bool:
        """
        Validate detection quality.
        
        Args:
            detection: Detection to validate
            frame_data: Original frame data
            
        Returns:
            True if detection passes quality checks
        """
        try:
            # Check bounding box validity
            if detection.bbox.area < 100:  # Minimum area threshold
                return False
            
            # Check aspect ratio (person should be taller than wide)
            aspect_ratio = detection.bbox.height / detection.bbox.width
            if aspect_ratio < 1.2:  # Minimum height/width ratio
                return False
            
            # Check confidence
            if detection.confidence < 0.3:  # Minimum confidence
                return False
            
            # Check if bbox is within frame bounds
            frame_width = frame_data.get("width", 1920)
            frame_height = frame_data.get("height", 1080)
            
            if not detection.bbox.normalized:
                if (detection.bbox.x < 0 or detection.bbox.y < 0 or 
                    detection.bbox.x2 > frame_width or detection.bbox.y2 > frame_height):
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating detection {detection.id}: {e}")
            return False
    
    async def filter_overlapping_detections(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Filter out overlapping detections using Non-Maximum Suppression.
        
        Args:
            detections: List of detections to filter
            iou_threshold: IoU threshold for overlap detection
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        try:
            # Sort by confidence (highest first)
            sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
            
            filtered_detections = []
            
            for detection in sorted_detections:
                # Check if this detection overlaps with any already selected
                should_keep = True
                
                for kept_detection in filtered_detections:
                    iou = self._calculate_iou(detection.bbox, kept_detection.bbox)
                    if iou > iou_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    filtered_detections.append(detection)
            
            logger.debug(f"Filtered {len(detections)} detections to {len(filtered_detections)}")
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Error filtering overlapping detections: {e}")
            return detections
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        try:
            # Calculate intersection area
            x1_inter = max(bbox1.x, bbox2.x)
            y1_inter = max(bbox1.y, bbox2.y)
            x2_inter = min(bbox1.x2, bbox2.x2)
            y2_inter = min(bbox1.y2, bbox2.y2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union area
            area1 = bbox1.area
            area2 = bbox2.area
            union_area = area1 + area2 - intersection_area
            
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
            
        except Exception as e:
            logger.warning(f"Error calculating IoU: {e}")
            return 0.0
    
    async def cleanup(self):
        """Clean up detection service resources."""
        try:
            if self.detector:
                await self.detector.cleanup()
                
            logger.info("Detection service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during detection service cleanup: {e}")
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information."""
        if self.detector:
            return self.detector.get_model_info()
        return {}
    
    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold."""
        if self.detector:
            self.detector.set_confidence_threshold(threshold)
            
    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        if self.detector:
            return self.detector.get_confidence_threshold()
        return 0.5
    
    async def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for detection."""
        if self.detector:
            return await self.detector.preprocess_image(frame)
        return frame