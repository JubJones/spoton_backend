"""
Detection video service - integrates RT-DETR person detection with video streaming.

This service extends RawVideoService to add RT-DETR-based person detection capabilities
as outlined in Phase 1: Foundation Setup of the DETECTION.md pipeline requirements.

Features:
- RT-DETR person detection on video frames
- Inherits all raw video streaming capabilities  
- Basic detection processing and frame annotation
- WebSocket streaming of detection results
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import numpy as np
import cv2

from app.core.config import settings
from app.services.raw_video_service import RawVideoService
from app.models.rtdetr_detector import RTDETRDetector
from app.api.websockets.connection_manager import binary_websocket_manager, MessageType
from app.api.websockets.frame_handler import frame_handler

logger = logging.getLogger(__name__)


class DetectionVideoService(RawVideoService):
    """
    Detection video service that extends RawVideoService with RT-DETR person detection.
    
    Phase 1 Implementation Features:
    - RT-DETR person detection on video frames
    - Basic frame processing and detection data extraction
    - WebSocket streaming of detection results
    - Inherits all raw video capabilities from parent class
    """
    
    def __init__(self):
        super().__init__()
        
        # RT-DETR detector instance (Phase 1)
        self.detector: Optional[RTDETRDetector] = None
        
        # Detection statistics
        self.detection_stats = {
            "total_frames_processed": 0,
            "total_detections_found": 0,
            "average_detection_time": 0.0,
            "successful_detections": 0,
            "failed_detections": 0
        }
        
        # Performance tracking
        self.detection_times: List[float] = []
        
        logger.info("DetectionVideoService initialized (Phase 1: RT-DETR Foundation)")
    
    async def initialize_detection_services(self, environment_id: str = "default") -> bool:
        """Initialize detection services including RT-DETR model loading."""
        try:
            logger.info(f"🚀 DETECTION SERVICE INIT: Starting detection service initialization for environment: {environment_id}")
            
            # First initialize parent services (video data manager, asset downloader)
            parent_initialized = await self.initialize_services(environment_id)
            if not parent_initialized:
                logger.error("❌ DETECTION SERVICE INIT: Failed to initialize parent video services")
                return False
            
            # Initialize RT-DETR detector
            logger.info("🧠 DETECTION SERVICE INIT: Loading RT-DETR model...")
            self.detector = RTDETRDetector(
                model_name=settings.RTDETR_MODEL_PATH.split("/")[-1],  # Extract filename
                confidence_threshold=settings.RTDETR_CONFIDENCE_THRESHOLD
            )
            
            # Load model
            await self.detector.load_model()
            
            # Warm up model for better performance
            await self.detector.warmup()
            
            logger.info("✅ DETECTION SERVICE INIT: Detection services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ DETECTION SERVICE INIT: Failed to initialize detection services: {e}")
            return False
    
    async def process_frame_with_detection(self, frame: np.ndarray, camera_id: str, frame_number: int) -> Dict[str, Any]:
        """
        Process frame with RT-DETR detection (Phase 1 implementation).
        
        Args:
            frame: Video frame as numpy array
            camera_id: Camera identifier  
            frame_number: Frame sequence number
            
        Returns:
            Detection data dictionary with bounding boxes and metadata
        """
        detection_start = time.time()
        
        try:
            if not self.detector:
                raise RuntimeError("RT-DETR detector not initialized")
            
            # Run RT-DETR detection
            detections = await self.detector.detect(frame)
            
            # Calculate processing time
            processing_time = (time.time() - detection_start) * 1000
            
            # Convert detections to the expected format
            detection_data = {
                "detections": [
                    {
                        "detection_id": f"det_{i:03d}",
                        "class_name": "person",
                        "class_id": 0,
                        "confidence": detection.confidence,
                        "bbox": {
                            "x1": detection.bbox.x1,
                            "y1": detection.bbox.y1, 
                            "x2": detection.bbox.x2,
                            "y2": detection.bbox.y2,
                            "width": detection.bbox.x2 - detection.bbox.x1,
                            "height": detection.bbox.y2 - detection.bbox.y1,
                            "center_x": (detection.bbox.x1 + detection.bbox.x2) / 2,
                            "center_y": (detection.bbox.y1 + detection.bbox.y2) / 2
                        },
                        "track_id": None,  # Phase 1: No tracking yet
                        "global_id": None,  # Phase 1: No re-ID yet
                        "map_coords": {"map_x": 0, "map_y": 0}  # Phase 1: No homography yet
                    }
                    for i, detection in enumerate(detections)
                ],
                "detection_count": len(detections),
                "processing_time_ms": processing_time
            }
            
            # Update statistics
            self.detection_times.append(processing_time)
            self.detection_stats["total_frames_processed"] += 1
            self.detection_stats["total_detections_found"] += len(detections)
            self.detection_stats["successful_detections"] += 1
            self._update_detection_stats()
            
            return detection_data
            
        except Exception as e:
            logger.error(f"❌ DETECTION PROCESSING: Error processing frame {frame_number} from camera {camera_id}: {e}")
            self.detection_stats["failed_detections"] += 1
            
            # Return empty detection result on error
            return {
                "detections": [],
                "detection_count": 0,
                "processing_time_ms": 0,
                "error": str(e)
            }
    
    async def run_detection_pipeline(self, task_id: uuid.UUID, environment_id: str):
        """
        Main detection pipeline that extends raw video streaming with RT-DETR detection.
        
        Phase 1 Process:
        1. Initialize detection services (including RT-DETR)
        2. Download video data (inherited from parent)
        3. Extract and process frames with detection
        4. Stream detection results via WebSocket
        """
        pipeline_start = time.time()
        
        try:
            logger.info(f"🎬 DETECTION PIPELINE: Starting detection pipeline for task {task_id}, environment {environment_id}")
            
            # Update task status
            await self._update_task_status(task_id, "INITIALIZING", 0.05, "Initializing detection services")
            
            # Step 1: Initialize detection services
            logger.info(f"🧠 DETECTION PIPELINE: Step 1/4 - Initializing detection services for task {task_id}")
            services_initialized = await self.initialize_detection_services(environment_id)
            if not services_initialized:
                raise RuntimeError("Failed to initialize detection services")
            
            await self._update_task_status(task_id, "DOWNLOADING", 0.25, "Downloading video data")
            
            # Step 2: Download video data (inherited method)
            logger.info(f"⬇️ DETECTION PIPELINE: Step 2/4 - Downloading video data for task {task_id}")
            video_data = await self._download_video_data(environment_id)
            if not video_data:
                raise RuntimeError("Failed to download video data")
            
            await self._update_task_status(task_id, "PROCESSING", 0.50, "Processing frames with RT-DETR detection")
            
            # Step 3: Process frames with detection
            logger.info(f"🔍 DETECTION PIPELINE: Step 3/4 - Processing frames with detection for task {task_id}")
            detection_success = await self._process_frames_with_detection(task_id, video_data)
            if not detection_success:
                raise RuntimeError("Failed to process frames with detection")
            
            await self._update_task_status(task_id, "STREAMING", 0.75, "Streaming detection results")
            
            # Step 4: Stream detection results (integrated with frame processing)
            logger.info(f"📡 DETECTION PIPELINE: Step 4/4 - Streaming detection results for task {task_id}")
            streaming_success = await self._stream_detection_results(task_id, video_data)
            if not streaming_success:
                raise RuntimeError("Failed to stream detection results")
            
            # Mark as completed
            await self._update_task_status(task_id, "COMPLETED", 1.0, "Detection pipeline completed successfully")
            
            # Update final statistics
            pipeline_time = time.time() - pipeline_start
            logger.info(f"✅ DETECTION PIPELINE: Pipeline completed successfully for task {task_id} in {pipeline_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ DETECTION PIPELINE: Error in detection pipeline for task {task_id}: {e}")
            await self._update_task_status(task_id, "FAILED", 0.0, f"Detection pipeline failed: {str(e)}")
            
        finally:
            # Cleanup (inherited from parent)
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            if environment_id in self.environment_tasks:
                del self.environment_tasks[environment_id]
    
    async def _process_frames_with_detection(self, task_id: uuid.UUID, video_data: Dict[str, Any]) -> bool:
        """Process video frames with RT-DETR detection."""
        try:
            logger.info(f"🔍 DETECTION PROCESSING: Processing frames with detection for task {task_id}")
            
            # Get frame count for progress tracking
            total_frames = min(
                data.get("frame_count", 0) for data in video_data.values() 
                if data.get("frame_count", 0) > 0
            )
            
            if total_frames == 0:
                logger.warning("No frames available for detection processing")
                return False
            
            frame_index = 0
            
            # Process frames from all cameras
            while frame_index < total_frames:
                # Check if task is still active
                if task_id not in self.active_tasks:
                    logger.info(f"🔍 DETECTION PROCESSING: Task {task_id} was stopped")
                    break
                
                # Read frames from all cameras
                camera_frames = {}
                camera_detections = {}
                all_frames_valid = True
                
                for camera_id, data in video_data.items():
                    cap = data.get("video_capture")
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            camera_frames[camera_id] = frame
                            
                            # Process frame with detection
                            detection_data = await self.process_frame_with_detection(
                                frame, camera_id, frame_index
                            )
                            camera_detections[camera_id] = detection_data
                        else:
                            all_frames_valid = False
                            break
                    else:
                        all_frames_valid = False
                        break
                
                if not all_frames_valid or not camera_frames:
                    logger.info(f"🔍 DETECTION PROCESSING: End of video reached at frame {frame_index}")
                    break
                
                # Log progress periodically
                if frame_index % 30 == 0:  # Every 30 frames
                    progress = 0.50 + (frame_index / total_frames) * 0.25  # 0.50-0.75 range
                    detection_count = sum(det["detection_count"] for det in camera_detections.values())
                    await self._update_task_status(
                        task_id, "PROCESSING", progress,
                        f"Processed frame {frame_index}/{total_frames} - Found {detection_count} detections"
                    )
                    logger.info(f"🔍 DETECTION PROCESSING: Frame {frame_index}/{total_frames} - {detection_count} detections")
                
                frame_index += 1
            
            # Cleanup video captures
            for data in video_data.values():
                cap = data.get("video_capture")
                if cap:
                    cap.release()
            
            logger.info(f"✅ DETECTION PROCESSING: Frame processing completed for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ DETECTION PROCESSING: Error processing frames: {e}")
            return False
    
    async def _stream_detection_results(self, task_id: uuid.UUID, video_data: Dict[str, Any]) -> bool:
        """Stream detection results via WebSocket (Phase 1 - basic implementation)."""
        try:
            logger.info(f"📡 DETECTION STREAMING: Starting detection result streaming for task {task_id}")
            
            # For Phase 1, we'll implement a simple streaming approach
            # This will be enhanced in later phases with real-time processing
            
            # Create a summary message for completed detection processing
            total_detections = sum(self.detection_stats.get("total_detections_found", 0) for _ in video_data)
            
            summary_message = {
                "type": MessageType.TRACKING_UPDATE.value,
                "task_id": str(task_id),
                "timestamp_processed_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "detection_processing",
                "summary": {
                    "total_frames_processed": self.detection_stats.get("total_frames_processed", 0),
                    "total_detections_found": self.detection_stats.get("total_detections_found", 0),
                    "average_detection_time_ms": self.detection_stats.get("average_detection_time", 0.0),
                    "cameras_processed": list(video_data.keys())
                },
                "cameras": {
                    camera_id: {
                        "processing_completed": True,
                        "status": "detection_complete"
                    }
                    for camera_id in video_data.keys()
                }
            }
            
            # Send summary via WebSocket
            success = await binary_websocket_manager.send_json_message(
                str(task_id), summary_message, MessageType.TRACKING_UPDATE
            )
            
            if success:
                logger.info(f"📡 DETECTION STREAMING: Sent detection summary for task {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ DETECTION STREAMING: Error streaming detection results: {e}")
            return False
    
    def _update_detection_stats(self):
        """Update detection statistics."""
        try:
            if self.detection_times:
                self.detection_stats["average_detection_time"] = sum(self.detection_times) / len(self.detection_times)
                
                # Keep only recent times
                if len(self.detection_times) > 100:
                    self.detection_times = self.detection_times[-50:]
                    
        except Exception as e:
            logger.error(f"Error updating detection stats: {e}")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        stats = dict(self.detection_stats)
        stats.update({
            "active_tasks_count": len(self.active_tasks),
            "total_tasks_count": len(self.tasks),
            "detector_loaded": self.detector is not None and getattr(self.detector, '_model_loaded_flag', False)
        })
        return stats


# Global detection video service instance
detection_video_service = DetectionVideoService()