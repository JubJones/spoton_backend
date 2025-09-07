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
import json

from app.core.config import settings
from app.services.raw_video_service import RawVideoService
from app.models.rtdetr_detector import RTDETRDetector
from app.utils.detection_annotator import DetectionAnnotator
from app.api.websockets.connection_manager import binary_websocket_manager, MessageType
from app.api.websockets.frame_handler import frame_handler
from app.services.homography_service import HomographyService
from app.services.handoff_detection_service import HandoffDetectionService

logger = logging.getLogger(__name__)


class DetectionVideoService(RawVideoService):
    """
    Detection video service that extends RawVideoService with RT-DETR person detection.
    
    Phase 4 Implementation Features:
    - RT-DETR person detection on video frames
    - Spatial intelligence with homography coordinate transformations
    - Camera handoff detection for cross-camera tracking
    - WebSocket streaming of enhanced detection results
    - Inherits all raw video capabilities from parent class
    """
    
    def __init__(self):
        super().__init__()
        
        # RT-DETR detector instance (Phase 1)
        self.detector: Optional[RTDETRDetector] = None
        
        # Phase 2: Detection annotator for bounding box visualization
        self.annotator = DetectionAnnotator()
        
        # Phase 4: Spatial intelligence services
        self.homography_service: Optional[HomographyService] = None
        self.handoff_service: Optional[HandoffDetectionService] = None
        
        # Detection statistics (enhanced for Phase 2)
        self.detection_stats = {
            "total_frames_processed": 0,
            "total_detections_found": 0,
            "average_detection_time": 0.0,
            "successful_detections": 0,
            "failed_detections": 0,
            "frames_annotated": 0,
            "websocket_messages_sent": 0,
            "annotation_time": 0.0
        }
        
        # Performance tracking
        self.detection_times: List[float] = []
        self.annotation_times: List[float] = []
        
        logger.info("DetectionVideoService initialized (Phase 4: Spatial Intelligence)")
    
    async def initialize_detection_services(self, environment_id: str = "default") -> bool:
        """Initialize detection services including RT-DETR model loading and spatial intelligence services."""
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
            
            # Phase 4: Initialize spatial intelligence services
            logger.info("🗺️ DETECTION SERVICE INIT: Initializing spatial intelligence services...")
            
            # Initialize HomographyService
            self.homography_service = HomographyService(settings)
            await self.homography_service.preload_all_homography_matrices()
            
            # Initialize HandoffDetectionService
            self.handoff_service = HandoffDetectionService()
            
            # Validate spatial intelligence configuration
            homography_validation = len(self.homography_service._homography_matrices) > 0
            handoff_validation = self.handoff_service.validate_configuration()
            
            logger.info(f"🗺️ SPATIAL INTELLIGENCE: Homography matrices loaded: {homography_validation}")
            logger.info(f"🗺️ SPATIAL INTELLIGENCE: Handoff configuration valid: {all(handoff_validation.values())}")
            
            logger.info("✅ DETECTION SERVICE INIT: Detection and spatial intelligence services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ DETECTION SERVICE INIT: Failed to initialize detection services: {e}")
            return False
    
    async def process_frame_with_detection(self, frame: np.ndarray, camera_id: str, frame_number: int) -> Dict[str, Any]:
        """
        Process frame with RT-DETR detection and Phase 4 spatial intelligence.
        
        Args:
            frame: Video frame as numpy array
            camera_id: Camera identifier  
            frame_number: Frame sequence number
            
        Returns:
            Detection data dictionary with bounding boxes, coordinates, and spatial metadata
        """
        detection_start = time.time()
        
        try:
            if not self.detector:
                raise RuntimeError("RT-DETR detector not initialized")
            
            # Run RT-DETR detection
            detections = await self.detector.detect(frame)
            
            # Calculate processing time
            processing_time = (time.time() - detection_start) * 1000
            
            # Get frame dimensions for spatial processing
            frame_height, frame_width = frame.shape[:2]
            
            # Convert detections to the expected format with Phase 4 enhancements
            enhanced_detections = []
            for i, detection in enumerate(detections):
                bbox_dict = {
                    "x1": detection.bbox.x1,
                    "y1": detection.bbox.y1, 
                    "x2": detection.bbox.x2,
                    "y2": detection.bbox.y2,
                    "width": detection.bbox.x2 - detection.bbox.x1,
                    "height": detection.bbox.y2 - detection.bbox.y1,
                    "center_x": (detection.bbox.x1 + detection.bbox.x2) / 2,
                    "center_y": (detection.bbox.y1 + detection.bbox.y2) / 2
                }
                
                # Phase 4: Apply spatial intelligence
                map_coords = {"map_x": 0, "map_y": 0}  # Default fallback
                handoff_triggered = False
                candidate_cameras = []
                
                # Homography coordinate transformation
                if self.homography_service:
                    # Use foot point (bottom center of bounding box) for mapping
                    foot_point = (bbox_dict["center_x"], bbox_dict["y2"])
                    
                    # Try Phase 4 JSON-based projection first
                    projected_coords = self.homography_service.project_to_map(camera_id, foot_point)
                    if projected_coords:
                        map_coords = {"map_x": projected_coords[0], "map_y": projected_coords[1]}
                    else:
                        # Fallback to existing preload system
                        env_id = self._get_environment_for_camera(camera_id)
                        if env_id:
                            from app.shared.types import CameraID
                            matrix = self.homography_service.get_homography_matrix(env_id, CameraID(camera_id))
                            if matrix is not None:
                                # Apply transformation using existing method
                                try:
                                    import cv2
                                    point = np.array([[foot_point[0], foot_point[1]]], dtype=np.float32)
                                    point = point.reshape(-1, 1, 2)
                                    transformed = cv2.perspectiveTransform(point, matrix)
                                    map_x, map_y = transformed[0, 0]
                                    if np.isfinite(map_x) and np.isfinite(map_y):
                                        map_coords = {"map_x": float(map_x), "map_y": float(map_y)}
                                except Exception as homography_error:
                                    logger.debug(f"Homography transformation failed: {homography_error}")
                
                # Handoff detection
                if self.handoff_service:
                    handoff_triggered, candidate_cameras = self.handoff_service.check_handoff_trigger(
                        camera_id, bbox_dict, frame_width, frame_height
                    )
                
                # Create enhanced detection object
                enhanced_detection = {
                    "detection_id": f"det_{i:03d}",
                    "class_name": "person",
                    "class_id": 0,
                    "confidence": detection.confidence,
                    "bbox": bbox_dict,
                    "track_id": None,  # Future: No tracking yet
                    "global_id": None,  # Future: No re-ID yet
                    "map_coords": map_coords,
                    # Phase 4: Spatial intelligence metadata
                    "spatial_data": {
                        "handoff_triggered": handoff_triggered,
                        "candidate_cameras": candidate_cameras,
                        "coordinate_system": "bev_map_meters" if map_coords["map_x"] != 0 or map_coords["map_y"] != 0 else None
                    }
                }
                
                enhanced_detections.append(enhanced_detection)
            
            detection_data = {
                "detections": enhanced_detections,
                "detection_count": len(detections),
                "processing_time_ms": processing_time,
                # Phase 4: Frame spatial metadata
                "spatial_metadata": {
                    "camera_id": camera_id,
                    "frame_dimensions": {"width": frame_width, "height": frame_height},
                    "homography_available": self.homography_service is not None,
                    "handoff_detection_enabled": self.handoff_service is not None
                }
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
                            
                            # Process frame with detection (Phase 2: Enhanced processing)
                            detection_data = await self.process_frame_with_detection(
                                frame, camera_id, frame_index
                            )
                            camera_detections[camera_id] = detection_data
                            
                            # Phase 2: Send real-time detection update via WebSocket
                            await self.send_detection_update(task_id, camera_id, frame, detection_data, frame_index)
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
    
    async def send_detection_update(self, task_id: uuid.UUID, camera_id: str, frame: np.ndarray, 
                                   detection_data: Dict[str, Any], frame_number: int):
        """
        Send detection update via WebSocket with Phase 4 spatial intelligence data.
        
        Follows DETECTION.md schema with populated homography and handoff data
        replacing the previous static null values with actual spatial intelligence results.
        """
        try:
            # Create detection overlay with annotated frames
            frame_overlay = self.annotator.create_detection_overlay(frame, detection_data["detections"])
            
            # Phase 4: Prepare homography data for WebSocket message
            homography_data = None
            if self.homography_service:
                homography_data = self.homography_service.get_homography_data(camera_id)
            
            # Phase 4: Prepare mapping coordinates data
            mapping_coordinates = []
            if "detections" in detection_data:
                for detection in detection_data["detections"]:
                    if "map_coords" in detection and (detection["map_coords"]["map_x"] != 0 or detection["map_coords"]["map_y"] != 0):
                        # Extract foot point used for projection
                        foot_point = {
                            "image_x": detection["bbox"]["center_x"],
                            "image_y": detection["bbox"]["y2"]  # Bottom of bounding box
                        }
                        
                        coord_data = {
                            "detection_id": detection["detection_id"],
                            "map_x": detection["map_coords"]["map_x"],
                            "map_y": detection["map_coords"]["map_y"],
                            "projection_successful": True,
                            "foot_point": foot_point,
                            "coordinate_system": detection.get("spatial_data", {}).get("coordinate_system", "bev_map_meters")
                        }
                        mapping_coordinates.append(coord_data)
            
            # Create WebSocket message compatible with frontend (same format as raw endpoint)
            detection_message = {
                "type": MessageType.TRACKING_UPDATE.value,  # Compatible with frontend expectation
                "task_id": str(task_id),
                "camera_id": camera_id,
                "global_frame_index": frame_number,  # Compatible key name
                "timestamp_processed_utc": datetime.now(timezone.utc).isoformat(),  # Compatible key name
                "mode": "detection_streaming",  # Distinguish from raw mode
                "camera_data": {
                    "frame_image_base64": frame_overlay["annotated_b64"],  # Use annotated frame for display
                    "original_frame_base64": frame_overlay["original_b64"],  # Keep original for reference
                    "tracks": [],  # Future: Will contain tracking data
                    "frame_width": frame_overlay["width"],
                    "frame_height": frame_overlay["height"], 
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                # Additional detection-specific data (won't interfere with frontend display)
                "detection_data": detection_data,
                "future_pipeline_data": {
                    "tracking_data": None,  # Future: Multi-object tracking data
                    "reid_data": None,      # Future: Cross-camera re-identification data
                    # Phase 4: Populated homography and mapping data
                    "homography_data": homography_data,
                    "mapping_coordinates": mapping_coordinates if mapping_coordinates else None
                }
            }
            
            # Send via WebSocket
            success = await binary_websocket_manager.send_json_message(
                str(task_id), detection_message, MessageType.TRACKING_UPDATE
            )
            
            if success:
                self.detection_stats["websocket_messages_sent"] += 1
                logger.debug(f"📡 DETECTION UPDATE: Sent Phase 4 detection update for task {task_id}, camera {camera_id}, frame {frame_number}")
            else:
                logger.warning(f"📡 DETECTION UPDATE: Failed to send detection update for task {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ DETECTION UPDATE: Error sending detection update: {e}")
            return False
    
    async def process_detection_task_simple(self, task_id: uuid.UUID, environment_id: str):
        """
        Simplified detection processing pipeline (detection-only).
        
        Focuses only on person detection with RT-DETR, sends results via WebSocket
        with static null values for future pipeline features (tracking, re-ID, homography).
        """
        pipeline_start = time.time()
        
        try:
            logger.info(f"🎬 DETECTION PIPELINE: Starting simplified detection pipeline for task {task_id}")
            
            # Initialize detection services
            await self._update_task_status(task_id, "INITIALIZING", 0.10, "Initializing RT-DETR detection services")
            services_initialized = await self.initialize_detection_services(environment_id)
            if not services_initialized:
                raise RuntimeError("Failed to initialize detection services")
            
            # Download video data
            await self._update_task_status(task_id, "DOWNLOADING", 0.30, "Downloading video data")
            video_data = await self._download_video_data(environment_id)
            if not video_data:
                raise RuntimeError("Failed to download video data")
            
            # Extract frames for processing
            await self._update_task_status(task_id, "EXTRACTING", 0.45, "Extracting frames from video data")
            frames_extracted = await self._extract_raw_frames(task_id, video_data)
            if not frames_extracted:
                raise RuntimeError("Failed to extract frames")
            
            # Process frames with detection only
            await self._update_task_status(task_id, "PROCESSING", 0.60, "Processing frames with RT-DETR detection")
            success = await self._process_frames_simple_detection(task_id, video_data)
            if not success:
                raise RuntimeError("Failed to process frames with detection")
            
            # Complete
            await self._update_task_status(task_id, "COMPLETED", 1.0, "Detection pipeline completed successfully")
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"✅ DETECTION PIPELINE: Pipeline completed in {pipeline_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ DETECTION PIPELINE: Error in detection pipeline: {e}")
            await self._update_task_status(task_id, "FAILED", 0.0, f"Detection pipeline failed: {str(e)}")
            
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            if environment_id in self.environment_tasks:
                del self.environment_tasks[environment_id]
    
    async def _process_frames_simple_detection(self, task_id: uuid.UUID, video_data: Dict[str, Any]) -> bool:
        """
        Process frames with simple RT-DETR detection only.
        
        Simplified version that focuses only on detection and annotation,
        sending WebSocket updates with static null values for future features.
        """
        try:
            logger.info(f"🔍 SIMPLE DETECTION: Starting frame processing with RT-DETR for task {task_id}")
            
            # Get total frame count
            frame_counts = [data.get("frame_count", 0) for data in video_data.values() if data.get("frame_count", 0) > 0]
            total_frames = min(frame_counts) if frame_counts else 0
            
            if total_frames == 0:
                logger.warning("No frames available for detection processing")
                return False
            
            frame_index = 0
            frames_processed = 0
            
            # Main processing loop
            while frame_index < total_frames:
                if task_id not in self.active_tasks:
                    logger.info(f"🔍 SIMPLE DETECTION: Task {task_id} was stopped")
                    break
                
                # Process all cameras for current frame
                any_frame_processed = False
                
                for camera_id, data in video_data.items():
                    cap = data.get("video_capture")
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Run RT-DETR detection on frame
                            detection_data = await self.process_frame_with_detection(
                                frame, camera_id, frame_index
                            )
                            
                            # Send WebSocket update with detection results
                            await self.send_detection_update(
                                task_id, camera_id, frame, detection_data, frame_index
                            )
                            
                            frames_processed += 1
                            any_frame_processed = True
                        else:
                            logger.debug(f"End of video reached for camera {camera_id}")
                            break
                    else:
                        break
                
                # Check if all cameras finished
                if not any_frame_processed:
                    logger.info(f"🔍 SIMPLE DETECTION: All cameras finished at frame {frame_index}")
                    break
                
                # Update progress every 30 frames
                if frame_index % 30 == 0:
                    progress = 0.60 + (frame_index / total_frames) * 0.35  # 0.60-0.95 range
                    await self._update_task_status(
                        task_id, "PROCESSING", progress,
                        f"Processed frame {frame_index}/{total_frames} - {frames_processed} detections sent"
                    )
                
                frame_index += 1
                
                # Small delay to prevent overwhelming WebSocket clients
                await asyncio.sleep(0.02)  # 20ms delay
            
            # Cleanup video captures
            for data in video_data.values():
                cap = data.get("video_capture")
                if cap:
                    cap.release()
            
            logger.info(f"✅ SIMPLE DETECTION: Completed processing - {frames_processed} frames processed")
            return True
            
        except Exception as e:
            logger.error(f"❌ SIMPLE DETECTION: Error in frame processing: {e}")
            return False
    
    async def _process_frames_with_realtime_streaming(self, task_id: uuid.UUID, video_data: Dict[str, Any]) -> bool:
        """
        Process frames with real-time detection and WebSocket streaming.
        
        This is the core Phase 2 processing method that handles frame-by-frame
        detection, annotation, and streaming to connected clients.
        """
        try:
            logger.info(f"🔍 PHASE 2 PROCESSING: Starting real-time frame processing for task {task_id}")
            
            # Get total frame count
            total_frames = min(
                data.get("frame_count", 0) for data in video_data.values() 
                if data.get("frame_count", 0) > 0
            )
            
            if total_frames == 0:
                logger.warning("No frames available for Phase 2 processing")
                return False
            
            frame_index = 0
            frames_streamed = 0
            
            # Main processing loop
            while frame_index < total_frames:
                if task_id not in self.active_tasks:
                    logger.info(f"🔍 PHASE 2 PROCESSING: Task {task_id} was stopped")
                    break
                
                # Process all cameras for current frame
                camera_frames_processed = 0
                
                for camera_id, data in video_data.items():
                    cap = data.get("video_capture")
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Detect + Annotate + Stream in real-time
                            detection_data = await self.process_frame_with_detection(
                                frame, camera_id, frame_index
                            )
                            
                            # Send real-time update
                            stream_success = await self.send_detection_update(
                                task_id, camera_id, frame, detection_data, frame_index
                            )
                            
                            if stream_success:
                                frames_streamed += 1
                            
                            camera_frames_processed += 1
                        else:
                            logger.debug(f"End of video reached for camera {camera_id}")
                            break
                    else:
                        break
                
                # Check if all cameras finished
                if camera_frames_processed == 0:
                    logger.info(f"🔍 PHASE 2 PROCESSING: All cameras finished at frame {frame_index}")
                    break
                
                # Update progress every 15 frames
                if frame_index % 15 == 0:
                    progress = 0.50 + (frame_index / total_frames) * 0.45  # 0.50-0.95 range
                    await self._update_task_status(
                        task_id, "PROCESSING", progress,
                        f"Streaming frame {frame_index}/{total_frames} - {frames_streamed} updates sent"
                    )
                
                frame_index += 1
                
                # Small delay to prevent overwhelming WebSocket clients
                await asyncio.sleep(0.01)  # 10ms delay
            
            # Cleanup video captures
            for data in video_data.values():
                cap = data.get("video_capture")
                if cap:
                    cap.release()
            
            logger.info(f"✅ PHASE 2 PROCESSING: Completed real-time processing - {frames_streamed} updates streamed")
            return True
            
        except Exception as e:
            logger.error(f"❌ PHASE 2 PROCESSING: Error in real-time processing: {e}")
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
        """Get detection statistics with Phase 4 spatial intelligence status."""
        stats = dict(self.detection_stats)
        stats.update({
            "active_tasks_count": len(self.active_tasks),
            "total_tasks_count": len(self.tasks),
            "detector_loaded": self.detector is not None and getattr(self.detector, '_model_loaded_flag', False),
            # Phase 4: Spatial intelligence status
            "spatial_intelligence": {
                "homography_service_loaded": self.homography_service is not None,
                "handoff_service_loaded": self.handoff_service is not None,
                "homography_matrices_count": len(self.homography_service._homography_matrices) if self.homography_service else 0,
                "handoff_configuration_valid": all(self.handoff_service.validate_configuration().values()) if self.handoff_service else False
            }
        })
        return stats
    
    def _get_environment_for_camera(self, camera_id: str) -> Optional[str]:
        """
        Get environment ID for a given camera ID by looking up VIDEO_SETS configuration.
        
        Args:
            camera_id: Camera identifier (e.g., 'c09')
            
        Returns:
            Environment ID (e.g., 'campus', 'factory') or None if not found
        """
        try:
            for video_set in settings.VIDEO_SETS:
                if video_set.cam_id == camera_id:
                    return video_set.env_id
            
            logger.debug(f"No environment found for camera {camera_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting environment for camera {camera_id}: {e}")
            return None


# Global detection video service instance
detection_video_service = DetectionVideoService()