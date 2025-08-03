"""
Enhanced processing pipeline orchestrator.

Coordinates the three core features:
1. Multi-view person detection
2. Cross-camera re-identification
3. Unified spatial mapping
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone
import time
import numpy as np

from app.core.config import settings
from app.infrastructure.cache.redis_client import redis_client
from app.infrastructure.database.session import get_db_session
from app.infrastructure.gpu import get_gpu_manager

# Import domain services
from app.domains.detection.services.detection_service import DetectionService
from app.domains.reid.services.reid_service import ReIDService
from app.domains.mapping.services.mapping_service import MappingService
from app.domains.mapping.services.trajectory_builder import TrajectoryBuilder
from app.domains.mapping.services.calibration_service import CalibrationService

# Import entities
from app.domains.detection.entities.detection import DetectionBatch
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import CoordinateSystem
from app.domains.mapping.models.coordinate_transformer import CoordinateTransformer

from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Enhanced orchestrator for the AI processing pipeline."""
    
    def __init__(self):
        self.tasks: Dict[uuid.UUID, Dict[str, Any]] = {}
        self.active_tasks: set = set()
        
        # Initialize core services
        self.detection_service: Optional[DetectionService] = None
        self.reid_service: Optional[ReIDService] = None
        self.mapping_service: Optional[MappingService] = None
        self.trajectory_builder: Optional[TrajectoryBuilder] = None
        self.calibration_service: Optional[CalibrationService] = None
        
        # Coordinate transformer for spatial mapping
        self.coordinate_transformer: Optional[CoordinateTransformer] = None
        
        # GPU manager for resource allocation
        self.gpu_manager = get_gpu_manager()
        
        # Pipeline statistics
        self.pipeline_stats = {
            "total_frames_processed": 0,
            "detection_runs": 0,
            "reid_runs": 0,
            "mapping_runs": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "average_processing_time": 0.0,
            "gpu_utilization": 0.0
        }
        
        # Performance tracking
        self.processing_times: List[float] = []
        
        logger.info("Enhanced PipelineOrchestrator initialized")
    
    async def initialize_services(self, environment_id: str = "default") -> bool:
        """Initialize all pipeline services."""
        try:
            logger.info("Initializing pipeline services...")
            
            # Initialize coordinate transformer
            self.coordinate_transformer = CoordinateTransformer(
                enable_caching=True,
                cache_size=1000
            )
            
            # Initialize calibration service
            self.calibration_service = CalibrationService(
                coordinate_transformer=self.coordinate_transformer
            )
            
            # Load calibration environment
            if not await self.calibration_service.load_calibration_environment(environment_id):
                logger.error(f"Failed to load calibration environment: {environment_id}")
                return False
            
            # Initialize detection service
            self.detection_service = DetectionService(
                model_type="faster_rcnn",
                enable_gpu=True,
                batch_size=4  # Process 4 cameras simultaneously
            )
            
            await self.detection_service.initialize_model()
            
            # Initialize ReID service
            self.reid_service = ReIDService(
                model_type="clip",
                enable_gpu=True,
                batch_size=16
            )
            
            await self.reid_service.initialize_model()
            
            # Initialize mapping service
            self.mapping_service = MappingService()
            
            # Register camera views from calibration
            camera_views = self.calibration_service.get_all_camera_views()
            for camera_view in camera_views:
                self.mapping_service.register_camera_view(camera_view)
            
            # Initialize trajectory builder
            self.trajectory_builder = TrajectoryBuilder(
                coordinate_transformer=self.coordinate_transformer,
                min_trajectory_length=3,
                max_gap_duration=2.0,
                smoothing_window=5
            )
            
            logger.info("Pipeline services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing pipeline services: {e}")
            return False
    
    async def initialize_task(self, environment_id: str) -> uuid.UUID:
        """Initialize a new processing task."""
        task_id = uuid.uuid4()
        current_time_utc = datetime.now(timezone.utc)
        
        task_data = {
            "task_id": task_id,
            "environment_id": environment_id,
            "status": "INITIALIZING",
            "progress": 0.0,
            "start_time": current_time_utc,
            "current_step": "Task initialization",
            "details": f"Initializing processing task for environment '{environment_id}'",
        }
        
        self.tasks[task_id] = task_data
        self.active_tasks.add(task_id)
        
        # Cache task state in Redis
        await redis_client.set_json_async(
            f"task:{task_id}:state", 
            task_data, 
            ex=3600  # 1 hour expiration
        )
        
        logger.info(f"Task {task_id} initialized for environment {environment_id}")
        return task_id
    
    async def get_task_status(self, task_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get the current status of a task."""
        # Try to get from memory first
        if task_id in self.tasks:
            return self.tasks[task_id]
        
        # Fall back to Redis
        task_data = await redis_client.get_json_async(f"task:{task_id}:state")
        if task_data:
            # Convert task_id back to UUID if needed
            if isinstance(task_data.get("task_id"), str):
                task_data["task_id"] = uuid.UUID(task_data["task_id"])
            return task_data
        
        return None
    
    async def update_task_status(
        self, 
        task_id: uuid.UUID, 
        status: str, 
        progress: float, 
        current_step: str,
        details: Optional[str] = None
    ):
        """Update task status and progress."""
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for status update")
            return
        
        self.tasks[task_id].update({
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "details": details or f"Processing {current_step}",
            "last_updated": datetime.now(timezone.utc)
        })
        
        # Update Redis cache
        await redis_client.set_json_async(
            f"task:{task_id}:state", 
            self.tasks[task_id], 
            ex=3600
        )
        
        logger.info(f"Task {task_id} status updated: {status} ({progress:.1%})")
    
    async def complete_task(self, task_id: uuid.UUID, success: bool = True):
        """Mark a task as completed."""
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for completion")
            return
        
        status = "COMPLETED" if success else "FAILED"
        self.tasks[task_id].update({
            "status": status,
            "progress": 1.0 if success else self.tasks[task_id].get("progress", 0.0),
            "end_time": datetime.now(timezone.utc),
            "current_step": "Task completed" if success else "Task failed",
        })
        
        # Update Redis cache
        await redis_client.set_json_async(
            f"task:{task_id}:state", 
            self.tasks[task_id], 
            ex=3600
        )
        
        # Remove from active tasks
        self.active_tasks.discard(task_id)
        
        logger.info(f"Task {task_id} marked as {status}")
    
    async def cleanup_task(self, task_id: uuid.UUID):
        """Clean up task resources."""
        if task_id in self.tasks:
            del self.tasks[task_id]
        
        self.active_tasks.discard(task_id)
        
        # Clean up Redis cache
        redis_client.connect().delete(f"task:{task_id}:state")
        
        logger.info(f"Task {task_id} cleaned up")
    
    def get_active_tasks(self) -> List[uuid.UUID]:
        """Get list of active task IDs."""
        return list(self.active_tasks)
    
    async def process_frame_batch(
        self, 
        task_id: uuid.UUID, 
        frame_batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a batch of frames through the three core features.
        
        Pipeline stages:
        1. Multi-view person detection
        2. Cross-camera re-identification
        3. Unified spatial mapping
        """
        if not self.is_initialized():
            raise RuntimeError("Pipeline services not initialized")
        
        batch_start_time = time.time()
        
        try:
            # Stage 1: Person Detection
            await self.update_task_status(
                task_id, 
                "PROCESSING", 
                0.3, 
                "Person detection in progress"
            )
            
            detection_results = await self._run_detection(frame_batch)
            
            # Stage 2: Re-identification
            await self.update_task_status(
                task_id, 
                "PROCESSING", 
                0.6, 
                "Cross-camera re-identification in progress"
            )
            
            reid_results = await self._run_reid(detection_results)
            
            # Stage 3: Spatial Mapping
            await self.update_task_status(
                task_id, 
                "PROCESSING", 
                0.9, 
                "Spatial mapping in progress"
            )
            
            mapping_results = await self._run_mapping(reid_results)
            
            # Calculate total processing time
            total_processing_time = time.time() - batch_start_time
            
            # Update performance tracking
            self.processing_times.append(total_processing_time)
            if len(self.processing_times) > 100:  # Keep only recent 100 measurements
                self.processing_times = self.processing_times[-100:]
            
            self.pipeline_stats["successful_batches"] += 1
            
            # Create final results
            final_results = {
                "detection_results": detection_results,
                "reid_results": reid_results,
                "mapping_results": mapping_results,
                "total_processing_time": total_processing_time,
                "stage_times": {
                    "detection": detection_results.get("processing_time", 0.0),
                    "reid": reid_results.get("processing_time", 0.0),
                    "mapping": mapping_results.get("processing_time", 0.0)
                },
                "batch_id": str(task_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_stats": self.get_pipeline_stats()
            }
            
            logger.info(f"Batch {task_id} processed successfully in {total_processing_time:.3f}s")
            
            return final_results
            
        except Exception as e:
            self.pipeline_stats["failed_batches"] += 1
            logger.error(f"Error processing frame batch for task {task_id}: {e}")
            await self.update_task_status(
                task_id, 
                "FAILED", 
                self.tasks[task_id].get("progress", 0.0), 
                "Processing failed",
                str(e)
            )
            raise
    
    async def _run_detection(self, frame_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run person detection on frame batch."""
        if not self.detection_service:
            raise RuntimeError("Detection service not initialized")
        
        try:
            start_time = time.time()
            
            # Extract frames from batch
            camera_frames = frame_batch.get("camera_frames", {})
            
            # Create detection batch
            detection_batch = DetectionBatch(
                camera_frames=camera_frames,
                batch_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Process detection batch
            detection_results = await self.detection_service.process_frame_batch(detection_batch)
            
            processing_time = time.time() - start_time
            
            self.pipeline_stats["detection_runs"] += 1
            self.pipeline_stats["total_frames_processed"] += len(camera_frames)
            
            logger.info(f"Detection completed in {processing_time:.3f}s with {len(detection_results.get('detections', []))} detections")
            
            return {
                "detection_batch": detection_results,
                "stage": "detection",
                "processing_time": processing_time,
                "detection_count": len(detection_results.get("detections", []))
            }
            
        except Exception as e:
            logger.error(f"Error in detection pipeline: {e}")
            raise
    
    async def _run_reid(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run re-identification on detection results."""
        if not self.reid_service:
            raise RuntimeError("ReID service not initialized")
        
        try:
            start_time = time.time()
            
            # Extract detection batch
            detection_batch = detection_results.get("detection_batch")
            if not detection_batch:
                logger.warning("No detection batch found for ReID")
                return {
                    "identities": {},
                    "stage": "reid",
                    "processing_time": 0.0,
                    "input_detections": detection_results
                }
            
            # Process ReID batch
            reid_results = await self.reid_service.process_detection_batch(detection_batch)
            
            processing_time = time.time() - start_time
            
            self.pipeline_stats["reid_runs"] += 1
            
            logger.info(f"ReID completed in {processing_time:.3f}s with {len(reid_results.get('identities', {}))} identities")
            
            return {
                "reid_results": reid_results,
                "stage": "reid",
                "processing_time": processing_time,
                "identity_count": len(reid_results.get("identities", {})),
                "input_detections": detection_results
            }
            
        except Exception as e:
            logger.error(f"Error in ReID pipeline: {e}")
            raise
    
    async def _run_mapping(self, reid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run spatial mapping on ReID results."""
        if not self.mapping_service or not self.trajectory_builder:
            raise RuntimeError("Mapping services not initialized")
        
        try:
            start_time = time.time()
            
            # Extract ReID results
            reid_data = reid_results.get("reid_results", {})
            detection_batch = reid_results.get("input_detections", {}).get("detection_batch")
            
            if not detection_batch:
                logger.warning("No detection batch found for mapping")
                return {
                    "trajectories": [],
                    "stage": "mapping",
                    "processing_time": 0.0,
                    "input_identities": reid_results
                }
            
            # Transform detections to map coordinates
            mapping_results = await self.mapping_service.transform_detections_to_map(detection_batch)
            
            # Build trajectories for identified persons
            trajectories = []
            identities = reid_data.get("identities", {})
            
            for identity_id, identity in identities.items():
                # Get detections for this identity
                identity_detections = []
                for detection_dict in mapping_results.get("transformed_detections", []):
                    detection = detection_dict.get("detection")
                    if detection and getattr(detection, "track_id", None) == identity.global_id:
                        identity_detections.append(detection)
                
                if identity_detections:
                    # Build trajectory
                    trajectory = await self.trajectory_builder.build_trajectory_from_detections(
                        person_identity=identity,
                        detections=identity_detections,
                        target_coordinate_system=CoordinateSystem.MAP
                    )
                    
                    if trajectory:
                        trajectories.append(trajectory)
            
            processing_time = time.time() - start_time
            
            self.pipeline_stats["mapping_runs"] += 1
            
            logger.info(f"Mapping completed in {processing_time:.3f}s with {len(trajectories)} trajectories")
            
            return {
                "mapping_results": mapping_results,
                "trajectories": trajectories,
                "stage": "mapping",
                "processing_time": processing_time,
                "trajectory_count": len(trajectories),
                "input_identities": reid_results
            }
            
        except Exception as e:
            logger.error(f"Error in mapping pipeline: {e}")
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        # Calculate GPU utilization
        gpu_utilization = 0.0
        if self.gpu_manager:
            gpu_utilization = self.gpu_manager.get_utilization()
        
        # Calculate average processing time
        avg_processing_time = 0.0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        return {
            **self.pipeline_stats,
            "gpu_utilization": gpu_utilization,
            "average_processing_time": avg_processing_time,
            "active_tasks": len(self.active_tasks),
            "services_initialized": all([
                self.detection_service is not None,
                self.reid_service is not None,
                self.mapping_service is not None,
                self.trajectory_builder is not None,
                self.calibration_service is not None
            ])
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics from all services."""
        stats = {}
        
        if self.detection_service:
            stats["detection"] = self.detection_service.get_detection_stats()
        
        if self.reid_service:
            stats["reid"] = self.reid_service.get_reid_stats()
        
        if self.mapping_service:
            stats["mapping"] = self.mapping_service.get_mapping_stats()
        
        if self.trajectory_builder:
            stats["trajectory_builder"] = self.trajectory_builder.get_builder_stats()
        
        if self.calibration_service:
            stats["calibration"] = self.calibration_service.get_service_stats()
        
        if self.coordinate_transformer:
            stats["coordinate_transformer"] = self.coordinate_transformer.get_transformation_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset all pipeline statistics."""
        self.pipeline_stats = {
            "total_frames_processed": 0,
            "detection_runs": 0,
            "reid_runs": 0,
            "mapping_runs": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "average_processing_time": 0.0,
            "gpu_utilization": 0.0
        }
        
        self.processing_times.clear()
        
        # Reset service stats
        if self.detection_service:
            self.detection_service.reset_stats()
        
        if self.reid_service:
            self.reid_service.reset_stats()
        
        if self.mapping_service:
            self.mapping_service.reset_stats()
        
        if self.trajectory_builder:
            self.trajectory_builder.reset_stats()
        
        if self.calibration_service:
            self.calibration_service.reset_stats()
        
        if self.coordinate_transformer:
            self.coordinate_transformer.reset_stats()
        
        logger.info("Pipeline statistics reset")
    
    async def cleanup_services(self):
        """Clean up all pipeline services."""
        try:
            if self.detection_service:
                await self.detection_service.cleanup()
            
            if self.reid_service:
                await self.reid_service.cleanup()
            
            if self.calibration_service:
                await self.calibration_service.cleanup()
            
            if self.coordinate_transformer:
                self.coordinate_transformer.cleanup()
            
            logger.info("Pipeline services cleaned up")
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
    
    def is_initialized(self) -> bool:
        """Check if all services are initialized."""
        return all([
            self.detection_service is not None,
            self.reid_service is not None,
            self.mapping_service is not None,
            self.trajectory_builder is not None,
            self.calibration_service is not None,
            self.coordinate_transformer is not None
        ])
    
    async def run_processing_pipeline(self, task_id: uuid.UUID, environment_id: str):
        """
        Main method to run the complete processing pipeline for a task.
        This is executed as a background task.
        """
        logger.info(f"Starting processing pipeline for task {task_id}, environment {environment_id}")
        
        try:
            # Update task status to processing
            await self.update_task_status(task_id, "INITIALIZING", 0.1, "Initializing services")
            
            # Initialize services for this environment
            services_initialized = await self.initialize_services(environment_id)
            
            if not services_initialized:
                await self.complete_task(task_id, success=False)
                logger.error(f"Failed to initialize services for task {task_id}")
                return
            
            await self.update_task_status(task_id, "DOWNLOADING", 0.2, "Downloading video data")
            
            # Simulate video processing pipeline
            # In a real implementation, this would:
            # 1. Download video segments from S3
            # 2. Extract frames 
            # 3. Process frames through detection -> reid -> mapping pipeline
            # 4. Stream results via WebSocket
            
            # For now, simulate the pipeline stages
            import asyncio
            
            stages = [
                ("DOWNLOADING", 0.3, "Downloading video segments"),
                ("EXTRACTING", 0.5, "Extracting frames from video"),
                ("PROCESSING", 0.7, "Running AI detection and tracking"),
                ("STREAMING", 0.9, "Streaming results to frontend"),
            ]
            
            for stage, progress, step_desc in stages:
                await self.update_task_status(task_id, stage, progress, step_desc)
                
                # Simulate processing time
                await asyncio.sleep(2.0)
                
                # Example frame batch processing (simplified)
                if stage == "PROCESSING":
                    try:
                        # Create a mock frame batch
                        mock_frame_batch = {
                            "task_id": str(task_id),
                            "environment_id": environment_id,
                            "camera_frames": {
                                "c01": {"frame_data": "mock_frame_data_c01"},
                                "c02": {"frame_data": "mock_frame_data_c02"}
                            },
                            "batch_index": 0,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Process the batch (this will use the actual pipeline components)
                        results = await self.process_frame_batch(task_id, mock_frame_batch)
                        
                        logger.info(f"Processed frame batch for task {task_id}: {len(results.get('detection_results', {}).get('detections', []))} detections")
                        
                    except Exception as e:
                        logger.warning(f"Frame processing failed for task {task_id}: {e}")
                        # Continue pipeline even if frame processing fails
            
            # Complete the task successfully
            await self.complete_task(task_id, success=True)
            logger.info(f"Processing pipeline completed successfully for task {task_id}")
            
        except Exception as e:
            logger.error(f"Processing pipeline failed for task {task_id}: {e}")
            await self.complete_task(task_id, success=False)


# Global orchestrator instance
orchestrator = PipelineOrchestrator()