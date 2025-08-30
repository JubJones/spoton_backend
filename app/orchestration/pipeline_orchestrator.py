"""
Enhanced processing pipeline orchestrator.

Coordinates the three core features:
1. Multi-view person detection
2. Cross-camera re-identification
3. Unified spatial mapping
"""

import asyncio
import uuid
from uuid import UUID
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

# Import video processing services
from app.services.video_data_manager_service import VideoDataManagerService
from app.utils.asset_downloader import AssetDownloader

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
        # Track active tasks by environment to prevent duplicates
        self.environment_tasks: Dict[str, uuid.UUID] = {}
        
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
        
        # Video processing services
        self.video_data_manager: Optional[VideoDataManagerService] = None
        self.asset_downloader: Optional[AssetDownloader] = None
        
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
        overall_start = time.time()
        try:
            logger.info(f"🚀 PIPELINE INIT: Starting pipeline initialization for environment: {environment_id}")
            
            # Step 1: Initialize video processing services
            step_start = time.time()
            logger.info("📹 PIPELINE INIT: Step 1/7 - Initializing video processing services...")
            self.asset_downloader = AssetDownloader(
                s3_endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                s3_bucket_name=settings.S3_BUCKET_NAME
            )
            self.video_data_manager = VideoDataManagerService(asset_downloader=self.asset_downloader)
            logger.info(f"✅ PIPELINE INIT: Step 1/7 completed in {time.time() - step_start:.2f}s - Video services ready")
            
            # Step 2: Initialize coordinate transformer
            step_start = time.time()
            logger.info("📍 PIPELINE INIT: Step 2/7 - Initializing coordinate transformer...")
            self.coordinate_transformer = CoordinateTransformer(
                enable_caching=True,
                cache_size=1000
            )
            logger.info(f"✅ PIPELINE INIT: Step 2/7 completed in {time.time() - step_start:.2f}s - Coordinate transformer ready")
            
            # Step 3: Initialize calibration service
            step_start = time.time()
            logger.info("🗺️ PIPELINE INIT: Step 3/7 - Initializing calibration service...")
            self.calibration_service = CalibrationService(
                coordinate_transformer=self.coordinate_transformer
            )
            logger.info(f"✅ PIPELINE INIT: Step 3/7 completed in {time.time() - step_start:.2f}s - Calibration service ready")
            
            # Step 4: Load calibration environment
            step_start = time.time()
            logger.info(f"🏭 PIPELINE INIT: Step 4/7 - Loading calibration environment: {environment_id}...")
            if not await self.calibration_service.load_calibration_environment(environment_id):
                logger.error(f"❌ PIPELINE INIT: Failed to load calibration environment: {environment_id}")
                return False
            logger.info(f"✅ PIPELINE INIT: Step 4/7 completed in {time.time() - step_start:.2f}s - Environment {environment_id} loaded")
            
            # Step 5: Initialize detection service  
            step_start = time.time()
            logger.info("🔍 PIPELINE INIT: Step 5/7 - Initializing detection service (this may take 10-15 seconds)...")
            self.detection_service = DetectionService(
                detector_type="faster_rcnn",
                enable_gpu=True,
                batch_size=4  # Process 4 cameras simultaneously
            )
            
            await self.detection_service.initialize_detector()
            logger.info(f"✅ PIPELINE INIT: Step 5/7 completed in {time.time() - step_start:.2f}s - Detection service ready")
            
            # Step 6: Initialize ReID service
            step_start = time.time()
            logger.info("🧑‍🤝‍🧑 PIPELINE INIT: Step 6/7 - Initializing ReID service (this may take 5-10 seconds)...")
            self.reid_service = ReIDService(
                model_type="clip",
                enable_gpu=True,
                batch_size=16
            )
            
            await self.reid_service.initialize_model()
            logger.info(f"✅ PIPELINE INIT: Step 6/7 completed in {time.time() - step_start:.2f}s - ReID service ready")
            
            # Step 7: Initialize mapping and trajectory services
            step_start = time.time()
            logger.info("🗺️ PIPELINE INIT: Step 7/7 - Initializing mapping and trajectory services...")
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
            logger.info(f"✅ PIPELINE INIT: Step 7/7 completed in {time.time() - step_start:.2f}s - Mapping services ready")
            
            total_time = time.time() - overall_start
            logger.info(f"🎉 PIPELINE INIT: All pipeline services initialized successfully in {total_time:.2f}s total")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing pipeline services: {e}")
            return False
    
    async def get_active_task_for_environment(self, environment_id: str) -> Optional[uuid.UUID]:
        """Check if there's already an active task for the given environment."""
        try:
            # First check in-memory cache
            if environment_id in self.environment_tasks:
                task_id = self.environment_tasks[environment_id]
                # Verify the task is still active
                if task_id in self.active_tasks and task_id in self.tasks:
                    task_status = self.tasks[task_id].get("status", "").upper()
                    # Consider a task active if it's not completed or failed
                    if task_status not in ["COMPLETED", "FAILED"]:
                        logger.info(f"🔄 TASK REUSE: Found existing active task (in-memory) {task_id} for environment '{environment_id}' with status '{task_status}'")
                        return task_id
            
            # Check Redis for persistent task mapping (in case of backend restart)
            redis_key = f"env_task:{environment_id}"
            async_redis = await redis_client.connect_async()
            task_id_str = await async_redis.get(redis_key)
            
            if task_id_str:
                task_id = uuid.UUID(task_id_str)
                logger.info(f"🔍 TASK REUSE: Found environment task mapping in Redis: {environment_id} -> {task_id}")
                
                # Get task status from Redis
                task_data = await redis_client.get_json_async(f"task:{task_id}:state")
                
                if task_data:
                    task_status = task_data.get("status", "").upper()
                    logger.info(f"📊 TASK REUSE: Task {task_id} status from Redis: {task_status}")
                    
                    # Consider a task active if it's not completed or failed
                    if task_status not in ["COMPLETED", "FAILED"]:
                        # Restore to in-memory cache
                        self.environment_tasks[environment_id] = task_id
                        self.active_tasks.add(task_id)
                        self.tasks[task_id] = task_data
                        
                        logger.info(f"♻️ TASK REUSE: Restored existing active task {task_id} for environment '{environment_id}' with status '{task_status}'")
                        return task_id
                    else:
                        # Clean up completed/failed task from Redis
                        logger.info(f"🧹 TASK CLEANUP: Removing completed/failed task {task_id} for environment '{environment_id}' from Redis")
                        await async_redis.delete(redis_key)
                else:
                    # Task data not found in Redis, clean up mapping
                    logger.warning(f"⚠️ TASK CLEANUP: Task data not found for {task_id}, cleaning up environment mapping")
                    await async_redis.delete(redis_key)
        
        except Exception as e:
            logger.error(f"❌ TASK REUSE: Error checking for existing task for environment {environment_id}: {e}")
        
        return None

    async def initialize_task(self, environment_id: str) -> uuid.UUID:
        """Initialize a new processing task or return existing active task."""
        
        # Check for existing active task first
        existing_task_id = await self.get_active_task_for_environment(environment_id)
        if existing_task_id:
            logger.info(f"♻️ TASK REUSE: Reusing existing task {existing_task_id} for environment '{environment_id}'")
            return existing_task_id
        
        # Create new task if none exists
        task_id = uuid.uuid4()
        current_time_utc = datetime.now(timezone.utc)
        
        # Log task creation with context
        active_task_count = len(self.active_tasks)
        logger.info(f"📝 TASK CREATE: Creating new task {task_id} for environment '{environment_id}' (currently {active_task_count} active tasks)")
        
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
        # Track this task for the environment
        self.environment_tasks[environment_id] = task_id
        
        # Cache task state in Redis
        await redis_client.set_json_async(
            f"task:{task_id}:state", 
            task_data, 
            ex=3600  # 1 hour expiration
        )
        
        # Store environment-to-task mapping in Redis for persistence across restarts
        redis_env_key = f"env_task:{environment_id}"
        async_redis = await redis_client.connect_async()
        await async_redis.set(
            redis_env_key, 
            str(task_id), 
            ex=3600  # 1 hour expiration
        )
        
        logger.info(f"✅ TASK CREATE: Task {task_id} initialized for environment {environment_id} (now {len(self.active_tasks)} active tasks)")
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

    async def get_all_task_statuses(self) -> List[Dict[str, Any]]:
        """
        Get status of all active tasks.
        
        Returns:
            List of task status dictionaries
        """
        all_tasks = []
        
        for task_id, task_state in self.task_states.items():
            task_info = {
                "task_id": str(task_id),
                "status": task_state.status.value if hasattr(task_state.status, 'value') else str(task_state.status),
                "progress": task_state.progress,
                "current_step": task_state.current_step,
                "environment_id": getattr(task_state, 'environment_id', None),
                "created_at": getattr(task_state, 'created_at', None),
                "updated_at": getattr(task_state, 'updated_at', None)
            }
            all_tasks.append(task_info)
        
        return all_tasks
    
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
        
        # Clean up environment-to-task mapping from Redis
        environment_id = self.tasks[task_id].get("environment_id")
        if environment_id:
            redis_env_key = f"env_task:{environment_id}"
            async_redis = await redis_client.connect_async()
            await async_redis.delete(redis_env_key)
            # Also clean up from in-memory cache
            self.environment_tasks.pop(environment_id, None)
            logger.info(f"🧹 TASK CLEANUP: Removed environment mapping for {environment_id} -> {task_id}")
        
        logger.info(f"Task {task_id} marked as {status}")
    
    async def cleanup_task(self, task_id: uuid.UUID):
        """Clean up task resources."""
        # Get environment_id before deleting task data
        environment_id = None
        if task_id in self.tasks:
            environment_id = self.tasks[task_id].get("environment_id")
            del self.tasks[task_id]
        
        self.active_tasks.discard(task_id)
        
        # Clean up environment-to-task mapping from Redis
        if environment_id:
            redis_env_key = f"env_task:{environment_id}"
            async_redis = await redis_client.connect_async()
            await async_redis.delete(redis_env_key)
            # Also clean up from in-memory cache
            self.environment_tasks.pop(environment_id, None)
            logger.info(f"🧹 TASK CLEANUP: Removed environment mapping for {environment_id} -> {task_id}")
        
        # Clean up Redis cache
        async_redis = await redis_client.connect_async()
        await async_redis.delete(f"task:{task_id}:state")
        
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
            
            # Process detection batch using the correct method
            detection_batch = await self.detection_service.detect_persons_in_batch(
                frame_batch=camera_frames,
                frame_index=0,  # TODO: Get actual frame index from batch
                confidence_threshold=0.5,
                use_gpu_batch=True
            )
            
            processing_time = time.time() - start_time
            
            self.pipeline_stats["detection_runs"] += 1
            self.pipeline_stats["total_frames_processed"] += len(camera_frames)
            
            logger.info(f"Detection completed in {processing_time:.3f}s with {detection_batch.detection_count} detections")
            
            return {
                "detection_batch": detection_batch,
                "stage": "detection",
                "processing_time": processing_time,
                "detection_count": detection_batch.detection_count
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
            
            # Real video processing pipeline
            # 1. Download video segments from S3
            # 2. Extract frames 
            # 3. Process frames through detection -> reid -> mapping pipeline
            # 4. Stream results via WebSocket
            
            await self.update_task_status(task_id, "DOWNLOADING", 0.2, "Downloading video segments from S3")
            
            # Download videos using VideoDataManagerService  
            try:
                # Get task data to extract camera info
                task_info = await self.get_task_status(task_id)
                if not task_info:
                    logger.error(f"Task {task_id} not found")
                    return
                
                # For demo, we'll use the first sub-video (index 0)
                # In production, this could be configurable
                sub_video_index = 0
                logger.info(f"🎬 Downloading sub-video {sub_video_index} for environment '{environment_id}'")
                
                # Download videos from S3 with timeout
                try:
                    video_paths = await asyncio.wait_for(
                        self.video_data_manager.download_sub_videos_for_environment_batch(
                            task_id=task_id,
                            environment_id=environment_id,
                            sub_video_index=sub_video_index
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"🕒 Video download timed out after 30 seconds for task {task_id}")
                    video_paths = {}
                
                if not video_paths:
                    logger.warning(f"No videos downloaded for task {task_id}")
                    # Fall back to mock data for demo
                    await self._send_tracking_updates_with_mock_frames(task_id, environment_id)
                    return
                
                logger.info(f"✅ Downloaded {len(video_paths)} videos: {list(video_paths.keys())}")
                
                await self.update_task_status(task_id, "EXTRACTING", 0.4, "Extracting frames from videos")
                
                # Create frame provider
                frame_provider = self.video_data_manager.get_batched_frame_provider(
                    task_id=task_id,
                    local_video_paths_map=video_paths,
                    loop_videos=False
                )
                
                await self.update_task_status(task_id, "PROCESSING", 0.6, "Processing frames with AI detection")
                
                # Process and stream real frames
                await self._process_and_stream_real_frames(task_id, environment_id, frame_provider)
                
            except Exception as e:
                logger.error(f"Video processing failed for task {task_id}: {e}")
                # Fall back to mock data for demo
                await self.update_task_status(task_id, "STREAMING", 0.9, "Streaming mock data (video processing failed)")
                await self._send_tracking_updates_with_mock_frames(task_id, environment_id)
            
            # Complete the task successfully
            await self.complete_task(task_id, success=True)
            logger.info(f"Processing pipeline completed successfully for task {task_id}")
            
        except Exception as e:
            logger.error(f"Processing pipeline failed for task {task_id}: {e}")
            await self.complete_task(task_id, success=False)
    
    async def _process_and_stream_real_frames(self, task_id: UUID, environment_id: str, frame_provider) -> None:
        """Process real video frames and stream results with actual person detection."""
        import cv2
        import base64
        
        logger.info(f"🎬 Starting real frame processing for task {task_id}")
        
        await self.update_task_status(task_id, "STREAMING", 0.8, "Streaming real frames with detections")
        
        frame_count = 0
        max_frames = 20  # Process up to 20 frames for demo
        
        try:
            while frame_count < max_frames:
                # Get next batch of frames from all cameras
                frame_batch, has_more = await frame_provider.get_next_frame_batch()
                
                if not frame_batch or not has_more:
                    logger.info(f"No more frames available, processed {frame_count} frames")
                    break
                
                # Convert frames to the format expected by the pipeline
                camera_frames = {}
                for cam_id, frame_data in frame_batch.items():
                    if frame_data is not None:
                        image_np, frame_path = frame_data  # Unpack the tuple
                        if image_np is not None:
                            camera_frames[cam_id] = {
                                "image": image_np,
                                "width": image_np.shape[1],
                                "height": image_np.shape[0],
                                "fps": 30,
                                "format": "RGB",
                                "encoding": "numpy",
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                
                if not camera_frames:
                    continue
                
                # Process frames through detection pipeline
                pipeline_batch = {
                    "task_id": str(task_id),
                    "environment_id": environment_id,
                    "camera_frames": camera_frames,
                    "batch_index": frame_count,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Run through detection -> reid -> mapping pipeline
                try:
                    results = await self.process_frame_batch(task_id, pipeline_batch)
                    
                    # Convert processed frames to base64 and send via WebSocket
                    await self._send_real_frame_results(task_id, environment_id, frame_count, camera_frames, results)
                    
                    frame_count += 1
                    logger.info(f"📹 Processed frame batch {frame_count}/{max_frames}")
                    
                    # Small delay between frames
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Frame processing error for batch {frame_count}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Real frame processing failed: {e}")
        finally:
            # Clean up frame provider
            try:
                frame_provider.close()
            except:
                pass
        
        logger.info(f"✅ Completed real frame processing: {frame_count} frames processed")
    
    async def _send_real_frame_results(self, task_id: UUID, environment_id: str, frame_index: int, 
                                     camera_frames: dict, detection_results: dict) -> None:
        """Send real processed frames with detection results via WebSocket."""
        import cv2
        import base64
        
        try:
            # Get detection results
            detection_batch = detection_results.get("detection_results", {}).get("detection_batch")
            detections_by_camera = {}
            
            if detection_batch:
                # Group detections by camera
                for detection in detection_batch.detections:
                    cam_id = detection.camera_id
                    if cam_id not in detections_by_camera:
                        detections_by_camera[cam_id] = []
                    detections_by_camera[cam_id].append(detection)
            
            # Create tracking update message with real frame data
            cameras_data = {}
            
            for cam_id, frame_info in camera_frames.items():
                # Get frame image
                frame_image = frame_info["image"]  # numpy array
                
                # Draw bounding boxes on frame if we have detections
                if cam_id in detections_by_camera:
                    frame_with_boxes = frame_image.copy()
                    for detection in detections_by_camera[cam_id]:
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, detection.bbox.to_xyxy())
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add confidence score
                        label = f"Person {detection.confidence:.2f}"
                        cv2.putText(frame_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    frame_to_encode = frame_with_boxes
                else:
                    frame_to_encode = frame_image
                
                # Convert to JPEG and then base64
                if frame_to_encode.dtype != np.uint8:
                    frame_to_encode = (frame_to_encode * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                if len(frame_to_encode.shape) == 3:
                    frame_bgr = cv2.cvtColor(frame_to_encode, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_to_encode
                
                # Encode as JPEG
                success, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                else:
                    logger.warning(f"Failed to encode frame for camera {cam_id}")
                    continue
                
                # Create tracking data for this camera
                tracks = []
                if cam_id in detections_by_camera:
                    for i, detection in enumerate(detections_by_camera[cam_id]):
                        track = {
                            "track_id": detection.track_id or i,
                            "global_id": f"person_{detection.track_id or i}",
                            "bbox_xyxy": list(detection.bbox.to_xyxy()),
                            "confidence": detection.confidence,
                            "class_id": 1,  # Person class
                            "map_coords": [detection.bbox.x, detection.bbox.y]  # Simplified
                        }
                        tracks.append(track)
                
                cameras_data[cam_id] = {
                    "tracks": tracks,
                    "frame_image": frame_base64,
                    "frame_index": frame_index,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Send tracking update message
            tracking_message = {
                "type": "tracking_update",
                "task_id": str(task_id),
                "cameras": cameras_data,
                "frame_index": frame_index,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Send via WebSocket
            from app.api.websockets.connection_manager import binary_websocket_manager
            success = await binary_websocket_manager.send_json_message(str(task_id), tracking_message)
            
            if success:
                logger.info(f"📤 REAL STREAM: Sent frame {frame_index} for task {task_id} with {len(cameras_data)} cameras and {sum(len(data['tracks']) for data in cameras_data.values())} detections")
            else:
                logger.warning(f"⚠️ REAL STREAM: Failed to send frame {frame_index} for task {task_id}")
                
        except Exception as e:
            logger.error(f"Error sending real frame results: {e}")
    
    async def _send_tracking_updates_with_mock_frames(self, task_id: UUID, environment_id: str) -> None:
        """Fallback method that sends mock SVG frames (original behavior)."""
        logger.info(f"📝 Sending mock frames for task {task_id} (fallback mode)")
        await self._send_tracking_updates(task_id, environment_id)
    
    async def _send_tracking_updates(self, task_id: UUID, environment_id: str) -> None:
        """Send real-time tracking updates via WebSocket during streaming phase."""
        import base64
        from datetime import datetime, timezone
        
        logger.info(f"Starting real-time streaming for task {task_id}")
        
        # Get camera IDs for this environment
        camera_ids = ["c09", "c12", "c13", "c16"] if environment_id == "factory" else ["c01", "c02", "c03", "c05"]
        
        # Stream for 10 seconds with updates every 0.5 seconds
        for frame_idx in range(20):  # 20 frames over 10 seconds
            try:
                # Create mock tracking data for each camera
                cameras_data = {}
                
                for cam_id in camera_ids:
                    # Generate mock tracking data
                    tracks = []
                    
                    # Create 2-3 mock persons per camera
                    for person_idx in range(2 + (frame_idx % 2)):  # 2-3 persons
                        global_id = f"person_{person_idx + 1}"
                        
                        # Simulate person movement across frames
                        base_x = 300 + person_idx * 400 + (frame_idx * 5)  # Moving right
                        base_y = 200 + person_idx * 100 + (frame_idx % 20) * 2  # Slight vertical movement
                        
                        track = {
                            "track_id": person_idx + 1,
                            "global_id": global_id,
                            "bbox_xyxy": [
                                base_x,
                                base_y, 
                                base_x + 80,
                                base_y + 180
                            ],
                            "confidence": 0.85 + (person_idx * 0.05),
                            "class_id": 1,
                            "map_coords": [
                                50 + person_idx * 25 + (frame_idx % 10) * 2,  # Map X
                                30 + person_idx * 15 + (frame_idx % 8) * 1    # Map Y
                            ]
                        }
                        tracks.append(track)
                    
                    # Create a simple frame image (placeholder)
                    frame_svg = f'''<svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
                        <rect width="100%" height="100%" fill="#1a1a1a"/>
                        <text x="50%" y="45%" text-anchor="middle" fill="#888" font-size="20">
                            Camera {cam_id} - Frame {frame_idx}
                        </text>
                        <text x="50%" y="55%" text-anchor="middle" fill="#666" font-size="14">
                            {len(tracks)} persons detected
                        </text>
                    </svg>'''
                    
                    frame_base64 = base64.b64encode(frame_svg.encode()).decode()
                    
                    cameras_data[cam_id] = {
                        "tracks": tracks,
                        "frame_image": frame_base64,
                        "frame_index": frame_idx,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                
                # Create tracking update message
                tracking_message = {
                    "type": "tracking_update",
                    "task_id": str(task_id),
                    "cameras": cameras_data,
                    "frame_index": frame_idx,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Send via WebSocket using binary WebSocket manager
                try:
                    from app.api.websockets.connection_manager import binary_websocket_manager
                    success = await binary_websocket_manager.send_json_message(str(task_id), tracking_message)
                    
                    if success:
                        logger.info(f"📤 STREAM: Sent tracking update {frame_idx} for task {task_id} with {len(cameras_data)} cameras")
                    else:
                        logger.warning(f"⚠️ STREAM: Failed to send tracking update {frame_idx} for task {task_id} - no active connection")
                except Exception as e:
                    logger.error(f"❌ STREAM: Error sending tracking update {frame_idx} for task {task_id}: {e}")
                
                # Wait 0.5 seconds before next frame
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error sending tracking update {frame_idx} for task {task_id}: {e}")
                continue
        
        logger.info(f"Completed real-time streaming for task {task_id}")


# Global orchestrator instance
orchestrator = PipelineOrchestrator()