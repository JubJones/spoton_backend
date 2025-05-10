import asyncio
import uuid
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone
import time
from pathlib import Path

from app.services.video_data_manager_service import VideoDataManagerService, BatchedFrameProvider
from app.services.multi_camera_frame_processor import MultiCameraFrameProcessor
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.reid_components import ReIDStateManager # For type hint and instantiation
from app.services.notification_service import NotificationService
from app.core.config import settings
from app.api.v1 import schemas as api_schemas # For WebSocket message structure
from app.common_types import CameraID, TrackedObjectData # For type hints

logger = logging.getLogger(__name__)

# In-memory store for task states. Consider Redis/DB for production.
PROCESSING_TASKS_DB: Dict[uuid.UUID, Dict[str, Any]] = {}
# In-memory store for ReIDStateManager instances per task
TASK_REID_MANAGERS: Dict[uuid.UUID, ReIDStateManager] = {}


class PipelineOrchestratorService:
    """
    Orchestrates the entire multi-camera video processing pipeline:
    - Task initialization and status management.
    - Sequential processing of sub-video batches across multiple cameras.
    - Frame batch provision to the MultiCameraFrameProcessor.
    - Notifications.
    """
    def __init__(
        self,
        video_data_manager: VideoDataManagerService,
        multi_camera_processor: MultiCameraFrameProcessor,
        tracker_factory: CameraTrackerFactory, # Added
        notification_service: NotificationService,
        # Detector and device are now primarily used by MultiCameraFrameProcessor
    ):
        self.video_data_manager = video_data_manager
        self.multi_camera_processor = multi_camera_processor
        self.tracker_factory = tracker_factory
        self.notification_service = notification_service
        # _models_loaded flag might now belong to MultiCameraFrameProcessor if it loads them,
        # or if models are passed pre-loaded. Dependencies.py handles loading.
        logger.info("PipelineOrchestratorService initialized.")

    async def initialize_task(self, environment_id: str) -> uuid.UUID:
        """Initializes a task state and its ReIDStateManager."""
        task_id = uuid.uuid4()
        current_time_utc = datetime.now(timezone.utc)
        PROCESSING_TASKS_DB[task_id] = {
            "status": "QUEUED",
            "progress": 0.0,
            "current_step": "Task Queued",
            "details": f"Task {task_id} for environment '{environment_id}' queued.",
            "start_time": current_time_utc,
            "last_update_time": current_time_utc,
            "environment_id": environment_id,
            "total_sub_video_batches": 0, # Will be updated
            "completed_sub_video_batches": 0,
            "total_frames_estimate": 0, # Will be updated
            "processed_frames_count": 0
        }
        # Create and store a ReIDStateManager for this task
        TASK_REID_MANAGERS[task_id] = ReIDStateManager(task_id=task_id)
        logger.info(f"Processing task {task_id} initialized for environment: {environment_id}. ReIDStateManager created.")
        await self._send_task_status_notification(task_id)
        return task_id

    async def _send_task_status_notification(self, task_id: uuid.UUID):
        """Helper to prepare and send a serializable task status update via WebSocket."""
        if task_id not in PROCESSING_TASKS_DB: return
        task_data = PROCESSING_TASKS_DB[task_id]
        serializable_task_data = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in task_data.items()}
        await self.notification_service.send_status_update(str(task_id), serializable_task_data)

    async def update_task_status(self, task_id: uuid.UUID, status: Optional[str] = None,
                                 progress: Optional[float] = None, current_step: Optional[str] = None,
                                 details: Optional[str] = None, increment_processed_frames: Optional[int] = None):
        """Updates task status and sends notification. Progress is overall task progress."""
        if task_id not in PROCESSING_TASKS_DB:
            logger.warning(f"Attempted to update status for unknown task_id: {task_id}")
            return

        task_data = PROCESSING_TASKS_DB[task_id]
        task_data["last_update_time"] = datetime.now(timezone.utc)
        if status: task_data["status"] = status
        if current_step: task_data["current_step"] = current_step
        if details: task_data["details"] = details
        
        if increment_processed_frames:
            task_data["processed_frames_count"] = task_data.get("processed_frames_count", 0) + increment_processed_frames

        # Calculate overall progress if components are available
        if task_data["total_sub_video_batches"] > 0 and task_data["total_frames_estimate"] > 0:
            # Progress based on sub-video batches primarily, then frames within current batch
            # This is a rough estimate.
            progress_from_sub_videos = (task_data["completed_sub_video_batches"] / task_data["total_sub_video_batches"]) * 0.9 # 90% weight
            
            # Progress within current sub-video batch (remaining 10%)
            # This part is harder to estimate accurately without knowing frames per sub-video batch
            # For now, let's simplify: progress updates mainly when a sub-video batch completes.
            # Or, use `progress` argument if explicitly provided.
            if progress is not None:
                 task_data["progress"] = max(0.0, min(1.0, progress))
            else:
                 task_data["progress"] = max(0.0, min(1.0, progress_from_sub_videos))


        log_msg = f"Task {task_id} status: {task_data['status']}"
        if current_step: log_msg += f" - Step: {current_step}"
        if "progress" in task_data : log_msg += f" - Progress: {task_data['progress']:.2%}"
        if details: log_msg += f" - Details: {details}"
        logger.info(log_msg)
        await self._send_task_status_notification(task_id)

    async def get_task_status(self, task_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Retrieves the status of a processing task."""
        task_data = PROCESSING_TASKS_DB.get(task_id)
        if not task_data: return None
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in task_data.items()}

    async def run_processing_pipeline(self, task_id: uuid.UUID, environment_id: str):
        """
        Main background process for a task.
        Processes sub-video batches sequentially. For each sub-video batch,
        it gets frames synchronously across cameras and sends to MultiCameraFrameProcessor.
        """
        logger.info(f"[Task {task_id}] Starting processing pipeline for environment: {environment_id}")
        pipeline_start_time = time.time()
        reid_manager = TASK_REID_MANAGERS.get(task_id)
        if not reid_manager:
            logger.error(f"[Task {task_id}] ReIDStateManager not found! Aborting pipeline.")
            await self.update_task_status(task_id, status="FAILED", current_step="Initialization Error", details="ReID Manager missing.")
            return

        try:
            await self.update_task_status(task_id, status="INITIALIZING", progress=0.01, current_step="Preparing Models & Data")
            # Models are loaded via dependencies.py when MultiCameraFrameProcessor is injected.

            max_sub_video_batches = self.video_data_manager.get_max_sub_videos_for_environment(environment_id)
            if max_sub_video_batches == 0:
                raise ValueError(f"No sub-videos found or configured for environment '{environment_id}'.")
            
            PROCESSING_TASKS_DB[task_id]["total_sub_video_batches"] = max_sub_video_batches
            # Estimate total frames (can be refined if BatchedFrameProvider gives total per video)
            # Assuming roughly 2000-3000 frames per sub-video as a placeholder
            PROCESSING_TASKS_DB[task_id]["total_frames_estimate"] = max_sub_video_batches * (settings.TARGET_FPS * 80) # Approx 80s video @ target FPS
            
            overall_processed_frames_task_count = 0

            for sub_video_idx in range(max_sub_video_batches):
                sub_video_batch_start_time = time.time()
                current_sub_video_human_idx = sub_video_idx + 1
                await self.update_task_status(
                    task_id, status="DOWNLOADING_DATA",
                    current_step=f"Downloading sub-videos batch {current_sub_video_human_idx}/{max_sub_video_batches}"
                )

                local_video_paths_map = await self.video_data_manager.download_sub_videos_for_environment_batch(
                    task_id, environment_id, sub_video_idx
                )

                if not local_video_paths_map:
                    logger.warning(f"[Task {task_id}] No local videos obtained for sub-video batch {current_sub_video_human_idx}. Skipping.")
                    PROCESSING_TASKS_DB[task_id]["completed_sub_video_batches"] += 1
                    continue

                frame_provider = self.video_data_manager.get_batched_frame_provider(task_id, local_video_paths_map)
                
                # Reset all trackers for this task before processing a new set of sub-videos
                await self.tracker_factory.reset_all_trackers_for_task(task_id)
                logger.info(f"[Task {task_id}] Trackers reset for sub-video batch {current_sub_video_human_idx}.")

                frames_in_current_sub_video_batch = 0
                global_frame_idx_for_task = overall_processed_frames_task_count # For ReID state timestamping

                while True:
                    await self.update_task_status(
                        task_id, status="PROCESSING",
                        current_step=f"Processing frames for sub-video batch {current_sub_video_human_idx} (Frame {frames_in_current_sub_video_batch})"
                    )
                    
                    frame_batch_dict, any_active = await frame_provider.get_next_frame_batch()
                    if not any_active and not frame_batch_dict: # Check if any frames were returned at all
                        logger.info(f"[Task {task_id}] End of sub-video batch {current_sub_video_human_idx}.")
                        break
                    
                    if not frame_batch_dict: # No frames in this specific call, but provider might still be active
                        await asyncio.sleep(0.001) # Brief pause if one cycle yields no frames but provider active
                        continue

                    global_frame_idx_for_task += 1 # Increment for each batch processed
                    frames_in_current_sub_video_batch +=1
                    
                    # Delegate processing of this synchronized frame batch
                    # This returns Dict[CameraID, List[TrackedObjectData]]
                    # Note: MultiCameraFrameProcessor has been refactored to return data, not send notifications.
                    processed_data_batch = await self.multi_camera_processor.process_frame_batch(
                        task_id, reid_manager, frame_batch_dict, global_frame_idx_for_task
                    )

                    # Send notifications based on the processed data
                    current_frame_timestamp_iso = datetime.now(timezone.utc).isoformat()
                    for cam_id_notify, tracked_objects_list in processed_data_batch.items():
                        # Find the original frame_path for this camera from frame_batch_dict
                        original_frame_data = frame_batch_dict.get(cam_id_notify)
                        frame_path_for_notify = original_frame_data[1] if original_frame_data else f"unknown_path_cam_{cam_id_notify}"

                        # Convert TrackedObjectData to api_schemas.TrackedPersonData
                        api_tracked_persons = [
                            api_schemas.TrackedPersonData(
                                track_id=obj.track_id,
                                global_person_id=obj.global_person_id,
                                bbox_img=obj.bbox_xyxy, # Assuming BoundingBoxXYXY is List[float]
                                confidence=obj.confidence
                            ) for obj in tracked_objects_list
                        ]

                        ws_payload = {
                            "camera_id": cam_id_notify,
                            "frame_timestamp": current_frame_timestamp_iso,
                            "frame_path": frame_path_for_notify, # Pseudo path from BatchedFrameProvider
                            "tracking_data": [p.model_dump() for p in api_tracked_persons]
                        }
                        await self.notification_service.send_tracking_update(str(task_id), ws_payload)

                    await self.update_task_status(task_id, increment_processed_frames=len(frame_batch_dict))
                    await asyncio.sleep(0.001) # Yield control briefly

                frame_provider.close()
                overall_processed_frames_task_count += frames_in_current_sub_video_batch
                PROCESSING_TASKS_DB[task_id]["completed_sub_video_batches"] += 1
                logger.info(
                    f"[Task {task_id}] Finished sub-video batch {current_sub_video_human_idx} "
                    f"in {(time.time() - sub_video_batch_start_time):.2f}s. "
                    f"Processed {frames_in_current_sub_video_batch} frame sets."
                )
            
            await self.update_task_status(task_id, status="COMPLETED", progress=1.0, current_step="All Processing Complete")
            pipeline_duration = time.time() - pipeline_start_time
            logger.info(f"[Task {task_id}] Pipeline completed in {pipeline_duration:.2f} seconds.")

        except ValueError as ve: # Specific error for configuration issues
            logger.error(f"[Task {task_id}] Configuration or setup error: {ve}", exc_info=True)
            await self.update_task_status(task_id, "FAILED", current_step="Configuration Error", details=str(ve))
        except Exception as e:
            pipeline_duration = time.time() - pipeline_start_time
            error_msg = f"Task failed after {pipeline_duration:.2f} seconds: {str(e)}"
            logger.exception(f"[Task {task_id}] Pipeline execution failed.")
            await self.update_task_status(task_id, "FAILED", current_step="Pipeline Failed", details=error_msg)
        finally:
            await self.video_data_manager.cleanup_task_data(task_id)
            await self.tracker_factory.clear_trackers_for_task(task_id)
            TASK_REID_MANAGERS.pop(task_id, None) # Clean up ReID manager
            logger.info(f"[Task {task_id}] Pipeline execution finished (Status: {PROCESSING_TASKS_DB.get(task_id, {}).get('status')}). Cleaned up resources.")