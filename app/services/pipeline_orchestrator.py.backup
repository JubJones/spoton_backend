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
from app.services.reid_components import ReIDStateManager
from app.services.notification_service import NotificationService
from app.core.config import settings
from app.api.v1 import schemas as api_schemas
from app.common_types import CameraID, FrameData # Added FrameData
from app.api.v1.schemas import (
    # MediaURLEntry, # No longer used here
    # WebSocketMediaAvailablePayload, # No longer used here
    WebSocketBatchProcessingCompletePayload,
    WebSocketTrackingMessagePayload
)
from app.utils.video_processing import encode_frame_to_base64 # Import new utility

logger = logging.getLogger(__name__)

PROCESSING_TASKS_DB: Dict[uuid.UUID, Dict[str, Any]] = {}
TASK_REID_MANAGERS: Dict[uuid.UUID, ReIDStateManager] = {}


class PipelineOrchestratorService:
    def __init__(
        self,
        video_data_manager: VideoDataManagerService,
        multi_camera_processor: MultiCameraFrameProcessor,
        tracker_factory: CameraTrackerFactory,
        notification_service: NotificationService,
    ):
        self.video_data_manager = video_data_manager
        self.multi_camera_processor = multi_camera_processor
        self.tracker_factory = tracker_factory
        self.notification_service = notification_service
        logger.info("PipelineOrchestratorService initialized.")

    async def initialize_task(self, environment_id: str) -> uuid.UUID:
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
            "total_sub_video_batches": 0,
            "completed_sub_video_batches": 0,
            "current_global_frame_index_offset": 0,
            "processed_frames_count_total": 0
        }
        TASK_REID_MANAGERS[task_id] = ReIDStateManager(task_id=task_id)
        logger.info(f"Processing task {task_id} initialized for environment: {environment_id}. ReIDStateManager created.")
        await self._send_task_status_notification(task_id)
        return task_id

    async def _send_task_status_notification(self, task_id: uuid.UUID):
        if task_id not in PROCESSING_TASKS_DB: return
        task_data = PROCESSING_TASKS_DB[task_id]
        serializable_task_data = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in task_data.items()
            if k not in ["current_global_frame_index_offset"]
        }
        await self.notification_service.send_status_update(str(task_id), serializable_task_data)

    async def update_task_status(self, task_id: uuid.UUID, status: Optional[str] = None,
                                 progress: Optional[float] = None, current_step: Optional[str] = None,
                                 details: Optional[str] = None, increment_processed_frames: Optional[int] = None):
        if task_id not in PROCESSING_TASKS_DB:
            logger.warning(f"Attempted to update status for unknown task_id: {task_id}")
            return
        task_data = PROCESSING_TASKS_DB[task_id]
        task_data["last_update_time"] = datetime.now(timezone.utc)
        if status: task_data["status"] = status
        if current_step: task_data["current_step"] = current_step
        if details: task_data["details"] = details

        if increment_processed_frames:
            task_data["processed_frames_count_total"] = task_data.get("processed_frames_count_total", 0) + increment_processed_frames

        if task_data["total_sub_video_batches"] > 0:
            overall_progress = (task_data["completed_sub_video_batches"] / task_data["total_sub_video_batches"])
            task_data["progress"] = max(0.0, min(1.0, progress if progress is not None else overall_progress))
        else:
            task_data["progress"] = progress if progress is not None else 0.0

        log_msg = f"Task {task_id} status: {task_data['status']}"
        if current_step: log_msg += f" - Step: {current_step}"
        if "progress" in task_data : log_msg += f" - Progress: {task_data['progress']:.2%}"
        logger.info(log_msg)
        await self._send_task_status_notification(task_id)


    async def get_task_status(self, task_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        task_data = PROCESSING_TASKS_DB.get(task_id)
        if not task_data: return None
        serializable_task_data = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in task_data.items()
            if k not in ["current_global_frame_index_offset"]
        }
        return serializable_task_data

    async def run_processing_pipeline(self, task_id: uuid.UUID, environment_id: str):
        logger.info(f"[Task {task_id}] Starting processing pipeline for environment: {environment_id}")
        pipeline_start_time = time.time()
        reid_manager = TASK_REID_MANAGERS.get(task_id)
        if not reid_manager:
            logger.error(f"[Task {task_id}] ReIDStateManager not found! Aborting pipeline.")
            await self.update_task_status(task_id, status="FAILED", current_step="Initialization Error", details="ReID Manager missing.")
            return

        try:
            await self.update_task_status(task_id, status="INITIALIZING", progress=0.01, current_step="Preparing Models & Data")

            max_sub_video_batches = self.video_data_manager.get_max_sub_videos_for_environment(environment_id)
            if max_sub_video_batches == 0:
                raise ValueError(f"No sub-videos found or configured for environment '{environment_id}'.")

            PROCESSING_TASKS_DB[task_id]["total_sub_video_batches"] = max_sub_video_batches
            PROCESSING_TASKS_DB[task_id]["current_global_frame_index_offset"] = 0

            for sub_video_batch_idx in range(max_sub_video_batches):
                sub_video_batch_start_time = time.time()
                current_sub_video_human_idx = sub_video_batch_idx + 1
                start_global_frame_index_for_this_batch = PROCESSING_TASKS_DB[task_id]["current_global_frame_index_offset"]

                await self.update_task_status(
                    task_id, status="DOWNLOADING_DATA",
                    current_step=f"Downloading sub-videos batch {current_sub_video_human_idx}/{max_sub_video_batches}"
                )

                local_video_paths_map = await self.video_data_manager.download_sub_videos_for_environment_batch(
                    task_id, environment_id, sub_video_batch_idx
                )

                if not local_video_paths_map:
                    logger.warning(f"[Task {task_id}] No local videos for sub-video batch {current_sub_video_human_idx}. Skipping.")
                    PROCESSING_TASKS_DB[task_id]["completed_sub_video_batches"] += 1
                    await self.update_task_status(task_id)
                    continue

                frame_provider = self.video_data_manager.get_batched_frame_provider(task_id, local_video_paths_map)

                # The 'media_available' notification is removed. Frontend gets images in tracking_update.

                await self.tracker_factory.reset_all_trackers_for_task(task_id)
                logger.info(f"[Task {task_id}] Trackers reset for sub-video batch {current_sub_video_human_idx}.")

                frames_processed_in_this_batch = 0
                while True:
                    frame_batch_dict, any_active = await frame_provider.get_next_frame_batch()
                    if not any_active and not frame_batch_dict:
                        logger.info(f"[Task {task_id}] End of sub-video batch {current_sub_video_human_idx}.")
                        break

                    if not frame_batch_dict:
                        await asyncio.sleep(0.001)
                        continue
                    
                    current_global_frame_idx = start_global_frame_index_for_this_batch + frames_processed_in_this_batch
                    
                    await self.update_task_status(
                        task_id, status="PROCESSING",
                        current_step=(
                            f"Processing sub-video {current_sub_video_human_idx}, "
                            f"local frame batch {frames_processed_in_this_batch + 1} "
                            f"(Global Index: {current_global_frame_idx})"
                        )
                    )

                    processed_data_batch_output = await self.multi_camera_processor.process_frame_batch(
                        task_id, environment_id, reid_manager, frame_batch_dict, current_global_frame_idx
                    )

                    ws_cameras_data: Dict[str, api_schemas.CameraTracksData] = {}
                    all_relevant_cam_ids_for_env = {
                        vs_config.cam_id for vs_config in settings.VIDEO_SETS if vs_config.env_id == environment_id
                    }

                    # Encode frames and prepare payload
                    for cam_id_str_config in all_relevant_cam_ids_for_env:
                        cam_id_obj = CameraID(cam_id_str_config)
                        image_source_filename = f"{current_global_frame_idx:06d}.jpg" # Example identifier
                        
                        current_frame_data_for_cam: Optional[FrameData] = frame_batch_dict.get(cam_id_obj)
                        frame_image_b64: Optional[str] = None
                        if current_frame_data_for_cam:
                            frame_np, _ = current_frame_data_for_cam
                            if frame_np is not None:
                                frame_image_b64 = await encode_frame_to_base64(
                                    frame_np, jpeg_quality=settings.FRAME_JPEG_QUALITY
                                )
                        
                        tracks_for_this_cam_data = processed_data_batch_output.get(cam_id_obj, [])
                        
                        api_person_data_list = [
                            api_schemas.TrackedPersonData(
                                track_id=obj.track_id, global_id=obj.global_person_id,
                                bbox_xyxy=obj.bbox_xyxy, confidence=obj.confidence,
                                class_id=settings.PERSON_CLASS_ID, map_coords=obj.map_coords
                            ) for obj in tracks_for_this_cam_data
                        ]
                        ws_cameras_data[cam_id_str_config] = api_schemas.CameraTracksData(
                            image_source=image_source_filename,
                            frame_image_base64=frame_image_b64, # Add encoded image
                            tracks=api_person_data_list
                        )

                    tracking_payload = WebSocketTrackingMessagePayload(
                        global_frame_index=current_global_frame_idx,
                        scene_id=environment_id,
                        timestamp_processed_utc=datetime.now(timezone.utc).isoformat(),
                        cameras=ws_cameras_data
                    )
                    await self.notification_service.send_tracking_update(
                        str(task_id), tracking_payload.model_dump(exclude_none=True)
                    )
                    
                    frames_processed_in_this_batch += 1
                    await self.update_task_status(task_id, increment_processed_frames=1)
                    await asyncio.sleep(0.001) # Yield control

                frame_provider.close()
                
                PROCESSING_TASKS_DB[task_id]["current_global_frame_index_offset"] += frames_processed_in_this_batch
                
                batch_complete_payload = WebSocketBatchProcessingCompletePayload(
                    sub_video_batch_index=sub_video_batch_idx
                )
                # This notification's role for frontend is diminished, but can be kept for backend state logic
                await self.notification_service.send_batch_processing_complete_notification(
                    str(task_id), batch_complete_payload
                )

                PROCESSING_TASKS_DB[task_id]["completed_sub_video_batches"] += 1
                await self.update_task_status(task_id)
                logger.info(
                    f"[Task {task_id}] Finished sub-video batch {current_sub_video_human_idx} "
                    f"in {(time.time() - sub_video_batch_start_time):.2f}s. "
                    f"Processed {frames_processed_in_this_batch} frame batches from it."
                )

            await self.update_task_status(task_id, status="COMPLETED", progress=1.0, current_step="All Processing Complete")
            logger.info(
                f"[Task {task_id}] Pipeline completed in {time.time() - pipeline_start_time:.2f} seconds. "
                f"Total global frame batches processed: {PROCESSING_TASKS_DB[task_id]['processed_frames_count_total']}"
            )

        except ValueError as ve:
            logger.error(f"[Task {task_id}] Configuration or setup error: {ve}", exc_info=True)
            await self.update_task_status(task_id, "FAILED", current_step="Configuration Error", details=str(ve))
        except Exception as e:
            error_msg = f"Task failed: {str(e)}"
            logger.exception(f"[Task {task_id}] Pipeline execution failed.")
            await self.update_task_status(task_id, "FAILED", current_step="Pipeline Failed", details=error_msg)
        finally:
            await self.video_data_manager.cleanup_task_data(task_id)
            await self.tracker_factory.clear_trackers_for_task(task_id)
            if self.multi_camera_processor:
                 await self.multi_camera_processor.clear_task_resources(task_id, environment_id)
            TASK_REID_MANAGERS.pop(task_id, None)
            PROCESSING_TASKS_DB.get(task_id, {}).pop("current_global_frame_index_offset", None)
            logger.info(f"[Task {task_id}] Pipeline execution finished (Status: {PROCESSING_TASKS_DB.get(task_id, {}).get('status')}). Cleaned up resources.")