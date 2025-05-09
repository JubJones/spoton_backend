import asyncio
import uuid
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone # Import timezone
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from app.services.video_data_manager_service import VideoDataManagerService
from app.services.notification_service import NotificationService
from app.models.base_models import AbstractDetector, AbstractTracker, BoundingBox, TrackedObject
from app.core.config import settings
from app.api.v1 import schemas

logger = logging.getLogger(__name__)

PROCESSING_TASKS_DB: Dict[uuid.UUID, Dict[str, Any]] = {}

class PipelineOrchestratorService:
    """
    Orchestrates the video processing pipeline: download, frame extraction,
    detection, tracking with Re-ID, and notification.
    """
    def __init__(
        self,
        video_data_manager: VideoDataManagerService,
        detector: AbstractDetector,
        tracker: AbstractTracker,
        notification_service: NotificationService,
        device: torch.device
    ):
        self.video_data_manager = video_data_manager
        self.detector = detector
        self.tracker = tracker
        self.notification_service = notification_service
        self.device = device
        self._models_loaded = False
        logger.info("PipelineOrchestratorService initialized.")

    async def _ensure_models_loaded(self):
        """Loads models if they haven't been loaded yet."""
        if not self._models_loaded:
            logger.info("Loading models for the pipeline...")
            start_time = time.time()
            try:
                await asyncio.gather(
                    self.detector.load_model(),
                    self.tracker.load_model()
                )
                self._models_loaded = True
                load_time = time.time() - start_time
                logger.info(f"Detector and Tracker models loaded successfully in {load_time:.2f} seconds.")
            except Exception as e:
                logger.exception("Failed to load models.")
                self._models_loaded = False
                raise RuntimeError("Model loading failed, cannot proceed.") from e
        else:
             logger.debug("Models already loaded.")

    async def initialize_task(self, environment_id: str) -> uuid.UUID:
        """Initializes a task state in the database."""
        task_id = uuid.uuid4()
        current_time_utc = datetime.now(timezone.utc) # Use timezone-aware datetime
        PROCESSING_TASKS_DB[task_id] = {
            "status": "QUEUED",
            "progress": 0.0,
            "current_step": "Task Queued",
            "details": f"Task {task_id} for environment '{environment_id}' has been queued.",
            "start_time": current_time_utc, # Store as datetime object
            "last_update_time": current_time_utc # Store as datetime object
        }
        logger.info(f"Processing task {task_id} initialized for environment: {environment_id}")
        # Send initial queued status
        await self._send_task_status_notification(task_id)
        return task_id

    async def _send_task_status_notification(self, task_id: uuid.UUID):
        """Helper to prepare and send a serializable task status update."""
        if task_id not in PROCESSING_TASKS_DB:
            return

        task_data = PROCESSING_TASKS_DB[task_id]
        
        # Create a serializable copy for notification
        serializable_task_data = task_data.copy()
        for key, value in serializable_task_data.items():
            if isinstance(value, datetime):
                serializable_task_data[key] = value.isoformat() # Convert datetime to ISO string

        await self.notification_service.send_status_update(str(task_id), serializable_task_data)


    async def update_task_status(self, task_id: uuid.UUID, status: str, progress: Optional[float] = None,
                                 current_step: Optional[str] = None, details: Optional[str] = None):
        """Updates the status of a task and sends a notification."""
        if task_id not in PROCESSING_TASKS_DB:
            logger.warning(f"Attempted to update status for unknown task_id: {task_id}")
            return

        task_data = PROCESSING_TASKS_DB[task_id]
        task_data["status"] = status
        task_data["last_update_time"] = datetime.now(timezone.utc) # Update with timezone-aware datetime
        if progress is not None:
            task_data["progress"] = max(0.0, min(1.0, progress))
        if current_step is not None:
            task_data["current_step"] = current_step
        if details is not None:
             task_data["details"] = details

        log_msg = f"Task {task_id} status updated: {status}"
        if current_step: log_msg += f" - Step: {current_step}"
        if progress is not None: log_msg += f" - Progress: {task_data['progress']:.2%}"
        if details: log_msg += f" - Details: {details}"
        logger.info(log_msg)

        # Send notification via WebSocket using the helper
        await self._send_task_status_notification(task_id)

    async def get_task_status(self, task_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Retrieves the status of a processing task."""
        task_data = PROCESSING_TASKS_DB.get(task_id)
        if not task_data:
            return None
        
        # Return a serializable copy for API responses as well
        serializable_task_data = task_data.copy()
        for key, value in serializable_task_data.items():
            if isinstance(value, datetime):
                serializable_task_data[key] = value.isoformat()
        return serializable_task_data

    def _parse_tracker_output(self, tracker_output_np: np.ndarray, cam_id: str) -> List[TrackedObject]:
        """Parses the raw numpy array output from the BoxMOT tracker."""
        parsed_objects: List[TrackedObject] = []
        if tracker_output_np is None or tracker_output_np.size == 0:
            return parsed_objects

        num_cols = tracker_output_np.shape[1]
        if num_cols < 5:
            logger.warning(f"[{cam_id}] Tracker output has too few columns ({num_cols}) to parse. Skipping.")
            return parsed_objects

        for row in tracker_output_np:
            try:
                x1, y1, x2, y2 = map(float, row[0:4])
                track_id = int(row[4])
                conf = float(row[5]) if num_cols > 5 and np.isfinite(row[5]) else None
                cls_id = int(row[6]) if num_cols > 6 and np.isfinite(row[6]) else None
                global_id_val = int(row[7]) if num_cols > 7 and np.isfinite(row[7]) else None
                global_id = None if global_id_val is not None and global_id_val < 0 else global_id_val

                if x2 <= x1 or y2 <= y1: continue
                if track_id < 0: continue

                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                tracked_obj = TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    global_id=global_id
                )
                parsed_objects.append(tracked_obj)
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(f"[{cam_id}] Error parsing tracker output row: {row}. Error: {e}", exc_info=False)
                continue
        return parsed_objects

    async def run_processing_pipeline(self, task_id: uuid.UUID, environment_id: str):
        """
        The main background process for a task. Runs the full pipeline.
        """
        # Initial status update is now handled by initialize_task
        logger.info(f"[Task {task_id}] Starting processing pipeline for environment: {environment_id}")
        pipeline_start_time = time.time()

        try:
            await self.update_task_status(task_id, "INITIALIZING", progress=0.01, current_step="Pipeline Startup") # Small progress
            
            await self._ensure_models_loaded()
            await self.update_task_status(task_id, "INITIALIZING", progress=0.05, current_step="Models Loaded")

            await self.update_task_status(task_id, "PREPARING_DATA", progress=0.1, current_step="Downloading/Preparing Videos & Frames")
            initial_frames_map = await self.video_data_manager.prepare_initial_frames_for_environment(
                environment_id, target_fps=settings.TARGET_FPS, jpeg_quality=settings.FRAME_JPEG_QUALITY
            )

            if not initial_frames_map:
                raise RuntimeError(f"No initial frames could be prepared for environment {environment_id}.")

            total_frames_to_process = sum(len(paths) for paths in initial_frames_map.values())
            if total_frames_to_process == 0:
                 raise RuntimeError(f"No frames extracted for any camera in environment {environment_id}.")
            logger.info(f"[Task {task_id}] Total frames to process: {total_frames_to_process}")
            processed_frames_count = 0
            start_progress = 0.2

            await self.update_task_status(task_id, "PROCESSING", progress=start_progress, current_step="Starting Frame Processing")
            active_camera_ids = sorted(initial_frames_map.keys())

            for cam_id in active_camera_ids:
                frame_paths = initial_frames_map.get(cam_id, [])
                if not frame_paths:
                    logger.warning(f"[Task {task_id}][{cam_id}] No frames found to process.")
                    continue

                logger.info(f"[Task {task_id}][{cam_id}] Processing {len(frame_paths)} frames...")
                await self.update_task_status(task_id, "PROCESSING", current_step=f"Processing Camera {cam_id}")

                await self.tracker.reset()
                logger.info(f"[Task {task_id}][{cam_id}] Tracker state reset.")

                for frame_path_str in frame_paths:
                    frame_path = Path(frame_path_str)
                    if not frame_path.is_file():
                        logger.warning(f"[Task {task_id}][{cam_id}] Frame file not found: {frame_path}. Skipping.")
                        processed_frames_count += 1
                        continue

                    frame_start_time = time.time()
                    try:
                        frame_bgr = await asyncio.to_thread(cv2.imread, frame_path_str)
                        if frame_bgr is None:
                            logger.warning(f"[Task {task_id}][{cam_id}] Failed to load frame: {frame_path}. Skipping.")
                            processed_frames_count += 1
                            continue
                    except Exception as load_err:
                        logger.error(f"[Task {task_id}][{cam_id}] Error loading frame {frame_path}: {load_err}")
                        processed_frames_count += 1
                        continue

                    detections: List = await self.detector.detect(frame_bgr)
                    detections_np = np.array([d.to_tracker_format() for d in detections]) if detections else np.empty((0, 6))
                    tracker_output_np = await self.tracker.update(detections_np, frame_bgr)
                    tracked_objects: List[TrackedObject] = self._parse_tracker_output(tracker_output_np, cam_id)

                    tracking_data_payload = [
                         schemas.TrackedPersonData(
                             track_id=obj.track_id,
                             global_person_id=str(obj.global_id) if obj.global_id is not None else None,
                             bbox_img=obj.bbox.to_list(),
                             confidence=obj.confidence,
                         ) for obj in tracked_objects
                     ]
                    
                    # Use timezone-aware UTC now for frame_timestamp and convert to ISO string
                    current_frame_timestamp = datetime.now(timezone.utc).isoformat()

                    ws_payload = {
                        "camera_id": cam_id,
                        "frame_timestamp": current_frame_timestamp,
                        "frame_path": frame_path_str,
                        "tracking_data": [p.model_dump() for p in tracking_data_payload]
                    }
                    await self.notification_service.send_tracking_update(str(task_id), ws_payload)

                    processed_frames_count += 1
                    current_progress = start_progress + (1.0 - start_progress) * (processed_frames_count / total_frames_to_process)
                    if processed_frames_count % 20 == 0:
                         frame_proc_time = (time.time() - frame_start_time) * 1000
                         await self.update_task_status(
                             task_id, "PROCESSING", progress=current_progress,
                             current_step=f"Processing Camera {cam_id} ({processed_frames_count}/{total_frames_to_process})",
                             details=f"Frame proc time: {frame_proc_time:.1f} ms. Found {len(tracked_objects)} tracks."
                         )
                    await asyncio.sleep(0.001)
                logger.info(f"[Task {task_id}][{cam_id}] Finished processing frames.")

            pipeline_duration = time.time() - pipeline_start_time
            await self.update_task_status(
                task_id, "COMPLETED", progress=1.0, current_step="Processing Complete",
                details=f"Task completed successfully in {pipeline_duration:.2f} seconds."
            )
            logger.info(f"[Task {task_id}] Pipeline completed in {pipeline_duration:.2f} seconds.")

        except Exception as e:
            pipeline_duration = time.time() - pipeline_start_time
            error_msg = f"Task failed after {pipeline_duration:.2f} seconds: {e}"
            logger.exception(f"[Task {task_id}] Pipeline execution failed.")
            await self.update_task_status(task_id, "FAILED", current_step="Pipeline Failed", details=error_msg)
        finally:
            logger.info(f"[Task {task_id}] Pipeline execution finished (Status: {PROCESSING_TASKS_DB.get(task_id, {}).get('status')}).")