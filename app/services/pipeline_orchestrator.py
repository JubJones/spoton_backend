import asyncio
import uuid
from typing import Dict, Any, Optional, List
import logging

from app.api.v1.schemas import ProcessingTaskStartRequest, ProcessingTaskCreateResponse, TaskStatusResponse
from app.services.video_data_manager_service import VideoDataManagerService
from app.api.websockets import manager as websocket_manager # Assuming global manager instance for WebSockets

logger = logging.getLogger(__name__)

# In-memory store for task statuses. Replace with Redis/DB for production.
# task_id -> {"status": "...", "progress": 0.0, "details": "..."}
PROCESSING_TASKS_DB: Dict[uuid.UUID, Dict[str, Any]] = {}

class PipelineOrchestratorService:
    """
    Orchestrates the entire processing pipeline for a given task.
    This includes data preparation, detection, tracking, Re-ID, etc.
    """
    def __init__(self, video_data_manager: VideoDataManagerService):
        self.video_data_manager = video_data_manager
        # Inject other services like Detector, Tracker, ReIDService, HomographyService later

    async def _run_full_pipeline_background(
        self,
        task_id: uuid.UUID,
        params: ProcessingTaskStartRequest
    ):
        """
        The main background process for a task.
        """
        PROCESSING_TASKS_DB[task_id]["status"] = "INITIALIZING"
        PROCESSING_TASKS_DB[task_id]["details"] = f"Task {task_id} initializing for environment {params.environment_id}."
        logger.info(PROCESSING_TASKS_DB[task_id]["details"])
        await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})


        # 1. Prepare Initial Data (Download first videos, extract frames)
        try:
            PROCESSING_TASKS_DB[task_id]["status"] = "PREPARING_DATA"
            PROCESSING_TASKS_DB[task_id]["details"] = f"Preparing initial video data for environment: {params.environment_id}."
            await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})
            
            # This returns a map: {cam_id: [frame_paths_for_cam_1], ...}
            initial_frames_map = await self.video_data_manager.prepare_initial_frames_for_environment(
                params.environment_id
            )
            
            if not initial_frames_map:
                msg = f"No initial frames could be prepared for environment {params.environment_id}. Task cannot proceed."
                logger.error(msg)
                PROCESSING_TASKS_DB[task_id]["status"] = "FAILED"
                PROCESSING_TASKS_DB[task_id]["details"] = msg
                await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})
                return

            PROCESSING_TASKS_DB[task_id]["details"] = f"Initial video data prepared. Found frames for {len(initial_frames_map)} cameras."
            PROCESSING_TASKS_DB[task_id]["progress"] = 0.1 # Arbitrary progress after data prep
            logger.info(PROCESSING_TASKS_DB[task_id]["details"])
            await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})

        except Exception as e:
            error_msg = f"Error during data preparation for task {task_id}: {e}"
            logger.exception(error_msg)
            PROCESSING_TASKS_DB[task_id]["status"] = "FAILED"
            PROCESSING_TASKS_DB[task_id]["details"] = error_msg
            await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})
            return

        # --- Placeholder for actual processing loop ---
        # 2. Loop through frames/time, perform detection, tracking, Re-ID, etc.
        # For now, we'll simulate some processing.
        PROCESSING_TASKS_DB[task_id]["status"] = "PROCESSING"
        PROCESSING_TASKS_DB[task_id]["details"] = "Starting frame processing loop (simulated)."
        await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})
        
        # Example: Iterate through prepared frames (simplified)
        total_frames_to_simulate = sum(len(frames) for frames in initial_frames_map.values())
        processed_frames_count = 0

        for cam_id, frame_paths in initial_frames_map.items():
            if not frame_paths:
                continue
            logger.info(f"[Task {task_id}] Simulating processing for camera {cam_id} with {len(frame_paths)} frames.")
            for frame_idx, frame_path in enumerate(frame_paths):
                # In a real scenario:
                # - Load frame: image = cv2.imread(frame_path)
                # - Detect: detections = await detector.detect(image)
                # - Track: tracks = await tracker.update(detections, image)
                # - Re-ID, Homography, etc.
                # - Send WebSocketTrackingMessage
                
                await asyncio.sleep(0.05) # Simulate work for each frame
                
                processed_frames_count += 1
                current_progress = 0.1 + (0.8 * (processed_frames_count / total_frames_to_simulate)) if total_frames_to_simulate > 0 else 0.9
                PROCESSING_TASKS_DB[task_id]["progress"] = min(current_progress, 0.9) # Cap at 0.9 before final completion
                
                # Send a mock tracking update via WebSocket
                mock_tracking_data = {
                    "type": "tracking_update", # Differentiate from status updates
                    "payload": {
                        "camera_id": cam_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "image_url": f"file://{frame_path}", # Example local file URL
                        "tracking_data": [
                            {"global_person_id": f"person_{cam_id}_{frame_idx}", "bbox_img": [10,10,50,100], "map_coordinates": {"x": frame_idx * 0.1, "y": 1.5}}
                        ]
                    }
                }
                await websocket_manager.broadcast_to_task(str(task_id), mock_tracking_data)

                if processed_frames_count % 20 == 0: # Update status less frequently
                    PROCESSING_TASKS_DB[task_id]["details"] = f"Processed {processed_frames_count}/{total_frames_to_simulate} frames."
                    await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})


        PROCESSING_TASKS_DB[task_id]["status"] = "COMPLETED"
        PROCESSING_TASKS_DB[task_id]["progress"] = 1.0
        PROCESSING_TASKS_DB[task_id]["details"] = f"Task {task_id} completed successfully. Processed all initial frames."
        logger.info(PROCESSING_TASKS_DB[task_id]["details"])
        await websocket_manager.broadcast_to_task(str(task_id), {"type": "status_update", **PROCESSING_TASKS_DB[task_id]})


    async def start_processing_task(
        self, params: ProcessingTaskStartRequest
    ) -> Tuple[uuid.UUID, Dict[str, Any]]:
        """
        Initializes a processing task and returns its ID and initial status.
        The actual pipeline runs in the background.
        """
        task_id = uuid.uuid4()
        PROCESSING_TASKS_DB[task_id] = {
            "status": "QUEUED",
            "progress": 0.0,
            "details": f"Task {task_id} for environment '{params.environment_id}' has been queued."
        }
        # The background task itself will be started by the endpoint
        logger.info(f"Processing task {task_id} queued for environment: {params.environment_id}")
        return task_id, PROCESSING_TASKS_DB[task_id]

    async def get_task_status(self, task_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieves the status of a processing task.
        """
        return PROCESSING_TASKS_DB.get(task_id)