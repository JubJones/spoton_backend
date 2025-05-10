from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import Dict, Any
import uuid
import logging

from app.api.v1 import schemas
from app.services.pipeline_orchestrator import PipelineOrchestratorService, PROCESSING_TASKS_DB
from app.dependencies import get_pipeline_orchestrator
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/start",
    response_model=schemas.ProcessingTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new person tracking and Re-ID processing task"
)
async def start_processing_task_endpoint(
    params: schemas.ProcessingTaskStartRequest,
    background_tasks: BackgroundTasks,
    orchestrator: PipelineOrchestratorService = Depends(get_pipeline_orchestrator)
):
    """
    Initiates a background task to process video data for a specified environment.

    This involves:
    1.  Downloading the first sub-video for each camera in the environment.
    2.  Extracting frames from these videos.
    3.  Running the detection, tracking (BotSort), and Re-ID (CLIP) pipeline.
    4.  Sending tracking results via WebSocket.

    The process runs entirely in the background. Use the returned URLs to monitor
    status and receive real-time updates.
    """
    try:
        # Initialize task state in the orchestrator
        task_id = await orchestrator.initialize_task(params.environment_id)

        # Add the main pipeline execution method to background tasks
        background_tasks.add_task(
            orchestrator.run_processing_pipeline, # Public method to run the full pipeline
            task_id=task_id,
            environment_id=params.environment_id
        )

        logger.info(f"Background task added for processing task ID: {task_id} for environment {params.environment_id}")

        # Construct response URLs relative to the API prefix
        # Ensure leading slashes are handled correctly depending on how frontend constructs URLs
        status_url = f"{settings.API_V1_PREFIX}/processing-tasks/{task_id}/status"
        # WebSocket path is usually relative to the base URL, not the API prefix
        websocket_url = f"/ws/tracking/{task_id}"

        return schemas.ProcessingTaskCreateResponse(
            task_id=task_id,
            message=f"Processing task for environment '{params.environment_id}' initiated.",
            status_url=status_url,
            websocket_url=websocket_url
        )
    except ValueError as ve:
        logger.warning(f"Value error starting task for environment {params.environment_id}: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error starting processing task for environment {params.environment_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error starting task.")

@router.get(
    "/{task_id}/status",
    response_model=schemas.TaskStatusResponse,
    summary="Get the status of a processing task"
)
async def get_processing_task_status_endpoint(
    task_id: uuid.UUID,
    orchestrator: PipelineOrchestratorService = Depends(get_pipeline_orchestrator) # Use orchestrator method
):
    """
    Retrieves the current status, progress, and details of a specific processing task.
    """
    status_info = await orchestrator.get_task_status(task_id)

    if status_info is None:
        logger.warning(f"Status requested for unknown or missing processing task_id: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing task not found.")

    # Ensure all required fields for TaskStatusResponse are present
    return schemas.TaskStatusResponse(
        task_id=task_id,
        status=status_info.get("status", "UNKNOWN"),
        progress=status_info.get("progress", 0.0),
        current_step=status_info.get("current_step"),
        details=status_info.get("details")
    )