from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import Dict, Any # List not used in this skeleton
import uuid

from app.api.v1 import schemas
from app.services.pipeline_orchestrator import PipelineOrchestratorService, PROCESSING_TASKS_DB
from app.dependencies import get_pipeline_orchestrator
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/start", 
    response_model=schemas.ProcessingTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new person tracking and analytics processing task"
)
async def start_processing_task_endpoint(
    params: schemas.ProcessingTaskStartRequest,
    background_tasks: BackgroundTasks,
    orchestrator: PipelineOrchestratorService = Depends(get_pipeline_orchestrator)
):
    """
    Initiates a retrospective analysis task for a specified environment.
    This involves:
    1. Downloading the first sub-video for each camera in the environment.
    2. Extracting frames from these videos.
    3. Running the detection, tracking, and Re-ID pipeline on these frames (simulated for now).
    
    The process runs in the background. Status can be polled and updates are sent via WebSocket.
    """
    try:
        task_id, initial_status = await orchestrator.start_processing_task(params)
        
        # Add the main pipeline execution to background tasks
        background_tasks.add_task(
            orchestrator._run_full_pipeline_background, # Note: _run_full_pipeline_background is an internal method
            task_id=task_id,
            params=params
        )
        
        logger.info(f"Background task added for processing task ID: {task_id}")
        
        return schemas.ProcessingTaskCreateResponse(
            task_id=task_id,
            message=initial_status.get("details", "Processing task initiated."),
            status_url=f"{settings.API_V1_PREFIX}/processing-tasks/{task_id}/status",
            websocket_url=f"/ws/tracking/{task_id}" # Relative path for WebSocket
        )
    except Exception as e:
        logger.exception(f"Error starting processing task for environment {params.environment_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get(
    "/{task_id}/status", 
    response_model=schemas.TaskStatusResponse,
    summary="Get the status of a processing task"
)
async def get_processing_task_status_endpoint(
    task_id: uuid.UUID,
    orchestrator: PipelineOrchestratorService = Depends(get_pipeline_orchestrator) # Or directly use PROCESSING_TASKS_DB
):
    """
    Retrieves the current status of a specific processing task.
    """
    # status_info = await orchestrator.get_task_status(task_id)
    status_info = PROCESSING_TASKS_DB.get(task_id) # Direct access for simplicity
    
    if not status_info:
        logger.warning(f"Status requested for unknown processing task_id: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing task not found.")
    
    return schemas.TaskStatusResponse(task_id=task_id, **status_info)

# Placeholder for other control endpoints if needed in the future, e.g., stop task
# @router.post("/{task_id}/control", summary="Send control commands to a task")
# async def control_processing_task(task_id: str, command: Any):
#     raise HTTPException(status_code=501, detail="Control endpoint not implemented yet.")