from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import Dict, Any
import uuid
import logging
from datetime import datetime

from app.api.v1 import schemas
from app.orchestration.pipeline_orchestrator import orchestrator, PipelineOrchestrator
from app.core.dependencies import get_pipeline_orchestrator
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/environments",
    response_model=Dict[str, Any],
    summary="Get available environments for task processing"
)
async def get_available_environments():
    """
    Get list of available environments that can be processed.
    Returns environment configurations from the system settings.
    """
    try:
        # Get available environments from config (VideoSetEnvironmentConfig objects)
        video_sets = getattr(settings, 'VIDEO_SETS', [])
        
        # Group cameras by environment
        env_groups = {}
        for video_config in video_sets:
            env_id = video_config.env_id
            if env_id not in env_groups:
                env_groups[env_id] = {
                    "cameras": [],
                    "total_sub_videos": 0
                }
            env_groups[env_id]["cameras"].append(video_config.cam_id)
            env_groups[env_id]["total_sub_videos"] += video_config.num_sub_videos
        
        # Create environment list
        environments = []
        for env_id, env_data in env_groups.items():
            environment = {
                "environment_id": env_id,
                "name": f"{env_id.title()} Environment",
                "description": f"Environment with {len(env_data['cameras'])} cameras and {env_data['total_sub_videos']} video segments",
                "camera_count": len(env_data["cameras"]),
                "cameras": env_data["cameras"],
                "available": True,
                "total_sub_videos": env_data["total_sub_videos"]
            }
            environments.append(environment)
        
        # If no environments in config, return default ones
        if not environments:
            environments = [
                {
                    "environment_id": "factory",
                    "name": "Factory Environment",
                    "description": "Factory monitoring environment with 4 cameras",
                    "camera_count": 4,
                    "cameras": ["c09", "c12", "c13", "c16"],
                    "available": True
                },
                {
                    "environment_id": "campus",
                    "name": "Campus Environment", 
                    "description": "Campus monitoring environment with 4 cameras",
                    "camera_count": 4,
                    "cameras": ["c01", "c02", "c03", "c05"],
                    "available": True
                }
            ]
        
        return {
            "status": "success",
            "data": {
                "environments": environments,
                "total_count": len(environments)
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Error getting available environments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving available environments"
        )


@router.post(
    "/start",
    response_model=schemas.ProcessingTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new person tracking and Re-ID processing task"
)
async def start_processing_task_endpoint(
    params: schemas.ProcessingTaskStartRequest,
    background_tasks: BackgroundTasks,
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
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
        logger.info(f"🎬 API REQUEST: Start processing task endpoint called for environment: {params.environment_id}")
        
        # Initialize task state in the orchestrator
        task_id = await orchestrator.initialize_task(params.environment_id)
        logger.info(f"🆔 API REQUEST: Task ID {task_id} created for environment {params.environment_id}")

        # Add the main pipeline execution method to background tasks
        background_tasks.add_task(
            orchestrator.run_processing_pipeline, # Public method to run the full pipeline
            task_id=task_id,
            environment_id=params.environment_id
        )

        logger.info(f"🚀 API REQUEST: Background task added for processing task ID: {task_id} for environment {params.environment_id}")

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
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator) # Use orchestrator method
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

@router.get(
    "",
    response_model=Dict[str, Any],
    summary="List all processing tasks"
)
async def list_processing_tasks(
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """
    Get list of all processing tasks with their current status.
    """
    try:
        # Get all task statuses from orchestrator
        all_tasks = await orchestrator.get_all_task_statuses()
        
        return {
            "status": "success",
            "data": {
                "tasks": all_tasks,
                "total_count": len(all_tasks)
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
    except Exception as e:
        logger.error(f"Error listing processing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving processing tasks"
        )

@router.get(
    "/{task_id}",
    response_model=Dict[str, Any],
    summary="Get specific processing task details"
)
async def get_processing_task_details(
    task_id: uuid.UUID,
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """
    Get detailed information about a specific processing task.
    """
    try:
        # Get task status and details
        task_info = await orchestrator.get_task_status(task_id)
        
        if task_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Processing task not found"
            )
            
        return {
            "status": "success",
            "data": {
                "task_id": str(task_id),
                "status": task_info.get("status", "UNKNOWN"),
                "progress": task_info.get("progress", 0.0),
                "current_step": task_info.get("current_step"),
                "details": task_info.get("details"),
                "environment_id": task_info.get("environment_id"),
                "created_at": task_info.get("created_at"),
                "updated_at": task_info.get("updated_at")
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving processing task details"
        )
