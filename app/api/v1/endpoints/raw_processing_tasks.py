from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import Dict, Any
import uuid
import logging
from datetime import datetime

from app.api.v1 import schemas
from app.core.config import settings
from app.services.raw_video_service import RawVideoService
from app.core.dependencies import get_raw_video_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/environments",
    response_model=Dict[str, Any],
    summary="Get available environments for raw video streaming"
)
async def get_available_environments_raw():
    """
    Get list of available environments for raw video streaming.
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
                "name": f"{env_id.title()} Environment (Raw)",
                "description": f"Raw video streaming environment with {len(env_data['cameras'])} cameras and {env_data['total_sub_videos']} video segments",
                "camera_count": len(env_data["cameras"]),
                "cameras": env_data["cameras"],
                "available": True,
                "total_sub_videos": env_data["total_sub_videos"],
                "mode": "raw_streaming"
            }
            environments.append(environment)
        
        # If no environments in config, return default ones
        if not environments:
            environments = [
                {
                    "environment_id": "factory",
                    "name": "Factory Environment (Raw)",
                    "description": "Raw video streaming factory environment with 4 cameras",
                    "camera_count": 4,
                    "cameras": ["c09", "c12", "c13", "c16"],
                    "available": True,
                    "mode": "raw_streaming"
                },
                {
                    "environment_id": "campus",
                    "name": "Campus Environment (Raw)", 
                    "description": "Raw video streaming campus environment with 4 cameras",
                    "camera_count": 4,
                    "cameras": ["c01", "c02", "c03", "c05"],
                    "available": True,
                    "mode": "raw_streaming"
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
        logger.error(f"Error getting available environments for raw streaming: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving available environments"
        )


@router.post(
    "/start",
    response_model=schemas.ProcessingTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new raw video streaming task (no AI processing)"
)
async def start_raw_processing_task_endpoint(
    params: schemas.ProcessingTaskStartRequest,
    background_tasks: BackgroundTasks,
    raw_video_service: RawVideoService = Depends(get_raw_video_service)
):
    """
    Initiates a background task to stream raw video data for a specified environment without AI processing.

    This involves:
    1. Downloading the first sub-video for each camera in the environment.
    2. Extracting frames from these videos.
    3. Sending raw frames via WebSocket (no detection, tracking, or Re-ID).

    The process runs entirely in the background. Use the returned URLs to monitor
    status and receive real-time raw video frames.
    """
    try:
        logger.info(f"🎬 RAW API REQUEST: Start raw processing task endpoint called for environment: {params.environment_id}")
        
        # Initialize task state in the raw video service
        task_id = await raw_video_service.initialize_raw_task(params.environment_id)
        logger.info(f"🆔 RAW API REQUEST: Task ID {task_id} created for environment {params.environment_id}")

        # Add the main raw pipeline execution method to background tasks
        background_tasks.add_task(
            raw_video_service.run_raw_streaming_pipeline,
            task_id=task_id,
            environment_id=params.environment_id
        )

        logger.info(f"🚀 RAW API REQUEST: Background task added for raw processing task ID: {task_id} for environment {params.environment_id}")

        # Construct response URLs relative to the API prefix
        status_url = f"{settings.API_V1_PREFIX}/raw-processing-tasks/{task_id}/status"
        websocket_url = f"/ws/raw-tracking/{task_id}"

        return schemas.ProcessingTaskCreateResponse(
            task_id=task_id,
            message=f"Raw video streaming task for environment '{params.environment_id}' initiated.",
            status_url=status_url,
            websocket_url=websocket_url
        )
    except ValueError as ve:
        logger.warning(f"Value error starting raw task for environment {params.environment_id}: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error starting raw processing task for environment {params.environment_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error starting raw task.")


@router.get(
    "/{task_id}/status",
    response_model=schemas.TaskStatusResponse,
    summary="Get the status of a raw processing task"
)
async def get_raw_processing_task_status_endpoint(
    task_id: uuid.UUID,
    raw_video_service: RawVideoService = Depends(get_raw_video_service)
):
    """
    Retrieves the current status, progress, and details of a specific raw processing task.
    """
    status_info = await raw_video_service.get_raw_task_status(task_id)

    if status_info is None:
        logger.warning(f"Status requested for unknown or missing raw processing task_id: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw processing task not found.")

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
    summary="List all raw processing tasks"
)
async def list_raw_processing_tasks(
    raw_video_service: RawVideoService = Depends(get_raw_video_service)
):
    """
    Get list of all raw processing tasks with their current status.
    """
    try:
        # Get all task statuses from raw video service
        all_tasks = await raw_video_service.get_all_raw_task_statuses()
        
        return {
            "status": "success",
            "data": {
                "tasks": all_tasks,
                "total_count": len(all_tasks)
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
    except Exception as e:
        logger.error(f"Error listing raw processing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving raw processing tasks"
        )


@router.get(
    "/{task_id}",
    response_model=Dict[str, Any],
    summary="Get specific raw processing task details"
)
async def get_raw_processing_task_details(
    task_id: uuid.UUID,
    raw_video_service: RawVideoService = Depends(get_raw_video_service)
):
    """
    Get detailed information about a specific raw processing task.
    """
    try:
        # Get task status and details
        task_info = await raw_video_service.get_raw_task_status(task_id)
        
        if task_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Raw processing task not found"
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
                "updated_at": task_info.get("updated_at"),
                "mode": "raw_streaming"
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting raw processing task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving raw processing task details"
        )


@router.delete(
    "/{task_id}/stop",
    response_model=Dict[str, Any],
    summary="Stop and cleanup a raw processing task"
)
async def stop_raw_processing_task(
    task_id: uuid.UUID,
    raw_video_service: RawVideoService = Depends(get_raw_video_service)
):
    """
    Stop and cleanup a raw processing task.
    This is useful for handling stuck streaming sessions.
    """
    try:
        logger.info(f"🛑 RAW API REQUEST: Stop raw processing task {task_id}")
        
        # Stop the task in the raw video service
        success = await raw_video_service.stop_raw_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Raw processing task not found or already stopped"
            )
        
        logger.info(f"✅ RAW API REQUEST: Successfully stopped raw processing task {task_id}")
        
        return {
            "status": "success",
            "data": {
                "task_id": str(task_id),
                "message": "Raw processing task stopped successfully",
                "stopped_at": str(datetime.utcnow().isoformat())
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping raw processing task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error stopping raw processing task"
        )


@router.delete(
    "/environment/{environment_id}/cleanup",
    response_model=Dict[str, Any],
    summary="Cleanup all tasks for an environment"
)
async def cleanup_environment_tasks(
    environment_id: str,
    raw_video_service: RawVideoService = Depends(get_raw_video_service)
):
    """
    Stop and cleanup all raw processing tasks for a specific environment.
    This is useful for resolving "active streaming session already exists" errors.
    """
    try:
        logger.info(f"🧹 RAW API REQUEST: Cleanup environment tasks for {environment_id}")
        
        # Get all tasks for this environment
        all_tasks = await raw_video_service.get_all_raw_task_statuses()
        environment_tasks = [
            task for task in all_tasks 
            if task.get("environment_id") == environment_id
        ]
        
        stopped_tasks = []
        
        # Stop each task for this environment
        for task in environment_tasks:
            task_id = uuid.UUID(task["task_id"])
            success = await raw_video_service.stop_raw_task(task_id)
            if success:
                stopped_tasks.append(str(task_id))
                logger.info(f"✅ RAW CLEANUP: Stopped task {task_id} for environment {environment_id}")
        
        logger.info(f"🧹 RAW API REQUEST: Cleanup completed for environment {environment_id}, stopped {len(stopped_tasks)} tasks")
        
        return {
            "status": "success",
            "data": {
                "environment_id": environment_id,
                "message": f"Cleaned up {len(stopped_tasks)} tasks for environment {environment_id}",
                "stopped_tasks": stopped_tasks,
                "cleanup_at": str(datetime.utcnow().isoformat())
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up environment {environment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error cleaning up environment tasks"
        )
