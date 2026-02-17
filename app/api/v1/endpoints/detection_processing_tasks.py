"""
Detection processing tasks API endpoints.

Provides REST API endpoints for YOLO detection-enabled video processing tasks.

Endpoints:
- GET /environments - List detection-capable environments
- POST /start - Start detection processing task
- GET /{task_id}/status - Get task status
- GET /{task_id} - Get task details
- GET / - List all detection tasks
- DELETE /{task_id}/stop - Stop detection task
- DELETE /environment/{environment_id}/cleanup - Cleanup environment tasks
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import Dict, Any
import uuid
import logging
from datetime import datetime

from app.api.v1 import schemas
from app.core.config import settings
from app.services.detection_video_service import DetectionVideoService
from app.core.dependencies import get_detection_video_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/environments",
    response_model=Dict[str, Any],
    summary="Get available environments for YOLO detection processing"
)
async def get_available_detection_environments():
    """
    Get list of available environments for YOLO detection processing.
    Returns environment configurations with detection capabilities.
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
        
        # Create environment list with detection capabilities
        environments = []
        for env_id, env_data in env_groups.items():
            environment = {
                "environment_id": env_id,
                "name": f"{env_id.title()} Environment (YOLO Detection)",
                "description": f"YOLO detection environment with {len(env_data['cameras'])} cameras and {env_data['total_sub_videos']} video segments",
                "camera_count": len(env_data["cameras"]),
                "cameras": env_data["cameras"],
                "available": True,
                "total_sub_videos": env_data["total_sub_videos"],
                "mode": "detection_processing",
                "detection_features": {
                    "model": "YOLO11-L",
                    "person_detection": True,
                    "confidence_threshold": settings.YOLO_CONFIDENCE_THRESHOLD,
                    "real_time_processing": True
                }
            }
            environments.append(environment)
        
        # If no environments in config, return default ones with detection features
        if not environments:
            for env_id, template in settings.ENVIRONMENT_TEMPLATES.items():
                camera_definitions = template.get("cameras", {})
                camera_ids = list(camera_definitions.keys())
                total_sub_videos = sum(
                    cam_cfg.get("num_sub_videos", 0) for cam_cfg in camera_definitions.values()
                )
                environments.append(
                    {
                        "environment_id": env_id,
                        "name": f"{template.get('name', env_id.title() + ' Environment')} (YOLO Detection)",
                        "description": template.get(
                            "description",
                            f"YOLO detection environment with {len(camera_ids)} cameras"
                        ),
                        "camera_count": len(camera_ids),
                        "cameras": camera_ids,
                        "total_sub_videos": total_sub_videos,
                        "available": True,
                        "mode": "detection_processing",
                        "detection_features": {
                            "model": "YOLO11-L",
                            "person_detection": True,
                            "confidence_threshold": settings.YOLO_CONFIDENCE_THRESHOLD,
                            "real_time_processing": True
                        }
                    }
                )

        return {
            "status": "success",
            "data": {
                "environments": environments,
                "total_count": len(environments),
                "detection_capabilities": {
                    "model_type": "YOLO",
                    "model_variant": "yolo11l",
                    "supported_classes": ["person"],
                    "real_time_inference": True,
                    "confidence_threshold": settings.YOLO_CONFIDENCE_THRESHOLD,
                    "input_resolution": f"{settings.YOLO_INPUT_SIZE}x{settings.YOLO_INPUT_SIZE}"
                }
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Error getting available environments for detection processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving available detection environments"
        )


@router.post(
    "/start",
    response_model=schemas.ProcessingTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new YOLO detection processing task"
)
async def start_detection_processing_task_endpoint(
    params: schemas.ProcessingTaskStartRequest,
    background_tasks: BackgroundTasks,
    detection_service: DetectionVideoService = Depends(get_detection_video_service)
):
    """
    Initiates a background task for YOLO person detection processing on video data.

    Simplified Detection Process:
    1. Initialize YOLO model and detection services
    2. Download video data for the specified environment
    3. Process frames with YOLO person detection
    4. Stream detection results via WebSocket with annotated frames
    5. Future pipeline features (tracking, re-ID, homography) are sent as static null values

    The process runs entirely in the background. Use the returned URLs to monitor
    status and receive real-time detection updates.
    """
    try:
        logger.info(f"🎬 DETECTION API REQUEST: Start detection processing task endpoint called for environment: {params.environment_id}")
        
        # Concurrency check
        active_detection_tasks = [
            t for t in detection_service.active_tasks 
            # We can check mode primarily, though active_tasks is a set of UUIDs.
            # Ideally we check the task state but for raw speed we can check count of active_tasks
            # Since detection service inherits from raw, let's just count all active tasks for now 
            # or filter if possible. Because we don't have easy access to task types in active_tasks set 
            # without lookup, we'll count all.
        ]
        if len(detection_service.active_tasks) >= settings.MAX_CONCURRENT_DETECTION_TASKS:
             raise HTTPException(
                 status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                 detail=f"Maximum concurrent detection tasks ({settings.MAX_CONCURRENT_DETECTION_TASKS}) reached. Please stop an existing task first."
             )

        # Prepare task options instead of mutating global settings
        task_options = {}
        if params.enable_tracking is not None:
             task_options["enable_tracking"] = bool(params.enable_tracking)

        # Initialize task state in the detection service
        task_id = await detection_service.initialize_raw_task(params.environment_id, options=task_options)
        logger.info(f"🆔 DETECTION API REQUEST: Task ID {task_id} created for environment {params.environment_id}")

        # Add the simplified detection pipeline to background tasks
        background_tasks.add_task(
            detection_service.process_detection_task_simple,
            task_id=task_id,
            environment_id=params.environment_id
        )

        logger.info(f"🚀 DETECTION API REQUEST: Background task added for detection processing task ID: {task_id} for environment {params.environment_id}")

        # Construct response URLs relative to the API prefix
        status_url = f"{settings.API_V1_PREFIX}/detection-processing-tasks/{task_id}/status"
        websocket_url = f"/ws/tracking/{task_id}"

        response = schemas.ProcessingTaskCreateResponse(
            task_id=task_id,
            message=f"YOLO detection processing task for environment '{params.environment_id}' initiated.",
            status_url=status_url,
            websocket_url=websocket_url
        )

        return response
    except ValueError as ve:
        logger.warning(f"Value error starting detection task for environment {params.environment_id}: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error starting detection processing task for environment {params.environment_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error starting detection task.")


@router.get(
    "/{task_id}/status",
    response_model=schemas.TaskStatusResponse,
    summary="Get the status of a detection processing task"
)
async def get_detection_processing_task_status_endpoint(
    task_id: uuid.UUID,
    detection_service: DetectionVideoService = Depends(get_detection_video_service)
):
    """
    Retrieves the current status, progress, and details of a specific detection processing task.
    """
    status_info = await detection_service.get_raw_task_status(task_id)  # Inherits from parent

    if status_info is None:
        logger.warning(f"Status requested for unknown or missing detection processing task_id: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection processing task not found.")

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
    summary="List all detection processing tasks"
)
async def list_detection_processing_tasks(
    detection_service: DetectionVideoService = Depends(get_detection_video_service)
):
    """
    Get list of all detection processing tasks with their current status.
    """
    try:
        # Get all task statuses from detection service (inherits from parent)
        all_tasks = await detection_service.get_all_raw_task_statuses()
        
        # Add detection-specific metadata
        enhanced_tasks = []
        for task in all_tasks:
            enhanced_task = dict(task)
            enhanced_task["mode"] = "detection_processing"
            enhanced_task["detection_model"] = "YOLO11-L"
            enhanced_tasks.append(enhanced_task)
        
        # Get detection statistics
        detection_stats = detection_service.get_detection_stats()
        
        return {
            "status": "success",
            "data": {
                "tasks": enhanced_tasks,
                "total_count": len(enhanced_tasks),
                "detection_statistics": detection_stats
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
    except Exception as e:
        logger.error(f"Error listing detection processing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving detection processing tasks"
        )


@router.get(
    "/{task_id}",
    response_model=Dict[str, Any],
    summary="Get specific detection processing task details"
)
async def get_detection_processing_task_details(
    task_id: uuid.UUID,
    detection_service: DetectionVideoService = Depends(get_detection_video_service)
):
    """
    Get detailed information about a specific detection processing task.
    """
    try:
        # Get task status and details (inherits from parent)
        task_info = await detection_service.get_raw_task_status(task_id)
        
        if task_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Detection processing task not found"
            )
        
        # Get detection statistics for this service
        detection_stats = detection_service.get_detection_stats()
            
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
                "mode": "detection_processing",
                "detection_model": "YOLO11-L",
                "detection_statistics": detection_stats
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detection processing task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving detection processing task details"
        )


@router.delete(
    "/{task_id}/stop",
    response_model=Dict[str, Any],
    summary="Stop and cleanup a detection processing task"
)
async def stop_detection_processing_task(
    task_id: uuid.UUID,
    detection_service: DetectionVideoService = Depends(get_detection_video_service)
):
    """
    Stop and cleanup a detection processing task.
    This is useful for handling stuck detection sessions.
    """
    try:
        logger.info(f"🛑 DETECTION API REQUEST: Stop detection processing task {task_id}")
        
        # Stop the task in the detection service (inherits from parent)
        success = await detection_service.stop_raw_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Detection processing task not found or already stopped"
            )
        
        logger.info(f"✅ DETECTION API REQUEST: Successfully stopped detection processing task {task_id}")
        
        return {
            "status": "success",
            "data": {
                "task_id": str(task_id),
                "message": "Detection processing task stopped successfully",
                "stopped_at": str(datetime.utcnow().isoformat()),
                "mode": "detection_processing"
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping detection processing task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error stopping detection processing task"
        )


@router.delete(
    "/environment/{environment_id}/cleanup",
    response_model=Dict[str, Any],
    summary="Cleanup all detection tasks for an environment"
)
async def cleanup_detection_environment_tasks(
    environment_id: str,
    detection_service: DetectionVideoService = Depends(get_detection_video_service)
):
    """
    Stop and cleanup all detection processing tasks for a specific environment.
    This is useful for resolving "active detection session already exists" errors.
    """
    try:
        logger.info(f"🧹 DETECTION API REQUEST: Cleanup environment tasks for {environment_id}")
        
        # Get all tasks for this environment (inherits from parent)
        all_tasks = await detection_service.get_all_raw_task_statuses()
        environment_tasks = [
            task for task in all_tasks 
            if task.get("environment_id") == environment_id
        ]
        
        stopped_tasks = []
        
        # Stop each task for this environment
        for task in environment_tasks:
            task_uuid = uuid.UUID(task["task_id"])
            success = await detection_service.stop_raw_task(task_uuid)
            if success:
                stopped_tasks.append(str(task_uuid))
                logger.info(f"✅ DETECTION CLEANUP: Stopped task {task_uuid} for environment {environment_id}")
        
        logger.info(f"🧹 DETECTION API REQUEST: Cleanup completed for environment {environment_id}, stopped {len(stopped_tasks)} tasks")
        
        return {
            "status": "success",
            "data": {
                "environment_id": environment_id,
                "message": f"Cleaned up {len(stopped_tasks)} detection tasks for environment {environment_id}",
                "stopped_tasks": stopped_tasks,
                "cleanup_at": str(datetime.utcnow().isoformat()),
                "mode": "detection_processing"
            },
            "timestamp": str(datetime.utcnow().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up detection environment {environment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error cleaning up detection environment tasks"
        )
