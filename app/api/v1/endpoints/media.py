"""
API Endpoints for serving media files (sub-videos).

NOTE: With the shift to sending frame images directly via WebSockets,
this endpoint's role for primary real-time frontend display is deprecated.
It may be retained for debugging, direct sub-video downloads, or alternative use cases.
"""
import logging
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get(
    "/tasks/{task_id}/environments/{environment_id}/cameras/{camera_id}/sub_videos/{sub_video_filename}",
    summary="Serve a specific sub-video file (DEPRECATED for primary frontend display)",
    response_class=StreamingResponse 
)
async def serve_sub_video(
    task_id: uuid.UUID,
    environment_id: str,
    camera_id: str,
    sub_video_filename: str,
    request: Request 
):
    """
    Streams a specific sub-video file that has been downloaded by the backend
    as part of a processing task.

    The file path is constructed based on the application's local video download directory
    and the provided path parameters.

    **Note:** This endpoint is considered deprecated for the primary real-time frontend display,
    as frame images are now sent directly via WebSockets in `tracking_update` messages.
    It can be used for debugging or allowing users to download full sub-video segments.
    """
    try:
        base_download_dir = Path(settings.LOCAL_VIDEO_DOWNLOAD_DIR)
        video_file_path = base_download_dir / str(task_id) / environment_id / camera_id / sub_video_filename

        logger.debug(f"Attempting to serve video file: {video_file_path}")

        if not video_file_path.is_file():
            logger.warning(
                f"Video file not found for task {task_id}, env {environment_id}, "
                f"cam {camera_id}, filename {sub_video_filename}. Expected at: {video_file_path}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found."
            )

        media_type = "video/mp4" 
        return FileResponse(
            path=video_file_path,
            media_type=media_type,
            filename=sub_video_filename
        )

    except HTTPException:
        raise 
    except Exception as e:
        logger.exception(
            f"Error serving video file {sub_video_filename} for task {task_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while serving video file."
        )