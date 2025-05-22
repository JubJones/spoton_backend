"""
API Endpoints for serving media files (sub-videos).
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
    summary="Serve a specific sub-video file",
    response_class=StreamingResponse # Use StreamingResponse for better large file handling
)
async def serve_sub_video(
    task_id: uuid.UUID,
    environment_id: str,
    camera_id: str,
    sub_video_filename: str,
    request: Request # To get base URL if needed, or for range requests later
):
    """
    Streams a specific sub-video file that has been downloaded by the backend
    as part of a processing task.

    The file path is constructed based on the application's local video download directory
    and the provided path parameters.
    """
    try:
        # Construct the path to the video file in the local cache
        # This path structure must match exactly how VideoDataManagerService saves files.
        # LOCAL_VIDEO_DOWNLOAD_DIR / task_id / environment_id / camera_id / sub_video_filename
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

        # Determine appropriate media type (basic for now)
        media_type = "video/mp4" # Assume MP4, can be more dynamic if needed

        # FastAPI's FileResponse handles range requests automatically if the underlying
        # starlette.responses.FileResponse is used, which it is.
        # For more control or if using a raw StreamingResponse with a file iterator,
        # range request handling would need to be implemented manually.
        return FileResponse(
            path=video_file_path,
            media_type=media_type,
            filename=sub_video_filename # Suggests a download filename to the browser
        )

    except HTTPException:
        raise # Re-raise HTTPException to ensure FastAPI handles it
    except Exception as e:
        logger.exception(
            f"Error serving video file {sub_video_filename} for task {task_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while serving video file."
        )