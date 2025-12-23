from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging

from app.utils.mjpeg_streamer import mjpeg_streamer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{task_id}/{camera_id}")
async def stream_video(task_id: str, camera_id: str):
    """
    Stream MJPEG video for a specific task and camera.
    
    CORS headers are explicitly set to allow canvas capture in browsers.
    """
    try:
        # Check if task/camera exists (optional validation logic could go here)
        
        return StreamingResponse(
            mjpeg_streamer.stream_generator(task_id, camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
    except Exception as e:
        logger.error(f"Error starting stream for {task_id}/{camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Stream error")
