"""
Enhanced Media API Endpoints

Provides image processing endpoints for frontend integration including
camera frames with overlays, cropped person images, and multi-camera views.
"""
import logging
from pathlib import Path
import uuid
from typing import Optional, List
import base64

from fastapi import APIRouter, HTTPException, Request, status, Depends, Query
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel

from app.core.config import settings
from app.domains.visualization.services.frame_composition_service import FrameCompositionService
from app.domains.visualization.services.image_caching_service import ImageCachingService
from app.domains.visualization.entities.overlay_config import OverlayConfig

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


# ======================== NEW IMAGE PROCESSING ENDPOINTS ========================

# Dependency injection
def get_frame_composition_service() -> FrameCompositionService:
    return FrameCompositionService()

def get_image_caching_service() -> ImageCachingService:
    return ImageCachingService()


class OverlayConfigRequest(BaseModel):
    """Request model for overlay configuration."""
    bbox_color: Optional[List[int]] = None
    bbox_thickness: Optional[int] = None
    focus_color: Optional[List[int]] = None
    show_person_id: Optional[bool] = None
    show_confidence: Optional[bool] = None
    overlay_quality: Optional[int] = None


@router.get(
    "/frames/{task_id}/{camera_id}",
    summary="Get current camera frame with overlays",
    response_class=Response
)
async def get_camera_frame_with_overlays(
    task_id: uuid.UUID,
    camera_id: str,
    focused_person_id: Optional[str] = Query(None, description="Person ID to focus on"),
    quality: Optional[int] = Query(85, ge=10, le=100, description="JPEG quality"),
    thumbnail: Optional[bool] = Query(False, description="Return thumbnail version"),
    frame_composition_service: FrameCompositionService = Depends(get_frame_composition_service),
    image_caching_service: ImageCachingService = Depends(get_image_caching_service)
):
    """
    Get the current camera frame with overlays including bounding boxes,
    person IDs, and visual enhancements.
    
    This endpoint provides processed frames for real-time display with
    configurable overlay options and caching support.
    """
    try:
        # Check cache first
        if thumbnail:
            cached_data = image_caching_service.get_cached_thumbnail(
                camera_id, 0, (320, 240)  # Default thumbnail size
            )
            if cached_data:
                return Response(content=cached_data, media_type="image/jpeg")
        
        # Return a simple SVG placeholder for now (avoids PIL/Pillow issues)
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#1a1a1a"/>
  <rect x="2" y="2" width="636" height="476" fill="none" stroke="#333" stroke-width="2"/>
  <text x="50%" y="45%" text-anchor="middle" fill="#888" font-family="Arial, sans-serif" font-size="20">
    Camera {camera_id}
  </text>
  <text x="50%" y="52%" text-anchor="middle" fill="#666" font-family="Arial, sans-serif" font-size="14">
    Task: {str(task_id)[:8]}...
  </text>
  <text x="50%" y="58%" text-anchor="middle" fill="#666" font-family="Arial, sans-serif" font-size="14">
    Waiting for real-time data...
  </text>'''
        
        if focused_person_id:
            svg_content += f'''
  <text x="50%" y="64%" text-anchor="middle" fill="#666" font-family="Arial, sans-serif" font-size="14">
    Focus: {focused_person_id}
  </text>'''
        else:
            svg_content += f'''
  <text x="50%" y="64%" text-anchor="middle" fill="#666" font-family="Arial, sans-serif" font-size="14">
    No person focused
  </text>'''
            
        svg_content += '''
</svg>'''
        
        return Response(
            content=svg_content.encode('utf-8'),
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache", 
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving camera frame {camera_id} for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing camera frame"
        )


@router.get(
    "/persons/{global_person_id}/image",
    summary="Get cropped person image",
    response_class=Response
)
async def get_cropped_person_image(
    global_person_id: str,
    camera_id: Optional[str] = Query(None, description="Specific camera ID"),
    frame_index: Optional[int] = Query(None, description="Specific frame index"),
    quality: Optional[int] = Query(85, ge=10, le=100, description="JPEG quality"),
    size: Optional[str] = Query(None, regex="^\\d+x\\d+$", description="Image size (e.g., '150x200')"),
    image_caching_service: ImageCachingService = Depends(get_image_caching_service)
):
    """
    Get a cropped image of a specific person.
    
    This endpoint serves cropped person images that can be displayed
    as thumbnails or for person selection interfaces.
    """
    try:
        # Try to get from cache
        if camera_id and frame_index is not None:
            cached_image = image_caching_service.get_cached_cropped_image(
                global_person_id, camera_id, frame_index
            )
            if cached_image:
                return Response(
                    content=cached_image.image_data,
                    media_type=f"image/{cached_image.image_format}"
                )
        
        # TODO: Generate cropped image if not cached
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cropped image not found for person {global_person_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving cropped image for person {global_person_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing cropped image"
        )


@router.get(
    "/frames/{task_id}/{camera_id}/raw",
    summary="Get raw camera frame without overlays",
    response_class=Response
)
async def get_raw_camera_frame(
    task_id: uuid.UUID,
    camera_id: str,
    frame_index: Optional[int] = Query(None, description="Specific frame index"),
    quality: Optional[int] = Query(85, ge=10, le=100, description="JPEG quality")
):
    """
    Get the raw camera frame without any overlays or processing.
    
    Useful for debugging or when overlays are applied client-side.
    """
    try:
        # Return a simple placeholder without PIL dependency
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="320" height="240" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#1a1a1a"/>
  <rect x="2" y="2" width="316" height="236" fill="none" stroke="#333" stroke-width="2"/>
  <text x="50%" y="45%" text-anchor="middle" fill="#888" font-family="Arial, sans-serif" font-size="16">
    Camera {camera_id}
  </text>
  <text x="50%" y="55%" text-anchor="middle" fill="#666" font-family="Arial, sans-serif" font-size="12">
    Task: {str(task_id)[:8]}...
  </text>
  <text x="50%" y="65%" text-anchor="middle" fill="#666" font-family="Arial, sans-serif" font-size="12">
    Raw frame placeholder
  </text>
</svg>'''
        
        return Response(
            content=svg_content.encode('utf-8'),
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving raw frame {camera_id} for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing raw frame"
        )


@router.post(
    "/frames/overlay-config",
    summary="Configure overlay settings"
)
async def configure_overlay_settings(
    config: OverlayConfigRequest,
    task_id: Optional[uuid.UUID] = Query(None, description="Apply to specific task")
):
    """
    Configure visual overlay settings for frames.
    
    This endpoint allows customization of bounding box colors,
    text display options, and visual effects.
    """
    try:
        # Create overlay config from request
        overlay_config = OverlayConfig()
        
        # Update with provided settings
        if config.bbox_color:
            overlay_config.bbox_color = tuple(config.bbox_color)
        if config.bbox_thickness is not None:
            overlay_config.bbox_thickness = config.bbox_thickness
        if config.focus_color:
            overlay_config.focus_color = tuple(config.focus_color)
        if config.show_person_id is not None:
            overlay_config.show_person_id = config.show_person_id
        if config.show_confidence is not None:
            overlay_config.show_confidence = config.show_confidence
        if config.overlay_quality is not None:
            overlay_config.overlay_quality = config.overlay_quality
        
        # TODO: Save configuration (in database or cache)
        
        return {
            "message": "Overlay configuration updated successfully",
            "config": overlay_config.to_dict(),
            "task_id": str(task_id) if task_id else None
        }
        
    except Exception as e:
        logger.error(f"Error updating overlay configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating overlay configuration"
        )


@router.get(
    "/frames/{task_id}/multi-camera",
    summary="Get synchronized multi-camera view",
    response_class=Response
)
async def get_multi_camera_view(
    task_id: uuid.UUID,
    layout: Optional[str] = Query("grid", regex="^(grid|horizontal|vertical)$", description="Layout style"),
    size: Optional[str] = Query("1280x720", regex="^\\d+x\\d+$", description="Output size"),
    cameras: Optional[str] = Query(None, description="Comma-separated camera IDs"),
    quality: Optional[int] = Query(85, ge=10, le=100, description="JPEG quality"),
    frame_composition_service: FrameCompositionService = Depends(get_frame_composition_service)
):
    """
    Get a composed view of multiple cameras in a single image.
    
    This endpoint creates a grid or linear layout of multiple camera feeds
    synchronized to the same timestamp.
    """
    try:
        # Parse size
        width, height = map(int, size.split('x'))
        target_size = (width, height)
        
        # Parse camera list
        camera_list = []
        if cameras:
            camera_list = [cam.strip() for cam in cameras.split(',') if cam.strip()]
        
        # TODO: Implement multi-camera composition
        # This would need to:
        # 1. Get current frames from all specified cameras
        # 2. Compose them using the frame_composition_service
        # 3. Return the composed image
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Multi-camera view not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating multi-camera view for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating multi-camera view"
        )


@router.get(
    "/cache/stats",
    summary="Get image cache statistics"
)
async def get_cache_statistics(
    image_caching_service: ImageCachingService = Depends(get_image_caching_service)
):
    """
    Get statistics about image cache usage and performance.
    
    Useful for monitoring and debugging cache efficiency.
    """
    try:
        stats = image_caching_service.get_cache_statistics()
        return {
            "cache_stats": stats,
            "timestamp": "2025-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving cache statistics"
        )