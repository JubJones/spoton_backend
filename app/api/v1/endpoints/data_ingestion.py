"""
Data ingestion API endpoints for managing DagHub S3 dataset downloads.

Provides REST API endpoints for:
- Dataset discovery and structure
- Batch downloading with progress tracking
- Verification and integrity checking
- Download session management
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.dagshub_s3_service import get_dagshub_s3_service, DagHubS3Service, DownloadProgress

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models

class DatasetStructureResponse(BaseModel):
    """Response model for dataset structure."""
    structure: Dict[str, Dict[str, List[str]]] = Field(..., description="Dataset structure mapping")
    total_video_sets: int = Field(..., description="Total number of video sets")
    total_cameras: int = Field(..., description="Total number of cameras")
    total_videos: int = Field(..., description="Total number of video files")
    discovery_timestamp: datetime = Field(..., description="When structure was discovered")


class DownloadRequest(BaseModel):
    """Request model for batch downloads."""
    video_sets: Optional[List[str]] = Field(None, description="Specific video sets to download (default: all)")
    cameras: Optional[List[str]] = Field(None, description="Specific cameras to download (default: all)")
    max_concurrent: Optional[int] = Field(None, ge=1, le=50, description="Maximum concurrent downloads")
    force_redownload: bool = Field(False, description="Force re-download of existing files")


class DownloadResponse(BaseModel):
    """Response model for download operations."""
    progress_id: str = Field(..., description="Unique progress tracking ID")
    message: str = Field(..., description="Response message")
    download_started: bool = Field(..., description="Whether download was started successfully")


class ProgressResponse(BaseModel):
    """Response model for download progress."""
    progress_id: str = Field(..., description="Progress tracking ID")
    progress: Dict = Field(..., description="Progress information")


class VerificationRequest(BaseModel):
    """Request model for dataset verification."""
    video_sets: Optional[List[str]] = Field(None, description="Specific video sets to verify (default: all)")
    cameras: Optional[List[str]] = Field(None, description="Specific cameras to verify (default: all)")


class VerificationResponse(BaseModel):
    """Response model for dataset verification."""
    verification_results: Dict = Field(..., description="Verification results")
    verification_timestamp: datetime = Field(..., description="When verification was performed")


# API Endpoints

@router.get("/dataset/structure", 
           response_model=DatasetStructureResponse,
           summary="Get Dataset Structure",
           description="Discover and return the complete structure of the DagHub dataset")
async def get_dataset_structure(
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> DatasetStructureResponse:
    """Get the complete dataset structure from DagHub S3."""
    try:
        logger.info("API request: Get dataset structure")
        
        structure = dagshub_service.get_dataset_structure()
        
        # Calculate statistics
        total_video_sets = len(structure)
        total_cameras = sum(len(cameras) for cameras in structure.values())
        total_videos = sum(
            len(videos) 
            for cameras in structure.values() 
            for videos in cameras.values()
        )
        
        return DatasetStructureResponse(
            structure=structure,
            total_video_sets=total_video_sets,
            total_cameras=total_cameras,
            total_videos=total_videos,
            discovery_timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting dataset structure: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dataset structure: {str(e)}"
        )


@router.post("/dataset/download",
            response_model=DownloadResponse,
            summary="Start Batch Download",
            description="Start a batch download of video datasets with progress tracking")
async def start_batch_download(
    request: DownloadRequest,
    background_tasks: BackgroundTasks,
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> DownloadResponse:
    """Start a batch download of videos from DagHub S3."""
    try:
        progress_id = str(uuid.uuid4())
        
        logger.info(f"API request: Start batch download [ID: {progress_id}]")
        logger.info(f"Request parameters: {request.dict()}")
        
        # Validate request parameters
        if request.video_sets:
            structure = dagshub_service.get_dataset_structure()
            invalid_sets = [vs for vs in request.video_sets if vs not in structure]
            if invalid_sets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid video sets: {invalid_sets}. Available: {list(structure.keys())}"
                )
        
        # Start download in background
        background_tasks.add_task(
            _execute_batch_download,
            dagshub_service,
            progress_id,
            request.video_sets,
            request.cameras,
            request.max_concurrent
        )
        
        return DownloadResponse(
            progress_id=progress_id,
            message=f"Batch download started with ID: {progress_id}",
            download_started=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch download: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch download: {str(e)}"
        )


async def _execute_batch_download(
    dagshub_service: DagHubS3Service,
    progress_id: str,
    video_sets: Optional[List[str]],
    cameras: Optional[List[str]],
    max_concurrent: Optional[int]
):
    """Execute batch download in background."""
    try:
        logger.info(f"Executing batch download [ID: {progress_id}]")
        
        progress = await dagshub_service.batch_download_videos(
            video_sets=video_sets,
            cameras=cameras,
            max_concurrent=max_concurrent,
            progress_id=progress_id
        )
        
        logger.info(f"Batch download completed [ID: {progress_id}]: "
                   f"{progress.completed_files} completed, {progress.failed_files} failed")
        
    except Exception as e:
        logger.error(f"Error in background batch download [ID: {progress_id}]: {e}")


@router.get("/dataset/download/{progress_id}/progress",
           response_model=ProgressResponse,
           summary="Get Download Progress",
           description="Get the current progress of a batch download")
async def get_download_progress(
    progress_id: str,
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> ProgressResponse:
    """Get download progress by ID."""
    try:
        progress = dagshub_service.get_download_progress(progress_id)
        
        if not progress:
            raise HTTPException(
                status_code=404,
                detail=f"Download progress not found for ID: {progress_id}"
            )
        
        return ProgressResponse(
            progress_id=progress_id,
            progress=progress.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting download progress: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get download progress: {str(e)}"
        )


@router.get("/dataset/downloads",
           summary="List Download Sessions",
           description="List all download sessions and their progress")
async def list_download_sessions(
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> Dict[str, Any]:
    """List all download sessions and their progress."""
    try:
        sessions = dagshub_service.list_download_sessions()
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing download sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list download sessions: {str(e)}"
        )


@router.post("/dataset/verify",
            response_model=VerificationResponse,
            summary="Verify Dataset Integrity",
            description="Verify the integrity of downloaded dataset files")
async def verify_dataset_integrity(
    request: VerificationRequest,
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> VerificationResponse:
    """Verify integrity of downloaded dataset."""
    try:
        logger.info("API request: Verify dataset integrity")
        logger.info(f"Request parameters: {request.dict()}")
        
        verification_results = await dagshub_service.verify_dataset_integrity(
            video_sets=request.video_sets,
            cameras=request.cameras
        )
        
        return VerificationResponse(
            verification_results=verification_results,
            verification_timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error verifying dataset integrity: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify dataset integrity: {str(e)}"
        )


@router.delete("/dataset/downloads/{progress_id}",
              summary="Clear Download Session",
              description="Clear a specific download session from memory")
async def clear_download_session(
    progress_id: str,
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> Dict[str, str]:
    """Clear a download session from memory."""
    try:
        if progress_id in dagshub_service._download_progress:
            del dagshub_service._download_progress[progress_id]
            return {
                "message": f"Download session {progress_id} cleared successfully",
                "progress_id": progress_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Download session not found: {progress_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing download session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear download session: {str(e)}"
        )


@router.get("/dataset/inventory",
           summary="Get File Inventory",
           description="Get a complete inventory of video files in the dataset")
async def get_dataset_inventory(
    video_sets: Optional[List[str]] = Query(None, description="Filter by video sets"),
    cameras: Optional[List[str]] = Query(None, description="Filter by cameras"),
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> Dict:
    """Get complete inventory of video files."""
    try:
        logger.info("API request: Get dataset inventory")
        
        inventory = dagshub_service.create_video_file_inventory(
            video_sets=video_sets,
            cameras=cameras
        )
        
        # Convert to serializable format
        inventory_data = []
        for file_meta in inventory:
            inventory_data.append({
                "video_set": file_meta.video_set,
                "camera_id": file_meta.camera_id,
                "sub_video_id": file_meta.sub_video_id,
                "s3_key": file_meta.s3_key,
                "local_path": str(file_meta.local_path),
                "status": file_meta.status.value,
                "file_size": file_meta.file_size,
                "checksum": file_meta.checksum
            })
        
        return {
            "total_files": len(inventory_data),
            "files": inventory_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset inventory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dataset inventory: {str(e)}"
        )


@router.get("/health",
           summary="Data Ingestion Health Check",
           description="Check the health of the data ingestion system")
async def health_check(
    dagshub_service: DagHubS3Service = Depends(get_dagshub_s3_service)
) -> Dict[str, Any]:
    """Health check for data ingestion system."""
    try:
        # Test S3 connection
        structure = dagshub_service.get_dataset_structure()
        
        return {
            "status": "healthy",
            "service": "data_ingestion",
            "s3_connection": "active",
            "available_video_sets": list(structure.keys()),
            "total_video_files": sum(
                len(videos) 
                for cameras in structure.values() 
                for videos in cameras.values()
            ),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data ingestion health check failed: {e}")
        return {
            "status": "degraded",
            "service": "data_ingestion",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }