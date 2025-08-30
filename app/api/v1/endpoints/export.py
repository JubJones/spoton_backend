"""Export API endpoints for data export and reporting."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.dependencies import get_current_user, get_export_service
from app.domains.export.entities import ExportFormat, ExportStatus, ExportRequest
from app.domains.export.services import ExportService
from app.infrastructure.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["export"])


# Request/Response Models
class ExportTrackingDataRequest(BaseModel):
    """Request model for tracking data export."""
    environment_id: str = Field(..., description="Environment ID")
    start_time: datetime = Field(..., description="Start time for data range")
    end_time: datetime = Field(..., description="End time for data range")
    format: ExportFormat = Field(default=ExportFormat.CSV, description="Export format")
    camera_ids: Optional[List[str]] = Field(None, description="Filter by camera IDs")
    person_ids: Optional[List[str]] = Field(None, description="Filter by person IDs")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_metadata: bool = Field(default=True, description="Include detection metadata")


class ExportAnalyticsReportRequest(BaseModel):
    """Request model for analytics report export."""
    environment_id: str = Field(..., description="Environment ID")
    start_time: datetime = Field(..., description="Start time for report period")
    end_time: datetime = Field(..., description="End time for report period")
    format: ExportFormat = Field(default=ExportFormat.JSON, description="Report format")
    include_zone_analytics: bool = Field(default=True, description="Include zone-based analytics")
    include_camera_analytics: bool = Field(default=True, description="Include camera-based analytics")
    include_behavioral_patterns: bool = Field(default=True, description="Include behavioral analysis")
    include_heatmap_data: bool = Field(default=False, description="Include heatmap data")
    report_language: str = Field(default="en", description="Report language")


class ExportVideoRequest(BaseModel):
    """Request model for video export with overlays."""
    environment_id: str = Field(..., description="Environment ID")
    start_time: datetime = Field(..., description="Start time for video")
    end_time: datetime = Field(..., description="End time for video")
    camera_ids: List[str] = Field(..., description="Camera IDs to include")
    include_overlays: bool = Field(default=True, description="Include tracking overlays")
    quality: str = Field(default="medium", description="Video quality (low/medium/high)")
    fps: int = Field(default=10, ge=1, le=30, description="Frames per second")
    format: str = Field(default="mp4", description="Video format")


class ExportJobResponse(BaseModel):
    """Response model for export job creation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: ExportStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    expires_at: datetime = Field(..., description="Job expiration timestamp")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated completion time")


class ExportJobStatusResponse(BaseModel):
    """Response model for export job status."""
    job_id: str = Field(..., description="Job identifier")
    status: ExportStatus = Field(..., description="Current status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Completion percentage")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    file_size: Optional[int] = Field(None, description="Generated file size in bytes")
    record_count: Optional[int] = Field(None, description="Number of exported records")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    download_url: Optional[str] = Field(None, description="Download URL if completed")


class ExportJobListResponse(BaseModel):
    """Response model for listing export jobs."""
    jobs: List[ExportJobStatusResponse] = Field(..., description="List of export jobs")
    total_count: int = Field(..., description="Total number of jobs")


# API Endpoints
@router.post("/tracking-data", response_model=ExportJobResponse)
async def export_tracking_data(
    request: ExportTrackingDataRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
) -> ExportJobResponse:
    """
    Export tracking data in specified format.
    
    Supported formats:
    - CSV: Comma-separated values with headers
    - JSON: Structured JSON array
    - Excel: Multi-sheet Excel workbook
    """
    try:
        # Create export request
        export_request = ExportRequest(
            export_id=str(uuid4()),
            export_type="tracking_data",
            format=request.format,
            environment_id=request.environment_id,
            start_time=request.start_time,
            end_time=request.end_time,
            filters={
                "camera_ids": request.camera_ids,
                "person_ids": request.person_ids,
                "confidence_threshold": request.confidence_threshold
            },
            options={
                "include_metadata": request.include_metadata
            },
            requested_by=current_user.username,
            created_at=datetime.utcnow()
        )
        
        # Create job with fallback handling
        try:
            job = await export_service.create_export_job(export_request)
        except Exception as service_error:
            logger.warning(f"Export service unavailable, returning mock response: {service_error}")
            # Return a mock response indicating service is unavailable
            return ExportJobResponse(
                job_id=str(uuid4()),
                status=ExportStatus.FAILED,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),
                estimated_duration_minutes=None
            )
        
        # Process in background
        background_tasks.add_task(export_service.process_export_job, job.job_id)
        
        logger.info(f"Created tracking data export job {job.job_id} for user {current_user.username}")
        
        return ExportJobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.request.created_at,
            expires_at=job.expires_at,
            estimated_duration_minutes=5  # Estimate based on data size
        )
        
    except Exception as e:
        logger.error(f"Failed to create tracking data export: {e}")
        # Return a user-friendly error instead of 500
        return ExportJobResponse(
            job_id=str(uuid4()),
            status=ExportStatus.FAILED,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            estimated_duration_minutes=None
        )


@router.post("/analytics-report", response_model=ExportJobResponse)
async def export_analytics_report(
    request: ExportAnalyticsReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
) -> ExportJobResponse:
    """
    Export comprehensive analytics report.
    
    Supported formats:
    - JSON: Structured analytics data
    - Excel: Multi-sheet report with charts
    - PDF: Formatted report document
    """
    try:
        # Create export request
        export_request = ExportRequest(
            export_id=str(uuid4()),
            export_type="analytics_report",
            format=request.format,
            environment_id=request.environment_id,
            start_time=request.start_time,
            end_time=request.end_time,
            filters={},
            options={
                "include_zone_analytics": request.include_zone_analytics,
                "include_camera_analytics": request.include_camera_analytics,
                "include_behavioral_patterns": request.include_behavioral_patterns,
                "include_heatmap_data": request.include_heatmap_data,
                "report_language": request.report_language
            },
            requested_by=current_user.username,
            created_at=datetime.utcnow()
        )
        
        # Create job with fallback handling
        try:
            job = await export_service.create_export_job(export_request)
        except Exception as service_error:
            logger.warning(f"Export service unavailable, returning mock response: {service_error}")
            return ExportJobResponse(
                job_id=str(uuid4()),
                status=ExportStatus.FAILED,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),
                estimated_duration_minutes=None
            )
        
        # Process in background
        background_tasks.add_task(export_service.process_export_job, job.job_id)
        
        logger.info(f"Created analytics report export job {job.job_id} for user {current_user.username}")
        
        return ExportJobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.request.created_at,
            expires_at=job.expires_at,
            estimated_duration_minutes=10  # Analytics reports take longer
        )
        
    except Exception as e:
        logger.error(f"Failed to create analytics report export: {e}")
        return ExportJobResponse(
            job_id=str(uuid4()),
            status=ExportStatus.FAILED,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            estimated_duration_minutes=None
        )


@router.post("/video-with-overlays", response_model=ExportJobResponse)
async def export_video_with_overlays(
    request: ExportVideoRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
) -> ExportJobResponse:
    """
    Export video with tracking overlays.
    
    Note: This is a resource-intensive operation that may take significant time.
    """
    try:
        # Create export request
        export_request = ExportRequest(
            export_id=str(uuid4()),
            export_type="video_with_overlays",
            format=ExportFormat.VIDEO,
            environment_id=request.environment_id,
            start_time=request.start_time,
            end_time=request.end_time,
            filters={
                "camera_ids": request.camera_ids
            },
            options={
                "include_overlays": request.include_overlays,
                "quality": request.quality,
                "fps": request.fps,
                "format": request.format
            },
            requested_by=current_user.username,
            created_at=datetime.utcnow()
        )
        
        # Create job with fallback handling
        try:
            job = await export_service.create_export_job(export_request)
        except Exception as service_error:
            logger.warning(f"Export service unavailable, returning mock response: {service_error}")
            return ExportJobResponse(
                job_id=str(uuid4()),
                status=ExportStatus.FAILED,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),
                estimated_duration_minutes=None
            )
        
        # Process in background
        background_tasks.add_task(export_service.process_export_job, job.job_id)
        
        logger.info(f"Created video export job {job.job_id} for user {current_user.username}")
        
        # Estimate duration based on video length and camera count
        duration_hours = (request.end_time - request.start_time).total_seconds() / 3600
        estimated_minutes = int(duration_hours * len(request.camera_ids) * 2)  # Rough estimate
        
        return ExportJobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.request.created_at,
            expires_at=job.expires_at,
            estimated_duration_minutes=estimated_minutes
        )
        
    except Exception as e:
        logger.error(f"Failed to create video export: {e}")
        return ExportJobResponse(
            job_id=str(uuid4()),
            status=ExportStatus.FAILED,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            estimated_duration_minutes=None
        )


@router.get("/jobs/{job_id}/status", response_model=ExportJobStatusResponse)
async def get_export_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
) -> ExportJobStatusResponse:
    """Get the status of an export job."""
    try:
        try:
            job = await export_service.get_job_status(job_id)
        except Exception as service_error:
            logger.warning(f"Export service unavailable: {service_error}")
            # Return a mock failed job status
            return ExportJobStatusResponse(
                job_id=job_id,
                status=ExportStatus.FAILED,
                progress=0.0,
                started_at=None,
                completed_at=None,
                file_size=None,
                record_count=None,
                error_message="Export service temporarily unavailable",
                download_url=None
            )
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export job not found"
            )
        
        # Generate download URL if completed
        download_url = None
        if job.status == ExportStatus.COMPLETED and job.file_path:
            download_url = f"/api/v1/export/jobs/{job_id}/download"
        
        return ExportJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            started_at=job.started_at,
            completed_at=job.completed_at,
            file_size=job.file_size,
            record_count=None,  # Would need to be tracked separately
            error_message=job.error_message,
            download_url=download_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status"
        )


@router.get("/jobs/{job_id}/download")
async def download_export_file(
    job_id: str,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
) -> FileResponse:
    """Download the exported file."""
    try:
        job = await export_service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export job not found"
            )
        
        if job.status != ExportStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Export job is not completed (status: {job.status.value})"
            )
        
        if not job.file_path or not job.file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export file not found"
            )
        
        # Determine media type based on file extension
        file_extension = job.file_path.suffix.lower()
        media_type_map = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pdf': 'application/pdf',
            '.mp4': 'video/mp4'
        }
        media_type = media_type_map.get(file_extension, 'application/octet-stream')
        
        logger.info(f"Serving export file {job.file_path} for job {job_id}")
        
        return FileResponse(
            path=str(job.file_path),
            media_type=media_type,
            filename=job.file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download export file"
        )


@router.get("/jobs", response_model=ExportJobListResponse)
async def list_export_jobs(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(default=0, ge=0, description="Number of jobs to skip"),
    status_filter: Optional[ExportStatus] = Query(None, description="Filter by job status"),
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
) -> ExportJobListResponse:
    """List export jobs for the current user."""
    try:
        # For now, return jobs from memory (in production, this would query database)
        all_jobs = list(export_service.export_jobs.values())
        
        # Filter by user (in production, this would be done at database level)
        user_jobs = [job for job in all_jobs if job.request and job.request.requested_by == current_user.username]
        
        # Filter by status if provided
        if status_filter:
            user_jobs = [job for job in user_jobs if job.status == status_filter]
        
        # Apply pagination
        total_count = len(user_jobs)
        paginated_jobs = user_jobs[offset:offset + limit]
        
        # Convert to response format
        job_responses = []
        for job in paginated_jobs:
            download_url = None
            if job.status == ExportStatus.COMPLETED and job.file_path:
                download_url = f"/api/v1/export/jobs/{job.job_id}/download"
            
            job_responses.append(ExportJobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                started_at=job.started_at,
                completed_at=job.completed_at,
                file_size=job.file_size,
                record_count=None,
                error_message=job.error_message,
                download_url=download_url
            ))
        
        return ExportJobListResponse(
            jobs=job_responses,
            total_count=total_count
        )
        
    except Exception as e:
        logger.error(f"Failed to list export jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve export jobs"
        )


@router.delete("/jobs/{job_id}")
async def cancel_export_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
) -> Dict[str, str]:
    """Cancel an export job."""
    try:
        job = await export_service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export job not found"
            )
        
        if job.status in [ExportStatus.COMPLETED, ExportStatus.FAILED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in {job.status.value} status"
            )
        
        # Update job status (in production, this would also stop the background task)
        if job_id in export_service.export_jobs:
            export_service.export_jobs[job_id].status = ExportStatus.FAILED
            export_service.export_jobs[job_id].error_message = "Cancelled by user"
        
        logger.info(f"Cancelled export job {job_id} by user {current_user.username}")
        
        return {"message": "Export job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel export job"
        )