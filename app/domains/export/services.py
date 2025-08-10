"""Export domain services for data export and reporting."""

import asyncio
import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from uuid import uuid4
import pandas as pd

from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.infrastructure.cache.redis_client import RedisClient
from app.services.analytics_engine import AnalyticsEngine
from app.core.config import settings

from .entities import (
    ExportJob,
    ExportRequest,
    ExportResult, 
    ExportFormat,
    ExportStatus,
    TrackingDataExport,
    AnalyticsReportData,
    VideoExportConfig
)

logger = logging.getLogger(__name__)


class ExportService:
    """Main export service coordinating all export operations."""
    
    def __init__(
        self,
        tracking_repository: TrackingRepository,
        redis_client: RedisClient,
        analytics_service: AnalyticsEngine,
        data_serialization_service: 'DataSerializationService',
        report_generator_service: 'ReportGeneratorService'
    ):
        self.tracking_repository = tracking_repository
        self.redis_client = redis_client
        self.analytics_service = analytics_service
        self.data_serialization_service = data_serialization_service
        self.report_generator_service = report_generator_service
        self.export_jobs: Dict[str, ExportJob] = {}
        self.export_dir = Path(settings.EXPORT_BASE_DIR)
        self.export_dir.mkdir(exist_ok=True)
    
    async def create_export_job(self, request: ExportRequest) -> ExportJob:
        """Create a new export job."""
        job_id = str(uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=settings.EXPORT_EXPIRY_HOURS)
        
        job = ExportJob(
            job_id=job_id,
            request=request,
            status=ExportStatus.PENDING,
            progress=0.0,
            started_at=None,
            completed_at=None,
            file_path=None,
            file_size=None,
            error_message=None,
            expires_at=expires_at
        )
        
        self.export_jobs[job_id] = job
        
        # Cache job info in Redis
        await self.redis_client.setex(
            f"export_job:{job_id}",
            settings.EXPORT_EXPIRY_HOURS * 3600,
            json.dumps({
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.request.created_at.isoformat(),
                "expires_at": job.expires_at.isoformat()
            })
        )
        
        logger.info(f"Created export job {job_id} for {request.export_type}")
        return job
    
    async def process_export_job(self, job_id: str) -> ExportResult:
        """Process an export job."""
        job = self.export_jobs.get(job_id)
        if not job:
            raise ValueError(f"Export job {job_id} not found")
        
        try:
            job.status = ExportStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await self._update_job_status(job)
            
            logger.info(f"Processing export job {job_id}: {job.request.export_type}")
            
            if job.request.export_type == "tracking_data":
                result = await self._export_tracking_data(job)
            elif job.request.export_type == "analytics_report":
                result = await self._export_analytics_report(job)
            elif job.request.export_type == "video_with_overlays":
                result = await self._export_video_with_overlays(job)
            else:
                raise ValueError(f"Unsupported export type: {job.request.export_type}")
            
            job.status = ExportStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.file_path = result.file_path
            job.file_size = result.file_size
            job.progress = 100.0
            
            await self._update_job_status(job)
            logger.info(f"Completed export job {job_id}")
            
            return result
            
        except Exception as e:
            job.status = ExportStatus.FAILED
            job.error_message = str(e)
            await self._update_job_status(job)
            logger.error(f"Export job {job_id} failed: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[ExportJob]:
        """Get export job status."""
        # Try memory first
        job = self.export_jobs.get(job_id)
        if job:
            return job
        
        # Try Redis cache
        cached_data = await self.redis_client.get(f"export_job:{job_id}")
        if cached_data:
            data = json.loads(cached_data)
            # Create minimal job object for status checking
            job = ExportJob(
                job_id=job_id,
                request=None,  # Not needed for status
                status=ExportStatus(data["status"]),
                progress=data["progress"],
                started_at=None,
                completed_at=None,
                file_path=None,
                file_size=None,
                error_message=None,
                expires_at=datetime.fromisoformat(data["expires_at"])
            )
            return job
        
        return None
    
    async def _export_tracking_data(self, job: ExportJob) -> ExportResult:
        """Export tracking data."""
        request = job.request
        start_time = datetime.now()
        
        # Get tracking data from repository
        tracking_data = await self.tracking_repository.get_tracking_data_by_range(
            environment_id=request.environment_id,
            start_time=request.start_time,
            end_time=request.end_time,
            filters=request.filters
        )
        
        job.progress = 30.0
        await self._update_job_status(job)
        
        # Convert to export format
        export_data = []
        for record in tracking_data:
            export_record = TrackingDataExport(
                person_id=record.person_id,
                timestamp=record.timestamp,
                camera_id=record.camera_id,
                bbox_x=record.bbox_x,
                bbox_y=record.bbox_y,
                bbox_width=record.bbox_width,
                bbox_height=record.bbox_height,
                confidence=record.confidence,
                map_x=record.map_x,
                map_y=record.map_y,
                tracking_duration=record.tracking_duration,
                detection_metadata=record.metadata or {}
            )
            export_data.append(export_record)
        
        job.progress = 60.0
        await self._update_job_status(job)
        
        # Serialize data
        file_path = await self.data_serialization_service.serialize_data(
            data=export_data,
            format=request.format,
            filename=f"tracking_data_{request.environment_id}_{job.job_id}",
            export_dir=self.export_dir
        )
        
        job.progress = 90.0
        await self._update_job_status(job)
        
        file_size = file_path.stat().st_size if file_path.exists() else 0
        duration = (datetime.now() - start_time).total_seconds()
        
        return ExportResult(
            job_id=job.job_id,
            success=True,
            file_path=file_path,
            file_size=file_size,
            record_count=len(export_data),
            duration_seconds=duration,
            error_message=None,
            metadata={
                "environment_id": request.environment_id,
                "time_range": f"{request.start_time} to {request.end_time}",
                "format": request.format.value
            }
        )
    
    async def _export_analytics_report(self, job: ExportJob) -> ExportResult:
        """Export analytics report."""
        request = job.request
        start_time = datetime.now()
        
        # Generate analytics report
        report_data = await self.analytics_service.generate_comprehensive_report(
            environment_id=request.environment_id,
            start_time=request.start_time,
            end_time=request.end_time,
            options=request.options
        )
        
        job.progress = 50.0
        await self._update_job_status(job)
        
        # Use report generator service
        file_path = await self.report_generator_service.generate_report(
            report_data=report_data,
            format=request.format,
            template_name="analytics_report",
            filename=f"analytics_report_{request.environment_id}_{job.job_id}",
            export_dir=self.export_dir
        )
        
        job.progress = 90.0
        await self._update_job_status(job)
        
        file_size = file_path.stat().st_size if file_path.exists() else 0
        duration = (datetime.now() - start_time).total_seconds()
        
        return ExportResult(
            job_id=job.job_id,
            success=True,
            file_path=file_path,
            file_size=file_size,
            record_count=1,  # One report
            duration_seconds=duration,
            error_message=None,
            metadata={
                "environment_id": request.environment_id,
                "report_type": "analytics",
                "format": request.format.value
            }
        )
    
    async def _export_video_with_overlays(self, job: ExportJob) -> ExportResult:
        """Export video with overlays."""
        # This would integrate with video processing service
        # For now, return placeholder
        raise NotImplementedError("Video export with overlays not yet implemented")
    
    async def _update_job_status(self, job: ExportJob):
        """Update job status in cache."""
        await self.redis_client.setex(
            f"export_job:{job.job_id}",
            settings.EXPORT_EXPIRY_HOURS * 3600,
            json.dumps({
                "job_id": job.job_id,
                "status": job.status.value,
                "progress": job.progress,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message
            })
        )


class DataSerializationService:
    """Service for serializing data in different formats."""
    
    async def serialize_data(
        self,
        data: List[Any],
        format: ExportFormat,
        filename: str,
        export_dir: Path
    ) -> Path:
        """Serialize data to specified format."""
        
        if format == ExportFormat.CSV:
            return await self._serialize_to_csv(data, filename, export_dir)
        elif format == ExportFormat.JSON:
            return await self._serialize_to_json(data, filename, export_dir)
        elif format == ExportFormat.EXCEL:
            return await self._serialize_to_excel(data, filename, export_dir)
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    
    async def _serialize_to_csv(self, data: List[Any], filename: str, export_dir: Path) -> Path:
        """Serialize data to CSV format."""
        file_path = export_dir / f"{filename}.csv"
        
        if not data:
            # Create empty file
            file_path.touch()
            return file_path
        
        # Convert dataclass objects to dictionaries
        dict_data = []
        for item in data:
            if hasattr(item, '__dict__'):
                dict_data.append(item.__dict__)
            else:
                dict_data.append(item)
        
        # Write CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            if dict_data:
                fieldnames = dict_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dict_data)
        
        logger.info(f"Exported {len(data)} records to CSV: {file_path}")
        return file_path
    
    async def _serialize_to_json(self, data: List[Any], filename: str, export_dir: Path) -> Path:
        """Serialize data to JSON format."""
        file_path = export_dir / f"{filename}.json"
        
        # Convert dataclass objects to dictionaries
        json_data = []
        for item in data:
            if hasattr(item, '__dict__'):
                item_dict = item.__dict__.copy()
                # Convert datetime objects to ISO strings
                for key, value in item_dict.items():
                    if isinstance(value, datetime):
                        item_dict[key] = value.isoformat()
                json_data.append(item_dict)
            else:
                json_data.append(item)
        
        # Write JSON
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} records to JSON: {file_path}")
        return file_path
    
    async def _serialize_to_excel(self, data: List[Any], filename: str, export_dir: Path) -> Path:
        """Serialize data to Excel format."""
        file_path = export_dir / f"{filename}.xlsx"
        
        if not data:
            # Create empty Excel file
            pd.DataFrame().to_excel(file_path, index=False)
            return file_path
        
        # Convert dataclass objects to dictionaries
        dict_data = []
        for item in data:
            if hasattr(item, '__dict__'):
                dict_data.append(item.__dict__)
            else:
                dict_data.append(item)
        
        # Create DataFrame and export to Excel
        df = pd.DataFrame(dict_data)
        df.to_excel(file_path, index=False)
        
        logger.info(f"Exported {len(data)} records to Excel: {file_path}")
        return file_path


class ReportGeneratorService:
    """Service for generating formatted reports."""
    
    def __init__(self):
        self.template_dir = Path("templates/reports")
        self.template_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(
        self,
        report_data: Any,
        format: ExportFormat,
        template_name: str,
        filename: str,
        export_dir: Path
    ) -> Path:
        """Generate formatted report."""
        
        if format == ExportFormat.JSON:
            return await self._generate_json_report(report_data, filename, export_dir)
        elif format == ExportFormat.PDF:
            return await self._generate_pdf_report(report_data, template_name, filename, export_dir)
        elif format == ExportFormat.EXCEL:
            return await self._generate_excel_report(report_data, filename, export_dir)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    async def _generate_json_report(self, report_data: Any, filename: str, export_dir: Path) -> Path:
        """Generate JSON report."""
        file_path = export_dir / f"{filename}.json"
        
        # Convert to dictionary if needed
        if hasattr(report_data, '__dict__'):
            data_dict = report_data.__dict__.copy()
            # Convert datetime objects to ISO strings
            for key, value in data_dict.items():
                if isinstance(value, datetime):
                    data_dict[key] = value.isoformat()
            report_data = data_dict
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated JSON report: {file_path}")
        return file_path
    
    async def _generate_pdf_report(self, report_data: Any, template_name: str, filename: str, export_dir: Path) -> Path:
        """Generate PDF report."""
        # PDF generation would require additional libraries like reportlab or weasyprint
        # For now, return placeholder
        raise NotImplementedError("PDF report generation not yet implemented")
    
    async def _generate_excel_report(self, report_data: Any, filename: str, export_dir: Path) -> Path:
        """Generate Excel report."""
        file_path = export_dir / f"{filename}.xlsx"
        
        # Create Excel workbook with multiple sheets
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            
            if hasattr(report_data, 'zone_analytics') and report_data.zone_analytics:
                # Zone analytics sheet
                zone_df = pd.DataFrame.from_dict(report_data.zone_analytics, orient='index')
                zone_df.to_excel(writer, sheet_name='Zone Analytics')
            
            if hasattr(report_data, 'camera_analytics') and report_data.camera_analytics:
                # Camera analytics sheet
                camera_df = pd.DataFrame.from_dict(report_data.camera_analytics, orient='index')
                camera_df.to_excel(writer, sheet_name='Camera Analytics')
            
            if hasattr(report_data, 'movement_patterns') and report_data.movement_patterns:
                # Movement patterns sheet
                movement_df = pd.DataFrame(report_data.movement_patterns)
                movement_df.to_excel(writer, sheet_name='Movement Patterns')
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Persons', 'Unique Persons', 'Avg Dwell Time', 'Peak Occupancy'],
                'Value': [
                    getattr(report_data, 'total_persons', 0),
                    getattr(report_data, 'unique_persons', 0),
                    getattr(report_data, 'avg_dwell_time', 0),
                    getattr(report_data, 'peak_occupancy', 0)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Generated Excel report: {file_path}")
        return file_path