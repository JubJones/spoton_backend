"""Export domain entities for data export and reporting."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path


class ExportFormat(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"
    VIDEO = "video"


class ExportStatus(str, Enum):
    """Export job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ExportRequest:
    """Export request configuration."""
    export_id: str
    export_type: str  # tracking_data, analytics_report, video_with_overlays
    format: ExportFormat
    environment_id: str
    start_time: datetime
    end_time: datetime
    filters: Dict[str, Any]
    options: Dict[str, Any]
    requested_by: str
    created_at: datetime


@dataclass 
class ExportJob:
    """Export job tracking entity."""
    job_id: str
    request: ExportRequest
    status: ExportStatus
    progress: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    file_path: Optional[Path]
    file_size: Optional[int]
    error_message: Optional[str]
    expires_at: datetime


@dataclass
class ExportResult:
    """Export operation result."""
    job_id: str
    success: bool
    file_path: Optional[Path]
    file_size: Optional[int]
    record_count: Optional[int]
    duration_seconds: float
    error_message: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ReportConfig:
    """Report generation configuration."""
    report_id: str
    title: str
    description: str
    template_path: str
    data_sources: List[str]
    parameters: Dict[str, Any]
    output_formats: List[ExportFormat]
    schedule: Optional[str]  # Cron expression for scheduled reports


@dataclass
class TrackingDataExport:
    """Tracking data export structure."""
    person_id: str
    timestamp: datetime
    camera_id: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    confidence: float
    map_x: Optional[float]
    map_y: Optional[float]
    tracking_duration: float
    detection_metadata: Dict[str, Any]


@dataclass
class AnalyticsReportData:
    """Analytics report data structure."""
    environment_id: str
    report_period: str
    total_persons: int
    unique_persons: int 
    avg_dwell_time: float
    peak_occupancy: int
    zone_analytics: Dict[str, Dict[str, float]]
    camera_analytics: Dict[str, Dict[str, float]]
    behavioral_patterns: Dict[str, Any]
    movement_patterns: List[Dict[str, Any]]


@dataclass
class VideoExportConfig:
    """Video export configuration."""
    video_id: str
    camera_ids: List[str]
    start_time: datetime
    end_time: datetime
    include_overlays: bool
    overlay_config: Dict[str, Any]
    quality: str  # low, medium, high
    fps: int
    resolution: tuple[int, int]
    codec: str