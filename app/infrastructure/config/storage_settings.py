"""
Storage infrastructure configuration settings.

Clean infrastructure configuration for storage systems following Phase 6:
Configuration consolidation requirements. Handles S3, local storage, and
export settings with proper validation and environment management.
"""
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backend types."""
    S3 = "s3"
    LOCAL = "local"
    HYBRID = "hybrid"


class S3Provider(Enum):
    """Supported S3 providers."""
    AWS = "aws"
    DAGSHUB = "dagshub"
    MINIO = "minio"
    CUSTOM = "custom"


class CompressionType(Enum):
    """Supported compression types for exports."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"


class S3Settings(BaseModel):
    """S3-compatible storage configuration settings."""
    
    # Connection settings
    endpoint_url: Optional[str] = Field(
        default="https://s3.dagshub.com",
        description="S3 endpoint URL"
    )
    access_key_id: Optional[str] = Field(
        default=None,
        description="S3 access key ID"
    )
    secret_access_key: Optional[str] = Field(
        default=None,
        description="S3 secret access key"
    )
    region: str = Field(
        default="us-east-1",
        description="S3 region"
    )
    
    # Bucket settings
    bucket_name: str = Field(
        default="spoton_ml",
        min_length=3,
        max_length=63,
        description="S3 bucket name"
    )
    provider: S3Provider = Field(
        default=S3Provider.DAGSHUB,
        description="S3 service provider"
    )
    
    # DagHub specific settings
    dagshub_repo_owner: str = Field(
        default="Jwizzed",
        description="DagHub repository owner"
    )
    dagshub_repo_name: str = Field(
        default="spoton_ml",
        description="DagHub repository name"
    )
    
    # Performance settings
    multipart_threshold: int = Field(
        default=64 * 1024 * 1024,  # 64MB
        ge=5 * 1024 * 1024,  # Minimum 5MB
        description="Multipart upload threshold (bytes)"
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent S3 operations"
    )
    connect_timeout: int = Field(
        default=60,
        ge=5,
        le=300,
        description="S3 connection timeout (seconds)"
    )
    read_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="S3 read timeout (seconds)"
    )
    
    # Retry settings
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for S3 operations"
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff factor for retries"
    )
    
    @validator('bucket_name')
    def validate_bucket_name(cls, v):
        """Validate S3 bucket name format."""
        import re
        # S3 bucket naming rules (simplified)
        if not re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$', v):
            raise ValueError("Bucket name must be lowercase, start/end with alphanumeric, contain only letters, numbers, and hyphens")
        if '..' in v or '--' in v:
            raise ValueError("Bucket name cannot contain consecutive dots or hyphens")
        return v
    
    def validate_s3_settings(self) -> None:
        """Validate S3-specific settings."""
        # Check credentials are provided
        if not self.access_key_id or not self.secret_access_key:
            logger.warning("S3 credentials not configured - operations may fail")
        
        # Validate endpoint URL format
        if self.endpoint_url and not self.endpoint_url.startswith(('http://', 'https://')):
            logger.warning(f"S3 endpoint URL should include protocol: {self.endpoint_url}")


class LocalStorageSettings(BaseModel):
    """Local filesystem storage configuration settings."""
    
    # Base directories
    base_storage_path: str = Field(
        default="./storage",
        description="Base local storage directory"
    )
    video_download_dir: str = Field(
        default="./downloaded_videos",
        description="Directory for downloaded video files"
    )
    frame_extraction_dir: str = Field(
        default="./extracted_frames",
        description="Directory for extracted frames"
    )
    temp_storage_path: str = Field(
        default="./temp",
        description="Temporary storage directory"
    )
    cache_dir: str = Field(
        default="./cache",
        description="Cache directory for processed files"
    )
    
    # Storage limits
    max_storage_size_gb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum total storage size (GB)"
    )
    temp_file_retention_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Temporary file retention (hours)"
    )
    cache_retention_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Cache retention period (days)"
    )
    
    # Performance settings
    enable_compression: bool = Field(
        default=True,
        description="Enable compression for stored files"
    )
    compression_level: int = Field(
        default=6,
        ge=1,
        le=9,
        description="Compression level (1=fast, 9=best)"
    )
    io_buffer_size: int = Field(
        default=64 * 1024,  # 64KB
        ge=8 * 1024,  # 8KB minimum
        le=1024 * 1024,  # 1MB maximum
        description="I/O buffer size (bytes)"
    )
    
    def validate_local_storage_settings(self) -> None:
        """Validate local storage settings and create directories."""
        directories = [
            self.base_storage_path,
            self.video_download_dir,
            self.frame_extraction_dir,
            self.temp_storage_path,
            self.cache_dir
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Validated storage directory: {dir_path}")
            except Exception as e:
                logger.error(f"Cannot create storage directory {dir_path}: {e}")
                raise
    
    def get_storage_usage(self) -> Dict[str, int]:
        """
        Get current storage usage for each directory (placeholder).
        
        Returns:
            Dictionary mapping directory to size in bytes
        """
        # This would contain actual disk usage calculation
        return {
            'base_storage': 0,
            'video_download': 0,
            'frame_extraction': 0,
            'temp_storage': 0,
            'cache': 0
        }


class ExportSettings(BaseModel):
    """Data export configuration settings."""
    
    # Export directories
    export_base_dir: str = Field(
        default="./exports",
        description="Base directory for exported data"
    )
    export_temp_dir: str = Field(
        default="./exports/temp",
        description="Temporary directory for export processing"
    )
    
    # Export lifecycle
    export_expiry_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Export file expiry time (hours)"
    )
    cleanup_interval_hours: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Cleanup check interval (hours)"
    )
    
    # Export formats
    supported_formats: List[str] = Field(
        default=["json", "csv", "xml"],
        description="Supported export formats"
    )
    default_format: str = Field(
        default="json",
        description="Default export format"
    )
    compression_type: CompressionType = Field(
        default=CompressionType.GZIP,
        description="Export compression type"
    )
    
    # Performance settings
    max_export_size_mb: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum export file size (MB)"
    )
    batch_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Batch size for export processing"
    )
    
    @validator('default_format')
    def validate_default_format(cls, v, values):
        """Validate default format is in supported formats."""
        supported = values.get('supported_formats', [])
        if supported and v not in supported:
            raise ValueError(f"Default format '{v}' not in supported formats: {supported}")
        return v
    
    def validate_export_settings(self) -> None:
        """Validate export settings and create directories."""
        directories = [self.export_base_dir, self.export_temp_dir]
        
        for directory in directories:
            dir_path = Path(directory)
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Validated export directory: {dir_path}")
            except Exception as e:
                logger.error(f"Cannot create export directory {dir_path}: {e}")
                raise


class StorageSettings(BaseSettings):
    """
    Consolidated storage infrastructure configuration.
    
    Handles S3, local storage, and export settings with proper validation and
    environment management. Follows Phase 6 configuration consolidation requirements.
    """
    
    # Storage backend selection
    storage_backend: StorageBackend = Field(
        default=StorageBackend.HYBRID,
        description="Primary storage backend"
    )
    
    # Storage components
    s3: S3Settings = Field(default_factory=S3Settings)
    local: LocalStorageSettings = Field(default_factory=LocalStorageSettings)
    exports: ExportSettings = Field(default_factory=ExportSettings)
    
    # Global storage settings
    enable_caching: bool = Field(
        default=True,
        description="Enable storage caching layer"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL (seconds)"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(
        default=True,
        description="Enable storage metrics collection"
    )
    metrics_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Metrics collection interval (seconds)"
    )
    
    def __init__(self, **kwargs):
        """Initialize storage settings with validation."""
        super().__init__(**kwargs)
        self._validate_all_settings()
        logger.debug("StorageSettings initialized successfully")
    
    def validate_config(self) -> None:
        """Validate complete storage configuration."""
        self._validate_all_settings()
    
    def _validate_all_settings(self) -> None:
        """Validate all storage settings."""
        try:
            self.s3.validate_s3_settings()
            self.local.validate_local_storage_settings()
            self.exports.validate_export_settings()
            
            # Cross-component validation
            if self.storage_backend == StorageBackend.S3 and not self.s3.access_key_id:
                logger.warning("S3 backend selected but credentials not configured")
            
            if self.storage_backend == StorageBackend.LOCAL:
                total_size_gb = self.local.max_storage_size_gb
                if total_size_gb < 10:
                    logger.warning(f"Local storage limit ({total_size_gb}GB) may be too small")
            
        except Exception as e:
            logger.error(f"Storage settings validation failed: {e}")
            raise
    
    def get_active_storage_paths(self) -> Dict[str, str]:
        """
        Get active storage paths based on backend configuration.
        
        Returns:
            Dictionary mapping path type to active path
        """
        paths = {
            'base_storage': self.local.base_storage_path,
            'temp_storage': self.local.temp_storage_path,
            'cache': self.local.cache_dir,
            'exports': self.exports.export_base_dir
        }
        
        if self.storage_backend in [StorageBackend.S3, StorageBackend.HYBRID]:
            paths.update({
                's3_bucket': self.s3.bucket_name,
                's3_endpoint': self.s3.endpoint_url or ''
            })
        
        return paths
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive storage configuration summary.
        
        Returns:
            Dictionary containing configuration overview
        """
        return {
            'backend': self.storage_backend.value,
            's3': {
                'provider': self.s3.provider.value,
                'bucket': self.s3.bucket_name,
                'endpoint': self.s3.endpoint_url,
                'credentials_configured': bool(self.s3.access_key_id),
                'multipart_threshold_mb': self.s3.multipart_threshold // (1024 * 1024)
            },
            'local': {
                'base_path': self.local.base_storage_path,
                'max_size_gb': self.local.max_storage_size_gb,
                'compression_enabled': self.local.enable_compression,
                'temp_retention_hours': self.local.temp_file_retention_hours
            },
            'exports': {
                'base_dir': self.exports.export_base_dir,
                'expiry_hours': self.exports.export_expiry_hours,
                'supported_formats': self.exports.supported_formats,
                'compression': self.exports.compression_type.value
            },
            'features': {
                'caching_enabled': self.enable_caching,
                'metrics_enabled': self.enable_metrics,
                'cache_ttl_seconds': self.cache_ttl_seconds
            },
            'validation_timestamp': datetime.utcnow().isoformat()
        }
    
    class Config:
        env_prefix = "STORAGE_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        use_enum_values = True


# Default storage settings factory
def get_storage_settings() -> StorageSettings:
    """
    Get storage settings with environment-specific defaults.
    
    Returns:
        Configured StorageSettings instance
    """
    return StorageSettings()