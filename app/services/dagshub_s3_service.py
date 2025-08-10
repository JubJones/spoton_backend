"""
DagHub S3 service for enhanced data ingestion capabilities.

Provides DagHub-specific S3 operations with proper authentication,
batch downloading, progress tracking, and validation features.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Set
import uuid
from datetime import datetime
import json
import hashlib
from dataclasses import dataclass
from enum import Enum

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
from botocore.config import Config

from app.infrastructure.config.storage_settings import get_storage_settings, S3Provider
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class DownloadStatus(Enum):
    """Download status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VideoFileMetadata:
    """Metadata for a video file in the dataset."""
    video_set: str  # e.g., "video_s14", "video_s37"
    camera_id: str  # e.g., "c09", "c12", "c13", "c16"
    sub_video_id: str  # e.g., "sub_video_01", "sub_video_02"
    s3_key: str  # Full S3 key path
    local_path: Path  # Local destination path
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    status: DownloadStatus = DownloadStatus.PENDING


@dataclass
class DownloadProgress:
    """Progress tracking for downloads."""
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_file: Optional[str] = None
    error_messages: List[str] = None

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage based on files."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100

    @property
    def is_complete(self) -> bool:
        """Check if download is complete."""
        return (self.completed_files + self.failed_files + self.skipped_files) >= self.total_files

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "completion_percentage": self.completion_percentage,
            "is_complete": self.is_complete,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_file": self.current_file,
            "error_count": len(self.error_messages),
            "recent_errors": self.error_messages[-5:] if self.error_messages else []
        }


class DagHubS3Service:
    """
    Enhanced S3 service specifically designed for DagHub integration.
    
    Provides comprehensive data ingestion capabilities including:
    - Batch downloading with progress tracking
    - Resume capability for interrupted downloads
    - File validation and verification
    - Error handling and retry logic
    """

    def __init__(self):
        """Initialize DagHub S3 service with configuration."""
        self.storage_settings = get_storage_settings()
        self.s3_settings = self.storage_settings.s3
        
        # Validate DagHub configuration
        if self.s3_settings.provider != S3Provider.DAGSHUB:
            logger.warning(f"S3 provider is {self.s3_settings.provider}, expected DAGSHUB")
        
        # Initialize S3 client with optimized configuration
        self.s3_client = self._create_s3_client()
        
        # Progress tracking
        self._download_progress: Dict[str, DownloadProgress] = {}
        
        logger.info(f"DagHubS3Service initialized for bucket: {self.s3_settings.bucket_name}")

    def _create_s3_client(self) -> boto3.client:
        """Create optimized S3 client for DagHub."""
        try:
            # Create optimized configuration
            config = Config(
                region_name=self.s3_settings.region,
                retries={
                    'max_attempts': self.s3_settings.max_retries,
                    'mode': 'adaptive',
                    'total_max_attempts': self.s3_settings.max_retries + 2
                },
                max_pool_connections=self.s3_settings.max_concurrency,
                signature_version='s3v4',
                read_timeout=self.s3_settings.read_timeout,
                connect_timeout=self.s3_settings.connect_timeout
            )
            
            # Create S3 client
            client = boto3.client(
                's3',
                endpoint_url=self.s3_settings.endpoint_url,
                aws_access_key_id=self.s3_settings.access_key_id,
                aws_secret_access_key=self.s3_settings.secret_access_key,
                config=config
            )
            
            # Test connection
            logger.info("Testing DagHub S3 connection...")
            client.head_bucket(Bucket=self.s3_settings.bucket_name)
            logger.info("✅ DagHub S3 connection successful")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create DagHub S3 client: {e}")
            raise

    def get_dataset_structure(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the complete dataset structure from DagHub S3.
        
        Returns:
            Dictionary mapping video_set -> camera_id -> list of sub_videos
            
        Example:
            {
                "video_s14": {
                    "c09": ["sub_video_01.mp4", "sub_video_02.mp4", ...],
                    "c12": ["sub_video_01.mp4", "sub_video_02.mp4", ...],
                    ...
                },
                "video_s37": {
                    "c01": ["sub_video_01.mp4", "sub_video_02.mp4", ...],
                    ...
                }
            }
        """
        logger.info("Discovering dataset structure from DagHub S3...")
        
        try:
            # List all objects in the bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_settings.bucket_name)
            
            structure = {}
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Parse key structure: video_set/camera_id/sub_video_XX.mp4
                    parts = key.split('/')
                    if len(parts) == 3 and parts[2].endswith('.mp4'):
                        video_set, camera_id, filename = parts
                        
                        if video_set not in structure:
                            structure[video_set] = {}
                        if camera_id not in structure[video_set]:
                            structure[video_set][camera_id] = []
                            
                        structure[video_set][camera_id].append(filename)
            
            # Sort sub-videos for consistent ordering
            for video_set in structure:
                for camera_id in structure[video_set]:
                    structure[video_set][camera_id].sort()
            
            logger.info(f"Dataset structure discovery complete:")
            for video_set, cameras in structure.items():
                logger.info(f"  {video_set}: {len(cameras)} cameras, {sum(len(vids) for vids in cameras.values())} total videos")
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to discover dataset structure: {e}")
            raise

    def create_video_file_inventory(self, 
                                    video_sets: Optional[List[str]] = None,
                                    cameras: Optional[List[str]] = None) -> List[VideoFileMetadata]:
        """
        Create comprehensive inventory of video files to download.
        
        Args:
            video_sets: Specific video sets to include (default: all)
            cameras: Specific cameras to include (default: all)
            
        Returns:
            List of VideoFileMetadata objects
        """
        logger.info("Creating video file inventory...")
        
        structure = self.get_dataset_structure()
        inventory = []
        
        # Filter video sets if specified
        if video_sets:
            structure = {k: v for k, v in structure.items() if k in video_sets}
        
        for video_set, cameras_dict in structure.items():
            # Filter cameras if specified
            if cameras:
                cameras_dict = {k: v for k, v in cameras_dict.items() if k in cameras}
            
            for camera_id, sub_videos in cameras_dict.items():
                for sub_video in sub_videos:
                    s3_key = f"{video_set}/{camera_id}/{sub_video}"
                    
                    # Create local path structure
                    local_path = (
                        Path(self.storage_settings.local.video_download_dir) /
                        video_set / camera_id / sub_video
                    )
                    
                    metadata = VideoFileMetadata(
                        video_set=video_set,
                        camera_id=camera_id,
                        sub_video_id=sub_video.replace('.mp4', ''),
                        s3_key=s3_key,
                        local_path=local_path
                    )
                    
                    inventory.append(metadata)
        
        logger.info(f"Created inventory of {len(inventory)} video files")
        return inventory

    async def get_file_metadata(self, s3_key: str) -> Optional[Dict]:
        """Get metadata for a specific S3 object."""
        try:
            response = await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=self.s3_settings.bucket_name,
                Key=s3_key
            )
            return {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag', '').strip('"')
            }
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"File not found: {s3_key}")
                return None
            logger.error(f"Error getting metadata for {s3_key}: {e}")
            return None

    async def download_file_with_validation(self, 
                                            file_metadata: VideoFileMetadata,
                                            progress_callback: Optional[callable] = None) -> bool:
        """
        Download a single file with validation and error handling.
        
        Args:
            file_metadata: VideoFileMetadata object
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create parent directory
            file_metadata.local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists and validate
            if file_metadata.local_path.exists():
                if await self._validate_existing_file(file_metadata):
                    logger.debug(f"File already exists and is valid: {file_metadata.local_path}")
                    file_metadata.status = DownloadStatus.SKIPPED
                    return True
                else:
                    logger.info(f"Existing file invalid, re-downloading: {file_metadata.local_path}")
            
            # Get file metadata from S3
            s3_metadata = await self.get_file_metadata(file_metadata.s3_key)
            if not s3_metadata:
                logger.error(f"Cannot get S3 metadata for: {file_metadata.s3_key}")
                file_metadata.status = DownloadStatus.FAILED
                return False
            
            file_metadata.file_size = s3_metadata['size']
            file_metadata.checksum = s3_metadata['etag']
            
            # Download file
            logger.info(f"Downloading {file_metadata.s3_key} -> {file_metadata.local_path}")
            file_metadata.status = DownloadStatus.IN_PROGRESS
            
            await asyncio.to_thread(
                self.s3_client.download_file,
                Bucket=self.s3_settings.bucket_name,
                Key=file_metadata.s3_key,
                Filename=str(file_metadata.local_path)
            )
            
            # Validate downloaded file
            if await self._validate_downloaded_file(file_metadata):
                file_metadata.status = DownloadStatus.COMPLETED
                logger.info(f"✅ Successfully downloaded: {file_metadata.s3_key}")
                
                if progress_callback:
                    progress_callback(file_metadata)
                
                return True
            else:
                file_metadata.status = DownloadStatus.FAILED
                logger.error(f"❌ Downloaded file validation failed: {file_metadata.s3_key}")
                return False
                
        except Exception as e:
            file_metadata.status = DownloadStatus.FAILED
            logger.error(f"❌ Download failed for {file_metadata.s3_key}: {e}")
            return False

    async def _validate_existing_file(self, file_metadata: VideoFileMetadata) -> bool:
        """Validate an existing local file."""
        try:
            # Check file size
            local_size = file_metadata.local_path.stat().st_size
            if local_size == 0:
                return False
            
            # For now, basic validation - can be enhanced with checksum verification
            return True
            
        except Exception as e:
            logger.debug(f"Error validating existing file {file_metadata.local_path}: {e}")
            return False

    async def _validate_downloaded_file(self, file_metadata: VideoFileMetadata) -> bool:
        """Validate a downloaded file."""
        try:
            # Check file exists and has size
            if not file_metadata.local_path.exists():
                return False
            
            local_size = file_metadata.local_path.stat().st_size
            if local_size == 0:
                return False
            
            # Validate size matches S3 metadata
            if file_metadata.file_size and local_size != file_metadata.file_size:
                logger.error(f"Size mismatch: local={local_size}, expected={file_metadata.file_size}")
                return False
            
            # Basic video file validation (check if it's a valid MP4)
            if file_metadata.local_path.suffix.lower() == '.mp4':
                # Could add more sophisticated video validation here
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating downloaded file {file_metadata.local_path}: {e}")
            return False

    async def batch_download_videos(self, 
                                    video_sets: Optional[List[str]] = None,
                                    cameras: Optional[List[str]] = None,
                                    max_concurrent: Optional[int] = None,
                                    progress_id: Optional[str] = None) -> DownloadProgress:
        """
        Download multiple videos with progress tracking and concurrency control.
        
        Args:
            video_sets: Specific video sets to download (default: all)
            cameras: Specific cameras to download (default: all)
            max_concurrent: Maximum concurrent downloads (default: from config)
            progress_id: Unique identifier for tracking progress
            
        Returns:
            DownloadProgress object with results
        """
        if max_concurrent is None:
            max_concurrent = self.s3_settings.max_concurrency
        
        if progress_id is None:
            progress_id = str(uuid.uuid4())
        
        logger.info(f"Starting batch download [ID: {progress_id}]")
        logger.info(f"Parameters: video_sets={video_sets}, cameras={cameras}, max_concurrent={max_concurrent}")
        
        # Create file inventory
        inventory = self.create_video_file_inventory(video_sets, cameras)
        
        # Initialize progress tracking
        progress = DownloadProgress(
            total_files=len(inventory),
            start_time=datetime.utcnow()
        )
        self._download_progress[progress_id] = progress
        
        # Progress callback
        def update_progress(file_metadata: VideoFileMetadata):
            if file_metadata.status == DownloadStatus.COMPLETED:
                progress.completed_files += 1
                if file_metadata.file_size:
                    progress.downloaded_bytes += file_metadata.file_size
            elif file_metadata.status == DownloadStatus.FAILED:
                progress.failed_files += 1
                progress.error_messages.append(f"Failed: {file_metadata.s3_key}")
            elif file_metadata.status == DownloadStatus.SKIPPED:
                progress.skipped_files += 1
            
            progress.current_file = file_metadata.s3_key
            
            # Log progress periodically
            total_processed = progress.completed_files + progress.failed_files + progress.skipped_files
            if total_processed % 10 == 0 or progress.is_complete:
                logger.info(f"Progress [{progress_id}]: {progress.completion_percentage:.1f}% "
                           f"({progress.completed_files}/{progress.total_files} completed, "
                           f"{progress.failed_files} failed, {progress.skipped_files} skipped)")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(file_metadata: VideoFileMetadata):
            async with semaphore:
                return await self.download_file_with_validation(file_metadata, update_progress)
        
        # Execute downloads
        logger.info(f"Starting {len(inventory)} downloads with max_concurrent={max_concurrent}")
        
        tasks = [download_with_semaphore(file_meta) for file_meta in inventory]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Finalize progress
        progress.end_time = datetime.utcnow()
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                progress.failed_files += 1
                progress.error_messages.append(f"Exception: {inventory[i].s3_key} - {str(result)}")
                inventory[i].status = DownloadStatus.FAILED
        
        logger.info(f"Batch download complete [{progress_id}]:")
        logger.info(f"  ✅ Completed: {progress.completed_files}")
        logger.info(f"  ⏭️  Skipped: {progress.skipped_files}")
        logger.info(f"  ❌ Failed: {progress.failed_files}")
        logger.info(f"  📊 Success Rate: {(progress.completed_files + progress.skipped_files) / progress.total_files * 100:.1f}%")
        
        return progress

    def get_download_progress(self, progress_id: str) -> Optional[DownloadProgress]:
        """Get download progress by ID."""
        return self._download_progress.get(progress_id)

    def list_download_sessions(self) -> Dict[str, Dict]:
        """List all download sessions and their progress."""
        return {
            progress_id: progress.to_dict()
            for progress_id, progress in self._download_progress.items()
        }

    async def verify_dataset_integrity(self, 
                                       video_sets: Optional[List[str]] = None,
                                       cameras: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Verify integrity of downloaded dataset.
        
        Returns:
            Dictionary with verification results
        """
        logger.info("Starting dataset integrity verification...")
        
        inventory = self.create_video_file_inventory(video_sets, cameras)
        
        results = {
            'total_files': len(inventory),
            'existing_files': 0,
            'missing_files': 0,
            'invalid_files': 0,
            'total_size_bytes': 0,
            'missing_file_list': [],
            'invalid_file_list': []
        }
        
        for file_metadata in inventory:
            if file_metadata.local_path.exists():
                results['existing_files'] += 1
                results['total_size_bytes'] += file_metadata.local_path.stat().st_size
                
                # Basic validation
                if not await self._validate_existing_file(file_metadata):
                    results['invalid_files'] += 1
                    results['invalid_file_list'].append(str(file_metadata.local_path))
            else:
                results['missing_files'] += 1
                results['missing_file_list'].append(str(file_metadata.local_path))
        
        results['completeness_percentage'] = (results['existing_files'] / results['total_files']) * 100
        results['integrity_score'] = ((results['existing_files'] - results['invalid_files']) / results['total_files']) * 100
        
        logger.info(f"Dataset verification complete:")
        logger.info(f"  📁 Files: {results['existing_files']}/{results['total_files']} ({results['completeness_percentage']:.1f}% complete)")
        logger.info(f"  ✅ Integrity: {results['integrity_score']:.1f}%")
        logger.info(f"  💾 Total Size: {results['total_size_bytes'] / (1024**3):.2f} GB")
        
        return results


# Factory function for dependency injection
def get_dagshub_s3_service() -> DagHubS3Service:
    """Get DagHub S3 service instance."""
    return DagHubS3Service()