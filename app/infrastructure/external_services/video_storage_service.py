"""
Video Storage Service implementation.

Modernized video storage service that integrates the clean S3 service
with video-specific operations and maintains backward compatibility.
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import uuid
from datetime import datetime, timezone

from app.shared.types import CameraID
from app.core.config import settings, VideoSetEnvironmentConfig
from .s3_service import ModernS3Service, create_s3_service, S3OperationResult

logger = logging.getLogger(__name__)


class VideoStorageService:
    """
    Modern video storage service with clean architecture patterns.
    
    Provides video-specific operations on top of the general S3 service,
    including multi-camera video downloads and batch operations.
    """
    
    def __init__(self, s3_service: Optional[ModernS3Service] = None):
        """
        Initialize video storage service.
        
        Args:
            s3_service: S3 service instance (creates default if not provided)
        """
        self.s3_service = s3_service or create_s3_service()
        self._download_cache: Dict[str, bool] = {}
        logger.info("VideoStorageService initialized")
    
    async def download_video_set(
        self, 
        video_set_name: str,
        task_id: uuid.UUID,
        overwrite: bool = False
    ) -> Dict[CameraID, Path]:
        """
        Download a complete video set for multi-camera processing.
        
        Args:
            video_set_name: Name of the video set configuration
            task_id: Unique task identifier for organizing downloads
            overwrite: Whether to overwrite existing local files
            
        Returns:
            Dictionary mapping camera IDs to local video file paths
            
        Raises:
            ValueError: If video set configuration not found
        """
        # Get video set configuration
        video_config = self._get_video_set_config(video_set_name)
        if not video_config:
            raise ValueError(f"Video set configuration not found: {video_set_name}")
        
        video_paths_map: Dict[CameraID, Path] = {}
        download_results = []
        
        logger.info(f"[Task {task_id}] Starting download for video set: {video_set_name}")
        
        for camera_id, s3_key in video_config.camera_video_keys.items():
            try:
                # Create local path for video file
                local_path = self._create_local_video_path(task_id, camera_id, s3_key)
                
                # Download video file
                result = await self.s3_service.download_file(
                    s3_key=s3_key,
                    local_path=str(local_path),
                    overwrite=overwrite
                )
                
                download_results.append(result)
                
                if result.success:
                    video_paths_map[camera_id] = local_path
                    logger.info(
                        f"[Task {task_id}][{camera_id}] Downloaded: {s3_key} -> {local_path.name}"
                    )
                else:
                    logger.error(
                        f"[Task {task_id}][{camera_id}] Failed to download {s3_key}: {result.error_message}"
                    )
                
            except Exception as e:
                logger.error(
                    f"[Task {task_id}][{camera_id}] Unexpected error downloading {s3_key}: {e}",
                    exc_info=True
                )
        
        # Log download summary
        successful_downloads = sum(1 for r in download_results if r.success)
        total_downloads = len(download_results)
        
        logger.info(
            f"[Task {task_id}] Video set download completed: "
            f"{successful_downloads}/{total_downloads} successful"
        )
        
        if not video_paths_map:
            raise RuntimeError(f"No videos downloaded successfully for video set: {video_set_name}")
        
        return video_paths_map
    
    def _get_video_set_config(self, video_set_name: str) -> Optional[VideoSetEnvironmentConfig]:
        """Get video set configuration by name."""
        if not hasattr(settings, 'VIDEO_SETS') or not settings.VIDEO_SETS:
            logger.error("No VIDEO_SETS configuration found in settings")
            return None
        
        video_config = settings.VIDEO_SETS.get(video_set_name)
        if not video_config:
            available_sets = ", ".join(settings.VIDEO_SETS.keys())
            logger.error(
                f"Video set '{video_set_name}' not found. Available sets: {available_sets}"
            )
            return None
        
        return video_config
    
    def _create_local_video_path(self, task_id: uuid.UUID, camera_id: CameraID, s3_key: str) -> Path:
        """Create local file path for downloaded video."""
        # Extract filename from S3 key
        filename = Path(s3_key).name
        if not filename:
            filename = f"{camera_id}_video.mp4"
        
        # Create organized directory structure
        local_dir = Path(settings.LOCAL_STORAGE_PATH) / "tasks" / str(task_id) / "videos"
        local_dir.mkdir(parents=True, exist_ok=True)
        
        return local_dir / f"{camera_id}_{filename}"
    
    async def verify_video_set_availability(
        self, 
        video_set_name: str
    ) -> Dict[CameraID, bool]:
        """
        Verify that all videos in a video set are available in S3.
        
        Args:
            video_set_name: Name of the video set configuration
            
        Returns:
            Dictionary mapping camera IDs to availability status
        """
        video_config = self._get_video_set_config(video_set_name)
        if not video_config:
            return {}
        
        availability_map: Dict[CameraID, bool] = {}
        
        logger.info(f"Verifying availability for video set: {video_set_name}")
        
        for camera_id, s3_key in video_config.camera_video_keys.items():
            try:
                exists = await self.s3_service.file_exists(s3_key)
                availability_map[camera_id] = exists
                
                if exists:
                    logger.debug(f"[{camera_id}] Video available: {s3_key}")
                else:
                    logger.warning(f"[{camera_id}] Video not found: {s3_key}")
                    
            except Exception as e:
                logger.error(f"[{camera_id}] Error checking availability of {s3_key}: {e}")
                availability_map[camera_id] = False
        
        available_count = sum(1 for available in availability_map.values() if available)
        total_count = len(availability_map)
        
        logger.info(
            f"Video set '{video_set_name}' availability: {available_count}/{total_count} videos available"
        )
        
        return availability_map
    
    async def get_video_set_metadata(
        self, 
        video_set_name: str
    ) -> Dict[CameraID, Optional[Dict]]:
        """
        Get metadata for all videos in a video set.
        
        Args:
            video_set_name: Name of the video set configuration
            
        Returns:
            Dictionary mapping camera IDs to video metadata
        """
        video_config = self._get_video_set_config(video_set_name)
        if not video_config:
            return {}
        
        metadata_map: Dict[CameraID, Optional[Dict]] = {}
        
        for camera_id, s3_key in video_config.camera_video_keys.items():
            try:
                metadata = await self.s3_service.get_file_metadata(s3_key)
                metadata_map[camera_id] = metadata
                
                if metadata:
                    file_size_mb = metadata.get('ContentLength', 0) / (1024 * 1024)
                    logger.debug(
                        f"[{camera_id}] Video metadata: {s3_key} ({file_size_mb:.1f} MB)"
                    )
                else:
                    logger.warning(f"[{camera_id}] No metadata found: {s3_key}")
                    
            except Exception as e:
                logger.error(f"[{camera_id}] Error getting metadata for {s3_key}: {e}")
                metadata_map[camera_id] = None
        
        return metadata_map
    
    async def cleanup_task_videos(self, task_id: uuid.UUID) -> int:
        """
        Clean up local video files for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Number of files cleaned up
        """
        task_video_dir = Path(settings.LOCAL_STORAGE_PATH) / "tasks" / str(task_id) / "videos"
        
        if not task_video_dir.exists():
            logger.debug(f"[Task {task_id}] No video directory to clean up")
            return 0
        
        cleaned_count = 0
        try:
            for video_file in task_video_dir.glob("*"):
                if video_file.is_file():
                    video_file.unlink()
                    cleaned_count += 1
            
            # Remove empty directory
            if task_video_dir.exists():
                task_video_dir.rmdir()
            
            # Clean up parent task directory if empty
            task_dir = task_video_dir.parent
            if task_dir.exists() and not any(task_dir.iterdir()):
                task_dir.rmdir()
            
            logger.info(f"[Task {task_id}] Cleaned up {cleaned_count} video files")
            
        except Exception as e:
            logger.error(f"[Task {task_id}] Error during video cleanup: {e}")
        
        return cleaned_count
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage service statistics.
        
        Returns:
            Dictionary containing service statistics
        """
        s3_stats = await self.s3_service.get_service_statistics()
        
        # Add video-specific statistics
        local_storage_path = Path(settings.LOCAL_STORAGE_PATH) / "tasks"
        task_count = 0
        total_video_files = 0
        total_storage_mb = 0
        
        if local_storage_path.exists():
            task_dirs = [d for d in local_storage_path.iterdir() if d.is_dir()]
            task_count = len(task_dirs)
            
            for task_dir in task_dirs:
                video_dir = task_dir / "videos"
                if video_dir.exists():
                    for video_file in video_dir.glob("*"):
                        if video_file.is_file():
                            total_video_files += 1
                            total_storage_mb += video_file.stat().st_size / (1024 * 1024)
        
        return {
            's3_statistics': s3_stats,
            'local_storage': {
                'active_tasks': task_count,
                'total_video_files': total_video_files,
                'total_storage_mb': round(total_storage_mb, 2),
                'storage_path': str(local_storage_path)
            },
            'cache_size': len(self._download_cache)
        }
    
    async def list_available_video_sets(self) -> List[str]:
        """
        List all configured video sets.
        
        Returns:
            List of video set names
        """
        if not hasattr(settings, 'VIDEO_SETS') or not settings.VIDEO_SETS:
            return []
        
        return list(settings.VIDEO_SETS.keys())
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        await self.s3_service.cleanup()
        self._download_cache.clear()
        logger.debug("VideoStorageService cleaned up")


# Global service instance
_video_storage_service: Optional[VideoStorageService] = None


def get_video_storage_service() -> VideoStorageService:
    """
    Get the global video storage service instance.
    
    Returns:
        VideoStorageService instance
    """
    global _video_storage_service
    
    if _video_storage_service is None:
        _video_storage_service = VideoStorageService()
        logger.info("Global VideoStorageService instance created")
    
    return _video_storage_service