"""
Module for managing and providing application dependencies.
Leverages FastAPI's dependency injection system and app.state for preloaded components.
"""
from functools import lru_cache
from typing import Optional
import logging
import uuid
from dataclasses import dataclass

from fastapi import Depends, Request, HTTPException, status, Header
import torch

from app.core.config import settings
# Keys are still useful for conceptual grouping or if we switched to setattr/getattr
# from app.core.event_handlers import (
#     DETECTOR_KEY, TRACKER_FACTORY_KEY, HOMOGRAPHY_SERVICE_KEY, COMPUTE_DEVICE_KEY
# )
from app.utils.asset_downloader import AssetDownloader
from app.services.video_data_manager_service import VideoDataManagerService
from app.services.notification_service import NotificationService
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.multi_camera_frame_processor import MultiCameraFrameProcessor
from app.services.homography_service import HomographyService

from app.models.base_models import AbstractDetector
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.infrastructure.cache.redis_client import RedisClient
from app.services.analytics_engine import AnalyticsEngine

from app.api.websockets import binary_websocket_manager as websocket_manager
from app.services.raw_video_service import raw_video_service, RawVideoService
from app.services.detection_video_service import detection_video_service, DetectionVideoService
from app.services.playback_status_store import PlaybackStatusStore
from app.services.task_runtime_registry import TaskRuntimeRegistry
from app.dependencies import (
    get_playback_status_store as global_playback_status_store,
    get_task_runtime_registry as global_task_runtime_registry,
)

logger = logging.getLogger(__name__)


# Mock classes for graceful fallback when services are unavailable
class MockTrackingRepository:
    """Mock tracking repository for fallback scenarios."""
    
    async def get_tracking_data_by_range(self, **kwargs):
        """Return empty tracking data."""
        return []


class MockRedisClient:
    """Mock Redis client for fallback scenarios."""
    
    async def setex(self, key: str, timeout: int, value: str):
        """Mock setex operation."""
        pass
    
    async def get(self, key: str):
        """Mock get operation."""
        return None


class MockAnalyticsEngine:
    """Mock analytics engine for fallback scenarios."""
    
    async def generate_comprehensive_report(self, **kwargs):
        """Return mock analytics report."""
        return {
            "total_persons": 0,
            "unique_persons": 0,
            "avg_dwell_time": 0,
            "peak_occupancy": 0,
            "zone_analytics": {},
            "camera_analytics": {},
            "movement_patterns": []
        }


@dataclass
class User:
    """Simple user model for export endpoints."""
    username: str
    user_id: str = None
    
    def __post_init__(self):
        if self.user_id is None:
            self.user_id = str(uuid.uuid4())


def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """Simple authentication dependency. In production, this would validate JWT tokens."""
    # For development/demo purposes, create a default user
    # In production, this would validate the authorization header and extract user info
    if authorization:
        # Extract user info from token (simplified)
        username = "admin"  # Would extract from JWT
    else:
        username = "anonymous"
    
    return User(username=username)


# --- Accessing Preloaded Components from app.state ---

def get_compute_device(request: Request) -> torch.device:
    """Retrieves the pre-configured compute device from app.state."""
    # Direct attribute access
    if not hasattr(request.app.state, 'compute_device') or request.app.state.compute_device is None:
        logger.error("Compute device not found in app.state. Startup might have failed.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Compute device not initialized.")
    return request.app.state.compute_device

def get_detector(request: Request) -> AbstractDetector:
    """Retrieves the preloaded detector instance from app.state."""
    # Direct attribute access
    if not hasattr(request.app.state, 'detector') or request.app.state.detector is None:
        logger.error("Detector instance not found in app.state (attribute 'detector'). Model preloading might have failed.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Detector model not available.")
    return request.app.state.detector

def get_camera_tracker_factory(request: Request) -> CameraTrackerFactory:
    """Retrieves the preloaded CameraTrackerFactory instance from app.state."""
    # Direct attribute access
    if not hasattr(request.app.state, 'tracker_factory') or request.app.state.tracker_factory is None:
        logger.error("CameraTrackerFactory instance not found in app.state (attribute 'tracker_factory'). Preloading might have failed.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Tracker factory not available.")
    return request.app.state.tracker_factory

def get_homography_service(request: Request) -> HomographyService:
    """Retrieves the preloaded HomographyService instance from app.state."""
    # Direct attribute access
    if not hasattr(request.app.state, 'homography_service') or request.app.state.homography_service is None:
        logger.error("HomographyService instance not found in app.state (attribute 'homography_service'). Preloading might have failed.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Homography service not available.")
    return request.app.state.homography_service


# --- Other Singleton Services (can use lru_cache or be instantiated once) ---

@lru_cache()
def get_asset_downloader() -> AssetDownloader:
    """Dependency provider for AssetDownloader."""
    return AssetDownloader(
        s3_endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        s3_bucket_name=settings.S3_BUCKET_NAME
    )

@lru_cache()
def get_video_data_manager_service(
    asset_downloader: AssetDownloader = Depends(get_asset_downloader)
) -> VideoDataManagerService:
    """Dependency provider for VideoDataManagerService."""
    return VideoDataManagerService(asset_downloader=asset_downloader)

@lru_cache()
def get_notification_service() -> NotificationService:
    """Dependency provider for NotificationService (singleton)."""
    return NotificationService(manager=websocket_manager)


# --- Services that depend on preloaded components ---

@lru_cache()
def get_multi_camera_frame_processor(
    detector: AbstractDetector = Depends(get_detector),
    tracker_factory: CameraTrackerFactory = Depends(get_camera_tracker_factory),
    homography_service: HomographyService = Depends(get_homography_service),
    notification_service: NotificationService = Depends(get_notification_service),
    device: torch.device = Depends(get_compute_device)
) -> MultiCameraFrameProcessor:
    """Dependency provider for MultiCameraFrameProcessor."""
    logger.debug("Initializing MultiCameraFrameProcessor instance (or returning cached).")
    return MultiCameraFrameProcessor(
        detector=detector,
        tracker_factory=tracker_factory,
        homography_service=homography_service,
        notification_service=notification_service,
        device=device
    )

# Note: Legacy PipelineOrchestrator removed with Re-ID pipeline deprecation.

# --- Export Services ---

@lru_cache()
def get_tracking_repository() -> TrackingRepository:
    """Dependency provider for TrackingRepository."""
    try:
        return TrackingRepository()
    except Exception as e:
        logger.warning(f"Failed to create TrackingRepository, using mock: {e}")
        return MockTrackingRepository()

@lru_cache()
def get_redis_client() -> RedisClient:
    """Dependency provider for RedisClient."""
    try:
        return RedisClient()
    except Exception as e:
        logger.warning(f"Failed to create RedisClient, using mock: {e}")
        return MockRedisClient()

@lru_cache()
def get_analytics_service() -> AnalyticsEngine:
    """Dependency provider for AnalyticsEngine."""
    try:
        return AnalyticsEngine()
    except Exception as e:
        logger.warning(f"Failed to create AnalyticsEngine, using mock: {e}")
        return MockAnalyticsEngine()


@lru_cache()
def get_raw_video_service(
    playback_status_store: PlaybackStatusStore = Depends(global_playback_status_store),
    playback_runtime_registry: TaskRuntimeRegistry = Depends(global_task_runtime_registry),
) -> RawVideoService:
    """Dependency provider for RawVideoService."""
    logger.debug("Initializing RawVideoService instance (or returning cached).")
    service = raw_video_service
    service.attach_playback_interfaces(
        status_store=playback_status_store,
        runtime_registry=playback_runtime_registry,
    )
    return service


@lru_cache()
def get_detection_video_service(
    playback_status_store: PlaybackStatusStore = Depends(global_playback_status_store),
    playback_runtime_registry: TaskRuntimeRegistry = Depends(global_task_runtime_registry),
) -> DetectionVideoService:
    """Dependency provider for DetectionVideoService."""
    logger.debug("Initializing DetectionVideoService instance (or returning cached).")
    service = detection_video_service
    service.attach_playback_interfaces(
        status_store=playback_status_store,
        runtime_registry=playback_runtime_registry,
    )
    return service
