"""
Module for managing and providing application dependencies.
Leverages FastAPI's dependency injection system and app.state for preloaded components.
"""
from functools import lru_cache
from typing import Optional
import logging
import uuid

from fastapi import Depends, Request, HTTPException, status
import torch

from app.core.config import settings
# Keys are still useful for conceptual grouping or if we switched to setattr/getattr
# from app.core.event_handlers import (
#     DETECTOR_KEY, TRACKER_FACTORY_KEY, HOMOGRAPHY_SERVICE_KEY, COMPUTE_DEVICE_KEY
# )
from app.utils.asset_downloader import AssetDownloader
from app.services.video_data_manager_service import VideoDataManagerService
# Legacy pipeline orchestrator removed with Re-ID deprecation.
class PipelineOrchestratorService:
    def __init__(self, *args, **kwargs):
        pass
    async def initialize_task(self, *args, **kwargs):
        raise NotImplementedError("Pipeline orchestrator has been removed.")
    async def run_processing_pipeline(self, *args, **kwargs):
        raise NotImplementedError("Pipeline orchestrator has been removed.")
    async def get_task_status(self, *args, **kwargs):
        return None
from app.services.notification_service import NotificationService
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.multi_camera_frame_processor import MultiCameraFrameProcessor
from app.services.homography_service import HomographyService
from app.services.playback_status_store import PlaybackStatusStore
from app.services.task_runtime_registry import TaskRuntimeRegistry
from app.services.playback_control_service import PlaybackControlService

from app.models.base_models import AbstractDetector

from app.api.websockets import binary_websocket_manager as websocket_manager

logger = logging.getLogger(__name__)


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


@lru_cache()
def get_playback_status_store() -> PlaybackStatusStore:
    """Provide a shared in-memory store for playback state."""

    return PlaybackStatusStore()


@lru_cache()
def get_task_runtime_registry() -> TaskRuntimeRegistry:
    """Provide a global registry coordinating playback runtime state."""

    return TaskRuntimeRegistry()


def get_playback_control_service(
    status_store: PlaybackStatusStore = Depends(get_playback_status_store),
    runtime_registry: TaskRuntimeRegistry = Depends(get_task_runtime_registry),
) -> PlaybackControlService:
    """Dependency provider for playback control orchestration."""

    return PlaybackControlService(
        status_store=status_store,
        runtime_registry=runtime_registry,
        timeout_seconds=settings.PLAYBACK_CONTROL_TIMEOUT_SECONDS,
    )


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

@lru_cache()
def get_pipeline_orchestrator(
    video_data_manager: VideoDataManagerService = Depends(get_video_data_manager_service),
    multi_camera_processor: MultiCameraFrameProcessor = Depends(get_multi_camera_frame_processor),
    tracker_factory: CameraTrackerFactory = Depends(get_camera_tracker_factory),
    notification_service: NotificationService = Depends(get_notification_service)
) -> PipelineOrchestratorService:
    """Deprecated: Pipeline orchestrator removed; this returns a stub."""
    logger.debug("PipelineOrchestratorService requested but has been removed.")
    return PipelineOrchestratorService()
