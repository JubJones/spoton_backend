"""
Module for managing and providing application dependencies.
Leverages FastAPI's dependency injection system and `functools.lru_cache`.
"""

from functools import lru_cache
from typing import Optional
import logging
import uuid # Added for task_id typing in factory if needed

from fastapi import Depends, Request # Added Request for potential task-scoped dependencies
import torch

from app.core.config import settings
from app.utils.asset_downloader import AssetDownloader
from app.utils.device_utils import get_selected_device
from app.services.video_data_manager_service import VideoDataManagerService
from app.services.pipeline_orchestrator import PipelineOrchestratorService
from app.services.notification_service import NotificationService
from app.services.camera_tracker_factory import CameraTrackerFactory # New
from app.services.multi_camera_frame_processor import MultiCameraFrameProcessor # New
# ReIDStateManager is now created per task within PipelineOrchestratorService
# from app.services.reid_components import ReIDStateManager

from app.models.base_models import AbstractDetector, AbstractTracker # Keep AbstractTracker for type hint
from app.models.detectors import FasterRCNNDetector
# BotSortTracker is now managed by CameraTrackerFactory, not directly injected as a singleton.

from app.api.websockets import manager as websocket_manager

logger = logging.getLogger(__name__)

# Cached device selection
@lru_cache()
def get_compute_device() -> torch.device:
    """Gets the compute device based on environment settings."""
    return get_selected_device(requested_device="auto")

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

# --- Model Dependencies ---

@lru_cache()
def get_detector(device: torch.device = Depends(get_compute_device)) -> AbstractDetector:
    """
    Dependency provider for the object detector (singleton).
    The detector model is loaded upon first call by its load_model method.
    """
    logger.info(f"Initializing detector (Type: {settings.DETECTOR_TYPE})...")
    if settings.DETECTOR_TYPE.lower() == "fasterrcnn":
        detector = FasterRCNNDetector() # Device assignment happens in its load_model
    else:
        logger.error(f"Unsupported DETECTOR_TYPE: {settings.DETECTOR_TYPE}. Defaulting to FasterRCNN.")
        detector = FasterRCNNDetector()
    # Actual model loading from disk/network is deferred to detector.load_model(),
    # which will be called by the service that uses it (e.g., PipelineOrchestrator/MultiCameraFrameProcessor)
    # or via an application startup event if pre-loading is desired.
    # For now, MultiCameraFrameProcessor will ensure models are loaded.
    logger.info("Detector instance created (model loading deferred to first use or explicit call).")
    return detector

# --- Service Dependencies ---

@lru_cache()
def get_notification_service() -> NotificationService:
    """Dependency provider for NotificationService (singleton)."""
    return NotificationService(manager=websocket_manager)

@lru_cache()
def get_camera_tracker_factory(
    device: torch.device = Depends(get_compute_device)
) -> CameraTrackerFactory:
    """
    Dependency provider for CameraTrackerFactory (singleton).
    The factory itself is a singleton; it manages per-camera tracker instances.
    """
    logger.info("Initializing CameraTrackerFactory...")
    return CameraTrackerFactory(device=device)


# Models (detector, tracker) should be loaded before MultiCameraFrameProcessor uses them.
# This can be handled by PipelineOrchestratorService ensuring models are ready.
# Or, MultiCameraFrameProcessor can take AbstractDetector and CameraTrackerFactory
# and call their load methods if not already loaded.
# For simplicity, let's assume PipelineOrchestrator or an app startup hook ensures loading.
# The dependencies here provide the *instances* or *factories*.

@lru_cache()
def get_multi_camera_frame_processor(
    detector: AbstractDetector = Depends(get_detector),
    tracker_factory: CameraTrackerFactory = Depends(get_camera_tracker_factory),
    notification_service: NotificationService = Depends(get_notification_service), # May not be needed here if orchestrator handles it
    device: torch.device = Depends(get_compute_device)
) -> MultiCameraFrameProcessor:
    """Dependency provider for MultiCameraFrameProcessor (singleton)."""
    logger.info("Initializing MultiCameraFrameProcessor...")
    # MultiCameraFrameProcessor will need the ReIDStateManager, but it's per-task.
    # So, ReIDStateManager cannot be a singleton dependency here.
    # PipelineOrchestrator will create/retrieve it per task and pass it.
    return MultiCameraFrameProcessor(
        detector=detector,
        tracker_factory=tracker_factory,
        notification_service=notification_service, # Orchestrator will use this
        device=device
    )

@lru_cache()
def get_pipeline_orchestrator(
    video_data_manager: VideoDataManagerService = Depends(get_video_data_manager_service),
    multi_camera_processor: MultiCameraFrameProcessor = Depends(get_multi_camera_frame_processor),
    tracker_factory: CameraTrackerFactory = Depends(get_camera_tracker_factory),
    notification_service: NotificationService = Depends(get_notification_service)
    # Detector and device are implicitly handled by multi_camera_processor now
) -> PipelineOrchestratorService:
    """Dependency provider for PipelineOrchestratorService (singleton)."""
    logger.info("Initializing PipelineOrchestratorService...")
    return PipelineOrchestratorService(
        video_data_manager=video_data_manager,
        multi_camera_processor=multi_camera_processor,
        tracker_factory=tracker_factory,
        notification_service=notification_service
    )

# --- Ensuring models are loaded ---
# This is a conceptual placement. In a real app, this might be part of `app.main.lifespan`
# or called by the orchestrator before starting any processing.
async def ensure_models_are_loaded(
    detector: AbstractDetector = Depends(get_detector),
    # Tracker models are loaded by CameraTrackerFactory when a tracker is first requested
):
    """
    A dependency that ensures core models (like the detector) are loaded.
    Tracker models are loaded on-demand by their factory.
    """
    if not hasattr(detector, '_model_loaded_flag') or not detector._model_loaded_flag: #Requires detector to have this flag
        logger.info("Ensuring detector model is loaded via dependency...")
        await detector.load_model()
        # setattr(detector, '_model_loaded_flag', True) # Example flag
    # else: logger.debug("Detector model presumed to be already loaded.")

# You might inject `Depends(ensure_models_are_loaded)` into endpoints or services that
# trigger processing, or call it explicitly in `on_startup`.
# For this refactor, PipelineOrchestrator can call `detector.load_model()`
# and `tracker_factory.get_tracker()` will handle tracker model loading.
# The `_ensure_models_loaded` method in the old PipelineOrchestratorService can be adapted.
# In the new structure, `MultiCameraFrameProcessor` could call `await self.detector.load_model()`
# in its `__init__` or a dedicated `async def initialize_models(self)` method.
# For now, let's assume the `PipelineOrchestratorService` will manage this,
# perhaps by calling a load method on `MultiCameraFrameProcessor` or directly on dependencies.
# The current `FasterRCNNDetector.load_model` already logs and handles `self.model is not None`.
# And `CameraTrackerFactory.get_tracker` also handles loading for new tracker instances.
# So, explicit call from orchestrator should be fine.