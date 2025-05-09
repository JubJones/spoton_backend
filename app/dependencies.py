"""
Module for managing and providing application dependencies.

This module centralizes the creation and provision of various services,
utilities, and machine learning models used throughout the application.
It leverages FastAPI's dependency injection system and `functools.lru_cache`
to efficiently manage and reuse instances of these components.

Key responsibilities include:
- Providing access to configuration settings.
- Managing asset downloading.
- Initializing and providing machine learning models (detectors, trackers).
- Providing service instances (e.g., VideoDataManagerService, PipelineOrchestratorService).
- Determining the appropriate compute device (CPU/GPU).
"""

from functools import lru_cache
from typing import Optional
import logging

from fastapi import Depends
import torch

from app.core.config import settings
from app.utils.asset_downloader import AssetDownloader
from app.utils.device_utils import get_selected_device
from app.services.video_data_manager_service import VideoDataManagerService
from app.services.pipeline_orchestrator import PipelineOrchestratorService
from app.services.notification_service import NotificationService
# Import models and base classes
from app.models.base_models import AbstractDetector, AbstractTracker
from app.models.detectors import FasterRCNNDetector
from app.models.trackers import BotSortTracker
from app.api.websockets import manager as websocket_manager

logger = logging.getLogger(__name__)

# Cached device selection
@lru_cache()
def get_compute_device() -> torch.device:
    """Gets the compute device based on environment settings."""
    return get_selected_device(requested_device="auto")

@lru_cache()
def get_asset_downloader() -> AssetDownloader:
    """Dependency provider for AssetDownloader utility."""
    return AssetDownloader(
        dagshub_repo_owner=settings.DAGSHUB_REPO_OWNER,
        dagshub_repo_name=settings.DAGSHUB_REPO_NAME
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
    Dependency provider for the object detector.
    Loads the model upon first call.
    """
    logger.info(f"Initializing detector (Type: {settings.DETECTOR_TYPE})...")
    # Currently only supports FasterRCNN, can be extended with a factory pattern
    if settings.DETECTOR_TYPE.lower() == "fasterrcnn":
        detector = FasterRCNNDetector()
    else:
        logger.error(f"Unsupported DETECTOR_TYPE: {settings.DETECTOR_TYPE}. Defaulting to FasterRCNN.")
        detector = FasterRCNNDetector() # Fallback

    # Note: load_model is async, but dependencies are typically synchronous.
    # Model loading should ideally happen during application startup (lifespan)
    # or handled carefully here (e.g., using an async setup function or risking blocking).
    # For simplicity in this example, we call it here, but beware of blocking I/O.
    # A better approach uses app lifespan events.
    # await detector.load_model() # This won't work directly in sync function
    # --- Alternative: Load in lifespan, store in app.state ---
    # Or, make this dependency async if FastAPI supports async dependencies that load models.
    # For now, we assume load_model is called within the service that uses it, or in lifespan.
    logger.info("Detector instance created (model loading deferred to usage/lifespan).")
    return detector

@lru_cache()
def get_tracker(device: torch.device = Depends(get_compute_device)) -> AbstractTracker:
    """
    Dependency provider for the tracker.
    Loads the model upon first call.
    """
    logger.info(f"Initializing tracker (Type: {settings.TRACKER_TYPE})...")
    if settings.TRACKER_TYPE.lower() == "botsort":
        tracker = BotSortTracker()
    else:
        logger.error(f"Unsupported TRACKER_TYPE: {settings.TRACKER_TYPE}. Defaulting to BotSort.")
        tracker = BotSortTracker()

    # Similar loading concern as the detector. Deferred to usage/lifespan ideally.
    logger.info("Tracker instance created (model loading deferred to usage/lifespan).")
    return tracker


@lru_cache()
def get_notification_service() -> NotificationService:
    """Dependency provider for NotificationService."""
    # Inject the global websocket manager
    return NotificationService(manager=websocket_manager)


@lru_cache()
def get_pipeline_orchestrator(
    video_data_manager: VideoDataManagerService = Depends(get_video_data_manager_service),
    detector: AbstractDetector = Depends(get_detector),
    tracker: AbstractTracker = Depends(get_tracker),
    notification_service: NotificationService = Depends(get_notification_service),
    device: torch.device = Depends(get_compute_device)
) -> PipelineOrchestratorService:
    """
    Dependency provider for PipelineOrchestratorService.
    Injects necessary service and model dependencies.
    """
    return PipelineOrchestratorService(
        video_data_manager=video_data_manager,
        detector=detector,
        tracker=tracker,
        notification_service=notification_service,
        device=device
    )