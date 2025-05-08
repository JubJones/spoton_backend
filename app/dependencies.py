from functools import lru_cache
from fastapi import Depends
from typing import Optional

from app.core.config import settings
from app.utils.asset_downloader import AssetDownloader # MODIFIED IMPORT
from app.services.video_data_manager_service import VideoDataManagerService
from app.services.pipeline_orchestrator import PipelineOrchestratorService
# from app.models.detectors import FasterRCNNDetector
# from app.models.base_models import AbstractDetector

@lru_cache()
def get_asset_downloader() -> AssetDownloader: # Renamed function for clarity
    """
    Dependency provider for AssetDownloader utility.
    Uses DagsHub configuration from settings.
    """
    return AssetDownloader( # Class name updated
        dagshub_repo_owner=settings.DAGSHUB_REPO_OWNER,
        dagshub_repo_name=settings.DAGSHUB_REPO_NAME
        # s3_endpoint_url=settings.S3_ENDPOINT_URL # Pass if set and needed by AssetDownloader
    )

@lru_cache()
def get_video_data_manager_service(
    asset_downloader: AssetDownloader = Depends(get_asset_downloader) # Depends on renamed function
) -> VideoDataManagerService:
    """
    Dependency provider for VideoDataManagerService.
    Injects an AssetDownloader instance.
    """
    return VideoDataManagerService(asset_downloader=asset_downloader)

@lru_cache()
def get_pipeline_orchestrator(
    video_data_manager: VideoDataManagerService = Depends(get_video_data_manager_service)
    # detector: AbstractDetector = Depends(get_person_detector)
) -> PipelineOrchestratorService:
    """
    Dependency provider for PipelineOrchestratorService.
    Injects necessary service dependencies.
    """
    return PipelineOrchestratorService(
        video_data_manager=video_data_manager
        # detector=detector
    )

# Example for an AI model dependency (Strategy Pattern context)
# @lru_cache()
# async def get_person_detector() -> AbstractDetector:
#     detector = FasterRCNNDetector()
#     await detector.load_model()
#     return detector