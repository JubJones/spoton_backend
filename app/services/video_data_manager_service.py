import os
import asyncio
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple

from app.core.config import settings, VideoSetEnvironmentConfig
from app.utils.asset_downloader import AssetDownloader # MODIFIED IMPORT
from app.utils.video_processing import extract_frames_from_video_to_disk

logger = logging.getLogger(__name__)

class VideoDataManagerService:
    """
    Manages downloading videos and extracting frames for processing.
    Focuses on preparing initial data for a given environment.
    """
    def __init__(self, asset_downloader: AssetDownloader): # Type hint uses the new class name
        self.asset_downloader = asset_downloader
        self.video_sets_config: List[VideoSetEnvironmentConfig] = settings.VIDEO_SETS
        self.local_video_dir: str = settings.LOCAL_VIDEO_DOWNLOAD_DIR
        self.local_frame_dir: str = settings.LOCAL_FRAME_EXTRACTION_DIR

    async def prepare_initial_frames_for_environment(
        self, environment_id: str
    ) -> Dict[str, List[str]]:
        """
        Downloads the first sub-video for each camera in the specified environment
        and extracts frames from them.

        Args:
            environment_id: The ID of the environment (e.g., "campus", "factory").

        Returns:
            A dictionary mapping camera_id to a list of paths of its extracted frames.
            Example: {"c01": ["/path/to/frames/c01/frame_000000.jpg", ...], ...}
        """
        logger.info(f"Preparing initial frames for environment: {environment_id}")
        cam_frames_map: Dict[str, List[str]] = {}
        
        cameras_in_env = [
            vs_config for vs_config in self.video_sets_config if vs_config.env_id == environment_id
        ]

        if not cameras_in_env:
            logger.warning(f"No camera configurations found for environment: {environment_id}")
            return cam_frames_map

        for cam_config in cameras_in_env:
            first_sub_video_idx = 1 
            if cam_config.num_sub_videos < first_sub_video_idx:
                logger.warning(f"Camera {cam_config.cam_id} in env {environment_id} configured with {cam_config.num_sub_videos} sub-videos, cannot get sub-video {first_sub_video_idx}.")
                continue

            video_filename = cam_config.sub_video_filename_pattern.format(idx=first_sub_video_idx)
            remote_video_key = f"{cam_config.remote_base_key.strip('/')}/{video_filename}"
            
            local_cam_video_download_dir = Path(self.local_video_dir) / cam_config.env_id / cam_config.cam_id
            local_video_path = local_cam_video_download_dir / video_filename
            
            local_cam_frame_extraction_dir = Path(self.local_frame_dir) / cam_config.env_id / cam_config.cam_id / Path(video_filename).stem
            
            local_cam_video_download_dir.mkdir(parents=True, exist_ok=True)
            local_cam_frame_extraction_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Processing camera {cam_config.cam_id} in {environment_id}: downloading {remote_video_key}")

            if os.path.exists(local_video_path):
                 logger.info(f"Video {local_video_path} already exists locally. Skipping download.")
                 download_successful = True
            else:
                download_successful = await self.asset_downloader.download_file_from_dagshub(
                    remote_s3_key=remote_video_key,
                    local_destination_path=str(local_video_path)
                )

            if download_successful:
                logger.info(f"Extracting frames for {local_video_path} at 23 FPS.")
                extracted_paths, status_msg = await asyncio.to_thread(
                    extract_frames_from_video_to_disk,
                    video_path=str(local_video_path),
                    output_folder=str(local_cam_frame_extraction_dir),
                    frame_filename_prefix=f"{Path(video_filename).stem}_frame_",
                    target_fps=23,
                    jpeg_quality=95
                )
                if extracted_paths:
                    cam_frames_map[cam_config.cam_id] = extracted_paths
                else:
                    logger.warning(f"No frames extracted for {cam_config.cam_id} from {video_filename}. Status: {status_msg}")
            else:
                logger.error(f"Failed to download video for camera {cam_config.cam_id}: {remote_video_key}")
        
        logger.info(f"Finished preparing initial frames for environment: {environment_id}. Cameras processed: {len(cam_frames_map)}.")
        return cam_frames_map