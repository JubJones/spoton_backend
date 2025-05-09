import os
import asyncio
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple

from app.core.config import settings, VideoSetEnvironmentConfig
from app.utils.asset_downloader import AssetDownloader
from app.utils.video_processing import extract_frames_from_video_to_disk

logger = logging.getLogger(__name__)

class VideoDataManagerService:
    """
    Manages downloading videos and extracting frames for processing.
    Focuses on preparing initial data for a given environment.
    """
    def __init__(self, asset_downloader: AssetDownloader):
        self.asset_downloader = asset_downloader
        self.video_sets_config: List[VideoSetEnvironmentConfig] = settings.VIDEO_SETS
        self.local_video_dir: str = settings.LOCAL_VIDEO_DOWNLOAD_DIR
        self.local_frame_dir: str = settings.LOCAL_FRAME_EXTRACTION_DIR
        logger.info("VideoDataManagerService initialized.")

    async def prepare_initial_frames_for_environment(
        self, environment_id: str, target_fps: int, jpeg_quality: int
    ) -> Dict[str, List[str]]:
        """
        Downloads the first sub-video for each camera in the specified environment
        and extracts frames from them.

        Args:
            environment_id: The ID of the environment (e.g., "campus", "factory").
            target_fps: Target FPS for frame extraction.
            jpeg_quality: Quality for saving JPEG frames.

        Returns:
            A dictionary mapping camera_id to a list of full paths of its extracted frames.
            Example: {"c01": ["/path/to/frames/c01/frame_000000.jpg", ...], ...}
            Returns an empty dictionary if the environment or cameras are not found.
        """
        logger.info(f"Preparing initial frames for environment: {environment_id}")
        cam_frames_map: Dict[str, List[str]] = {}

        # Find camera configurations for the requested environment
        cameras_in_env = [
            vs_config for vs_config in self.video_sets_config if vs_config.env_id == environment_id
        ]

        if not cameras_in_env:
            logger.warning(f"No camera configurations found for environment: {environment_id}. Cannot prepare data.")
            return cam_frames_map # Return empty map

        download_tasks = []
        extraction_infos = [] # Store info needed for extraction after download

        # --- Download Phase ---
        for cam_config in cameras_in_env:
            # Process only the first sub-video (index 1)
            first_sub_video_idx = 1
            if cam_config.num_sub_videos < first_sub_video_idx:
                logger.warning(f"Camera {cam_config.cam_id} in env {environment_id} configured with "
                               f"{cam_config.num_sub_videos} sub-videos, cannot get sub-video {first_sub_video_idx}.")
                continue

            video_filename = cam_config.sub_video_filename_pattern.format(idx=first_sub_video_idx)
            remote_video_key = f"{cam_config.remote_base_key.strip('/')}/{video_filename}"

            # Define local paths
            local_cam_video_download_dir = Path(self.local_video_dir) / environment_id / cam_config.cam_id
            local_video_path = local_cam_video_download_dir / video_filename
            local_cam_frame_extraction_dir = Path(self.local_frame_dir) / environment_id / cam_config.cam_id / Path(video_filename).stem

            # Ensure directories exist
            local_cam_video_download_dir.mkdir(parents=True, exist_ok=True)
            local_cam_frame_extraction_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"[{environment_id}/{cam_config.cam_id}] Queuing download: {remote_video_key}")

            # Add download task if file doesn't exist locally
            if not local_video_path.exists():
                 download_tasks.append(
                     self.asset_downloader.download_file_from_dagshub(
                         remote_s3_key=remote_video_key,
                         local_destination_path=str(local_video_path)
                     )
                 )
            else:
                 logger.info(f"[{environment_id}/{cam_config.cam_id}] Video already exists locally: {local_video_path}")
                 # Simulate successful download result for consistency if file exists
                 download_tasks.append(asyncio.sleep(0, result=True)) # Non-blocking success

            # Store info needed for extraction regardless of download status
            extraction_infos.append({
                "cam_id": cam_config.cam_id,
                "local_video_path": str(local_video_path),
                "local_frame_dir": str(local_cam_frame_extraction_dir),
                "frame_prefix": f"{Path(video_filename).stem}_frame_"
            })

        # Run downloads concurrently
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

        # --- Extraction Phase ---
        extraction_tasks = []
        successful_downloads = 0
        failed_downloads = 0

        for i, result in enumerate(download_results):
            info = extraction_infos[i]
            if isinstance(result, Exception):
                logger.error(f"[{environment_id}/{info['cam_id']}] Download failed: {result}")
                failed_downloads += 1
            elif result is True: # Check for explicit True success
                successful_downloads += 1
                logger.info(f"[{environment_id}/{info['cam_id']}] Download successful or file existed. Queueing frame extraction from {info['local_video_path']}")
                # Run extraction in thread pool as it's CPU/Disk bound
                extraction_tasks.append(
                    asyncio.to_thread(
                        extract_frames_from_video_to_disk,
                        video_path=info["local_video_path"],
                        output_folder=info["local_frame_dir"],
                        frame_filename_prefix=info["frame_prefix"],
                        target_fps=target_fps,
                        jpeg_quality=jpeg_quality
                    )
                )
            else: # Should not happen if we simulate True for existing files, but handle unexpected results
                 logger.warning(f"[{environment_id}/{info['cam_id']}] Download result was neither True nor Exception: {result}. Skipping extraction.")
                 failed_downloads += 1

        logger.info(f"Download results: {successful_downloads} succeeded, {failed_downloads} failed.")

        if not extraction_tasks:
            logger.warning(f"No videos available for frame extraction in environment {environment_id}.")
            return cam_frames_map # Return empty if no tasks

        # Run extractions concurrently
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # Process extraction results
        for i, result in enumerate(extraction_results):
            # Find the corresponding info based on the order of successful downloads
            # This assumes the order of extraction_tasks matches the order of successfully downloaded infos
            # It's safer to map results back using a unique identifier if order isn't guaranteed
            # For now, assume order is maintained.
            original_info_index = -1
            success_counter = 0
            for j, dl_res in enumerate(download_results):
                if isinstance(dl_res, Exception) or dl_res is not True: continue # Skip failed downloads
                if success_counter == i:
                    original_info_index = j
                    break
                success_counter += 1

            if original_info_index == -1:
                 logger.error(f"Could not map extraction result back to original video info (Index {i}). Skipping.")
                 continue

            info = extraction_infos[original_info_index]
            cam_id = info['cam_id']

            if isinstance(result, Exception):
                logger.error(f"[{environment_id}/{cam_id}] Frame extraction failed: {result}", exc_info=True)
            elif isinstance(result, tuple) and len(result) == 2:
                extracted_paths, status_msg = result
                if extracted_paths:
                    cam_frames_map[cam_id] = extracted_paths
                    logger.info(f"[{environment_id}/{cam_id}] Frame extraction successful. {status_msg}")
                else:
                    logger.warning(f"[{environment_id}/{cam_id}] No frames extracted. {status_msg}")
            else:
                 logger.error(f"[{environment_id}/{cam_id}] Unexpected result type from frame extraction: {type(result)}")

        logger.info(f"Finished preparing initial frames for environment: {environment_id}. "
                    f"Frames extracted for {len(cam_frames_map)} cameras.")
        return cam_frames_map