"""
Module for managing video data, including downloading and frame extraction.
"""
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
    Checks for existing videos and frames to avoid redundant operations.
    """
    def __init__(self, asset_downloader: AssetDownloader):
        """
        Initializes the VideoDataManagerService.

        Args:
            asset_downloader: An instance of AssetDownloader for fetching remote files.
        """
        self.asset_downloader = asset_downloader
        self.video_sets_config: List[VideoSetEnvironmentConfig] = settings.VIDEO_SETS
        self.local_video_dir: str = settings.LOCAL_VIDEO_DOWNLOAD_DIR
        self.local_frame_dir: str = settings.LOCAL_FRAME_EXTRACTION_DIR
        logger.info("VideoDataManagerService initialized.")

    async def prepare_initial_frames_for_environment(
        self, environment_id: str, target_fps: int, jpeg_quality: int
    ) -> Dict[str, List[str]]:
        """
        Prepares initial frames for each camera in the specified environment.

        This involves:
        1. Identifying the first sub-video for each camera in the environment.
        2. Checking if the sub-video already exists locally. If not, download it.
        3. Checking if frames for this sub-video are already extracted locally.
           If yes, use them (cache hit).
        4. If frames are not cached, extract them from the local video.

        Args:
            environment_id: The ID of the environment (e.g., "campus", "factory").
            target_fps: Target FPS for frame extraction.
            jpeg_quality: Quality for saving JPEG frames.

        Returns:
            A dictionary mapping camera_id to a list of full paths of its extracted frames.
            Example: {"c01": ["/path/to/frames/c01/frame_000000.jpg", ...], ...}
            Returns an empty list for a camera if its video download or frame processing fails.
        """
        logger.info(f"Preparing initial frames for environment: {environment_id}")
        cam_frames_map: Dict[str, List[str]] = {}

        cameras_in_env = [
            vs_config for vs_config in self.video_sets_config if vs_config.env_id == environment_id
        ]

        if not cameras_in_env:
            logger.warning(f"No camera configurations found for environment: {environment_id}. Cannot prepare data.")
            return cam_frames_map

        download_tasks = []
        # Stores metadata for each video, used to manage download and extraction logic
        video_processing_metadata_list: List[Dict[str, Any]] = [] 

        for cam_config in cameras_in_env:
            first_sub_video_idx = 1 # Process only the first sub-video as per current design
            if cam_config.num_sub_videos < first_sub_video_idx:
                logger.warning(
                    f"Camera {cam_config.cam_id} in env {environment_id} configured with "
                    f"{cam_config.num_sub_videos} sub-videos, cannot get sub-video {first_sub_video_idx}."
                )
                cam_frames_map[cam_config.cam_id] = [] # Mark as failed/skipped early
                continue

            video_filename = cam_config.sub_video_filename_pattern.format(idx=first_sub_video_idx)
            remote_video_key = f"{cam_config.remote_base_key.strip('/')}/{video_filename}"

            local_cam_video_download_dir = Path(self.local_video_dir) / environment_id / cam_config.cam_id
            local_video_path = local_cam_video_download_dir / video_filename
            # Frame directory is specific to this video file's stem
            local_cam_frame_extraction_dir = Path(self.local_frame_dir) / environment_id / cam_config.cam_id / Path(video_filename).stem

            local_cam_video_download_dir.mkdir(parents=True, exist_ok=True)
            # Note: local_cam_frame_extraction_dir is created later if extraction occurs or checked if caching.

            current_video_info = {
                "cam_id": cam_config.cam_id,
                "local_video_path": str(local_video_path),
                "local_frame_dir": str(local_cam_frame_extraction_dir),
                "frame_prefix": f"{Path(video_filename).stem}_frame_" # e.g., "sub_video_01_frame_"
            }
            video_processing_metadata_list.append(current_video_info)

            if not local_video_path.exists():
                logger.info(f"[{environment_id}/{cam_config.cam_id}] Queuing download: {remote_video_key} to {local_video_path}")
                download_tasks.append(
                    self.asset_downloader.download_file_from_dagshub(
                        remote_s3_key=remote_video_key,
                        local_destination_path=str(local_video_path)
                    )
                )
            else:
                logger.info(f"[{environment_id}/{cam_config.cam_id}] Video already exists locally: {local_video_path}")
                download_tasks.append(asyncio.sleep(0, result=True)) # Simulate non-blocking success

        # Run downloads concurrently
        download_results = []
        if download_tasks:
            download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
        else: 
            logger.info(f"[{environment_id}] No video downloads were queued (e.g., all videos skipped or no cameras).")


        # --- Frame Processing Phase (Cache Check & New Extraction) ---
        actual_extraction_coroutines = [] 
        # Stores how to get frames for each video that was successfully available (downloaded or existed)
        cam_frame_sourcing_references = [] 

        successful_downloads_count = 0
        failed_downloads_count = 0
        
        for i, video_meta in enumerate(video_processing_metadata_list):
            cam_id = video_meta['cam_id']
            
            if i >= len(download_results): 
                 logger.warning(f"[{environment_id}/{cam_id}] Mismatch: No download result for video metadata. Skipping.")
                 cam_frames_map[cam_id] = [] # Ensure it's marked
                 continue

            download_result = download_results[i]

            if isinstance(download_result, Exception) or download_result is not True:
                logger.error(f"[{environment_id}/{cam_id}] Download failed for video {video_meta['local_video_path']}: {download_result}")
                failed_downloads_count += 1
                cam_frames_map[cam_id] = [] # Record failure
                continue
            
            # Video is successfully downloaded or already existed
            successful_downloads_count +=1
            logger.info(f"[{environment_id}/{cam_id}] Video ready: {video_meta['local_video_path']}. Checking for existing frames...")

            frame_dir_path = Path(video_meta["local_frame_dir"])
            frame_prefix = video_meta["frame_prefix"]
            existing_frames = []

            if frame_dir_path.is_dir():
                # Glob for .jpg files matching the naming convention and sort them
                existing_frames = sorted([str(f) for f in frame_dir_path.glob(f"{frame_prefix}*.jpg")])
            
            if existing_frames:
                logger.info(f"[{environment_id}/{cam_id}] Found {len(existing_frames)} existing frames in {frame_dir_path}. Using cached frames.")
                cam_frame_sourcing_references.append({
                    "cam_id": cam_id,
                    "status": "cached", # Indicates frames were found in cache
                    "data": (existing_frames, f"Used {len(existing_frames)} cached frames from {frame_dir_path}.")
                })
            else:
                # Ensure frame directory exists before queueing extraction
                frame_dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"[{environment_id}/{cam_id}] No cached frames found in {frame_dir_path}. Queueing new frame extraction.")
                coro = asyncio.to_thread(
                    extract_frames_from_video_to_disk,
                    video_path=video_meta["local_video_path"],
                    output_folder=str(frame_dir_path),
                    frame_filename_prefix=frame_prefix,
                    target_fps=target_fps,
                    jpeg_quality=jpeg_quality
                )
                actual_extraction_coroutines.append(coro)
                cam_frame_sourcing_references.append({
                    "cam_id": cam_id,
                    "status": "needs_extraction", # Indicates frames need to be extracted
                    "future_index": len(actual_extraction_coroutines) - 1 # Index in actual_extraction_coroutines
                })
        
        logger.info(
            f"Download summary for env '{environment_id}': {successful_downloads_count} videos ready for frame processing, "
            f"{failed_downloads_count} failed/skipped video downloads."
        )

        # Run new extractions if any are needed
        newly_extracted_frame_results = []
        if actual_extraction_coroutines:
            logger.info(f"[{environment_id}] Starting {len(actual_extraction_coroutines)} new frame extraction tasks.")
            newly_extracted_frame_results = await asyncio.gather(*actual_extraction_coroutines, return_exceptions=True)
            logger.info(f"[{environment_id}] Finished {len(actual_extraction_coroutines)} new frame extraction tasks.")
        else:
            logger.info(f"[{environment_id}] No new frame extractions were needed (all cached or downloads failed).")

        # Populate cam_frames_map using the references
        for ref in cam_frame_sourcing_references:
            cam_id = ref["cam_id"]
            if ref["status"] == "cached":
                frames, msg = ref["data"]
                cam_frames_map[cam_id] = frames
                # Message for cached frames already logged when found
            elif ref["status"] == "needs_extraction":
                future_idx = ref["future_index"]
                if future_idx < len(newly_extracted_frame_results):
                    result = newly_extracted_frame_results[future_idx]
                    if isinstance(result, Exception):
                        logger.error(f"[{environment_id}/{cam_id}] Frame extraction task failed: {result}", exc_info=True)
                        cam_frames_map[cam_id] = []
                    elif isinstance(result, tuple) and len(result) == 2:
                        frames, msg = result
                        cam_frames_map[cam_id] = frames
                        logger.info(f"[{environment_id}/{cam_id}] Frame extraction successful. {msg}")
                        if not frames: # Extraction succeeded but returned no paths
                            logger.warning(f"[{environment_id}/{cam_id}] Extraction reported success but returned no frame paths.")
                    else: # Should not happen if extract_frames_from_video_to_disk is consistent
                        logger.error(f"[{environment_id}/{cam_id}] Unexpected result type from extraction task: {type(result)}")
                        cam_frames_map[cam_id] = []
                else: # Should not happen if indexing logic is correct
                    logger.error(f"[{environment_id}/{cam_id}] Logic error: Mismatch in extraction results index for future_index {future_idx}.")
                    cam_frames_map[cam_id] = []
            
            # Safety net: ensure cam_id from references is in map, even if with empty list
            if cam_id not in cam_frames_map: 
                logger.warning(f"[{environment_id}/{cam_id}] Cam ID from references was not added to cam_frames_map. Setting to empty list.")
                cam_frames_map[cam_id] = []

        # Ensure all originally configured cameras for the env (that weren't skipped early due to config) have an entry
        for cam_config in cameras_in_env:
            if cam_config.cam_id not in cam_frames_map:
                # This case implies the camera was skipped before even video_processing_metadata_list was populated
                # or some other very early exit for this cam_id. It should already have [] from the initial loop.
                logger.debug(
                    f"[{environment_id}/{cam_config.cam_id}] Ensuring entry in map for camera. "
                    "It was likely skipped early or already handled."
                )
                # If it's missing (shouldn't be if initial loop is correct), add empty list.
                if cam_config.cam_id not in cam_frames_map : cam_frames_map[cam_config.cam_id] = []


        logger.info(
            f"Finished preparing initial frames for environment: {environment_id}. "
            f"Frame data map (cam_id: num_frames): { {k: f'{len(v)} frames' for k, v in cam_frames_map.items()} }"
        )
        return cam_frames_map