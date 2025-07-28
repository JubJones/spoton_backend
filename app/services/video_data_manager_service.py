"""
Module for managing video data, including downloading and frame extraction
in batches suitable for synchronized multi-camera processing.
"""
import os
import asyncio
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple, AsyncGenerator
import uuid
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import math # For math.ceil

import cv2 # OpenCV for video processing

from app.core.config import settings, VideoSetEnvironmentConfig
from app.utils.asset_downloader import AssetDownloader
from app.shared.types import CameraID, FrameBatch, FrameData

logger = logging.getLogger(__name__)

class BatchedFrameProvider:
    """
    Provides frames from multiple video files in synchronized batches.
    Each batch contains one frame from each video, corresponding to the same
    temporal index.
    """
    def __init__(
        self,
        task_id: uuid.UUID,
        video_paths_map: Dict[CameraID, Path], # CameraID -> Path to local video file
        target_fps: int,
        jpeg_quality: int, # Not used for frame providing, but kept for consistency if saving
        loop_videos: bool = False # Whether to loop videos if they end
    ):
        self.task_id = task_id
        self.video_paths_map = video_paths_map
        self.target_fps = target_fps
        self.loop_videos = loop_videos

        self.video_captures: Dict[CameraID, cv2.VideoCapture] = {}
        self.video_actual_fps: Dict[CameraID, float] = {}
        self.frame_skip_intervals: Dict[CameraID, int] = {}
        self.frame_counters_read: Dict[CameraID, int] = defaultdict(int)
        self.total_frames_in_video: Dict[CameraID, int] = {} # Original number of frames
        self.num_processed_frames_per_video: Dict[CameraID, int] = {} # Expected frames after skipping
        self.current_frame_index_processed_per_cam: Dict[CameraID, int] = defaultdict(int)

        self._is_open = False
        self._open_videos()

    def _open_videos(self):
        """Opens all video files using OpenCV VideoCapture."""
        for cam_id, video_path in self.video_paths_map.items():
            if not video_path.exists():
                logger.error(f"[Task {self.task_id}][{cam_id}] Video file not found: {video_path}. Will skip this camera.")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"[Task {self.task_id}][{cam_id}] Could not open video: {video_path}. Will skip this camera.")
                continue

            self.video_captures[cam_id] = cap
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_actual_fps[cam_id] = actual_fps if actual_fps > 0 else 25.0
            total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames_in_video[cam_id] = total_original_frames

            skip_interval = 1
            if self.target_fps > 0 and self.video_actual_fps[cam_id] > self.target_fps:
                skip_interval = max(1, round(self.video_actual_fps[cam_id] / self.target_fps))
            self.frame_skip_intervals[cam_id] = skip_interval

            if total_original_frames > 0 and skip_interval > 0:
                self.num_processed_frames_per_video[cam_id] = math.ceil(total_original_frames / skip_interval)
            else:
                self.num_processed_frames_per_video[cam_id] = 0

            logger.info(
                f"[Task {self.task_id}][{cam_id}] Opened video {video_path.name}. "
                f"Actual FPS: {self.video_actual_fps[cam_id]:.2f}, "
                f"Target FPS: {self.target_fps}, Skip Interval: {skip_interval}, "
                f"Total Original Frames: {self.total_frames_in_video[cam_id]}, "
                f"Est. Processed Frames: {self.num_processed_frames_per_video[cam_id]}."
            )
        self._is_open = True

    def get_num_processed_frames_for_camera(self, cam_id: CameraID) -> int:
        """
        Returns the estimated number of frames that will be processed (yielded)
        for a given camera from its current sub-video, considering skip intervals.
        """
        return self.num_processed_frames_per_video.get(cam_id, 0)

    async def get_next_frame_batch(self) -> Tuple[FrameBatch, bool]:
        """
        Asynchronously reads the next frame from each video according to skip intervals
        and returns them as a batch.

        Returns:
            A tuple: (FrameBatch, bool indicating if any video still has frames).
            FrameBatch: Dict mapping CameraID to Optional[FrameData (image_np, frame_pseudo_path)].
                        None if a camera's video ended or failed to read.
            bool: True if at least one video capture is still active and provided a frame,
                  False if all videos have ended.
        """
        if not self._is_open:
            logger.warning(f"[Task {self.task_id}] Attempted to get frames, but provider is not open.")
            return {}, False

        current_batch: FrameBatch = {}
        any_video_active = False

        tasks = []
        for cam_id, cap in self.video_captures.items():
            tasks.append(self._read_frame_for_camera(cam_id, cap))

        results = await asyncio.gather(*tasks)

        for cam_id_res, frame_data_res in results:
            current_batch[cam_id_res] = frame_data_res
            if frame_data_res is not None:
                any_video_active = True

        if not any_video_active and self.loop_videos and self.video_captures:
            logger.info(f"[Task {self.task_id}] All videos ended, looping enabled. Re-opening videos.")
            self.close()
            self._open_videos()
            if self._is_open:
                 return await self.get_next_frame_batch()
            else:
                 return {}, False

        return current_batch, any_video_active

    async def _read_frame_for_camera(self, cam_id: CameraID, cap: cv2.VideoCapture) -> Tuple[CameraID, Optional[FrameData]]:
        """Helper to read a frame from a single camera respecting skip interval."""
        frame_image_np: Optional[np.ndarray] = None
        pseudo_frame_path = ""

        if cap.isOpened():
            for _ in range(self.frame_skip_intervals[cam_id]):
                ret, frame = await asyncio.to_thread(cap.read)
                self.frame_counters_read[cam_id] += 1
                if not ret:
                    break
                frame_image_np = frame
            
            if frame_image_np is not None:
                self.current_frame_index_processed_per_cam[cam_id] +=1
                video_file_name = Path(self.video_paths_map[cam_id]).name
                pseudo_frame_path = f"cam_{cam_id}/{video_file_name}/frame_{self.current_frame_index_processed_per_cam[cam_id]:06d}.jpg"
                return cam_id, (frame_image_np, pseudo_frame_path)

        return cam_id, None

    def close(self):
        """Releases all video capture objects."""
        logger.info(f"[Task {self.task_id}] Closing BatchedFrameProvider.")
        for cam_id, cap in self.video_captures.items():
            if cap.isOpened():
                cap.release()
                logger.debug(f"[Task {self.task_id}][{cam_id}] Released video capture.")
        self.video_captures.clear()
        self._is_open = False

class VideoDataManagerService:
    """
    Manages downloading videos and providing frames in batches.
    """
    def __init__(self, asset_downloader: AssetDownloader):
        self.asset_downloader = asset_downloader
        self.video_sets_config: List[VideoSetEnvironmentConfig] = settings.VIDEO_SETS
        self.local_video_dir_base: Path = Path(settings.LOCAL_VIDEO_DOWNLOAD_DIR)
        logger.info("VideoDataManagerService initialized.")

    async def download_sub_videos_for_environment_batch(
        self, task_id: uuid.UUID, environment_id: str, sub_video_index: int
    ) -> Dict[CameraID, Path]:
        """
        Downloads a specific sub-video (by index) for all relevant cameras
        in the given environment for a specific task.

        Args:
            task_id: The unique ID of the processing task.
            environment_id: The environment (e.g., "campus").
            sub_video_index: The 0-based index of the sub-video.

        Returns:
            A dictionary mapping CameraID to the local Path of the downloaded sub-video.
        """
        logger.info(
            f"[Task {task_id}] Downloading sub_video index {sub_video_index} for env '{environment_id}'."
        )
        downloaded_video_paths: Dict[CameraID, Path] = {}
        
        task_specific_video_dir = self.local_video_dir_base / str(task_id)
        task_specific_video_dir.mkdir(parents=True, exist_ok=True)

        cameras_in_env = [
            vs_config for vs_config in self.video_sets_config if vs_config.env_id == environment_id
        ]

        if not cameras_in_env:
            logger.warning(f"[Task {task_id}] No camera configurations for environment: {environment_id}.")
            return {}

        download_coroutines = []
        video_metadata_for_download = []

        for cam_config in cameras_in_env:
            if sub_video_index >= cam_config.num_sub_videos:
                logger.warning(
                    f"[Task {task_id}][{cam_config.cam_id}] Requested sub-video index {sub_video_index} "
                    f"is out of bounds (total: {cam_config.num_sub_videos}). Skipping."
                )
                continue

            video_filename = cam_config.sub_video_filename_pattern.format(idx=sub_video_index + 1)
            remote_video_key = f"{cam_config.remote_base_key.strip('/')}/{video_filename}"
            
            local_cam_video_download_dir = task_specific_video_dir / environment_id / cam_config.cam_id
            local_cam_video_download_dir.mkdir(parents=True, exist_ok=True)
            local_video_path = local_cam_video_download_dir / video_filename

            video_metadata_for_download.append({
                "cam_id": CameraID(cam_config.cam_id),
                "remote_key": remote_video_key,
                "local_path": local_video_path
            })

            if not local_video_path.exists():
                logger.debug(f"[Task {task_id}][{cam_config.cam_id}] Queuing download: {remote_video_key} to {local_video_path}")
                download_coroutines.append(
                    self.asset_downloader.download_file_from_dagshub(
                        remote_s3_key=remote_video_key,
                        local_destination_path=str(local_video_path)
                    )
                )
            else:
                logger.debug(f"[Task {task_id}][{cam_config.cam_id}] Video already exists locally: {local_video_path}")
                async def _mock_download_success(): return True
                download_coroutines.append(_mock_download_success())
        
        if not download_coroutines:
            logger.info(f"[Task {task_id}] No videos to download for sub-video index {sub_video_index}, env '{environment_id}'.")
            return {}

        download_results = await asyncio.gather(*download_coroutines, return_exceptions=True)

        for i, meta in enumerate(video_metadata_for_download):
            cam_id, local_path = meta["cam_id"], meta["local_path"]
            result = download_results[i]
            if isinstance(result, Exception) or not result:
                logger.error(
                    f"[Task {task_id}][{cam_id}] Failed to download {meta['remote_key']}: {result}"
                )
            else:
                downloaded_video_paths[cam_id] = local_path
        
        logger.info(
            f"[Task {task_id}] Downloaded {len(downloaded_video_paths)} videos for sub-video index {sub_video_index}."
        )
        return downloaded_video_paths

    def get_batched_frame_provider(
        self,
        task_id: uuid.UUID,
        local_video_paths_map: Dict[CameraID, Path],
        loop_videos: bool = False
    ) -> BatchedFrameProvider:
        """
        Creates and returns a BatchedFrameProvider for the given set of local video files.
        """
        logger.info(f"[Task {task_id}] Creating BatchedFrameProvider for {len(local_video_paths_map)} videos.")
        return BatchedFrameProvider(
            task_id=task_id,
            video_paths_map=local_video_paths_map,
            target_fps=settings.TARGET_FPS,
            jpeg_quality=settings.FRAME_JPEG_QUALITY,
            loop_videos=loop_videos
        )

    async def cleanup_task_data(self, task_id: uuid.UUID):
        """Removes downloaded video data for a specific task."""
        task_specific_video_dir = self.local_video_dir_base / str(task_id)
        if task_specific_video_dir.exists():
            try:
                import shutil
                await asyncio.to_thread(shutil.rmtree, task_specific_video_dir)
                logger.info(f"[Task {task_id}] Cleaned up downloaded video data from {task_specific_video_dir}.")
            except Exception as e:
                logger.error(f"[Task {task_id}] Error cleaning up task data {task_specific_video_dir}: {e}", exc_info=True)
        else:
            logger.info(f"[Task {task_id}] No video data directory found at {task_specific_video_dir} to cleanup.")

    def get_max_sub_videos_for_environment(self, environment_id: str) -> int:
        """
        Determines the maximum number of sub-videos any camera in the specified
        environment has. This helps determine how many sub-video batches to process.
        """
        max_subs = 0
        cameras_in_env = [
            vs_config for vs_config in self.video_sets_config if vs_config.env_id == environment_id
        ]
        if not cameras_in_env:
            return 0
        for cam_config in cameras_in_env:
            if cam_config.num_sub_videos > max_subs:
                max_subs = cam_config.num_sub_videos
        return max_subs