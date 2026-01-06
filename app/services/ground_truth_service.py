"""
Ground Truth Service

Responsible for loading and serving pre-computed detection/track data from disk.
Simulates a perfect detector/tracker for testing and validation purposes.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import math

from app.core.config import settings

logger = logging.getLogger(__name__)

class GroundTruthService:
    """
    Service to load and serve ground truth data from MOTChallenge-like format files.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or settings.GROUND_TRUTH_DATA_DIR
        self._cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {} # camera_id -> {frame_num -> [tracks]}
        logger.info(f"GroundTruthService initialized with data_dir: {self.data_dir}")
        self._inspect_data_dir()

    def _inspect_data_dir(self):
        """Helper to log contents of the data directory to debug mounting issues."""
        try:
            if os.path.exists(self.data_dir):
                contents = os.listdir(self.data_dir)
                logger.info(f"📂 [DEBUG] Contents of {self.data_dir}: {contents}")
            else:
                logger.error(f"❌ [DEBUG] Directory {self.data_dir} does NOT exist!")
        except Exception as e:
            logger.error(f"❌ [DEBUG] Error listing {self.data_dir}: {e}")

    def load_tracks_for_camera(self, camera_id: str) -> bool:
        """
        Loads ground truth data for a specific camera.
        Expected file path: <data_dir>/<camera_id>/gt/gt.txt or similar.
        We will try standard MOT folder structure: {data_dir}/train/{camera_id}/gt/gt.txt
        Or direct: {data_dir}/{camera_id}.txt
        """
        
        # Try a few common path patterns
        possible_paths = [
            Path(self.data_dir) / camera_id / "gt" / "gt.txt",
            Path(self.data_dir) / "train" / camera_id / "gt" / "gt.txt",
            Path(self.data_dir) / f"{camera_id}.txt",
        ]

        gt_file = None
        for p in possible_paths:
            if p.exists():
                gt_file = p
                break
        
        if not gt_file:
            logger.warning(f"Ground truth file not found for camera {camera_id}. Searched: {possible_paths}")
            # Debug: Check if camera dir exists
            cam_dir = Path(self.data_dir) / camera_id
            if cam_dir.exists():
                logger.info(f"📂 [DEBUG] Camera dir {cam_dir} exists. Contents: {os.listdir(cam_dir)}")
            else:
                logger.warning(f"⚠️ [DEBUG] Camera dir {cam_dir} does NOT exist.")
            return False

        logger.info(f"Loading ground truth for {camera_id} from {gt_file}")
        
        tracks_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    
                    # Format: frame, id, left, top, width, height, conf, class, visibility
                    frame_num = int(parts[0])
                    track_id = int(parts[1])
                    left = float(parts[2])
                    top = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])
                    
                    # Optional fields
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    class_id = int(float(parts[7])) if len(parts) > 7 else 1  # Default to person
                    visibility = float(parts[8]) if len(parts) > 8 else 1.0

                    # Simulate detection/track object
                    track_data = {
                        "track_id": track_id,
                        "global_id": f"global_{track_id}", # Simple pass-through for now, or could come from file if format differs
                        "class_id": class_id,
                        "confidence": conf,
                        "bbox_xyxy": [left, top, left + width, top + height],
                        "bbox": {
                            "x1": left,
                            "y1": top,
                            "x2": left + width,
                            "y2": top + height,
                            "width": width,
                            "height": height,
                            "center_x": left + width / 2,
                            "center_y": top + height / 2
                        }, 
                        "visibility": visibility
                    }
                    
                    if frame_num not in tracks_by_frame:
                        tracks_by_frame[frame_num] = []
                    tracks_by_frame[frame_num].append(track_data)
            
            # 1. TRACE: Ingestion
            total_frames = len(tracks_by_frame)
            total_tracks = sum(len(t) for t in tracks_by_frame.values())
            first_frame_idx = sorted(list(tracks_by_frame.keys()))[0] if tracks_by_frame else -1
            sample_track = tracks_by_frame[first_frame_idx][0] if tracks_by_frame and tracks_by_frame[first_frame_idx] else "None"
            
            logger.info(
                f"🛡️ [GT-TRACE-1] Ingested {gt_file}. "
                f"Total Frames: {total_frames}. Total Tracks: {total_tracks}. "
                f"Frame Range: {first_frame_idx} -> {sorted(list(tracks_by_frame.keys()))[-1]}. "
                f"Sample Track (Frame {first_frame_idx}): {sample_track}"
            )

            self._cache[camera_id] = tracks_by_frame
            logger.info(f"Loaded {len(tracks_by_frame)} frames of GT data for {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load GT file {gt_file}: {e}")
            return False

    def get_tracks_for_frame(self, camera_id: str, frame_number: int) -> List[Dict[str, Any]]:
        """
        Retrieve tracks for a given frame and camera.
        """
        if camera_id not in self._cache:
             # Lazy load
             if not self.load_tracks_for_camera(camera_id):
                 if frame_number % 60 == 0:
                     logger.warning(f"🛡️ [GT-FAIL] Cache miss for {camera_id}. Load FAILED. Path issue?")
                 return []
        
        tracks = self._cache.get(camera_id, {}).get(frame_number, [])
        
        # 2. TRACE: Utilization (Lookup) - Log every 60 frames to avoid spam, or on error
        if frame_number % 60 == 0:
             logger.info(f"🛡️ [GT-TRACE-2] Lookup Cam={camera_id} Frame={frame_number}. Found {len(tracks)} tracks.")

        if not tracks:
             # Force log on EVERY blank frame for a moment to verify execution path
             frames = sorted(list(self._cache[camera_id].keys()))
             min_frame = frames[0] if frames else 'N/A'
             max_frame = frames[-1] if frames else 'N/A'
             sample_frames = frames[:10] if frames else []
             logger.error(
                 f"🛡️ [GT-MISS] Cam={camera_id} Frame={frame_number} NOT FOUND. "
                 f"Total Frames: {len(frames)}. Range: [{min_frame}, {max_frame}]. "
                 f"Sample keys: {sample_frames}"
             )
        
        return tracks
