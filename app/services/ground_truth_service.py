
import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class GroundTruthService:
    """
    Service to load and serve Ground Truth data for detections and tracking.
    Bypasses ML models when enabled.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Cache: camera_id -> {frame_number -> [tracks]}
        self._cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        logger.info(f"GroundTruthService initialized with data_dir: {self.data_dir}")

    def load_tracks_for_camera(self, camera_id: str) -> bool:
        """
        Load GT tracks for a specific camera from standard MOT format file.
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        if camera_id in self._cache:
            return True

        # Potential paths for gt.txt
        # 1. {data_dir}/{camera_id}/gt/gt.txt (Standard MTMMC)
        # 2. {data_dir}/train/{camera_id}/gt/gt.txt (Some MTMMC variants)
        # 3. {data_dir}/{camera_id}.txt (Simple format)
        
        candidates = [
            Path(self.data_dir) / camera_id / "gt" / "gt.txt",
            Path(self.data_dir) / "train" / camera_id / "gt" / "gt.txt",
            Path(self.data_dir) / f"{camera_id}.txt"
        ]
        
        gt_path = None
        for p in candidates:
            if p.exists():
                gt_path = p
                break
        
        if not gt_path:
            logger.error(f"GT file not found for camera {camera_id}. Searched: {candidates}")
            return False

        try:
            logger.info(f"Loading GT data from {gt_path}...")
            # Load data using numpy for speed
            # Format: frame, id, left, top, width, height, conf, class, visibility
            data = np.loadtxt(str(gt_path), delimiter=',')
            data = np.atleast_2d(data)
            
            # Organize by frame
            self._cache[camera_id] = {}
            
            for row in data:
                frame_num = int(row[0])
                track_id = int(row[1])
                x1 = float(row[2])
                y1 = float(row[3])
                w = float(row[4])
                h = float(row[5])
                
                # Check formatting (some datasets leverage -1 for unused fields)
                conf = float(row[6]) if len(row) > 6 else 1.0
                cls_id = int(row[7]) if len(row) > 7 else 1  # Assume class 1 (person) if missing
                
                # MTMMC often puts class at pos 7 (index 8? no 0-indexed) -> index 7
                # Standard MOT16/17: 7th column is confidence, 8th is class, 9th visibility.
                
                # Create track object compatible with system
                track = {
                    "track_id": track_id,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x1 + w, y1 + h],
                    "xyxy": [x1, y1, x1 + w, y1 + h], # Redundant but safe
                    "det_class": cls_id
                }
                
                if frame_num not in self._cache[camera_id]:
                    self._cache[camera_id][frame_num] = []
                self._cache[camera_id][frame_num].append(track)
                
            logger.info(f"Loaded {len(data)} tracks for {camera_id} across {len(self._cache[camera_id])} frames.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GT data for {camera_id}: {e}")
            return False

    def get_tracks_for_frame(self, camera_id: str, frame_number: int) -> List[Dict[str, Any]]:
        """Retrieve tracks for a specific frame."""
        # Ensure loaded
        if camera_id not in self._cache:
            success = self.load_tracks_for_camera(camera_id)
            if not success:
                return []
        
        return self._cache[camera_id].get(frame_number, [])
