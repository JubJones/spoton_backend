"""
Service for processing a batch of frames from multiple cameras simultaneously.
This involves detection, per-camera tracking, and cross-camera Re-ID.
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

import numpy as np
import torch

from app.models.base_models import AbstractDetector # Removed BoundingBox as it's not used here
from app.common_types import CameraID, TrackKey, GlobalID, FeatureVector, FrameBatch, RawDetection, TrackedObjectData, BoundingBoxXYXY, TrackID
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.reid_components import ReIDStateManager
from app.services.notification_service import NotificationService # Retained
from app.core.config import settings
# from app.api.v1 import schemas as api_schemas # Not used directly in this file after changes

logger = logging.getLogger(__name__)

class MultiCameraFrameProcessor:
    """
    Orchestrates processing for a batch of frames, one from each camera,
    performing detection, per-camera tracking, and global Re-ID.
    """

    def __init__(
        self,
        detector: AbstractDetector,
        tracker_factory: CameraTrackerFactory,
        notification_service: NotificationService,
        device: torch.device,
    ):
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.notification_service = notification_service # Retained for orchestrator's potential use
        self.device = device
        self._detector_model_loaded = False
        logger.info("MultiCameraFrameProcessor initialized.")

    async def _ensure_detector_loaded(self):
        """Ensures the detector model is loaded."""
        if not self._detector_model_loaded:
            logger.info(f"[MultiCameraFrameProcessor] Detector model not yet loaded. Loading now...")
            try:
                await self.detector.load_model()
                self._detector_model_loaded = True
                logger.info(f"[MultiCameraFrameProcessor] Detector model loaded successfully.")
            except Exception as e:
                logger.error(f"[MultiCameraFrameProcessor] CRITICAL: Failed to load detector model: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load detector model: {e}") from e

    def _parse_raw_tracker_output(
        self, task_id: uuid.UUID, camera_id: CameraID, tracker_output_np: np.ndarray
    ) -> List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]]:
        """Parses raw tracker output (BoxMOT format with features)."""
        # This function's internal logic remains the same as provided in the prompt.
        # It correctly extracts TrackKey, BoundingBoxXYXY, and Optional[FeatureVector].
        parsed_tracks: List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]] = []
        if tracker_output_np is None or tracker_output_np.size == 0:
            return parsed_tracks

        num_cols = tracker_output_np.shape[1]
        if num_cols < 7:
            logger.warning(
                f"[Task {task_id}][{camera_id}] Tracker output has {num_cols} columns. Expected >= 7 for basic track data + features. Features might be missing."
            )

        for row_idx, row in enumerate(tracker_output_np):
            try:
                x1, y1, x2, y2 = map(float, row[0:4])
                track_id_val = row[4] # Track ID from tracker
                
                if not np.isfinite(track_id_val) or track_id_val < 0:
                    continue
                track_id_int = int(track_id_val)

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox_xyxy = BoundingBoxXYXY([x1, y1, x2, y2])
                cam_id_typed = CameraID(camera_id) # camera_id is already CameraID from signature
                track_id_typed = TrackID(track_id_int) # NewType for clarity
                track_key: TrackKey = (cam_id_typed, track_id_typed)
                
                feature_vector: Optional[FeatureVector] = None
                if num_cols > 7: # Features start at column index 7
                    feature_data = row[7:]
                    if feature_data.size > 0 and np.isfinite(feature_data).all():
                        feature_vector = FeatureVector(feature_data.astype(np.float32))
                
                parsed_tracks.append((track_key, bbox_xyxy, feature_vector))
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(
                    f"[Task {task_id}][{camera_id}] Error parsing raw tracker output row {row_idx}: {row}. Error: {e}", exc_info=False
                )
        return parsed_tracks

    async def process_frame_batch(
        self,
        task_id: uuid.UUID,
        reid_manager: ReIDStateManager,
        frame_batch: FrameBatch,
        processed_frame_count: int
    ) -> Dict[CameraID, List[TrackedObjectData]]:
        """Processes a batch of frames with new Re-ID logic."""
        batch_start_time = asyncio.get_event_loop().time()
        logger.info(f"[Task {task_id}] MFProc: Processing batch for global frame count {processed_frame_count}.")

        await self._ensure_detector_loaded()

        aggregated_parsed_track_data_this_batch: Dict[TrackKey, Tuple[BoundingBoxXYXY, Optional[FeatureVector]]] = {}
        active_track_keys_this_batch_set: Set[TrackKey] = set() # Using a set for efficient management of active keys

        # --- Detection Phase ---
        per_camera_detections_tasks = []
        valid_cam_ids_in_batch = [cam_id for cam_id, data in frame_batch.items() if data is not None]

        async def detect_for_camera(cam_id_local: CameraID, frame_image_np_local: np.ndarray) -> Tuple[CameraID, List[RawDetection]]:
            # This helper's internal logic remains the same.
            raw_model_detections = await self.detector.detect(frame_image_np_local)
            converted_detections = [
                RawDetection(
                    bbox_xyxy=BoundingBoxXYXY(d.bbox.to_list()), 
                    confidence=d.confidence, 
                    class_id=d.class_id
                ) for d in raw_model_detections
            ]
            return cam_id_local, converted_detections

        for cam_id_loop in valid_cam_ids_in_batch:
            frame_data_loop = frame_batch[cam_id_loop]
            if frame_data_loop:
                frame_image_np_loop, _ = frame_data_loop
                per_camera_detections_tasks.append(detect_for_camera(cam_id_loop, frame_image_np_loop))
        
        detection_results: List[Any] = await asyncio.gather(*per_camera_detections_tasks, return_exceptions=True)

        # --- Tracking and Feature Parsing Phase ---
        for i, det_res_or_exc in enumerate(detection_results):
            cam_id_from_det_task_order = valid_cam_ids_in_batch[i]
            
            if isinstance(det_res_or_exc, Exception):
                logger.error(f"[Task {task_id}][{cam_id_from_det_task_order}] Detection failed: {det_res_or_exc}", exc_info=True)
                continue
            if not (isinstance(det_res_or_exc, tuple) and len(det_res_or_exc) == 2):
                logger.error(f"[Task {task_id}][{cam_id_from_det_task_order}] Unexpected result type from detection: {type(det_res_or_exc)}.")
                continue

            cam_id, raw_detections_list = det_res_or_exc
            frame_data = frame_batch.get(cam_id)
            if not frame_data: 
                logger.warning(f"[Task {task_id}][{cam_id}] Frame data missing post-detection. Skipping tracking.")
                continue
            frame_image_np, _ = frame_data

            detections_np_for_tracker = np.array(
                [(*d.bbox_xyxy, d.confidence, d.class_id) for d in raw_detections_list], dtype=np.float32
            ) if raw_detections_list else np.empty((0, 6))

            try:
                tracker_instance = await self.tracker_factory.get_tracker(task_id, cam_id)
                raw_tracker_output_np: np.ndarray = await tracker_instance.update(detections_np_for_tracker, frame_image_np)
                
                # Parse tracker output for this camera
                parsed_tracks_this_camera = self._parse_raw_tracker_output(task_id, cam_id, raw_tracker_output_np)
                for track_key, bbox, feature in parsed_tracks_this_camera:
                    aggregated_parsed_track_data_this_batch[track_key] = (bbox, feature)
                    active_track_keys_this_batch_set.add(track_key)
            except Exception as e:
                logger.error(f"[Task {task_id}][{cam_id}] Error during tracking or parsing: {e}", exc_info=True)
        
        # --- Re-ID Association Phase ---
        # Prepare features for ReID (only those with actual feature vectors)
        features_for_reid_input: Dict[TrackKey, FeatureVector] = {
            tk: feat for tk, (_, feat) in aggregated_parsed_track_data_this_batch.items() if feat is not None
        }

        if features_for_reid_input or active_track_keys_this_batch_set: # Proceed if features or active tracks (for lifecycle)
            logger.debug(
                f"[Task {task_id}] MFProc: Calling ReID association. "
                f"Features for ReID: {len(features_for_reid_input)}. "
                f"Total active tracks this batch: {len(active_track_keys_this_batch_set)}."
            )
            # This call now encapsulates the new Re-ID decision logic, matching, conflict resolution, etc.
            # It modifies reid_manager's state directly.
            await reid_manager.associate_features_and_update_state(
                features_for_reid_input, # Only tracks with features are candidates for matching
                active_track_keys_this_batch_set, # All active tracks are needed for lifecycle updates
                processed_frame_count
            )
        else:
            logger.debug(f"[Task {task_id}] MFProc: No features or active tracks for ReID processing for frame count {processed_frame_count}.")


        # --- Construct Final Output for this Batch ---
        final_batch_results: Dict[CameraID, List[TrackedObjectData]] = defaultdict(list)
        # Iterate over the tracks that were active in this batch (from aggregated_parsed_track_data_this_batch)
        for track_key, (bbox_xyxy, original_feature) in aggregated_parsed_track_data_this_batch.items():
            cam_id_active, track_id_active = track_key # track_id_active is TrackID type (int)
            
            # Get the authoritative GlobalID from the ReIDStateManager (which was updated by the new logic)
            gid_assigned: Optional[GlobalID] = reid_manager.track_to_global_id.get(track_key) # GlobalID is str/UUID

            feature_list_for_output: Optional[List[float]] = None
            if original_feature is not None:
                # The feature stored in TrackedObjectData could be the original one,
                # or a normalized one if consistency is desired. ReIDStateManager works with normalized internally.
                # For output, providing the original (if available) or normalized is a choice.
                # Let's assume the original parsed feature is fine for output.
                feature_list_for_output = list(original_feature) # Convert np.ndarray to list

            track_obj_data = TrackedObjectData(
                camera_id=cam_id_active, # Already CameraID type
                track_id=track_id_active, # Already TrackID type (int)
                global_person_id=gid_assigned, # GlobalID type (str/UUID)
                bbox_xyxy=bbox_xyxy, # BoundingBoxXYXY type (List[float])
                confidence=None, # Confidence is not explicitly carried through this particular data flow path.
                                 # Tracker output parsing could be extended to include it.
                feature_vector=feature_list_for_output # List[float] or None
            )
            final_batch_results[cam_id_active].append(track_obj_data)

        logger.info(
            f"[Task {task_id}] MFProc: Batch for frame count {processed_frame_count} finished in "
            f"{(asyncio.get_event_loop().time() - batch_start_time):.3f}s. "
            f"Total active tracks: {len(active_track_keys_this_batch_set)}. "
            f"Tracks with features for ReID input: {len(features_for_reid_input)}."
        )
        return dict(final_batch_results)