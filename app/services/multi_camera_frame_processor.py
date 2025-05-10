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

from app.models.base_models import AbstractDetector, BoundingBox
from app.common_types import CameraID, TrackKey, GlobalID, FeatureVector, FrameBatch, RawDetection, TrackedObjectData, BoundingBoxXYXY, TrackID # Added TrackID
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.reid_components import ReIDStateManager, associate_tracks_to_gallery, update_reid_state_after_frame
from app.services.notification_service import NotificationService
from app.core.config import settings
from app.api.v1 import schemas as api_schemas

logger = logging.getLogger(__name__)

REID_REFRESH_INTERVAL = 10

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
        self.notification_service = notification_service
        self.device = device
        self._detector_model_loaded = False
        logger.info("MultiCameraFrameProcessor initialized.")

    async def _ensure_detector_loaded(self):
        """Ensures the detector model is loaded. Calls load_model() if not already loaded."""
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
        """
        Parses the raw numpy array output from a BoxMOT tracker.
        BoxMOT's BotSort with ReID typically returns: [x1, y1, x2, y2, track_id, conf, cls_id, feature_vector...]
        The features start from column index 7.
        """
        parsed_tracks: List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]] = []
        if tracker_output_np is None or tracker_output_np.size == 0:
            return parsed_tracks

        num_cols = tracker_output_np.shape[1]
        if num_cols < 7:
            logger.warning(
                f"[Task {task_id}][{camera_id}] Tracker output has too few columns ({num_cols}). Expected >= 7 for track data. Skipping parsing for this output."
            )
            return parsed_tracks

        for row_idx, row in enumerate(tracker_output_np):
            try:
                x1, y1, x2, y2 = map(float, row[0:4])
                track_id_val = row[4] 
                
                if not np.isfinite(track_id_val) or track_id_val < 0:
                    continue
                track_id_int = int(track_id_val) # Convert to int after validation

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox_xyxy = BoundingBoxXYXY([x1, y1, x2, y2])
                # --- Corrected TrackKey instantiation ---
                # Create CameraID and TrackID instances explicitly if desired for type safety with NewType
                # then create the tuple.
                cam_id_typed = CameraID(camera_id) # camera_id is already CameraID type from method signature
                track_id_typed = TrackID(track_id_int)
                track_key: TrackKey = (cam_id_typed, track_id_typed)
                # --- End of correction ---
                
                feature_vector: Optional[FeatureVector] = None
                if num_cols > 7:
                    feature_data = row[7:]
                    if feature_data.size > 0 and np.isfinite(feature_data).all():
                        feature_vector = FeatureVector(feature_data.astype(np.float32))

                parsed_tracks.append((track_key, bbox_xyxy, feature_vector))
            except (ValueError, IndexError, TypeError) as e: # Catch TypeError as well
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
        batch_start_time = asyncio.get_event_loop().time()
        logger.info(f"[Task {task_id}] MFProc: Processing batch for frame count {processed_frame_count}.")

        await self._ensure_detector_loaded()

        active_track_keys_current_batch: Set[TrackKey] = set()
        parsed_track_info_map: Dict[TrackKey, Tuple[BoundingBoxXYXY, Optional[FeatureVector]]] = {}
        
        per_camera_detections_tasks = []
        valid_cam_ids_in_batch = [cam_id for cam_id, data in frame_batch.items() if data is not None]

        async def detect_for_camera(cam_id_local: CameraID, frame_image_np_local: np.ndarray) -> Tuple[CameraID, List[RawDetection]]:
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

        valid_detection_results = []
        for i, det_res_or_exc in enumerate(detection_results):
            original_cam_id_index = i # Assuming order is maintained from valid_cam_ids_in_batch
            if isinstance(det_res_or_exc, Exception):
                cam_id_failed = valid_cam_ids_in_batch[original_cam_id_index]
                logger.error(f"[Task {task_id}][{cam_id_failed}] Detection failed: {det_res_or_exc}", exc_info=True)
            elif isinstance(det_res_or_exc, tuple) and len(det_res_or_exc) == 2: # Expected (CameraID, List[RawDetection])
                valid_detection_results.append(det_res_or_exc)
            else: # Unexpected result type
                 cam_id_unknown_res = valid_cam_ids_in_batch[original_cam_id_index]
                 logger.error(f"[Task {task_id}][{cam_id_unknown_res}] Unexpected result type from detection task: {type(det_res_or_exc)}. Value: {str(det_res_or_exc)[:200]}")

        for cam_id, raw_detections_list in valid_detection_results:
            frame_data = frame_batch.get(cam_id) # Use .get() for safety
            if not frame_data: 
                logger.warning(f"[Task {task_id}][{cam_id}] Frame data missing after detection. Skipping tracking.")
                continue
            frame_image_np, _ = frame_data

            detections_np_for_tracker = np.array(
                [(*d.bbox_xyxy, d.confidence, d.class_id) for d in raw_detections_list], dtype=np.float32
            ) if raw_detections_list else np.empty((0, 6))

            try:
                tracker_instance = await self.tracker_factory.get_tracker(task_id, cam_id)
                raw_tracker_output_np: np.ndarray = await tracker_instance.update(detections_np_for_tracker, frame_image_np)
                
                # Pass the correct CameraID type to _parse_raw_tracker_output
                parsed_tracks_this_camera = self._parse_raw_tracker_output(task_id, CameraID(cam_id), raw_tracker_output_np)


                for track_key, bbox, feature in parsed_tracks_this_camera:
                    active_track_keys_current_batch.add(track_key)
                    norm_feature = reid_manager._normalize_embedding(feature) if feature is not None else None
                    parsed_track_info_map[track_key] = (bbox, norm_feature)
                    
            except Exception as e:
                logger.error(f"[Task {task_id}][{cam_id}] Error during tracking update: {e}", exc_info=True)
        
        features_for_association: Dict[TrackKey, FeatureVector] = {
            tk: feat for tk, (_, feat) in parsed_track_info_map.items() if feat is not None
        }

        if features_for_association:
            logger.debug(f"[Task {task_id}] MFProc: Associating {len(features_for_association)} tracks with features.")
            _ = associate_tracks_to_gallery(
                reid_manager, features_for_association, processed_frame_count
            )
        else:
            logger.debug(f"[Task {task_id}] MFProc: No tracks with features to associate for frame count {processed_frame_count}.")

        update_reid_state_after_frame(reid_manager, active_track_keys_current_batch, processed_frame_count)

        final_batch_results: Dict[CameraID, List[TrackedObjectData]] = defaultdict(list)
        for track_key_active, (bbox_active, norm_feat_active) in parsed_track_info_map.items():
            cam_id_active, track_id_active = track_key_active
            gid_assigned = reid_manager.track_to_global_id.get(track_key_active)
            
            track_obj_data = TrackedObjectData(
                camera_id=cam_id_active,
                track_id=track_id_active,
                global_person_id=gid_assigned,
                bbox_xyxy=bbox_active,
                confidence=None, 
                feature_vector=list(norm_feat_active) if norm_feat_active is not None else None
            )
            final_batch_results[cam_id_active].append(track_obj_data)

        logger.info(
            f"[Task {task_id}] MFProc: Batch for frame count {processed_frame_count} finished in "
            f"{(asyncio.get_event_loop().time() - batch_start_time):.3f}s. "
            f"Active tracks this batch: {len(active_track_keys_current_batch)}. "
            f"Tracks for ReID Assoc: {len(features_for_association)}."
        )
        return dict(final_batch_results)