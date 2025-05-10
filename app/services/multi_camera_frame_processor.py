"""
Service for processing a batch of frames from multiple cameras simultaneously.
This involves detection, per-camera tracking, handoff trigger detection, and cross-camera Re-ID.
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

import numpy as np
import torch

from app.models.base_models import AbstractDetector
from app.common_types import (
    CameraID, TrackKey, GlobalID, FeatureVector, FrameBatch, RawDetection,
    TrackedObjectData, BoundingBoxXYXY, TrackID,
    HandoffTriggerInfo, ExitRuleModel, # NEW IMPORTS
    QUADRANT_REGIONS_TEMPLATE, DIRECTION_TO_QUADRANTS_MAP # NEW IMPORTS
)
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.reid_components import ReIDStateManager
from app.services.notification_service import NotificationService
from app.core.config import settings # For accessing handoff settings

logger = logging.getLogger(__name__)

class MultiCameraFrameProcessor:
    """
    Orchestrates processing for a batch of frames, one from each camera,
    performing detection, per-camera tracking, handoff detection, and global Re-ID.
    """

    def __init__(
        self,
        detector: AbstractDetector,
        tracker_factory: CameraTrackerFactory,
        notification_service: NotificationService, # Retained for orchestrator's use
        device: torch.device,
    ):
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.notification_service = notification_service
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
        parsed_tracks: List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]] = []
        if tracker_output_np is None or tracker_output_np.size == 0:
            return parsed_tracks

        num_cols = tracker_output_np.shape[1]
        # BoxMOT trackers usually output [x1,y1,x2,y2, track_id, conf, cls_id, (+optional features)]
        # So, 7 columns for base, features start at index 7.
        if num_cols < 7: # Check for minimum required columns (x1,y1,x2,y2,id,conf,cls)
            logger.warning(
                f"[Task {task_id}][{camera_id}] Tracker output has {num_cols} columns. Expected >= 7 for track data + optional features. Features might be missing or data corrupt."
            )
            # Attempt to parse basic info if at least 5 columns (x1,y1,x2,y2,id)
            if num_cols < 5: return parsed_tracks


        for row_idx, row in enumerate(tracker_output_np):
            try:
                x1, y1, x2, y2 = map(float, row[0:4])
                track_id_val = row[4]
                
                if not np.isfinite(track_id_val) or track_id_val < 0:
                    # logger.debug(f"[{camera_id}] Invalid track ID {track_id_val} in row {row_idx}. Skipping.")
                    continue
                track_id_int = int(track_id_val)

                if x2 <= x1 or y2 <= y1: # Invalid bbox
                    # logger.debug(f"[{camera_id}] Invalid bbox {row[0:4]} for T:{track_id_int}. Skipping.")
                    continue

                bbox_xyxy = BoundingBoxXYXY([x1, y1, x2, y2])
                track_key: TrackKey = (camera_id, TrackID(track_id_int))
                
                feature_vector: Optional[FeatureVector] = None
                # Features are expected after id, conf, cls. If conf/cls are present, features start at index 7.
                if num_cols > 7: 
                    feature_data = row[7:]
                    if feature_data.size > 0 and np.isfinite(feature_data).all():
                        feature_vector = FeatureVector(feature_data.astype(np.float32))
                
                parsed_tracks.append((track_key, bbox_xyxy, feature_vector))
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(
                    f"[Task {task_id}][{camera_id}] Error parsing raw tracker output row {row_idx}: {row}. Error: {e}", exc_info=False
                )
        return parsed_tracks

    # --- NEW: Handoff Trigger Detection Logic (adapted from POC) ---
    def _check_handoff_triggers_for_camera(
        self,
        task_id: uuid.UUID, # Added for logging context
        environment_id: str, # Added to fetch correct config
        camera_id: CameraID,
        tracked_dets_np: np.ndarray, # Output from this camera's tracker
        frame_shape: Tuple[int, int] # (height, width) of the current frame for this camera
    ) -> List[HandoffTriggerInfo]:
        """
        Checks if active tracks in a camera's frame overlap significantly
        with predefined exit quadrants, based on configured ExitRuleModels.
        """
        triggers_found: List[HandoffTriggerInfo] = []
        cam_detail_key = (environment_id, str(camera_id)) # Config key
        cam_handoff_config = settings.CAMERA_HANDOFF_DETAILS.get(cam_detail_key)

        if not cam_handoff_config or not cam_handoff_config.exit_rules:
            # logger.debug(f"[{camera_id}] No handoff rules configured or found. Skipping trigger check.")
            return triggers_found
        
        exit_rules: List[ExitRuleModel] = cam_handoff_config.exit_rules
        min_overlap_ratio = settings.MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT

        if min_overlap_ratio <= 0:
            return triggers_found # Handoff trigger check disabled by ratio

        H, W = frame_shape
        if H <= 0 or W <= 0:
            logger.warning(f"[Task {task_id}][{camera_id}] Invalid frame shape {frame_shape} for handoff check.")
            return triggers_found

        processed_track_ids_this_cam_frame = set() # Avoid multiple triggers for the same track in one frame

        for rule in exit_rules:
            relevant_quadrant_names = DIRECTION_TO_QUADRANTS_MAP.get(rule.direction, [])
            if not relevant_quadrant_names:
                continue

            # Calculate actual quadrant coordinates for this frame shape
            exit_regions_coords = []
            for qn in relevant_quadrant_names:
                template_func = QUADRANT_REGIONS_TEMPLATE.get(qn)
                if template_func:
                    exit_regions_coords.append(template_func(W, H))
            
            if not exit_regions_coords:
                continue

            for track_data_row in tracked_dets_np:
                # BoxMOT output: [x1,y1,x2,y2, track_id, conf, cls_id, ...]
                if len(track_data_row) < 5: continue 
                try:
                    track_id_val = track_data_row[4]
                    if not np.isfinite(track_id_val) or track_id_val < 0: continue
                    track_id_int = TrackID(int(track_id_val))

                    if track_id_int in processed_track_ids_this_cam_frame:
                        continue # Already triggered a rule for this track in this frame

                    bbox_xyxy_list = BoundingBoxXYXY(list(map(float, track_data_row[0:4])))
                    x1, y1, x2, y2 = bbox_xyxy_list
                    bbox_w, bbox_h = x2 - x1, y2 - y1
                    if bbox_w <= 0 or bbox_h <= 0: continue
                    bbox_area = float(bbox_w * bbox_h)

                    total_intersection_area = 0.0
                    for qx1, qy1, qx2, qy2 in exit_regions_coords:
                        inter_x1, inter_y1 = max(x1, float(qx1)), max(y1, float(qy1))
                        inter_x2, inter_y2 = min(x2, float(qx2)), min(y2, float(qy2))
                        inter_w = max(0.0, inter_x2 - inter_x1)
                        inter_h = max(0.0, inter_y2 - inter_y1)
                        total_intersection_area += float(inter_w * inter_h)
                    
                    if bbox_area > 1e-5 and (total_intersection_area / bbox_area) >= min_overlap_ratio:
                        source_track_key: TrackKey = (camera_id, track_id_int)
                        trigger_info = HandoffTriggerInfo(
                            source_track_key=source_track_key, rule=rule, source_bbox=bbox_xyxy_list
                        )
                        triggers_found.append(trigger_info)
                        processed_track_ids_this_cam_frame.add(track_id_int)
                        logger.info(
                            f"[Task {task_id}][{camera_id}] HANDOFF TRIGGER: Track {source_track_key} "
                            f"matched rule '{rule.direction}' -> Cam '{rule.target_cam_id}' "
                            f"Area '{rule.target_entry_area}'. Overlap: {total_intersection_area/bbox_area:.2f}"
                        )
                        break # A track can only trigger one rule per frame effectively
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(
                        f"[Task {task_id}][{camera_id}] Error processing track for handoff: {track_data_row}. Error: {e}", exc_info=False
                    )
        return triggers_found


    async def process_frame_batch(
        self,
        task_id: uuid.UUID,
        environment_id: str, # NEW: Needed to fetch camera-specific handoff configs
        reid_manager: ReIDStateManager,
        frame_batch: FrameBatch,
        processed_frame_count: int # Global frame counter for the task
    ) -> Dict[CameraID, List[TrackedObjectData]]:
        """Processes a batch of frames with new Re-ID logic and handoff detection."""
        batch_start_time = asyncio.get_event_loop().time()
        logger.info(f"[Task {task_id}][Env {environment_id}] MFProc: Processing batch for global frame count {processed_frame_count}.")

        await self._ensure_detector_loaded()

        aggregated_parsed_track_data_this_batch: Dict[TrackKey, Tuple[BoundingBoxXYXY, Optional[FeatureVector]]] = {}
        active_track_keys_this_batch_set: Set[TrackKey] = set()
        all_handoff_triggers_this_batch: List[HandoffTriggerInfo] = [] # Collect all triggers

        # --- Detection Phase ---
        per_camera_detections_tasks = []
        valid_cam_ids_in_batch = [cam_id for cam_id, data in frame_batch.items() if data is not None]
        camera_frame_shapes: Dict[CameraID, Tuple[int, int]] = {} # Store frame shapes

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
                camera_frame_shapes[cam_id_loop] = frame_image_np_loop.shape[:2] # Store H, W
                per_camera_detections_tasks.append(detect_for_camera(cam_id_loop, frame_image_np_loop))
        
        detection_results: List[Any] = await asyncio.gather(*per_camera_detections_tasks, return_exceptions=True)

        # --- Tracking, Feature Parsing, and Handoff Trigger Detection Phase ---
        for i, det_res_or_exc in enumerate(detection_results):
            cam_id_from_det_task_order = valid_cam_ids_in_batch[i]
            
            if isinstance(det_res_or_exc, Exception):
                logger.error(f"[Task {task_id}][{cam_id_from_det_task_order}] Detection failed: {det_res_or_exc}", exc_info=True)
                continue
            # ... (rest of detection result validation) ...
            cam_id, raw_detections_list = det_res_or_exc
            frame_data = frame_batch.get(cam_id)
            if not frame_data: continue
            frame_image_np, _ = frame_data
            current_frame_shape = camera_frame_shapes.get(cam_id)
            if not current_frame_shape:
                logger.warning(f"[Task {task_id}][{cam_id}] Frame shape missing. Skipping tracking & handoff.")
                continue


            detections_np_for_tracker = np.array(
                [(*d.bbox_xyxy, d.confidence, d.class_id) for d in raw_detections_list], dtype=np.float32
            ) if raw_detections_list else np.empty((0, 6))

            try:
                tracker_instance = await self.tracker_factory.get_tracker(task_id, cam_id)
                raw_tracker_output_np: np.ndarray = await tracker_instance.update(detections_np_for_tracker, frame_image_np)
                
                parsed_tracks_this_camera = self._parse_raw_tracker_output(task_id, cam_id, raw_tracker_output_np)
                for track_key, bbox, feature in parsed_tracks_this_camera:
                    aggregated_parsed_track_data_this_batch[track_key] = (bbox, feature)
                    active_track_keys_this_batch_set.add(track_key)

                # Check for handoff triggers for this camera
                if raw_tracker_output_np.size > 0: # Only check if there were tracks
                    handoff_triggers_this_cam = self._check_handoff_triggers_for_camera(
                        task_id, environment_id, cam_id, raw_tracker_output_np, current_frame_shape
                    )
                    all_handoff_triggers_this_batch.extend(handoff_triggers_this_cam)

            except Exception as e:
                logger.error(f"[Task {task_id}][{cam_id}] Error during tracking, parsing or handoff check: {e}", exc_info=True)
        
        # --- Re-ID Association Phase ---
        features_for_reid_input: Dict[TrackKey, FeatureVector] = {
            tk: feat for tk, (_, feat) in aggregated_parsed_track_data_this_batch.items() if feat is not None
        }
        
        # Create active_triggers_map for ReIDStateManager
        active_triggers_map_for_reid: Dict[TrackKey, HandoffTriggerInfo] = {
            trigger.source_track_key: trigger for trigger in all_handoff_triggers_this_batch
        }

        if features_for_reid_input or active_track_keys_this_batch_set:
            await reid_manager.associate_features_and_update_state(
                features_for_reid_input,
                active_track_keys_this_batch_set,
                active_triggers_map_for_reid, # Pass the trigger map
                processed_frame_count
            )
        else:
            logger.debug(f"[Task {task_id}] MFProc: No features or active tracks for ReID for frame {processed_frame_count}.")


        # --- Construct Final Output for this Batch ---
        final_batch_results: Dict[CameraID, List[TrackedObjectData]] = defaultdict(list)
        for track_key, (bbox_xyxy, original_feature) in aggregated_parsed_track_data_this_batch.items():
            cam_id_active, track_id_active = track_key
            gid_assigned: Optional[GlobalID] = reid_manager.track_to_global_id.get(track_key)
            # Homography/map_coords projection would happen here if enabled, using self.homography_service
            # For now, map_coords will be None unless homography is added back.
            map_coords_output: Optional[Tuple[float, float]] = None # Placeholder

            track_obj_data = TrackedObjectData(
                camera_id=cam_id_active,
                track_id=track_id_active,
                global_person_id=gid_assigned,
                bbox_xyxy=bbox_xyxy,
                feature_vector=list(original_feature) if original_feature is not None else None,
                map_coords=map_coords_output # Add map_coords to output
            )
            final_batch_results[cam_id_active].append(track_obj_data)

        logger.info(
            f"[Task {task_id}] MFProc: Batch for frame count {processed_frame_count} finished in "
            f"{(asyncio.get_event_loop().time() - batch_start_time):.3f}s. "
            f"Active tracks: {len(active_track_keys_this_batch_set)}. "
            f"Features for ReID: {len(features_for_reid_input)}. "
            f"Handoff Triggers: {len(all_handoff_triggers_this_batch)}."
        )
        return dict(final_batch_results)