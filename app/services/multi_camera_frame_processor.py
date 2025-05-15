"""
Service for processing a batch of frames from multiple cameras simultaneously.
This involves detection, per-camera tracking, handoff trigger detection, map projection, and cross-camera Re-ID.
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2 

from app.models.base_models import AbstractDetector
from app.common_types import (
    CameraID, TrackKey, GlobalID, FeatureVector, FrameBatch, RawDetection,
    TrackedObjectData, BoundingBoxXYXY, TrackID,
    HandoffTriggerInfo, ExitRuleModel, CameraHandoffDetailConfig,
    QUADRANT_REGIONS_TEMPLATE, DIRECTION_TO_QUADRANTS_MAP
)
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.reid_components import ReIDStateManager
from app.services.notification_service import NotificationService
from app.services.homography_service import HomographyService # New import
from app.core.config import settings

logger = logging.getLogger(__name__)

class MultiCameraFrameProcessor:
    """
    Orchestrates processing for a batch of frames, one from each camera,
    performing detection, per-camera tracking, handoff detection, map projection, and global Re-ID.
    Relies on pre-loaded detector and homography matrices.
    """

    def __init__(
        self,
        detector: AbstractDetector,
        tracker_factory: CameraTrackerFactory,
        homography_service: HomographyService, # New dependency
        notification_service: NotificationService,
        device: torch.device,
    ):
        self.detector = detector # Assumed to be pre-loaded
        self.tracker_factory = tracker_factory
        self.homography_service = homography_service # Store the service
        self.notification_service = notification_service
        self.device = device
        # No internal _detector_model_loaded flag needed if startup guarantees it.
        # No internal _homography_matrices_cache needed as HomographyService handles it.
        logger.info("MultiCameraFrameProcessor initialized (expects pre-loaded components).")

    # Removed _ensure_detector_loaded method, as detector is expected to be loaded at startup.
    # Removed _load_homography_matrix_for_camera method, HomographyService handles this.

    def _project_point_to_map(
        self, image_point_xy: Tuple[float, float], homography_matrix: Optional[np.ndarray]
    ) -> Optional[List[float]]:
        """Projects a single image point (x, y) to map coordinates [X, Y] using homography."""
        if homography_matrix is None:
            return None
        try:
            img_pt_np = np.array([[image_point_xy]], dtype=np.float32)
            map_pt_np = cv2.perspectiveTransform(img_pt_np, homography_matrix)
            if map_pt_np is not None and map_pt_np.shape == (1, 1, 2):
                return [float(map_pt_np[0, 0, 0]), float(map_pt_np[0, 0, 1])]
            else:
                logger.debug(f"Perspective transform for {image_point_xy} returned unexpected shape or None.")
                return None
        except Exception as e:
            logger.debug(f"Error projecting point {image_point_xy}: {e}")
            return None

    def _parse_raw_tracker_output(
        self, task_id: uuid.UUID, camera_id: CameraID, tracker_output_np: np.ndarray
    ) -> List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]]:
        """Parses raw tracker output (BoxMOT format with features)."""
        parsed_tracks: List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]] = []
        if tracker_output_np is None or tracker_output_np.size == 0:
            return parsed_tracks

        num_cols = tracker_output_np.shape[1]
        # BoxMOT BotSort often outputs: x1, y1, x2, y2, track_id, conf, cls_id, {feature_vector}
        # So, >= 7 columns is expected if features are there. At least 5 without features.
        if num_cols < 7: 
            if num_cols >= 5: 
                 logger.debug(f"[Task {task_id}][{camera_id}] Tracker output has {num_cols} columns. Expected >= 7 for track data + features. Features might be missing or tracker output format changed.")
            else:
                 logger.warning(f"[Task {task_id}][{camera_id}] Tracker output has {num_cols} columns. Expected >= 5 for minimal track data (xyxy, id). Data possibly corrupt.")
                 return parsed_tracks

        for row_idx, row in enumerate(tracker_output_np):
            try:
                x1, y1, x2, y2 = map(float, row[0:4])
                track_id_val = row[4] # track_id
                
                # Basic validation
                if not np.isfinite(track_id_val) or track_id_val < 0: # Basic check for valid ID
                    # logger.warning(f"[Task {task_id}][{camera_id}] Invalid track ID in output row {row_idx}: {track_id_val}")
                    continue 
                track_id_int = int(track_id_val)

                if x2 <= x1 or y2 <= y1: # Invalid bbox
                    # logger.warning(f"[Task {task_id}][{camera_id}] Invalid bbox in output row {row_idx}: {[x1,y1,x2,y2]}")
                    continue

                bbox_xyxy = BoundingBoxXYXY([x1, y1, x2, y2])
                track_key: TrackKey = (camera_id, TrackID(track_id_int))
                
                feature_vector: Optional[FeatureVector] = None
                # Check if features are present (columns beyond the standard 7: xyxy, id, conf, cls)
                if num_cols > 7: 
                    feature_data = row[7:] # Assuming features start from the 8th column
                    if feature_data.size > 0 and np.isfinite(feature_data).all(): # Check if feature data is valid
                        feature_vector = FeatureVector(feature_data.astype(np.float32))
                
                parsed_tracks.append((track_key, bbox_xyxy, feature_vector))
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(
                    f"[Task {task_id}][{camera_id}] Error parsing raw tracker output row {row_idx}: {row}. Error: {e}", exc_info=False
                )
        return parsed_tracks

    def _check_handoff_triggers_for_camera(
        self,
        task_id: uuid.UUID,
        environment_id: str,
        camera_id: CameraID,
        tracked_dets_np: np.ndarray, # Raw tracker output with at least [x1,y1,x2,y2,id]
        frame_shape: Tuple[int, int] # (H, W)
    ) -> List[HandoffTriggerInfo]:
        """Checks if active tracks in a camera's frame overlap significantly with predefined exit quadrants."""
        triggers_found: List[HandoffTriggerInfo] = []
        cam_detail_key = (environment_id, str(camera_id)) # Config key is (env_str, cam_str)
        cam_handoff_config: Optional[CameraHandoffDetailConfig] = settings.CAMERA_HANDOFF_DETAILS.get(cam_detail_key)

        if not cam_handoff_config or not cam_handoff_config.exit_rules:
            return triggers_found # No rules, no triggers
        
        exit_rules: List[ExitRuleModel] = cam_handoff_config.exit_rules
        min_overlap_ratio = settings.MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT

        if min_overlap_ratio <= 0: # If ratio is 0 or less, effectively disabled
            return triggers_found

        H, W = frame_shape
        if H <= 0 or W <= 0:
            logger.warning(f"[Task {task_id}][{camera_id}] Invalid frame shape {frame_shape} for handoff check.")
            return triggers_found

        # To avoid multiple triggers for the same track_id in one frame from different rules
        processed_track_ids_this_cam_frame = set()

        for rule in exit_rules:
            relevant_quadrant_names = DIRECTION_TO_QUADRANTS_MAP.get(rule.direction, [])
            if not relevant_quadrant_names:
                continue # Rule direction not mapped to quadrants

            # Calculate coordinates of all relevant exit quadrants for this rule
            exit_regions_coords = []
            for qn in relevant_quadrant_names:
                template_func = QUADRANT_REGIONS_TEMPLATE.get(qn)
                if template_func:
                    exit_regions_coords.append(template_func(W, H)) # (qx1, qy1, qx2, qy2)
            
            if not exit_regions_coords:
                continue # No valid quadrant regions for this rule direction

            # Iterate over each tracked object from this camera's frame
            for track_data_row in tracked_dets_np:
                # Ensure track_data_row has at least 5 elements (x1,y1,x2,y2, track_id)
                if len(track_data_row) < 5: continue 
                try:
                    track_id_val = track_data_row[4]
                    if not np.isfinite(track_id_val) or track_id_val < 0: continue
                    track_id_int = TrackID(int(track_id_val))

                    # If this track_id already triggered a handoff in this frame (by another rule), skip
                    if track_id_int in processed_track_ids_this_cam_frame:
                        continue

                    bbox_xyxy_list = BoundingBoxXYXY(list(map(float, track_data_row[0:4])))
                    x1, y1, x2, y2 = bbox_xyxy_list
                    bbox_w, bbox_h = x2 - x1, y2 - y1
                    if bbox_w <= 0 or bbox_h <= 0: continue # Invalid bbox
                    bbox_area = float(bbox_w * bbox_h)

                    # Calculate total intersection area of the bbox with all relevant exit quadrants for this rule
                    total_intersection_area = 0.0
                    for qx1, qy1, qx2, qy2 in exit_regions_coords:
                        # Calculate intersection rectangle
                        inter_x1, inter_y1 = max(x1, float(qx1)), max(y1, float(qy1))
                        inter_x2, inter_y2 = min(x2, float(qx2)), min(y2, float(qy2))
                        inter_w = max(0.0, inter_x2 - inter_x1) # Ensure non-negative width
                        inter_h = max(0.0, inter_y2 - inter_y1) # Ensure non-negative height
                        total_intersection_area += float(inter_w * inter_h)
                    
                    # Check if overlap ratio meets threshold
                    if bbox_area > 1e-5 and (total_intersection_area / bbox_area) >= min_overlap_ratio:
                        source_track_key: TrackKey = (camera_id, track_id_int)
                        trigger_info = HandoffTriggerInfo(
                            source_track_key=source_track_key, rule=rule, source_bbox=bbox_xyxy_list
                        )
                        triggers_found.append(trigger_info)
                        processed_track_ids_this_cam_frame.add(track_id_int) # Mark this track as processed for handoff in this frame
                        logger.info(
                            f"[Task {task_id}][{camera_id}] HANDOFF TRIGGER: Track {source_track_key} "
                            f"matched rule '{rule.direction}' -> Cam '{rule.target_cam_id}' "
                            f"Area '{rule.target_entry_area}'. Overlap: {total_intersection_area/bbox_area:.2f}"
                        )
                        break # Move to the next track, this one has triggered
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(
                        f"[Task {task_id}][{camera_id}] Error processing track for handoff: {track_data_row}. Error: {e}", exc_info=False
                    )
        return triggers_found


    async def process_frame_batch(
        self,
        task_id: uuid.UUID,
        environment_id: str,
        reid_manager: ReIDStateManager,
        frame_batch: FrameBatch,
        processed_frame_count: int # This is the global batch index
    ) -> Dict[CameraID, List[TrackedObjectData]]:
        """Processes a batch of frames with Re-ID, handoff detection, and map projection."""
        batch_start_time = asyncio.get_event_loop().time()
        # Detector is assumed to be loaded via on_startup.
        # Log that we are starting to process a batch.
        logger.info(f"[Task {task_id}][Env {environment_id}] MFProc: Processing batch for global frame count {processed_frame_count}.")

        aggregated_parsed_track_data_this_batch: Dict[TrackKey, Tuple[BoundingBoxXYXY, Optional[FeatureVector]]] = {}
        active_track_keys_this_batch_set: Set[TrackKey] = set()
        all_handoff_triggers_this_batch: List[HandoffTriggerInfo] = []
        camera_frame_shapes: Dict[CameraID, Tuple[int, int]] = {} # Stores (H, W) for each camera's frame
        confidences_map: Dict[TrackKey, float] = {}

        # --- Detection Phase (Parallel for all cameras in batch) ---
        per_camera_detections_tasks = []
        valid_cam_ids_in_batch = [cam_id for cam_id, data in frame_batch.items() if data is not None]

        # Helper coroutine for detection on a single camera's frame
        async def detect_for_camera(cam_id_local: CameraID, frame_image_np_local: np.ndarray) -> Tuple[CameraID, List[RawDetection]]:
            raw_model_detections = await self.detector.detect(frame_image_np_local) # self.detector is pre-loaded
            converted_detections = [
                RawDetection(
                    bbox_xyxy=BoundingBoxXYXY(d.bbox.to_list()), 
                    confidence=d.confidence, 
                    class_id=d.class_id
                ) for d in raw_model_detections
            ]
            return cam_id_local, converted_detections

        for cam_id_loop in valid_cam_ids_in_batch:
            frame_data_loop = frame_batch[cam_id_loop] # Should be Some, based on valid_cam_ids_in_batch
            if frame_data_loop: # Should always be true here
                frame_image_np_loop, _ = frame_data_loop
                camera_frame_shapes[cam_id_loop] = frame_image_np_loop.shape[:2] # Store H, W
                per_camera_detections_tasks.append(detect_for_camera(cam_id_loop, frame_image_np_loop))
        
        detection_results: List[Any] = await asyncio.gather(*per_camera_detections_tasks, return_exceptions=True)

        # --- Tracking, Feature Parsing, and Handoff Trigger Detection Phase (Iterate through detection results) ---
        for i, det_res_or_exc in enumerate(detection_results):
            cam_id_from_det_task_order = valid_cam_ids_in_batch[i] # Get cam_id based on task submission order
            
            if isinstance(det_res_or_exc, Exception):
                logger.error(f"[Task {task_id}][{cam_id_from_det_task_order}] Detection failed: {det_res_or_exc}", exc_info=True)
                continue # Skip this camera for this batch if detection failed
            
            # If no exception, unpack result
            cam_id, raw_detections_list = det_res_or_exc
            frame_data = frame_batch.get(cam_id) # Should exist
            if not frame_data: continue # Should not happen if valid_cam_ids_in_batch was correct
            frame_image_np, _ = frame_data
            current_frame_shape = camera_frame_shapes.get(cam_id) # Get stored H, W
            if not current_frame_shape:
                logger.warning(f"[Task {task_id}][{cam_id}] Frame shape missing for cam_id. Skipping tracking & handoff.")
                continue

            # Convert RawDetections to NumPy array for tracker
            detections_np_for_tracker = np.array(
                [(*d.bbox_xyxy, d.confidence, d.class_id) for d in raw_detections_list], dtype=np.float32
            ) if raw_detections_list else np.empty((0, 6))

            try:
                tracker_instance = await self.tracker_factory.get_tracker(task_id, cam_id) # Factory manages preloading via prototype
                raw_tracker_output_np: np.ndarray = await tracker_instance.update(detections_np_for_tracker, frame_image_np)
                
                # Parse tracker output to extract track_key, bbox, and features
                parsed_tracks_this_camera = self._parse_raw_tracker_output(task_id, cam_id, raw_tracker_output_np)
                for track_key, bbox, feature in parsed_tracks_this_camera:
                    aggregated_parsed_track_data_this_batch[track_key] = (bbox, feature)
                    active_track_keys_this_batch_set.add(track_key)
                    # Store confidence if available in raw_tracker_output_np (BotSort: x,y,x,y,id,conf,cls,feat)
                    original_track_row = next((row for row in raw_tracker_output_np if int(row[4]) == track_key[1]), None)
                    if original_track_row is not None and len(original_track_row) > 5: # index 5 for confidence
                        confidences_map[track_key] = float(original_track_row[5])


                # Perform handoff trigger check if there were any tracks
                if raw_tracker_output_np.size > 0: # Check if tracker produced any output
                    handoff_triggers_this_cam = self._check_handoff_triggers_for_camera(
                        task_id, environment_id, cam_id, raw_tracker_output_np, current_frame_shape
                    )
                    all_handoff_triggers_this_batch.extend(handoff_triggers_this_cam)

            except Exception as e:
                logger.error(f"[Task {task_id}][{cam_id}] Error during tracking, parsing or handoff check: {e}", exc_info=True)
        
        # --- Re-ID Association Phase ---
        # Collect all features from successfully tracked objects this batch
        features_for_reid_input: Dict[TrackKey, FeatureVector] = {
            tk: feat for tk, (_, feat) in aggregated_parsed_track_data_this_batch.items() if feat is not None
        }
        # Map active triggers for quick lookup during Re-ID
        active_triggers_map_for_reid: Dict[TrackKey, HandoffTriggerInfo] = {
            trigger.source_track_key: trigger for trigger in all_handoff_triggers_this_batch
        }

        if features_for_reid_input or active_track_keys_this_batch_set: # If any features or any active tracks
            await reid_manager.associate_features_and_update_state(
                features_for_reid_input, # Only tracks with features
                active_track_keys_this_batch_set, # All tracks that were active (even if no feature)
                active_triggers_map_for_reid,
                processed_frame_count # Global frame/batch counter for lifecycle management
            )
        else:
            logger.debug(f"[Task {task_id}] MFProc: No features or active tracks for ReID for frame count {processed_frame_count}.")

        # --- Construct Final Output for this Batch (including map projection) ---
        final_batch_results: Dict[CameraID, List[TrackedObjectData]] = defaultdict(list)
        for track_key, (bbox_xyxy, original_feature) in aggregated_parsed_track_data_this_batch.items():
            cam_id_active, track_id_active = track_key
            gid_assigned: Optional[GlobalID] = reid_manager.track_to_global_id.get(track_key)
            
            # Map projection using HomographyService
            map_coords_output: Optional[List[float]] = None
            # Retrieve pre-computed homography matrix
            homography_matrix_current_cam = self.homography_service.get_homography_matrix(environment_id, cam_id_active)
            if homography_matrix_current_cam is not None:
                foot_point_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
                foot_point_y = bbox_xyxy[3] 
                map_coords_output = self._project_point_to_map((foot_point_x, foot_point_y), homography_matrix_current_cam)

            track_obj_data = TrackedObjectData(
                camera_id=cam_id_active,
                track_id=track_id_active,
                global_person_id=gid_assigned,
                bbox_xyxy=bbox_xyxy,
                confidence=confidences_map.get(track_key), # Get stored confidence
                feature_vector=list(original_feature) if original_feature is not None else None,
                map_coords=map_coords_output
            )
            final_batch_results[cam_id_active].append(track_obj_data)

        logger.info(
            f"[Task {task_id}] MFProc: Batch for frame count {processed_frame_count} finished in "
            f"{(asyncio.get_event_loop().time() - batch_start_time):.3f}s. "
            f"Active tracks: {len(active_track_keys_this_batch_set)}. "
            f"Features for ReID: {len(features_for_reid_input)}. "
            f"Handoff Triggers: {len(all_handoff_triggers_this_batch)}."
        )
        return dict(final_batch_results) # Convert defaultdict to dict for safety

    async def clear_task_resources(self, task_id: uuid.UUID, environment_id: str):
        """
        Placeholder for any specific cleanup MultiCameraFrameProcessor might need per task.
        Homography matrices are global, managed by HomographyService, not cleared per task here.
        Detector is global. Tracker instances are cleared by CameraTrackerFactory.
        """
        logger.info(f"[Task {task_id}][Env {environment_id}] MultiCameraFrameProcessor: No task-specific resources to clear directly in this processor for now.")