"""
Service for processing a batch of frames from multiple cameras simultaneously.
This involves detection, per-camera tracking, handoff trigger detection, and map projection.
"""
import asyncio
import logging
import uuid
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2 

from app.models.base_models import AbstractDetector
from app.shared.types import (
    CameraID, TrackKey, GlobalID, FeatureVector, FrameBatch, RawDetection,
    TrackedObjectData, BoundingBoxXYXY, TrackID,
    HandoffTriggerInfo, ExitRuleModel, CameraHandoffDetailConfig,
    QUADRANT_REGIONS_TEMPLATE # MODIFIED: DIRECTION_TO_QUADRANTS_MAP removed
)
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.notification_service import NotificationService
from app.services.homography_service import HomographyService 
from app.services.geometric import (
    BottomPointExtractor,
    ImagePoint,
    WorldPlaneTransformer,
    WorldPoint,
    ROICalculator,
    ROIShape,
    GeometricMatcher,
    MetricsCollector,
    InverseHomographyProjector,
    ProjectedImagePoint,
    DebugOverlay,
    ReprojectionDebugger,
)
from app.core.config import settings
from app.tracing import analytics_event_tracer

logger = logging.getLogger(__name__)

class MultiCameraFrameProcessor:
    """
    Orchestrates processing for a batch of frames, one from each camera,
    performing detection, per-camera tracking, handoff detection, and map projection.
    Relies on pre-loaded detector and homography matrices.
    """

    def __init__(
        self,
        detector: AbstractDetector,
        tracker_factory: CameraTrackerFactory,
        homography_service: HomographyService, 
        notification_service: NotificationService,
        device: torch.device,
        bottom_point_extractor: Optional[BottomPointExtractor] = None,
        world_plane_transformer: Optional[WorldPlaneTransformer] = None,
    ):
        self.detector = detector 
        self.tracker_factory = tracker_factory
        self.homography_service = homography_service 
        self.notification_service = notification_service
        self.device = device
        self.bottom_point_extractor = bottom_point_extractor or BottomPointExtractor(
            validation_enabled=getattr(settings, "ENABLE_POINT_VALIDATION", True)
        )
        self.world_plane_transformer = world_plane_transformer or WorldPlaneTransformer.from_settings()
        roi_shape_value = str(getattr(settings, "ROI_SHAPE", "circular")).lower()
        self.roi_shape = ROIShape._value2member_map_.get(roi_shape_value, ROIShape.CIRCULAR)
        self.roi_calculator = ROICalculator(
            base_radius=getattr(settings, "ROI_BASE_RADIUS", 1.5),
            max_walking_speed=getattr(settings, "ROI_MAX_WALKING_SPEED", 1.5),
            min_radius=getattr(settings, "ROI_MIN_RADIUS", 0.5),
            max_radius=getattr(settings, "ROI_MAX_RADIUS", 10.0),
        )
        self.geometric_matcher = GeometricMatcher(
            exact_match_confidence=getattr(settings, "EXACT_MATCH_CONFIDENCE", 0.95),
            closest_match_confidence=getattr(settings, "CLOSEST_MATCH_CONFIDENCE", 0.70),
            distance_penalty_factor=getattr(settings, "DISTANCE_PENALTY_FACTOR", 0.1),
        )
        self.metrics_collector = MetricsCollector(
            high_confidence_threshold=getattr(settings, "HIGH_CONFIDENCE_THRESHOLD", 0.8),
        )

        self.enable_debug_reprojection = bool(getattr(settings, "ENABLE_DEBUG_REPROJECTION", False))
        self.inverse_projector: Optional[InverseHomographyProjector] = None
        self.debug_overlay: Optional[DebugOverlay] = None
        self.reprojection_debugger: Optional[ReprojectionDebugger] = None
        self._frame_store: Dict[str, np.ndarray] = {}

        if self.enable_debug_reprojection:
            try:
                self.debug_overlay = DebugOverlay(
                    radius_px=getattr(settings, "DEBUG_OVERLAY_RADIUS_PX", 6)
                )
                self.reprojection_debugger = ReprojectionDebugger(
                    frame_provider=self._provide_frame_for_debug,
                    output_dir=getattr(settings, "DEBUG_REPROJECTION_OUTPUT_DIR", "app/debug_outputs"),
                    sampling_rate=getattr(settings, "DEBUG_FRAME_SAMPLING_RATE", 1),
                    max_frames_per_camera=getattr(settings, "DEBUG_MAX_FRAMES_PER_CAMERA", 500),
                )
            except Exception as exc:
                logger.warning("Failed to initialize reprojection debugger: %s", exc)
                self.enable_debug_reprojection = False

            if self.enable_debug_reprojection and self.world_plane_transformer:
                try:
                    self.inverse_projector = InverseHomographyProjector(
                        world_plane_transformer=self.world_plane_transformer
                    )
                except Exception as exc:
                    logger.warning("Failed to initialize inverse projector: %s", exc)
                    self.inverse_projector = None

        logger.info("MultiCameraFrameProcessor initialized (expects pre-loaded components).")


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

    async def _emit_geometric_metrics(self, environment_id: str) -> None:
        """Publish aggregated geometric metrics for observability."""
        try:
            extraction_stats = self.bottom_point_extractor.get_statistics()
            transformation_stats = (
                self.world_plane_transformer.get_statistics()
                if self.world_plane_transformer
                else None
            )
            roi_stats = self.roi_calculator.get_statistics()
            metrics_summary = self.metrics_collector.get_metrics().to_dict()
            await analytics_event_tracer.record_geometric_metrics(
                environment_id=environment_id,
                camera_id=None,
                extraction_stats=extraction_stats,
                transformation_stats=transformation_stats,
                roi_stats=roi_stats,
                matcher_stats=self.geometric_matcher.get_statistics(),
                metrics_summary=metrics_summary,
            )
        except Exception as exc:
            logger.debug("MultiCameraFrameProcessor metrics emission failed: %s", exc)

    def _apply_geometric_matching(
        self,
        final_batch_results: Dict[CameraID, List[TrackedObjectData]],
        processed_frame_count: int,
        camera_frame_shapes: Dict[CameraID, Tuple[int, int]],
        camera_homographies: Dict[CameraID, Optional[np.ndarray]],
        environment_id: str,
    ) -> None:
        """Perform cross-camera geometric matching using ROIs and world coordinates."""
        min_confidence = getattr(settings, "MIN_MATCH_CONFIDENCE", 0.5)
        if len(final_batch_results) < 2:
            return

        timestamp_value = float(processed_frame_count)
        debug_enabled = bool(self.enable_debug_reprojection and self.reprojection_debugger and self.debug_overlay)

        candidates_by_camera: Dict[CameraID, List[WorldPoint]] = {}
        track_lookup_by_camera: Dict[str, Dict[int, TrackedObjectData]] = {}

        for cam_id, track_data in final_batch_results.items():
            candidate_points: List[WorldPoint] = []
            track_lookup: Dict[int, TrackedObjectData] = {}

            for track in track_data:
                try:
                    track_lookup[int(track.track_id)] = track
                except (TypeError, ValueError):
                    continue

                if track.map_coords is None:
                    continue

                try:
                    candidate_points.append(
                        WorldPoint(
                            x=float(track.map_coords[0]),
                            y=float(track.map_coords[1]),
                            camera_id=str(cam_id),
                            person_id=int(track.track_id),
                            original_image_point=(0.0, 0.0),
                            frame_number=processed_frame_count,
                            timestamp=timestamp_value,
                            transformation_quality=float(track.transformation_quality or 1.0),
                        )
                    )
                except Exception as exc:
                    logger.debug(
                        "Skipping candidate creation for camera %s track %s: %s",
                        cam_id,
                        track.track_id,
                        exc,
                    )

            candidates_by_camera[cam_id] = candidate_points
            track_lookup_by_camera[str(cam_id)] = track_lookup

        for source_camera, tracks in final_batch_results.items():
            for track in tracks:
                if track.map_coords is None or not track.search_roi:
                    continue

                try:
                    roi_shape = ROIShape(track.search_roi.get("shape", "circular"))
                    roi = SearchROI(
                        center=(float(track.search_roi["center_x"]), float(track.search_roi["center_y"])),
                        radius=float(track.search_roi.get("radius") or 0.0),
                        width=track.search_roi.get("width"),
                        height=track.search_roi.get("height"),
                        shape=roi_shape,
                        source_camera=str(source_camera),
                        dest_camera=None,
                        person_id=int(track.track_id),
                        timestamp=track.search_roi.get("timestamp"),
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to rehydrate ROI for camera %s track %s: %s",
                        source_camera,
                        track.track_id,
                        exc,
                    )
                    continue

                source_world_point = WorldPoint(
                    x=float(track.map_coords[0]),
                    y=float(track.map_coords[1]),
                    camera_id=str(source_camera),
                    person_id=int(track.track_id),
                    original_image_point=(0.0, 0.0),
                    frame_number=processed_frame_count,
                    timestamp=timestamp_value,
                    transformation_quality=float(track.transformation_quality or 1.0),
                )

                best_match = None
                best_confidence = -1.0

                for dest_camera, candidates in candidates_by_camera.items():
                    if dest_camera == source_camera or not candidates:
                        continue

                    dest_camera_str = str(dest_camera)
                    roi.dest_camera = dest_camera_str

                    start_time = time.perf_counter()
                    match_result = self.geometric_matcher.match_person(
                        source_world_point,
                        candidates,
                        roi,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                    self.metrics_collector.record_match(
                        match_result=match_result,
                        processing_time_ms=elapsed_ms,
                        transformation_quality=source_world_point.transformation_quality,
                    )

                    if debug_enabled:
                        projected_point = None
                        if self.inverse_projector:
                            projected_point = self.inverse_projector.project(source_world_point, dest_camera_str)
                        else:
                            projected_point = self._project_world_to_image(
                                dest_camera=dest_camera_str,
                                world_point=source_world_point,
                                camera_homographies=camera_homographies,
                                environment_id=environment_id,
                            )

                        if projected_point:
                            actual_point_tuple: Optional[Tuple[float, float]] = None
                            if match_result.matched_person_id is not None:
                                dest_lookup = track_lookup_by_camera.get(dest_camera_str, {})
                                matched_track = dest_lookup.get(match_result.matched_person_id)
                                if matched_track:
                                    frame_shape = camera_frame_shapes.get(dest_camera)
                                    actual_point_tuple = self._bottom_center_from_bbox(
                                        matched_track.bbox_xyxy,
                                        frame_shape,
                                    )

                            has_actual = actual_point_tuple is not None
                            self.metrics_collector.record_reprojection_event(has_actual)
                            if has_actual and actual_point_tuple is not None:
                                error_px = self._compute_pixel_error(projected_point, actual_point_tuple)
                                projected_point.reprojection_error_px = error_px
                                self.metrics_collector.record_reprojection_error(error_px)

                            if self.reprojection_debugger and self.debug_overlay:
                                self.reprojection_debugger.emit(
                                    dest_camera_str,
                                    processed_frame_count,
                                    self.debug_overlay,
                                    projected_point,
                                    actual_point_tuple,
                                )

                    if not match_result.is_successful():
                        continue

                    if match_result.confidence < min_confidence:
                        continue

                    if match_result.confidence > best_confidence:
                        best_confidence = match_result.confidence
                        best_match = match_result

                if not best_match:
                    continue

                track.geometric_match = {
                    "role": "source",
                    "dest_camera": best_match.dest_camera,
                    "matched_track_id": best_match.matched_person_id,
                    "match_type": best_match.match_type.value,
                    "confidence": best_match.confidence,
                    "distance_m": best_match.spatial_distance,
                    "candidates_in_roi": best_match.candidates_in_roi,
                    "roi_radius": best_match.roi_radius,
                }

                dest_camera_id = best_match.dest_camera
                if not dest_camera_id:
                    continue

                dest_tracks = final_batch_results.get(CameraID(dest_camera_id))
                if dest_tracks:
                    for dest_track in dest_tracks:
                        if int(dest_track.track_id) == best_match.matched_person_id:
                            dest_track.geometric_match = {
                                "role": "target",
                                "source_camera": str(source_camera),
                                "source_track_id": int(track.track_id),
                                "match_type": best_match.match_type.value,
                                "confidence": best_match.confidence,
                                "distance_m": best_match.spatial_distance,
                                "candidates_in_roi": best_match.candidates_in_roi,
                                "roi_radius": best_match.roi_radius,
                            }
                            break

    @staticmethod
    def _compute_pixel_error(predicted_point: ProjectedImagePoint, actual_point: Tuple[float, float]) -> float:
        dx = predicted_point.x - actual_point[0]
        dy = predicted_point.y - actual_point[1]
        return float((dx ** 2 + dy ** 2) ** 0.5)

    @staticmethod
    def _bottom_center_from_bbox(
        bbox_xyxy: BoundingBoxXYXY,
        frame_shape: Optional[Tuple[int, int]],
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        center_x = (x1 + x2) / 2.0
        bottom_y = y2

        if frame_shape:
            height, width = frame_shape
            center_x = max(0.0, min(center_x, width - 1))
            bottom_y = max(0.0, min(bottom_y, height - 1))

        return (center_x, bottom_y)

    def _project_world_to_image(
        self,
        dest_camera: str,
        world_point: WorldPoint,
        camera_homographies: Dict[CameraID, Optional[np.ndarray]],
        environment_id: str,
    ) -> Optional[ProjectedImagePoint]:
        cam_id = CameraID(dest_camera)
        matrix = camera_homographies.get(cam_id)

        if matrix is None and self.homography_service:
            try:
                matrix = self.homography_service.get_homography_matrix(environment_id, cam_id)
                camera_homographies[cam_id] = matrix
            except Exception:
                matrix = None

        if matrix is None:
            return None

        try:
            inverse_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None

        input_world = np.array([[[world_point.x, world_point.y]]], dtype=np.float32)
        try:
            projected = cv2.perspectiveTransform(input_world, inverse_matrix)
        except cv2.error:
            return None

        x_px = float(projected[0, 0, 0])
        y_px = float(projected[0, 0, 1])

        if not np.isfinite(x_px) or not np.isfinite(y_px):
            return None

        return ProjectedImagePoint(
            x=x_px,
            y=y_px,
            camera_id=dest_camera,
            person_id=world_point.person_id,
            source_camera_id=str(world_point.camera_id),
            world_point=(world_point.x, world_point.y),
            frame_number=world_point.frame_number,
            timestamp=world_point.timestamp,
        )

    def _provide_frame_for_debug(self, camera_id: str, frame_number: int) -> Optional[np.ndarray]:
        frame = self._frame_store.get(camera_id)
        if frame is None:
            return None
        return frame.copy()

    def _parse_raw_tracker_output(
        self, task_id: uuid.UUID, camera_id: CameraID, tracker_output_np: np.ndarray
    ) -> List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]]:
        """Parses raw tracker output (BoxMOT format with features)."""
        parsed_tracks: List[Tuple[TrackKey, BoundingBoxXYXY, Optional[FeatureVector]]] = []
        if tracker_output_np is None or tracker_output_np.size == 0:
            return parsed_tracks

        num_cols = tracker_output_np.shape[1]
        
        if num_cols < 7: 
            if num_cols >= 5: 
                 logger.debug(f"[Task {task_id}][{camera_id}] Tracker output has {num_cols} columns. Expected >= 7 for track data + features. Features might be missing or tracker output format changed.")
            else:
                 logger.warning(f"[Task {task_id}][{camera_id}] Tracker output has {num_cols} columns. Expected >= 5 for minimal track data (xyxy, id). Data possibly corrupt.")
                 return parsed_tracks

        for row_idx, row in enumerate(tracker_output_np):
            try:
                x1, y1, x2, y2 = map(float, row[0:4])
                track_id_val = row[4] 
                
                if not np.isfinite(track_id_val) or track_id_val < 0: 
                    continue 
                track_id_int = int(track_id_val)

                if x2 <= x1 or y2 <= y1: 
                    continue

                bbox_xyxy = BoundingBoxXYXY([x1, y1, x2, y2])
                track_key: TrackKey = (camera_id, TrackID(track_id_int))
                
                feature_vector: Optional[FeatureVector] = None
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

    def _check_handoff_triggers_for_camera(
        self,
        task_id: uuid.UUID,
        environment_id: str,
        camera_id: CameraID,
        tracked_dets_np: np.ndarray, 
        frame_shape: Tuple[int, int] 
    ) -> List[HandoffTriggerInfo]:
        """
        Checks if active tracks in a camera's frame overlap significantly with predefined exit quadrants
        as specified in the `ExitRuleModel.source_exit_quadrant`.
        """
        triggers_found: List[HandoffTriggerInfo] = []
        cam_detail_key = (environment_id, str(camera_id)) 
        cam_handoff_config: Optional[CameraHandoffDetailConfig] = settings.CAMERA_HANDOFF_DETAILS.get(cam_detail_key)

        if not cam_handoff_config or not cam_handoff_config.exit_rules:
            return triggers_found 
        
        exit_rules: List[ExitRuleModel] = cam_handoff_config.exit_rules
        min_overlap_ratio = settings.MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT

        if min_overlap_ratio <= 0: 
            return triggers_found

        H, W = frame_shape
        if H <= 0 or W <= 0:
            logger.warning(f"[Task {task_id}][{camera_id}] Invalid frame shape {frame_shape} for handoff check.")
            return triggers_found

        processed_track_ids_this_cam_frame = set()

        # Pre-extract bbox and track arrays for vectorized ops
        if tracked_dets_np is None or tracked_dets_np.size == 0:
            return triggers_found
        valid_rows = tracked_dets_np[:, 0:5]
        x1_arr = valid_rows[:, 0].astype(float)
        y1_arr = valid_rows[:, 1].astype(float)
        x2_arr = valid_rows[:, 2].astype(float)
        y2_arr = valid_rows[:, 3].astype(float)
        track_id_vals = valid_rows[:, 4]

        # Filter valid tracks
        finite_mask = np.isfinite(track_id_vals) & (track_id_vals >= 0)
        # Positive area mask
        pos_area_mask = (x2_arr > x1_arr) & (y2_arr > y1_arr)
        base_mask = finite_mask & pos_area_mask

        for rule in exit_rules:
            source_quadrant_name = rule.source_exit_quadrant
            quadrant_template_func = QUADRANT_REGIONS_TEMPLATE.get(source_quadrant_name)
            if not quadrant_template_func:
                logger.warning(f"[Task {task_id}][{camera_id}] Rule specifies unknown source_exit_quadrant '{source_quadrant_name}'. Skipping rule.")
                continue

            qx1, qy1, qx2, qy2 = quadrant_template_func(W, H)

            # Vectorized intersection with quadrant
            inter_x1 = np.maximum(x1_arr, float(qx1))
            inter_y1 = np.maximum(y1_arr, float(qy1))
            inter_x2 = np.minimum(x2_arr, float(qx2))
            inter_y2 = np.minimum(y2_arr, float(qy2))
            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            bbox_area = (x2_arr - x1_arr) * (y2_arr - y1_arr)

            overlap_ratio = np.divide(inter_area, bbox_area, out=np.zeros_like(inter_area), where=bbox_area > 1e-5)
            mask = base_mask & (overlap_ratio >= min_overlap_ratio)

            indices = np.where(mask)[0]
            for idx in indices:
                tid_val = track_id_vals[idx]
                try:
                    track_id_int = TrackID(int(tid_val))
                    if track_id_int in processed_track_ids_this_cam_frame:
                        continue
                    bbox_xyxy_list = BoundingBoxXYXY([x1_arr[idx], y1_arr[idx], x2_arr[idx], y2_arr[idx]])
                    source_track_key: TrackKey = (camera_id, track_id_int)
                    trigger_info = HandoffTriggerInfo(
                        source_track_key=source_track_key, rule=rule, source_bbox=bbox_xyxy_list
                    )
                    triggers_found.append(trigger_info)
                    processed_track_ids_this_cam_frame.add(track_id_int)
                    logger.info(
                        f"[Task {task_id}][{camera_id}] HANDOFF TRIGGER: Track {source_track_key} "
                        f"exiting quadrant '{rule.source_exit_quadrant}' -> Cam '{rule.target_cam_id}' "
                        f"Area '{rule.target_entry_area}'. Overlap: {float(overlap_ratio[idx]):.2f}"
                    )
                except Exception as e:
                    logger.warning(f"[Task {task_id}][{camera_id}] Error processing vectorized handoff idx {idx}: {e}", exc_info=False)
        return triggers_found


    async def process_frame_batch(
        self,
        task_id: uuid.UUID,
        environment_id: str,
        assoc_manager: object,
        frame_batch: FrameBatch,
        processed_frame_count: int 
    ) -> Dict[CameraID, List[TrackedObjectData]]:
        """Processes a batch of frames with Re-ID, handoff detection, and map projection."""
        batch_start_time = asyncio.get_event_loop().time()
        
        logger.info(f"[Task {task_id}][Env {environment_id}] MFProc: Processing batch for global frame count {processed_frame_count}.")

        aggregated_parsed_track_data_this_batch: Dict[TrackKey, Tuple[BoundingBoxXYXY, Optional[FeatureVector]]] = {}
        active_track_keys_this_batch_set: Set[TrackKey] = set()
        all_handoff_triggers_this_batch: List[HandoffTriggerInfo] = []
        camera_frame_shapes: Dict[CameraID, Tuple[int, int]] = {} 
        camera_homographies: Dict[CameraID, Optional[np.ndarray]] = {}
        confidences_map: Dict[TrackKey, float] = {}

        per_camera_detections_tasks = []
        valid_cam_ids_in_batch = [cam_id for cam_id, data in frame_batch.items() if data is not None]

        if self.enable_debug_reprojection:
            self._frame_store = {}
            for cam_id_loop in valid_cam_ids_in_batch:
                frame_data_loop = frame_batch.get(cam_id_loop)
                if frame_data_loop and frame_data_loop[0] is not None:
                    self._frame_store[str(cam_id_loop)] = frame_data_loop[0]
        else:
            self._frame_store = {}

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
                camera_frame_shapes[cam_id_loop] = frame_image_np_loop.shape[:2] 
                per_camera_detections_tasks.append(detect_for_camera(cam_id_loop, frame_image_np_loop))
        
        detection_results: List[Any] = await asyncio.gather(*per_camera_detections_tasks, return_exceptions=True)

        for i, det_res_or_exc in enumerate(detection_results):
            cam_id_from_det_task_order = valid_cam_ids_in_batch[i] 
            
            if isinstance(det_res_or_exc, Exception):
                logger.error(f"[Task {task_id}][{cam_id_from_det_task_order}] Detection failed: {det_res_or_exc}", exc_info=True)
                continue 
            
            cam_id, raw_detections_list = det_res_or_exc
            frame_data = frame_batch.get(cam_id) 
            if not frame_data: continue 
            frame_image_np, _ = frame_data
            current_frame_shape = camera_frame_shapes.get(cam_id) 
            if not current_frame_shape:
                logger.warning(f"[Task {task_id}][{cam_id}] Frame shape missing for cam_id. Skipping tracking & handoff.")
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
                    original_track_row = next((row for row in raw_tracker_output_np if int(row[4]) == track_key[1]), None)
                    if original_track_row is not None and len(original_track_row) > 5: 
                        confidences_map[track_key] = float(original_track_row[5])

                if raw_tracker_output_np.size > 0: 
                    handoff_triggers_this_cam = self._check_handoff_triggers_for_camera(
                        task_id, environment_id, cam_id, raw_tracker_output_np, current_frame_shape
                    )
                    all_handoff_triggers_this_batch.extend(handoff_triggers_this_cam)

            except Exception as e:
                logger.error(f"[Task {task_id}][{cam_id}] Error during tracking, parsing or handoff check: {e}", exc_info=True)
        
        features_for_assoc_input: Dict[TrackKey, FeatureVector] = {
            tk: feat for tk, (_, feat) in aggregated_parsed_track_data_this_batch.items() if feat is not None
        }
        active_triggers_map_for_assoc: Dict[TrackKey, HandoffTriggerInfo] = {
            trigger.source_track_key: trigger for trigger in all_handoff_triggers_this_batch
        }

        if features_for_assoc_input or active_track_keys_this_batch_set: 
            await assoc_manager.associate_features_and_update_state(
                features_for_assoc_input, 
                active_track_keys_this_batch_set, 
                active_triggers_map_for_assoc,
                processed_frame_count 
            )
        else:
            logger.debug(f"[Task {task_id}] MFProc: No features or active tracks for association for frame count {processed_frame_count}.")

        final_batch_results: Dict[CameraID, List[TrackedObjectData]] = defaultdict(list)
        for track_key, (bbox_xyxy, original_feature) in aggregated_parsed_track_data_this_batch.items():
            cam_id_active, track_id_active = track_key
            gid_assigned: Optional[GlobalID] = assoc_manager.track_to_global_id.get(track_key)
            
            map_coords_output: Optional[List[float]] = None
            transformation_quality: Optional[float] = None
            homography_matrix_current_cam = self.homography_service.get_homography_matrix(environment_id, cam_id_active)
            camera_homographies[cam_id_active] = homography_matrix_current_cam
            frame_shape_for_cam = camera_frame_shapes.get(cam_id_active)
            bottom_point: Optional[ImagePoint] = None

            if frame_shape_for_cam:
                frame_height, frame_width = frame_shape_for_cam
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_xyxy
                bbox_width = bbox_x2 - bbox_x1
                bbox_height = bbox_y2 - bbox_y1

                try:
                    bottom_point = self.bottom_point_extractor.extract_point(
                        bbox_x=bbox_x1,
                        bbox_y=bbox_y1,
                        bbox_width=bbox_width,
                        bbox_height=bbox_height,
                        camera_id=cam_id_active,
                        person_id=track_id_active,
                        frame_number=processed_frame_count,
                        timestamp=None,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                except ValueError as exc:
                    logger.debug(
                        "[Task %s][%s] Bottom point extraction failed for track %s: %s",
                        task_id,
                        cam_id_active,
                        track_id_active,
                        exc,
                    )

            if self.world_plane_transformer and bottom_point:
                try:
                    world_point = self.world_plane_transformer.transform_point(bottom_point)
                    map_coords_output = [world_point.x, world_point.y]
                    transformation_quality = world_point.transformation_quality
                except (KeyError, ValueError) as exc:
                    logger.debug(
                        "[Task %s][%s] World-plane transform failed for track %s: %s",
                        task_id,
                        cam_id_active,
                        track_id_active,
                        exc,
                    )

            if map_coords_output is None and homography_matrix_current_cam is not None and bottom_point:
                map_coords_output = self._project_point_to_map((bottom_point.x, bottom_point.y), homography_matrix_current_cam)
                transformation_quality = transformation_quality or 0.8

            search_roi_payload: Optional[Dict[str, Optional[float]]] = None
            if map_coords_output is not None:
                try:
                    roi = self.roi_calculator.calculate_roi(
                        (map_coords_output[0], map_coords_output[1]),
                        time_elapsed=0.0,
                        transformation_quality=transformation_quality if transformation_quality is not None else 1.0,
                        shape=self.roi_shape,
                        source_camera=str(cam_id_active),
                        dest_camera=None,
                        person_id=int(track_id_active),
                        timestamp=None,
                    )
                    search_roi_payload = roi.to_dict()
                except Exception as exc:
                    logger.debug(
                        "[Task %s][%s] ROI calculation failed for track %s: %s",
                        task_id,
                        cam_id_active,
                        track_id_active,
                        exc,
                    )

            track_obj_data = TrackedObjectData(
                camera_id=cam_id_active,
                track_id=track_id_active,
                global_person_id=gid_assigned,
                bbox_xyxy=bbox_xyxy,
                confidence=confidences_map.get(track_key), 
                feature_vector=list(original_feature) if original_feature is not None else None,
                map_coords=map_coords_output,
                search_roi=search_roi_payload,
                transformation_quality=transformation_quality,
            )
            final_batch_results[cam_id_active].append(track_obj_data)

        logger.info(
            f"[Task {task_id}] MFProc: Batch for frame count {processed_frame_count} finished in "
            f"{(asyncio.get_event_loop().time() - batch_start_time):.3f}s. "
            f"Active tracks: {len(active_track_keys_this_batch_set)}. "
            f"Features for association: {len(features_for_assoc_input)}. "
            f"Handoff Triggers: {len(all_handoff_triggers_this_batch)}."
        )
        self._apply_geometric_matching(
            final_batch_results,
            processed_frame_count,
            camera_frame_shapes,
            camera_homographies,
            environment_id,
        )
        if self.enable_debug_reprojection:
            self._frame_store.clear()
        await self._emit_geometric_metrics(environment_id)
        return dict(final_batch_results) 

    async def clear_task_resources(self, task_id: uuid.UUID, environment_id: str):
        """
        Placeholder for any specific cleanup MultiCameraFrameProcessor might need per task.
        """
        logger.info(f"[Task {task_id}][Env {environment_id}] MultiCameraFrameProcessor: No task-specific resources to clear directly in this processor for now.")
