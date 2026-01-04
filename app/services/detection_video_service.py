"""
Detection video service - integrates YOLO person detection with video streaming.

This service extends RawVideoService to add YOLO-based person detection capabilities.

Features:
- YOLO11-L person detection on video frames
- Inherits all raw video streaming capabilities  
- Basic detection processing and frame annotation
- WebSocket streaming of detection results
"""

import asyncio
import copy
from pathlib import Path
import uuid
import time
import logging
import math
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timezone
import numpy as np
import cv2
import json

from app.core.config import settings
from app.services.raw_video_service import RawVideoService
from app.models.yolo_detector import YOLODetector
from app.utils.detection_annotator import DetectionAnnotator
from app.utils.mjpeg_streamer import mjpeg_streamer
from app.api.websockets.connection_manager import binary_websocket_manager, MessageType
from app.api.websockets.focus_handler import focus_tracking_handler
from app.api.websockets.frame_handler import frame_handler
from app.services.homography_service import HomographyService
from app.services.handoff_detection_service import HandoffDetectionService
from app.services.trail_management_service import TrailManagementService
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.geometric import (
    BottomPointExtractor,
    ImagePoint,
    WorldPlaneTransformer,
    WorldPoint,
    ROICalculator,
    ROIShape,
    InverseHomographyProjector,
    ProjectedImagePoint,
    DebugOverlay,
    ReprojectionDebugger,
)
from app.services.space_based_matcher import SpaceBasedMatcher
from app.services.feature_extraction_service import FeatureExtractionService
from app.services.handoff_manager import HandoffManager
from app.services.global_person_registry import GlobalPersonRegistry
from app.tracing import analytics_event_tracer
from app.shared.types import CameraID, TrackID

logger = logging.getLogger(__name__)

# Dedicated file logger for [SPEED_DEBUG] timing logs ONLY
class SpeedDebugFilter(logging.Filter):
    """Filter to only allow [SPEED_DEBUG] messages"""
    def filter(self, record):
        return '[SPEED_DEBUG]' in record.getMessage()

speed_optimize_logger = logging.getLogger("speed_debug_pipeline")
speed_optimize_logger.setLevel(logging.INFO)
_speed_log_path = Path("speed_debug.log")
_speed_file_handler = logging.FileHandler(_speed_log_path, mode='a', encoding='utf-8')
_speed_file_handler.setLevel(logging.INFO)
_speed_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
_speed_file_handler.addFilter(SpeedDebugFilter())  # Only [SPEED_DEBUG] messages
speed_optimize_logger.addHandler(_speed_file_handler)
speed_optimize_logger.propagate = True  # Also show in console


class DetectionVideoService(RawVideoService):
    """
    Detection video service that extends RawVideoService with YOLO person detection.
    
    Phase 4 Implementation Features:
    - YOLO11-L person detection on video frames
    - Spatial intelligence with homography coordinate transformations
    - Camera handoff detection for cross-camera tracking
    - WebSocket streaming of enhanced detection results
    - Inherits all raw video capabilities from parent class
    """
    
    def __init__(
        self,
        *,
        homography_service: Optional[HomographyService] = None,
        tracker_factory: Optional[CameraTrackerFactory] = None,
        trail_service: Optional[TrailManagementService] = None,
        bottom_point_extractor: Optional[BottomPointExtractor] = None,
        world_plane_transformer: Optional[WorldPlaneTransformer] = None,
    ):
        super().__init__()
        
        # YOLO detector instance
        self.detector: Optional[YOLODetector] = None
        # Maintain detectors by environment to support separate weights
        self.detectors_by_env: Dict[str, YOLODetector] = {}
        # Track the resolved weights per environment for logging/debug
        self.detector_weights_by_env: Dict[str, str] = {}
        
        # Phase 2: Detection annotator for bounding box visualization
        self.annotator = DetectionAnnotator()
        
        # Phase 4: Spatial intelligence services
        self.homography_service: Optional[HomographyService] = homography_service
        self.handoff_service: Optional[HandoffDetectionService] = None
        
        # Trail management service for 2D mapping feature
        self.trail_service = trail_service or TrailManagementService(trail_length=getattr(settings, 'TRAIL_LENGTH', 3))

        # Phase 1: Bottom-center point extraction service
        self.bottom_point_extractor = bottom_point_extractor or BottomPointExtractor(
            validation_enabled=getattr(settings, "ENABLE_POINT_VALIDATION", True)
        )

        # Phase 2: World-plane transformer for geometric normalization
        self.world_plane_transformer = (
            world_plane_transformer or WorldPlaneTransformer.from_settings()
        )

        roi_shape_value = str(getattr(settings, "ROI_SHAPE", "circular")).lower()
        self.roi_shape = ROIShape._value2member_map_.get(roi_shape_value, ROIShape.CIRCULAR)
        self.roi_calculator = ROICalculator(
            base_radius=getattr(settings, "ROI_BASE_RADIUS", 1.5),
            max_walking_speed=getattr(settings, "ROI_MAX_WALKING_SPEED", 1.5),
            min_radius=getattr(settings, "ROI_MIN_RADIUS", 0.5),
            max_radius=getattr(settings, "ROI_MAX_RADIUS", 10.0),
        )
        self.enable_debug_reprojection = bool(getattr(settings, "ENABLE_DEBUG_REPROJECTION", False))
        self.debug_overlay: Optional[DebugOverlay] = None
        self.reprojection_debugger: Optional[ReprojectionDebugger] = None
        self.inverse_projector: Optional[InverseHomographyProjector] = None
        self._inverse_homography_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._debug_frame_store: Dict[Tuple[str, int], np.ndarray] = {}

        if self.enable_debug_reprojection:
            try:
                self.debug_overlay = DebugOverlay(
                    radius_px=getattr(settings, "DEBUG_OVERLAY_RADIUS_PX", 6)
                )
                self.reprojection_debugger = ReprojectionDebugger(
                    frame_provider=self._provide_debug_frame,
                    output_dir=getattr(settings, "DEBUG_REPROJECTION_OUTPUT_DIR", "app/debug_outputs"),
                    sampling_rate=getattr(settings, "DEBUG_FRAME_SAMPLING_RATE", 1),
                    max_frames_per_camera=getattr(settings, "DEBUG_MAX_FRAMES_PER_CAMERA", 500),
                )
            except Exception as exc:
                logger.warning("Failed to initialize detection reprojection debugger: %s", exc)
                self.enable_debug_reprojection = False

            if self.enable_debug_reprojection and self.world_plane_transformer:
                try:
                    self.inverse_projector = InverseHomographyProjector(
                        world_plane_transformer=self.world_plane_transformer
                    )
                except Exception as exc:
                    logger.warning("Failed to initialize inverse projector for detection pipeline: %s", exc)
                    self.inverse_projector = None
        
        self.global_registry = GlobalPersonRegistry()
        
        # Phase 1: Space-Based Matching
        self.space_based_matcher = SpaceBasedMatcher(registry=self.global_registry)

        # Phase 2: Re-ID Services
        self.feature_extraction_service: Optional[FeatureExtractionService] = None
        self.handoff_manager: Optional[HandoffManager] = None
        self.active_track_ids: Dict[str, Set[int]] = {} # camera_id -> set of track_ids
        
        # Detection statistics (enhanced for Phase 2)
        self.detection_stats = {
            "total_frames_processed": 0,
            "total_detections_found": 0,
            "average_detection_time": 0.0,
            "successful_detections": 0,
            "failed_detections": 0,
            "frames_annotated": 0,
            "websocket_messages_sent": 0,
            "annotation_time": 0.0
        }
        
        # Performance tracking
        self.detection_times: List[float] = []
        self.annotation_times: List[float] = []
        
        # --- Core Integration Architecture: Tracking scaffolding ---
        # Per-camera trackers are managed via a factory. We keep references by camera_id.
        self.camera_trackers: Dict[str, Any] = {}
        self.tracker_factory: Optional[CameraTrackerFactory] = tracker_factory

        # Enhanced statistics
        self.tracking_stats = {
            "total_tracks_created": 0,
            "cross_camera_handoffs": 0,
            "average_track_length": 0.0
        }

        # Frontend event cache

        logger.info("DetectionVideoService initialized (Phase 4: Spatial Intelligence)")

    async def _emit_geometric_metrics(self, environment_id: str, camera_id: str) -> None:
        """Publish geometric extraction and transformation stats to analytics pipeline."""
        try:
            extraction_stats = self.bottom_point_extractor.get_statistics()
            transformation_stats = (
                self.world_plane_transformer.get_statistics()
                if self.world_plane_transformer
                else None
            )
            roi_stats = self.roi_calculator.get_statistics()
            await analytics_event_tracer.record_geometric_metrics(
                environment_id=environment_id,
                camera_id=camera_id,
                extraction_stats=extraction_stats,
                transformation_stats=transformation_stats,
                roi_stats=roi_stats,
                matcher_stats=None,
                metrics_summary=None,
            )
        except Exception as exc:
            pass # logger.debug("Failed to emit geometric metrics: %s", exc)
    
    async def initialize_detection_services(self, environment_id: str = "default", task_id: Optional[uuid.UUID] = None) -> bool:
        """Initialize detection services including YOLO model loading and spatial intelligence services."""
        try:
            logger.info(f"🚀 DETECTION SERVICE INIT: Starting detection service initialization for environment: {environment_id}")
            
            # First initialize parent services (video data manager, asset downloader)
            parent_initialized = await self.initialize_services(environment_id)
            if not parent_initialized:
                logger.error("❌ DETECTION SERVICE INIT: Failed to initialize parent video services")
                return False
            
            # Initialize YOLO detector for the requested environment
            weights_path = self._resolve_yolo_weights_for_environment(environment_id)

            if environment_id not in self.detectors_by_env:
                logger.info(f"🧠 DETECTION SERVICE INIT: Loading YOLO model for '{environment_id}' from: {weights_path}")
                detector = YOLODetector(
                    model_name=weights_path,
                    confidence_threshold=settings.YOLO_CONFIDENCE_THRESHOLD
                )
                await detector.load_model()
                await detector.warmup()
                self.detectors_by_env[environment_id] = detector
                self.detector_weights_by_env[environment_id] = weights_path
            else:
                # If already loaded (preloaded at startup), just use it
                logger.info(f"🧠 DETECTION SERVICE INIT: Using preloaded YOLO model for '{environment_id}'")

            # Maintain backward-compatible single-detector reference for existing methods
            self.detector = self.detectors_by_env.get(environment_id)
            
            # Phase 4: Initialize spatial intelligence services (via DI if available)
            logger.info("🗺️ DETECTION SERVICE INIT: Initializing spatial intelligence services...")
            if self.homography_service is None:
                try:
                    # Fallback to local instance (degraded mode)
                    self.homography_service = HomographyService(settings)
                    if getattr(settings, 'PRELOAD_HOMOGRAPHY', True):
                        await self.homography_service.preload_all_homography_matrices()
                    logger.warning("HomographyService was not injected; created local instance.")
                except Exception as e:
                    logger.warning(f"HomographyService initialization failed: {e}")
            
            # Initialize HandoffDetectionService
            self.handoff_service = HandoffDetectionService()

            # Initialize Re-ID Services (Phase 2)
            if self.feature_extraction_service is None:
                try:
                    self.feature_extraction_service = FeatureExtractionService()
                    logger.info("🧠 RE-ID: FeatureExtractionService initialized (on-demand)")
                except Exception as e:
                    logger.error(f"❌ RE-ID INIT FAILED: {e}")
                    self.feature_extraction_service = None
            else:
                logger.info("🧠 RE-ID: Using preloaded FeatureExtractionService")
            
            if self.handoff_manager is None:
                self.handoff_manager = HandoffManager()
            
            # Validate spatial intelligence configuration
            homography_validation = bool(getattr(self.homography_service, "json_homography_matrices", {}))
            handoff_validation = self.handoff_service.validate_configuration()
            
            logger.info(f"🗺️ SPATIAL INTELLIGENCE: Homography matrices loaded: {homography_validation}")
            logger.info(f"🗺️ SPATIAL INTELLIGENCE: Handoff configuration valid: {all(handoff_validation.values())}")
            
            # Trail cleanup runs globally (app startup); do not start per-task here

            # Core Integration: Optionally initialize tracking scaffolding
            try:
                if settings.TRACKING_ENABLED and task_id is not None:
                    await self.initialize_tracking_services(task_id, environment_id)
            except Exception as e:
                logger.warning(f"TRACKING init skipped or failed (non-blocking): {e}")
            
            logger.info("✅ DETECTION SERVICE INIT: Detection and spatial intelligence services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ DETECTION SERVICE INIT: Failed to initialize detection services: {e}")
            return False

    def _resolve_yolo_weights_for_environment(self, environment_id: str) -> str:
        """Resolve the best YOLO weights path for the given environment.

        Resolution order per environment:
        0) If USE_TENSORRT=true AND CUDA available, prefer TensorRT engine files (.engine)
        1) Settings override (YOLO_MODEL_PATH_CAMPUS / YOLO_MODEL_PATH_FACTORY)
        2) External base dir (EXTERNAL_WEIGHTS_BASE_DIR) with 'yolo11l_<env>.[pt|engine]'
        3) Local weights directory './weights/yolo11l_<env>.[pt|engine]'
        4) Global default: settings.YOLO_MODEL_PATH or YOLO_MODEL_PATH_TENSORRT
        
        Note: Automatically falls back to .pt if CUDA is not available (e.g., on Mac/CPU).
        """
        try:
            import os
            import torch
            
            env_key = (environment_id or "").strip().lower()
            use_tensorrt_setting = getattr(settings, "USE_TENSORRT", False)
            
            # TensorRT requires CUDA - automatically disable on CPU/Mac
            cuda_available = torch.cuda.is_available()
            use_tensorrt = use_tensorrt_setting and cuda_available
            
            if use_tensorrt_setting and not cuda_available:
                logger.warning(
                    "USE_TENSORRT=true but CUDA not available (running on CPU/Mac). "
                    "Falling back to PyTorch model (.pt)"
                )
            
            # Determine file extension based on TensorRT setting
            ext = ".engine" if use_tensorrt else ".pt"
            
            # 1) Settings override
            if env_key == "factory" and getattr(settings, "YOLO_MODEL_PATH_FACTORY", None):
                candidate = settings.YOLO_MODEL_PATH_FACTORY
                # If TensorRT enabled, try .engine version first
                if use_tensorrt:
                    engine_candidate = candidate.replace(".pt", ".engine")
                    if os.path.isfile(engine_candidate):
                        logger.info(f"🚀 Using TensorRT engine: {engine_candidate}")
                        return engine_candidate
                if os.path.isfile(candidate):
                    return candidate
                    
            if env_key == "campus" and getattr(settings, "YOLO_MODEL_PATH_CAMPUS", None):
                candidate = settings.YOLO_MODEL_PATH_CAMPUS
                # If TensorRT enabled, try .engine version first
                if use_tensorrt:
                    engine_candidate = candidate.replace(".pt", ".engine")
                    if os.path.isfile(engine_candidate):
                        logger.info(f"🚀 Using TensorRT engine: {engine_candidate}")
                        return engine_candidate
                if os.path.isfile(candidate):
                    return candidate

            # 2) External base dir (YOLO-specific naming)
            if getattr(settings, "EXTERNAL_WEIGHTS_BASE_DIR", None):
                # Try TensorRT first if enabled
                if use_tensorrt:
                    candidate = os.path.join(settings.EXTERNAL_WEIGHTS_BASE_DIR, f"yolo11l_{env_key}.engine")
                    if os.path.isfile(candidate):
                        logger.info(f"🚀 Using TensorRT engine: {candidate}")
                        return candidate
                # Fallback to .pt
                candidate = os.path.join(settings.EXTERNAL_WEIGHTS_BASE_DIR, f"yolo11l_{env_key}.pt")
                if os.path.isfile(candidate):
                    return candidate

            # 3) Local weights dir (YOLO-specific naming)
            # Try TensorRT first if enabled
            if use_tensorrt:
                candidate = os.path.join("weights", f"yolo11l_{env_key}.engine")
                if os.path.isfile(candidate):
                    logger.info(f"🚀 Using TensorRT engine: {candidate}")
                    return candidate
            # Fallback to .pt
            candidate = os.path.join("weights", f"yolo11l_{env_key}.pt")
            if os.path.isfile(candidate):
                return candidate

            # 4) Global default - TensorRT path if enabled, else PyTorch
            if use_tensorrt:
                tensorrt_path = getattr(settings, "YOLO_MODEL_PATH_TENSORRT", None)
                if tensorrt_path and os.path.isfile(tensorrt_path):
                    logger.info(f"🚀 Using TensorRT engine: {tensorrt_path}")
                    return tensorrt_path
                # Warn if TensorRT enabled but no engine found
                logger.warning(f"USE_TENSORRT=true but no .engine file found. Falling back to PyTorch model.")
            
            return settings.YOLO_MODEL_PATH
        except Exception:
            return settings.YOLO_MODEL_PATH

    async def _apply_reid_logic(self, tracks: List[Dict[str, Any]], frame: np.ndarray, camera_id: str, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """
        Apply Re-ID logic:
        1. Detect Entry (New Tracks) -> Extract -> Match against HandoffManager -> Assign Global ID
        2. Detect Exit (Handoff Zone) -> Extract -> Register to HandoffManager
        3. Assign Global ID to tracks
        """
        # Early exit if Re-ID is disabled via config
        if not getattr(settings, 'REID_ENABLED', True):
            return tracks
        
        if not self.feature_extraction_service or not self.handoff_manager:
            return tracks

        # Initialize cooldown tracker if not present
        if not hasattr(self, "_reid_cooldowns"):
             self._reid_cooldowns: Dict[str, float] = {}
        
        # Cooldown configuration (hardcoded for now, could be in settings)
        COOLDOWN_SECONDS = 0.5 
        now = time.time()

        current_track_ids = set()
        if camera_id not in self.active_track_ids:
            self.active_track_ids[camera_id] = set()
        
        previous_track_ids = self.active_track_ids[camera_id]

        for track in tracks:
            track_id = int(track['track_id'])
            current_track_ids.add(track_id)
            
            # Convert bbox_xyxy [x1,y1,x2,y2] to dict format for handoff_service
            bbox_xyxy = track.get('bbox_xyxy')
            if not bbox_xyxy or len(bbox_xyxy) < 4:
                continue
            x1, y1, x2, y2 = bbox_xyxy[:4]
            bbox = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2 - x1,
                'height': y2 - y1,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
            }
            
            # Unique key for this track in this camera
            track_key = f"{camera_id}_{track_id}"
            
            # Check if this is a NEW track (Entry Event)
            if track_id not in previous_track_ids:
                # Trigger Re-ID Match ONLY if in Handoff Zone (Boundary Trigger)
                # This prevents Re-ID from overriding Spatial Matcher for center-frame detections
                try:
                    in_zone = False
                    if self.handoff_service:
                         in_zone, _ = self.handoff_service.check_handoff_trigger(
                            camera_id, bbox, frame_width, frame_height
                         )
                    
                    if in_zone:
                        # logger.info(f"[RE-ID] 🟢 New Track {track_id} in {camera_id} detected in Handoff Zone. Triggering Global Search.")
                        
                        # Crop image
                        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                        # Clamp
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_width, x2), min(frame_height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            patch = frame[y1:y2, x1:x2]
                            embedding = await self.feature_extraction_service.extract_async(patch)
                            
                            match_found = False
                            if embedding is not None:
                                # Search for match in HandoffManager
                                match_global_id, score = self.handoff_manager.find_match(embedding, camera_id)
                                if match_global_id:
                                    # logger.info(f"[RE-ID] ✅ MATCH FOUND: Track {track_id} in {camera_id} matched to Global ID {match_global_id} (Score: {score:.2f})")
                                    # Updated Phase 3: Use Registry
                                    self.global_registry.assign_identity(camera_id, track_id, match_global_id)
                                    
                                    track['global_id'] = match_global_id
                                    match_found = True
                                    
                                    # Set cooldown to prevent immediate re-extraction
                                    self._reid_cooldowns[track_key] = now
                                else:
                                     pass # logger.debug(f"[RE-ID] ⚪ No match for new Track {track_id} in {camera_id}. Assigned temporary ID.")
                    else:
                        pass # logger.debug(f"[RE-ID] New Track {track_id} in {camera_id} NOT in Handoff Zone. Skipping Re-ID search (relying on spatial/local).")

                except Exception as e:
                    logger.error(f"Re-ID Entry Error: {e}")

            # Check if track is in Handoff Zone (Exit Event Candidate)
            if self.handoff_service:
                is_handoff, _ = self.handoff_service.check_handoff_trigger(
                     camera_id, bbox, frame_width, frame_height
                )
                if is_handoff:
                     # Check Cooldown
                     last_time = self._reid_cooldowns.get(track_key, 0.0)
                     if (now - last_time) >= COOLDOWN_SECONDS:
                         # This person is leaving? Extract and Register
                         try:
                            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_width, x2), min(frame_height, y2)
                            
                            if x2 > x1 and y2 > y1:
                                patch = frame[y1:y2, x1:x2]
                                embedding = await self.feature_extraction_service.extract_async(patch)
                                if embedding is not None:
                                    # Use current global_id if available, else track_id as temporary global_id
                                    # Phase 3: Get from registry first
                                    current_gid = self.global_registry.get_global_id(camera_id, track_id)
                                    gid = current_gid or track.get('global_id') or f"temp_{camera_id}_{track_id}"
                                    
                                    # logger.info(f"[RE-ID] 🚪 Track {track_id} (Global: {gid}) entered Handoff Zone in {camera_id}. Registering exit.")
                                    self.handoff_manager.register_exit(gid, embedding, camera_id)
                                    pass # logger.debug(f"[RE-ID] 📤 Registered {gid} for handoff. Embedding size: {embedding.shape}.")
                                    
                                    # Update cooldown
                                    self._reid_cooldowns[track_key] = now
                         except Exception as e:
                            logger.error(f"Re-ID Exit Error: {e}")
                     # else: logger.debug("Skipping Re-ID due to cooldown") 

        # Update active tracks
        self.active_track_ids[camera_id] = current_track_ids
        
        # Cleanup cooldowns for stale tracks (simple version: strict memory management would be better)
        # For now, just let it grow slightly or clean periodically? 
        # Let's do a quick lazy cleanup if dict gets too big
        if len(self._reid_cooldowns) > 1000:
             # Keep only keys in current active list across all cameras (simplified)
             # Actually active_track_ids is structured by camera.
             active_keys = {f"{c}_{t}" for c, t_set in self.active_track_ids.items() for t in t_set}
             self._reid_cooldowns = {k: v for k, v in self._reid_cooldowns.items() if k in active_keys}

        # Final pass: Ensure all tracks have their assigned global_id from registry
        for track in tracks:
            t_id = int(track['track_id'])
            g_id = self.global_registry.get_global_id(camera_id, t_id)
            if g_id:
                track['global_id'] = g_id
        
        return tracks

    # --- Core Integration Architecture Methods (scaffolding) ---
    async def initialize_tracking_services(self, task_id: uuid.UUID, environment_id: str) -> bool:
        """Initialize per-camera trackers via factory."""
        try:
            # Lazily create tracker factory using compute device heuristics from BoxMOT
            if self.tracker_factory is None:
                # Device selection handled internally by trackers; we just instantiate the factory
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tracker_factory = CameraTrackerFactory(device=device)

            # Determine cameras for environment from config
            camera_ids = [vc.cam_id for vc in settings.VIDEO_SETS if vc.env_id == environment_id]
            logger.info(f"Cameras found in config for env {environment_id}: {camera_ids}")
            if not camera_ids:
                 logger.warning(f"No cameras found for env {environment_id} in settings.VIDEO_SETS")

            # Create trackers for each camera (keyed by the real task_id)
            self.camera_trackers = {}
            for camera_id in camera_ids:
                try:
                    tracker = await self.tracker_factory.get_tracker(task_id=task_id, camera_id=camera_id)
                    self.camera_trackers[camera_id] = tracker
                    logger.info(f"Initialized tracker for {camera_id}")
                except Exception as e:
                    logger.warning(f"Failed to initialize tracker for camera {camera_id}: {e}")

            logger.info(f"Tracking scaffolding prepared for environment {environment_id} (trackers: {len(self.camera_trackers)})")
            return True
        except Exception as e:
            logger.warning(f"initialize_tracking_services encountered an issue (non-blocking): {e}")
            return False

    async def process_frame_with_tracking(self, frame: np.ndarray, camera_id: str, frame_number: int) -> Dict[str, Any]:
        """Process frame with detection, then integrate tracking and spatial intelligence.

        - Runs existing detection + spatial intelligence pipeline
        - If tracking is enabled and a tracker is available for the camera, updates tracks
        - Enhances track data with map coordinates and short trajectories
        """
        import time as _time
        _total_start = _time.perf_counter()
        
        # Step 1: Run detection pipeline (includes spatial intelligence)
        _det_start = _time.perf_counter()
        detection_data = await self.process_frame_with_detection(frame, camera_id, frame_number)
        _det_time = (_time.perf_counter() - _det_start) * 1000

        # Step 2: Tracking integration (via tracker factory)
        # Step 2: Tracking integration (via tracker factory)
        tracks: List[Dict[str, Any]] = []
        tracker_used = False
        _track_time = 0.0
        _spatial_time = 0.0
        _reid_time = 0.0
        
        try:
            if settings.TRACKING_ENABLED:
                # Lazy initialization check
                if camera_id not in self.camera_trackers:
                    try:
                        # logger.info(f"Tracker missing for {camera_id}. Attempting lazy initialization...")
                        tracker = await self.tracker_factory.get_tracker("lazy_init_task", camera_id)
                        if tracker:
                            self.camera_trackers[camera_id] = tracker
                            # logger.info(f"Lazy initialization successful for {camera_id}")
                    except Exception as e:
                        logger.error(f"Lazy tracker init failed for {camera_id}: {e}")

            if settings.TRACKING_ENABLED and camera_id in self.camera_trackers:
                tracker = self.camera_trackers.get(camera_id)
                # Convert detections to BoxMOT format: [x1, y1, x2, y2, conf, cls]
                np_dets = self._convert_detections_to_boxmot_format(detection_data.get("detections", []))
                
                _track_start = _time.perf_counter()
                tracked_np = await tracker.update(np_dets, frame)
                _track_time = (_time.perf_counter() - _track_start) * 1000
                
                # ... (logging skipped for brevity in replacement) ...

                # Convert tracked output to track dicts
                tracks = self._convert_boxmot_to_track_data(tracked_np, camera_id)
                
                # Enhance with spatial intelligence (map coords + short trajectory)
                _spatial_start = _time.perf_counter()
                tracks = await self._enhance_tracks_with_spatial_intelligence(tracks, camera_id, frame_number)
                _spatial_time = (_time.perf_counter() - _spatial_start) * 1000
                
                # Apply Re-ID (Phase 2)
                _reid_start = _time.perf_counter()
                frame_height, frame_width = frame.shape[:2]
                tracks = await self._apply_reid_logic(tracks, frame, camera_id, frame_width, frame_height)
                _reid_time = (_time.perf_counter() - _reid_start) * 1000
                
                tracker_used = True
            else:
                if frame_number % 30 == 0:
                    logger.info(
                        "Tracking skipped for camera %s (enabled=%s, tracker_available=%s)",
                        camera_id,
                        settings.TRACKING_ENABLED,
                        camera_id in self.camera_trackers
                    )
        except Exception as e:
            logger.warning(f"Tracking integration failed for camera {camera_id} on frame {frame_number}: {e}")
            tracks = []

        # Step 3: Attach tracks to detection data for downstream consumers
        detection_data["tracks"] = tracks

        try:
            self._associate_detections_with_tracks(camera_id, detection_data.get("detections"), tracks)
            # if tracker_used and logger.isEnabledFor(logging.DEBUG):
            #     pass
        except Exception as exc:
            pass
            # logger.debug(
            #     "Detection-track association failed for %s frame %s: %s",
            #     camera_id,
            #     frame_number,
            #     exc,
            # )
        
        # Frontend: emit auxiliary events (non-blocking)
        _viz_start = _time.perf_counter()
        try:
            if getattr(settings, 'ENABLE_ENHANCED_VISUALIZATION', True):
                await self._emit_frontend_events(task_id=None, camera_id=camera_id, frame_number=frame_number, tracks=tracks)
        except Exception:
            pass
        _viz_time = (_time.perf_counter() - _viz_start) * 1000
        
        # Total time
        _total_time = (_time.perf_counter() - _total_start) * 1000
        
        # Log timing breakdown every 10 frames or if slow (>100ms)
        if frame_number % 10 == 0 or _total_time > 100:
            speed_optimize_logger.info(
                "[SPEED_OPTIMIZE] Cam=%s Frame=%d | Total=%.1fms | Det=%.1fms Track=%.1fms Spatial=%.1fms ReID=%.1fms Viz=%.1fms | Tracks=%d",
                camera_id, frame_number, _total_time,
                _det_time, _track_time, _spatial_time, _reid_time, _viz_time,
                len(tracks)
            )
        
        return detection_data

    def get_tracking_stats(self) -> Dict[str, Any]:
        """Return current tracking stats without affecting existing API."""
        return {
            **self.tracking_stats,
            "trackers_initialized": len(self.camera_trackers)
        }
    
    async def process_frame_with_detection(self, frame: np.ndarray, camera_id: str, frame_number: int) -> Dict[str, Any]:
        """
        Process frame with YOLO detection and Phase 4 spatial intelligence.
        
        Args:
            frame: Video frame as numpy array
            camera_id: Camera identifier  
            frame_number: Frame sequence number
            
        Returns:
            Detection data dictionary with bounding boxes, coordinates, and spatial metadata
        """
        import time as _time
        _proc_start = _time.perf_counter()
        _yolo_time = 0.0
        _spatial_loop_time = 0.0
        
        try:
            # Select detector by environment of the camera, fallback to default
            env_for_camera = self._get_environment_for_camera(camera_id) or "default"
            detector = self.detectors_by_env.get(env_for_camera) or self.detector
            if not detector:
                raise RuntimeError("YOLO detector not initialized for the current environment")
            
            # Run YOLO detection
            _yolo_start = _time.perf_counter()
            detections = await detector.detect(frame)
            _yolo_time = (_time.perf_counter() - _yolo_start) * 1000
            
            # Calculate processing time
            processing_time = (_time.perf_counter() - _proc_start) * 1000
            
            # Get frame dimensions for spatial processing
            frame_height, frame_width = frame.shape[:2]
            
            # Convert detections to the expected format with Phase 4 enhancements
            _spatial_start = _time.perf_counter()
            enhanced_detections = []
            for i, detection in enumerate(detections):
                bbox_dict = {
                    "x1": detection.bbox.x1,
                    "y1": detection.bbox.y1, 
                    "x2": detection.bbox.x2,
                    "y2": detection.bbox.y2,
                    "width": detection.bbox.x2 - detection.bbox.x1,
                    "height": detection.bbox.y2 - detection.bbox.y1,
                    "center_x": (detection.bbox.x1 + detection.bbox.x2) / 2,
                    "center_y": (detection.bbox.y1 + detection.bbox.y2) / 2
                }
                
                # Phase 4: Apply spatial intelligence
                map_coords = {"map_x": 0, "map_y": 0}  # Default fallback
                projection_success = False
                transformation_quality: Optional[float] = None
                handoff_triggered = False
                candidate_cameras = []
                search_roi_payload: Optional[Dict[str, Optional[float]]] = None

                bottom_point: Optional[ImagePoint] = None
                try:
                    bottom_point = self.bottom_point_extractor.extract_point(
                        bbox_x=bbox_dict["x1"],
                        bbox_y=bbox_dict["y1"],
                        bbox_width=bbox_dict["width"],
                        bbox_height=bbox_dict["height"],
                        camera_id=CameraID(camera_id),
                        person_id=None,
                        frame_number=frame_number,
                        timestamp=None,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                except ValueError as exc:
                    pass # logger.debug(
                    #     "Bottom point validation failed for detection %s in camera %s: %s",
                    #     i,
                    #     camera_id,
                    #     exc,
                    # )

                world_point: Optional[WorldPoint] = None
                if self.world_plane_transformer and bottom_point:
                    try:
                        world_point = self.world_plane_transformer.transform_point(bottom_point)
                        transformation_quality = world_point.transformation_quality
                        map_coords = {"map_x": world_point.x, "map_y": world_point.y}
                        projection_success = transformation_quality >= 0.5
                    except (KeyError, ValueError) as exc:
                        logger.info(
                            "World-plane transform failed for camera %s detection %s: %s",
                            camera_id,
                            i,
                            exc,
                        )

                # Homography coordinate transformation fallback (legacy)
                if not projection_success and self.homography_service and bottom_point:
                    fallback_point = (bottom_point.x, bottom_point.y)
                    projected_coords = self.homography_service.project_to_map(camera_id, fallback_point)
                    if projected_coords:
                        candidate_map_x, candidate_map_y = projected_coords
                        if self.homography_service.validate_map_coordinate(camera_id, candidate_map_x, candidate_map_y):
                            map_coords = {"map_x": candidate_map_x, "map_y": candidate_map_y}
                            projection_success = True
                            transformation_quality = transformation_quality or 0.8  # Legacy validation success
                            pass
                            # logger.debug(
                            #     "Fallback homography map coords accepted for %s: (%.4f, %.4f)",
                            #     camera_id,
                            #     candidate_map_x,
                            #     candidate_map_y,
                            # )
                        else:
                            pass
                            # logger.info(
                            #     "Fallback projected coords out of bounds for %s: (%.4f, %.4f)",
                            #     camera_id,
                            #     candidate_map_x,
                            #     candidate_map_y,
                            # )

                if projection_success and map_coords:
                    try:
                        roi = self.roi_calculator.calculate_roi(
                            (map_coords["map_x"], map_coords["map_y"]),
                            time_elapsed=0.0,
                            transformation_quality=transformation_quality if transformation_quality is not None else 1.0,
                            shape=self.roi_shape,
                            source_camera=camera_id,
                            dest_camera=None,
                            person_id=None,
                            timestamp=None,
                        )
                        search_roi_payload = roi.to_dict()
                    except Exception as exc:
                        pass
                        # logger.debug(
                        #     "ROI calculation failed for detection %s in camera %s: %s",
                        #     i,
                        #     camera_id,
                        #     exc,
                        # )

                # Handoff detection
                if self.handoff_service:
                    handoff_triggered, candidate_cameras = self.handoff_service.check_handoff_trigger(
                        camera_id, bbox_dict, frame_width, frame_height
                    )
                
                # Create enhanced detection object
                enhanced_detection = {
                    "detection_id": f"det_{i:03d}",
                    "class_name": "person",
                    "class_id": 0,
                    "confidence": detection.confidence,
                    "bbox": bbox_dict,
                    "track_id": None,  # Future: No tracking yet
                    "global_id": None,  # Future: No re-ID yet
                    "map_coords": map_coords,
                    # Phase 4: Spatial intelligence metadata
                    "spatial_data": {
                        "handoff_triggered": handoff_triggered,
                        "candidate_cameras": candidate_cameras,
                        "coordinate_system": "world_meters" if projection_success else None,
                        "projection_successful": projection_success,
                        "transformation_quality": transformation_quality,
                        "search_roi": search_roi_payload,
                    }
                }
                
                enhanced_detections.append(enhanced_detection)
            
            _spatial_loop_time = (_time.perf_counter() - _spatial_start) * 1000
            
            detection_data = {
                "detections": enhanced_detections,
                "detection_count": len(detections),
                "processing_time_ms": processing_time,
                # Phase 4: Frame spatial metadata
                "spatial_metadata": {
                    "camera_id": camera_id,
                    "frame_dimensions": {"width": frame_width, "height": frame_height},
                    "homography_available": self.homography_service is not None,
                    "handoff_detection_enabled": self.handoff_service is not None
                }
            }
            
            # Update statistics
            self.detection_times.append(processing_time)
            self.detection_stats["total_frames_processed"] += 1
            self.detection_stats["total_detections_found"] += len(detections)
            self.detection_stats["successful_detections"] += 1
            self._update_detection_stats()

            await self._emit_geometric_metrics(environment_id=env_for_camera, camera_id=camera_id)
            
            # Total time for this method
            _total_det_time = (_time.perf_counter() - _proc_start) * 1000
            
            # Log granular timing every 10 frames or if slow
            if frame_number % 10 == 0 or _total_det_time > 50:
                speed_optimize_logger.info(
                    "[SPEED_OPTIMIZE] DET Cam=%s Frame=%d | Total=%.1fms | YOLO=%.1fms SpatialLoop=%.1fms | Dets=%d",
                    camera_id, frame_number, _total_det_time,
                    _yolo_time, _spatial_loop_time, len(detections)
                )
            
            return detection_data
            
        except Exception as e:
            logger.error(f"❌ DETECTION PROCESSING: Error processing frame {frame_number} from camera {camera_id}: {e}")
            self.detection_stats["failed_detections"] += 1
            
            # Return empty detection result on error
            return {
                "detections": [],
                "detection_count": 0,
                "processing_time_ms": 0,
                "error": str(e)
            }
    
    async def run_detection_pipeline(self, task_id: uuid.UUID, environment_id: str):
        """
        Main detection pipeline that extends raw video streaming with YOLO detection.
        
        Phase 1 Process:
        1. Initialize detection services (including YOLO)
        2. Download video data (inherited from parent)
        3. Extract and process frames with detection
        4. Stream detection results via WebSocket
        """
        pipeline_start = time.time()
        
        try:
            logger.info(f"🎬 DETECTION PIPELINE: Starting detection pipeline for task {task_id}, environment {environment_id}")
            
            # Update task status
            await self._update_task_status(task_id, "INITIALIZING", 0.05, "Initializing detection services")
            
            # Step 1: Initialize detection services
            logger.info(f"🧠 DETECTION PIPELINE: Step 1/4 - Initializing detection services for task {task_id}")
            services_initialized = await self.initialize_detection_services(environment_id, task_id)
            if not services_initialized:
                raise RuntimeError("Failed to initialize detection services")
            
            await self._update_task_status(task_id, "DOWNLOADING", 0.25, "Downloading video data")
            
            # Step 2: Download video data (inherited method)
            logger.info(f"⬇️ DETECTION PIPELINE: Step 2/4 - Downloading video data for task {task_id}")
            video_data = await self._download_video_data(environment_id)
            if not video_data:
                raise RuntimeError("Failed to download video data")
            
            await self._update_task_status(task_id, "PROCESSING", 0.50, "Processing frames with YOLO detection")
            
            # Step 3: Process frames with detection
            logger.info(f"🔍 DETECTION PIPELINE: Step 3/4 - Processing frames with detection for task {task_id}")
            detection_success = await self._process_frames_with_detection(task_id, video_data)
            if self._task_marked_stopped(task_id):
                logger.info(f"🛑 DETECTION PIPELINE: Task {task_id} stopped early (no active WebSocket clients)")
                return
            if not detection_success:
                raise RuntimeError("Failed to process frames with detection")

            await self._update_task_status(task_id, "STREAMING", 0.75, "Streaming detection results")

            # Step 4: Stream detection results (integrated with frame processing)
            logger.info(f"📡 DETECTION PIPELINE: Step 4/4 - Streaming detection results for task {task_id}")
            streaming_success = await self._stream_detection_results(task_id, video_data)
            if self._task_marked_stopped(task_id):
                logger.info(f"🛑 DETECTION PIPELINE: Task {task_id} stopped before summary streaming (no clients)")
                return
            if not streaming_success:
                raise RuntimeError("Failed to stream detection results")
            
            # Mark as completed
            await self._update_task_status(task_id, "COMPLETED", 1.0, "Detection pipeline completed successfully")
            
            # Update final statistics
            pipeline_time = time.time() - pipeline_start
            logger.info(f"✅ DETECTION PIPELINE: Pipeline completed successfully for task {task_id} in {pipeline_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ DETECTION PIPELINE: Error in detection pipeline for task {task_id}: {e}")
            await self._update_task_status(task_id, "FAILED", 0.0, f"Detection pipeline failed: {str(e)}")
            
        finally:
            # Cleanup (inherited from parent)
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            if environment_id in self.environment_tasks:
                del self.environment_tasks[environment_id]
            self._clear_client_watch(task_id)
    
    async def _process_frames_with_detection(self, task_id: uuid.UUID, video_data: Dict[str, Any]) -> bool:
        """Process video frames with YOLO detection."""
        try:
            logger.info(f"🔍 DETECTION PROCESSING: Processing frames with detection for task {task_id}")
            
            # Get frame count for progress tracking
            total_frames = min(
                data.get("frame_count", 0) for data in video_data.values() 
                if data.get("frame_count", 0) > 0
            )
            
            if total_frames == 0:
                logger.warning("No frames available for detection processing")
                return False
            
            frame_index = 0
            aborted_due_to_no_clients = False
            
            # FPS tracking for detection pipeline
            fps_start_time = time.time()
            fps_frame_count = 0
            
            # Process frames from all cameras
            while frame_index < total_frames:
                # Check if task is still active
                if task_id not in self.active_tasks:
                    logger.info(f"🔍 DETECTION PROCESSING: Task {task_id} was stopped")
                    break
                if not self._should_continue_stream(task_id, detection_mode=True):
                    aborted_due_to_no_clients = True
                    break

                # Read frames from all cameras
                camera_frames = {}
                camera_detections = {}
                all_frames_valid = True
                
                for camera_id, data in video_data.items():
                    cap = data.get("video_capture")
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            camera_frames[camera_id] = frame
                            
                            # Process frame with enhanced flow stub (detection -> tracking pipeline)
                            detection_data = await self.process_frame_with_tracking(
                                frame, camera_id, frame_index
                            )

                            camera_detections[camera_id] = detection_data
                            
                            # Log analytics (moved logic here to ensure we have data)
                            environment_id = None
                            camera_config = data.get("config")
                            if camera_config is not None:
                                environment_id = getattr(camera_config, "env_id", None)
                            environment_id = environment_id or "default"

                            detection_count = detection_data.get("detection_count", 0)
                            detections_payload = detection_data.get("detections", []) or []
                            average_confidence = None
                            if detection_count > 0:
                                average_confidence = sum(
                                    float(det.get("confidence", 0.0)) for det in detections_payload
                                ) / detection_count

                            await analytics_event_tracer.record_detection_batch(
                                environment_id=environment_id,
                                camera_id=camera_id,
                                detections=detection_count,
                                unique_entities=detection_count,
                                average_confidence=average_confidence,
                                trace_id=str(task_id),
                                timestamp=datetime.now(timezone.utc),
                            )

                        else:
                            all_frames_valid = False
                            break
                    else:
                        all_frames_valid = False
                        break
                
                if not all_frames_valid or not camera_frames:
                    logger.info(f"🔍 DETECTION PROCESSING: End of video reached at frame {frame_index}")
                    break
                
                # --- Phase 1: Cross-Camera Space-Based Matching ---
                # Run matching algorithm on checking all cameras for this frame
                try:
                    self.space_based_matcher.match_across_cameras(camera_detections)
                except Exception as e:
                    logger.error(f"Space-based matching failed: {e}")

                # Update 'is_matched' flag for correct coloring
                # precise timing: MUST be done after match_across_cameras updates global IDs
                if self.space_based_matcher:
                    # logger.warning("DEBUG: Updating is_matched flags...") # Confirm block entry
                    for camera_id, detection_data in camera_detections.items():
                        for track in detection_data.get("tracks", []):
                            gid = track.get("global_id")
                            if gid:
                                is_shared = self.space_based_matcher.is_global_id_shared(gid)
                                track["is_matched"] = is_shared
                                
                                # Log visibility status (WARNING to ensure visibility)
                                if is_shared:
                                    pass # logger.warning(f"COLOR DEBUG: Cam {camera_id} Track {track.get('track_id')} Global {gid} Shared=True -> COLORING")
                                else:
                                    # Log unmatched too (limited to avoid infinite spam if possible, but for short test it is fine)
                                    pass # logger.warning(f"COLOR DEBUG: Cam {camera_id} Track {track.get('track_id')} Global {gid} Shared=False -> GREEN")

                # --- Send Updates Loop ---
                # Now that global_ids are injected, send updates to frontend
                for camera_id, detection_data in camera_detections.items():
                    frame = camera_frames.get(camera_id)
                    if frame is not None:
                        # Phase 2: Send real-time detection update via WebSocket
                        await self.send_detection_update(task_id, camera_id, frame, detection_data, frame_index)
                        # Emit auxiliary events tied to this task
                        if getattr(settings, 'ENABLE_ENHANCED_VISUALIZATION', True):
                            await self._emit_frontend_events(task_id, camera_id, frame_index, detection_data.get("tracks", []))
                    else:
                        all_frames_valid = False
                        break
                
                if not all_frames_valid or not camera_frames:
                    logger.info(f"🔍 DETECTION PROCESSING: End of video reached at frame {frame_index}")
                    break
                
                # --- Phase 4 Debug: Reprojection ---
                if self.enable_debug_reprojection and self.reprojection_debugger:
                    # 1. Populate debug store
                    for cid, frm in camera_frames.items():
                        self._debug_frame_store[(cid, frame_index)] = frm
                    
                    # 2. Collect world points
                    debug_payload = {}
                    for cid, det_data in camera_detections.items():
                         w_points = self._collect_world_points(det_data, cid, frame_index)
                         debug_payload[cid] = {"world_points": w_points}
                    
                    # 3. Resolve environment (heuristic)
                    sample_env_id = "default"
                    for d in video_data.values():
                        cfg = d.get("config")
                        if cfg and getattr(cfg, "env_id", None):
                            sample_env_id = cfg.env_id
                            break

                    # 4. Emit debug frame
                    self._emit_reprojection_debug_frame(sample_env_id, frame_index, debug_payload)

                    # 5. Cleanup
                    for cid in camera_frames.keys():
                        self._debug_frame_store.pop((cid, frame_index), None)

                # Log progress periodically
                if frame_index % 30 == 0:  # Every 30 frames
                    progress = 0.50 + (frame_index / total_frames) * 0.25  # 0.50-0.75 range
                    detection_count = sum(det["detection_count"] for det in camera_detections.values())
                    await self._update_task_status(
                        task_id, "PROCESSING", progress,
                        f"Processed frame {frame_index}/{total_frames} - Found {detection_count} detections"
                    )
                    # FPS logging
                    fps_frame_count += 30
                    fps_elapsed = time.time() - fps_start_time
                    if fps_elapsed > 0:
                        current_fps = fps_frame_count / fps_elapsed
                        logger.info(f"[FPS_DEBUG] Detection Pipeline FPS={current_fps:.1f} (frames={fps_frame_count} elapsed={fps_elapsed:.1f}s)")
                    logger.info(f"🔍 DETECTION PROCESSING: Frame {frame_index}/{total_frames} - {detection_count} detections")
                
                frame_index += 1
            
            # Cleanup video captures
            for data in video_data.values():
                cap = data.get("video_capture")
                if cap:
                    cap.release()

            # Record uptime snapshots for the processed cameras (simplified 100% uptime assumption)
            for camera_id, data in video_data.items():
                camera_config = data.get("config")
                environment_id = getattr(camera_config, "env_id", None) if camera_config else None
                environment_id = environment_id or "default"
                await analytics_event_tracer.record_camera_uptime(
                    environment_id=environment_id,
                    camera_id=camera_id,
                    uptime_percent=100.0,
                    samples=1,
                    trace_id=str(task_id),
                )

            if aborted_due_to_no_clients:
                reason = "Detection stopped - no active WebSocket clients"
                self._mark_task_stopped_due_to_idle_clients(task_id, reason)
                logger.info(f"🛑 DETECTION PROCESSING: {reason} (task {task_id})")
                return True

            logger.info(f"✅ DETECTION PROCESSING: Frame processing completed for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ DETECTION PROCESSING: Error processing frames: {e}")
            return False
    
    async def _stream_detection_results(self, task_id: uuid.UUID, video_data: Dict[str, Any]) -> bool:
        """Stream detection results via WebSocket (Phase 1 - basic implementation)."""
        try:
            logger.info(f"📡 DETECTION STREAMING: Starting detection result streaming for task {task_id}")
            
            # For Phase 1, we'll implement a simple streaming approach
            # This will be enhanced in later phases with real-time processing
            
            # Create a summary message for completed detection processing
            total_detections = sum(self.detection_stats.get("total_detections_found", 0) for _ in video_data)
            
            summary_message = {
                "type": MessageType.TRACKING_UPDATE.value,
                "task_id": str(task_id),
                "timestamp_processed_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "detection_processing",
                "summary": {
                    "total_frames_processed": self.detection_stats.get("total_frames_processed", 0),
                    "total_detections_found": self.detection_stats.get("total_detections_found", 0),
                    "average_detection_time_ms": self.detection_stats.get("average_detection_time", 0.0),
                    "cameras_processed": list(video_data.keys())
                },
                "cameras": {
                    camera_id: {
                        "processing_completed": True,
                        "status": "detection_complete"
                    }
                    for camera_id in video_data.keys()
                }
            }
            
            # Send summary via WebSocket
            success = await binary_websocket_manager.send_json_message(
                str(task_id), summary_message, MessageType.TRACKING_UPDATE
            )
            
            if success:
                logger.info(f"📡 DETECTION STREAMING: Sent detection summary for task {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ DETECTION STREAMING: Error streaming detection results: {e}")
            return False
    
    async def send_detection_update(self, task_id: uuid.UUID, camera_id: str, frame: np.ndarray,
                                   detection_data: Dict[str, Any], frame_number: int):
        """
        Send detection update via WebSocket with Phase 4 spatial intelligence data.
        
        Follows DETECTION.md schema with populated homography and handoff data
        replacing the previous static null values with actual spatial intelligence results.
        """
        import time as _time
        _func_start = _time.perf_counter()
        
        try:
            # Step 1: Apply focus filter
            _focus_start = _time.perf_counter()
            payload_data = detection_data
            focus_applied = False
            if task_id is not None:
                payload_data, focus_applied = self._apply_focus_filter(task_id, camera_id, detection_data)
            _focus_time = (_time.perf_counter() - _focus_start) * 1000

            # Retrieve handoff zones for visualization if available
            handoff_zones = None
            if self.handoff_service and hasattr(self.handoff_service, 'camera_zones'):
                 handoff_zones = self.handoff_service.camera_zones.get(camera_id)

            # Step 2: MJPEG Streaming - encode and push frame
            _mjpeg_start = _time.perf_counter()
            if frame is not None:
                jpeg_bytes = self.annotator.frame_to_jpeg_bytes(frame)
                if jpeg_bytes:
                    await mjpeg_streamer.push_frame(str(task_id), camera_id, jpeg_bytes)
            _mjpeg_time = (_time.perf_counter() - _mjpeg_start) * 1000

            # Step 3: Prepare homography and mapping data
            _mapping_start = _time.perf_counter()
            homography_data = None
            if self.homography_service:
                homography_data = self.homography_service.get_homography_data(camera_id)
            
            # Phase 4: Prepare mapping coordinates data WITH TRAILS
            mapping_coordinates: List[Dict[str, Any]] = []
            for detection in payload_data.get("detections", []):
                    # Accept valid coordinates, including (0,0). Validate bounds if possible.
                    if "map_coords" in detection:
                        mx = detection["map_coords"].get("map_x")
                        my = detection["map_coords"].get("map_y")
                        is_valid = (
                            mx is not None and my is not None and
                            (self.homography_service.validate_map_coordinate(camera_id, mx, my) if self.homography_service else True)
                        )
                        if not is_valid:
                            continue
                        
                        # GET trail (already updated in _enhance_tracks_with_spatial_intelligence)
                        trail = self.trail_service.get_trail(
                            camera_id=camera_id,
                            detection_id=detection["detection_id"]
                        )
                        
                        # Convert trail to frontend format
                        trail_data = [
                            {
                                "x": point.map_x,
                                "y": point.map_y,
                                "frame_offset": point.frame_offset,
                                "timestamp": point.timestamp.isoformat()
                            }
                            for point in trail[:3]
                        ]
                        
                        # Extract foot point used for projection
                        foot_point = {
                            "image_x": detection["bbox"]["center_x"],
                            "image_y": detection["bbox"]["y2"]
                        }
                        
                        coord_data = {
                            "detection_id": detection["detection_id"],
                            "map_x": mx,
                            "map_y": my,
                            "projection_successful": True,
                            "foot_point": foot_point,
                            "coordinate_system": detection.get("spatial_data", {}).get("coordinate_system", "bev_map_meters"),
                            "trail": trail_data
                        }
                        mapping_coordinates.append(coord_data)
            _mapping_time = (_time.perf_counter() - _mapping_start) * 1000

            # Step 4: Create WebSocket message
            _msg_start = _time.perf_counter()
            detection_message = {
                "type": MessageType.TRACKING_UPDATE.value,
                "task_id": str(task_id),
                "camera_id": camera_id,
                "global_frame_index": frame_number,
                "timestamp_processed_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "detection_streaming",
                "message_type": "detection_update",
                "camera_data": {
                    "frame_image_base64": None,
                    "original_frame_base64": None,
                    "tracks": payload_data.get("tracks", []),
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0], 
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "detection_data": payload_data,
                "future_pipeline_data": {
                    "tracking_data": {
                        "track_count": len(payload_data.get("tracks", []))
                    } if payload_data.get("tracks") is not None else None,
                    "homography_data": homography_data,
                    "mapping_coordinates": mapping_coordinates if mapping_coordinates else None
                }
            }

            if focus_applied:
                detection_message["focus"] = payload_data.get("focus_metadata", {})
            _msg_time = (_time.perf_counter() - _msg_start) * 1000
            
            # Optional on-disk frame cache (sampled)
            try:
                if getattr(settings, 'STORE_EXTRACTED_FRAMES', False) and annotated_frame is not None:
                    sample_rate = int(getattr(settings, 'FRAME_CACHE_SAMPLE_RATE', 0))
                    if sample_rate and frame_number % sample_rate == 0:
                        cache_dir = Path(getattr(settings, 'FRAME_CACHE_DIR', './extracted_frames')) / str(task_id) / str(camera_id)
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        out_path = cache_dir / f"frame_{frame_number:06d}.jpg"
                        
                        bytes_to_write = locals().get('jpeg_bytes')
                        if not bytes_to_write:
                             bytes_to_write = self.annotator.frame_to_jpeg_bytes(annotated_frame)
                        
                        if bytes_to_write:
                            await asyncio.to_thread(out_path.write_bytes, bytes_to_write)
            except Exception as e:
                pass

            # Step 5: Send via WebSocket
            _ws_start = _time.perf_counter()
            success = await binary_websocket_manager.send_json_message(
                str(task_id), detection_message, MessageType.TRACKING_UPDATE
            )
            _ws_time = (_time.perf_counter() - _ws_start) * 1000
            
            _func_total = (_time.perf_counter() - _func_start) * 1000
            
            # Log granular timing
            speed_optimize_logger.info(
                "[SPEED_DEBUG] send_detection_update | Cam=%s Frame=%d | Total=%.1fms | FocusFilter=%.1fms MJPEG=%.1fms Mapping=%.1fms MsgBuild=%.1fms WsSend=%.1fms",
                camera_id, frame_number, _func_total,
                _focus_time, _mjpeg_time, _mapping_time, _msg_time, _ws_time
            )
            
            if success:
                self.detection_stats["websocket_messages_sent"] += 1
            else:
                logger.warning(f"📡 DETECTION UPDATE: Failed to send detection update for task {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ DETECTION UPDATE: Error sending detection update: {e}")
            return False

    async def _emit_frontend_events(self, task_id: Optional[uuid.UUID], camera_id: str, frame_number: int, tracks: List[Dict[str, Any]]):
        """Emit auxiliary WebSocket events for frontend: trajectories and tracking stats."""
        # 1) Emit person_trajectory_update
        try:
            trajectory_payload = {
                "type": MessageType.TRACKING_UPDATE.value,
                "message_type": "person_trajectory_update",
                "camera_id": camera_id,
                "global_frame_index": frame_number,
                "trajectories": [
                    {
                        "track_id": t.get("track_id"),
                        "global_id": t.get("global_id"),
                        "trajectory": t.get("trajectory", [])
                    }
                    for t in tracks if t.get("trajectory")
                ]
            }
            if task_id is not None:
                await binary_websocket_manager.send_json_message(str(task_id), trajectory_payload, MessageType.TRACKING_UPDATE)
        except Exception:
            pass

        # 2) Emit tracking_statistics snapshot occasionally
        try:
            if frame_number % 30 == 0:  # every ~1-2 seconds depending on FPS
                stats_payload = {
                    "type": MessageType.TRACKING_UPDATE.value,
                    "message_type": "tracking_statistics",
                    "camera_id": camera_id,
                    "global_frame_index": frame_number,
                    "stats": self.get_tracking_stats()
                }
                if task_id is not None:
                    await binary_websocket_manager.send_json_message(str(task_id), stats_payload, MessageType.TRACKING_UPDATE)
        except Exception:
            pass

    def _convert_detections_to_boxmot_format(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Convert detection dicts to BoxMOT format array [x1, y1, x2, y2, conf, cls]."""
        try:
            if not detections:
                return np.empty((0, 6), dtype=np.float32)
            rows: List[List[float]] = []
            for det in detections:
                bbox = det.get("bbox", {})
                x1 = float(bbox.get("x1", 0.0))
                y1 = float(bbox.get("y1", 0.0))
                x2 = float(bbox.get("x2", 0.0))
                y2 = float(bbox.get("y2", 0.0))
                conf = float(det.get("confidence", 0.0))
                cls_id = float(det.get("class_id", 0))
                rows.append([x1, y1, x2, y2, conf, cls_id])
            return np.array(rows, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed converting detections to tracker format: {e}")
            return np.empty((0, 6), dtype=np.float32)

    def _convert_boxmot_to_track_data(self, tracked_dets: np.ndarray, camera_id: str) -> List[Dict[str, Any]]:
        """Convert BoxMOT output to our track data format expected by the frontend."""
        tracks: List[Dict[str, Any]] = []
        try:
            if tracked_dets is None or tracked_dets.size == 0:
                return tracks
            # Expected columns (best-effort, BoxMOT versions vary):
            # [x1, y1, x2, y2, track_id, conf, cls, ...]
            for row in tracked_dets:
                x1 = float(row[0])
                y1 = float(row[1])
                x2 = float(row[2])
                y2 = float(row[3])
                track_id = int(row[4]) if row.shape[0] > 4 else -1
                confidence = float(row[5]) if row.shape[0] > 5 else None
                class_id = int(row[6]) if row.shape[0] > 6 else 0
                track = {
                    "track_id": track_id,
                    "global_id": None,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    # Spatial fields populated later
                    "map_coords": None,
                    "trajectory": [],
                    # Tracker-provided metadata (placeholders if unavailable)
                    "age": None,
                    "status": "active",
                    # Identity placeholders
                    "reid_confidence": None,
                    "last_seen_camera": camera_id,
                    "is_focused": False
                }
                tracks.append(track)
            return tracks
        except Exception as e:
            logger.warning(f"Failed converting tracker output to track data: {e}")
            return []

    def _associate_detections_with_tracks(
        self,
        camera_id: str,
        detections: Optional[List[Dict[str, Any]]],
        tracks: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Promote tracker identifiers onto detection entries for stability.

        Matches detections to tracks using IoU and center distance so the frontend can
        rely on a consistent `detection_id` instead of frame-local indexes.
        
        OPTIMIZED: Uses spatial grid indexing for O(n) average-case instead of O(n*m).
        """

        if not detections or not tracks:
            return

        assigned_track_ids: Set[int] = set()
        
        # Build spatial grid index for tracks (100px cell size)
        CELL_SIZE = 100.0
        track_grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        
        for track in tracks:
            track_bbox = track.get("bbox_xyxy")
            if not track_bbox or len(track_bbox) != 4:
                continue
            if track.get("track_id") is None:
                continue
            
            # Index by center cell
            cx = (float(track_bbox[0]) + float(track_bbox[2])) / 2.0
            cy = float(track_bbox[3])  # foot point
            cell = (int(cx // CELL_SIZE), int(cy // CELL_SIZE))
            
            if cell not in track_grid:
                track_grid[cell] = []
            track_grid[cell].append(track)

        for detection in detections:
            bbox = detection.get("bbox") or {}
            x1 = bbox.get("x1")
            y1 = bbox.get("y1")
            x2 = bbox.get("x2")
            y2 = bbox.get("y2")

            if None in (x1, y1, x2, y2):
                continue

            detection_bbox = [float(x1), float(y1), float(x2), float(y2)]
            det_center = ((detection_bbox[0] + detection_bbox[2]) / 2.0, detection_bbox[3])
            
            # Get detection's cell and check neighboring cells
            det_cell = (int(det_center[0] // CELL_SIZE), int(det_center[1] // CELL_SIZE))
            
            best_track: Optional[Dict[str, Any]] = None
            best_iou = 0.0
            best_center_dist = float("inf")

            # Check 3x3 neighborhood of cells for potential matches
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    neighbor_cell = (det_cell[0] + dx, det_cell[1] + dy)
                    candidate_tracks = track_grid.get(neighbor_cell, [])
                    
                    for track in candidate_tracks:
                        track_bbox = track.get("bbox_xyxy")
                        track_id = track.get("track_id")

                        iou = self._calculate_iou(track_bbox, detection_bbox)
                        track_center = ((float(track_bbox[0]) + float(track_bbox[2])) / 2.0, float(track_bbox[3]))
                        center_dist = math.hypot(track_center[0] - det_center[0], track_center[1] - det_center[1])

                        already_assigned = track_id in assigned_track_ids
                        if (
                            iou > best_iou + 1e-6
                            or (
                                abs(iou - best_iou) <= 1e-6
                                and (center_dist < best_center_dist - 1e-6 or (center_dist <= best_center_dist + 1e-6 and not already_assigned))
                            )
                        ):
                            best_track = track
                            best_iou = iou
                            best_center_dist = center_dist

            if not best_track:
                continue

            track_id = best_track.get("track_id")
            if track_id is None:
                continue

            if best_iou < 0.1 and best_center_dist > 96.0:
                # Avoid weak matches that would cause identity jumps.
                continue

            assigned_track_ids.add(track_id)

            original_detection_id = detection.get("detection_id")
            if original_detection_id and original_detection_id != f"track_{track_id:03d}":
                detection.setdefault("metadata", {})
                if isinstance(detection["metadata"], dict):
                    detection["metadata"].setdefault("original_detection_id", original_detection_id)
                else:
                    detection["metadata"] = {"original_detection_id": original_detection_id}

            detection["detection_id"] = f"track_{track_id:03d}"
            detection["track_id"] = track_id

            global_id = best_track.get("global_id")
            detection["global_id"] = str(global_id) if global_id is not None else None
            detection["tracking_key"] = f"{camera_id}:track:{track_id}"
            detection["track_assignment_iou"] = round(best_iou, 4)
            detection["track_assignment_center_distance"] = round(best_center_dist, 2)

            # Priority 2 Enhancement: Pre-matched detection+track data
            # Embed full track data on detection so frontend can display directly
            # without running expensive IoU matching every frame
            detection["bbox_xyxy"] = best_track.get("bbox_xyxy")
            detection["track_confidence"] = best_track.get("confidence")
            detection["is_matched"] = best_track.get("is_matched", False)
            detection["class_id"] = best_track.get("class_id", detection.get("class_id", 0))

            # logger.debug(
            #     "Detection %s (camera=%s) associated with track %s -> detection_id=%s iou=%.3f",
            #     original_detection_id,
            #     camera_id,
            #     track_id,
            #     detection["detection_id"],
            #     best_iou,
            # )

    @staticmethod
    def _calculate_iou(box_a: List[float], box_b: List[float]) -> float:
        try:
            ax1, ay1, ax2, ay2 = map(float, box_a)
            bx1, by1, bx2, by2 = map(float, box_b)
        except (TypeError, ValueError):
            return 0.0

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    FOCUS_IOU_UPDATE_THRESHOLD: float = 0.55

    def _apply_focus_filter(
        self,
        task_id: uuid.UUID,
        camera_id: str,
        detection_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        focus_state = focus_tracking_handler.get_focus_state(str(task_id))
        if not focus_state or not focus_state.has_active_focus():
            return detection_data, False

        focus_target = focus_state.get_focus_target_summary()
        if not focus_target:
            return detection_data, False

        filtered = copy.deepcopy(detection_data)
        filtered_metadata = filtered.setdefault("focus_metadata", {})
        filtered_metadata.update(focus_target)

        # Cross-camera tracking: filter by global_id match, not camera_id
        focused_global_id = focus_target.get("focused_person_id")
        
        # Find tracks with matching global_id in this camera
        matching_tracks = [
            t for t in detection_data.get("tracks", [])
            if t.get("global_id") == focused_global_id
        ]
        
        if not matching_tracks:
            # No matching global_id in this camera - hide all tracks
            filtered["detections"] = []
            filtered["tracks"] = []
            filtered["detection_count"] = 0
            filtered_metadata["active_on_camera"] = False
            return filtered, True
        
        filtered_metadata["active_on_camera"] = True

        target_track_id = focus_target.get("track_id")
        target_detection_id = focus_target.get("detection_id")
        focus_bbox: Optional[List[float]] = None

        if focus_target.get("bbox"):
            bbox_dict = focus_target["bbox"]
            focus_bbox = [
                bbox_dict.get("x1"),
                bbox_dict.get("y1"),
                bbox_dict.get("x2"),
                bbox_dict.get("y2"),
            ]

        best_track: Optional[Dict[str, Any]] = None
        best_track_iou = -1.0
        track_candidates: List[Dict[str, Any]] = []
        # Iterate over matching_tracks (filtered by global_id) instead of all tracks
        for track in matching_tracks:
            track_bbox = track.get("bbox_xyxy")
            if track_bbox is None:
                continue

            iou = self._calculate_iou(track_bbox, focus_bbox) if focus_bbox is not None else -1.0
            track_candidates.append({
                "track_id": track.get("track_id"),
                "iou": iou,
            })

            if target_track_id is not None and track.get("track_id") == target_track_id:
                best_track = track
                best_track_iou = 1.0
                focus_bbox = track_bbox
                break

            if focus_bbox is not None and iou > best_track_iou:
                best_track = track
                best_track_iou = iou

        if best_track is None and matching_tracks:
            best_track = matching_tracks[0]
            best_track_iou = 0.0
            focus_bbox = best_track.get("bbox_xyxy")

        filtered_tracks: List[Dict[str, Any]] = []
        if best_track is not None:
            filtered_metadata["track_id"] = best_track.get("track_id")
            track_copy = copy.deepcopy(best_track)
            track_copy["is_focused"] = True
            filtered_tracks.append(track_copy)
            if track_copy.get("bbox_xyxy"):
                focus_state.record_observation(
                    camera_id=camera_id,
                    bbox=track_copy.get("bbox_xyxy"),
                    detection_id=filtered_metadata.get("detection_id"),
                    track_id=track_copy.get("track_id"),
                    confidence=track_copy.get("confidence"),
                )

        if focus_bbox is None:
            return filtered, True

        best_detection: Optional[Dict[str, Any]] = None
        best_iou = -1.0
        target_detection_match: Optional[Dict[str, Any]] = None
        target_detection_iou = -1.0

        reference_bbox = best_track.get("bbox_xyxy") if best_track is not None else focus_bbox
        detection_candidates: List[Dict[str, Any]] = []

        for detection in detection_data.get("detections", []):
            bbox = detection.get("bbox") or {}
            det_bbox = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]
            if reference_bbox is None or any(v is None for v in det_bbox):
                continue

            iou = self._calculate_iou(reference_bbox, det_bbox)
            detection_candidates.append({
                "detection_id": detection.get("detection_id"),
                "iou": iou,
            })

            if detection.get("detection_id") == target_detection_id:
                target_detection_match = detection
                target_detection_iou = iou

            if iou > best_iou:
                best_detection = detection
                best_iou = iou

        # Prefer the explicit detection id when available
        chosen_detection = None
        chosen_iou = -1.0
        if target_detection_match is not None:
            chosen_detection = target_detection_match
            chosen_iou = target_detection_iou
        else:
            chosen_detection = best_detection
            chosen_iou = best_iou

        filtered_detections: List[Dict[str, Any]] = []
        if chosen_detection and (chosen_iou >= self.FOCUS_IOU_UPDATE_THRESHOLD or chosen_iou == 1.0):
            det_copy = copy.deepcopy(chosen_detection)
            det_copy["is_focused"] = True
            filtered_detections.append(det_copy)
            bbox = det_copy.get("bbox") or {}
            focus_state.record_observation(
                camera_id=camera_id,
                bbox=[bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")],
                detection_id=det_copy.get("detection_id"),
                track_id=det_copy.get("track_id"),
                confidence=det_copy.get("confidence"),
            )
            filtered_metadata["detection_id"] = det_copy.get("detection_id")
            best_iou = chosen_iou
        else:
            if focus_bbox:
                x1, y1, x2, y2 = focus_bbox
                width = (x2 - x1) if None not in focus_bbox else None
                height = (y2 - y1) if None not in focus_bbox else None
                center_x = (x1 + x2) / 2 if None not in (x1, x2) else None
                center_y = (y1 + y2) / 2 if None not in (y1, y2) else None
                filtered_detections.append({
                    "detection_id": target_detection_id or "focus_placeholder",
                    "class_name": "person",
                    "class_id": 0,
                    "confidence": focus_target.get("confidence", 1.0),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": width,
                        "height": height,
                        "center_x": center_x,
                        "center_y": center_y,
                    },
                    "is_focused": True,
                })
                focus_state.record_observation(
                    camera_id=camera_id,
                    bbox=focus_bbox,
                    detection_id=target_detection_id,
                    track_id=target_track_id,
                    confidence=focus_target.get("confidence"),
                )
                filtered_metadata["detection_id"] = target_detection_id
                best_iou = target_detection_iou if target_detection_match is not None else best_iou

        filtered_metadata["match_iou"] = best_iou
        filtered["tracks"] = filtered_tracks
        filtered["detections"] = filtered_detections
        filtered["detection_count"] = len(filtered_detections)

        logger.info(
            'Focus filter applied: task=%s camera=%s selected_track=%s selected_det=%s match_iou=%.3f',
            str(task_id),
            camera_id,
            filtered_metadata.get('track_id'),
            filtered_metadata.get('detection_id'),
            best_iou,
        )
        pass # logger.debug(
            #     'Focus filter candidates',
            #     extra={
            #         'task_id': str(task_id),
            #         'camera_id': camera_id,
            #         'track_candidates': track_candidates,
            #         'detection_candidates': detection_candidates,
            #         'best_track_iou': best_track_iou,
            #         'target_track_id': target_track_id,
            #         'target_detection_id': target_detection_id,
            #     },
            # )

        return filtered, True

    async def _enhance_tracks_with_spatial_intelligence(self, tracks: List[Dict[str, Any]], camera_id: str, frame_number: int) -> List[Dict[str, Any]]:
        """Compute map coordinates and short trajectories for tracks using homography and trail service."""
        # DEBUG TRACE: Verify execution
        # DEBUG TRACE: Verify execution
        # if frame_number % 30 == 0:
        #      logger.info(f"EnhanceTracks: Processing {len(tracks)} tracks for {camera_id}. HomographyService? {self.homography_service is not None}")
        
        if not tracks:
            return []
        enhanced: List[Dict[str, Any]] = []
        for track in tracks:
            try:
                x1, y1, x2, y2 = track["bbox_xyxy"]
                center_x = (x1 + x2) / 2.0
                foot_y = y2
                map_coords = None
                if self.homography_service:
                    projected = self.homography_service.project_to_map(camera_id, (center_x, foot_y))
                    if projected:
                        mx, my = projected
                        if self.homography_service.validate_map_coordinate(camera_id, mx, my):
                            map_coords = {"map_x": mx, "map_y": my}
                            # Update trajectory trail keyed by track
                            trail = await self.trail_service.update_trail(
                                camera_id=camera_id,
                                detection_id=f"track_{track['track_id']}",
                                map_x=mx,
                                map_y=my
                            )
                            trajectory = [[p.map_x, p.map_y] for p in trail[:3]]
                            track["trajectory"] = trajectory
                        else:
                            # DEBUG TRACE: Log rejection
                            logger.info(f"SpatialEnhancement: Rejected coords for cam {camera_id}: ({mx:.2f}, {my:.2f})")
                track["map_coords"] = map_coords
                enhanced.append(track)
            except Exception as e:
                pass # logger.debug(f"Spatial enhancement failed for track {track.get('track_id')}: {e}")
                enhanced.append(track)
        return enhanced
    
    async def process_detection_task_simple(self, task_id: uuid.UUID, environment_id: str):
        """
        Simplified detection processing pipeline (detection-only).
        
        Focuses only on person detection with YOLO, sends results via WebSocket
        with static null values for future pipeline features (homography-only details).
        """
        pipeline_start = time.time()
        
        try:
            logger.info(f"🎬 DETECTION PIPELINE: Starting simplified detection pipeline for task {task_id}")
            
            # Initialize detection services
            await self._update_task_status(task_id, "INITIALIZING", 0.10, "Initializing YOLO detection services")
            services_initialized = await self.initialize_detection_services(environment_id)
            if not services_initialized:
                raise RuntimeError("Failed to initialize detection services")
            
            # Download video data
            await self._update_task_status(task_id, "DOWNLOADING", 0.30, "Downloading video data")
            video_data = await self._download_video_data(environment_id)
            if not video_data:
                raise RuntimeError("Failed to download video data")
            
            # Extract frames for processing
            await self._update_task_status(task_id, "EXTRACTING", 0.45, "Extracting frames from video data")
            frames_extracted = await self._extract_raw_frames(task_id, video_data)
            if not frames_extracted:
                raise RuntimeError("Failed to extract frames")
            
            # Process frames with detection only
            await self._update_task_status(task_id, "PROCESSING", 0.60, "Processing frames with YOLO detection")
            success = await self._process_frames_simple_detection(task_id, environment_id, video_data)
            if self._task_marked_stopped(task_id):
                logger.info(f"🛑 DETECTION PIPELINE: Task {task_id} stopped early (no active WebSocket clients)")
                return
            if not success:
                raise RuntimeError("Failed to process frames with detection")
            
            # Complete
            await self._update_task_status(task_id, "COMPLETED", 1.0, "Detection pipeline completed successfully")
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"✅ DETECTION PIPELINE: Pipeline completed in {pipeline_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ DETECTION PIPELINE: Error in detection pipeline: {e}")
            await self._update_task_status(task_id, "FAILED", 0.0, f"Detection pipeline failed: {str(e)}")
            
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            if environment_id in self.environment_tasks:
                del self.environment_tasks[environment_id]
            # Explicit tracker teardown for this task
            try:
                if self.tracker_factory:
                    await self.tracker_factory.clear_trackers_for_task(task_id)
            except Exception:
                pass
            self._clear_client_watch(task_id)
            await self._cleanup_playback_task(task_id)
    
    async def _process_frames_simple_detection(self, task_id: uuid.UUID, environment_id: str, video_data: Dict[str, Any]) -> bool:
        """
        Process frames with simple YOLO detection only.
        
        Simplified version that focuses only on detection and annotation,
        sending WebSocket updates with static null values for future features.
        """
        try:
            logger.info(f"🔍 SIMPLE DETECTION: Starting frame processing with YOLO for task {task_id}")
            
            # Get total frame count
            frame_counts = [data.get("frame_count", 0) for data in video_data.values() if data.get("frame_count", 0) > 0]
            total_frames = min(frame_counts) if frame_counts else 0
            
            if total_frames == 0:
                logger.warning("No frames available for detection processing")
                return False
            
            frame_index = 0
            frames_processed = 0
            aborted_due_to_no_clients = False
            
            # Main processing loop
            while frame_index < total_frames:
                if task_id not in self.active_tasks:
                    logger.info(f"🔍 SIMPLE DETECTION: Task {task_id} was stopped")
                    break

                await self._wait_for_playback(task_id)

                if task_id not in self.active_tasks:
                    logger.info(f"🔍 SIMPLE DETECTION: Task {task_id} stopped during pause wait")
                    break

                if not self._should_continue_stream(task_id, detection_mode=True):
                    aborted_due_to_no_clients = True
                    break
                
                # FRAME SKIP: Only process every Nth frame for performance
                frame_skip = getattr(settings, 'FRAME_SKIP', 1)
                if frame_skip > 1 and frame_index % frame_skip != 0:
                    frame_index += 1
                    continue
                
                # Process all cameras for current frame using BATCH INFERENCE
                frame_camera_data = {} # Buffer for (frame, detection_data)
                frame_debug_payload: Dict[str, Dict[str, Any]] = {}
                any_frame_processed = False
                
                import time as _time
                _batch_start = _time.perf_counter()
                _read_time = 0.0
                _batch_det_time = 0.0
                _track_time = 0.0
                _space_match_time = 0.0
                _ws_time = 0.0
                
                # === PHASE 1: Collect all frames from cameras ===
                _read_start = _time.perf_counter()
                frames_batch: List[np.ndarray] = []
                camera_order: List[Tuple[str, np.ndarray]] = []
                
                for camera_id, data in video_data.items():
                    cap = data.get("video_capture")
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            if self.enable_debug_reprojection:
                                self._debug_frame_store[(camera_id, frame_index)] = frame.copy()
                            frames_batch.append(frame)
                            camera_order.append((camera_id, frame))
                            any_frame_processed = True
                        else:
                            pass # End of video
                            break
                    else:
                        break
                _read_time = (_time.perf_counter() - _read_start) * 1000
                
                # Check if all cameras finished
                if not any_frame_processed or not frames_batch:
                    logger.info(f"🔍 SIMPLE DETECTION: All cameras finished at frame {frame_index}")
                    break
                
                # === PHASE 2: Run batch detection (single GPU call) ===
                _det_start = _time.perf_counter()
                
                # Get detector (use first camera's environment, they should all match)
                first_camera_id = camera_order[0][0] if camera_order else None
                env_for_batch = self._get_environment_for_camera(first_camera_id) or "default" if first_camera_id else "default"
                detector = self.detectors_by_env.get(env_for_batch) or self.detector
                
                if detector and len(frames_batch) > 1:
                    # Use batch inference for multiple cameras
                    try:
                        batch_detections = await detector.detect_batch(frames_batch)
                    except Exception as e:
                        logger.error(f"Batch detection failed, falling back to sequential: {e}")
                        # Fallback: sequential detection
                        batch_detections = []
                        for frame in frames_batch:
                            try:
                                dets = await detector.detect(frame)
                                batch_detections.append(dets)
                            except Exception:
                                batch_detections.append([])
                elif detector:
                    # Single camera, use regular detect
                    batch_detections = [await detector.detect(frames_batch[0])]
                else:
                    batch_detections = [[] for _ in frames_batch]
                    
                _batch_det_time = (_time.perf_counter() - _det_start) * 1000
                
                # === PHASE 3: Process each camera with pre-computed detections ===
                _track_start = _time.perf_counter()
                for (camera_id, frame), raw_detections in zip(camera_order, batch_detections):
                    # Convert raw detections to enhanced format with spatial intelligence
                    detection_data = self._process_detections_to_format(
                        raw_detections, frame, camera_id, frame_index
                    )
                    
                    # Run tracking on the detections (without re-running detection)
                    detection_data = await self._process_tracking_with_predetected(
                        frame, camera_id, frame_index, detection_data
                    )
                    
                    frame_camera_data[camera_id] = (frame, detection_data)
                _track_time = (_time.perf_counter() - _track_start) * 1000


                # 2. Run Space-Based Matching (Cross-Camera)
                _space_start = _time.perf_counter()
                if self.space_based_matcher and self.space_based_matcher.enabled:
                    try:
                        # Extract detection dicts for matcher
                        camera_detections_map = {cid: ddata for cid, (_, ddata) in frame_camera_data.items()}
                        self.space_based_matcher.match_across_cameras(camera_detections_map)
                    except Exception as e:
                         logger.error(f"Space-based matching failed in simple loop: {e}")

                # Update 'is_matched' flag for correct coloring (Fix for Simple Loop)
                if self.space_based_matcher:
                    for camera_id, (frame, detection_data) in frame_camera_data.items():
                         for track in detection_data.get("tracks", []):
                            gid = track.get("global_id")
                            if gid:
                                is_shared = self.space_based_matcher.is_global_id_shared(gid)
                                track["is_matched"] = is_shared
                                # DEBUG LOG (Temporary)
                                pass # logger.warning(f"COLOR DEBUG SIMPLE: Cam {camera_id} Track {track.get('track_id')} Global {gid} Shared={is_shared}")
                                pass # logger.info(f"[INFO]COLOR DEBUG SIMPLE: Cam {camera_id} Track {track.get('track_id')} Global {gid} Shared={is_shared}")
                _space_match_time = (_time.perf_counter() - _space_start) * 1000

                # 3. Send Updates and Emit Debug
                _ws_start = _time.perf_counter()
                for camera_id, (frame, detection_data) in frame_camera_data.items():
                    # Collect debug points (now with global_ids injected)
                    if self.enable_debug_reprojection:
                        world_points = self._collect_world_points(
                            detection_data=detection_data,
                            camera_id=camera_id,
                            frame_number=frame_index,
                        )
                        if world_points:
                            frame_debug_payload[camera_id] = {"world_points": world_points}
                    
                    # Send WebSocket update
                    await self.send_detection_update(
                        task_id, camera_id, frame, detection_data, frame_index
                    )
                    if getattr(settings, 'ENABLE_ENHANCED_VISUALIZATION', True):
                         await self._emit_frontend_events(task_id, camera_id, frame_index, detection_data.get("tracks", []))
                    
                    frames_processed += 1
                _ws_time = (_time.perf_counter() - _ws_start) * 1000

                # 4. Emit Debug Frame
                if self.enable_debug_reprojection and frame_debug_payload:
                    self._emit_reprojection_debug_frame(
                        environment_id=environment_id,
                        frame_number=frame_index,
                        frame_payload=frame_debug_payload,
                    )
                    for cam_key in frame_debug_payload.keys():
                        self._debug_frame_store.pop((cam_key, frame_index), None)
                elif self.enable_debug_reprojection:
                    for key in list(self._debug_frame_store.keys()):
                        if key[1] == frame_index:
                            self._debug_frame_store.pop(key, None)
                
                # Total batch time
                _batch_time = (_time.perf_counter() - _batch_start) * 1000
                
                # Log frame batch timing EVERY FRAME for debugging
                speed_optimize_logger.info(
                    "[SPEED_DEBUG] BATCH Frame=%d | Total=%.1fms | Read=%.1fms BatchDet=%.1fms Track=%.1fms SpaceMatch=%.1fms WsSend=%.1fms | Cams=%d FPS=%.1f",
                    frame_index, _batch_time,
                    _read_time, _batch_det_time, _track_time, _space_match_time, _ws_time,
                    len(frame_camera_data), 1000.0 / _batch_time if _batch_time > 0 else 0
                )
                
                # Update progress every 30 frames
                if frame_index % 30 == 0:
                    progress = 0.60 + (frame_index / total_frames) * 0.35  # 0.60-0.95 range
                    await self._update_task_status(
                        task_id, "PROCESSING", progress,
                        f"Processed frame {frame_index}/{total_frames} - {frames_processed} detections sent"
                    )

                await self._record_playback_progress(task_id, frame_index)

                frame_index += 1
                
                # Yield to event loop (no delay for max FPS)
                await asyncio.sleep(0)
            
            # Cleanup video captures
            for data in video_data.values():
                cap = data.get("video_capture")
                if cap:
                    cap.release()

            if aborted_due_to_no_clients:
                reason = "Detection stopped - no active WebSocket clients"
                self._mark_task_stopped_due_to_idle_clients(task_id, reason)
                logger.info(f"🛑 SIMPLE DETECTION: {reason} (task {task_id})")
                return True

            return True

        except Exception as e:
            logger.error(f"❌ SIMPLE DETECTION: Error in frame processing: {e}")
            return False
    
    def _process_detections_to_format(
        self, 
        raw_detections: List, 
        frame: np.ndarray, 
        camera_id: str, 
        frame_number: int
    ) -> Dict[str, Any]:
        """
        Convert raw Detection objects to enhanced format with spatial intelligence.
        
        This is extracted from process_frame_with_detection to allow batch detection
        results to be processed without re-running the detector.
        
        Args:
            raw_detections: List of Detection objects from detector
            frame: Video frame for dimension reference
            camera_id: Camera identifier
            frame_number: Frame sequence number
            
        Returns:
            Detection data dictionary with enhanced spatial metadata
        """
        import time as _time
        _func_start = _time.perf_counter()
        
        frame_height, frame_width = frame.shape[:2]
        
        enhanced_detections = []
        for i, detection in enumerate(raw_detections):
            bbox_dict = {
                "x1": detection.bbox.x1,
                "y1": detection.bbox.y1, 
                "x2": detection.bbox.x2,
                "y2": detection.bbox.y2,
                "width": detection.bbox.x2 - detection.bbox.x1,
                "height": detection.bbox.y2 - detection.bbox.y1,
                "center_x": (detection.bbox.x1 + detection.bbox.x2) / 2,
                "center_y": (detection.bbox.y1 + detection.bbox.y2) / 2
            }
            
            # Apply spatial intelligence
            map_coords = {"map_x": 0, "map_y": 0}
            projection_success = False
            transformation_quality: Optional[float] = None
            handoff_triggered = False
            candidate_cameras = []
            search_roi_payload: Optional[Dict[str, Optional[float]]] = None

            bottom_point: Optional[ImagePoint] = None
            try:
                bottom_point = self.bottom_point_extractor.extract_point(
                    bbox_x=bbox_dict["x1"],
                    bbox_y=bbox_dict["y1"],
                    bbox_width=bbox_dict["width"],
                    bbox_height=bbox_dict["height"],
                    camera_id=CameraID(camera_id),
                    person_id=None,
                    frame_number=frame_number,
                    timestamp=None,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            except ValueError:
                pass

            world_point: Optional[WorldPoint] = None
            if self.world_plane_transformer and bottom_point:
                try:
                    world_point = self.world_plane_transformer.transform_point(bottom_point)
                    transformation_quality = world_point.transformation_quality
                    map_coords = {"map_x": world_point.x, "map_y": world_point.y}
                    projection_success = transformation_quality >= 0.5
                except (KeyError, ValueError):
                    pass

            # Homography coordinate transformation fallback (legacy)
            if not projection_success and self.homography_service and bottom_point:
                fallback_point = (bottom_point.x, bottom_point.y)
                projected_coords = self.homography_service.project_to_map(camera_id, fallback_point)
                if projected_coords:
                    candidate_map_x, candidate_map_y = projected_coords
                    if self.homography_service.validate_map_coordinate(camera_id, candidate_map_x, candidate_map_y):
                        map_coords = {"map_x": candidate_map_x, "map_y": candidate_map_y}
                        projection_success = True
                        transformation_quality = transformation_quality or 0.8

            if projection_success and map_coords:
                try:
                    roi = self.roi_calculator.calculate_roi(
                        (map_coords["map_x"], map_coords["map_y"]),
                        time_elapsed=0.0,
                        transformation_quality=transformation_quality if transformation_quality is not None else 1.0,
                        shape=self.roi_shape,
                        source_camera=camera_id,
                        dest_camera=None,
                        person_id=None,
                        timestamp=None,
                    )
                    search_roi_payload = roi.to_dict()
                except Exception:
                    pass

            # Handoff detection
            if self.handoff_service:
                handoff_triggered, candidate_cameras = self.handoff_service.check_handoff_trigger(
                    camera_id, bbox_dict, frame_width, frame_height
                )
            
            enhanced_detection = {
                "detection_id": f"det_{i:03d}",
                "class_name": "person",
                "class_id": 0,
                "confidence": detection.confidence,
                "bbox": bbox_dict,
                "track_id": None,
                "global_id": None,
                "map_coords": map_coords,
                "spatial_data": {
                    "handoff_triggered": handoff_triggered,
                    "candidate_cameras": candidate_cameras,
                    "coordinate_system": "world_meters" if projection_success else None,
                    "projection_successful": projection_success,
                    "transformation_quality": transformation_quality,
                    "search_roi": search_roi_payload,
                }
            }
            
            enhanced_detections.append(enhanced_detection)
        
        detection_data = {
            "detections": enhanced_detections,
            "detection_count": len(raw_detections),
            "processing_time_ms": 0,  # Not tracked in batch mode
            "spatial_metadata": {
                "camera_id": camera_id,
                "frame_dimensions": {"width": frame_width, "height": frame_height},
                "homography_available": self.homography_service is not None,
                "handoff_detection_enabled": self.handoff_service is not None
            }
        }
        
        # Update statistics
        self.detection_stats["total_frames_processed"] += 1
        self.detection_stats["total_detections_found"] += len(raw_detections)
        self.detection_stats["successful_detections"] += 1
        
        _func_total = (_time.perf_counter() - _func_start) * 1000
        
        # Log granular timing
        speed_optimize_logger.info(
            "[SPEED_DEBUG] _process_detections_to_format | Cam=%s Frame=%d | Total=%.1fms | Dets=%d",
            camera_id, frame_number, _func_total, len(raw_detections)
        )
        
        return detection_data

    async def _process_tracking_with_predetected(
        self, 
        frame: np.ndarray, 
        camera_id: str, 
        frame_number: int,
        detection_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run tracking and Re-ID on pre-computed detection data.
        
        This is similar to process_frame_with_tracking but skips the detection
        step since detections are already provided from batch inference.
        
        Args:
            frame: Video frame
            camera_id: Camera identifier
            frame_number: Frame sequence number
            detection_data: Pre-computed detection data from _process_detections_to_format
            
        Returns:
            Detection data with tracks added
        """
        import time as _time
        _func_start = _time.perf_counter()
        
        tracks: List[Dict[str, Any]] = []
        _tracker_init_time = 0.0
        _convert_dets_time = 0.0
        _tracker_update_time = 0.0
        _convert_tracks_time = 0.0
        _spatial_intel_time = 0.0
        _reid_time = 0.0
        
        try:
            if settings.TRACKING_ENABLED:
                # Lazy initialization check
                _init_start = _time.perf_counter()
                if camera_id not in self.camera_trackers:
                    try:
                        tracker = await self.tracker_factory.get_tracker("lazy_init_task", camera_id)
                        if tracker:
                            self.camera_trackers[camera_id] = tracker
                    except Exception as e:
                        logger.error(f"Lazy tracker init failed for {camera_id}: {e}")
                _tracker_init_time = (_time.perf_counter() - _init_start) * 1000

            if settings.TRACKING_ENABLED and camera_id in self.camera_trackers:
                tracker = self.camera_trackers.get(camera_id)
                
                # Convert detections to BoxMOT format
                _conv_start = _time.perf_counter()
                np_dets = self._convert_detections_to_boxmot_format(detection_data.get("detections", []))
                _convert_dets_time = (_time.perf_counter() - _conv_start) * 1000
                
                # Update tracker
                _update_start = _time.perf_counter()
                tracked_np = await tracker.update(np_dets, frame)
                _tracker_update_time = (_time.perf_counter() - _update_start) * 1000
                
                # Convert to track dicts
                _conv_tracks_start = _time.perf_counter()
                tracks = self._convert_boxmot_to_track_data(tracked_np, camera_id)
                _convert_tracks_time = (_time.perf_counter() - _conv_tracks_start) * 1000
                
                # Enhance with spatial intelligence
                _spatial_start = _time.perf_counter()
                tracks = await self._enhance_tracks_with_spatial_intelligence(tracks, camera_id, frame_number)
                _spatial_intel_time = (_time.perf_counter() - _spatial_start) * 1000
                
                # Apply Re-ID
                _reid_start = _time.perf_counter()
                frame_height, frame_width = frame.shape[:2]
                tracks = await self._apply_reid_logic(tracks, frame, camera_id, frame_width, frame_height)
                _reid_time = (_time.perf_counter() - _reid_start) * 1000
                
        except Exception as e:
            logger.warning(f"Tracking failed for camera {camera_id} on frame {frame_number}: {e}")
            tracks = []

        detection_data["tracks"] = tracks
        
        try:
            self._associate_detections_with_tracks(camera_id, detection_data.get("detections"), tracks)
        except Exception:
            pass
        
        _func_total = (_time.perf_counter() - _func_start) * 1000
        
        # Log granular timing
        speed_optimize_logger.info(
            "[SPEED_DEBUG] _process_tracking | Cam=%s Frame=%d | Total=%.1fms | TrackerInit=%.1fms ConvDets=%.1fms TrackerUpdate=%.1fms ConvTracks=%.1fms Spatial=%.1fms ReID=%.1fms | Tracks=%d",
            camera_id, frame_number, _func_total,
            _tracker_init_time, _convert_dets_time, _tracker_update_time, _convert_tracks_time, _spatial_intel_time, _reid_time,
            len(tracks)
        )
        
        return detection_data

    
    def _collect_world_points(
        self,
        detection_data: Dict[str, Any],
        camera_id: str,
        frame_number: int,
    ) -> List[WorldPoint]:
        results: List[WorldPoint] = []

        tracks = detection_data.get("tracks") or []
        
        # DEBUG TRACE
        if self.enable_debug_reprojection and frame_number % 30 == 0 and len(tracks) > 0:
             has_gid = sum(1 for t in tracks if t.get("global_id"))
             logging.info(f"CollectWorldPoints Cam={camera_id}: {len(tracks)} tracks, {has_gid} have global_id")

        for track in tracks:
            map_coords = track.get("map_coords")
            bbox = track.get("bbox_xyxy")
            if not map_coords or not bbox or len(map_coords) < 2 or len(bbox) < 4:
                continue

            try:
                person_id = TrackID(int(track.get("track_id")))
            except (TypeError, ValueError):
                person_id = None

            x1, y1, x2, y2 = map(float, bbox[:4])
            center_x = (x1 + x2) / 2.0
            bottom_y = y2

            try:
                if isinstance(map_coords, dict):
                    wx = float(map_coords.get("map_x", 0.0))
                    wy = float(map_coords.get("map_y", 0.0))
                elif isinstance(map_coords, (list, tuple)) and len(map_coords) >= 2:
                    wx = float(map_coords[0])
                    wy = float(map_coords[1])
                else:
                    continue

                results.append(
                    WorldPoint(
                        x=wx,
                        y=wy,
                        camera_id=CameraID(camera_id),
                        person_id=person_id,
                        original_image_point=(center_x, bottom_y),
                        frame_number=frame_number,
                        timestamp=float(frame_number),
                        global_id=track.get("global_id"),
                        transformation_quality=float(track.get("transformation_quality") or 1.0),
                        is_matched=track.get("is_matched", False),
                    )
                )
            except Exception:
                continue

        detections = detection_data.get("detections") or []
        for det in detections:
            # Skip if this detection is already covered by a track
            if det.get("track_id"):
                continue

            map_coords = det.get("map_coords") or {}
            if not map_coords:
                continue

            bbox_meta = det.get("bbox") or {}
            x1 = float(bbox_meta.get("x1", 0.0))
            y1 = float(bbox_meta.get("y1", 0.0))
            x2 = float(bbox_meta.get("x2", x1))
            y2 = float(bbox_meta.get("y2", y1))
            center_x = (x1 + x2) / 2.0
            bottom_y = y2

            spatial = det.get("spatial_data") or {}
            quality = spatial.get("transformation_quality")
            try:
                results.append(
                    WorldPoint(
                        x=float(map_coords.get("map_x")),
                        y=float(map_coords.get("map_y")),
                        camera_id=CameraID(camera_id),
                        person_id=None,
                        original_image_point=(center_x, bottom_y),
                        frame_number=frame_number,
                        timestamp=float(frame_number),
                        global_id=det.get("global_id"),
                        transformation_quality=float(quality) if quality is not None else 1.0,
                    )
                )
            except Exception:
                continue

        return results

    def _emit_reprojection_debug_frame(
        self,
        environment_id: str,
        frame_number: int,
        frame_payload: Dict[str, Dict[str, Any]],
    ) -> None:
        if not (
            self.enable_debug_reprojection
            and self.reprojection_debugger
            and self.debug_overlay
        ):
            return

        # Prepare in-memory canvases for all cameras involved
        # distinct from _debug_frame_store which holds clean originals
        canvases: Dict[str, np.ndarray] = {}
        
        # Helper to get canvas (lazy copy)
        def get_canvas(cam_id: str) -> Optional[np.ndarray]:
            if cam_id in canvases:
                return canvases[cam_id]
            
            clean_frame = self._debug_frame_store.get((cam_id, frame_number))
            if clean_frame is not None:
                canvases[cam_id] = clean_frame.copy()
                return canvases[cam_id]
            return None

        for source_camera, payload in frame_payload.items():
            world_points: List[WorldPoint] = payload.get("world_points", [])
            if not world_points:
                continue

            for world_point in world_points:
                for dest_camera in frame_payload.keys():
                    if dest_camera == source_camera:
                        continue

                    projected = self._project_world_point_to_camera(
                        environment_id=environment_id,
                        dest_camera=dest_camera,
                        world_point=world_point,
                    )
                    if not projected:
                        continue
                    
                    # Draw on in-memory canvas
                    canvas = get_canvas(dest_camera)
                    if canvas is not None:
                        self.debug_overlay.draw_prediction(
                            frame=canvas,
                            predicted_point=projected,
                            actual_point=None
                        )

        # Save all modified canvases
        for cam_id, canvas in canvases.items():
            self.reprojection_debugger.save_frame(
                camera_id=cam_id,
                frame_number=frame_number,
                frame=canvas
            )

    def _project_world_point_to_camera(
        self,
        environment_id: str,
        dest_camera: str,
        world_point: WorldPoint,
    ) -> Optional[ProjectedImagePoint]:
        if self.inverse_projector:
            projected = self.inverse_projector.project(world_point, dest_camera)
            if projected:
                return projected

        inverse_matrix = self._get_inverse_homography_matrix(environment_id, dest_camera)
        if inverse_matrix is None:
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

        is_matched = False
        if world_point.global_id and self.space_based_matcher:
             is_matched = self.space_based_matcher.is_global_id_shared(world_point.global_id)

        return ProjectedImagePoint(
            x=x_px,
            y=y_px,
            camera_id=dest_camera,
            person_id=world_point.person_id,
            source_camera_id=str(world_point.camera_id),
            world_point=(world_point.x, world_point.y),
            frame_number=world_point.frame_number,
            timestamp=world_point.timestamp,
            global_id=world_point.global_id,
            is_matched=is_matched,
        )

    def _get_inverse_homography_matrix(
        self,
        environment_id: str,
        camera_id: str,
    ) -> Optional[np.ndarray]:
        cache_key = (environment_id, camera_id)
        if cache_key in self._inverse_homography_cache:
            return self._inverse_homography_cache[cache_key]

        if not self.homography_service:
            return None

        try:
            matrix = self.homography_service.get_homography_matrix(environment_id, CameraID(camera_id))
        except Exception as exc:
            pass # logger.debug(
            #     "Homography lookup failed for env %s camera %s: %s",
            #     environment_id,
            #     camera_id,
            #     exc,
            # )
            return None

        if matrix is None:
            return None

        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            # logger.debug(
            #     "Homography matrix for env %s camera %s is not invertible",
            #     environment_id,
            #     camera_id,
            # )
            return None

        self._inverse_homography_cache[cache_key] = inverse
        return inverse

    def _provide_debug_frame(self, camera_id: str, frame_number: int) -> Optional[np.ndarray]:
        frame = self._debug_frame_store.get((camera_id, frame_number))
        if frame is None:
            return None
        return frame.copy()

    async def _process_frames_with_realtime_streaming(self, task_id: uuid.UUID, video_data: Dict[str, Any]) -> bool:
        """
        Process frames with real-time detection and WebSocket streaming.
        
        This is the core Phase 2 processing method that handles frame-by-frame
        detection, annotation, and streaming to connected clients.
        """
        try:
            logger.info(f"🔍 PHASE 2 PROCESSING: Starting real-time frame processing for task {task_id}")
            
            # Get total frame count
            total_frames = min(
                data.get("frame_count", 0) for data in video_data.values() 
                if data.get("frame_count", 0) > 0
            )
            
            if total_frames == 0:
                logger.warning("No frames available for Phase 2 processing")
                return False
            
            frame_index = 0
            frames_streamed = 0
            aborted_due_to_no_clients = False
            
            # Main processing loop
            while frame_index < total_frames:
                if task_id not in self.active_tasks:
                    logger.info(f"🔍 PHASE 2 PROCESSING: Task {task_id} was stopped")
                    break

                await self._wait_for_playback(task_id)

                if task_id not in self.active_tasks:
                    logger.info(f"🔍 PHASE 2 PROCESSING: Task {task_id} stopped during pause wait")
                    break

                if not self._should_continue_stream(task_id, detection_mode=True):
                    aborted_due_to_no_clients = True
                    break

                # Process all cameras for current frame
                camera_frames_processed = 0
                
                for camera_id, data in video_data.items():
                    cap = data.get("video_capture")
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Detect + Track + Annotate + Stream in real-time (stubbed)
                            detection_data = await self.process_frame_with_tracking(
                                frame, camera_id, frame_index
                            )
                            
                            # Send real-time update
                            stream_success = await self.send_detection_update(
                                task_id, camera_id, frame, detection_data, frame_index
                            )
                            
                            if stream_success:
                                frames_streamed += 1
                            
                            camera_frames_processed += 1
                        else:
                            pass # logger.debug(f"End of video reached for camera {camera_id}")
                            break
                    else:
                        break
                
                # Check if all cameras finished
                if camera_frames_processed == 0:
                    logger.info(f"🔍 PHASE 2 PROCESSING: All cameras finished at frame {frame_index}")
                    break
                
                # Update progress every 15 frames
                if frame_index % 15 == 0:
                    progress = 0.50 + (frame_index / total_frames) * 0.45  # 0.50-0.95 range
                    await self._update_task_status(
                        task_id, "PROCESSING", progress,
                        f"Streaming frame {frame_index}/{total_frames} - {frames_streamed} updates sent"
                    )

                await self._record_playback_progress(task_id, frame_index)

                frame_index += 1
                
                # Small delay to prevent overwhelming WebSocket clients
                await asyncio.sleep(0.01)  # 10ms delay
            
            # Cleanup video captures
            for data in video_data.values():
                cap = data.get("video_capture")
                if cap:
                    cap.release()

            if aborted_due_to_no_clients:
                reason = "Detection stopped - no active WebSocket clients"
                self._mark_task_stopped_due_to_idle_clients(task_id, reason)
                logger.info(f"🛑 PHASE 2 PROCESSING: {reason} (task {task_id})")
                return True

            logger.info(f"✅ PHASE 2 PROCESSING: Completed real-time processing - {frames_streamed} updates streamed")
            return True
            
        except Exception as e:
            logger.error(f"❌ PHASE 2 PROCESSING: Error in real-time processing: {e}")
            return False
    
    def _update_detection_stats(self):
        """Update detection statistics."""
        try:
            if self.detection_times:
                self.detection_stats["average_detection_time"] = sum(self.detection_times) / len(self.detection_times)
                
                # Keep only recent times
                if len(self.detection_times) > 100:
                    self.detection_times = self.detection_times[-50:]
                    
        except Exception as e:
            logger.error(f"Error updating detection stats: {e}")
    
    async def _start_trail_cleanup_task(self):
        """Start background task for trail cleanup to prevent memory leaks."""
        async def cleanup_loop():
            """Background cleanup loop for trail management."""
            while True:
                try:
                    await self.trail_service.cleanup_old_trails(max_age_seconds=30)
                    await asyncio.sleep(10)  # Cleanup every 10 seconds
                except Exception as e:
                    logger.warning(f"Trail cleanup error: {e}")
                    await asyncio.sleep(30)  # Longer sleep on error
        
        asyncio.create_task(cleanup_loop())
        logger.info("🧹 TRAIL CLEANUP: Background cleanup task started")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics with Phase 4 spatial intelligence status."""
        stats = dict(self.detection_stats)
        stats.update({
            "active_tasks_count": len(self.active_tasks),
            "total_tasks_count": len(self.tasks),
            "detector_loaded": self.detector is not None and getattr(self.detector, '_model_loaded_flag', False),
            # Phase 4: Spatial intelligence status
            "spatial_intelligence": {
                "homography_service_loaded": self.homography_service is not None,
                "handoff_service_loaded": self.handoff_service is not None,
                "homography_matrices_count": len(getattr(self.homography_service, "json_homography_matrices", {})) if self.homography_service else 0,
                "handoff_configuration_valid": all(self.handoff_service.validate_configuration().values()) if self.handoff_service else False
            },
            # Trail management statistics for 2D mapping
            "trail_management": {
                "service_loaded": self.trail_service is not None,
                "total_cameras": len(self.trail_service.trails) if self.trail_service else 0,
                "total_active_trails": sum(len(camera_trails) for camera_trails in self.trail_service.trails.values()) if self.trail_service else 0
            }
        })
        return stats
    
    def _get_environment_for_camera(self, camera_id: str) -> Optional[str]:
        # Get environment ID for a given camera ID by looking up VIDEO_SETS configuration.
        try:
            for video_set in settings.VIDEO_SETS:
                if video_set.cam_id == camera_id:
                    return video_set.env_id
            
            # logger.debug(f"No environment found for camera {camera_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting environment for camera {camera_id}: {e}")
            return None


# Global detection video service instance
detection_video_service = DetectionVideoService()
