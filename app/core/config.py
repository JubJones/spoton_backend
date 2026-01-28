from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Tuple, Set, Any
from pathlib import Path
import math
import copy

from app.shared.types import CameraID, ExitRuleModel, QuadrantName, CameraHandoffDetailConfig

class VideoSetEnvironmentConfig(BaseModel):
    remote_base_key: str = Field(..., description="Base S3 key (e.g., 'video_s37/c01')")
    env_id: str = Field(..., description="Environment ID (e.g., 'campus')")
    cam_id: str = Field(..., description="Camera ID (e.g., 'c01')")
    num_sub_videos: int = Field(..., gt=0, description="Total number of sub-videos available for this camera.")
    sub_video_filename_pattern: str = Field(default="sub_video_{idx:02d}.mp4", description="Filename pattern. '{idx:02d}' for index.")


def _resolve_local_videos_base_dir() -> str:
    """Prefer an existing local videos root if available."""
    candidate_paths = [
        Path("/app/videos"),
        Path.cwd() / "videos",
        Path(__file__).resolve().parents[2] / "videos",
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path.resolve())
    return str(candidate_paths[0])


BASE_ENVIRONMENT_CONFIGURATION: Dict[str, Dict[str, Any]] = {
    "campus": {
        "name": "Campus Environment",
        "environment_type": "campus",
        "description": "Default campus environment with multiple zones and cameras",
        "timezone": "UTC",
        "operating_hours": {
            "monday": {"start": "06:00", "end": "22:00"},
            "tuesday": {"start": "06:00", "end": "22:00"},
            "wednesday": {"start": "06:00", "end": "22:00"},
            "thursday": {"start": "06:00", "end": "22:00"},
            "friday": {"start": "06:00", "end": "22:00"},
            "saturday": {"start": "08:00", "end": "18:00"},
            "sunday": {"start": "10:00", "end": "16:00"}
        },
        "capacity_limits": {"total": 100, "entrance": 8, "main_area": 50},
        "data_retention_days": 180,
        "analytics_enabled": True,
        "possible_camera_overlaps": [("c09", "c12"), ("c12", "c13"), ("c13", "c16")],
        "layout": {
            "layout_id": "campus_layout",
            "name": "Campus Layout",
            "bounds": {"min": (0, 0), "max": (70, 30)},
            "scale_meters_per_pixel": 0.1,
            "reference_points": [
                {"point": (0, 0), "description": "Origin point"},
                {"point": (35, 15), "description": "Center point"}
            ],
            "walls": [],
            "doors": [],
            "landmarks": []
        },
        "zones": [
            {
                "zone_id": "entrance",
                "name": "Campus Entrance",
                "zone_type": "entrance",
                "boundary_points": [(0, 0), (10, 0), (10, 5), (0, 5)],
                "capacity_limit": 8,
                "monitoring_enabled": True,
                "occupancy_threshold": 6,
                "description": "Default campus entrance zone"
            },
            {
                "zone_id": "main_area",
                "name": "Campus Main Area",
                "zone_type": "main_area",
                "boundary_points": [(10, 0), (60, 0), (60, 30), (10, 30)],
                "capacity_limit": 50,
                "monitoring_enabled": True,
                "occupancy_threshold": 40,
                "description": "Default campus main area zone"
            },
            {
                "zone_id": "corridor",
                "name": "Campus Corridor",
                "zone_type": "corridor",
                "boundary_points": [(60, 5), (70, 5), (70, 15), (60, 15)],
                "capacity_limit": 15,
                "monitoring_enabled": True,
                "occupancy_threshold": 12,
                "description": "Default campus corridor zone"
            },
            {
                "zone_id": "exit",
                "name": "Campus Exit",
                "zone_type": "exit",
                "boundary_points": [(60, 0), (70, 0), (70, 5), (60, 5)],
                "capacity_limit": 8,
                "monitoring_enabled": True,
                "occupancy_threshold": 6,
                "description": "Default campus exit zone"
            }
        ],
        "cameras": {
            "c01": {
                "display_name": "Campus Gate Camera",
                "remote_base_key": "video_s37/c01",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (10.0, 5.0),
                "resolution": (1920, 1080),
                "field_of_view": 60.0,
                "orientation": 0.0,
                "handoff_exit_rules": [
                    {
                        "source_exit_quadrant": "upper_right",
                        "target_cam_id": "c03",
                        "target_entry_area": "bottom_left",
                        "notes": None
                    }
                ],

            },
            "c02": {
                "display_name": "Campus Plaza Camera",
                "remote_base_key": "video_s37/c02",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (30.0, 6.0),
                "resolution": (1920, 1080),
                "field_of_view": 60.0,
                "orientation": 0.0,
                "handoff_exit_rules": [
                    {
                        "source_exit_quadrant": "upper_right",
                        "target_cam_id": "c05",
                        "target_entry_area": "upper left",
                        "notes": None
                    }
                ],

            },
            "c03": {
                "display_name": "Campus Walkway Camera",
                "remote_base_key": "video_s37/c03",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (50.0, 7.0),
                "resolution": (1920, 1080),
                "field_of_view": 60.0,
                "orientation": 0.0,
                "handoff_exit_rules": [
                    {
                        "source_exit_quadrant": "bottom_left",
                        "target_cam_id": "c01",
                        "target_entry_area": "upper_right",
                        "notes": None
                    },
                    {
                        "source_exit_quadrant": "upper_right",
                        "target_cam_id": "c05",
                        "target_entry_area": "upper left",
                        "notes": None
                    }
                ],

            },
            "c05": {
                "display_name": "Campus Commons Camera",
                "remote_base_key": "video_s37/c05",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (68.0, 4.0),
                "resolution": (1920, 1080),
                "field_of_view": 60.0,
                "orientation": 0.0,
                "handoff_exit_rules": [
                    {
                        "source_exit_quadrant": "upper_left",
                        "target_cam_id": "c02",
                        "target_entry_area": "upper_right",
                        "notes": None
                    },
                    {
                        "source_exit_quadrant": "upper_left",
                        "target_cam_id": "c03",
                        "target_entry_area": "upper_right",
                        "notes": None
                    }
                ],

            }
        }
    },
    "factory": {
        "name": "Factory Environment",
        "environment_type": "factory",
        "description": "Default factory environment with production lines and quality control",
        "timezone": "UTC",
        "operating_hours": {
            "monday": {"start": "06:00", "end": "18:00"},
            "tuesday": {"start": "06:00", "end": "18:00"},
            "wednesday": {"start": "06:00", "end": "18:00"},
            "thursday": {"start": "06:00", "end": "18:00"},
            "friday": {"start": "06:00", "end": "18:00"}
        },
        "capacity_limits": {"total": 40, "production_line_1": 12, "production_line_2": 12},
        "data_retention_days": 60,
        "analytics_enabled": True,
        "possible_camera_overlaps": [("c01", "c03"), ("c02", "c03"), ("c03", "c05")],
        "layout": {
            "layout_id": "factory_layout",
            "name": "Factory Floor Layout",
            "bounds": {"min": (0, 0), "max": (75, 10)},
            "scale_meters_per_pixel": 0.1,
            "reference_points": [
                {"point": (0, 0), "description": "Factory entrance"},
                {"point": (37.5, 5), "description": "Factory center"}
            ],
            "walls": [],
            "doors": [],
            "landmarks": []
        },
        "zones": [
            {
                "zone_id": "production_line_1",
                "name": "Production Line 1",
                "zone_type": "main_area",
                "boundary_points": [(5, 0), (25, 0), (25, 10), (5, 10)],
                "capacity_limit": 12,
                "monitoring_enabled": True,
                "occupancy_threshold": 10,
                "description": "Default production line 1 zone"
            },
            {
                "zone_id": "production_line_2",
                "name": "Production Line 2",
                "zone_type": "main_area",
                "boundary_points": [(25, 0), (45, 0), (45, 10), (25, 10)],
                "capacity_limit": 12,
                "monitoring_enabled": True,
                "occupancy_threshold": 10,
                "description": "Default production line 2 zone"
            },
            {
                "zone_id": "quality_control",
                "name": "Quality Control",
                "zone_type": "restricted",
                "boundary_points": [(45, 0), (65, 0), (65, 10), (45, 10)],
                "capacity_limit": 5,
                "monitoring_enabled": True,
                "occupancy_threshold": 4,
                "description": "Default quality control zone"
            },
            {
                "zone_id": "factory_exit",
                "name": "Factory Exit",
                "zone_type": "exit",
                "boundary_points": [(65, 0), (75, 0), (75, 5), (65, 5)],
                "capacity_limit": 6,
                "monitoring_enabled": True,
                "occupancy_threshold": 5,
                "description": "Default factory exit zone"
            }
        ],
        "cameras": {
            "c09": {
                "display_name": "Factory Entry Camera",
                "remote_base_key": "video_s14/c09",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (5.0, 2.0),
                "resolution": (1920, 1080),
                "field_of_view": 70.0,
                "orientation": 0.0,
                "handoff_exit_rules": [
                    {
                        "source_exit_quadrant": "lower_left",
                        "target_cam_id": "c13",
                        "target_entry_area": "upper right",
                        "notes": "wait; overlap c13/c16 possible"
                    },
                    {
                        "source_exit_quadrant": "lower_right",
                        "target_cam_id": "c13",
                        "target_entry_area": "upper right",
                        "notes": "wait; overlap c13/c16 possible"
                    }
                ],

            },
            "c12": {
                "display_name": "Factory Main Area Camera",
                "remote_base_key": "video_s14/c12",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (25.0, 15.0),
                "resolution": (1920, 1080),
                "field_of_view": 70.0,
                "orientation": 0.0,
                "handoff_exit_rules": [
                    {
                        "source_exit_quadrant": "upper_left",
                        "target_cam_id": "c13",
                        "target_entry_area": "upper left",
                        "notes": "overlap c13 possible"
                    },
                    {
                        "source_exit_quadrant": "lower_left",
                        "target_cam_id": "c13",
                        "target_entry_area": "upper left",
                        "notes": "overlap c13 possible"
                    }
                ],

            },
            "c13": {
                "display_name": "Factory Corridor Camera",
                "remote_base_key": "video_s14/c13",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (45.0, 10.0),
                "resolution": (1920, 1080),
                "field_of_view": 70.0,
                "orientation": 0.0,
                "handoff_exit_rules": [
                    {
                        "source_exit_quadrant": "upper_right",
                        "target_cam_id": "c09",
                        "target_entry_area": "down",
                        "notes": "wait; overlap c09 possible"
                    },
                    {
                        "source_exit_quadrant": "lower_right",
                        "target_cam_id": "c09",
                        "target_entry_area": "down",
                        "notes": "wait; overlap c09 possible"
                    },
                    {
                        "source_exit_quadrant": "upper_left",
                        "target_cam_id": "c12",
                        "target_entry_area": "upper left",
                        "notes": "overlap c12 possible"
                    },
                    {
                        "source_exit_quadrant": "lower_left",
                        "target_cam_id": "c12",
                        "target_entry_area": "upper left",
                        "notes": "overlap c12 possible"
                    }
                ],

            },
            "c16": {
                "display_name": "Factory Exit Camera",
                "remote_base_key": "video_s14/c16",
                "num_sub_videos": 4,
                "sub_video_filename_pattern": "sub_video_{idx:02d}.mp4",
                "position": (65.0, 2.0),
                "resolution": (1920, 1080),
                "field_of_view": 70.0,
                "orientation": 0.0,
                "handoff_exit_rules": [],

            }
        }
    }
}


def _build_video_sets_from_templates(templates: Dict[str, Dict[str, Any]]) -> List[VideoSetEnvironmentConfig]:
    video_sets: List[VideoSetEnvironmentConfig] = []
    for env_id, env_cfg in templates.items():
        for cam_id, cam_cfg in env_cfg.get("cameras", {}).items():
            video_sets.append(
                VideoSetEnvironmentConfig(
                    remote_base_key=cam_cfg["remote_base_key"],
                    env_id=env_id,
                    cam_id=cam_id,
                    num_sub_videos=cam_cfg.get("num_sub_videos", 1),
                    sub_video_filename_pattern=cam_cfg.get("sub_video_filename_pattern", "sub_video_{idx:02d}.mp4")
                )
            )
    return video_sets


def _build_camera_handoff_details_from_templates(
    templates: Dict[str, Dict[str, Any]]
) -> Dict[Tuple[str, str], CameraHandoffDetailConfig]:
    details: Dict[Tuple[str, str], CameraHandoffDetailConfig] = {}
    for env_id, env_cfg in templates.items():
        for cam_id, cam_cfg in env_cfg.get("cameras", {}).items():
            exit_rules = [
                ExitRuleModel(
                    source_exit_quadrant=QuadrantName(rule["source_exit_quadrant"]),
                    target_cam_id=CameraID(rule["target_cam_id"]),
                    target_entry_area=rule["target_entry_area"],
                    notes=rule.get("notes")
                )
                for rule in cam_cfg.get("handoff_exit_rules", [])
            ]
            details[(env_id, cam_id)] = CameraHandoffDetailConfig(
                exit_rules=exit_rules,
                homography_matrix_path=cam_cfg.get("homography_matrix_path")
            )
    return details


def _build_possible_camera_overlaps_from_templates(
    templates: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, str]]:
    overlaps: List[Tuple[str, str]] = []
    for env_cfg in templates.values():
        for pair in env_cfg.get("possible_camera_overlaps", []):
            overlaps.append(tuple(pair))
    return overlaps


def _build_camera_handoff_zones_from_templates(
    templates: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Dict[str, float]]]:
    zones: Dict[str, List[Dict[str, float]]] = {}
    for env_cfg in templates.values():
        for cam_id, cam_cfg in env_cfg.get("cameras", {}).items():
            if cam_cfg.get("handoff_zones"):
                zones[cam_id] = list(cam_cfg["handoff_zones"])
    return zones


class Settings(BaseSettings):
    APP_NAME: str = "SpotOn Backend"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True # Controls application debug mode (e.g. detailed errors), separate from logging level.
    # LOG_LEVEL is read directly by logging config or uvicorn, not defined here as a setting.
    
    # Production Endpoint Control
    ENABLE_ANALYTICS_ENDPOINTS: bool = Field(default=True, description="Enable analytics API endpoints")
    ENABLE_EXPORT_ENDPOINTS: bool = Field(default=True, description="Enable data export API endpoints")
    ENABLE_AUTH_ENDPOINTS: bool = Field(default=True, description="Enable authentication API endpoints")
    ENABLE_ADMIN_ENDPOINTS: bool = Field(default=False, description="Enable admin-only API endpoints")
    PRODUCTION_MODE: bool = Field(default=False, description="Production deployment mode with enhanced security")
    
    # Security Configuration
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"], description="Allowed CORS origins")
    ALLOWED_HOSTS: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"], description="Allowed host headers")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Rate limit requests per minute per IP")
    ENABLE_SECURITY_HEADERS: bool = Field(default=True, description="Enable security headers middleware")
    ENABLE_REQUEST_LOGGING: bool = Field(default=True, description="Enable security request logging")
    MAX_REQUEST_SIZE_MB: int = Field(default=10, description="Maximum request size in MB")

    S3_ENDPOINT_URL: Optional[str] = "https://s3.dagshub.com"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: str = "spoton_ml"
    DAGSHUB_REPO_OWNER: str = "Jwizzed"
    DAGSHUB_REPO_NAME: str = "spoton_ml"
    LOCAL_VIDEO_DOWNLOAD_DIR: str = "./downloaded_videos"
    # Base directory to read pre-downloaded/local videos from
    LOCAL_VIDEOS_BASE_DIR: str = Field(default_factory=_resolve_local_videos_base_dir)
    LOCAL_FRAME_EXTRACTION_DIR: str = "./extracted_frames"
    # Phase 6: I/O and caching controls
    MAX_DOWNLOAD_CONCURRENCY: int = Field(default=3, description="Max concurrent S3 downloads")
    STORE_EXTRACTED_FRAMES: bool = Field(default=False, description="Store extracted/annotated frames to disk while streaming")
    FRAME_CACHE_DIR: str = Field(default="./extracted_frames", description="Directory for on-disk frame cache")
    FRAME_CACHE_SAMPLE_RATE: int = Field(default=0, description="Save every Nth frame (0 disables saving)")
    # S3 transfer settings
    S3_CONNECT_TIMEOUT: int = Field(default=10, description="S3 connect timeout (seconds)")
    S3_READ_TIMEOUT: int = Field(default=60, description="S3 read timeout (seconds)")
    S3_MAX_ATTEMPTS: int = Field(default=5, description="S3 max retries")
    S3_MAX_TRANSFER_CONCURRENCY: int = Field(default=4, description="S3 multipart transfer concurrency")
    S3_MULTIPART_THRESHOLD_MB: int = Field(default=16, description="S3 multipart threshold (MB)")

    ENVIRONMENT_TEMPLATES: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: copy.deepcopy(BASE_ENVIRONMENT_CONFIGURATION)
    )
    VIDEO_SETS: List[VideoSetEnvironmentConfig] = Field(default_factory=list)
    CAMERA_HANDOFF_DETAILS: Dict[Tuple[str, str], CameraHandoffDetailConfig] = Field(default_factory=dict)
    CAMERA_HANDOFF_ZONES: Dict[str, List[Dict[str, float]]] = Field(default_factory=dict)
    POSSIBLE_CAMERA_OVERLAPS: List[Tuple[str, str]] = Field(default_factory=list)
    MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT: float = Field(default=0.40)
    MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT: float = Field(default=0.40)
    HOMOGRAPHY_FILE_PATH: str = Field(default="/app/homography_data/homography_20251024-103317.json", description="JSON file containing camera-to-world homography matrices")
    WORLD_BOUNDS_X_MIN: float = Field(default=-2000.0, description="Minimum world X coordinate (meters)")
    WORLD_BOUNDS_X_MAX: float = Field(default=2000.0, description="Maximum world X coordinate (meters)")
    WORLD_BOUNDS_Y_MIN: float = Field(default=-2000.0, description="Minimum world Y coordinate (meters)")
    WORLD_BOUNDS_Y_MAX: float = Field(default=2000.0, description="Maximum world Y coordinate (meters)")
    ENABLE_POINT_VALIDATION: bool = Field(default=True, description="Enable image-space point validation checks")
    ENABLE_BOUNDS_VALIDATION: bool = Field(default=True, description="Enable world bounds validation after homography projection")
    ROI_BASE_RADIUS: float = Field(default=1.5, description="Base ROI radius in meters")
    ROI_MAX_WALKING_SPEED: float = Field(default=1.5, description="Maximum walking speed in m/s for ROI expansion")
    ROI_MIN_RADIUS: float = Field(default=0.5, description="Minimum ROI radius in meters")
    ROI_MAX_RADIUS: float = Field(default=10.0, description="Maximum ROI radius in meters")
    ROI_SHAPE: str = Field(default="circular", description="ROI shape: 'circular' or 'rectangular'")
    EXACT_MATCH_CONFIDENCE: float = Field(default=0.95, description="Base confidence for exact geometric matches")
    CLOSEST_MATCH_CONFIDENCE: float = Field(default=0.70, description="Base confidence when choosing closest candidate")
    DISTANCE_PENALTY_FACTOR: float = Field(default=0.1, description="Confidence reduction per meter of separation")
    MIN_MATCH_CONFIDENCE: float = Field(default=0.5, description="Minimum confidence required to accept geometric match")
    MIN_TRACK_CONFIDENCE_FOR_MATCHING: float = Field(default=0.3, description="Minimum track confidence to include in geometric matching and debug visualization")
    HIGH_CONFIDENCE_THRESHOLD: float = Field(default=0.8, description="Threshold for counting matches as high-confidence")
    
    # Phase 1: Space-Based Matching (Spatial Intelligence)
    SPATIAL_MATCH_ENABLED: bool = Field(default=True, description="Enable space-based cross-camera matching")
    SPATIAL_MATCH_THRESHOLD: float = Field(default=300.0, description="Max world distance (units/pixels) to consider a potential match. (E.g. 300 = 3 meters at 100px/m)")
    SPATIAL_EDGE_MARGIN: float = Field(default=0.05, description="Margin (0.0-1.0) to ignore detections near frame edges")
    SPATIAL_VELOCITY_GATE: bool = Field(default=True, description="Prevent matching tracks moving in opposite directions")
    SPATIAL_MATCH_MIN_OVERLAP_FRAMES: int = Field(default=3, description="Require N consecutive frames of proximity before merging IDs")

    ENABLE_DEBUG_REPROJECTION: bool = Field(default=False, description="Enable reprojection debugging overlays")
    DEBUG_OVERLAY_RADIUS_PX: int = Field(default=6, description="Radius (px) for predicted marker visualization")
    DEBUG_REPROJECTION_OUTPUT_DIR: str = Field(default="app/debug_outputs", description="Directory for saved reprojection frames")
    DEBUG_FRAME_SAMPLING_RATE: int = Field(default=1, description="Persist every Nth frame for reprojection debugging")
    DEBUG_MAX_FRAMES_PER_CAMERA: int = Field(default=500, description="Maximum saved debug frames per camera per run")

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: Optional[str] = None
    POSTGRES_USER: str = "spoton_user"
    POSTGRES_PASSWORD: str = "spoton_password"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "spotondb"
    DATABASE_URL: Optional[str] = None
    # Database toggles and pool settings (Phase 7)
    DB_ENABLED: bool = Field(default=True, description="Enable database integration (TimescaleDB)")
    DB_POOL_SIZE: int = Field(default=20, description="SQLAlchemy QueuePool size")
    DB_MAX_OVERFLOW: int = Field(default=30, description="SQLAlchemy QueuePool max_overflow")
    DB_POOL_RECYCLE: int = Field(default=3600, description="Seconds to recycle DB connections")
    DB_POOL_PRE_PING: bool = Field(default=True, description="Enable pool_pre_ping to detect stale connections")

    DETECTOR_TYPE: str = "yolo"
    PERSON_CLASS_ID: int = 1
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    DETECTION_USE_AMP: bool = False
    
    
    # YOLO Configuration (YOLO11-L)
    # TensorRT Optimization: Set USE_TENSORRT=true and provide .engine file path for 3-5x speedup
    USE_TENSORRT: bool = Field(default=True, description="Use TensorRT engine for YOLO inference (requires .engine file)")
    YOLO_MODEL_PATH: str = "/app/weights/yolo26m.pt" # Default PT model 
    YOLO_MODEL_PATH_TENSORRT: str = Field(default="/app/weights/yolo26m.engine", description="TensorRT engine path (used when USE_TENSORRT=true)")
    # Optional per-environment model overrides. If provided and file exists,
    # detection will use these weights for the corresponding environment.
    YOLO_MODEL_PATH_CAMPUS: Optional[str] = None
    YOLO_MODEL_PATH_FACTORY: Optional[str] = None
    # Optional external weights directory (e.g., sibling repo with weights)
    EXTERNAL_WEIGHTS_BASE_DIR: Optional[str] = None
    YOLO_CONFIDENCE_THRESHOLD: float = 0.3  # Lowered from 0.5 for NMS-free YOLO26
    YOLO_NMS_THRESHOLD: float = 0.45
    YOLO_INPUT_SIZE: int = 480  # Reduced from 640 to 480 for higher FPS
    DETECTION_ANNOTATION_ENABLED: bool = True
    DETECTION_SAVE_ORIGINAL_FRAMES: bool = True
    TRACKER_TYPE: str = "bytetrack"
    WEIGHTS_DIR: str = "./weights" 
    TRACKER_HALF_PRECISION: bool = False
    TRACKER_PER_CLASS: bool = False
    # (Re-ID model settings removed)

    # --- Core Integration Architecture: Feature Flags ---
    TRACKING_ENABLED: bool = True
    TRAJECTORY_SMOOTHING_ENABLED: bool = True
    # Migration strategy feature flags (phased rollout)
    ENABLE_INTRA_CAMERA_TRACKING: bool = True
    ENABLE_TRAJECTORY_TRACKING: bool = True
    ENABLE_ENHANCED_VISUALIZATION: bool = True
    # Tracking parameters
    TRACK_BUFFER_SIZE: int = 20  # Reduced from 30 for faster tracking
    TRACK_CONFIDENCE_THRESHOLD: float = 0.5
    
    # (Re-ID batch/gallery settings removed)
    REID_ENABLED: bool = Field(default=True, description="Enable Re-ID feature extraction (disable for performance)")
    HANDOFF_ZONE_THRESHOLD: float = 0.2
    TRACKER_CONFIG_PATH: Optional[str] = None
    
    # Performance Optimization Settings
    FRAME_SKIP: int = Field(default=2, description="Process every Nth frame (1=all, 2=every other, 3=every 3rd)")
    BATCH_ACCUMULATION_SIZE: int = Field(default=4, description="Number of frames to accumulate per camera before batch inference (e.g., 4 cams × 4 frames = 16 batch)")
    SPATIAL_MATCH_SKIP_STATIC: bool = Field(default=False, description="Skip spatial matching for tracks that haven't moved")

    # (Re-ID similarity settings removed)

    TARGET_FPS: int = 23 
    FRAME_JPEG_QUALITY=85  # Good balance

    # WebSocket client connection guards
    STREAMING_CLIENT_INITIAL_GRACE_SECONDS: float = Field(
        default=15.0,
        description="Seconds to wait for an initial WebSocket client before stopping the stream"
    )
    STREAMING_CLIENT_IDLE_TIMEOUT_SECONDS: float = Field(
        default=12.0,
        description="Seconds to continue processing after the last WebSocket client disconnects"
    )
    DETECTION_CLIENT_INITIAL_GRACE_SECONDS: Optional[float] = Field(
        default=None,
        description="Override initial grace period (in seconds) for detection tasks; falls back to STREAMING_CLIENT_INITIAL_GRACE_SECONDS when unset"
    )
    DETECTION_CLIENT_IDLE_TIMEOUT_SECONDS: Optional[float] = Field(
        default=None,
        description="Override idle timeout (in seconds) for detection tasks; falls back to STREAMING_CLIENT_IDLE_TIMEOUT_SECONDS when unset"
    )

    ENABLE_PLAYBACK_CONTROL: bool = Field(
        default=True,
        description="Enable playback control API endpoints and coordination services",
    )
    PLAYBACK_CONTROL_TIMEOUT_SECONDS: float = Field(
        default=1.0,
        description="Maximum seconds to wait for playback pause/resume acknowledgement",
    )

    # Lazy initialization toggles to reduce cold start
    PRELOAD_TRACKER_FACTORY: bool = Field(default=True, description="Preload prototype tracker at startup")
    PRELOAD_HOMOGRAPHY: bool = Field(default=True, description="Preload homography matrices at startup")
    PRELOAD_YOLO_DETECTOR: bool = Field(default=True, description="Preload YOLO detector at startup")
    PRELOAD_REID_MODEL: bool = Field(default=True, description="Preload Re-ID model at startup")
    PRELOAD_ENVIRONMENTS: List[str] = Field(default_factory=lambda: ["campus", "factory"], description="Environments to preload models for")

    # Trail management settings
    START_TRAIL_CLEANUP: bool = Field(default=True, description="Start global trail cleanup background task")
    TRAIL_CLEANUP_INTERVAL_SECONDS: int = Field(default=10, description="Interval between trail cleanup runs")
    TRAIL_MAX_AGE_SECONDS: int = Field(default=30, description="Max age for trail points before cleanup")
    TRAIL_LENGTH: int = Field(default=3, description="Number of points to retain per trail")

    # WebSocket payload limits
    WS_TRACKING_TRAJECTORY_POINTS_LIMIT: int = Field(default=50, description="Max trajectory points to include per person in WS payloads")

    # Export settings
    EXPORT_BASE_DIR: str = "./exports"
    EXPORT_EXPIRY_HOURS: int = 24

    # (Re-ID derived thresholds removed)

    model_config = { # Pydantic V2 uses model_config instead of Config class
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._apply_environment_templates()
        
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            
        if not hasattr(self, 'REDIS_URL') or not self.REDIS_URL:
             # Construct Redis URL if not present (although not explicitly defined as a field yet, useful for clients)
            auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            self.REDIS_URL = f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    def _apply_environment_templates(self) -> None:
        templates = self.ENVIRONMENT_TEMPLATES
        self.VIDEO_SETS = _build_video_sets_from_templates(templates)
        self.CAMERA_HANDOFF_DETAILS = _build_camera_handoff_details_from_templates(templates)
        self.POSSIBLE_CAMERA_OVERLAPS = _build_possible_camera_overlaps_from_templates(templates)
        self.CAMERA_HANDOFF_ZONES = _build_camera_handoff_zones_from_templates(templates)

    # (Re-ID weights path removed)
    


    @property
    def normalized_possible_camera_overlaps(self) -> Set[Tuple[CameraID, CameraID]]:
        return {tuple(sorted((CameraID(c1), CameraID(c2)))) for c1, c2 in self.POSSIBLE_CAMERA_OVERLAPS}

settings = Settings()
