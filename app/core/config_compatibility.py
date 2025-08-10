"""
Backward compatibility wrappers for Phase 6 configuration consolidation.

Provides seamless compatibility layer between old configuration system
and new consolidated configuration, ensuring zero-downtime migration.
"""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import os
import warnings
from functools import wraps

from app.core.config_consolidated import ConsolidatedSettings, get_settings

logger = logging.getLogger(__name__)


class CompatibilityWarning(UserWarning):
    """Warning for deprecated configuration access patterns."""
    pass


def deprecated_config_access(component: str, old_attr: str, new_path: str):
    """Decorator to mark deprecated configuration access methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"Accessing '{old_attr}' is deprecated. Use '{new_path}' instead. "
                f"Component: {component}",
                CompatibilityWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class LegacySettingsWrapper:
    """
    Backward compatibility wrapper for legacy Settings class.
    
    Provides property-based access to consolidated configuration
    while maintaining the same interface as the original Settings class.
    """
    
    def __init__(self, consolidated_settings: ConsolidatedSettings):
        """Initialize with consolidated settings instance."""
        self._consolidated = consolidated_settings
        logger.debug("LegacySettingsWrapper initialized for backward compatibility")
    
    # Core application settings - Direct mapping
    @property
    def APP_NAME(self) -> str:
        return self._consolidated.core.app_name
    
    @property
    def API_V1_PREFIX(self) -> str:
        return self._consolidated.core.api_v1_prefix
    
    @property
    @deprecated_config_access("core", "DEBUG", "consolidated_settings.core.debug")
    def DEBUG(self) -> bool:
        return self._consolidated.core.debug
    
    # Storage settings - Map to storage configuration
    @property
    @deprecated_config_access("storage", "S3_ENDPOINT_URL", "consolidated_settings.storage_config.s3.endpoint_url")
    def S3_ENDPOINT_URL(self) -> Optional[str]:
        try:
            return self._consolidated.storage_config.s3.endpoint_url
        except Exception:
            return None
    
    @property
    @deprecated_config_access("storage", "AWS_ACCESS_KEY_ID", "consolidated_settings.storage_config.s3.access_key_id")
    def AWS_ACCESS_KEY_ID(self) -> Optional[str]:
        try:
            return self._consolidated.storage_config.s3.access_key_id
        except Exception:
            return None
    
    @property
    @deprecated_config_access("storage", "AWS_SECRET_ACCESS_KEY", "consolidated_settings.storage_config.s3.secret_access_key")
    def AWS_SECRET_ACCESS_KEY(self) -> Optional[str]:
        try:
            return self._consolidated.storage_config.s3.secret_access_key
        except Exception:
            return None
    
    @property
    @deprecated_config_access("storage", "S3_BUCKET_NAME", "consolidated_settings.storage_config.s3.bucket_name")
    def S3_BUCKET_NAME(self) -> str:
        try:
            return self._consolidated.storage_config.s3.bucket_name
        except Exception:
            return "spoton_ml"
    
    @property
    @deprecated_config_access("storage", "DAGSHUB_REPO_OWNER", "consolidated_settings.storage_config.s3.dagshub_repo_owner")
    def DAGSHUB_REPO_OWNER(self) -> str:
        try:
            return self._consolidated.storage_config.s3.dagshub_repo_owner
        except Exception:
            return "Jwizzed"
    
    @property
    @deprecated_config_access("storage", "DAGSHUB_REPO_NAME", "consolidated_settings.storage_config.s3.dagshub_repo_name")
    def DAGSHUB_REPO_NAME(self) -> str:
        try:
            return self._consolidated.storage_config.s3.dagshub_repo_name
        except Exception:
            return "spoton_ml"
    
    @property
    @deprecated_config_access("storage", "LOCAL_VIDEO_DOWNLOAD_DIR", "consolidated_settings.storage_config.local.video_download_dir")
    def LOCAL_VIDEO_DOWNLOAD_DIR(self) -> str:
        try:
            return self._consolidated.storage_config.local.video_download_dir
        except Exception:
            return "./downloaded_videos"
    
    @property
    @deprecated_config_access("storage", "LOCAL_FRAME_EXTRACTION_DIR", "consolidated_settings.storage_config.local.frame_extraction_dir")
    def LOCAL_FRAME_EXTRACTION_DIR(self) -> str:
        try:
            return self._consolidated.storage_config.local.frame_extraction_dir
        except Exception:
            return "./extracted_frames"
    
    # Video set configuration - Map to camera configuration
    @property
    @deprecated_config_access("cameras", "VIDEO_SETS", "consolidated_settings.camera_config.video_sets")
    def VIDEO_SETS(self) -> List[Any]:
        try:
            return self._consolidated.camera_config.video_sets
        except Exception:
            return []
    
    @property
    @deprecated_config_access("cameras", "CAMERA_HANDOFF_DETAILS", "consolidated_settings.camera_config.camera_handoff_details")
    def CAMERA_HANDOFF_DETAILS(self) -> Dict[Tuple[str, str], Any]:
        try:
            return self._consolidated.camera_config.camera_handoff_details
        except Exception:
            return {}
    
    @property
    @deprecated_config_access("cameras", "MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT", "consolidated_settings.camera_config.min_bbox_overlap_ratio")
    def MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT(self) -> float:
        try:
            return self._consolidated.camera_config.min_bbox_overlap_ratio
        except Exception:
            return 0.40
    
    @property
    @deprecated_config_access("cameras", "HOMOGRAPHY_DATA_DIR", "consolidated_settings.camera_config.homography_data_dir")
    def HOMOGRAPHY_DATA_DIR(self) -> str:
        try:
            return self._consolidated.camera_config.homography_data_dir
        except Exception:
            return "./homography_points"
    
    @property
    @deprecated_config_access("cameras", "POSSIBLE_CAMERA_OVERLAPS", "consolidated_settings.camera_config.possible_camera_overlaps")
    def POSSIBLE_CAMERA_OVERLAPS(self) -> List[Tuple[str, str]]:
        try:
            return self._consolidated.camera_config.possible_camera_overlaps
        except Exception:
            return []
    
    # Database settings - Map to database configuration
    @property
    @deprecated_config_access("database", "REDIS_HOST", "consolidated_settings.database_config.redis.host")
    def REDIS_HOST(self) -> str:
        try:
            return self._consolidated.database_config.redis.host
        except Exception:
            return "localhost"
    
    @property
    @deprecated_config_access("database", "REDIS_PORT", "consolidated_settings.database_config.redis.port")
    def REDIS_PORT(self) -> int:
        try:
            return self._consolidated.database_config.redis.port
        except Exception:
            return 6379
    
    @property
    @deprecated_config_access("database", "REDIS_DB", "consolidated_settings.database_config.redis.database")
    def REDIS_DB(self) -> int:
        try:
            return self._consolidated.database_config.redis.database
        except Exception:
            return 0
    
    @property
    @deprecated_config_access("database", "REDIS_PASSWORD", "consolidated_settings.database_config.redis.password")
    def REDIS_PASSWORD(self) -> Optional[str]:
        try:
            return self._consolidated.database_config.redis.password
        except Exception:
            return None
    
    @property
    @deprecated_config_access("database", "POSTGRES_USER", "consolidated_settings.database_config.postgresql.user")
    def POSTGRES_USER(self) -> str:
        try:
            return self._consolidated.database_config.postgresql.user
        except Exception:
            return "spoton_user"
    
    @property
    @deprecated_config_access("database", "POSTGRES_PASSWORD", "consolidated_settings.database_config.postgresql.password")
    def POSTGRES_PASSWORD(self) -> str:
        try:
            return self._consolidated.database_config.postgresql.password
        except Exception:
            return "spoton_password"
    
    @property
    @deprecated_config_access("database", "POSTGRES_SERVER", "consolidated_settings.database_config.postgresql.host")
    def POSTGRES_SERVER(self) -> str:
        try:
            return self._consolidated.database_config.postgresql.host
        except Exception:
            return "localhost"
    
    @property
    @deprecated_config_access("database", "POSTGRES_PORT", "consolidated_settings.database_config.postgresql.port")
    def POSTGRES_PORT(self) -> int:
        try:
            return self._consolidated.database_config.postgresql.port
        except Exception:
            return 5432
    
    @property
    @deprecated_config_access("database", "POSTGRES_DB", "consolidated_settings.database_config.postgresql.database")
    def POSTGRES_DB(self) -> str:
        try:
            return self._consolidated.database_config.postgresql.database
        except Exception:
            return "spotondb"
    
    @property
    @deprecated_config_access("database", "DATABASE_URL", "consolidated_settings.database_config.postgresql.database_url")
    def DATABASE_URL(self) -> Optional[str]:
        try:
            return self._consolidated.database_config.postgresql.database_url
        except Exception:
            return None
    
    # AI model settings - Map to AI configuration
    @property
    @deprecated_config_access("ai", "DETECTOR_TYPE", "consolidated_settings.ai_config.detector.detector_type")
    def DETECTOR_TYPE(self) -> str:
        try:
            return self._consolidated.ai_config.detector.detector_type.value
        except Exception:
            return "fasterrcnn"
    
    @property
    @deprecated_config_access("ai", "PERSON_CLASS_ID", "consolidated_settings.ai_config.detector.person_class_id")
    def PERSON_CLASS_ID(self) -> int:
        try:
            return self._consolidated.ai_config.detector.person_class_id
        except Exception:
            return 1
    
    @property
    @deprecated_config_access("ai", "DETECTION_CONFIDENCE_THRESHOLD", "consolidated_settings.ai_config.detector.confidence_threshold")
    def DETECTION_CONFIDENCE_THRESHOLD(self) -> float:
        try:
            return self._consolidated.ai_config.detector.confidence_threshold
        except Exception:
            return 0.5
    
    @property
    @deprecated_config_access("ai", "DETECTION_USE_AMP", "consolidated_settings.ai_config.detector.use_amp")
    def DETECTION_USE_AMP(self) -> bool:
        try:
            return self._consolidated.ai_config.detector.use_amp
        except Exception:
            return False
    
    @property
    @deprecated_config_access("ai", "TRACKER_TYPE", "consolidated_settings.ai_config.tracker.tracker_type")
    def TRACKER_TYPE(self) -> str:
        try:
            return self._consolidated.ai_config.tracker.tracker_type.value
        except Exception:
            return "botsort"
    
    @property
    @deprecated_config_access("ai", "WEIGHTS_DIR", "consolidated_settings.ai_config.weights_dir")
    def WEIGHTS_DIR(self) -> str:
        try:
            return self._consolidated.ai_config.weights_dir
        except Exception:
            return "./weights"
    
    @property
    @deprecated_config_access("ai", "REID_WEIGHTS_PATH", "consolidated_settings.ai_config.reid.weights_filename")
    def REID_WEIGHTS_PATH(self) -> str:
        try:
            return self._consolidated.ai_config.reid.weights_filename
        except Exception:
            return "clip_market1501.pt"
    
    @property
    @deprecated_config_access("ai", "TRACKER_HALF_PRECISION", "consolidated_settings.ai_config.tracker.half_precision")
    def TRACKER_HALF_PRECISION(self) -> bool:
        try:
            return self._consolidated.ai_config.tracker.half_precision
        except Exception:
            return False
    
    @property
    @deprecated_config_access("ai", "TRACKER_PER_CLASS", "consolidated_settings.ai_config.tracker.per_class")
    def TRACKER_PER_CLASS(self) -> bool:
        try:
            return self._consolidated.ai_config.tracker.per_class
        except Exception:
            return False
    
    @property
    @deprecated_config_access("ai", "REID_MODEL_TYPE", "consolidated_settings.ai_config.reid.reid_model_type")
    def REID_MODEL_TYPE(self) -> str:
        try:
            return self._consolidated.ai_config.reid.reid_model_type.value
        except Exception:
            return "clip"
    
    @property
    @deprecated_config_access("ai", "REID_MODEL_HALF_PRECISION", "consolidated_settings.ai_config.reid.half_precision")
    def REID_MODEL_HALF_PRECISION(self) -> bool:
        try:
            return self._consolidated.ai_config.reid.half_precision
        except Exception:
            return False
    
    # ReID-specific settings
    @property
    @deprecated_config_access("ai", "REID_SIMILARITY_METHOD", "consolidated_settings.ai_config.reid.similarity_method")
    def REID_SIMILARITY_METHOD(self) -> str:
        try:
            return self._consolidated.ai_config.reid.similarity_method.value
        except Exception:
            return "faiss_l2"
    
    @property
    @deprecated_config_access("ai", "REID_SIMILARITY_THRESHOLD", "consolidated_settings.ai_config.reid.similarity_threshold")
    def REID_SIMILARITY_THRESHOLD(self) -> float:
        try:
            return self._consolidated.ai_config.reid.similarity_threshold
        except Exception:
            return 0.65
    
    @property
    @deprecated_config_access("ai", "REID_L2_DISTANCE_THRESHOLD", "consolidated_settings.ai_config.reid.l2_distance_threshold")
    def REID_L2_DISTANCE_THRESHOLD(self) -> Optional[float]:
        try:
            return self._consolidated.ai_config.reid.l2_distance_threshold
        except Exception:
            return None
    
    @property
    @deprecated_config_access("ai", "REID_GALLERY_EMA_ALPHA", "consolidated_settings.ai_config.reid.gallery_ema_alpha")
    def REID_GALLERY_EMA_ALPHA(self) -> float:
        try:
            return self._consolidated.ai_config.reid.gallery_ema_alpha
        except Exception:
            return 0.9
    
    @property
    @deprecated_config_access("ai", "REID_REFRESH_INTERVAL_FRAMES", "consolidated_settings.ai_config.reid.refresh_interval_frames")
    def REID_REFRESH_INTERVAL_FRAMES(self) -> int:
        try:
            return self._consolidated.ai_config.reid.refresh_interval_frames
        except Exception:
            return 10
    
    @property
    @deprecated_config_access("ai", "REID_LOST_TRACK_BUFFER_FRAMES", "consolidated_settings.ai_config.reid.lost_track_buffer_frames")
    def REID_LOST_TRACK_BUFFER_FRAMES(self) -> int:
        try:
            return self._consolidated.ai_config.reid.lost_track_buffer_frames
        except Exception:
            return 200
    
    @property
    @deprecated_config_access("ai", "REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES", "consolidated_settings.ai_config.reid.main_gallery_prune_interval_frames")
    def REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES(self) -> int:
        try:
            return self._consolidated.ai_config.reid.main_gallery_prune_interval_frames
        except Exception:
            return 500
    
    @property
    @deprecated_config_access("ai", "REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES", "consolidated_settings.ai_config.derived_l2_distance_threshold")
    def REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES(self) -> int:
        try:
            return self._consolidated.ai_config.reid.lost_track_buffer_frames * 2
        except Exception:
            return 400
    
    @property
    @deprecated_config_access("ai", "TARGET_FPS", "consolidated_settings.ai_config.target_fps")
    def TARGET_FPS(self) -> int:
        try:
            return self._consolidated.ai_config.target_fps
        except Exception:
            return 23
    
    @property
    @deprecated_config_access("ai", "FRAME_JPEG_QUALITY", "consolidated_settings.ai_config.frame_jpeg_quality")
    def FRAME_JPEG_QUALITY(self) -> int:
        try:
            return self._consolidated.ai_config.frame_jpeg_quality
        except Exception:
            return 90
    
    # Export settings - Map to storage configuration
    @property
    @deprecated_config_access("storage", "EXPORT_BASE_DIR", "consolidated_settings.storage_config.exports.export_base_dir")
    def EXPORT_BASE_DIR(self) -> str:
        try:
            return self._consolidated.storage_config.exports.export_base_dir
        except Exception:
            return "./exports"
    
    @property
    @deprecated_config_access("storage", "EXPORT_EXPIRY_HOURS", "consolidated_settings.storage_config.exports.export_expiry_hours")
    def EXPORT_EXPIRY_HOURS(self) -> int:
        try:
            return self._consolidated.storage_config.exports.export_expiry_hours
        except Exception:
            return 24
    
    # Computed properties - Legacy compatibility
    @property
    def derived_l2_distance_threshold(self) -> float:
        """Legacy computed property for L2 distance threshold."""
        try:
            return self._consolidated.ai_config.derived_l2_distance_threshold
        except Exception:
            import math
            return math.sqrt(max(0, 2 * (1 - 0.65)))
    
    @property
    def resolved_reid_weights_path(self) -> Path:
        """Legacy computed property for ReID weights path."""
        try:
            return self._consolidated.ai_config.resolved_reid_weights_path
        except Exception:
            return Path("./weights/clip_market1501.pt")
    
    @property
    def resolved_homography_base_path(self) -> Path:
        """Legacy computed property for homography base path."""
        try:
            return self._consolidated.camera_config.resolved_homography_base_path
        except Exception:
            return Path("./homography_points")
    
    @property
    def normalized_possible_camera_overlaps(self) -> set:
        """Legacy computed property for normalized camera overlaps."""
        try:
            return self._consolidated.camera_config.normalized_possible_camera_overlaps
        except Exception:
            return set()
    
    # Model configuration compatibility - For Pydantic V2
    @property
    def model_config(self) -> Dict[str, Any]:
        """Pydantic model configuration compatibility."""
        return {
            "extra": "ignore",
            "env_file": ".env",
            "env_file_encoding": "utf-8"
        }


# Global backward compatibility instance
def create_legacy_settings() -> LegacySettingsWrapper:
    """
    Create legacy settings wrapper with consolidated settings.
    
    Returns:
        Legacy settings wrapper instance
    """
    consolidated_settings = get_settings()
    return LegacySettingsWrapper(consolidated_settings)


# Compatibility utilities
def enable_compatibility_warnings():
    """Enable deprecation warnings for configuration access."""
    warnings.filterwarnings("default", category=CompatibilityWarning)
    logger.info("Configuration compatibility warnings enabled")


def disable_compatibility_warnings():
    """Disable deprecation warnings for configuration access."""
    warnings.filterwarnings("ignore", category=CompatibilityWarning)
    logger.info("Configuration compatibility warnings disabled")


def get_migration_status() -> Dict[str, Any]:
    """
    Get migration status and recommendations.
    
    Returns:
        Dictionary containing migration progress and recommendations
    """
    # Check for legacy configuration usage (simplified)
    legacy_env_vars = [
        'DETECTOR_TYPE', 'TRACKER_TYPE', 'REID_MODEL_TYPE',
        'REDIS_HOST', 'POSTGRES_SERVER', 'S3_BUCKET_NAME'
    ]
    
    legacy_usage = {var: var in os.environ for var in legacy_env_vars}
    legacy_count = sum(legacy_usage.values())
    total_vars = len(legacy_env_vars)
    
    migration_progress = ((total_vars - legacy_count) / total_vars) * 100
    
    return {
        'migration_progress_percent': round(migration_progress, 1),
        'legacy_variables_remaining': legacy_count,
        'total_variables_checked': total_vars,
        'legacy_usage_details': legacy_usage,
        'recommendations': [
            "Gradually migrate environment variables to new consolidated format",
            "Use consolidated configuration properties instead of legacy Settings",
            "Test thoroughly with compatibility warnings enabled",
            "Update deployment scripts to use new environment variable names"
        ],
        'next_steps': [
            "Enable compatibility warnings in development",
            "Update CI/CD pipelines with new configuration format",
            "Document migration path for team members",
            "Schedule removal of compatibility layer after full migration"
        ]
    }


# Migration helper functions
def print_migration_guide():
    """Print migration guide for developers."""
    guide = """
    📋 Configuration Migration Guide - Phase 6 Consolidated Configuration
    
    🔄 Migration Steps:
    1. Replace legacy Settings imports:
       OLD: from app.core.config import settings
       NEW: from app.core.config_consolidated import consolidated_settings
    
    2. Update configuration access patterns:
       OLD: settings.DETECTOR_TYPE
       NEW: consolidated_settings.ai_config.detector.detector_type.value
    
    3. Use new configuration modules:
       - Core: consolidated_settings.core.*
       - AI: consolidated_settings.ai_config.*
       - Database: consolidated_settings.database_config.*
       - Storage: consolidated_settings.storage_config.*
       - Cameras: consolidated_settings.camera_config.*
    
    4. Update environment variables (gradual migration):
       OLD: DETECTOR_TYPE=fasterrcnn
       NEW: AI_DETECTOR_DETECTOR_TYPE=fasterrcnn
    
    5. Enable validation and monitoring:
       - Use validate_configuration_health() for health checks
       - Generate documentation with generate_configuration_documentation()
       - Monitor migration status with get_migration_status()
    
    ⚠️  Backward Compatibility:
    - Legacy access patterns work during migration period
    - Deprecation warnings help identify usage to update
    - Compatibility layer will be removed in future release
    
    📚 For detailed documentation, see config_validator.py
    """
    print(guide)