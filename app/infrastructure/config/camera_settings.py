"""
Camera and video infrastructure configuration settings.

Clean infrastructure configuration for camera systems following Phase 6:
Configuration consolidation requirements. Handles camera configurations,
video sets, homography matrices, and handoff rules with proper validation.
"""
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class QuadrantName(Enum):
    """Camera view quadrant names for handoff rules."""
    UPPER_LEFT = "upper_left"
    UPPER_RIGHT = "upper_right" 
    LOWER_LEFT = "lower_left"
    LOWER_RIGHT = "lower_right"


class CameraType(Enum):
    """Types of camera installations."""
    FIXED = "fixed"
    PTZ = "ptz"
    DOME = "dome"
    MOBILE = "mobile"


class EnvironmentType(Enum):
    """Environment types for camera deployments."""
    CAMPUS = "campus"
    FACTORY = "factory"
    RETAIL = "retail"
    OFFICE = "office"


class VideoSetConfig(BaseModel):
    """Video set configuration for specific camera/environment combinations."""
    
    remote_base_key: str = Field(
        ...,
        min_length=1,
        description="Base S3 key (e.g., 'video_s37/c01')"
    )
    env_id: str = Field(
        ...,
        min_length=1,
        description="Environment ID (e.g., 'campus')"
    )
    cam_id: str = Field(
        ...,
        min_length=1,
        description="Camera ID (e.g., 'c01')"
    )
    num_sub_videos: int = Field(
        ...,
        gt=0,
        le=100,
        description="Total number of sub-videos available"
    )
    sub_video_filename_pattern: str = Field(
        default="sub_video_{idx:02d}.mp4",
        description="Filename pattern with {idx} placeholder"
    )
    
    @validator('sub_video_filename_pattern')
    def validate_filename_pattern(cls, v):
        """Validate filename pattern contains proper placeholder."""
        if '{idx' not in v:
            raise ValueError("Filename pattern must contain {idx} placeholder")
        return v


class ExitRule(BaseModel):
    """Camera handoff exit rule configuration."""
    
    source_exit_quadrant: QuadrantName = Field(
        ...,
        description="Source camera exit quadrant"
    )
    target_cam_id: str = Field(
        ...,
        min_length=1,
        description="Target camera ID"
    )
    target_entry_area: str = Field(
        ...,
        min_length=1,
        description="Target camera entry area description"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes for handoff rule"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for handoff"
    )


class HandoffConfig(BaseModel):
    """Camera handoff configuration for environment/camera combination."""
    
    exit_rules: List[ExitRule] = Field(
        default_factory=list,
        description="List of exit rules for this camera"
    )
    homography_matrix_path: str = Field(
        ...,
        min_length=1,
        description="Path to homography matrix file"
    )
    overlap_cameras: List[str] = Field(
        default_factory=list,
        description="List of cameras that may overlap with this one"
    )
    
    def validate_handoff_config(self) -> None:
        """Validate handoff configuration."""
        if self.exit_rules:
            # Check for duplicate exit quadrants
            quadrants = [rule.source_exit_quadrant for rule in self.exit_rules]
            if len(quadrants) != len(set(quadrants)):
                logger.warning("Duplicate exit quadrants detected in handoff rules")
        
        # Validate homography path
        homography_path = Path(self.homography_matrix_path)
        if not homography_path.suffix in ['.npz', '.npy']:
            logger.warning(f"Homography file should be .npz or .npy format: {self.homography_matrix_path}")


class CameraConfig(BaseModel):
    """Individual camera configuration."""
    
    # Basic camera properties
    camera_id: str = Field(
        ...,
        min_length=1,
        description="Unique camera identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable camera name"
    )
    camera_type: CameraType = Field(
        default=CameraType.FIXED,
        description="Type of camera installation"
    )
    
    # Technical specifications
    resolution: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Camera resolution (width, height)"
    )
    frame_rate: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Camera frame rate (fps)"
    )
    field_of_view: float = Field(
        default=60.0,
        gt=0.0,
        le=180.0,
        description="Camera field of view (degrees)"
    )
    
    # Physical properties
    position: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        description="Camera physical position coordinates"
    )
    orientation: float = Field(
        default=0.0,
        ge=0.0,
        lt=360.0,
        description="Camera orientation angle (degrees)"
    )
    
    # Operational settings
    is_active: bool = Field(
        default=True,
        description="Whether camera is currently active"
    )
    quality_level: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Video quality level (1=highest, 5=lowest)"
    )
    
    # Calibration information
    last_calibrated: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last calibration"
    )
    calibration_accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calibration accuracy score"
    )
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate camera resolution."""
        width, height = v
        if width < 640 or height < 480:
            raise ValueError(f"Resolution {width}x{height} is below minimum (640x480)")
        if width > 4096 or height > 4096:
            raise ValueError(f"Resolution {width}x{height} exceeds maximum (4096x4096)")
        return v
    
    @validator('position')
    def validate_position(cls, v):
        """Validate camera position coordinates."""
        required_keys = ['x', 'y']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Position must contain '{key}' coordinate")
        return v


class CameraSettings(BaseSettings):
    """
    Consolidated camera and video infrastructure configuration.
    
    Handles camera configurations, video sets, homography matrices, and handoff rules
    with proper validation and environment management. Follows Phase 6 configuration
    consolidation requirements.
    """
    
    # Video set configurations
    video_sets: List[VideoSetConfig] = Field(
        default_factory=lambda: [
            # Campus environment
            VideoSetConfig(remote_base_key="video_s37/c01", env_id="campus", cam_id="c01", num_sub_videos=4),
            VideoSetConfig(remote_base_key="video_s37/c02", env_id="campus", cam_id="c02", num_sub_videos=4),
            VideoSetConfig(remote_base_key="video_s37/c03", env_id="campus", cam_id="c03", num_sub_videos=4),
            VideoSetConfig(remote_base_key="video_s37/c05", env_id="campus", cam_id="c05", num_sub_videos=4),
            # Factory environment
            VideoSetConfig(remote_base_key="video_s14/c09", env_id="factory", cam_id="c09", num_sub_videos=4),
            VideoSetConfig(remote_base_key="video_s14/c12", env_id="factory", cam_id="c12", num_sub_videos=4),
            VideoSetConfig(remote_base_key="video_s14/c13", env_id="factory", cam_id="c13", num_sub_videos=4),
            VideoSetConfig(remote_base_key="video_s14/c16", env_id="factory", cam_id="c16", num_sub_videos=4),
        ],
        description="List of video set configurations"
    )
    
    # Camera handoff configurations
    camera_handoff_details: Dict[Tuple[str, str], HandoffConfig] = Field(
        default_factory=lambda: {
            # Campus handoffs
            ("campus", "c01"): HandoffConfig(
                exit_rules=[
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_RIGHT,
                        target_cam_id="c03",
                        target_entry_area="bottom_left"
                    )
                ],
                homography_matrix_path="homography_points_c01_scene_s47.npz"
            ),
            ("campus", "c02"): HandoffConfig(
                exit_rules=[
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_RIGHT,
                        target_cam_id="c05",
                        target_entry_area="upper left"
                    )
                ],
                homography_matrix_path="homography_points_c02_scene_s47.npz"
            ),
            ("campus", "c03"): HandoffConfig(
                exit_rules=[
                    ExitRule(
                        source_exit_quadrant=QuadrantName.LOWER_LEFT,
                        target_cam_id="c01",
                        target_entry_area="upper_right"
                    ),
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_RIGHT,
                        target_cam_id="c05",
                        target_entry_area="upper left"
                    )
                ],
                homography_matrix_path="homography_points_c03_scene_s47.npz"
            ),
            ("campus", "c05"): HandoffConfig(
                exit_rules=[
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_LEFT,
                        target_cam_id="c02",
                        target_entry_area="upper_right"
                    ),
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_LEFT,
                        target_cam_id="c03",
                        target_entry_area="upper_right"
                    )
                ],
                homography_matrix_path="homography_points_c05_scene_s47.npz"
            ),
            # Factory handoffs
            ("factory", "c09"): HandoffConfig(
                exit_rules=[
                    ExitRule(
                        source_exit_quadrant=QuadrantName.LOWER_LEFT,
                        target_cam_id="c13",
                        target_entry_area="upper right",
                        notes="wait; overlap c13/c16 possible"
                    ),
                    ExitRule(
                        source_exit_quadrant=QuadrantName.LOWER_RIGHT,
                        target_cam_id="c13",
                        target_entry_area="upper right",
                        notes="wait; overlap c13/c16 possible"
                    )
                ],
                homography_matrix_path="homography_points_c09_scene_s14.npz"
            ),
            ("factory", "c12"): HandoffConfig(
                exit_rules=[
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_LEFT,
                        target_cam_id="c13",
                        target_entry_area="upper left",
                        notes="overlap c13 possible"
                    ),
                    ExitRule(
                        source_exit_quadrant=QuadrantName.LOWER_LEFT,
                        target_cam_id="c13",
                        target_entry_area="upper left",
                        notes="overlap c13 possible"
                    )
                ],
                homography_matrix_path="homography_points_c12_scene_s14.npz"
            ),
            ("factory", "c13"): HandoffConfig(
                exit_rules=[
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_RIGHT,
                        target_cam_id="c09",
                        target_entry_area="down",
                        notes="wait; overlap c09 possible"
                    ),
                    ExitRule(
                        source_exit_quadrant=QuadrantName.LOWER_RIGHT,
                        target_cam_id="c09",
                        target_entry_area="down",
                        notes="wait; overlap c09 possible"
                    ),
                    ExitRule(
                        source_exit_quadrant=QuadrantName.UPPER_LEFT,
                        target_cam_id="c12",
                        target_entry_area="upper left",
                        notes="overlap c12 possible"
                    ),
                    ExitRule(
                        source_exit_quadrant=QuadrantName.LOWER_LEFT,
                        target_cam_id="c12",
                        target_entry_area="upper left",
                        notes="overlap c12 possible"
                    )
                ],
                homography_matrix_path="homography_points_c13_scene_s14.npz"
            ),
            ("factory", "c16"): HandoffConfig(
                exit_rules=[],
                homography_matrix_path="homography_points_c16_scene_s14.npz"
            ),
        },
        description="Camera handoff configuration details"
    )
    
    # Overlap detection settings
    min_bbox_overlap_ratio: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum bounding box overlap ratio for quadrant detection"
    )
    possible_camera_overlaps: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("c09", "c12"), ("c12", "c13"), ("c13", "c16"),
            ("c01", "c03"), ("c02", "c03"), ("c03", "c05")
        ],
        description="List of camera pairs that may have overlapping views"
    )
    
    # Homography settings
    homography_data_dir: str = Field(
        default="./homography_points",
        description="Directory containing homography matrix files"
    )
    homography_cache_enabled: bool = Field(
        default=True,
        description="Enable homography matrix caching"
    )
    homography_validation_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Maximum acceptable homography error threshold"
    )
    
    def __init__(self, **kwargs):
        """Initialize camera settings with validation."""
        super().__init__(**kwargs)
        self._validate_all_settings()
        logger.debug("CameraSettings initialized successfully")
    
    @property
    def resolved_homography_base_path(self) -> Path:
        """Get resolved path to homography data directory."""
        return Path(self.homography_data_dir).resolve()
    
    @property
    def normalized_possible_camera_overlaps(self) -> Set[Tuple[str, str]]:
        """Get normalized camera overlap pairs (sorted tuples)."""
        return {tuple(sorted((c1, c2))) for c1, c2 in self.possible_camera_overlaps}
    
    def get_video_set_by_camera(self, env_id: str, cam_id: str) -> Optional[VideoSetConfig]:
        """
        Get video set configuration for specific environment/camera combination.
        
        Args:
            env_id: Environment identifier
            cam_id: Camera identifier
            
        Returns:
            VideoSetConfig if found, None otherwise
        """
        for video_set in self.video_sets:
            if video_set.env_id == env_id and video_set.cam_id == cam_id:
                return video_set
        return None
    
    def get_handoff_config(self, env_id: str, cam_id: str) -> Optional[HandoffConfig]:
        """
        Get handoff configuration for specific environment/camera combination.
        
        Args:
            env_id: Environment identifier
            cam_id: Camera identifier
            
        Returns:
            HandoffConfig if found, None otherwise
        """
        return self.camera_handoff_details.get((env_id, cam_id))
    
    def get_cameras_by_environment(self, env_id: str) -> List[str]:
        """
        Get list of camera IDs for specific environment.
        
        Args:
            env_id: Environment identifier
            
        Returns:
            List of camera IDs in the environment
        """
        return [vs.cam_id for vs in self.video_sets if vs.env_id == env_id]
    
    def get_environments(self) -> List[str]:
        """
        Get list of all configured environments.
        
        Returns:
            List of unique environment IDs
        """
        return list(set(vs.env_id for vs in self.video_sets))
    
    def validate_config(self) -> None:
        """Validate complete camera configuration."""
        self._validate_all_settings()
    
    def _validate_all_settings(self) -> None:
        """Validate all camera settings."""
        try:
            # Validate homography directory exists
            homography_path = self.resolved_homography_base_path
            if not homography_path.exists():
                logger.warning(f"Homography data directory not found: {homography_path}")
                homography_path.mkdir(parents=True, exist_ok=True)
            
            # Validate video sets
            self._validate_video_sets()
            
            # Validate handoff configurations
            self._validate_handoff_configurations()
            
            # Cross-validation between video sets and handoff configs
            self._cross_validate_configurations()
            
        except Exception as e:
            logger.error(f"Camera settings validation failed: {e}")
            raise
    
    def _validate_video_sets(self) -> None:
        """Validate video set configurations."""
        env_cam_pairs = set()
        
        for video_set in self.video_sets:
            # Check for duplicates
            pair = (video_set.env_id, video_set.cam_id)
            if pair in env_cam_pairs:
                raise ValueError(f"Duplicate video set configuration: {pair}")
            env_cam_pairs.add(pair)
            
            # Validate filename pattern
            try:
                test_filename = video_set.sub_video_filename_pattern.format(idx=1)
                if not test_filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    logger.warning(f"Unusual video file extension in pattern: {test_filename}")
            except Exception as e:
                raise ValueError(f"Invalid filename pattern in video set {pair}: {e}")
    
    def _validate_handoff_configurations(self) -> None:
        """Validate handoff configurations."""
        for (env_id, cam_id), handoff_config in self.camera_handoff_details.items():
            handoff_config.validate_handoff_config()
            
            # Check homography file path
            homography_file = self.resolved_homography_base_path / handoff_config.homography_matrix_path
            if not homography_file.exists():
                logger.warning(f"Homography file not found for {env_id}/{cam_id}: {homography_file}")
    
    def _cross_validate_configurations(self) -> None:
        """Cross-validate video sets and handoff configurations."""
        video_set_pairs = {(vs.env_id, vs.cam_id) for vs in self.video_sets}
        handoff_pairs = set(self.camera_handoff_details.keys())
        
        # Check for missing handoff configurations
        missing_handoffs = video_set_pairs - handoff_pairs
        if missing_handoffs:
            logger.warning(f"Video sets without handoff configurations: {missing_handoffs}")
        
        # Check for orphaned handoff configurations
        orphaned_handoffs = handoff_pairs - video_set_pairs
        if orphaned_handoffs:
            logger.warning(f"Handoff configurations without video sets: {orphaned_handoffs}")
        
        # Validate handoff target cameras exist
        for (env_id, cam_id), handoff_config in self.camera_handoff_details.items():
            for exit_rule in handoff_config.exit_rules:
                target_pair = (env_id, exit_rule.target_cam_id)
                if target_pair not in video_set_pairs:
                    logger.warning(f"Handoff target camera not found: {target_pair} referenced by {env_id}/{cam_id}")
    
    def get_camera_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive camera configuration summary.
        
        Returns:
            Dictionary containing configuration overview
        """
        environments = {}
        for env_id in self.get_environments():
            cameras = self.get_cameras_by_environment(env_id)
            environments[env_id] = {
                'camera_count': len(cameras),
                'camera_ids': cameras,
                'video_sets': len([vs for vs in self.video_sets if vs.env_id == env_id]),
                'handoff_rules': sum(
                    len(hc.exit_rules) 
                    for (e, c), hc in self.camera_handoff_details.items() 
                    if e == env_id
                )
            }
        
        return {
            'environments': environments,
            'total_cameras': len(self.video_sets),
            'total_video_sets': len(self.video_sets),
            'total_handoff_configs': len(self.camera_handoff_details),
            'overlap_pairs': len(self.possible_camera_overlaps),
            'settings': {
                'homography_data_dir': self.homography_data_dir,
                'min_overlap_ratio': self.min_bbox_overlap_ratio,
                'homography_cache_enabled': self.homography_cache_enabled
            },
            'validation_timestamp': datetime.utcnow().isoformat()
        }
    
    class Config:
        env_prefix = "CAMERA_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        use_enum_values = True


# Default camera settings factory
def get_camera_settings() -> CameraSettings:
    """
    Get camera settings with environment-specific defaults.
    
    Returns:
        Configured CameraSettings instance
    """
    return CameraSettings()