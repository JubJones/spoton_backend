from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Tuple, Set
from pathlib import Path
from app.common_types import CameraID, ExitRuleModel, ExitDirection, CameraHandoffDetailConfig

# Define a model for individual video set configurations
class VideoSetEnvironmentConfig(BaseModel):
    remote_base_key: str = Field(..., description="Base S3 key (e.g., 'video_s37/c01')")
    env_id: str = Field(..., description="Environment ID (e.g., 'campus')")
    cam_id: str = Field(..., description="Camera ID (e.g., 'c01')")
    num_sub_videos: int = Field(..., gt=0, description="Total number of sub-videos available for this camera.")
    sub_video_filename_pattern: str = Field(default="sub_video_{idx:02d}.mp4", description="Filename pattern. '{idx:02d}' for index.")


class Settings(BaseSettings):
    APP_NAME: str = "SpotOn Backend"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # --- S3 (DagsHub/Generic) Configuration ---
    S3_ENDPOINT_URL: Optional[str] = "https://s3.dagshub.com"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: str = "spoton_ml"

    # --- DagsHub Specific Configuration ---
    DAGSHUB_REPO_OWNER: str = "Jwizzed"
    DAGSHUB_REPO_NAME: str = "spoton_ml"

    # Local Data Directories
    LOCAL_VIDEO_DOWNLOAD_DIR: str = "./downloaded_videos"
    LOCAL_FRAME_EXTRACTION_DIR: str = "./extracted_frames"

    # Video Source Definitions
    VIDEO_SETS: List[VideoSetEnvironmentConfig] = [
        # --- Campus (s37) Environment - Commented Out ---
        # VideoSetEnvironmentConfig(remote_base_key="video_s37/c01", env_id="campus", cam_id="c01", num_sub_videos=4),
        # VideoSetEnvironmentConfig(remote_base_key="video_s37/c02", env_id="campus", cam_id="c02", num_sub_videos=4),
        # VideoSetEnvironmentConfig(remote_base_key="video_s37/c03", env_id="campus", cam_id="c03", num_sub_videos=4),
        # VideoSetEnvironmentConfig(remote_base_key="video_s37/c05", env_id="campus", cam_id="c05", num_sub_videos=4),
        # --- Factory (s14) Environment ---
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c09", env_id="factory", cam_id="c09", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c12", env_id="factory", cam_id="c12", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c13", env_id="factory", cam_id="c13", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c16", env_id="factory", cam_id="c16", num_sub_videos=4),
    ]

    # --- Handoff and Detailed Camera Configuration ---
    # Homography matrix paths are relative to HOMOGRAPHY_DATA_DIR
    # For factory (s14 videos), homography files refer to scene s10.
    CAMERA_HANDOFF_DETAILS: Dict[Tuple[str, str], CameraHandoffDetailConfig] = {
        # --- Campus Environment - Commented Out ---
        # ("campus", "c01"): CameraHandoffDetailConfig(exit_rules=[
        #     ExitRuleModel(direction=ExitDirection("right"), target_cam_id=CameraID("c02"), target_entry_area="left_side"),
        #     ExitRuleModel(direction=ExitDirection("down"), target_cam_id=CameraID("c03"), target_entry_area="top_side"),
        # ], homography_matrix_path="homography_points_c01_scene_campus.npz"), # Actual file might be _scene_s37 based on video
        # ("campus", "c02"): CameraHandoffDetailConfig(exit_rules=[
        #     ExitRuleModel(direction=ExitDirection("left"), target_cam_id=CameraID("c01"), target_entry_area="right_side"),
        #     ExitRuleModel(direction=ExitDirection("down"), target_cam_id=CameraID("c05"), target_entry_area="top_right"),
        # ], homography_matrix_path="homography_points_c02_scene_campus.npz"),
        # ("campus", "c03"): CameraHandoffDetailConfig(exit_rules=[
        #     ExitRuleModel(direction=ExitDirection("up"), target_cam_id=CameraID("c01"), target_entry_area="bottom_side"),
        # ], homography_matrix_path="homography_points_c03_scene_campus.npz"),
        #  ("campus", "c05"): CameraHandoffDetailConfig(exit_rules=[
        #     ExitRuleModel(direction=ExitDirection("up"), target_cam_id=CameraID("c02"), target_entry_area="bottom_right"),
        # ], homography_matrix_path="homography_points_c05_scene_campus.npz"),
        # --- Factory Environment (Videos: s14, Homography Scene: s10) ---
        ("factory", "c09"): CameraHandoffDetailConfig(
            exit_rules=[
                ExitRuleModel(direction=ExitDirection("down"), target_cam_id=CameraID("c13"), target_entry_area="upper right", notes="wait; overlap c13/c16 possible"),
            ],
            homography_matrix_path="homography_points_c09_scene_s14.npz" # Aligned with actual file
        ),
        ("factory", "c12"): CameraHandoffDetailConfig(
            exit_rules=[
                ExitRuleModel(direction=ExitDirection("left"), target_cam_id=CameraID("c13"), target_entry_area="upper left", notes="overlap c13 possible"),
            ],
            homography_matrix_path="homography_points_c12_scene_s14.npz" # Aligned with actual file
        ),
        ("factory", "c13"): CameraHandoffDetailConfig(
            exit_rules=[
                 ExitRuleModel(direction=ExitDirection("right"), target_cam_id=CameraID("c09"), target_entry_area="down", notes="wait; overlap c09 possible"),
                 ExitRuleModel(direction=ExitDirection("left"), target_cam_id=CameraID("c12"), target_entry_area="upper left", notes="overlap c12 possible"),
            ],
            homography_matrix_path="homography_points_c13_scene_s14.npz" # Aligned with actual file
        ),
        ("factory", "c16"): CameraHandoffDetailConfig(
            exit_rules=[], # No exit rules defined from c16 in user example
            homography_matrix_path="homography_points_c16_scene_s14.npz" # Aligned with actual file
        ),
    }
    MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT: float = Field(default=0.40, description="Min BBox area ratio in an exit quadrant to trigger handoff.")
    HOMOGRAPHY_DATA_DIR: str = Field(default="./homography_points", description="Directory relative to app root for homography .npz files.")


    POSSIBLE_CAMERA_OVERLAPS: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            # --- Campus Overlaps - Commented Out ---
            # ("c01", "c02"), ("c01", "c03"),
            # ("c02", "c05"),
            # --- Factory Overlaps ---
            ("c09", "c12"), ("c12", "c13"), ("c13", "c16")
        ],
        description="List of camera ID pairs that have potential visual overlap."
    )

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # TimescaleDB (PostgreSQL) Configuration
    POSTGRES_USER: str = "spoton_user"
    POSTGRES_PASSWORD: str = "spoton_password"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "spotondb"
    DATABASE_URL: Optional[str] = None

    # AI Model & Pipeline Configuration
    DETECTOR_TYPE: str = "fasterrcnn"
    PERSON_CLASS_ID: int = 1
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    DETECTION_USE_AMP: bool = False

    TRACKER_TYPE: str = "botsort"
    WEIGHTS_DIR: str = "./weights" # This is for model weights like .pt files
    REID_WEIGHTS_PATH: str = "clip_market1501.pt"
    TRACKER_HALF_PRECISION: bool = False
    TRACKER_PER_CLASS: bool = False
    
    REID_SIMILARITY_THRESHOLD: float = 0.65
    REID_GALLERY_EMA_ALPHA: float = 0.9
    REID_REFRESH_INTERVAL_FRAMES: int = 10
    REID_LOST_TRACK_BUFFER_FRAMES: int = 200
    REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES: int = 500
    REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES: int = REID_LOST_TRACK_BUFFER_FRAMES * 2

    TARGET_FPS: int = 23 
    FRAME_JPEG_QUALITY: int = 90

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

    @property
    def resolved_reid_weights_path(self) -> Path:
        weights_dir_in_container = Path(self.WEIGHTS_DIR) # e.g., /app/weights
        reid_weights_file_path = weights_dir_in_container / self.REID_WEIGHTS_PATH
        return reid_weights_file_path.resolve()
    
    @property
    def resolved_homography_base_path(self) -> Path:
        """Returns the resolved base path for homography files, relative to app root."""
        return Path(self.HOMOGRAPHY_DATA_DIR).resolve()

    @property
    def normalized_possible_camera_overlaps(self) -> Set[Tuple[CameraID, CameraID]]:
        return {tuple(sorted((CameraID(c1), CameraID(c2)))) for c1, c2 in self.POSSIBLE_CAMERA_OVERLAPS}

settings = Settings()