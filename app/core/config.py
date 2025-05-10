from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from app.common_types import CameraID

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
    AWS_ACCESS_KEY_ID: Optional[str] = None # Read from .env
    AWS_SECRET_ACCESS_KEY: Optional[str] = None # Read from .env
    S3_BUCKET_NAME: str = "spoton_ml" # Read from .env, defaults to "spoton_ml"

    # --- DagsHub Specific Configuration (if other DagsHub library features are used) ---
    DAGSHUB_REPO_OWNER: str = "Jwizzed"
    DAGSHUB_REPO_NAME: str = "spoton_ml"

    # Local Data Directories
    LOCAL_VIDEO_DOWNLOAD_DIR: str = "./downloaded_videos"
    LOCAL_FRAME_EXTRACTION_DIR: str = "./extracted_frames"

    # Video Source Definitions
    VIDEO_SETS: List[VideoSetEnvironmentConfig] = [
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c01", env_id="campus", cam_id="c01", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c02", env_id="campus", cam_id="c02", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c03", env_id="campus", cam_id="c03", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c05", env_id="campus", cam_id="c05", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c09", env_id="factory", cam_id="c09", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c12", env_id="factory", cam_id="c12", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c13", env_id="factory", cam_id="c13", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c16", env_id="factory", cam_id="c16", num_sub_videos=4),
    ]
    # --- NEW: Define possible camera overlaps for conceptual handoff influence ---
    # Each tuple represents a pair of camera IDs that might have an overlap.
    # Example: [("c01", "c02"), ("c02", "c03")]
    POSSIBLE_CAMERA_OVERLAPS: List[Tuple[str, str]] = Field(default_factory=list, description="List of camera ID pairs that have potential visual overlap.")


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

    # --- AI Model & Pipeline Configuration ---
    DETECTOR_TYPE: str = "fasterrcnn"
    PERSON_CLASS_ID: int = 1
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    DETECTION_USE_AMP: bool = False

    TRACKER_TYPE: str = "botsort"
    WEIGHTS_DIR: str = "./weights"
    REID_WEIGHTS_PATH: str = "clip_market1501.pt"
    TRACKER_HALF_PRECISION: bool = False
    TRACKER_PER_CLASS: bool = False
    
    # --- Re-ID Logic Configuration (NEW/MODIFIED) ---
    REID_SIMILARITY_THRESHOLD: float = 0.65 # As per POC
    REID_GALLERY_EMA_ALPHA: float = 0.9 # Exponential Moving Average for gallery updates
    REID_REFRESH_INTERVAL_FRAMES: int = 10 # Processed frames between ReID updates for a track
    REID_LOST_TRACK_BUFFER_FRAMES: int = 200 # Frames before a lost track feature is purged from lost gallery
    REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES: int = 500 # How often to check for main gallery pruning
    REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES: int = REID_LOST_TRACK_BUFFER_FRAMES * 2 # Prune if GID unseen for this many frames in main gallery

    TARGET_FPS: int = 23
    FRAME_JPEG_QUALITY: int = 90

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

    @property
    def resolved_reid_weights_path(self) -> Path:
        """
        Returns the full, resolved path within the configured WEIGHTS_DIR for the ReID weights.
        """
        weights_dir_in_container = Path(self.WEIGHTS_DIR)
        reid_weights_file_path = weights_dir_in_container / self.REID_WEIGHTS_PATH
        return reid_weights_file_path.resolve()
    
    @property
    def normalized_possible_camera_overlaps(self) -> List[Tuple[CameraID, CameraID]]:
        """Returns a normalized list of camera overlaps (sorted tuples)."""
        # Ensure CameraID type consistency if needed, though str comparison works
        return [tuple(sorted(pair)) for pair in self.POSSIBLE_CAMERA_OVERLAPS]


settings = Settings()