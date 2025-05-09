from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# Define a model for individual video set configurations
class VideoSetEnvironmentConfig(BaseModel):
    remote_base_key: str = Field(..., description="Base S3 key in DagsHub repo (e.g., 'video_s37/c01')")
    env_id: str = Field(..., description="Environment ID (e.g., 'campus')")
    cam_id: str = Field(..., description="Camera ID (e.g., 'c01')")
    num_sub_videos: int = Field(..., gt=0, description="Total number of sub-videos available for this camera.")
    sub_video_filename_pattern: str = Field(default="sub_video_{idx:02d}.mp4", description="Filename pattern. '{idx:02d}' for index.")

class Settings(BaseSettings):
    APP_NAME: str = "SpotOn Backend"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # DagsHub Configuration
    DAGSHUB_REPO_OWNER: str = "Jwizzed"
    DAGSHUB_REPO_NAME: str = "spoton_ml"

    # Local Data Directories
    LOCAL_VIDEO_DOWNLOAD_DIR: str = "./downloaded_videos"
    LOCAL_FRAME_EXTRACTION_DIR: str = "./extracted_frames"

    # Video Source Definitions
    VIDEO_SETS: List[VideoSetEnvironmentConfig] = [
        # Campus Cameras (s47)
        # TODO: this is duplicate, the remote base key already tell the cam id already.
        VideoSetEnvironmentConfig(remote_base_key="video_s47/c01", env_id="campus", cam_id="c01", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s47/c02", env_id="campus", cam_id="c02", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s47/c03", env_id="campus", cam_id="c03", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s47/c05", env_id="campus", cam_id="c05", num_sub_videos=4),
        # Factory Cameras (s10 / s14 reference)
        # Assuming s14 structure applies based on repo names
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c09", env_id="factory", cam_id="c09", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c12", env_id="factory", cam_id="c12", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c13", env_id="factory", cam_id="c13", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s14/c16", env_id="factory", cam_id="c16", num_sub_videos=4),
    ]

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
    DATABASE_URL: Optional[str] = None # Will be constructed if needed

    # --- AI Model & Pipeline Configuration ---
    # Detector Settings
    DETECTOR_TYPE: str = "fasterrcnn" # Type of detector to use
    PERSON_CLASS_ID: int = 1          # Class ID for 'person' (1 for torchvision FasterRCNN)
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    DETECTION_USE_AMP: bool = False   # Use AMP for detector inference (CUDA only)

    # Tracker & Re-ID Settings (Integrated via BoxMOT)
    TRACKER_TYPE: str = "botsort" # Type of tracker to use (BoxMOT compatible)
    WEIGHTS_DIR: str = "./weights" # Directory where ReID weights are stored
    REID_WEIGHTS_PATH: str = "clip_market1501.pt" # ReID weights filename (relative to WEIGHTS_DIR or downloaded by BoxMOT)
    TRACKER_HALF_PRECISION: bool = False # Use half precision for tracker/ReID (CUDA only)
    TRACKER_PER_CLASS: bool = False      # Use per-class tracking logic

    # --- Video Processing ---
    TARGET_FPS: int = 23 # Target FPS for frame extraction
    FRAME_JPEG_QUALITY: int = 90

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

    @property
    def resolved_reid_weights_path(self) -> Optional[Path]:
        """Returns the absolute path to the ReID weights file, or None if not found."""
        base_dir = Path(self.WEIGHTS_DIR)
        weights_file = base_dir / self.REID_WEIGHTS_PATH
        if weights_file.is_file():
            return weights_file.resolve()
        # Return the raw path as identifier if not found locally (BoxMOT might download)
        # Or return None if local existence is mandatory
        # Let's return the relative path for BoxMOT to handle potentially
        return Path(self.REID_WEIGHTS_PATH)
        # return None # If local file must exist

settings = Settings()