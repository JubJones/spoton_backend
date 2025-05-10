from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Tuple
from pathlib import Path

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
    # These might be redundant if all S3 access is through boto3 with the above credentials.
    # However, keeping them in case the dagshub library is used elsewhere.
    DAGSHUB_REPO_OWNER: str = "Jwizzed"
    DAGSHUB_REPO_NAME: str = "spoton_ml" # This is often the same as S3_BUCKET_NAME for DagsHub

    # Local Data Directories
    LOCAL_VIDEO_DOWNLOAD_DIR: str = "./downloaded_videos" # Relative to WORKDIR /app -> /app/downloaded_videos
    LOCAL_FRAME_EXTRACTION_DIR: str = "./extracted_frames" # Relative to WORKDIR /app -> /app/extracted_frames

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
    WEIGHTS_DIR: str = "./weights" # Relative to WORKDIR /app -> /app/weights
    REID_WEIGHTS_PATH: str = "clip_market1501.pt" # Filename of the ReID weights
    TRACKER_HALF_PRECISION: bool = False
    TRACKER_PER_CLASS: bool = False
    REID_SIMILARITY_THRESHOLD: float = 0.65

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
        BoxMOT will attempt to download to this path if the file doesn't exist.
        The WORKDIR in the Docker container is /app.
        So, self.WEIGHTS_DIR (e.g., "./weights") becomes /app/weights.
        """
        weights_dir_in_container = Path(self.WEIGHTS_DIR)
        reid_weights_file_path = weights_dir_in_container / self.REID_WEIGHTS_PATH
        return reid_weights_file_path.resolve()

settings = Settings()