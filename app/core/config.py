from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict

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
    # S3_ENDPOINT_URL: Optional[str] = None # For DagsHub if explicit endpoint is needed

    # Local Data Directories
    LOCAL_VIDEO_DOWNLOAD_DIR: str = "./downloaded_videos"
    LOCAL_FRAME_EXTRACTION_DIR: str = "./extracted_frames"

    # Video Source Definitions (similar to VIDEO_SETS_TO_DOWNLOAD)
    # This could also be loaded from a separate JSON/YAML file.
    VIDEO_SETS: List[VideoSetEnvironmentConfig] = [
        # Campus Cameras
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c01", env_id="campus", cam_id="c01", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c02", env_id="campus", cam_id="c02", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c03", env_id="campus", cam_id="c03", num_sub_videos=4),
        VideoSetEnvironmentConfig(remote_base_key="video_s37/c05", env_id="campus", cam_id="c05", num_sub_videos=4),
        # Factory Cameras
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
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    DATABASE_URL: Optional[str] = None

    # AI Model Paths/Configuration
    DETECTOR_MODEL_PATH: Optional[str] = None
    CLIP_MODEL_NAME: str = "ViT-B/32"

    # Detector Settings
    PERSON_CLASS_ID: int = 1
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    DETECTION_USE_AMP: bool = False

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()