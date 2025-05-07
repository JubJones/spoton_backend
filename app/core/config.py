from pydantic_settings import BaseSettings
from typing import List, Optional # Ensure List is imported if Dict is used later with it. The provided snippet uses List for CAMERA_HOMOGRAPHIES type hint if uncommented.

class Settings(BaseSettings):
    """
    Application settings.

    Values are loaded from environment variables.
    A .env file can be used for local development.
    """
    APP_NAME: str = "SpotOn Backend"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # S3 Configuration
    S3_BUCKET_NAME: str
    S3_ACCESS_KEY_ID: str
    S3_SECRET_ACCESS_KEY: str
    S3_REGION_NAME: str
    S3_ENDPOINT_URL: Optional[str] = None # For MinIO or other S3-compatible services

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
    DATABASE_URL: Optional[str] = None #SQLALCHEMY_DATABASE_URL if using SQLAlchemy

    # AI Model Paths/Configuration (example)
    DETECTOR_MODEL_PATH: Optional[str] = None
    TRACKER_CONFIG_PATH: Optional[str] = None
    CLIP_MODEL_NAME: str = "ViT-B/32" # Example CLIP model

    # Homography matrices (could be loaded from a file or DB)
    # Example: CAMERA_HOMOGRAPHIES: Dict[str, List[List[float]]] = {} # If using this, ensure Dict is imported from typing

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Pydantic V2: use `model_config = SettingsConfigDict(extra='ignore')`

settings = Settings()
