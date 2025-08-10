"""
AI model infrastructure configuration settings.

Clean infrastructure configuration for AI models following Phase 6:
Configuration consolidation requirements. Handles AI-specific settings
with proper validation and environment management.
"""
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DetectorType(Enum):
    """Supported person detection models."""
    FASTER_RCNN = "fasterrcnn"
    YOLO = "yolo"


class TrackerType(Enum):
    """Supported multi-object tracker types."""
    BOTSORT = "botsort"
    BYTETRACK = "bytetrack"


class ReIDModelType(Enum):
    """Supported re-identification model types."""
    CLIP = "clip"
    RESNET = "resnet"


class SimilarityMethod(Enum):
    """Re-ID similarity calculation methods."""
    COSINE = "cosine"
    L2_DERIVED = "l2_derived"
    L2_EXPLICIT = "l2_explicit"
    INNER_PRODUCT = "inner_product"
    FAISS_IP = "faiss_ip"
    FAISS_L2 = "faiss_l2"


class DetectorSettings(BaseModel):
    """Person detector configuration settings."""
    
    detector_type: DetectorType = Field(
        default=DetectorType.FASTER_RCNN,
        description="Type of person detection model"
    )
    person_class_id: int = Field(
        default=1,
        ge=0,
        le=100,
        description="Class ID for person detection (COCO: 1)"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )
    use_amp: bool = Field(
        default=False,
        description="Enable automatic mixed precision"
    )
    nms_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Non-maximum suppression threshold"
    )
    
    def validate_detector_settings(self) -> None:
        """Validate detector-specific settings."""
        if self.confidence_threshold < 0.1:
            logger.warning(f"Low confidence threshold ({self.confidence_threshold}) may produce many false positives")
        
        if self.confidence_threshold > 0.9:
            logger.warning(f"High confidence threshold ({self.confidence_threshold}) may miss valid detections")


class TrackerSettings(BaseModel):
    """Multi-object tracker configuration settings."""
    
    tracker_type: TrackerType = Field(
        default=TrackerType.BOTSORT,
        description="Type of multi-object tracker"
    )
    max_age: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum frames to keep lost tracks"
    )
    n_init: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum detections to confirm track"
    )
    max_iou_distance: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum IoU distance for association"
    )
    max_cosine_distance: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum cosine distance for features"
    )
    half_precision: bool = Field(
        default=False,
        description="Enable half precision inference"
    )
    per_class: bool = Field(
        default=False,
        description="Enable per-class tracking"
    )
    
    def validate_tracker_settings(self) -> None:
        """Validate tracker-specific settings."""
        if self.max_age < self.n_init:
            logger.warning("max_age should be greater than n_init for stable tracking")


class ReIDSettings(BaseModel):
    """Re-identification model configuration settings."""
    
    reid_model_type: ReIDModelType = Field(
        default=ReIDModelType.CLIP,
        description="Type of re-identification model"
    )
    weights_filename: str = Field(
        default="clip_market1501.pt",
        description="ReID model weights filename"
    )
    half_precision: bool = Field(
        default=False,
        description="Enable half precision inference"
    )
    feature_dimension: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Feature vector dimension"
    )
    
    # Similarity settings
    similarity_method: SimilarityMethod = Field(
        default=SimilarityMethod.FAISS_L2,
        description="Re-ID similarity calculation method"
    )
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for re-identification"
    )
    l2_distance_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Explicit L2 distance threshold (if None, derived from similarity)"
    )
    
    # Gallery management
    gallery_ema_alpha: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Exponential moving average alpha for gallery updates"
    )
    refresh_interval_frames: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Frames between gallery refreshes"
    )
    lost_track_buffer_frames: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Frames to keep lost track features"
    )
    main_gallery_prune_interval_frames: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Frames between gallery pruning"
    )
    
    @validator('main_gallery_prune_interval_frames')
    def validate_pruning_interval(cls, v, values):
        """Ensure pruning interval is reasonable relative to buffer frames."""
        buffer_frames = values.get('lost_track_buffer_frames', 200)
        if v <= buffer_frames:
            raise ValueError("Pruning interval should be greater than lost track buffer frames")
        return v
    
    def validate_reid_settings(self) -> None:
        """Validate re-identification specific settings."""
        if self.similarity_threshold < 0.3:
            logger.warning(f"Low similarity threshold ({self.similarity_threshold}) may cause identity switches")
        
        if self.similarity_threshold > 0.9:
            logger.warning(f"High similarity threshold ({self.similarity_threshold}) may miss valid re-identifications")


class AIModelSettings(BaseSettings):
    """
    Consolidated AI model infrastructure configuration.
    
    Handles all AI model settings with proper validation and environment management.
    Follows Phase 6 configuration consolidation requirements.
    """
    
    # Model components
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    reid: ReIDSettings = Field(default_factory=ReIDSettings)
    
    # Model storage paths
    weights_dir: str = Field(
        default="./weights",
        description="Base directory for model weights"
    )
    model_cache_dir: str = Field(
        default="./model_cache",
        description="Directory for cached model files"
    )
    
    # Processing settings
    target_fps: int = Field(
        default=23,
        ge=1,
        le=60,
        description="Target processing frame rate"
    )
    frame_jpeg_quality: int = Field(
        default=90,
        ge=10,
        le=100,
        description="JPEG quality for processed frames"
    )
    
    def __init__(self, **kwargs):
        """Initialize AI model settings with validation."""
        super().__init__(**kwargs)
        self._validate_all_settings()
        logger.debug("AIModelSettings initialized successfully")
    
    @property
    def resolved_reid_weights_path(self) -> Path:
        """Get resolved path to ReID model weights."""
        weights_dir_path = Path(self.weights_dir)
        reid_weights_path = weights_dir_path / self.reid.weights_filename
        return reid_weights_path.resolve()
    
    @property
    def derived_l2_distance_threshold(self) -> float:
        """
        Calculate L2 distance threshold from cosine similarity threshold.
        Formula: d = sqrt(2 * (1 - s_cos)) for L2-normalized vectors.
        """
        import math
        return math.sqrt(max(0, 2 * (1 - self.reid.similarity_threshold)))
    
    def validate_model_weights_exist(self) -> Dict[str, bool]:
        """
        Validate that required model weight files exist.
        
        Returns:
            Dictionary mapping model type to existence status
        """
        validation_results = {}
        
        # Check ReID weights
        reid_weights_path = self.resolved_reid_weights_path
        validation_results['reid_weights'] = reid_weights_path.exists()
        
        if not validation_results['reid_weights']:
            logger.warning(f"ReID weights not found: {reid_weights_path}")
        
        # Create directories if they don't exist
        for directory in [self.weights_dir, self.model_cache_dir]:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            validation_results[f'{directory}_dir'] = dir_path.exists()
        
        return validation_results
    
    def validate_config(self) -> None:
        """Validate complete AI model configuration."""
        self._validate_all_settings()
        
        # Validate model weight files
        weight_validation = self.validate_model_weights_exist()
        if not all(weight_validation.values()):
            missing = [k for k, v in weight_validation.items() if not v]
            logger.warning(f"Missing AI model components: {missing}")
    
    def _validate_all_settings(self) -> None:
        """Validate all AI model settings."""
        try:
            self.detector.validate_detector_settings()
            self.tracker.validate_tracker_settings()
            self.reid.validate_reid_settings()
            
            # Cross-component validation
            if self.target_fps > 30 and self.detector.use_amp == False:
                logger.info("Consider enabling AMP for high FPS processing")
            
        except Exception as e:
            logger.error(f"AI model settings validation failed: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive AI model configuration summary.
        
        Returns:
            Dictionary containing configuration overview
        """
        return {
            'detector': {
                'type': self.detector.detector_type.value,
                'confidence_threshold': self.detector.confidence_threshold,
                'amp_enabled': self.detector.use_amp
            },
            'tracker': {
                'type': self.tracker.tracker_type.value,
                'max_age': self.tracker.max_age,
                'n_init': self.tracker.n_init
            },
            'reid': {
                'type': self.reid.reid_model_type.value,
                'similarity_method': self.reid.similarity_method.value,
                'similarity_threshold': self.reid.similarity_threshold,
                'weights_path': str(self.resolved_reid_weights_path)
            },
            'processing': {
                'target_fps': self.target_fps,
                'frame_quality': self.frame_jpeg_quality
            },
            'validation_timestamp': datetime.utcnow().isoformat()
        }
    
    class Config:
        env_prefix = "AI_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        use_enum_values = True


# Default AI model settings factory
def get_ai_model_settings() -> AIModelSettings:
    """
    Get AI model settings with environment-specific defaults.
    
    Returns:
        Configured AIModelSettings instance
    """
    return AIModelSettings()