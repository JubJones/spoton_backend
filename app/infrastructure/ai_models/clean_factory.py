"""
Clean AI Model Factory implementation.

Implements Factory Pattern with clean architecture principles,
proper integration with existing model abstractions, and caching support.
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import torch
import numpy as np

from app.core.config import settings
from app.models.base_models import (
    AbstractDetector, AbstractTracker, AbstractFeatureExtractor,
    Detection, TrackedObject, BoundingBox
)

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported AI model types."""
    DETECTOR = "detector"
    TRACKER = "tracker"
    REID = "reid"


class ModelCreationError(Exception):
    """Exception raised when model creation fails."""
    pass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for AI model creation."""
    model_type: str
    device: torch.device
    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    additional_params: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """Registry for AI model implementations."""
    
    def __init__(self):
        self._detectors: Dict[str, Type[AbstractDetector]] = {}
        self._trackers: Dict[str, Type[AbstractTracker]] = {}
        self._reid_models: Dict[str, Type[AbstractFeatureExtractor]] = {}
        
    def register_detector(self, name: str, detector_class: Type[AbstractDetector]) -> None:
        """Register a detector implementation."""
        self._detectors[name] = detector_class
        logger.debug(f"Registered detector: {name} -> {detector_class.__name__}")
    
    def register_tracker(self, name: str, tracker_class: Type[AbstractTracker]) -> None:
        """Register a tracker implementation."""
        self._trackers[name] = tracker_class
        logger.debug(f"Registered tracker: {name} -> {tracker_class.__name__}")
    
    def register_reid_model(self, name: str, reid_class: Type[AbstractFeatureExtractor]) -> None:
        """Register a ReID model implementation."""
        self._reid_models[name] = reid_class
        logger.debug(f"Registered ReID model: {name} -> {reid_class.__name__}")
    
    def get_detector(self, name: str) -> Type[AbstractDetector]:
        """Get detector class by name."""
        if name not in self._detectors:
            available = ", ".join(self._detectors.keys())
            raise ModelCreationError(f"Unknown detector: {name}. Available: {available}")
        return self._detectors[name]
    
    def get_tracker(self, name: str) -> Type[AbstractTracker]:
        """Get tracker class by name."""
        if name not in self._trackers:
            available = ", ".join(self._trackers.keys())
            raise ModelCreationError(f"Unknown tracker: {name}. Available: {available}")
        return self._trackers[name]
    
    def get_reid_model(self, name: str) -> Type[AbstractFeatureExtractor]:
        """Get ReID model class by name."""
        if name not in self._reid_models:
            available = ", ".join(self._reid_models.keys())
            raise ModelCreationError(f"Unknown ReID model: {name}. Available: {available}")
        return self._reid_models[name]
    
    def list_available(self, model_type: ModelType) -> List[str]:
        """List available models for given type."""
        if model_type == ModelType.DETECTOR:
            return list(self._detectors.keys())
        elif model_type == ModelType.TRACKER:
            return list(self._trackers.keys())
        elif model_type == ModelType.REID:
            return list(self._reid_models.keys())
        else:
            return []


class CleanAIModelFactory:
    """
    Clean AI Model Factory implementation.
    
    Implements Factory Pattern with proper separation of concerns,
    clean interfaces, and integration with existing model abstractions.
    Follows infrastructure layer patterns for AI model creation.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize AI model factory with clean architecture patterns.
        
        Args:
            device: Compute device for model inference
        """
        self.device = device or self._detect_optimal_device()
        self._registry = ModelRegistry()
        self._model_cache: Dict[str, Any] = {}
        self._load_count: Dict[str, int] = {}
        
        logger.info(f"CleanAIModelFactory initialized with device: {self.device}")
        self._register_built_in_models()
    
    def _detect_optimal_device(self) -> torch.device:
        """Detect optimal compute device for AI inference."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for AI inference")
        return device
    
    def _register_built_in_models(self) -> None:
        """Register built-in model implementations."""
        try:
            # Register existing detector implementations
            from app.models.detectors import FasterRCNNDetector
            self._registry.register_detector("faster_rcnn", FasterRCNNDetector)
            
            # Register existing tracker implementations  
            from app.models.trackers import BoxMOTTracker
            self._registry.register_tracker("boxmot", BoxMOTTracker)
            
            # Register existing ReID implementations
            from app.models.feature_extractors import CLIPFeatureExtractor
            self._registry.register_reid_model("clip", CLIPFeatureExtractor)
            
            logger.info("Built-in model implementations registered")
            
        except ImportError as e:
            logger.warning(f"Some built-in models not available: {e}")
    
    def register_detector(self, detector_type: str, detector_class: Type[AbstractDetector]) -> None:
        """Register a detector implementation with validation."""
        if not issubclass(detector_class, AbstractDetector):
            raise ModelCreationError(f"Detector must implement AbstractDetector interface")
        self._registry.register_detector(detector_type, detector_class)
    
    def register_tracker(self, tracker_type: str, tracker_class: Type[AbstractTracker]) -> None:
        """Register a tracker implementation with validation."""
        if not issubclass(tracker_class, AbstractTracker):
            raise ModelCreationError(f"Tracker must implement AbstractTracker interface")
        self._registry.register_tracker(tracker_type, tracker_class)
    
    def register_reid_model(self, reid_type: str, reid_class: Type[AbstractFeatureExtractor]) -> None:
        """Register a ReID model implementation with validation."""
        if not issubclass(reid_class, AbstractFeatureExtractor):
            raise ModelCreationError(f"ReID model must implement AbstractFeatureExtractor interface")
        self._registry.register_reid_model(reid_type, reid_class)
    
    def create_detector(
        self, 
        detector_type: Optional[str] = None,
        cache_enabled: bool = True,
        **kwargs
    ) -> AbstractDetector:
        """
        Create detector instance with clean architecture patterns.
        
        Args:
            detector_type: Type of detector (defaults to settings.DETECTOR_TYPE)
            cache_enabled: Whether to cache model instances
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured detector instance
            
        Raises:
            ModelCreationError: If detector creation fails
        """
        detector_type = detector_type or settings.DETECTOR_TYPE
        cache_key = f"detector_{detector_type}_{hash(str(kwargs))}"
        
        # Return cached instance if available
        if cache_enabled and cache_key in self._model_cache:
            self._load_count[cache_key] = self._load_count.get(cache_key, 0) + 1
            logger.debug(f"Returning cached detector: {detector_type}")
            return self._model_cache[cache_key]
        
        try:
            # Create model configuration
            config = ModelConfig(
                model_type=detector_type,
                device=self.device,
                confidence_threshold=kwargs.get('confidence_threshold', 0.5),
                additional_params=kwargs
            )
            
            # Get detector class and create instance
            detector_class = self._registry.get_detector(detector_type)
            detector = self._create_detector_instance(detector_class, config)
            
            # Cache if enabled
            if cache_enabled:
                self._model_cache[cache_key] = detector
                self._load_count[cache_key] = 1
            
            logger.info(f"Created detector: {detector_type} on device: {self.device}")
            return detector
            
        except Exception as e:
            logger.error(f"Failed to create detector {detector_type}: {e}")
            raise ModelCreationError(f"Detector creation failed: {e}")
    
    def _create_detector_instance(self, detector_class: Type[AbstractDetector], config: ModelConfig) -> AbstractDetector:
        """Create detector instance with proper configuration."""
        # Initialize with device
        instance = detector_class(device=config.device)
        
        # Apply additional configuration if provided
        if config.additional_params:
            for param, value in config.additional_params.items():
                if hasattr(instance, param):
                    setattr(instance, param, value)
        
        return instance
    
    def create_tracker(
        self, 
        tracker_type: Optional[str] = None,
        cache_enabled: bool = True,
        **kwargs
    ) -> AbstractTracker:
        """
        Create tracker instance with clean architecture patterns.
        
        Args:
            tracker_type: Type of tracker (defaults to settings.TRACKER_TYPE)
            cache_enabled: Whether to cache model instances
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured tracker instance
            
        Raises:
            ModelCreationError: If tracker creation fails
        """
        tracker_type = tracker_type or settings.TRACKER_TYPE
        cache_key = f"tracker_{tracker_type}_{hash(str(kwargs))}"
        
        # Return cached instance if available
        if cache_enabled and cache_key in self._model_cache:
            self._load_count[cache_key] = self._load_count.get(cache_key, 0) + 1
            logger.debug(f"Returning cached tracker: {tracker_type}")
            return self._model_cache[cache_key]
        
        try:
            # Create model configuration
            config = ModelConfig(
                model_type=tracker_type,
                device=self.device,
                additional_params=kwargs
            )
            
            # Get tracker class and create instance
            tracker_class = self._registry.get_tracker(tracker_type)
            tracker = self._create_tracker_instance(tracker_class, config)
            
            # Cache if enabled
            if cache_enabled:
                self._model_cache[cache_key] = tracker
                self._load_count[cache_key] = 1
            
            logger.info(f"Created tracker: {tracker_type} on device: {self.device}")
            return tracker
            
        except Exception as e:
            logger.error(f"Failed to create tracker {tracker_type}: {e}")
            raise ModelCreationError(f"Tracker creation failed: {e}")
    
    def _create_tracker_instance(self, tracker_class: Type[AbstractTracker], config: ModelConfig) -> AbstractTracker:
        """Create tracker instance with proper configuration."""
        # Initialize with device
        instance = tracker_class(device=config.device)
        
        # Apply additional configuration if provided
        if config.additional_params:
            for param, value in config.additional_params.items():
                if hasattr(instance, param):
                    setattr(instance, param, value)
        
        return instance
    
    def create_reid_model(
        self, 
        reid_type: Optional[str] = None,
        cache_enabled: bool = True,
        **kwargs
    ) -> AbstractFeatureExtractor:
        """
        Create ReID model instance with clean architecture patterns.
        
        Args:
            reid_type: Type of ReID model (defaults to settings.REID_MODEL_TYPE)
            cache_enabled: Whether to cache model instances
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured ReID model instance
            
        Raises:
            ModelCreationError: If ReID model creation fails
        """
        reid_type = reid_type or settings.REID_MODEL_TYPE
        cache_key = f"reid_{reid_type}_{hash(str(kwargs))}"
        
        # Return cached instance if available
        if cache_enabled and cache_key in self._model_cache:
            self._load_count[cache_key] = self._load_count.get(cache_key, 0) + 1
            logger.debug(f"Returning cached ReID model: {reid_type}")
            return self._model_cache[cache_key]
        
        try:
            # Create model configuration
            config = ModelConfig(
                model_type=reid_type,
                device=self.device,
                additional_params=kwargs
            )
            
            # Get ReID class and create instance
            reid_class = self._registry.get_reid_model(reid_type)
            reid_model = self._create_reid_instance(reid_class, config)
            
            # Cache if enabled
            if cache_enabled:
                self._model_cache[cache_key] = reid_model
                self._load_count[cache_key] = 1
            
            logger.info(f"Created ReID model: {reid_type} on device: {self.device}")
            return reid_model
            
        except Exception as e:
            logger.error(f"Failed to create ReID model {reid_type}: {e}")
            raise ModelCreationError(f"ReID model creation failed: {e}")
    
    def _create_reid_instance(self, reid_class: Type[AbstractFeatureExtractor], config: ModelConfig) -> AbstractFeatureExtractor:
        """Create ReID model instance with proper configuration."""
        # Initialize with device
        instance = reid_class(device=config.device)
        
        # Apply additional configuration if provided
        if config.additional_params:
            for param, value in config.additional_params.items():
                if hasattr(instance, param):
                    setattr(instance, param, value)
        
        return instance
    
    def get_available_models(self, model_type: ModelType) -> List[str]:
        """Get list of available models for given type."""
        return self._registry.list_available(model_type)
    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detector types."""
        return self._registry.list_available(ModelType.DETECTOR)
    
    def get_available_trackers(self) -> List[str]:
        """Get list of available tracker types."""
        return self._registry.list_available(ModelType.TRACKER)
    
    def get_available_reid_models(self) -> List[str]:
        """Get list of available ReID model types."""
        return self._registry.list_available(ModelType.REID)
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate that required models are available for current configuration.
        
        Returns:
            Dictionary indicating availability of configured models
        """
        try:
            detector_available = settings.DETECTOR_TYPE in self.get_available_detectors()
            tracker_available = settings.TRACKER_TYPE in self.get_available_trackers()
            reid_available = settings.REID_MODEL_TYPE in self.get_available_reid_models()
            
            return {
                'detector_available': detector_available,
                'tracker_available': tracker_available,
                'reid_available': reid_available,
                'device_available': torch.cuda.is_available() if self.device.type == 'cuda' else True,
                'factory_ready': detector_available and tracker_available and reid_available
            }
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return {
                'detector_available': False,
                'tracker_available': False,
                'reid_available': False,
                'device_available': False,
                'factory_ready': False
            }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        return {
            'cached_models': len(self._model_cache),
            'total_loads': sum(self._load_count.values()),
            'cache_hits': sum(count - 1 for count in self._load_count.values()),
            'cache_keys': list(self._model_cache.keys())
        }
    
    def clear_cache(self, model_type: Optional[str] = None) -> int:
        """Clear model cache."""
        if model_type:
            # Clear specific model type
            keys_to_remove = [key for key in self._model_cache.keys() if key.startswith(model_type)]
            for key in keys_to_remove:
                del self._model_cache[key]
                del self._load_count[key]
            return len(keys_to_remove)
        else:
            # Clear all cache
            count = len(self._model_cache)
            self._model_cache.clear()
            self._load_count.clear()
            return count


# Global factory instance (will be configured during startup)
_clean_ai_model_factory: Optional[CleanAIModelFactory] = None


def get_clean_ai_model_factory() -> CleanAIModelFactory:
    """
    Get the global clean AI model factory instance.
    
    Returns:
        Configured AI model factory
        
    Raises:
        RuntimeError: If factory not initialized
    """
    if _clean_ai_model_factory is None:
        raise RuntimeError("Clean AI model factory not initialized. Call configure_clean_ai_models() during startup.")
    return _clean_ai_model_factory


def configure_clean_ai_models(device: Optional[torch.device] = None) -> CleanAIModelFactory:
    """
    Configure and initialize the global clean AI model factory.
    
    This should be called during application startup to register
    all available model implementations and validate configuration.
    
    Args:
        device: Compute device for model inference
        
    Returns:
        Configured AI model factory
        
    Raises:
        ModelCreationError: If factory configuration fails
    """
    global _clean_ai_model_factory
    
    try:
        _clean_ai_model_factory = CleanAIModelFactory(device)
        
        # Validate configuration
        config_status = _clean_ai_model_factory.validate_configuration()
        if not config_status['factory_ready']:
            missing = [k for k, v in config_status.items() if not v and k != 'factory_ready']
            logger.error(f"Clean AI factory configuration incomplete: {missing}")
            raise ModelCreationError(f"Missing required models: {missing}")
        
        logger.info("Clean AI model factory configured and validated successfully")
        logger.info(f"Available models - Detectors: {_clean_ai_model_factory.get_available_detectors()}")
        logger.info(f"Available models - Trackers: {_clean_ai_model_factory.get_available_trackers()}")
        logger.info(f"Available models - ReID: {_clean_ai_model_factory.get_available_reid_models()}")
        
        return _clean_ai_model_factory
        
    except Exception as e:
        logger.error(f"Failed to configure clean AI model factory: {e}")
        raise ModelCreationError(f"Factory configuration failed: {e}")