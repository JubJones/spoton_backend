"""
AI Model Factory for creating and managing AI model instances.

Provides a centralized way to create detection, tracking, and re-identification
models while maintaining the Strategy pattern for interchangeable implementations.

REFACTORED: This module has been modernized with clean architecture patterns.
See clean_factory.py for the new implementation that will replace this module.
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional
import logging
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelCreationError(Exception):
    """Exception raised when model creation fails."""
    pass


class BaseAIModel(ABC):
    """Base interface for all AI models."""
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the device where model is loaded."""
        pass


class BaseDetector(BaseAIModel):
    """Base interface for person detection models."""
    
    @abstractmethod
    def detect(self, image: torch.Tensor) -> Any:
        """Detect persons in image."""
        pass


class BaseTracker(BaseAIModel):
    """Base interface for person tracking models."""
    
    @abstractmethod
    def update(self, detections: Any, image: torch.Tensor) -> Any:
        """Update tracker with new detections."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass


class BaseReIDModel(BaseAIModel):
    """Base interface for person re-identification models."""
    
    @abstractmethod
    def extract_features(self, image: torch.Tensor, bbox: Any) -> torch.Tensor:
        """Extract features for person re-identification."""
        pass


class AIModelFactory:
    """
    Factory for creating AI model instances.
    
    Uses the Factory pattern to create different types of AI models
    based on configuration. Supports detector, tracker, and ReID models.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize AI model factory.
        
        Args:
            device: Compute device for model inference
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._detector_registry: Dict[str, Type[BaseDetector]] = {}
        self._tracker_registry: Dict[str, Type[BaseTracker]] = {}
        self._reid_registry: Dict[str, Type[BaseReIDModel]] = {}
        
        logger.info(f"AIModelFactory initialized with device: {self.device}")
    
    def register_detector(self, detector_type: str, detector_class: Type[BaseDetector]) -> None:
        """Register a detector implementation."""
        self._detector_registry[detector_type] = detector_class
        logger.debug(f"Registered detector: {detector_type} -> {detector_class.__name__}")
    
    def register_tracker(self, tracker_type: str, tracker_class: Type[BaseTracker]) -> None:
        """Register a tracker implementation."""
        self._tracker_registry[tracker_type] = tracker_class
        logger.debug(f"Registered tracker: {tracker_type} -> {tracker_class.__name__}")
    
    def register_reid_model(self, reid_type: str, reid_class: Type[BaseReIDModel]) -> None:
        """Register a ReID model implementation."""
        self._reid_registry[reid_type] = reid_class
        logger.debug(f"Registered ReID model: {reid_type} -> {reid_class.__name__}")
    
    def create_detector(
        self, 
        detector_type: Optional[str] = None, 
        **kwargs
    ) -> BaseDetector:
        """
        Create detector instance.
        
        Args:
            detector_type: Type of detector (defaults to settings.DETECTOR_TYPE)
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured detector instance
            
        Raises:
            ModelCreationError: If detector creation fails
        """
        detector_type = detector_type or settings.DETECTOR_TYPE
        
        if detector_type not in self._detector_registry:
            available = ", ".join(self._detector_registry.keys())
            raise ModelCreationError(
                f"Unknown detector type: {detector_type}. Available: {available}"
            )
        
        try:
            detector_class = self._detector_registry[detector_type]
            detector = detector_class(device=self.device, **kwargs)
            
            logger.info(f"Created detector: {detector_type}")
            return detector
            
        except Exception as e:
            logger.error(f"Failed to create detector {detector_type}: {e}")
            raise ModelCreationError(f"Detector creation failed: {e}")
    
    def create_tracker(
        self, 
        tracker_type: Optional[str] = None, 
        **kwargs
    ) -> BaseTracker:
        """
        Create tracker instance.
        
        Args:
            tracker_type: Type of tracker (defaults to settings.TRACKER_TYPE)
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured tracker instance
            
        Raises:
            ModelCreationError: If tracker creation fails
        """
        tracker_type = tracker_type or settings.TRACKER_TYPE
        
        if tracker_type not in self._tracker_registry:
            available = ", ".join(self._tracker_registry.keys())
            raise ModelCreationError(
                f"Unknown tracker type: {tracker_type}. Available: {available}"
            )
        
        try:
            tracker_class = self._tracker_registry[tracker_type]
            tracker = tracker_class(device=self.device, **kwargs)
            
            logger.info(f"Created tracker: {tracker_type}")
            return tracker
            
        except Exception as e:
            logger.error(f"Failed to create tracker {tracker_type}: {e}")
            raise ModelCreationError(f"Tracker creation failed: {e}")
    
    def create_reid_model(
        self, 
        reid_type: Optional[str] = None, 
        **kwargs
    ) -> BaseReIDModel:
        """
        Create ReID model instance.
        
        Args:
            reid_type: Type of ReID model (defaults to settings.REID_MODEL_TYPE)
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured ReID model instance
            
        Raises:
            ModelCreationError: If ReID model creation fails
        """
        reid_type = reid_type or settings.REID_MODEL_TYPE
        
        if reid_type not in self._reid_registry:
            available = ", ".join(self._reid_registry.keys())
            raise ModelCreationError(
                f"Unknown ReID model type: {reid_type}. Available: {available}"
            )
        
        try:
            reid_class = self._reid_registry[reid_type]
            reid_model = reid_class(device=self.device, **kwargs)
            
            logger.info(f"Created ReID model: {reid_type}")
            return reid_model
            
        except Exception as e:
            logger.error(f"Failed to create ReID model {reid_type}: {e}")
            raise ModelCreationError(f"ReID model creation failed: {e}")
    
    def get_available_detectors(self) -> list[str]:
        """Get list of available detector types."""
        return list(self._detector_registry.keys())
    
    def get_available_trackers(self) -> list[str]:
        """Get list of available tracker types."""
        return list(self._tracker_registry.keys())
    
    def get_available_reid_models(self) -> list[str]:
        """Get list of available ReID model types."""
        return list(self._reid_registry.keys())
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate that required models are available for current configuration.
        
        Returns:
            Dictionary indicating availability of configured models
        """
        return {
            'detector_available': settings.DETECTOR_TYPE in self._detector_registry,
            'tracker_available': settings.TRACKER_TYPE in self._tracker_registry,
            'reid_available': settings.REID_MODEL_TYPE in self._reid_registry,
        }


# Global factory instance (will be configured during startup)
_ai_model_factory: Optional[AIModelFactory] = None


def get_ai_model_factory() -> AIModelFactory:
    """
    Get the global AI model factory instance.
    
    Returns:
        Configured AI model factory
        
    Raises:
        RuntimeError: If factory not initialized
    """
    if _ai_model_factory is None:
        raise RuntimeError("AI model factory not initialized. Call configure_ai_models() during startup.")
    return _ai_model_factory


def configure_ai_models(device: Optional[torch.device] = None) -> AIModelFactory:
    """
    Configure and initialize the global AI model factory.
    
    This should be called during application startup to register
    all available model implementations.
    
    Args:
        device: Compute device for model inference
        
    Returns:
        Configured AI model factory
    """
    global _ai_model_factory
    
    _ai_model_factory = AIModelFactory(device)
    
    # Register model implementations here
    # This will be expanded as we migrate existing models
    
    # Import and use the new clean factory as the primary implementation
    try:
        from .clean_factory import configure_clean_ai_models
        clean_factory = configure_clean_ai_models(device)
        logger.info("Clean AI model factory configured and ready (new implementation)")
        return clean_factory
    except Exception as e:
        logger.warning(f"Failed to load clean factory, using legacy: {e}")
    
    logger.info("AI model factory configured and ready (legacy implementation)")
    return _ai_model_factory