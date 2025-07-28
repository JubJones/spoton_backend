"""
Abstract base detector interface.

Defines the contract for all detection models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class AbstractDetector(ABC):
    """Abstract base class for object detectors."""
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the detection model."""
        pass
    
    @abstractmethod
    async def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: Dict with x, y, width, height, normalized
            - confidence: Float confidence score
            - class_id: Integer class identifier
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    @abstractmethod
    async def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection."""
        pass
    
    @abstractmethod
    async def postprocess_detections(
        self, 
        raw_output: Any, 
        image_shape: tuple
    ) -> List[Dict[str, Any]]:
        """Postprocess raw model output to detections."""
        pass
    
    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """Get list of supported detection classes."""
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for detections."""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        pass
    
    @abstractmethod
    async def batch_detect(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in multiple images simultaneously.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of detection lists, one per input image
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        pass
    
    @abstractmethod
    async def warm_up(self) -> None:
        """Warm up the model with dummy inference."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up model resources."""
        pass

class DetectionResult:
    """Standard detection result format."""
    
    def __init__(
        self,
        bbox: Dict[str, float],
        confidence: float,
        class_id: int,
        class_name: Optional[str] = None
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }

class DetectorFactory:
    """Factory for creating detector instances."""
    
    _detectors = {}
    
    @classmethod
    def register_detector(cls, name: str, detector_class: type):
        """Register a detector class."""
        cls._detectors[name] = detector_class
    
    @classmethod
    def create_detector(cls, name: str, **kwargs) -> AbstractDetector:
        """Create a detector instance."""
        if name not in cls._detectors:
            raise ValueError(f"Unknown detector type: {name}")
        
        return cls._detectors[name](**kwargs)
    
    @classmethod
    def get_available_detectors(cls) -> List[str]:
        """Get list of available detector types."""
        return list(cls._detectors.keys())