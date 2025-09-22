"""
Abstract base ReID model interface.
Defines the contract for all person re-identification models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.domains.reid.entities.feature_vector import FeatureVector


class AbstractReIDModel(ABC):
    """Abstract base class for person re-identification models."""
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the ReID model."""
        pass
    
    @abstractmethod
    async def extract_features(self, image: np.ndarray) -> FeatureVector:
        """
        Extract features from a single person image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Feature vector
        """
        pass
    
    @abstractmethod
    async def extract_features_batch(self, images: List[np.ndarray]) -> List[FeatureVector]:
        """
        Extract features from multiple person images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of feature vectors
        """
        pass
    
    @abstractmethod
    def add_to_database(self, feature_vector: FeatureVector, person_id: str) -> None:
        """
        Add feature vector to the database.
        
        Args:
            feature_vector: Feature vector to add
            person_id: Person identifier
        """
        pass
    
    @abstractmethod
    def find_similar_persons(
        self, 
        query_features: FeatureVector,
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find similar persons in the database.
        
        Args:
            query_features: Query feature vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (person_id, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def clear_database(self) -> None:
        """Clear the feature database."""
        pass
    
    @abstractmethod
    def get_database_size(self) -> int:
        """Get the size of the feature database."""
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
    async def preprocess_image(self, image: np.ndarray) -> Any:
        """Preprocess image for feature extraction."""
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
    
    @abstractmethod
    def calculate_similarity(
        self, 
        features1: FeatureVector, 
        features2: FeatureVector
    ) -> float:
        """
        Calculate similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def update_feature_in_database(
        self, 
        person_id: str, 
        new_features: FeatureVector
    ) -> bool:
        """
        Update feature vector for a person in the database.
        
        Args:
            person_id: Person identifier
            new_features: New feature vector
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def remove_from_database(self, person_id: str) -> bool:
        """
        Remove person from the database.
        
        Args:
            person_id: Person identifier
            
        Returns:
            True if removal successful, False otherwise
        """
        pass


class ReIDModelFactory:
    """Factory for creating ReID model instances."""
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a ReID model class."""
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> AbstractReIDModel:
        """Create a ReID model instance."""
        if name not in cls._models:
            raise ValueError(f"Unknown ReID model type: {name}")
        
        return cls._models[name](**kwargs)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available ReID model types."""
        return list(cls._models.keys())


class ReIDResult:
    """Standard ReID result format."""
    
    def __init__(
        self,
        person_id: str,
        similarity_score: float,
        features: FeatureVector,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.person_id = person_id
        self.similarity_score = similarity_score
        self.features = features
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "person_id": self.person_id,
            "similarity_score": self.similarity_score,
            "features": self.features.to_dict(),
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        return f"ReIDResult(person_id={self.person_id}, similarity={self.similarity_score:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()