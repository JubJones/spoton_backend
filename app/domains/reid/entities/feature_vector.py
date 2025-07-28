"""
Feature vector entity for ReID representation.

Contains feature vectors used for person re-identification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

@dataclass(frozen=True)
class FeatureVector:
    """Feature vector value object for ReID."""
    
    vector: List[float]
    extraction_timestamp: datetime
    model_version: str
    confidence: float = 1.0
    
    # Metadata
    detection_id: Optional[str] = None
    camera_id: Optional[str] = None
    frame_index: Optional[int] = None
    
    def __post_init__(self):
        """Validate feature vector."""
        if not self.vector:
            raise ValueError("Feature vector cannot be empty")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if not self.model_version:
            raise ValueError("Model version cannot be empty")
        
        # Validate vector values
        for i, value in enumerate(self.vector):
            if not isinstance(value, (int, float)) or np.isnan(value):
                raise ValueError(f"Invalid value at index {i}: {value}")
    
    @property
    def dimension(self) -> int:
        """Get feature vector dimension."""
        return len(self.vector)
    
    @property
    def norm(self) -> float:
        """Get L2 norm of feature vector."""
        return float(np.linalg.norm(self.vector))
    
    @property
    def is_normalized(self) -> bool:
        """Check if feature vector is L2 normalized."""
        return abs(self.norm - 1.0) < 1e-6
    
    def normalize(self) -> 'FeatureVector':
        """Normalize feature vector to unit length."""
        if self.is_normalized:
            return self
        
        norm = self.norm
        if norm == 0:
            return self
        
        normalized_vector = [v / norm for v in self.vector]
        
        return FeatureVector(
            vector=normalized_vector,
            extraction_timestamp=self.extraction_timestamp,
            model_version=self.model_version,
            confidence=self.confidence,
            detection_id=self.detection_id,
            camera_id=self.camera_id,
            frame_index=self.frame_index
        )
    
    def cosine_similarity(self, other: 'FeatureVector') -> float:
        """Calculate cosine similarity with another feature vector."""
        if self.dimension != other.dimension:
            raise ValueError("Feature vectors must have same dimension")
        
        # Normalize both vectors
        norm_self = self.normalize()
        norm_other = other.normalize()
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(norm_self.vector, norm_other.vector))
        
        # Clamp to [-1, 1] to handle numerical errors
        return max(-1.0, min(1.0, dot_product))
    
    def euclidean_distance(self, other: 'FeatureVector') -> float:
        """Calculate Euclidean distance with another feature vector."""
        if self.dimension != other.dimension:
            raise ValueError("Feature vectors must have same dimension")
        
        distance_squared = sum((a - b) ** 2 for a, b in zip(self.vector, other.vector))
        return float(np.sqrt(distance_squared))
    
    def manhattan_distance(self, other: 'FeatureVector') -> float:
        """Calculate Manhattan distance with another feature vector."""
        if self.dimension != other.dimension:
            raise ValueError("Feature vectors must have same dimension")
        
        return sum(abs(a - b) for a, b in zip(self.vector, other.vector))
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.vector, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vector": self.vector,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "model_version": self.model_version,
            "confidence": self.confidence,
            "detection_id": self.detection_id,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "dimension": self.dimension,
            "norm": self.norm,
            "is_normalized": self.is_normalized
        }
    
    @classmethod
    def from_numpy(
        cls,
        vector: np.ndarray,
        extraction_timestamp: datetime,
        model_version: str,
        confidence: float = 1.0,
        **kwargs
    ) -> 'FeatureVector':
        """Create feature vector from numpy array."""
        return cls(
            vector=vector.tolist(),
            extraction_timestamp=extraction_timestamp,
            model_version=model_version,
            confidence=confidence,
            **kwargs
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVector':
        """Create feature vector from dictionary."""
        return cls(
            vector=data["vector"],
            extraction_timestamp=datetime.fromisoformat(data["extraction_timestamp"]),
            model_version=data["model_version"],
            confidence=data.get("confidence", 1.0),
            detection_id=data.get("detection_id"),
            camera_id=data.get("camera_id"),
            frame_index=data.get("frame_index")
        )

@dataclass
class FeatureVectorBatch:
    """Batch of feature vectors for efficient processing."""
    
    vectors: List[FeatureVector]
    batch_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate feature vector batch."""
        if not self.vectors:
            raise ValueError("Feature vector batch cannot be empty")
        
        # Check dimension consistency
        if len(self.vectors) > 1:
            first_dim = self.vectors[0].dimension
            for i, vector in enumerate(self.vectors[1:], 1):
                if vector.dimension != first_dim:
                    raise ValueError(f"Dimension mismatch at index {i}: {vector.dimension} != {first_dim}")
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.vectors)
    
    @property
    def dimension(self) -> int:
        """Get feature vector dimension."""
        return self.vectors[0].dimension if self.vectors else 0
    
    def to_numpy_matrix(self) -> np.ndarray:
        """Convert to numpy matrix (batch_size x dimension)."""
        if not self.vectors:
            return np.array([], dtype=np.float32)
        
        matrix = np.array([v.vector for v in self.vectors], dtype=np.float32)
        return matrix
    
    def normalize_all(self) -> 'FeatureVectorBatch':
        """Normalize all feature vectors in batch."""
        normalized_vectors = [v.normalize() for v in self.vectors]
        return FeatureVectorBatch(
            vectors=normalized_vectors,
            batch_timestamp=self.batch_timestamp
        )
    
    def filter_by_confidence(self, min_confidence: float) -> 'FeatureVectorBatch':
        """Filter vectors by minimum confidence."""
        filtered_vectors = [v for v in self.vectors if v.confidence >= min_confidence]
        return FeatureVectorBatch(
            vectors=filtered_vectors,
            batch_timestamp=self.batch_timestamp
        )
    
    def get_average_vector(self) -> Optional[FeatureVector]:
        """Get average feature vector."""
        if not self.vectors:
            return None
        
        if len(self.vectors) == 1:
            return self.vectors[0]
        
        # Calculate element-wise average
        avg_vector = [0.0] * self.dimension
        total_confidence = 0.0
        
        for vector in self.vectors:
            for i, value in enumerate(vector.vector):
                avg_vector[i] += value * vector.confidence
            total_confidence += vector.confidence
        
        if total_confidence > 0:
            avg_vector = [val / total_confidence for val in avg_vector]
        
        return FeatureVector(
            vector=avg_vector,
            extraction_timestamp=self.batch_timestamp,
            model_version=self.vectors[0].model_version,
            confidence=total_confidence / len(self.vectors)
        )
    
    def pairwise_similarities(self) -> List[List[float]]:
        """Calculate pairwise cosine similarities."""
        n = len(self.vectors)
        similarities = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarities[i][j] = 1.0
                else:
                    sim = self.vectors[i].cosine_similarity(self.vectors[j])
                    similarities[i][j] = sim
                    similarities[j][i] = sim
        
        return similarities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vectors": [v.to_dict() for v in self.vectors],
            "batch_timestamp": self.batch_timestamp.isoformat(),
            "size": self.size,
            "dimension": self.dimension
        }