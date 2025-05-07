from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple, Optional
import numpy as np # Commonly used for image data and embeddings

# Define common data structures (can be Pydantic models for more rigor)

class BoundingBox:
    """Represents a bounding box."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

class Detection:
    """Represents a detected object."""
    def __init__(self, bbox: BoundingBox, confidence: float, class_id: Any, class_name: Optional[str] = None):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

class TrackedObject:
    """Represents a tracked object within a single camera view."""
    def __init__(self, temporary_track_id: Any, bbox: BoundingBox, confidence: Optional[float] = None,
                 class_id: Optional[Any] = None, feature_embedding: Optional[np.ndarray] = None,
                 state: Optional[str] = None, age: Optional[int] = None):
        self.temporary_track_id = temporary_track_id
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.feature_embedding = feature_embedding # Optional, if tracker also extracts features or it's added later
        self.state = state # e.g., 'active', 'lost'
        self.age = age # Number of frames tracked


class AbstractDetector(ABC):
    """Abstract base class for object detectors (Strategy Pattern)."""

    @abstractmethod
    async def load_model(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Loads the detection model into memory.
        This can be async if model loading involves I/O.
        """
        pass

    @abstractmethod
    async def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Performs object detection on an image.

        Args:
            image: A NumPy array representing the image (e.g., in BGR or RGB format).

        Returns:
            A list of Detection objects.
        """
        pass


class AbstractTracker(ABC):
    """Abstract base class for intra-camera trackers (Strategy Pattern)."""

    @abstractmethod
    async def load_model(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """Loads and initializes the tracker."""
        pass

    @abstractmethod
    async def update(self, detections: List[Detection], image: np.ndarray) -> List[TrackedObject]:
        """
        Updates tracks with new detections for the current frame.

        Args:
            detections: A list of Detection objects from the detector for the current frame.
            image: The current frame as a NumPy array.

        Returns:
            A list of TrackedObject instances representing currently active tracks.
        """
        pass

    @abstractmethod
    async def reset(self):
        """Resets the tracker's state (e.g., for a new video sequence)."""
        pass


class AbstractFeatureExtractor(ABC):
    """Abstract base class for appearance feature extractors (Strategy Pattern)."""

    @abstractmethod
    async def load_model(self, model_name_or_path: Optional[str] = None, config: Optional[Dict] = None):
        """Loads the feature extraction model."""
        pass

    @abstractmethod
    async def extract_features(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Extracts an embedding vector from an image crop of a person.

        Args:
            image_crop: A NumPy array representing the cropped image of a person.

        Returns:
            A 1D NumPy array (embedding vector).
        """
        pass
