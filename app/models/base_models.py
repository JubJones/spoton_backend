from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Optional
import numpy as np


class BoundingBox:
    """Represents a bounding box."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

class Detection:
    """Represents a detected object."""
    def __init__(self, bbox: BoundingBox, confidence: float, class_id: Any, class_name: Optional[str] = None):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def to_tracker_format(self) -> np.ndarray:
        """Converts detection to [x1, y1, x2, y2, conf, cls_id] format for trackers."""
        return np.array([
            self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2,
            self.confidence, self.class_id
        ], dtype=np.float32)




class AbstractDetector(ABC):
    """Abstract base class for object detectors (Strategy Pattern)."""

    @abstractmethod
    async def load_model(self):
        """
        Loads the detection model into memory.
        Should handle device placement (using self.device).
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
    async def load_model(self):
        """
        Loads and initializes the tracker. Should handle device placement.
        """
        pass

    @abstractmethod
    async def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Updates tracks with new detections for the current frame.

        Args:
            detections: A NumPy array of detections in [x1, y1, x2, y2, conf, cls_id] format.
            image: The current frame as a NumPy array (BGR).

        Returns:
            A NumPy array representing tracked objects, typically in a format like
            [x1, y1, x2, y2, track_id, conf, cls_id, global_id (optional), ...].
            The exact format depends on the underlying BoxMOT tracker implementation.
        """
        pass

    @abstractmethod
    async def reset(self):
        """Resets the tracker's state (e.g., for a new video sequence)."""
        pass


