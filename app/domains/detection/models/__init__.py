"""
Detection models module.
Contains all object detection model implementations.
"""

from .base_detector import (
    AbstractDetector,
    DetectionResult,
    DetectorFactory
)

# Import specific detector implementations to register them
from .faster_rcnn_detector import FasterRCNNDetector
from .yolo_detector import YOLODetector
from .rtdetr_detector import RTDETRDetector

__all__ = [
    'AbstractDetector',
    'DetectionResult',
    'DetectorFactory',
    'FasterRCNNDetector',
    'YOLODetector',
    'RTDETRDetector'
]

# Available detector types
AVAILABLE_DETECTORS = DetectorFactory.get_available_detectors()