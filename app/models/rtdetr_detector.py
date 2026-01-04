"""
Backward compatibility shim for RT-DETR detector.
The actual implementation has been migrated to YOLO (yolo_detector.py).
"""

# Import from new YOLO detector module for backward compatibility
from .yolo_detector import YOLODetector

# Backward compatibility alias
RTDETRDetector = YOLODetector

__all__ = ['RTDETRDetector', 'YOLODetector']
