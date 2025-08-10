"""
Visualization Domain Entities

Core entities for image processing and visual enhancement.
"""

from .overlay_config import OverlayConfig
from .cropped_image import CroppedImage
from .visual_frame import VisualFrame

__all__ = [
    "OverlayConfig",
    "CroppedImage", 
    "VisualFrame"
]