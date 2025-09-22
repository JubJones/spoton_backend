"""
ReID models module.
Contains all person re-identification model implementations.
"""

from .base_reid_model import (
    AbstractReIDModel,
    ReIDModelFactory,
    ReIDResult
)

# Import specific model implementations to register them
from .clip_reid_model import CLIPReIDModel

__all__ = [
    'AbstractReIDModel',
    'ReIDModelFactory',
    'ReIDResult',
    'CLIPReIDModel'
]

# Available ReID model types
AVAILABLE_MODELS = ReIDModelFactory.get_available_models()