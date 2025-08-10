# AI Models infrastructure

# Legacy factory (maintained for backward compatibility)
from .factory import (
    AIModelFactory,
    get_ai_model_factory,
    configure_ai_models,
    ModelCreationError
)

# Clean factory (new clean architecture implementation)
from .clean_factory import (
    CleanAIModelFactory,
    get_clean_ai_model_factory,
    configure_clean_ai_models,
    ModelType,
    ModelConfig,
    ModelRegistry
)

__all__ = [
    # Legacy factory
    "AIModelFactory",
    "get_ai_model_factory", 
    "configure_ai_models",
    # Clean factory (recommended)
    "CleanAIModelFactory",
    "get_clean_ai_model_factory",
    "configure_clean_ai_models",
    "ModelType",
    "ModelConfig", 
    "ModelRegistry",
    # Common
    "ModelCreationError"
]