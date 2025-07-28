"""
Mapping models module.
Contains all spatial mapping and coordinate transformation model implementations.
"""

from .homography_model import (
    HomographyModel,
    HomographyValidationResult
)

from .coordinate_transformer import (
    CoordinateTransformer,
    TransformationMode,
    TransformationPath,
    TransformationResult
)

from .calibration_loader import (
    CalibrationLoader,
    CalibrationData,
    CalibrationManifest,
    CameraIntrinsics,
    DistortionCoefficients
)

__all__ = [
    'HomographyModel',
    'HomographyValidationResult',
    'CoordinateTransformer', 
    'TransformationMode',
    'TransformationPath',
    'TransformationResult',
    'CalibrationLoader',
    'CalibrationData',
    'CalibrationManifest',
    'CameraIntrinsics',
    'DistortionCoefficients'
]

# Available mapping model types
AVAILABLE_MODELS = {
    'homography': HomographyModel,
    'coordinate_transformer': CoordinateTransformer,
    'calibration_loader': CalibrationLoader
}