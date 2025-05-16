"""
Unit tests for feature extractor models in app.models.feature_extractors.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, AsyncMock, PropertyMock, patch
from pathlib import Path

from app.models.feature_extractors import BoxMOTFeatureExtractor, BOXMOT_REID_AVAILABLE
# Assuming mock_settings is available from conftest.py

# Conditionally skip all tests in this file if BoxMOT ReidAutoBackend is not available
if not BOXMOT_REID_AVAILABLE:
    pytest.skip("Skipping feature_extractors tests: BoxMOT ReidAutoBackend not available", allow_module_level=True)


@pytest.fixture
def mock_reid_auto_backend_class(mocker):
    """Mocks the boxmot.appearance.reid_auto_backend.ReidAutoBackend class."""
    if not BOXMOT_REID_AVAILABLE: # Should not be reached if module is skipped
        return None, None

    mock_reid_instance = MagicMock()
    mock_reid_instance.get_features = MagicMock(return_value=np.empty((0, 512))) # Default empty features
    
    # Mock the underlying model within ReidAutoBackend if warmup is tested
    mock_reid_instance.model = MagicMock()
    mock_reid_instance.model.warmup = MagicMock()


    mock_reid_class = MagicMock(return_value=mock_reid_instance)
    mocker.patch("app.models.feature_extractors.ReidAutoBackend", mock_reid_class, create=True)
    return mock_reid_class, mock_reid_instance


@pytest.fixture
def boxmot_extractor_instance(mock_settings, mock_reid_auto_backend_class, mocker):
    """Provides an instance of BoxMOTFeatureExtractor with mocked dependencies."""
    if not BOXMOT_REID_AVAILABLE:
        return None

    mocker.patch("app.models.feature_extractors.settings", mock_settings)
    mock_settings.REID_MODEL_TYPE = "clip" # Example, can be varied
    mock_settings.REID_MODEL_HALF_PRECISION = False

    # Mock torch.device and get_boxmot_device_string
    test_device = torch.device("cpu") # For simplicity in these tests
    mocker.patch("torch.device", return_value=test_device)
    mocker.patch("app.utils.device_utils.get_boxmot_device_string", return_value="cpu")

    return BoxMOTFeatureExtractor(device=test_device)

@pytest.mark.asyncio
async def test_boxmot_extractor_init(boxmot_extractor_instance: BoxMOTFeatureExtractor, mock_settings):
    """Tests BoxMOTFeatureExtractor initialization."""
    if not boxmot_extractor_instance: pytest.skip("BoxMOT not available")
    extractor = boxmot_extractor_instance
    assert extractor.reid_model_path == mock_settings.resolved_reid_weights_path
    assert extractor.model_type == mock_settings.REID_MODEL_TYPE
    assert extractor.use_half == mock_settings.REID_MODEL_HALF_PRECISION
    assert extractor.reid_model_handler is None
    assert extractor._model_loaded_flag is False

@pytest.mark.asyncio
async def test_boxmot_extractor_load_model_success(boxmot_extractor_instance: BoxMOTFeatureExtractor, mock_reid_auto_backend_class, mock_settings, mocker):
    """Tests successful ReID model loading."""
    if not boxmot_extractor_instance: pytest.skip("BoxMOT not available")
    extractor = boxmot_extractor_instance
    mock_reid_class_constructor, mock_reid_instance = mock_reid_auto_backend_class # type: ignore
    mock_logger_info = mocker.patch("app.models.feature_extractors.logger.info")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))

    await extractor.load_model()

    assert extractor.reid_model_handler == mock_reid_instance
    assert extractor._model_loaded_flag is True
    mock_reid_class_constructor.assert_called_once_with(
        weights=mock_settings.resolved_reid_weights_path,
        device="cpu", # As mocked
        half=mock_settings.REID_MODEL_HALF_PRECISION and extractor.device.type == 'cuda', # Effective half
        model_type=mock_settings.REID_MODEL_TYPE
    )
    # Check if warmup on the inner model was called
    if hasattr(mock_reid_instance.model, "warmup"):
         mock_reid_instance.model.warmup.assert_called_once()

    mock_logger_info.assert_any_call(f"Standalone ReID model '{mock_settings.REID_MODEL_TYPE}' loaded successfully from '{mock_settings.resolved_reid_weights_path}'.")

@pytest.mark.asyncio
async def test_boxmot_extractor_load_model_failure(boxmot_extractor_instance: BoxMOTFeatureExtractor, mock_reid_auto_backend_class, mocker):
    """Tests ReID model loading failure."""
    if not boxmot_extractor_instance: pytest.skip("BoxMOT not available")
    extractor = boxmot_extractor_instance
    mock_reid_class_constructor, _ = mock_reid_auto_backend_class # type: ignore
    mock_reid_class_constructor.side_effect = RuntimeError("Load failed")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    mock_logger_exception = mocker.patch("app.models.feature_extractors.logger.exception")

    with pytest.raises(RuntimeError, match="Failed to load standalone ReID model: Load failed"):
        await extractor.load_model()

    assert extractor.reid_model_handler is None
    assert extractor._model_loaded_flag is False
    mock_logger_exception.assert_called_once()

@pytest.mark.asyncio
async def test_boxmot_extractor_get_features_model_not_loaded(boxmot_extractor_instance: BoxMOTFeatureExtractor, mocker):
    """Tests get_features call when model is not loaded."""
    if not boxmot_extractor_instance: pytest.skip("BoxMOT not available")
    extractor = boxmot_extractor_instance
    mock_logger_error = mocker.patch("app.models.feature_extractors.logger.error")
    dummy_bboxes = np.array([[0,0,10,10]])
    dummy_image = np.zeros((100,100,3), dtype=np.uint8)

    features = await extractor.get_features(dummy_bboxes, dummy_image)
    assert features is None
    mock_logger_error.assert_called_with("Standalone ReID model not loaded. Call load_model() first.")


@pytest.mark.asyncio
async def test_boxmot_extractor_get_features_successful(boxmot_extractor_instance: BoxMOTFeatureExtractor, mock_reid_auto_backend_class, mocker):
    """Tests successful feature extraction."""
    if not boxmot_extractor_instance: pytest.skip("BoxMOT not available")
    extractor = boxmot_extractor_instance
    await extractor.load_model()
    _, mock_reid_instance = mock_reid_auto_backend_class # type: ignore

    bboxes_np = np.array([[10, 10, 20, 20]], dtype=np.float32)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    expected_features = np.random.rand(1, 512).astype(np.float32)
    mock_reid_instance.get_features.return_value = expected_features

    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))

    output_features = await extractor.get_features(bboxes_np, image_np)

    mock_reid_instance.get_features.assert_called_once()
    # Check arguments to the underlying ReidAutoBackend.get_features
    assert np.array_equal(mock_reid_instance.get_features.call_args[0][0], bboxes_np)
    assert np.array_equal(mock_reid_instance.get_features.call_args[0][1], image_np)
    assert np.array_equal(output_features, expected_features) # type: ignore

@pytest.mark.asyncio
async def test_boxmot_extractor_get_features_empty_bboxes(boxmot_extractor_instance: BoxMOTFeatureExtractor, mocker):
    """Tests get_features with empty bounding boxes."""
    if not boxmot_extractor_instance: pytest.skip("BoxMOT not available")
    extractor = boxmot_extractor_instance
    await extractor.load_model()
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))

    empty_bboxes = np.empty((0, 4))
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)

    output_features = await extractor.get_features(empty_bboxes, image_np)
    assert output_features is not None
    assert output_features.shape == (0,0) # Expect empty array with 0 features, 0 dims based on current impl

    none_bboxes = None
    output_features_none = await extractor.get_features(none_bboxes, image_np) # type: ignore
    assert output_features_none is not None
    assert output_features_none.shape == (0,0) 