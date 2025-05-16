"""
Unit tests for tracker models in app.models.trackers.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, AsyncMock, PropertyMock, patch
from pathlib import Path
import re

from app.models.trackers import BotSortTracker

@pytest.fixture
def mock_boxmot_botsort_class(mocker):
    """Mocks the boxmot.trackers.botsort.BotSort class."""
    mock_botsort_instance = MagicMock() # This will be the instance returned by BotSort()
    # Mock the update method to be an async mock if BotSort.update is async,
    # or a regular MagicMock if it's sync and called via asyncio.to_thread.
    # BotSort.update is synchronous.
    mock_botsort_instance.update = MagicMock(return_value=np.empty((0,8))) # Default empty tracks
    mock_botsort_instance.reset = MagicMock()

    mock_botsort_class = MagicMock(return_value=mock_botsort_instance) # This is the mock for the class itself
    mocker.patch("app.models.trackers.BotSort", mock_botsort_class, create=True)
    return mock_botsort_class, mock_botsort_instance


@pytest.fixture
def botsort_tracker_instance(mock_settings, mock_boxmot_botsort_class, mocker):
    """Provides an instance of BotSortTracker with mocked dependencies."""
    mocker.patch("app.models.trackers.settings", mock_settings)
    
    actual_cpu_device = torch.device('cpu')
    # Patch torch.device where it's imported and used by BotSortTracker for self.device initialization
    mocker.patch("app.models.trackers.torch.device", return_value=actual_cpu_device)
    
    # Patch get_boxmot_device_string where it's imported and used by BotSortTracker
    # This prevents the actual get_boxmot_device_string (and its isinstance check) from running.
    mocker.patch("app.models.trackers.get_boxmot_device_string", return_value="cpu")
    
    return BotSortTracker()


@pytest.mark.asyncio
async def test_botsort_init(botsort_tracker_instance: BotSortTracker, mock_settings):
    """Tests BotSortTracker initialization."""
    tracker = botsort_tracker_instance
    assert tracker.reid_model_path == mock_settings.resolved_reid_weights_path
    assert tracker.use_half == mock_settings.TRACKER_HALF_PRECISION
    assert tracker.per_class == mock_settings.TRACKER_PER_CLASS
    assert tracker.tracker_instance is None
    assert tracker._model_loaded_flag is False

@pytest.mark.asyncio
async def test_botsort_load_model_success(botsort_tracker_instance: BotSortTracker, mock_boxmot_botsort_class, mock_settings, mocker):
    """Tests successful BotSort model loading."""
    tracker = botsort_tracker_instance
    mock_botsort_class_constructor, mock_botsort_instance = mock_boxmot_botsort_class
    mock_logger_info = mocker.patch("app.models.trackers.logger.info")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    await tracker.load_model()

    assert tracker.tracker_instance == mock_botsort_instance
    assert tracker._model_loaded_flag is True
    mock_botsort_class_constructor.assert_called_once_with(
        reid_weights=mock_settings.resolved_reid_weights_path,
        device="cpu", # As mocked in get_boxmot_device_string
        half=mock_settings.TRACKER_HALF_PRECISION and tracker.device.type == 'cuda', # Effective half
        per_class=mock_settings.TRACKER_PER_CLASS
    )
    mock_logger_info.assert_any_call(f"BotSort tracker instance created with ReID model '{mock_settings.resolved_reid_weights_path}'.")

@pytest.mark.asyncio
async def test_botsort_load_model_failure(botsort_tracker_instance: BotSortTracker, mock_boxmot_botsort_class, mocker):
    """Tests BotSort model loading failure."""
    tracker = botsort_tracker_instance
    mock_botsort_class_constructor, _ = mock_boxmot_botsort_class
    mock_botsort_class_constructor.side_effect = RuntimeError("Load failed")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    mock_logger_exception = mocker.patch("app.models.trackers.logger.exception")


    with pytest.raises(RuntimeError, match="Load failed"):
        await tracker.load_model()

    assert tracker.tracker_instance is None
    assert tracker._model_loaded_flag is False
    mock_logger_exception.assert_called_once()


@pytest.mark.asyncio
async def test_botsort_warmup(botsort_tracker_instance: BotSortTracker, mocker):
    """Tests tracker warmup."""
    tracker = botsort_tracker_instance
    await tracker.load_model() # Ensure model is loaded

    mock_update_internal = mocker.patch.object(tracker, "update", new_callable=AsyncMock)
    mock_logger_info = mocker.patch("app.models.trackers.logger.info")

    await tracker.warmup()

    mock_update_internal.assert_called_once()
    # Check that the arguments to update were a (0,6) numpy array for detections and a (640,480,3) image
    assert isinstance(mock_update_internal.call_args[0][0], np.ndarray)
    assert mock_update_internal.call_args[0][0].shape == (0, 6)
    assert isinstance(mock_update_internal.call_args[0][1], np.ndarray)
    assert mock_update_internal.call_args[0][1].shape == (640, 480, 3)
    mock_logger_info.assert_any_call("BotSortTracker warmup successful.")


@pytest.mark.asyncio
async def test_botsort_update_model_not_loaded(botsort_tracker_instance: BotSortTracker):
    """Tests update call when model is not loaded."""
    tracker = botsort_tracker_instance
    dummy_detections = np.empty((0, 6))
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    expected_message = "BotSort tracker not loaded. Call load_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        await tracker.update(dummy_detections, dummy_image)

@pytest.mark.asyncio
async def test_botsort_update_successful(botsort_tracker_instance: BotSortTracker, mock_boxmot_botsort_class, mocker):
    """Tests successful tracker update."""
    tracker = botsort_tracker_instance
    await tracker.load_model()
    _, mock_botsort_instance = mock_boxmot_botsort_class

    detections_np = np.array([[10, 10, 20, 20, 0.9, 0]], dtype=np.float32)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    expected_output = np.array([[10, 10, 20, 20, 1, 0.9, 0, 123]], dtype=np.float32) # x1,y1,x2,y2,track_id,conf,cls,global_id (example)
    mock_botsort_instance.update.return_value = expected_output

    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))

    output = await tracker.update(detections_np, image_np)

    mock_botsort_instance.update.assert_called_once()
    # Check that arguments to the underlying BoxMOT update were correct
    assert np.array_equal(mock_botsort_instance.update.call_args[0][0], detections_np)
    assert np.array_equal(mock_botsort_instance.update.call_args[0][1], image_np)
    assert np.array_equal(output, expected_output)

@pytest.mark.asyncio
async def test_botsort_update_invalid_detections_shape(botsort_tracker_instance: BotSortTracker, mocker):
    """Tests tracker update with invalid detections shape."""
    tracker = botsort_tracker_instance
    await tracker.load_model()
    mock_logger_error = mocker.patch("app.models.trackers.logger.error")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    invalid_detections = np.array([1, 2, 3, 4, 5]) # Not (N, 6)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)

    output = await tracker.update(invalid_detections, image_np) # type: ignore
    assert output.shape == (0, 8) # Expect empty output
    mock_logger_error.assert_called_with("Invalid detections shape: (5,). Expected (N, 6) for tracker update.")

@pytest.mark.asyncio
async def test_botsort_reset(botsort_tracker_instance: BotSortTracker, mock_boxmot_botsort_class, mocker):
    """Tests tracker reset."""
    tracker = botsort_tracker_instance
    await tracker.load_model()
    _, mock_botsort_instance = mock_boxmot_botsort_class
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    await tracker.reset()
    mock_botsort_instance.reset.assert_called_once()

@pytest.mark.asyncio
async def test_botsort_reset_reinitializes_if_no_reset_method(botsort_tracker_instance: BotSortTracker, mock_boxmot_botsort_class, mocker):
    """Tests that tracker re-initializes if its instance has no reset method (fallback)."""
    tracker = botsort_tracker_instance
    await tracker.load_model() # Initial load
    
    _, mock_botsort_instance = mock_boxmot_botsort_class
    del mock_botsort_instance.reset # Remove reset method to test fallback

    mock_load_model_on_tracker = mocker.spy(tracker, "load_model")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    await tracker.reset()
    
    # load_model should have been called twice: once initially, once during reset
    assert mock_load_model_on_tracker.call_count == 2
    assert tracker._model_loaded_flag is True # Should be true after re-init 