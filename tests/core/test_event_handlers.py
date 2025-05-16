# tests/core/test_event_handlers.py
"""
Unit tests for application lifecycle event handlers in app.core.event_handlers.
"""
import pytest
import torch
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from app.core import event_handlers # Import the module
# mock_settings is available from conftest.py

# Mock dependencies for event_handlers
@pytest.fixture
def mock_fastapi_app(mocker):
    """Mocks a FastAPI app instance with a MagicMock for app.state."""
    app_mock = MagicMock()
    app_mock.state = MagicMock() # Allow attributes to be set on state
    # Initialize state attributes to None to mimic pre-startup state
    app_mock.state.compute_device = None
    app_mock.state.detector = None
    app_mock.state.tracker_factory = None
    app_mock.state.homography_service = None
    return app_mock

@pytest.fixture
def mock_get_selected_device_util(mocker):
    """Mocks app.utils.device_utils.get_selected_device."""
    return mocker.patch("app.core.event_handlers.get_selected_device", return_value=torch.device("cpu"))

@pytest.fixture
def mock_detector_class_in_events(mocker):
    """Mocks FasterRCNNDetector class used in event_handlers."""
    mock_instance = MagicMock() # Instance of FasterRCNNDetector
    mock_instance.load_model = AsyncMock()
    mock_instance.warmup = AsyncMock()
    mock_instance._model_loaded_flag = True # Simulate successful load
    
    mock_class = MagicMock(return_value=mock_instance) # Constructor returns our instance
    mocker.patch("app.core.event_handlers.FasterRCNNDetector", mock_class)
    return mock_class, mock_instance

@pytest.fixture
def mock_tracker_factory_class_in_events(mocker):
    """Mocks CameraTrackerFactory class used in event_handlers."""
    mock_instance = MagicMock() # Instance of CameraTrackerFactory
    mock_instance.preload_prototype_tracker = AsyncMock()
    mock_instance._prototype_tracker_loaded = True # Simulate successful preload

    mock_class = MagicMock(return_value=mock_instance)
    mocker.patch("app.core.event_handlers.CameraTrackerFactory", mock_class)
    return mock_class, mock_instance

@pytest.fixture
def mock_homography_service_class_in_events(mocker):
    """Mocks HomographyService class used in event_handlers."""
    mock_instance = MagicMock() # Instance of HomographyService
    mock_instance.preload_all_homography_matrices = AsyncMock()
    mock_instance._preloaded = True # Simulate successful preload

    mock_class = MagicMock(return_value=mock_instance)
    mocker.patch("app.core.event_handlers.HomographyService", mock_class)
    return mock_class, mock_instance


@pytest.mark.asyncio
async def test_on_startup_successful_flow(
    mock_fastapi_app: MagicMock,
    mock_settings, # To pass to HomographyService if needed, or patch settings import in event_handlers
    mock_get_selected_device_util: MagicMock,
    mock_detector_class_in_events,
    mock_tracker_factory_class_in_events,
    mock_homography_service_class_in_events,
    mocker
):
    """Tests the successful execution of on_startup event handler."""
    # Patch settings where event_handlers imports it
    mocker.patch("app.core.event_handlers.settings", mock_settings)

    _, mock_detector_instance = mock_detector_class_in_events
    _, mock_tracker_factory_instance = mock_tracker_factory_class_in_events
    mock_homography_service_constructor, mock_homography_instance = mock_homography_service_class_in_events

    await event_handlers.on_startup(mock_fastapi_app)

    # 1. Device selection
    mock_get_selected_device_util.assert_called_once_with(requested_device="auto")
    assert mock_fastapi_app.state.compute_device == torch.device("cpu") # From mock_get_selected_device_util

    # 2. Detector
    mock_detector_class_in_events[0].assert_called_once() # Constructor
    mock_detector_instance.load_model.assert_called_once()
    mock_detector_instance.warmup.assert_called_once()
    assert mock_fastapi_app.state.detector == mock_detector_instance

    # 3. Tracker Factory
    mock_tracker_factory_class_in_events[0].assert_called_once_with(device=torch.device("cpu"))
    mock_tracker_factory_instance.preload_prototype_tracker.assert_called_once()
    assert mock_fastapi_app.state.tracker_factory == mock_tracker_factory_instance

    # 4. Homography Service
    mock_homography_service_constructor.assert_called_once_with(settings=mock_settings)
    mock_homography_instance.preload_all_homography_matrices.assert_called_once()
    assert mock_fastapi_app.state.homography_service == mock_homography_instance


@pytest.mark.asyncio
async def test_on_startup_device_selection_fails(mock_fastapi_app: MagicMock, mock_get_selected_device_util: MagicMock, mocker):
    """Tests on_startup when device selection utility raises an error."""
    mock_get_selected_device_util.side_effect = Exception("Device detection crashed")
    mock_logger_error = mocker.patch("app.core.event_handlers.logger.error")
    mock_logger_warning = mocker.patch("app.core.event_handlers.logger.warning")

    # Prevent other initializations from running by not mocking them fully or expecting calls
    mocker.patch("app.core.event_handlers.FasterRCNNDetector")
    mocker.patch("app.core.event_handlers.CameraTrackerFactory")
    mocker.patch("app.core.event_handlers.HomographyService")


    await event_handlers.on_startup(mock_fastapi_app)

    mock_logger_error.assert_called_with("Failed to determine compute device during startup: Device detection crashed", exc_info=True)
    mock_logger_warning.assert_called_with("Defaulting to CPU due to error in device selection.")
    assert mock_fastapi_app.state.compute_device == torch.device("cpu") # Should fallback to CPU

@pytest.mark.asyncio
async def test_on_startup_detector_preload_fails(mock_fastapi_app: MagicMock, mock_detector_class_in_events, mocker):
    """Tests on_startup when detector preloading fails."""
    mock_constructor, mock_instance = mock_detector_class_in_events
    mock_instance.load_model.side_effect = RuntimeError("Detector model load error")
    mock_logger_error = mocker.patch("app.core.event_handlers.logger.error")
    
    # Patch other services to prevent their init if detector is critical
    mocker.patch("app.core.event_handlers.CameraTrackerFactory")
    mocker.patch("app.core.event_handlers.HomographyService")
    mocker.patch("app.core.event_handlers.get_selected_device", return_value=torch.device("cpu"))


    await event_handlers.on_startup(mock_fastapi_app)

    mock_logger_error.assert_called_with("CRITICAL: Failed to preload detector model: Detector model load error", exc_info=True)
    assert mock_fastapi_app.state.detector is None # Should be marked as not loaded

@pytest.mark.asyncio
async def test_on_shutdown(mock_fastapi_app: MagicMock, mocker):
    """Tests the on_shutdown event handler."""
    mock_cuda_is_available = mocker.patch("torch.cuda.is_available", return_value=True)
    mock_cuda_empty_cache = mocker.patch("torch.cuda.empty_cache")
    mock_logger_info = mocker.patch("app.core.event_handlers.logger.info")

    await event_handlers.on_shutdown(mock_fastapi_app)

    mock_cuda_is_available.assert_called_once()
    mock_cuda_empty_cache.assert_called_once()
    mock_logger_info.assert_any_call("Cleared PyTorch CUDA cache.")
    mock_logger_info.assert_any_call("--- SpotOn Backend: Application Shutdown Tasks Completed ---")

@pytest.mark.asyncio
async def test_on_shutdown_cuda_not_available(mock_fastapi_app: MagicMock, mocker):
    """Tests on_shutdown when CUDA is not available."""
    mocker.patch("torch.cuda.is_available", return_value=False) # CUDA not available
    mock_cuda_empty_cache = mocker.patch("torch.cuda.empty_cache")

    await event_handlers.on_shutdown(mock_fastapi_app)
    mock_cuda_empty_cache.assert_not_called() # Should not be called 