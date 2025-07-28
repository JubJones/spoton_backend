"""
Unit tests for the HomographyService in app.services.homography_service.
"""
import pytest
import numpy as np
from pathlib import Path as StandardPath
from unittest.mock import MagicMock, PropertyMock, call, AsyncMock

from app.services.homography_service import HomographyService
from app.core.config import Settings, CameraHandoffDetailConfig
from app.shared.types import CameraID, ExitRuleModel


@pytest.fixture
def homography_service_instance(mock_settings):
    """Provides an instance of HomographyService with mocked settings."""
    return HomographyService(settings=mock_settings)

@pytest.fixture
def mock_np_load_for_homography(mocker):
    """Mocks numpy.load specifically for homography point files."""
    return mocker.patch("app.services.homography_service.np.load")

@pytest.fixture
def mock_cv2_find_homography_for_service(mocker):
    """Mocks cv2.findHomography specifically for the service tests."""
    mock_matrix = np.eye(3, dtype=np.float32)
    mask = np.ones((4,1), dtype=np.uint8)
    return mocker.patch("app.services.homography_service.cv2.findHomography", return_value=(mock_matrix, mask))

@pytest.fixture
def mock_path_constructor_factory(mocker):
    """
    Factory fixture to create a sophisticated mock for the pathlib.Path constructor.
    It allows configuring which path strings should make `is_file()` return True.
    Returns a function that applies the patch and allows setting existing files.
    """
    
    # This dictionary will store the state (which paths exist) for the mock
    # It's reset each time the factory is called by a test.
    _existing_files_for_current_test = set()

    def _path_constructor_side_effect(*args):
        # Reconstruct the path string as Path would
        path_str = str(StandardPath(*args))
        
        # Create a new MagicMock for this Path instance
        path_mock = MagicMock(spec=StandardPath)
        path_mock.__str__ = MagicMock(return_value=path_str)
        path_mock.name = StandardPath(path_str).name # Keep some real attributes
        
        # Configure is_file based on the current test's setup
        path_mock.is_file = MagicMock(return_value=(path_str in _existing_files_for_current_test))
        
        # Ensure path joining (division) returns another mocked Path
        def truediv_side_effect(other):
            # When path_mock / "filename" is called, 'other' is "filename"
            # We need to call the constructor mock again with the combined path parts
            return _path_constructor_side_effect(path_str, other) 
        path_mock.__truediv__ = MagicMock(side_effect=truediv_side_effect)
        
        return path_mock

    # Patch the Path constructor in the module where it's used
    patched_constructor = mocker.patch("app.services.homography_service.Path", side_effect=_path_constructor_side_effect)
    
    # Return a function that tests can use to set which files should "exist"
    def _configure_existing_files(paths_that_exist: list[str]):
        _existing_files_for_current_test.clear()
        _existing_files_for_current_test.update(paths_that_exist)
        return patched_constructor # Return the mock constructor for assertions if needed
        
    return _configure_existing_files


@pytest.mark.asyncio
async def test_homography_service_init(mock_settings):
    """Tests HomographyService initialization."""
    service = HomographyService(settings=mock_settings)
    assert service._settings == mock_settings
    assert not service._preloaded
    assert service._homography_matrices == {}


@pytest.mark.asyncio
async def test_preload_all_homography_matrices_file_not_found(
    homography_service_instance: HomographyService,
    mock_settings,
    mock_path_constructor_factory, 
    mocker
):
    """Tests preloading when a homography file is not found."""
    env_id = "test_env"
    cam_id_str = "c01"
    filename_str = "non_existent.npz"
    mock_settings.CAMERA_HANDOFF_DETAILS = {(env_id, cam_id_str): CameraHandoffDetailConfig(homography_matrix_path=filename_str)}
    base_path_obj_from_settings = mock_settings.resolved_homography_base_path
    base_path_str = str(base_path_obj_from_settings)

    # Configure: no files exist for this test
    mock_path_constructor_factory([]) # Pass empty list, so all is_file calls on mocked Paths return False
        
    mock_logger_warning = mocker.patch("app.services.homography_service.logger.warning")
    mocker.patch("app.services.homography_service.asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))

    await homography_service_instance.preload_all_homography_matrices()

    assert (env_id, CameraID(cam_id_str)) in homography_service_instance._homography_matrices
    assert homography_service_instance._homography_matrices[(env_id, CameraID(cam_id_str))] is None
    
    expected_missing_path = StandardPath(base_path_str) / filename_str
    mock_logger_warning.assert_any_call(
        f"Homography file not found for env '{env_id}', cam '{CameraID(cam_id_str)}': {expected_missing_path}"
    )

@pytest.mark.asyncio
async def test_get_homography_matrix_after_preload(homography_service_instance: HomographyService, mocker):
    """Tests getting a matrix after successful (mocked) preloading."""
    env_id = "test_env"
    cam_id = CameraID("c01")
    mock_matrix = np.eye(3)
    homography_service_instance._homography_matrices[(env_id, cam_id)] = mock_matrix
    homography_service_instance._preloaded = True

    retrieved_matrix = homography_service_instance.get_homography_matrix(env_id, cam_id)
    assert np.array_equal(retrieved_matrix, mock_matrix)

@pytest.mark.asyncio
async def test_get_homography_matrix_not_preloaded(homography_service_instance: HomographyService):
    """Tests that get_homography_matrix raises error if called before preloading."""
    with pytest.raises(RuntimeError, match="HomographyService: Matrices not preloaded."):
        homography_service_instance.get_homography_matrix("test_env", CameraID("c01"))

@pytest.mark.asyncio
async def test_preload_already_done(homography_service_instance: HomographyService, mocker):
    """Tests that preload does nothing if already preloaded."""
    homography_service_instance._preloaded = True
    mock_logger_info = mocker.patch("app.services.homography_service.logger.info")
    mock_compute_internal = mocker.patch.object(homography_service_instance, "_compute_and_cache_matrix", new_callable=AsyncMock)


    await homography_service_instance.preload_all_homography_matrices()
    mock_logger_info.assert_called_with("Homography matrices already preloaded.")
    mock_compute_internal.assert_not_called()