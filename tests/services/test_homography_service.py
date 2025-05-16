# tests/services/test_homography_service.py
"""
Unit tests for the HomographyService in app.services.homography_service.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, AsyncMock

from app.services.homography_service import HomographyService
from app.core.config import Settings, CameraHandoffDetailConfig # Import directly for type hints
from app.common_types import CameraID, ExitRuleModel, ExitDirection

# mock_settings fixture is available from conftest.py

@pytest.fixture
def homography_service_instance(mock_settings):
    """Provides an instance of HomographyService with mocked settings."""
    return HomographyService(settings=mock_settings)

@pytest.fixture
def mock_np_load_for_homography(mocker):
    """Mocks numpy.load specifically for homography point files."""
    return mocker.patch("numpy.load")

@pytest.fixture
def mock_cv2_find_homography_for_service(mocker):
    """Mocks cv2.findHomography specifically for the service tests."""
    # Default success: returns an identity matrix and a dummy mask
    mock_matrix = np.eye(3, dtype=np.float32)
    mask = np.ones((4,1), dtype=np.uint8) # Example mask
    return mocker.patch("cv2.findHomography", return_value=(mock_matrix, mask))


@pytest.mark.asyncio
async def test_homography_service_init(mock_settings):
    """Tests HomographyService initialization."""
    service = HomographyService(settings=mock_settings)
    assert service._settings == mock_settings
    assert not service._preloaded
    assert service._homography_matrices == {}

@pytest.mark.asyncio
async def test_preload_all_homography_matrices_success(
    homography_service_instance: HomographyService,
    mock_settings, # We need to configure CAMERA_HANDOFF_DETAILS on mock_settings
    mock_np_load_for_homography,
    mock_cv2_find_homography_for_service,
    mocker
):
    """Tests successful preloading of all homography matrices."""
    # Configure mock_settings for this test
    env_id = "test_env"
    cam_id1_str = "c01"
    cam_id2_str = "c02" # No homography path
    cam_id3_str = "c03" # File exists, but findHomography will fail for this one

    cam1_path_str = "points_c01.npz"
    cam3_path_str = "points_c03.npz"

    mock_settings.CAMERA_HANDOFF_DETAILS = {
        (env_id, cam_id1_str): CameraHandoffDetailConfig(homography_matrix_path=cam1_path_str),
        (env_id, cam_id2_str): CameraHandoffDetailConfig(homography_matrix_path=None), # No path
        (env_id, cam_id3_str): CameraHandoffDetailConfig(homography_matrix_path=cam3_path_str),
    }
    # resolved_homography_base_path is already mocked in conftest.py's mock_settings
    base_path = mock_settings.resolved_homography_base_path

    # Mock Path.is_file()
    def mock_is_file_side_effect(path_arg: Path):
        if path_arg == base_path / cam1_path_str: return True
        if path_arg == base_path / cam3_path_str: return True
        return False
    mocker.patch.object(Path, "is_file", side_effect=mock_is_file_side_effect)

    # Mock np.load return values
    img_pts_valid = np.array([[1,1],[2,1],[1,2],[2,2]], dtype=np.float32)
    map_pts_valid = np.array([[10,10],[20,10],[10,20],[20,20]], dtype=np.float32)
    mock_np_load_for_homography.side_effect = [
        {"image_points": img_pts_valid, "map_points": map_pts_valid}, # For c01
        {"image_points": img_pts_valid, "map_points": map_pts_valid}, # For c03
    ]

    # Mock cv2.findHomography: success for c01, failure for c03
    mock_cv2_find_homography_for_service.side_effect = [
        (np.eye(3, dtype=np.float32), np.ones(4)), # c01 success
        (None, None) # c03 failure
    ]
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    await homography_service_instance.preload_all_homography_matrices()

    assert homography_service_instance._preloaded is True
    # c01 should have a matrix
    assert (env_id, CameraID(cam_id1_str)) in homography_service_instance._homography_matrices
    assert homography_service_instance._homography_matrices[(env_id, CameraID(cam_id1_str))] is not None
    # c02 should be None (no path)
    assert (env_id, CameraID(cam_id2_str)) in homography_service_instance._homography_matrices
    assert homography_service_instance._homography_matrices[(env_id, CameraID(cam_id2_str))] is None
    # c03 should be None (findHomography failed)
    assert (env_id, CameraID(cam_id3_str)) in homography_service_instance._homography_matrices
    assert homography_service_instance._homography_matrices[(env_id, CameraID(cam_id3_str))] is None

    # Check calls to np.load and cv2.findHomography
    assert mock_np_load_for_homography.call_count == 2 # For c01 and c03
    assert mock_cv2_find_homography_for_service.call_count == 2 # For c01 and c03

@pytest.mark.asyncio
async def test_preload_all_homography_matrices_file_not_found(homography_service_instance: HomographyService, mock_settings, mocker):
    """Tests preloading when a homography file is not found."""
    env_id = "test_env"
    cam_id_str = "c01"
    path_str = "non_existent.npz"
    mock_settings.CAMERA_HANDOFF_DETAILS = {(env_id, cam_id_str): CameraHandoffDetailConfig(homography_matrix_path=path_str)}
    base_path = mock_settings.resolved_homography_base_path
    mocker.patch.object(Path, "is_file", return_value=False)
    mock_logger_warning = mocker.patch("app.services.homography_service.logger.warning")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    await homography_service_instance.preload_all_homography_matrices()

    assert homography_service_instance._homography_matrices[(env_id, CameraID(cam_id_str))] is None
    mock_logger_warning.assert_any_call(
        f"Homography file not found for env '{env_id}', cam '{CameraID(cam_id_str)}': {base_path / path_str}"
    )

@pytest.mark.asyncio
async def test_preload_all_homography_matrices_insufficient_points(
    homography_service_instance: HomographyService, mock_settings, mock_np_load_for_homography, mocker
):
    """Tests preloading with insufficient points in the file."""
    env_id = "test_env"
    cam_id_str = "c01"
    path_str = "points_few.npz"
    mock_settings.CAMERA_HANDOFF_DETAILS = {(env_id, cam_id_str): CameraHandoffDetailConfig(homography_matrix_path=path_str)}
    base_path = mock_settings.resolved_homography_base_path
    mocker.patch.object(Path, "is_file", return_value=True)

    img_pts_few = np.array([[1,1],[2,1]], dtype=np.float32) # Only 2 points
    map_pts_few = np.array([[10,10],[20,10]], dtype=np.float32)
    mock_np_load_for_homography.return_value = {"image_points": img_pts_few, "map_points": map_pts_few}
    mock_logger_warning = mocker.patch("app.services.homography_service.logger.warning")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    await homography_service_instance.preload_all_homography_matrices()

    assert homography_service_instance._homography_matrices[(env_id, CameraID(cam_id_str))] is None
    mock_logger_warning.assert_any_call(
        f"Insufficient/mismatched points in homography file for env '{env_id}', cam '{CameraID(cam_id_str)}': {base_path / path_str}"
    )

@pytest.mark.asyncio
async def test_get_homography_matrix_after_preload(homography_service_instance: HomographyService, mocker):
    """Tests getting a matrix after successful (mocked) preloading."""
    env_id = "test_env"
    cam_id = CameraID("c01")
    mock_matrix = np.eye(3)
    # Manually set up the state as if preloading occurred
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
    # Mock _compute_and_cache_matrix to ensure it's not called again
    mock_compute_internal = mocker.patch.object(homography_service_instance, "_compute_and_cache_matrix", new_callable=AsyncMock)


    await homography_service_instance.preload_all_homography_matrices()
    mock_logger_info.assert_called_with("Homography matrices already preloaded.")
    mock_compute_internal.assert_not_called() 