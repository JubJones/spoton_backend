"""
Unit tests for homography utility functions in app.utils.homography_utlis.
Note: The HomographyService is the primary user of this logic. These tests
are for the low-level utility functions themselves.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from app.utils.homography_utlis import load_homography_matrix_from_points_file, project_point_to_map
from app.common_types import CameraID

@pytest.fixture
def mock_np_load(mocker):
    """Mocks numpy.load."""
    return mocker.patch("numpy.load")

@pytest.fixture
def mock_cv2_find_homography(mocker):
    """Mocks cv2.findHomography."""
    mock_matrix = np.eye(3, dtype=np.float32)
    return mocker.patch("cv2.findHomography", return_value=(mock_matrix, np.array([1,1,1,1])))

@pytest.fixture
def mock_cv2_perspective_transform(mocker):
    """Mocks cv2.perspectiveTransform."""
    return mocker.patch("cv2.perspectiveTransform")

@pytest.fixture
def homography_points_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory for homography point files."""
    d = tmp_path / "homography_data"
    d.mkdir()
    return d

def create_dummy_homography_npz_file(
    dir_path: Path, camera_id: str, scene_id: str, image_points: np.ndarray, map_points: np.ndarray
):
    """Helper to create a dummy .npz file."""
    file_path = dir_path / f"homography_points_{camera_id}_scene_{scene_id}.npz"
    np.savez(file_path, image_points=image_points, map_points=map_points)
    return file_path


def test_load_homography_matrix_successful(
    homography_points_dir, mock_cv2_find_homography, mocker
):
    """Tests successful loading and computation of homography matrix."""
    cam_id = CameraID("c01")
    scene_id = "s01"
    img_pts = np.array([[1,1], [2,1], [1,2], [2,2]], dtype=np.float32)
    map_pts = np.array([[10,10], [20,10], [10,20], [20,20]], dtype=np.float32)
    create_dummy_homography_npz_file(homography_points_dir, str(cam_id), scene_id, img_pts, map_pts)

    matrix = load_homography_matrix_from_points_file(cam_id, scene_id, homography_points_dir)

    assert matrix is not None
    assert isinstance(matrix, np.ndarray)
    mock_cv2_find_homography.assert_called_once()
    # np.testing.assert_array_equal(matrix, np.eye(3, dtype=np.float32)) # Check if it's the mocked matrix

def test_load_homography_matrix_file_not_found(homography_points_dir, mocker):
    """Tests behavior when the .npz file is not found."""
    mock_logger_warning = mocker.patch("app.utils.homography_utlis.logger.warning")
    matrix = load_homography_matrix_from_points_file(CameraID("c99"), "s99", homography_points_dir)
    assert matrix is None
    mock_logger_warning.assert_called_once_with(
        f"Homography points file not found for Cam c99, Scene s99: {homography_points_dir / 'homography_points_c99_scene_s99.npz'}"
    )

def test_load_homography_matrix_insufficient_points(homography_points_dir, mocker):
    """Tests behavior with insufficient points in the .npz file."""
    cam_id = CameraID("c02")
    scene_id = "s02"
    img_pts = np.array([[1,1], [2,1]], dtype=np.float32) # Only 2 points
    map_pts = np.array([[10,10], [20,10]], dtype=np.float32)
    file_path = create_dummy_homography_npz_file(homography_points_dir, str(cam_id), scene_id, img_pts, map_pts)
    mock_logger_warning = mocker.patch("app.utils.homography_utlis.logger.warning")

    matrix = load_homography_matrix_from_points_file(cam_id, scene_id, homography_points_dir)
    assert matrix is None
    mock_logger_warning.assert_called_once_with(
        f"Insufficient points (<4) found in {file_path.name} for Cam {cam_id}."
    )

def test_load_homography_matrix_mismatched_points(homography_points_dir, mocker):
    """Tests behavior with mismatched point counts."""
    cam_id = CameraID("c03")
    scene_id = "s03"
    img_pts = np.array([[1,1], [2,1], [1,2], [2,2]], dtype=np.float32)
    map_pts = np.array([[10,10], [20,10], [10,20]], dtype=np.float32) # Only 3 map points
    file_path = create_dummy_homography_npz_file(homography_points_dir, str(cam_id), scene_id, img_pts, map_pts)
    mock_logger_error = mocker.patch("app.utils.homography_utlis.logger.error")

    matrix = load_homography_matrix_from_points_file(cam_id, scene_id, homography_points_dir)
    assert matrix is None
    mock_logger_error.assert_called_once_with(
        f"Mismatch in point counts in {file_path.name} for Cam {cam_id}."
    )


def test_load_homography_cv2_find_homography_fails(homography_points_dir, mock_cv2_find_homography, mocker):
    """Tests behavior when cv2.findHomography returns None."""
    mock_cv2_find_homography.return_value = (None, None) # Simulate failure
    cam_id = CameraID("c04")
    scene_id = "s04"
    img_pts = np.array([[1,1], [2,1], [1,2], [2,2]], dtype=np.float32)
    map_pts = np.array([[10,10], [20,10], [10,20], [20,20]], dtype=np.float32)
    file_path = create_dummy_homography_npz_file(homography_points_dir, str(cam_id), scene_id, img_pts, map_pts)
    mock_logger_error = mocker.patch("app.utils.homography_utlis.logger.error")

    matrix = load_homography_matrix_from_points_file(cam_id, scene_id, homography_points_dir)
    assert matrix is None
    mock_logger_error.assert_called_once_with(
         f"Homography calculation failed (cv2.findHomography returned None) for Cam {cam_id} using {file_path.name}."
    )


def test_project_point_to_map_successful(mock_cv2_perspective_transform):
    """Tests successful point projection."""
    homography_matrix = np.eye(3)
    image_point = (10.0, 20.0)
    mock_cv2_perspective_transform.return_value = np.array([[[100.0, 200.0]]], dtype=np.float32)

    map_point = project_point_to_map(image_point, homography_matrix)

    assert map_point == (100.0, 200.0)
    # Check that cv2.perspectiveTransform was called with correct args
    # The first arg to perspectiveTransform should be np.array([[[10., 20.]]], dtype=np.float32)
    call_args = mock_cv2_perspective_transform.call_args[0]
    assert np.array_equal(call_args[0], np.array([[image_point]], dtype=np.float32))
    assert np.array_equal(call_args[1], homography_matrix)


def test_project_point_to_map_matrix_is_none(mocker):
    """Tests projection when homography matrix is None."""
    mock_logger_debug = mocker.patch("app.utils.homography_utlis.logger.debug")
    map_point = project_point_to_map((10.0, 20.0), None) # type: ignore
    assert map_point is None
    mock_logger_debug.assert_called_with("Projection skipped: Homography matrix is None for point (10.0, 20.0)")

def test_project_point_to_map_transform_fails(mock_cv2_perspective_transform, mocker):
    """Tests projection when cv2.perspectiveTransform returns unexpected result."""
    mock_cv2_perspective_transform.return_value = None # Simulate failure
    mock_logger_warning = mocker.patch("app.utils.homography_utlis.logger.warning")
    homography_matrix = np.eye(3)
    image_point = (10.0, 20.0)

    map_point = project_point_to_map(image_point, homography_matrix)
    assert map_point is None
    mock_logger_warning.assert_called_once_with(
        f"Perspective transform returned unexpected shape or None for point {image_point}. Result: None"
    )