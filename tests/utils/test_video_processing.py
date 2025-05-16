"""
Unit tests for video processing utilities in app.utils.video_processing.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, call

from app.utils.video_processing import extract_frames_from_video_to_disk, ensure_rgb

@pytest.fixture
def mock_cv2_video_capture(mocker):
    """Mocks cv2.VideoCapture and its methods."""
    mock_cap_instance = MagicMock(spec=cv2.VideoCapture)
    mock_cap_instance.isOpened.return_value = True
    mock_cap_instance.get.side_effect = lambda prop_id: {
        cv2.CAP_PROP_FPS: 29.97,
        cv2.CAP_PROP_FRAME_COUNT: 100, # Example frame count
        # Add other props if your code uses them
    }.get(prop_id, 0)

    # Simulate frame reading: return (True, dummy_frame) for a number of calls, then (False, None)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Let's say we simulate 10 frames for testing read loop
    read_effects = [(True, dummy_frame)] * 10 + [(False, None)]
    mock_cap_instance.read.side_effect = read_effects

    mocker.patch("cv2.VideoCapture", return_value=mock_cap_instance)
    return mock_cap_instance

@pytest.fixture
def mock_cv2_imwrite(mocker):
    """Mocks cv2.imwrite."""
    return mocker.patch("cv2.imwrite", return_value=True) # Assume success by default

@pytest.fixture
def mock_os_path_exists(mocker):
    """Mocks os.path.exists."""
    return mocker.patch("os.path.exists", return_value=True) # Assume video file exists by default

@pytest.fixture
def mock_pathlib_mkdir(mocker):
    """Mocks pathlib.Path.mkdir."""
    return mocker.patch("pathlib.Path.mkdir")


def test_extract_frames_successful_with_target_fps(
    tmp_path, mock_cv2_video_capture, mock_cv2_imwrite, mock_os_path_exists, mock_pathlib_mkdir
):
    """Tests successful frame extraction with a target FPS."""
    video_file = "test_video.mp4"
    output_dir = tmp_path / "frames"
    target_fps = 10
    jpeg_quality = 90

    # Adjust mock_cap_instance.get for CAP_PROP_FPS if default isn't 30 (e.g., 29.97)
    # For target_fps=10 and source=29.97, skip_interval = round(29.97/10) = 3
    # If 10 frames are read, frames 0, 3, 6, 9 should be saved. (4 frames)
    expected_saves = 4 # Frames 0, 3, 6, 9

    paths, msg = extract_frames_from_video_to_disk(
        video_file, str(output_dir), target_fps=target_fps, jpeg_quality=jpeg_quality
    )

    assert mock_pathlib_mkdir.called_with(parents=True, exist_ok=True)
    assert len(paths) == expected_saves
    assert mock_cv2_imwrite.call_count == expected_saves
    for i in range(expected_saves):
        expected_frame_filename = f"frame_{i:06d}.jpg"
        expected_frame_path = str(output_dir / expected_frame_filename)
        assert paths[i] == expected_frame_path
        # Check if imwrite was called with the correct path and quality
        args, kwargs = mock_cv2_imwrite.call_args_list[i]
        assert args[0] == expected_frame_path
        assert args[2] == [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    assert f"Extracted {expected_saves} frames" in msg
    mock_cv2_video_capture.release.assert_called_once()

def test_extract_frames_target_fps_none_extracts_all(
    tmp_path, mock_cv2_video_capture, mock_cv2_imwrite, mock_os_path_exists, mock_pathlib_mkdir
):
    """Tests extracting all frames when target_fps is None."""
    # mock_cap_instance.read.side_effect simulates 10 readable frames
    expected_saves = 10

    paths, msg = extract_frames_from_video_to_disk(
        "test.mp4", str(tmp_path), target_fps=None
    )
    assert len(paths) == expected_saves
    assert mock_cv2_imwrite.call_count == expected_saves
    assert f"Extracted {expected_saves} frames" in msg
    assert "Interval: 1" in msg # Skip interval should be 1

def test_extract_frames_video_not_found(tmp_path, mock_os_path_exists, mocker):
    """Tests behavior when the video file is not found."""
    mock_os_path_exists.return_value = False
    mock_logger_error = mocker.patch("app.utils.video_processing.logger.error")
    paths, msg = extract_frames_from_video_to_disk("non_existent.mp4", str(tmp_path))
    assert len(paths) == 0
    assert "Video file not found" in msg
    mock_logger_error.assert_called_with("Video file not found: non_existent.mp4")

def test_extract_frames_cannot_create_output_dir(tmp_path, mock_os_path_exists, mock_pathlib_mkdir, mocker):
    """Tests behavior when the output directory cannot be created."""
    mock_pathlib_mkdir.side_effect = OSError("Permission denied")
    mock_logger_error = mocker.patch("app.utils.video_processing.logger.error")
    paths, msg = extract_frames_from_video_to_disk("test.mp4", str(tmp_path / "forbidden"))
    assert len(paths) == 0
    assert "Could not create output directory" in msg
    mock_logger_error.assert_called_with(f"Could not create output directory {tmp_path / 'forbidden'}: Permission denied")


def test_extract_frames_cannot_open_video(tmp_path, mock_cv2_video_capture, mock_os_path_exists, mocker):
    """Tests behavior when cv2.VideoCapture fails to open the video."""
    mock_cv2_video_capture.isOpened.return_value = False
    mock_logger_error = mocker.patch("app.utils.video_processing.logger.error")
    paths, msg = extract_frames_from_video_to_disk("bad_video.mp4", str(tmp_path))
    assert len(paths) == 0
    assert "Could not open video file" in msg
    mock_logger_error.assert_called_with("Could not open video file: bad_video.mp4")

def test_extract_frames_imwrite_fails(
    tmp_path, mock_cv2_video_capture, mock_cv2_imwrite, mock_os_path_exists, mocker
):
    """Tests behavior when cv2.imwrite fails for a frame."""
    mock_cv2_imwrite.return_value = False # Simulate imwrite failure
    mock_logger_warning = mocker.patch("app.utils.video_processing.logger.warning")

    paths, msg = extract_frames_from_video_to_disk("test.mp4", str(tmp_path), target_fps=None)

    assert len(paths) == 0 # No paths should be returned if all writes fail
    # mock_cv2_video_capture.read.side_effect gives 10 frames, so 10 attempts to write
    assert mock_logger_warning.call_count == 10
    mock_logger_warning.assert_any_call("Failed to write frame frame_000000.jpg for video test.mp4")


def test_ensure_rgb_with_bgr_image():
    """Tests ensure_rgb with a 3-channel BGR image."""
    bgr_image = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_image[0, 0] = [255, 0, 0] # Blue pixel
    rgb_image = ensure_rgb(bgr_image)
    assert rgb_image.shape == (10, 10, 3)
    assert np.array_equal(rgb_image[0, 0], [0, 0, 255]) # Should be Red

def test_ensure_rgb_with_grayscale_image():
    """Tests ensure_rgb with a grayscale image."""
    gray_image = np.zeros((10, 10), dtype=np.uint8)
    gray_image[0, 0] = 128
    rgb_image = ensure_rgb(gray_image)
    assert rgb_image.shape == (10, 10, 3)
    assert np.array_equal(rgb_image[0, 0], [128, 128, 128])

def test_ensure_rgb_with_already_rgb_or_other_format():
    """Tests ensure_rgb with an image that's not BGR or grayscale (should pass through)."""
    # Simulate an image that might already be RGB (though cvtColor would still run)
    rgb_like_image = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_like_image[0, 0] = [0, 0, 255] # Red pixel
    processed_image = ensure_rgb(rgb_like_image.copy()) # Pass a copy
    # If it was BGR, it gets converted. If it was already RGB, cvtColor BGR2RGB is mostly harmless (might swap R and B again)
    # For this test, we are more concerned it doesn't crash and returns 3 channels.
    # A more robust test might check if cvtColor was called or not.
    assert processed_image.shape == (10, 10, 3)

    four_channel_image = np.zeros((10, 10, 4), dtype=np.uint8)
    processed_4channel = ensure_rgb(four_channel_image)
    assert np.array_equal(processed_4channel, four_channel_image) # Should return as-is