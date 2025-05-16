# tests/services/test_video_data_manager_service.py
"""
Unit tests for VideoDataManagerService and BatchedFrameProvider
in app.services.video_data_manager_service.
"""
import pytest
import uuid
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, PropertyMock, call, patch

from app.services.video_data_manager_service import VideoDataManagerService, BatchedFrameProvider
from app.utils.asset_downloader import AssetDownloader
from app.core.config import VideoSetEnvironmentConfig
from app.common_types import CameraID
# mock_settings is available from conftest.py

@pytest.fixture
def mock_asset_downloader(mocker):
    """Mocks the AssetDownloader."""
    mock_downloader = MagicMock(spec=AssetDownloader)
    mock_downloader.download_file_from_dagshub = AsyncMock(return_value=True) # Default success
    return mock_downloader

@pytest.fixture
def video_data_manager_service_instance(mock_asset_downloader, mock_settings, mocker):
    """Provides an instance of VideoDataManagerService."""
    mocker.patch("app.services.video_data_manager_service.settings", mock_settings)
    return VideoDataManagerService(asset_downloader=mock_asset_downloader)

# --- VideoDataManagerService Tests ---

def test_video_data_manager_init(video_data_manager_service_instance: VideoDataManagerService, mock_asset_downloader, mock_settings):
    """Tests VideoDataManagerService initialization."""
    assert video_data_manager_service_instance.asset_downloader == mock_asset_downloader
    assert video_data_manager_service_instance.video_sets_config == mock_settings.VIDEO_SETS
    assert video_data_manager_service_instance.local_video_dir_base == Path(mock_settings.LOCAL_VIDEO_DOWNLOAD_DIR)

@pytest.mark.asyncio
async def test_download_sub_videos_successful(
    video_data_manager_service_instance: VideoDataManagerService,
    mock_asset_downloader: MagicMock, # Type hint for clarity
    mock_settings, # To access VIDEO_SETS and LOCAL_VIDEO_DOWNLOAD_DIR
    tmp_path # Pytest fixture for temp directory
):
    """Tests successful download of sub-videos for an environment batch."""
    task_id = uuid.uuid4()
    env_id = "test_env" # Matches mock_settings from conftest
    sub_video_idx = 0 # First sub-video

    # Override LOCAL_VIDEO_DOWNLOAD_DIR to use tmp_path for this test
    # This ensures cleanup and isolation. The conftest cleanup_test_dirs will handle the original mock one.
    original_local_dir = video_data_manager_service_instance.local_video_dir_base
    video_data_manager_service_instance.local_video_dir_base = tmp_path / "test_downloads"


    downloaded_paths = await video_data_manager_service_instance.download_sub_videos_for_environment_batch(
        task_id, env_id, sub_video_idx
    )

    # From mock_settings, test_env has c01 and c02
    assert len(downloaded_paths) == 2
    assert CameraID("c01") in downloaded_paths
    assert CameraID("c02") in downloaded_paths

    expected_calls = []
    for cam_config in mock_settings.VIDEO_SETS:
        if cam_config.env_id == env_id:
            video_filename = cam_config.sub_video_filename_pattern.format(idx=sub_video_idx + 1)
            remote_key = f"{cam_config.remote_base_key}/{video_filename}"
            
            local_path = tmp_path / "test_downloads" / str(task_id) / env_id / cam_config.cam_id / video_filename
            assert downloaded_paths[CameraID(cam_config.cam_id)] == local_path
            assert local_path.parent.exists()
            expected_calls.append(call(remote_s3_key=remote_key, local_destination_path=str(local_path)))
    
    mock_asset_downloader.download_file_from_dagshub.assert_has_calls(expected_calls, any_order=True)
    
    # Restore original path if it matters for other tests using the same instance (though fixtures usually recreate)
    video_data_manager_service_instance.local_video_dir_base = original_local_dir


@pytest.mark.asyncio
async def test_download_sub_videos_download_failure(
    video_data_manager_service_instance: VideoDataManagerService,
    mock_asset_downloader: MagicMock,
    tmp_path
):
    """Tests download failure for one of the videos."""
    task_id = uuid.uuid4()
    env_id = "test_env"
    sub_video_idx = 0
    video_data_manager_service_instance.local_video_dir_base = tmp_path / "test_downloads"


    # c01 succeeds, c02 fails
    async def mock_download_side_effect(remote_s3_key, local_destination_path):
        if "c02" in remote_s3_key:
            return False # Simulate failure for c02
        return True
    mock_asset_downloader.download_file_from_dagshub.side_effect = mock_download_side_effect

    downloaded_paths = await video_data_manager_service_instance.download_sub_videos_for_environment_batch(
        task_id, env_id, sub_video_idx
    )
    assert len(downloaded_paths) == 1 # Only c01 should be in paths
    assert CameraID("c01") in downloaded_paths
    assert CameraID("c02") not in downloaded_paths

@pytest.mark.asyncio
async def test_download_sub_videos_index_out_of_bounds(video_data_manager_service_instance: VideoDataManagerService, mock_asset_downloader: MagicMock, tmp_path):
    """Tests requesting a sub-video index that is too high."""
    task_id = uuid.uuid4()
    env_id = "test_env" # c01 and c02 have num_sub_videos = 1
    sub_video_idx = 1 # This is out of bounds (0-indexed)
    video_data_manager_service_instance.local_video_dir_base = tmp_path / "test_downloads"

    downloaded_paths = await video_data_manager_service_instance.download_sub_videos_for_environment_batch(
        task_id, env_id, sub_video_idx
    )
    assert len(downloaded_paths) == 0
    mock_asset_downloader.download_file_from_dagshub.assert_not_called()


def test_get_batched_frame_provider(video_data_manager_service_instance: VideoDataManagerService, mock_settings):
    """Tests creation of BatchedFrameProvider."""
    task_id = uuid.uuid4()
    local_paths = {CameraID("c01"): Path("vid1.mp4")}
    
    with patch("app.services.video_data_manager_service.BatchedFrameProvider") as MockProvider:
        provider_instance = video_data_manager_service_instance.get_batched_frame_provider(task_id, local_paths)
        MockProvider.assert_called_once_with(
            task_id=task_id,
            video_paths_map=local_paths,
            target_fps=mock_settings.TARGET_FPS,
            jpeg_quality=mock_settings.FRAME_JPEG_QUALITY,
            loop_videos=False # Default
        )
        assert provider_instance == MockProvider.return_value


@pytest.mark.asyncio
async def test_cleanup_task_data_dir_exists(video_data_manager_service_instance: VideoDataManagerService, tmp_path, mocker):
    """Tests cleanup of task data when the directory exists."""
    task_id = uuid.uuid4()
    task_dir = tmp_path / str(task_id)
    task_dir.mkdir()
    video_data_manager_service_instance.local_video_dir_base = tmp_path # Point to parent of task_id dir

    mock_shutil_rmtree = mocker.patch("shutil.rmtree")
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args: func(*args))


    await video_data_manager_service_instance.cleanup_task_data(task_id)
    mock_shutil_rmtree.assert_called_once_with(task_dir)

@pytest.mark.asyncio
async def test_cleanup_task_data_dir_not_exists(video_data_manager_service_instance: VideoDataManagerService, tmp_path, mocker):
    """Tests cleanup when task directory does not exist."""
    task_id = uuid.uuid4()
    video_data_manager_service_instance.local_video_dir_base = tmp_path
    mock_shutil_rmtree = mocker.patch("shutil.rmtree")

    await video_data_manager_service_instance.cleanup_task_data(task_id)
    mock_shutil_rmtree.assert_not_called()


def test_get_max_sub_videos_for_environment(video_data_manager_service_instance: VideoDataManagerService, mock_settings):
    """Tests calculation of max sub-videos."""
    # mock_settings.VIDEO_SETS has test_env with c01 (1 sub_video) and c02 (1 sub_video)
    assert video_data_manager_service_instance.get_max_sub_videos_for_environment("test_env") == 1
    
    # Add a more complex scenario to mock_settings for this test or create a new service instance
    mock_settings.VIDEO_SETS.append(
        VideoSetEnvironmentConfig(remote_base_key="v/c3", env_id="test_env_complex", cam_id="c03", num_sub_videos=5)
    )
    mock_settings.VIDEO_SETS.append(
        VideoSetEnvironmentConfig(remote_base_key="v/c4", env_id="test_env_complex", cam_id="c04", num_sub_videos=3)
    )
    assert video_data_manager_service_instance.get_max_sub_videos_for_environment("test_env_complex") == 5
    assert video_data_manager_service_instance.get_max_sub_videos_for_environment("non_existent_env") == 0
    # Clean up added settings if they persist due to fixture scope
    mock_settings.VIDEO_SETS = mock_settings.VIDEO_SETS[:-2]


# --- BatchedFrameProvider Tests ---
@pytest.fixture
def mock_cv2_video_capture(mocker):
    """Mocks cv2.VideoCapture for BatchedFrameProvider tests."""
    mock_cap_instance = MagicMock(spec=cv2.VideoCapture)
    # Default to opened
    mock_cap_instance.isOpened.return_value = True
    # Default FPS and frame count
    mock_cap_instance.get.side_effect = lambda prop_id: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
    }.get(prop_id, 0)
    # Simulate reading a few frames then ending
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    read_effects = [(True, dummy_frame)] * 3 + [(False, None)] # 3 good frames, then end
    mock_cap_instance.read.side_effect = read_effects
    mock_cap_instance.release = MagicMock()

    mocker.patch("cv2.VideoCapture", return_value=mock_cap_instance)
    return mock_cap_instance

@pytest.fixture
def video_files_for_provider(tmp_path):
    """Creates dummy video files for BatchedFrameProvider tests."""
    vid_dir = tmp_path / "provider_vids"
    vid_dir.mkdir()
    vid1_path = vid_dir / "cam1_video.mp4"
    vid2_path = vid_dir / "cam2_video.mp4"
    vid1_path.touch() # Create empty files, as cv2.VideoCapture is mocked
    vid2_path.touch()
    return {CameraID("cam1"): vid1_path, CameraID("cam2"): vid2_path}


def test_batched_frame_provider_init_and_open(video_files_for_provider, mock_cv2_video_capture, mock_settings):
    """Tests BatchedFrameProvider initialization and video opening."""
    task_id = uuid.uuid4()
    target_fps = 10 # 30fps source / 10fps target = skip interval 3

    provider = BatchedFrameProvider(task_id, video_files_for_provider, target_fps, 90)

    assert provider._is_open
    assert len(provider.video_captures) == 2
    assert CameraID("cam1") in provider.frame_skip_intervals
    assert provider.frame_skip_intervals[CameraID("cam1")] == 3 # 30/10
    mock_cv2_video_capture.assert_any_call(str(video_files_for_provider[CameraID("cam1")]))
    mock_cv2_video_capture.assert_any_call(str(video_files_for_provider[CameraID("cam2")]))
    provider.close() # Cleanup

@pytest.mark.asyncio
async def test_batched_frame_provider_get_next_batch(video_files_for_provider, mock_cv2_video_capture, mocker):
    """Tests getting synchronized frame batches."""
    task_id = uuid.uuid4()
    target_fps = 30 # Source 30fps / target 30fps = skip interval 1
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args: func(*args))


    # Configure mock_cv2_video_capture for 2 good frames for each camera
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    read_effects_cam1 = [(True, dummy_frame)] * 2 + [(False, None)]
    read_effects_cam2 = [(True, dummy_frame)] * 2 + [(False, None)]
    
    # Need to make VideoCapture return different mocks for different files
    mock_caps = {}
    def cap_side_effect(path_str):
        path_obj = Path(path_str)
        if path_obj not in mock_caps:
            new_mock_cap = MagicMock(spec=cv2.VideoCapture)
            new_mock_cap.isOpened.return_value = True
            new_mock_cap.get.return_value = 30.0 # FPS
            new_mock_cap.release = MagicMock()
            if "cam1" in path_str:
                new_mock_cap.read.side_effect = read_effects_cam1
            else: # cam2
                new_mock_cap.read.side_effect = read_effects_cam2
            mock_caps[path_obj] = new_mock_cap
        return mock_caps[path_obj]
    
    mocker.patch("cv2.VideoCapture", side_effect=cap_side_effect)


    provider = BatchedFrameProvider(task_id, video_files_for_provider, target_fps, 90)

    # Batch 1
    batch1, active1 = await provider.get_next_frame_batch()
    assert active1 is True
    assert CameraID("cam1") in batch1 and batch1[CameraID("cam1")] is not None
    assert CameraID("cam2") in batch1 and batch1[CameraID("cam2")] is not None
    assert batch1[CameraID("cam1")][1].endswith("frame_000001.jpg") # Check pseudo path

    # Batch 2
    batch2, active2 = await provider.get_next_frame_batch()
    assert active2 is True
    assert CameraID("cam1") in batch2 and batch2[CameraID("cam1")] is not None
    assert CameraID("cam2") in batch2 and batch2[CameraID("cam2")] is not None
    assert batch2[CameraID("cam1")][1].endswith("frame_000002.jpg")


    # Batch 3 (all videos should have ended)
    batch3, active3 = await provider.get_next_frame_batch()
    assert active3 is False
    assert CameraID("cam1") in batch3 and batch3[CameraID("cam1")] is None
    assert CameraID("cam2") in batch3 and batch3[CameraID("cam2")] is None
    
    provider.close()


def test_batched_frame_provider_video_file_not_found(tmp_path, mock_cv2_video_capture, mocker):
    """Tests BatchedFrameProvider when a video file does not exist."""
    task_id = uuid.uuid4()
    non_existent_path = tmp_path / "no_video.mp4" # Does not call .touch()
    video_map = {CameraID("cam_missing"): non_existent_path}
    mock_logger_error = mocker.patch("app.services.video_data_manager_service.logger.error")

    provider = BatchedFrameProvider(task_id, video_map, 30, 90)
    assert not provider._is_open # Should not fully open if a video fails critically
    assert len(provider.video_captures) == 0
    mock_logger_error.assert_any_call(f"[Task {task_id}][cam_missing] Video file not found: {non_existent_path}. Will skip this camera.")
    provider.close()


def test_batched_frame_provider_cannot_open_video(video_files_for_provider, mock_cv2_video_capture, mocker):
    """Tests BatchedFrameProvider when cv2.VideoCapture fails to open a file."""
    task_id = uuid.uuid4()
    mock_cv2_video_capture.isOpened.return_value = False # Simulate failure to open
    mock_logger_error = mocker.patch("app.services.video_data_manager_service.logger.error")

    provider = BatchedFrameProvider(task_id, video_files_for_provider, 30, 90)
    assert not provider._is_open
    assert len(provider.video_captures) == 0
    path_cam1 = video_files_for_provider[CameraID("cam1")]
    mock_logger_error.assert_any_call(
        f"[Task {task_id}][cam1] Could not open video: {path_cam1}. Will skip this camera."
    )
    provider.close() 