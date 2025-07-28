"""
Global fixtures for the SpotOn backend test suite.
"""
import pytest
from unittest.mock import MagicMock, PropertyMock
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set, Optional
import math

from app.core.config import Settings, VideoSetEnvironmentConfig, CameraHandoffDetailConfig
from app.shared.types import CameraID, ExitRuleModel, QuadrantName

@pytest.fixture(scope="session")
def mock_settings_base_values() -> Dict[str, Any]:
    """
    Provides a dictionary of base values for a mocked Settings object.
    Tests can override these by providing their own dictionary to mock_settings.
    """
    base_cosine_sim_thresh = 0.6 # For deriving L2
    return {
        "APP_NAME": "SpotOn Test Backend",
        "API_V1_PREFIX": "/api/v1/test",
        "DEBUG": True,
        "S3_ENDPOINT_URL": "http://localhost:9000/s3-test-bucket",
        "AWS_ACCESS_KEY_ID": "test_access_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret_key",
        "S3_BUCKET_NAME": "test-spoton-bucket",
        "DAGSHUB_REPO_OWNER": "test_owner",
        "DAGSHUB_REPO_NAME": "test_repo",
        "LOCAL_VIDEO_DOWNLOAD_DIR": "./test_downloaded_videos",
        "LOCAL_FRAME_EXTRACTION_DIR": "./test_extracted_frames",
        "VIDEO_SETS": [
            VideoSetEnvironmentConfig(remote_base_key="video_test/c01", env_id="test_env", cam_id="c01", num_sub_videos=1),
            VideoSetEnvironmentConfig(remote_base_key="video_test/c02", env_id="test_env", cam_id="c02", num_sub_videos=1),
        ],
        "CAMERA_HANDOFF_DETAILS": {
            ("test_env", "c01"): CameraHandoffDetailConfig(
                exit_rules=[
                    ExitRuleModel(source_exit_quadrant=QuadrantName("upper_right"), target_cam_id=CameraID("c02"), target_entry_area="left_side")
                ],
                homography_matrix_path="homography_points_c01_scene_test.npz"
            ),
            ("test_env", "c02"): CameraHandoffDetailConfig(
                exit_rules=[],
                homography_matrix_path="homography_points_c02_scene_test.npz"
            ),
        },
        "MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT": 0.35,
        "HOMOGRAPHY_DATA_DIR": "./test_homography_data",
        "POSSIBLE_CAMERA_OVERLAPS": [("c01", "c02")],
        "REDIS_HOST": "localhost",
        "REDIS_PORT": 6379,
        "REDIS_DB": 1, 
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "POSTGRES_SERVER": "localhost",
        "POSTGRES_PORT": 5433, 
        "POSTGRES_DB": "test_spotondb",
        "DETECTOR_TYPE": "fasterrcnn",
        "PERSON_CLASS_ID": 1,
        "DETECTION_CONFIDENCE_THRESHOLD": 0.5,
        "DETECTION_USE_AMP": False,
        "TRACKER_TYPE": "botsort",
        "WEIGHTS_DIR": "./test_weights",
        "REID_WEIGHTS_PATH": "test_clip_model.pt",
        "TRACKER_HALF_PRECISION": False,
        "TRACKER_PER_CLASS": False,
        "REID_MODEL_TYPE": "clip",
        "REID_MODEL_HALF_PRECISION": False,
        
        # --- MODIFIED/ADDED Re-ID Settings for Tests ---
        "REID_SIMILARITY_METHOD": "faiss_l2", # Default for tests
        "REID_SIMILARITY_THRESHOLD": base_cosine_sim_thresh, 
        "REID_L2_DISTANCE_THRESHOLD": None, # Let it be derived by default
        # --- END MODIFIED/ADDED ---
        
        "REID_GALLERY_EMA_ALPHA": 0.85,
        "REID_REFRESH_INTERVAL_FRAMES": 15,
        "REID_LOST_TRACK_BUFFER_FRAMES": 150,
        "REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES": 400,
        "REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES": 300,
        "TARGET_FPS": 10,
        "FRAME_JPEG_QUALITY": 90,
        "resolved_reid_weights_path": Path("./test_weights/test_clip_model.pt"),
        "resolved_homography_base_path": Path("./test_homography_data"),
        "normalized_possible_camera_overlaps": {tuple(sorted((CameraID("c01"), CameraID("c02"))))},
        # Add the derived_l2_distance_threshold for the mock property
        "derived_l2_distance_threshold": math.sqrt(max(0, 2 * (1 - base_cosine_sim_thresh))),
    }

@pytest.fixture
def mock_settings(mocker, mock_settings_base_values: Dict[str, Any]) -> MagicMock:
    """
    Provides a MagicMock instance of the application Settings.
    """
    mocked_settings = MagicMock(spec=Settings)

    for key, value in mock_settings_base_values.items():
        # For properties, mock them on the type of the mock
        if key in ["resolved_reid_weights_path", 
                   "resolved_homography_base_path", 
                   "normalized_possible_camera_overlaps",
                   "derived_l2_distance_threshold"]: # MODIFIED: Add new property
            prop_mock = PropertyMock(return_value=value)
            setattr(type(mocked_settings), key, prop_mock) 
        else:
            setattr(mocked_settings, key, value)

    mocked_settings.model_config = mock_settings_base_values.get("model_config", {"extra": "ignore"})
    
    return mocked_settings

# manage_test_dirs fixture remains the same
@pytest.fixture(scope="session", autouse=True)
def manage_test_dirs(mock_settings_base_values: Dict[str, Any]):
    import shutil
    import os
    project_root = Path(__file__).parent.parent 
    dirs_to_manage_relative = [
        mock_settings_base_values["LOCAL_VIDEO_DOWNLOAD_DIR"],
        mock_settings_base_values["LOCAL_FRAME_EXTRACTION_DIR"],
        mock_settings_base_values["WEIGHTS_DIR"],
        mock_settings_base_values["HOMOGRAPHY_DATA_DIR"],
    ]
    absolute_dirs_to_manage = [project_root / Path(d) for d in dirs_to_manage_relative]
    for test_dir_path in absolute_dirs_to_manage:
        try: os.makedirs(test_dir_path, exist_ok=True)
        except OSError as e: print(f"Warning: Could not create test directory {test_dir_path}: {e}")
    yield 
    for test_dir_path in absolute_dirs_to_manage:
        if test_dir_path.exists():
            try: shutil.rmtree(test_dir_path)
            except OSError as e: print(f"Warning: Error cleaning up test directory {test_dir_path}: {e}")