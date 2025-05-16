# tests/services/test_multi_camera_frame_processor.py
"""
Unit tests for MultiCameraFrameProcessor in app.services.multi_camera_frame_processor.
"""
import pytest
import uuid
import asyncio
import numpy as np
import torch
from unittest.mock import MagicMock, AsyncMock, PropertyMock, call, patch

from app.services.multi_camera_frame_processor import MultiCameraFrameProcessor
from app.models.base_models import AbstractDetector, Detection, BoundingBox
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.homography_service import HomographyService
from app.services.notification_service import NotificationService
from app.services.reid_components import ReIDStateManager
from app.common_types import (
    CameraID, TrackKey, GlobalID, FeatureVector, FrameBatch, FrameData, RawDetection,
    TrackedObjectData, BoundingBoxXYXY, TrackID,
    HandoffTriggerInfo, ExitRuleModel, ExitDirection, CameraHandoffDetailConfig
)
# mock_settings is available from conftest.py

@pytest.fixture
def mock_detector(mocker):
    """Mocks AbstractDetector."""
    mock = AsyncMock(spec=AbstractDetector)
    # Default behavior for detect: returns an empty list of Detection objects
    mock.detect.return_value = []
    return mock

@pytest.fixture
def mock_tracker_instance(mocker):
    """Mocks an AbstractTracker instance (returned by the factory)."""
    mock = AsyncMock() # More generic async mock
    # Default behavior for update: returns an empty numpy array (shape for no tracks)
    mock.update.return_value = np.empty((0, 8), dtype=np.float32) # BoxMOT with features might have 8+ cols
    return mock


@pytest.fixture
def mock_camera_tracker_factory(mocker, mock_tracker_instance):
    """Mocks CameraTrackerFactory."""
    mock = MagicMock(spec=CameraTrackerFactory)
    mock.get_tracker = AsyncMock(return_value=mock_tracker_instance)
    return mock

@pytest.fixture
def mock_homography_service(mocker):
    """Mocks HomographyService."""
    mock = MagicMock(spec=HomographyService)
    # Default: no homography matrix found
    mock.get_homography_matrix.return_value = None
    return mock

@pytest.fixture
def mock_notification_service(mocker):
    """Mocks NotificationService."""
    mock = MagicMock(spec=NotificationService)
    # NotificationService methods are async
    mock.send_tracking_update = AsyncMock()
    mock.send_status_update = AsyncMock()
    return mock

@pytest.fixture
def mock_reid_manager(mocker):
    """Mocks ReIDStateManager."""
    mock = MagicMock(spec=ReIDStateManager)
    mock.associate_features_and_update_state = AsyncMock()
    mock.track_to_global_id = {} # Mock the attribute directly
    return mock

@pytest.fixture
def multi_camera_processor_instance(
    mock_detector,
    mock_camera_tracker_factory,
    mock_homography_service,
    mock_notification_service, # Not directly used by MFProc, but good to have if needed
    mock_settings,
    mocker
):
    """Provides an instance of MultiCameraFrameProcessor with mocked dependencies."""
    # Patch settings in the multi_camera_frame_processor module
    mocker.patch("app.services.multi_camera_frame_processor.settings", mock_settings)
    test_device = torch.device("cpu")
    return MultiCameraFrameProcessor(
        detector=mock_detector,
        tracker_factory=mock_camera_tracker_factory,
        homography_service=mock_homography_service,
        notification_service=mock_notification_service, # Passed but not used by MFProc directly
        device=test_device
    )


def test_multi_camera_processor_init(multi_camera_processor_instance: MultiCameraFrameProcessor, mock_detector):
    """Tests MultiCameraFrameProcessor initialization."""
    assert multi_camera_processor_instance.detector == mock_detector
    # Add other assertions for initialized members if any

def test_project_point_to_map_with_matrix(multi_camera_processor_instance: MultiCameraFrameProcessor, mocker):
    """Tests _project_point_to_map with a valid homography matrix."""
    mock_cv2_transform = mocker.patch("cv2.perspectiveTransform")
    # Mock to return a transformed point: (1,1,2) shape
    mock_cv2_transform.return_value = np.array([[[100.0, 200.0]]], dtype=np.float32)
    
    matrix = np.eye(3, dtype=np.float32)
    point_xy = (10.0, 20.0)
    projected = multi_camera_processor_instance._project_point_to_map(point_xy, matrix)
    
    assert projected == [100.0, 200.0]
    # Check that cv2.perspectiveTransform was called correctly
    args, _ = mock_cv2_transform.call_args
    assert np.array_equal(args[0], np.array([[point_xy]], dtype=np.float32))
    assert np.array_equal(args[1], matrix)

def test_project_point_to_map_no_matrix(multi_camera_processor_instance: MultiCameraFrameProcessor):
    """Tests _project_point_to_map when homography matrix is None."""
    projected = multi_camera_processor_instance._project_point_to_map((10.0, 20.0), None)
    assert projected is None

def test_parse_raw_tracker_output_valid(multi_camera_processor_instance: MultiCameraFrameProcessor):
    """Tests _parse_raw_tracker_output with valid BoxMOT-like output."""
    task_id = uuid.uuid4()
    cam_id = CameraID("c1")
    # x1, y1, x2, y2, track_id, conf, cls_id, feat1, feat2 ...
    raw_output = np.array([
        [10, 10, 20, 20, 1, 0.9, 0, 0.1, 0.2], # Track 1 with features
        [30, 30, 40, 40, 2, 0.8, 0],          # Track 2 without features (cols < 7)
        [50, 50, 60, 60, 3, 0.7, 1, 0.3, 0.4, 0.5] # Track 3 with features
    ], dtype=np.float32)

    parsed = multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, raw_output)
    
    assert len(parsed) == 3
    tk1, bbox1, feat1 = parsed[0]
    assert tk1 == (cam_id, TrackID(1))
    assert bbox1 == [10.0, 10.0, 20.0, 20.0]
    assert feat1 is not None and np.array_equal(feat1, np.array([0.1, 0.2], dtype=np.float32))

    tk2, bbox2, feat2 = parsed[1] # Track 2 had less than 7 columns
    assert tk2 == (cam_id, TrackID(2))
    assert feat2 is None # Should be None as num_cols < 7

    tk3, bbox3, feat3 = parsed[2]
    assert tk3 == (cam_id, TrackID(3))
    assert feat3 is not None and np.array_equal(feat3, np.array([0.3, 0.4, 0.5], dtype=np.float32))


def test_parse_raw_tracker_output_empty_or_malformed(multi_camera_processor_instance: MultiCameraFrameProcessor):
    """Tests _parse_raw_tracker_output with empty or malformed input."""
    task_id = uuid.uuid4()
    cam_id = CameraID("c1")
    assert multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, np.empty((0,8))) == []
    assert multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, None) == [] # type: ignore
    # Malformed (too few columns)
    malformed = np.array([[10,10,20,20]], dtype=np.float32) # Only 4 columns
    parsed_malformed = multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, malformed)
    assert parsed_malformed == []

@pytest.mark.asyncio
async def test_process_frame_batch_basic_flow(
    multi_camera_processor_instance: MultiCameraFrameProcessor,
    mock_detector: AsyncMock,
    mock_camera_tracker_factory: MagicMock,
    mock_tracker_instance: AsyncMock, # The instance returned by factory.get_tracker
    mock_homography_service: MagicMock,
    mock_reid_manager: MagicMock,
    mock_settings, # For PERSON_CLASS_ID
    mocker
):
    """
    Tests the basic processing flow of a frame batch:
    detect -> track -> parse -> reid_associate -> project_map -> format_output.
    """
    task_id = uuid.uuid4()
    env_id = "test_env"
    frame_count = 0

    cam1_id = CameraID("c01")
    cam1_frame_np = np.zeros((100, 100, 3), dtype=np.uint8) # H, W, C
    cam1_frame_path = "c01/frame_000.jpg"
    frame_batch_input: FrameBatch = {cam1_id: (cam1_frame_np, cam1_frame_path)}

    # --- Mock Detector Output ---
    # Detector returns one Detection object
    mock_detection_obj = Detection(
        bbox=BoundingBox(x1=10, y1=10, x2=20, y2=20),
        confidence=0.9,
        class_id=mock_settings.PERSON_CLASS_ID,
        class_name="person"
    )
    mock_detector.detect.return_value = [mock_detection_obj]

    # --- Mock Tracker Output ---
    # Tracker output for the one detection, with features
    tracker_output_np = np.array([[10, 10, 20, 20, 1, 0.9, mock_settings.PERSON_CLASS_ID, 0.1, 0.2, 0.3]], dtype=np.float32)
    mock_tracker_instance.update.return_value = tracker_output_np

    # --- Mock Homography Service Output ---
    mock_homography_matrix = np.eye(3, dtype=np.float32)
    mock_homography_service.get_homography_matrix.return_value = mock_homography_matrix
    # Mock the actual projection calculation (perspectiveTransform)
    mocker.patch("cv2.perspectiveTransform", return_value=np.array([[[50.0, 75.0]]], dtype=np.float32))


    # --- Mock ReID Manager Output (what it stores after association) ---
    gid_assigned = GlobalID("global_person_1")
    # Simulate that reid_manager will assign this GID to the track_key
    def mock_associate_side_effect(*args, **kwargs):
        # args[0] is features_for_reid_input (Dict[TrackKey, FeatureVector])
        # Simulate that the track key (cam1_id, TrackID(1)) gets gid_assigned
        tk_from_tracker = (cam1_id, TrackID(1)) # Based on tracker_output_np track_id
        mock_reid_manager.track_to_global_id[tk_from_tracker] = gid_assigned
        return None
    mock_reid_manager.associate_features_and_update_state.side_effect = mock_associate_side_effect


    # --- Call the method under test ---
    batch_results = await multi_camera_processor_instance.process_frame_batch(
        task_id, env_id, mock_reid_manager, frame_batch_input, frame_count
    )

    # --- Assertions ---
    mock_detector.detect.assert_called_once_with(cam1_frame_np)
    mock_camera_tracker_factory.get_tracker.assert_called_once_with(task_id, cam1_id)
    
    # Check args to tracker_instance.update
    # Expected detections_np: [[10.0, 10.0, 20.0, 20.0, 0.9, mock_settings.PERSON_CLASS_ID]]
    expected_dets_for_tracker = np.array([[10.0, 10.0, 20.0, 20.0, 0.9, float(mock_settings.PERSON_CLASS_ID)]], dtype=np.float32)
    call_args_tracker_update = mock_tracker_instance.update.call_args[0]
    assert np.array_equal(call_args_tracker_update[0], expected_dets_for_tracker)
    assert np.array_equal(call_args_tracker_update[1], cam1_frame_np)
    
    # Check args to reid_manager.associate_features_and_update_state
    # Expected features_for_reid_input: {(cam1_id, TrackID(1)): FeatureVector([0.1,0.2,0.3])}
    # Expected active_track_keys: {(cam1_id, TrackID(1))}
    call_args_reid = mock_reid_manager.associate_features_and_update_state.call_args[0]
    expected_feat_vec = FeatureVector(np.array([0.1,0.2,0.3], dtype=np.float32))
    expected_track_key = (cam1_id, TrackID(1))
    assert expected_track_key in call_args_reid[0] # features_for_reid_input
    assert np.array_equal(call_args_reid[0][expected_track_key], expected_feat_vec)
    assert call_args_reid[1] == {expected_track_key} # active_track_keys_this_batch_set
    assert call_args_reid[3] == frame_count # processed_frame_count
    
    mock_homography_service.get_homography_matrix.assert_called_once_with(env_id, cam1_id)
    # cv2.perspectiveTransform was called by _project_point_to_map
    # The footpoint for bbox [10,10,20,20] is x=(10+20)/2=15, y=20
    mocker.patch("cv2.perspectiveTransform").assert_called_once() # Already mocked above
    persp_trans_args = mocker.patch("cv2.perspectiveTransform").call_args[0]
    assert np.array_equal(persp_trans_args[0], np.array([[(15.0, 20.0)]], dtype=np.float32))


    # Check final batch_results structure
    assert cam1_id in batch_results
    assert len(batch_results[cam1_id]) == 1
    tracked_obj: TrackedObjectData = batch_results[cam1_id][0]
    
    assert tracked_obj.camera_id == cam1_id
    assert tracked_obj.track_id == TrackID(1)
    assert tracked_obj.global_person_id == gid_assigned
    assert tracked_obj.bbox_xyxy == [10.0, 10.0, 20.0, 20.0]
    assert tracked_obj.confidence == 0.9 # From tracker output
    assert tracked_obj.feature_vector == [0.1, 0.2, 0.3]
    assert tracked_obj.map_coords == [50.0, 75.0] # From mocked perspectiveTransform


@pytest.mark.asyncio
async def test_process_frame_batch_no_detections(
    multi_camera_processor_instance: MultiCameraFrameProcessor,
    mock_detector: AsyncMock,
    mock_tracker_instance: AsyncMock,
    mock_reid_manager: MagicMock,
):
    """Tests processing a batch where the detector finds nothing."""
    task_id = uuid.uuid4()
    env_id = "test_env"
    frame_count = 1
    cam1_id = CameraID("c01")
    cam1_frame_np = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_batch_input: FrameBatch = {cam1_id: (cam1_frame_np, "path")}

    mock_detector.detect.return_value = [] # No detections

    batch_results = await multi_camera_processor_instance.process_frame_batch(
        task_id, env_id, mock_reid_manager, frame_batch_input, frame_count
    )

    mock_detector.detect.assert_called_once()
    # Tracker should be called with empty detections
    expected_empty_dets_for_tracker = np.empty((0,6), dtype=np.float32)
    call_args_tracker_update = mock_tracker_instance.update.call_args[0]
    assert np.array_equal(call_args_tracker_update[0], expected_empty_dets_for_tracker)

    # ReID manager should still be called, but with empty inputs for features
    # but potentially non-empty for active_track_keys if tracker maintained old tracks
    # In this case, tracker also returns no tracks.
    mock_reid_manager.associate_features_and_update_state.assert_called_once()
    call_args_reid = mock_reid_manager.associate_features_and_update_state.call_args[0]
    assert call_args_reid[0] == {} # Empty features_for_reid_input
    assert call_args_reid[1] == set() # Empty active_track_keys_this_batch_set
    
    assert cam1_id not in batch_results or len(batch_results[cam1_id]) == 0

# TODO: Add more tests for MultiCameraFrameProcessor:
# - Multiple cameras in a batch
# - Handoff trigger logic being called and its results influencing ReID (needs more complex ReID mock)
# - Error handling if detector or tracker fails for one camera in a multi-camera batch
# - _check_handoff_triggers_for_camera in more detail (this is complex itself)

</rewritten_file> 