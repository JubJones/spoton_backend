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
    mock.detect.return_value = []
    return mock

@pytest.fixture
def mock_tracker_instance(mocker):
    """Mocks an AbstractTracker instance (returned by the factory)."""
    mock = AsyncMock() 
    mock.update.return_value = np.empty((0, 8), dtype=np.float32) 
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
    mock.get_homography_matrix.return_value = None
    return mock

@pytest.fixture
def mock_notification_service(mocker):
    """Mocks NotificationService."""
    mock = MagicMock(spec=NotificationService)
    mock.send_tracking_update = AsyncMock()
    mock.send_status_update = AsyncMock()
    return mock

@pytest.fixture
def mock_reid_manager(mocker):
    """Mocks ReIDStateManager."""
    mock = MagicMock(spec=ReIDStateManager)
    mock.associate_features_and_update_state = AsyncMock()
    mock.track_to_global_id = {} 
    return mock

@pytest.fixture
def multi_camera_processor_instance(
    mock_detector,
    mock_camera_tracker_factory,
    mock_homography_service,
    mock_notification_service, 
    mock_settings,
    mocker
):
    """Provides an instance of MultiCameraFrameProcessor with mocked dependencies."""
    mocker.patch("app.services.multi_camera_frame_processor.settings", mock_settings)
    test_device = torch.device("cpu")
    return MultiCameraFrameProcessor(
        detector=mock_detector,
        tracker_factory=mock_camera_tracker_factory,
        homography_service=mock_homography_service,
        notification_service=mock_notification_service, 
        device=test_device
    )


def test_multi_camera_processor_init(multi_camera_processor_instance: MultiCameraFrameProcessor, mock_detector):
    """Tests MultiCameraFrameProcessor initialization."""
    assert multi_camera_processor_instance.detector == mock_detector

def test_project_point_to_map_with_matrix(multi_camera_processor_instance: MultiCameraFrameProcessor, mocker):
    """Tests _project_point_to_map with a valid homography matrix."""
    mock_cv2_transform = mocker.patch("app.services.multi_camera_frame_processor.cv2.perspectiveTransform")
    mock_cv2_transform.return_value = np.array([[[100.0, 200.0]]], dtype=np.float32)
    
    matrix = np.eye(3, dtype=np.float32)
    point_xy = (10.0, 20.0)
    projected = multi_camera_processor_instance._project_point_to_map(point_xy, matrix)
    
    assert projected == [100.0, 200.0]
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
    
    raw_output_with_features = np.array([
        [10, 10, 20, 20, 1, 0.9, 0, 0.1, 0.2, 0.3], 
        [50, 50, 60, 60, 3, 0.7, 1, 0.3, 0.4, 0.5] 
    ], dtype=np.float32)

    parsed = multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, raw_output_with_features)
    assert len(parsed) == 2
    tk1, bbox1, feat1 = parsed[0]
    assert tk1 == (cam_id, TrackID(1))
    assert bbox1 == [10.0, 10.0, 20.0, 20.0]
    assert feat1 is not None and np.array_equal(feat1, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    raw_output_no_features = np.array([ 
        [30, 30, 40, 40, 2, 0.8, 0], 
        [70, 70, 80, 80, 4, 0.6, 1],
    ], dtype=np.float32)
    parsed_no_feat = multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, raw_output_no_features)
    assert len(parsed_no_feat) == 2
    tk2, _, feat2 = parsed_no_feat[0]
    assert tk2 == (cam_id, TrackID(2))
    assert feat2 is None


def test_parse_raw_tracker_output_empty_or_malformed(multi_camera_processor_instance: MultiCameraFrameProcessor):
    """Tests _parse_raw_tracker_output with empty or malformed input."""
    task_id = uuid.uuid4()
    cam_id = CameraID("c1")
    assert multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, np.empty((0,8))) == []
    assert multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, None) == [] # type: ignore
    malformed = np.array([[10,10,20,20]], dtype=np.float32) 
    parsed_malformed = multi_camera_processor_instance._parse_raw_tracker_output(task_id, cam_id, malformed)
    assert parsed_malformed == []

@pytest.mark.asyncio
async def test_process_frame_batch_basic_flow(
    multi_camera_processor_instance: MultiCameraFrameProcessor,
    mock_detector: AsyncMock,
    mock_camera_tracker_factory: MagicMock,
    mock_tracker_instance: AsyncMock, 
    mock_homography_service: MagicMock,
    mock_reid_manager: MagicMock,
    mock_settings, 
    mocker
):
    task_id = uuid.uuid4()
    env_id = "test_env"
    frame_count = 0

    cam1_id = CameraID("c01")
    cam1_frame_np = np.zeros((100, 100, 3), dtype=np.uint8) 
    cam1_frame_path = "c01/frame_000.jpg"
    frame_batch_input: FrameBatch = {cam1_id: (cam1_frame_np, cam1_frame_path)}

    mock_detection_obj = Detection(
        bbox=BoundingBox(x1=10, y1=10, x2=20, y2=20),
        confidence=0.9,
        class_id=mock_settings.PERSON_CLASS_ID,
        class_name="person"
    )
    mock_detector.detect.return_value = [mock_detection_obj]

    tracker_output_np = np.array([[10, 10, 20, 20, 1, 0.9, mock_settings.PERSON_CLASS_ID, 0.1, 0.2, 0.3]], dtype=np.float32)
    mock_tracker_instance.update.return_value = tracker_output_np

    mock_homography_matrix = np.eye(3, dtype=np.float32)
    mock_homography_service.get_homography_matrix.return_value = mock_homography_matrix
    
    mock_cv2_perspective_transform = mocker.patch("app.services.multi_camera_frame_processor.cv2.perspectiveTransform", return_value=np.array([[[50.0, 75.0]]], dtype=np.float32))


    gid_assigned = GlobalID("global_person_1")
    def mock_associate_side_effect(*args, **kwargs):
        tk_from_tracker = (cam1_id, TrackID(1)) 
        mock_reid_manager.track_to_global_id[tk_from_tracker] = gid_assigned
        return None
    mock_reid_manager.associate_features_and_update_state.side_effect = mock_associate_side_effect

    batch_results = await multi_camera_processor_instance.process_frame_batch(
        task_id, env_id, mock_reid_manager, frame_batch_input, frame_count
    )

    mock_detector.detect.assert_called_once_with(cam1_frame_np)
    mock_camera_tracker_factory.get_tracker.assert_called_once_with(task_id, cam1_id)
    
    expected_dets_for_tracker = np.array([[10.0, 10.0, 20.0, 20.0, 0.9, float(mock_settings.PERSON_CLASS_ID)]], dtype=np.float32)
    call_args_tracker_update = mock_tracker_instance.update.call_args[0]
    assert np.array_equal(call_args_tracker_update[0], expected_dets_for_tracker)
    assert np.array_equal(call_args_tracker_update[1], cam1_frame_np)
    
    call_args_reid = mock_reid_manager.associate_features_and_update_state.call_args[0]
    expected_feat_vec = FeatureVector(np.array([0.1,0.2,0.3], dtype=np.float32))
    expected_track_key = (cam1_id, TrackID(1))
    assert expected_track_key in call_args_reid[0] 
    assert np.array_equal(call_args_reid[0][expected_track_key], expected_feat_vec)
    assert call_args_reid[1] == {expected_track_key} 
    assert call_args_reid[3] == frame_count 
    
    mock_homography_service.get_homography_matrix.assert_called_once_with(env_id, cam1_id)
    
    mock_cv2_perspective_transform.assert_called_once() 
    persp_trans_args = mock_cv2_perspective_transform.call_args[0]
    assert np.array_equal(persp_trans_args[0], np.array([[(15.0, 20.0)]], dtype=np.float32))

    assert cam1_id in batch_results
    assert len(batch_results[cam1_id]) == 1
    tracked_obj: TrackedObjectData = batch_results[cam1_id][0]
    
    assert tracked_obj.camera_id == cam1_id
    assert tracked_obj.track_id == TrackID(1)
    assert tracked_obj.global_person_id == gid_assigned
    assert tracked_obj.bbox_xyxy == [10.0, 10.0, 20.0, 20.0]
    assert tracked_obj.confidence == pytest.approx(0.9)
    assert tracked_obj.feature_vector == pytest.approx([0.1, 0.2, 0.3]) # MODIFIED
    assert tracked_obj.map_coords == [50.0, 75.0] 


@pytest.mark.asyncio
async def test_process_frame_batch_no_detections(
    multi_camera_processor_instance: MultiCameraFrameProcessor,
    mock_detector: AsyncMock,
    mock_tracker_instance: AsyncMock,
    mock_reid_manager: MagicMock,
):
    task_id = uuid.uuid4()
    env_id = "test_env"
    frame_count = 1
    cam1_id = CameraID("c01")
    cam1_frame_np = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_batch_input: FrameBatch = {cam1_id: (cam1_frame_np, "path")}

    mock_detector.detect.return_value = [] 

    batch_results = await multi_camera_processor_instance.process_frame_batch(
        task_id, env_id, mock_reid_manager, frame_batch_input, frame_count
    )

    mock_detector.detect.assert_called_once()
    expected_empty_dets_for_tracker = np.empty((0,6), dtype=np.float32)
    call_args_tracker_update = mock_tracker_instance.update.call_args[0]
    assert np.array_equal(call_args_tracker_update[0], expected_empty_dets_for_tracker)

    mock_reid_manager.associate_features_and_update_state.assert_not_called()
    
    assert cam1_id not in batch_results or len(batch_results[cam1_id]) == 0