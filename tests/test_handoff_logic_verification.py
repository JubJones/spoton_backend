import pytest
from unittest.mock import patch, MagicMock
from app.services.handoff_detection_service import HandoffDetectionService, CameraZone

# Mock settings
@patch("app.services.handoff_detection_service.settings")
def test_handoff_default_zones(mock_settings):
    # Setup: No configured zones
    mock_settings.CAMERA_HANDOFF_ZONES = {}
    mock_settings.MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT = 0.5
    mock_settings.POSSIBLE_CAMERA_OVERLAPS = []
    
    # Initialize service
    service = HandoffDetectionService()
    
    # Verify initially empty
    assert len(service.camera_zones) == 0
    
    # Test with a new camera ID
    camera_id = "test_cam_auto"
    frame_w, frame_h = 1000, 1000
    
    # Case 1: BBox in the center (should NOT trigger, but SHOULD initialize zones)
    bbox_center = {"center_x": 500, "center_y": 500, "width": 100, "height": 100}
    is_triggered, _ = service.check_handoff_trigger(camera_id, bbox_center, frame_w, frame_h)
    
    assert is_triggered is False
    assert camera_id in service.camera_zones
    assert len(service.camera_zones[camera_id]) == 4 # Should have 4 default zones
    
    # Case 2: BBox at the Left Edge (should trigger)
    # Zone is 0.0 to 0.1 (0 to 100px). BBox center at 50, width 100.
    bbox_left = {"center_x": 50, "center_y": 500, "width": 100, "height": 100}
    is_triggered, _ = service.check_handoff_trigger(camera_id, bbox_left, frame_w, frame_h)
    
    assert is_triggered is True
