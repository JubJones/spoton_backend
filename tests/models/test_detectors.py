"""
Unit tests for detector models in app.models.detectors.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, PropertyMock, call
import torchvision
import re

from app.models.detectors import FasterRCNNDetector
from app.models.base_models import Detection, BoundingBox
# Assuming mock_settings is available from conftest.py

@pytest.fixture
def mock_torchvision_fasterrcnn(mocker):
    """Mocks torchvision's FasterRCNN model and related components."""
    mock_model_instance = MagicMock(spec=torch.nn.Module) # More generic for .to, .eval
    mock_model_instance.return_value = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0), "scores": torch.empty(0)}] # Default empty prediction

    mock_weights_obj = MagicMock() # Removed spec, will define attributes directly
    mock_weights_obj.transforms = MagicMock(return_value=MagicMock(return_value=torch.empty(3, 100, 100))) # Mock transform method
    mock_weights_obj.meta = {'categories': ['__background__', 'person', 'car']} # Mock COCO names

    mocker.patch("torchvision.models.detection.fasterrcnn_resnet50_fpn_v2", return_value=mock_model_instance)

    # Mock the entire FasterRCNN_ResNet50_FPN_V2_Weights enum/class
    # When FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT is accessed, it will use this PropertyMock
    mock_weights_class = MagicMock()
    type(mock_weights_class).DEFAULT = PropertyMock(return_value=mock_weights_obj) # Use type() for class properties
    mocker.patch("torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights", mock_weights_class)
    # If the import is in app.models.detectors, this would be more targeted:
    # mocker.patch("app.models.detectors.FasterRCNN_ResNet50_FPN_V2_Weights", mock_weights_class)

    return mock_model_instance, mock_weights_obj

@pytest.fixture
def fasterrcnn_detector_instance(mock_settings, mock_torchvision_fasterrcnn, mocker):
    """Provides an instance of FasterRCNNDetector with mocked dependencies."""
    # Patch settings directly in the module where FasterRCNNDetector imports it
    mocker.patch("app.models.detectors.settings", mock_settings)
    mocker.patch("app.utils.video_processing.cv2.cvtColor", side_effect=lambda x, code: x) # Avoid actual cv2 call

    # Mock torch.device to return a real torch.device('cpu') to ensure compatibility with tensor.to()
    actual_cpu_device = torch.device('cpu')
    mocker.patch("app.models.detectors.torch.device", return_value=actual_cpu_device) # Patch where detector uses it
    # If FasterRCNNDetector initializes self.device via a global torch.device call, 
    # and not app.models.detectors.torch.device, then a broader patch might be needed, 
    # or ensure self.device is set correctly post-init in tests.
    # For now, assuming it uses torch.device imported within its own module scope.

    return FasterRCNNDetector()

@pytest.mark.asyncio
async def test_fasterrcnn_init(fasterrcnn_detector_instance: FasterRCNNDetector, mock_settings):
    """Tests FasterRCNNDetector initialization."""
    detector = fasterrcnn_detector_instance
    assert detector.person_class_id == mock_settings.PERSON_CLASS_ID
    assert detector.detection_confidence_threshold == mock_settings.DETECTION_CONFIDENCE_THRESHOLD
    assert detector.use_amp == mock_settings.DETECTION_USE_AMP
    assert detector.model is None
    assert detector._model_loaded_flag is False
    assert "person" in detector.coco_names

@pytest.mark.asyncio
async def test_fasterrcnn_load_model_success(fasterrcnn_detector_instance: FasterRCNNDetector, mock_torchvision_fasterrcnn, mocker):
    """Tests successful model loading."""
    detector = fasterrcnn_detector_instance
    mock_model, _ = mock_torchvision_fasterrcnn
    mock_logger_info = mocker.patch("app.models.detectors.logger.info")

    await detector.load_model()

    assert detector.model == mock_model
    assert detector.transforms is not None
    assert detector._model_loaded_flag is True
    mock_model.to.assert_called_once()
    mock_model.eval.assert_called_once()
    mock_logger_info.assert_any_call("Faster R-CNN model loaded and configured successfully.")

@pytest.mark.asyncio
async def test_fasterrcnn_load_model_already_loaded(fasterrcnn_detector_instance: FasterRCNNDetector, mocker):
    """Tests that load_model doesn't reload if already loaded."""
    detector = fasterrcnn_detector_instance
    detector.model = MagicMock() # Mark as loaded
    detector._model_loaded_flag = True
    mock_logger_info = mocker.patch("app.models.detectors.logger.info")

    await detector.load_model()
    mock_logger_info.assert_called_with("Faster R-CNN model already loaded.")
    # Ensure model.to() or model.eval() were not called again
    assert detector.model.to.call_count == 0

@pytest.mark.asyncio
async def test_fasterrcnn_load_model_failure(fasterrcnn_detector_instance: FasterRCNNDetector, mocker):
    """Tests model loading failure."""
    detector = fasterrcnn_detector_instance
    mocker.patch("torchvision.models.detection.fasterrcnn_resnet50_fpn_v2", side_effect=RuntimeError("Load failed"))
    mock_logger_exception = mocker.patch("app.models.detectors.logger.exception")

    with pytest.raises(RuntimeError, match="Load failed"):
        await detector.load_model()

    assert detector.model is None
    assert detector._model_loaded_flag is False
    mock_logger_exception.assert_called_once()

@pytest.mark.asyncio
async def test_fasterrcnn_warmup(fasterrcnn_detector_instance: FasterRCNNDetector, mocker):
    """Tests model warmup."""
    detector = fasterrcnn_detector_instance
    await detector.load_model() # Ensure model is loaded

    # Mock the detect method for the warmup call
    mock_detect_internal = mocker.patch.object(detector, "detect", autospec=True)
    mock_logger_info = mocker.patch("app.models.detectors.logger.info")

    await detector.warmup()

    mock_detect_internal.assert_called_once()
    # Check that the argument to detect was a numpy array of the default shape
    assert isinstance(mock_detect_internal.call_args[0][0], np.ndarray)
    assert mock_detect_internal.call_args[0][0].shape == (640, 480, 3)
    mock_logger_info.assert_any_call("FasterRCNN detector warmup successful.")

@pytest.mark.asyncio
async def test_fasterrcnn_detect_model_not_loaded(fasterrcnn_detector_instance: FasterRCNNDetector):
    """Tests detect call when model is not loaded."""
    detector = fasterrcnn_detector_instance
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    expected_message = "Detector model not loaded. Call load_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        await detector.detect(dummy_image)

@pytest.mark.asyncio
async def test_fasterrcnn_detect_successful(fasterrcnn_detector_instance: FasterRCNNDetector, mock_torchvision_fasterrcnn, mock_settings, mocker):
    """Tests successful detection and postprocessing."""
    detector = fasterrcnn_detector_instance
    await detector.load_model()
    mock_model, mock_weights = mock_torchvision_fasterrcnn

    # Configure mock_settings for this test
    mock_settings.PERSON_CLASS_ID = 1 # 'person' in our mocked coco_names
    mock_settings.DETECTION_CONFIDENCE_THRESHOLD = 0.6
    detector.person_class_id = mock_settings.PERSON_CLASS_ID
    detector.detection_confidence_threshold = mock_settings.DETECTION_CONFIDENCE_THRESHOLD
    detector.coco_names = mock_weights.meta['categories'] # Ensure it uses the mocked names

    dummy_image_np = np.zeros((480, 640, 3), dtype=np.uint8) # H, W, C

    # Mock model output: one person, one car
    mock_model_output = {
        "boxes": torch.tensor([[10, 10, 60, 60], [70, 70, 120, 120]], dtype=torch.float32), # x1,y1,x2,y2
        "labels": torch.tensor([1, 2]),  # 1 for person, 2 for car
        "scores": torch.tensor([0.9, 0.8], dtype=torch.float32)
    }
    mock_model.return_value = [mock_model_output] # Model expects list of tensors, returns list of dicts

    # Mock asyncio.to_thread for the model call
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args: func(*args))


    detections = await detector.detect(dummy_image_np)

    assert len(detections) == 1
    person_detection = detections[0]
    assert isinstance(person_detection, Detection)
    assert person_detection.class_id == 1
    assert person_detection.class_name == "person"
    assert person_detection.confidence == 0.9
    assert isinstance(person_detection.bbox, BoundingBox)
    assert person_detection.bbox.to_list() == [10.0, 10.0, 60.0, 60.0] # Exact match

@pytest.mark.asyncio
async def test_fasterrcnn_detect_no_persons_above_threshold(fasterrcnn_detector_instance: FasterRCNNDetector, mock_torchvision_fasterrcnn, mock_settings, mocker):
    """Tests detection when no persons meet the confidence threshold."""
    detector = fasterrcnn_detector_instance
    await detector.load_model()
    mock_model, _ = mock_torchvision_fasterrcnn

    mock_settings.PERSON_CLASS_ID = 1
    mock_settings.DETECTION_CONFIDENCE_THRESHOLD = 0.95 # High threshold
    detector.person_class_id = mock_settings.PERSON_CLASS_ID
    detector.detection_confidence_threshold = mock_settings.DETECTION_CONFIDENCE_THRESHOLD


    dummy_image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_model_output = {
        "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9], dtype=torch.float32) # Below threshold
    }
    mock_model.return_value = [mock_model_output]
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args: func(*args))

    detections = await detector.detect(dummy_image_np)
    assert len(detections) == 0

@pytest.mark.asyncio
async def test_fasterrcnn_detect_bbox_clipping(fasterrcnn_detector_instance: FasterRCNNDetector, mock_torchvision_fasterrcnn, mock_settings, mocker):
    """Tests that bounding boxes are clipped to image dimensions."""
    detector = fasterrcnn_detector_instance
    await detector.load_model()
    mock_model, _ = mock_torchvision_fasterrcnn
    mock_settings.PERSON_CLASS_ID = 1
    mock_settings.DETECTION_CONFIDENCE_THRESHOLD = 0.5
    detector.person_class_id = mock_settings.PERSON_CLASS_ID
    detector.detection_confidence_threshold = mock_settings.DETECTION_CONFIDENCE_THRESHOLD

    img_h, img_w = 100, 150
    dummy_image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    mock_model_output = {
        "boxes": torch.tensor([[-10, -5, img_w + 20, img_h + 30]], dtype=torch.float32), # Box outside bounds
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9], dtype=torch.float32)
    }
    mock_model.return_value = [mock_model_output]
    mocker.patch("asyncio.to_thread", side_effect=lambda func, *args: func(*args))

    detections = await detector.detect(dummy_image_np)
    assert len(detections) == 1
    assert detections[0].bbox.to_list() == [0.0, 0.0, float(img_w), float(img_h)] 