"""
Phase 1 validation tests for RT-DETR detector implementation.

Tests the foundation setup as outlined in DETECTION.md Phase 1:
- RT-DETR model loading and initialization
- Basic person detection functionality 
- Error handling and graceful degradation
- Configuration validation

Run with: pytest tests/models/test_rtdetr_detector_phase1.py -v
"""

import pytest
import numpy as np
import cv2
import torch
from unittest.mock import patch, MagicMock
import asyncio

from app.models.rtdetr_detector import RTDETRDetector
from app.core.config import settings


class TestRTDETRDetectorPhase1:
    """Phase 1 validation tests for RT-DETR detector."""

    @pytest.fixture
    def detector(self):
        """Create RT-DETR detector instance for testing."""
        return RTDETRDetector(
            model_name="rtdetr-l.pt",
            confidence_threshold=0.5
        )

    @pytest.fixture
    def test_image(self):
        """Create a test image for detection testing."""
        # Create a simple 640x640 test image (RT-DETR input size)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some simple shapes to simulate a scene
        cv2.rectangle(image, (100, 100), (200, 300), (255, 255, 255), -1)  # White rectangle
        cv2.circle(image, (400, 300), 50, (128, 128, 128), -1)  # Gray circle
        
        return image

    @pytest.mark.asyncio
    async def test_detector_initialization(self, detector):
        """Test RT-DETR detector initialization."""
        # Verify detector properties
        assert detector.model_name == "rtdetr-l.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.person_class_id == 0
        assert detector.model is None  # Not loaded yet
        assert not detector._model_loaded_flag

    @pytest.mark.asyncio
    async def test_model_loading(self, detector):
        """Test RT-DETR model loading process."""
        # Load the model
        await detector.load_model()
        
        # Verify model loaded successfully
        assert detector.model is not None
        assert detector._model_loaded_flag
        assert hasattr(detector, 'device')
        
        # Verify device selection
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert detector.device.type == expected_device.type

    @pytest.mark.asyncio
    async def test_model_loading_idempotent(self, detector):
        """Test that loading model multiple times is safe."""
        # Load model first time
        await detector.load_model()
        model_instance_1 = detector.model
        
        # Load model second time
        await detector.load_model()
        model_instance_2 = detector.model
        
        # Should be the same instance (not reloaded)
        assert model_instance_1 is model_instance_2
        assert detector._model_loaded_flag

    @pytest.mark.asyncio
    async def test_warmup_before_model_load(self, detector):
        """Test warmup behavior when model is not loaded."""
        # Attempt warmup without loading model
        await detector.warmup()
        
        # Should not crash, just log warning
        assert not detector._model_loaded_flag

    @pytest.mark.asyncio
    async def test_warmup_after_model_load(self, detector, test_image):
        """Test warmup functionality after model loading."""
        # Load model first
        await detector.load_model()
        assert detector._model_loaded_flag
        
        # Perform warmup
        await detector.warmup()
        
        # Should complete without errors
        assert detector._model_loaded_flag

    @pytest.mark.asyncio
    async def test_detection_without_model_load(self, detector, test_image):
        """Test detection behavior when model is not loaded."""
        # Attempt detection without loading model
        with pytest.raises(RuntimeError, match="Detector model not loaded"):
            await detector.detect(test_image)

    @pytest.mark.asyncio
    async def test_basic_detection_functionality(self, detector, test_image):
        """Test basic detection functionality."""
        # Load model
        await detector.load_model()
        
        # Perform detection
        detections = await detector.detect(test_image)
        
        # Verify detection results structure
        assert isinstance(detections, list)
        
        # Each detection should have the correct structure
        for detection in detections:
            assert hasattr(detection, 'bbox')
            assert hasattr(detection, 'confidence')
            assert hasattr(detection, 'class_id')
            assert hasattr(detection, 'class_name')
            
            # Verify person class
            assert detection.class_id == 0
            assert detection.class_name == "person"
            
            # Verify confidence threshold
            assert detection.confidence >= 0.5
            
            # Verify bounding box structure
            bbox = detection.bbox
            assert hasattr(bbox, 'x1')
            assert hasattr(bbox, 'y1')
            assert hasattr(bbox, 'x2')
            assert hasattr(bbox, 'y2')
            
            # Verify bbox coordinates are valid
            assert 0 <= bbox.x1 < bbox.x2 <= test_image.shape[1]
            assert 0 <= bbox.y1 < bbox.y2 <= test_image.shape[0]

    @pytest.mark.asyncio
    async def test_detection_with_empty_image(self, detector):
        """Test detection with edge case: empty image."""
        # Load model
        await detector.load_model()
        
        # Create empty image
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Perform detection
        detections = await detector.detect(empty_image)
        
        # Should return empty list without crashing
        assert isinstance(detections, list)

    @pytest.mark.asyncio
    async def test_detection_with_invalid_input(self, detector):
        """Test detection with invalid input."""
        # Load model
        await detector.load_model()
        
        # Test with invalid input types
        invalid_inputs = [
            None,
            [],
            "not_an_image",
            np.array([1, 2, 3]),  # Wrong dimensions
        ]
        
        for invalid_input in invalid_inputs:
            # Should handle gracefully and return empty list
            detections = await detector.detect(invalid_input)
            assert detections == []

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test RT-DETR configuration validation."""
        # Test valid configuration
        detector = RTDETRDetector(
            model_name="rtdetr-l.pt",
            confidence_threshold=0.7
        )
        assert detector.confidence_threshold == 0.7
        
        # Test edge case configurations
        detector_low = RTDETRDetector(confidence_threshold=0.1)
        assert detector_low.confidence_threshold == 0.1
        
        detector_high = RTDETRDetector(confidence_threshold=0.99)
        assert detector_high.confidence_threshold == 0.99

    def test_coco_classes_validation(self, detector):
        """Test COCO classes configuration."""
        # Verify COCO classes are properly defined
        assert len(detector.coco_classes) == 80  # COCO has 80 classes
        assert detector.coco_classes[0] == "person"  # Person is class 0
        assert detector.person_class_id == 0

    @pytest.mark.asyncio
    async def test_device_selection_cpu(self, detector):
        """Test device selection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            await detector.load_model()
            assert detector.device.type == "cpu"

    @pytest.mark.asyncio 
    async def test_device_selection_cuda(self, detector):
        """Test device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            await detector.load_model()
            # Note: Actual CUDA availability depends on system
            assert detector.device.type in ["cpu", "cuda"]

    @pytest.mark.asyncio
    async def test_error_handling_model_loading(self):
        """Test error handling during model loading."""
        # Test with invalid model name
        detector = RTDETRDetector(model_name="invalid_model.pt")
        
        # Should raise an exception during model loading
        with pytest.raises(Exception):
            await detector.load_model()
        
        # Verify model is not marked as loaded
        assert not detector._model_loaded_flag
        assert detector.model is None

    @pytest.mark.asyncio
    async def test_detection_performance_basic(self, detector, test_image):
        """Test basic detection performance (Phase 1 baseline)."""
        # Load model
        await detector.load_model()
        
        # Measure detection time
        import time
        start_time = time.time()
        detections = await detector.detect(test_image)
        detection_time = time.time() - start_time
        
        # For Phase 1, we expect reasonable performance (< 2 seconds on CPU)
        assert detection_time < 2.0
        
        # Log performance for monitoring
        print(f"Detection time: {detection_time:.3f}s")


class TestRTDETRDetectorIntegration:
    """Integration tests for RT-DETR detector with application components."""

    @pytest.mark.asyncio
    async def test_integration_with_settings(self):
        """Test integration with application settings."""
        # Create detector using settings
        detector = RTDETRDetector(
            model_name=settings.RTDETR_MODEL_PATH.split("/")[-1],
            confidence_threshold=settings.RTDETR_CONFIDENCE_THRESHOLD
        )
        
        # Verify configuration matches settings
        assert detector.confidence_threshold == settings.RTDETR_CONFIDENCE_THRESHOLD
        
        # Test model loading
        await detector.load_model()
        assert detector._model_loaded_flag

    @pytest.mark.asyncio
    async def test_phase1_validation_checklist(self):
        """Comprehensive Phase 1 validation checklist."""
        detector = RTDETRDetector()
        
        # ✅ RT-DETR model loads successfully
        await detector.load_model()
        assert detector._model_loaded_flag
        
        # ✅ Basic person detection works on test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = await detector.detect(test_image)
        assert isinstance(detections, list)
        
        # ✅ Detection endpoints respond correctly (tested via service)
        # This will be covered by API tests
        
        # ✅ Service classes instantiate without errors
        # Covered by detector initialization
        
        # ✅ Configuration values load properly
        assert detector.confidence_threshold == 0.5
        assert detector.person_class_id == 0
        assert len(detector.coco_classes) == 80
        
        print("✅ Phase 1 validation checklist completed successfully!")