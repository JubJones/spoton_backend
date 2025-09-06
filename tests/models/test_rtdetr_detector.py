"""
Test RT-DETR detector implementation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock


class TestRTDETRDetector:
    """Test RT-DETR detector functionality."""
    
    def test_rtdetr_detector_import(self):
        """Test that RT-DETR detector can be imported."""
        from app.domains.detection.models.rtdetr_detector import RTDETRDetector
        
        # Test basic initialization without loading model
        with patch('app.domains.detection.models.rtdetr_detector.TORCH_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.ULTRALYTICS_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.get_gpu_manager') as mock_gpu:
            
            mock_gpu.return_value.get_optimal_device.return_value = 'cpu'
            
            detector = RTDETRDetector()
            assert detector is not None
            assert detector.model_name == "rtdetr-l.pt"
            assert detector.confidence_threshold == 0.5
            assert not detector.is_loaded()
    
    def test_rtdetr_detector_factory_registration(self):
        """Test that RT-DETR detector is registered with factory."""
        from app.domains.detection.models import DetectorFactory
        
        available_detectors = DetectorFactory.get_available_detectors()
        assert 'rtdetr' in available_detectors
        assert 'rt-detr' in available_detectors
    
    def test_rtdetr_detector_model_info(self):
        """Test model info functionality."""
        from app.domains.detection.models.rtdetr_detector import RTDETRDetector
        
        with patch('app.domains.detection.models.rtdetr_detector.TORCH_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.ULTRALYTICS_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.get_gpu_manager') as mock_gpu:
            
            mock_gpu.return_value.get_optimal_device.return_value = 'cpu'
            
            detector = RTDETRDetector()
            model_info = detector.get_model_info()
            
            assert model_info['name'] == 'RT-DETR'
            assert model_info['architecture'] == 'rtdetr-l.pt'
            assert model_info['device'] == 'cpu'
            assert model_info['loaded'] == False
    
    def test_rtdetr_detector_supported_classes(self):
        """Test supported classes."""
        from app.domains.detection.models.rtdetr_detector import RTDETRDetector
        
        with patch('app.domains.detection.models.rtdetr_detector.TORCH_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.ULTRALYTICS_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.get_gpu_manager') as mock_gpu:
            
            mock_gpu.return_value.get_optimal_device.return_value = 'cpu'
            
            detector = RTDETRDetector()
            supported_classes = detector.get_supported_classes()
            
            assert supported_classes == ['person']
    
    def test_rtdetr_detector_confidence_threshold(self):
        """Test confidence threshold operations."""
        from app.domains.detection.models.rtdetr_detector import RTDETRDetector
        
        with patch('app.domains.detection.models.rtdetr_detector.TORCH_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.ULTRALYTICS_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.get_gpu_manager') as mock_gpu:
            
            mock_gpu.return_value.get_optimal_device.return_value = 'cpu'
            
            detector = RTDETRDetector()
            
            # Test initial threshold
            assert detector.get_confidence_threshold() == 0.5
            
            # Test setting threshold
            detector.set_confidence_threshold(0.7)
            assert detector.get_confidence_threshold() == 0.7
            
            # Test threshold bounds
            detector.set_confidence_threshold(1.5)  # Should be clamped to 1.0
            assert detector.get_confidence_threshold() == 1.0
            
            detector.set_confidence_threshold(-0.5)  # Should be clamped to 0.0
            assert detector.get_confidence_threshold() == 0.0

    @pytest.mark.asyncio
    async def test_rtdetr_detector_mock_detection(self):
        """Test detection with mocked ultralytics."""
        from app.domains.detection.models.rtdetr_detector import RTDETRDetector
        
        with patch('app.domains.detection.models.rtdetr_detector.TORCH_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.ULTRALYTICS_AVAILABLE', True), \
             patch('app.domains.detection.models.rtdetr_detector.get_gpu_manager') as mock_gpu, \
             patch('app.domains.detection.models.rtdetr_detector.RTDETR') as mock_rtdetr_class:
            
            mock_gpu.return_value.get_optimal_device.return_value = 'cpu'
            
            # Mock the RTDETR model
            mock_model = Mock()
            mock_rtdetr_class.return_value = mock_model
            mock_model.info = Mock()
            mock_model.to = Mock()
            
            # Mock detection result
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = Mock()
            mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]])
            mock_result.boxes.conf = Mock()
            mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
            mock_result.boxes.cls = Mock()
            mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([0])  # person class
            
            mock_model.return_value = [mock_result]
            
            detector = RTDETRDetector()
            await detector.load_model()
            
            # Test detection
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = await detector.detect(dummy_image)
            
            assert len(detections) == 1
            assert detections[0]['class_name'] == 'person'
            assert detections[0]['confidence'] == 0.8
            assert detections[0]['bbox']['x'] == 100
            assert detections[0]['bbox']['y'] == 100
            assert detections[0]['bbox']['width'] == 100
            assert detections[0]['bbox']['height'] == 100


class TestLegacyRTDETRDetector:
    """Test legacy RT-DETR detector functionality."""
    
    def test_legacy_rtdetr_detector_import(self):
        """Test that legacy RT-DETR detector can be imported."""
        from app.models.rtdetr_detector import RTDETRDetector
        
        with patch('app.models.rtdetr_detector.TORCH_AVAILABLE', True), \
             patch('app.models.rtdetr_detector.ULTRALYTICS_AVAILABLE', True), \
             patch('torch.device'), \
             patch('torch.cuda.is_available', return_value=False):
            
            detector = RTDETRDetector()
            assert detector is not None
            assert detector.model_name == "rtdetr-l.pt"
            assert detector.confidence_threshold == 0.5