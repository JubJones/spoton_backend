"""
Detection use cases for application layer.

Business logic for person detection operations including frame processing,
confidence validation, and detection result management.
Maximum 250 lines per plan.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.frame_id import FrameID
from app.domain.detection.entities.detection import Detection
from app.domain.detection.value_objects.confidence import Confidence
from app.domain.detection.value_objects.detection_class import DetectionClass
from app.domain.detection.services.detection_service import DetectionService
from app.domain.shared.value_objects.bounding_box import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class DetectionRequest:
    """Request for person detection."""
    camera_id: CameraID
    frame_id: FrameID
    frame_data: bytes
    timestamp: datetime
    confidence_threshold: float = 0.5


@dataclass
class DetectionResult:
    """Result of detection operation."""
    camera_id: CameraID
    frame_id: FrameID
    detections: List[Detection]
    processing_time_ms: float
    timestamp: datetime


class DetectionUseCase:
    """
    Detection use case for application layer.
    
    Orchestrates person detection operations with business logic
    for validation, filtering, and result processing.
    """
    
    def __init__(self, detection_service: DetectionService):
        """
        Initialize detection use case.
        
        Args:
            detection_service: Domain detection service
        """
        self.detection_service = detection_service
        
        # Processing statistics
        self._detection_stats = {
            'frames_processed': 0,
            'detections_found': 0,
            'high_confidence_detections': 0,
            'avg_processing_time_ms': 0.0
        }
        
        logger.debug("DetectionUseCase initialized")
    
    async def process_frame_for_detection(
        self,
        request: DetectionRequest
    ) -> DetectionResult:
        """
        Process frame for person detection.
        
        Args:
            request: Detection request with frame data
            
        Returns:
            Detection result with found detections
        """
        start_time = datetime.utcnow()
        
        try:
            # Perform detection using domain service
            detections = await self.detection_service.detect_persons_in_frame(
                camera_id=request.camera_id,
                frame_id=request.frame_id,
                frame_data=request.frame_data,
                confidence_threshold=request.confidence_threshold
            )
            
            # Filter valid detections
            valid_detections = self._filter_valid_detections(detections)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_detection_statistics(valid_detections, processing_time)
            
            # Create result
            result = DetectionResult(
                camera_id=request.camera_id,
                frame_id=request.frame_id,
                detections=valid_detections,
                processing_time_ms=processing_time,
                timestamp=request.timestamp
            )
            
            logger.debug(f"Processed frame {request.frame_id} from camera {request.camera_id}: "
                        f"{len(valid_detections)} detections found")
            
            return result
            
        except Exception as e:
            logger.error(f"Detection processing failed for frame {request.frame_id}: {e}")
            raise
    
    async def process_batch_frames(
        self,
        requests: List[DetectionRequest]
    ) -> List[DetectionResult]:
        """
        Process multiple frames for detection in batch.
        
        Args:
            requests: List of detection requests
            
        Returns:
            List of detection results
        """
        results = []
        
        for request in requests:
            try:
                result = await self.process_frame_for_detection(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch detection failed for frame {request.frame_id}: {e}")
                # Create empty result for failed frame
                results.append(DetectionResult(
                    camera_id=request.camera_id,
                    frame_id=request.frame_id,
                    detections=[],
                    processing_time_ms=0.0,
                    timestamp=request.timestamp
                ))
        
        logger.info(f"Processed batch of {len(requests)} frames")
        return results
    
    def validate_detection_confidence(
        self,
        detections: List[Detection],
        min_confidence: float = 0.5
    ) -> List[Detection]:
        """
        Validate detections based on confidence threshold.
        
        Args:
            detections: List of detections to validate
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of validated detections
        """
        return [
            detection for detection in detections
            if detection.confidence.value >= min_confidence
        ]
    
    def filter_detections_by_area(
        self,
        detections: List[Detection],
        min_area_pixels: int = 100
    ) -> List[Detection]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of detections to filter
            min_area_pixels: Minimum bounding box area in pixels
            
        Returns:
            List of filtered detections
        """
        filtered_detections = []
        
        for detection in detections:
            bbox_area = detection.bbox.width * detection.bbox.height
            if bbox_area >= min_area_pixels:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def get_high_confidence_detections(
        self,
        detections: List[Detection],
        confidence_threshold: float = 0.8
    ) -> List[Detection]:
        """
        Get only high confidence detections.
        
        Args:
            detections: List of detections
            confidence_threshold: High confidence threshold
            
        Returns:
            List of high confidence detections
        """
        return [
            detection for detection in detections
            if detection.is_high_confidence and detection.confidence.value >= confidence_threshold
        ]
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection processing statistics."""
        return self._detection_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self._detection_stats = {
            'frames_processed': 0,
            'detections_found': 0,
            'high_confidence_detections': 0,
            'avg_processing_time_ms': 0.0
        }
        logger.debug("Detection statistics reset")
    
    def _filter_valid_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections for validity."""
        valid_detections = []
        
        for detection in detections:
            # Check if detection class is person
            if detection.detection_class != DetectionClass.PERSON:
                continue
            
            # Check confidence threshold
            if not detection.is_high_confidence:
                continue
            
            # Check bounding box validity
            if not self._is_valid_bounding_box(detection.bbox):
                continue
            
            valid_detections.append(detection)
        
        return valid_detections
    
    def _is_valid_bounding_box(self, bbox: BoundingBox) -> bool:
        """Validate bounding box parameters."""
        # Check for positive dimensions
        if bbox.width <= 0 or bbox.height <= 0:
            return False
        
        # Check for reasonable size (not too small)
        min_size = 10  # pixels
        if bbox.width < min_size or bbox.height < min_size:
            return False
        
        # Check for reasonable aspect ratio (person-like)
        aspect_ratio = bbox.height / bbox.width
        if aspect_ratio < 1.2 or aspect_ratio > 4.0:  # Person should be taller than wide
            return False
        
        return True
    
    def _update_detection_statistics(
        self,
        detections: List[Detection],
        processing_time_ms: float
    ) -> None:
        """Update detection processing statistics."""
        self._detection_stats['frames_processed'] += 1
        self._detection_stats['detections_found'] += len(detections)
        
        # Count high confidence detections
        high_conf_count = len([d for d in detections if d.is_high_confidence])
        self._detection_stats['high_confidence_detections'] += high_conf_count
        
        # Update average processing time
        current_avg = self._detection_stats['avg_processing_time_ms']
        frame_count = self._detection_stats['frames_processed']
        
        self._detection_stats['avg_processing_time_ms'] = (
            (current_avg * (frame_count - 1) + processing_time_ms) / frame_count
        )


class DetectionValidationUseCase:
    """
    Detection validation use case.
    
    Specialized use case for validating and refining detection results
    based on business rules and quality criteria.
    """
    
    def __init__(self):
        """Initialize detection validation use case."""
        self.validation_stats = {
            'detections_validated': 0,
            'detections_rejected': 0,
            'rejection_reasons': {}
        }
        
        logger.debug("DetectionValidationUseCase initialized")
    
    def validate_detection_quality(
        self,
        detection: Detection,
        quality_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate detection quality based on criteria.
        
        Args:
            detection: Detection to validate
            quality_criteria: Optional quality criteria
            
        Returns:
            Validation result with quality assessment
        """
        criteria = quality_criteria or self._get_default_quality_criteria()
        
        validation_result = {
            'is_valid': True,
            'quality_score': 0.0,
            'rejection_reasons': [],
            'quality_metrics': {}
        }
        
        # Confidence check
        confidence_score = detection.confidence.value
        validation_result['quality_metrics']['confidence'] = confidence_score
        
        if confidence_score < criteria.get('min_confidence', 0.5):
            validation_result['is_valid'] = False
            validation_result['rejection_reasons'].append('Low confidence')
        
        # Bounding box quality
        bbox_quality = self._assess_bounding_box_quality(detection.bbox)
        validation_result['quality_metrics']['bbox_quality'] = bbox_quality
        
        if bbox_quality < criteria.get('min_bbox_quality', 0.6):
            validation_result['is_valid'] = False
            validation_result['rejection_reasons'].append('Poor bounding box quality')
        
        # Calculate overall quality score
        validation_result['quality_score'] = (confidence_score + bbox_quality) / 2
        
        # Update validation statistics
        self._update_validation_statistics(validation_result)
        
        return validation_result
    
    def _get_default_quality_criteria(self) -> Dict[str, Any]:
        """Get default quality criteria for validation."""
        return {
            'min_confidence': 0.6,
            'min_bbox_quality': 0.6,
            'min_area_pixels': 100,
            'max_area_pixels': 50000,
            'min_aspect_ratio': 1.2,
            'max_aspect_ratio': 4.0
        }
    
    def _assess_bounding_box_quality(self, bbox: BoundingBox) -> float:
        """Assess bounding box quality (0.0 to 1.0)."""
        quality_score = 1.0
        
        # Aspect ratio assessment
        aspect_ratio = bbox.height / bbox.width
        if aspect_ratio < 1.2 or aspect_ratio > 4.0:
            quality_score *= 0.5
        
        # Size assessment
        area = bbox.width * bbox.height
        if area < 100:  # Too small
            quality_score *= 0.3
        elif area > 50000:  # Too large
            quality_score *= 0.7
        
        return quality_score
    
    def _update_validation_statistics(self, validation_result: Dict[str, Any]) -> None:
        """Update validation statistics."""
        self.validation_stats['detections_validated'] += 1
        
        if not validation_result['is_valid']:
            self.validation_stats['detections_rejected'] += 1
            
            # Count rejection reasons
            for reason in validation_result['rejection_reasons']:
                if reason not in self.validation_stats['rejection_reasons']:
                    self.validation_stats['rejection_reasons'][reason] = 0
                self.validation_stats['rejection_reasons'][reason] += 1