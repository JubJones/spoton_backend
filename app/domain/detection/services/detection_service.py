"""
Detection domain service providing core detection business logic.

Encapsulates pure business logic for detection operations while maintaining
proper domain boundaries and single responsibility.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import numpy as np

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.frame_id import FrameID
from app.domain.shared.value_objects.bounding_box import BoundingBox
from app.domain.detection.entities.detection import Detection, DetectionBatch
from app.domain.detection.value_objects.confidence import Confidence
from app.domain.detection.value_objects.detection_class import DetectionClass, DetectionClassType

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Detection domain service for core detection business logic.
    
    Provides business rules and operations for detection entities
    while maintaining domain integrity and proper boundaries.
    Maximum file size: 300 lines as per refactoring plan.
    """
    
    def __init__(self, min_confidence_threshold: float = 0.3):
        """
        Initialize detection service.
        
        Args:
            min_confidence_threshold: Minimum confidence for valid detections
        """
        self.min_confidence_threshold = min_confidence_threshold
        self._detection_stats = {
            'total_processed': 0,
            'total_valid': 0,
            'total_filtered': 0
        }
        
        logger.debug(f"DetectionService initialized with threshold {min_confidence_threshold}")
    
    def create_detection(
        self,
        camera_id: CameraID,
        frame_id: FrameID,
        bbox_data: Dict[str, float],
        confidence: float,
        class_id: int,
        timestamp: Optional[datetime] = None,
        **metadata
    ) -> Detection:
        """
        Create a new detection with validation.
        
        Args:
            camera_id: Camera identifier
            frame_id: Frame identifier
            bbox_data: Bounding box data dictionary
            confidence: Detection confidence (0-1)
            class_id: Detection class ID
            timestamp: Detection timestamp
            **metadata: Additional metadata
            
        Returns:
            New Detection instance
            
        Raises:
            ValueError: If detection data is invalid
        """
        # Create bounding box from data
        if 'width' in bbox_data and 'height' in bbox_data:
            bbox = BoundingBox.from_xywh(
                x=bbox_data['x'],
                y=bbox_data['y'],
                w=bbox_data['width'],
                h=bbox_data['height'],
                normalized=bbox_data.get('normalized', False)
            )
        else:
            bbox = BoundingBox.from_coordinates(
                x1=bbox_data['x1'],
                y1=bbox_data['y1'],
                x2=bbox_data['x2'],
                y2=bbox_data['y2'],
                normalized=bbox_data.get('normalized', False)
            )
        
        # Create confidence value object
        confidence_obj = Confidence.from_float(confidence)
        
        # Create detection class
        detection_class = DetectionClass.from_legacy_id(class_id)
        
        # Create detection
        detection = Detection.create(
            camera_id=camera_id,
            frame_id=frame_id,
            bbox=bbox,
            confidence=confidence_obj,
            detection_class=detection_class,
            timestamp=timestamp or datetime.utcnow(),
            processing_metadata=metadata
        )
        
        self._detection_stats['total_processed'] += 1
        
        # Validate detection meets business rules
        if self.is_valid_detection(detection):
            self._detection_stats['total_valid'] += 1
            logger.debug(f"Created valid detection: {detection}")
            return detection
        else:
            self._detection_stats['total_filtered'] += 1
            raise ValueError(f"Detection does not meet validation criteria: {detection}")
    
    def is_valid_detection(self, detection: Detection) -> bool:
        """
        Check if detection meets business validation rules.
        
        Args:
            detection: Detection to validate
            
        Returns:
            True if detection is valid
        """
        # Check confidence threshold
        if not detection.confidence.meets_threshold(self.min_confidence_threshold):
            logger.debug(f"Detection rejected: confidence {detection.confidence} below threshold {self.min_confidence_threshold}")
            return False
        
        # Check bounding box area (minimum size filter)
        min_area = 100.0  # Minimum 10x10 pixel area
        if detection.bbox_area < min_area:
            logger.debug(f"Detection rejected: area {detection.bbox_area} below minimum {min_area}")
            return False
        
        # Check aspect ratio (filter very thin/wide boxes)
        aspect_ratio = detection.bbox.aspect_ratio
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            logger.debug(f"Detection rejected: aspect ratio {aspect_ratio} outside valid range")
            return False
        
        # Additional domain-specific validations can be added here
        
        return True
    
    def filter_detections(
        self,
        detections: List[Detection],
        confidence_threshold: Optional[float] = None,
        class_filter: Optional[DetectionClassType] = None,
        area_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Filter detections based on business rules.
        
        Args:
            detections: List of detections to filter
            confidence_threshold: Optional confidence threshold override
            class_filter: Optional class type filter
            area_threshold: Optional minimum area threshold
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        threshold = confidence_threshold or self.min_confidence_threshold
        
        for detection in detections:
            # Check confidence
            if not detection.confidence.meets_threshold(threshold):
                continue
            
            # Check class filter
            if class_filter and not detection.detection_class.matches_type(class_filter):
                continue
            
            # Check area filter
            if area_threshold and detection.bbox_area < area_threshold:
                continue
            
            # Apply general validation
            if self.is_valid_detection(detection):
                filtered.append(detection)
        
        logger.debug(f"Filtered {len(detections)} detections to {len(filtered)}")
        return filtered
    
    def create_detection_batch(
        self,
        detections: List[Detection],
        processing_time_ms: float = 0.0,
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionBatch:
        """
        Create a detection batch with validation.
        
        Args:
            detections: List of Detection instances
            processing_time_ms: Processing time in milliseconds
            batch_metadata: Additional batch metadata
            
        Returns:
            DetectionBatch instance
        """
        # Validate all detections
        valid_detections = self.filter_detections(detections)
        
        batch = DetectionBatch.create(
            detections=valid_detections,
            processing_time_ms=processing_time_ms,
            metadata=batch_metadata or {}
        )
        
        logger.info(f"Created detection batch with {batch.detection_count} detections")
        return batch
    
    def merge_duplicate_detections(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Merge duplicate detections using Non-Maximum Suppression logic.
        
        Args:
            detections: List of detections to process
            iou_threshold: IoU threshold for considering detections as duplicates
            
        Returns:
            List of detections with duplicates merged
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(
            detections, 
            key=lambda d: float(d.confidence), 
            reverse=True
        )
        
        merged = []
        used_indices = set()
        
        for i, detection in enumerate(sorted_detections):
            if i in used_indices:
                continue
            
            # Find overlapping detections
            overlapping = [detection]
            
            for j, other_detection in enumerate(sorted_detections[i + 1:], i + 1):
                if j in used_indices:
                    continue
                
                # Check if same camera and similar class
                if (detection.camera_id == other_detection.camera_id and
                    detection.detection_class.class_type == other_detection.detection_class.class_type):
                    
                    # Check IoU overlap
                    iou = detection.bbox.iou(other_detection.bbox)
                    if iou >= iou_threshold:
                        overlapping.append(other_detection)
                        used_indices.add(j)
            
            # Merge overlapping detections (keep highest confidence)
            if len(overlapping) > 1:
                # For now, just keep the highest confidence detection
                # More sophisticated merging logic can be added here
                logger.debug(f"Merged {len(overlapping)} overlapping detections")
            
            merged.append(detection)
        
        logger.debug(f"NMS reduced {len(detections)} detections to {len(merged)}")
        return merged
    
    def calculate_detection_statistics(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Calculate comprehensive detection statistics.
        
        Args:
            detections: List of detections to analyze
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'total_count': 0,
                'person_count': 0,
                'vehicle_count': 0,
                'average_confidence': 0.0,
                'confidence_distribution': {},
                'camera_distribution': {},
                'area_statistics': {}
            }
        
        # Basic counts
        total_count = len(detections)
        person_count = sum(1 for d in detections if d.is_person)
        vehicle_count = sum(1 for d in detections if d.is_vehicle)
        
        # Confidence statistics
        confidences = [float(d.confidence) for d in detections]
        average_confidence = sum(confidences) / len(confidences)
        
        confidence_distribution = {
            'high (>0.8)': sum(1 for c in confidences if c > 0.8),
            'medium (0.5-0.8)': sum(1 for c in confidences if 0.5 <= c <= 0.8),
            'low (<0.5)': sum(1 for c in confidences if c < 0.5)
        }
        
        # Camera distribution
        camera_counts = {}
        for detection in detections:
            camera_key = str(detection.camera_id)
            camera_counts[camera_key] = camera_counts.get(camera_key, 0) + 1
        
        # Area statistics
        areas = [d.bbox_area for d in detections]
        area_stats = {
            'min': min(areas),
            'max': max(areas),
            'average': sum(areas) / len(areas),
            'median': sorted(areas)[len(areas) // 2]
        }
        
        return {
            'total_count': total_count,
            'person_count': person_count,
            'vehicle_count': vehicle_count,
            'average_confidence': average_confidence,
            'confidence_distribution': confidence_distribution,
            'camera_distribution': camera_counts,
            'area_statistics': area_stats,
            'processing_statistics': self._detection_stats.copy()
        }
    
    def get_high_quality_detections(
        self,
        detections: List[Detection],
        quality_threshold: float = 0.8
    ) -> List[Detection]:
        """
        Extract high-quality detections for further processing.
        
        Args:
            detections: Input detections
            quality_threshold: Quality threshold (confidence-based)
            
        Returns:
            High-quality detections
        """
        high_quality = []
        
        for detection in detections:
            # Multi-factor quality assessment
            confidence_score = float(detection.confidence)
            area_score = min(1.0, detection.bbox_area / 10000.0)  # Normalize to reasonable area
            aspect_score = min(1.0, 1.0 / abs(detection.bbox.aspect_ratio - 0.7))  # Prefer ~human aspect ratio
            
            # Weighted quality score
            quality_score = (
                confidence_score * 0.6 +
                area_score * 0.2 +
                aspect_score * 0.2
            )
            
            if quality_score >= quality_threshold:
                # Add quality metadata
                enhanced_detection = detection.add_metadata('quality_score', quality_score)
                high_quality.append(enhanced_detection)
        
        logger.debug(f"Selected {len(high_quality)} high-quality detections from {len(detections)}")
        return high_quality
    
    def get_processing_statistics(self) -> Dict[str, int]:
        """Get detection processing statistics."""
        return self._detection_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._detection_stats = {
            'total_processed': 0,
            'total_valid': 0,
            'total_filtered': 0
        }
        logger.debug("Detection processing statistics reset")