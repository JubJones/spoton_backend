"""
Detection repository interface for application layer.

Abstract interface defining detection data access operations.
Follows repository pattern with clean separation of concerns.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.frame_id import FrameID
from app.domain.detection.entities.detection import Detection
from app.domain.shared.value_objects.time_range import TimeRange


class DetectionRepository(ABC):
    """
    Abstract detection repository interface.
    
    Defines the contract for detection data persistence and retrieval
    without coupling to specific database implementations.
    """
    
    @abstractmethod
    async def save_detection(self, detection: Detection) -> bool:
        """
        Save a detection to persistent storage.
        
        Args:
            detection: Detection entity to save
            
        Returns:
            True if save successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def save_detections_batch(self, detections: List[Detection]) -> int:
        """
        Save multiple detections in batch operation.
        
        Args:
            detections: List of detection entities
            
        Returns:
            Number of detections successfully saved
        """
        pass
    
    @abstractmethod
    async def get_detections_by_camera(
        self,
        camera_id: CameraID,
        time_range: Optional[TimeRange] = None,
        limit: int = 1000
    ) -> List[Detection]:
        """
        Get detections for specific camera.
        
        Args:
            camera_id: Camera identifier
            time_range: Optional time range filter
            limit: Maximum number of detections to return
            
        Returns:
            List of detection entities
        """
        pass
    
    @abstractmethod
    async def get_detections_by_frame(
        self,
        camera_id: CameraID,
        frame_id: FrameID
    ) -> List[Detection]:
        """
        Get all detections for a specific frame.
        
        Args:
            camera_id: Camera identifier
            frame_id: Frame identifier
            
        Returns:
            List of detection entities for the frame
        """
        pass
    
    @abstractmethod
    async def get_detection_count(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None
    ) -> int:
        """
        Get count of detections matching criteria.
        
        Args:
            camera_id: Optional camera filter
            time_range: Optional time range filter
            
        Returns:
            Count of matching detections
        """
        pass
    
    @abstractmethod
    async def delete_old_detections(
        self,
        older_than: datetime,
        batch_size: int = 1000
    ) -> int:
        """
        Delete detections older than specified date.
        
        Args:
            older_than: Delete detections before this timestamp
            batch_size: Number of records to delete per batch
            
        Returns:
            Number of detections deleted
        """
        pass
    
    @abstractmethod
    async def get_detection_statistics(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None
    ) -> Dict[str, Any]:
        """
        Get detection statistics for analysis.
        
        Args:
            camera_id: Optional camera filter
            time_range: Optional time range filter
            
        Returns:
            Dictionary containing detection statistics
        """
        pass
    
    @abstractmethod
    async def exists_detection(
        self,
        camera_id: CameraID,
        frame_id: FrameID,
        detection_bbox: Dict[str, float]
    ) -> bool:
        """
        Check if similar detection already exists.
        
        Args:
            camera_id: Camera identifier
            frame_id: Frame identifier
            detection_bbox: Bounding box coordinates
            
        Returns:
            True if similar detection exists
        """
        pass