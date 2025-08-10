"""
Tracking repository interface for application layer.

Abstract interface defining tracking data access operations.
Follows repository pattern with clean separation of concerns.
"""
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from abc import ABC, abstractmethod

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.tracking.entities.track import Track
from app.domain.tracking.value_objects.track_id import TrackID
from app.domain.shared.value_objects.time_range import TimeRange


class TrackingRepository(ABC):
    """
    Abstract tracking repository interface.
    
    Defines the contract for track data persistence and retrieval
    without coupling to specific database implementations.
    """
    
    @abstractmethod
    async def save_track(self, track: Track) -> bool:
        """
        Save a track to persistent storage.
        
        Args:
            track: Track entity to save
            
        Returns:
            True if save successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def update_track(self, track: Track) -> bool:
        """
        Update an existing track in persistent storage.
        
        Args:
            track: Updated track entity
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_track_by_id(
        self,
        camera_id: CameraID,
        track_id: TrackID
    ) -> Optional[Track]:
        """
        Get track by camera and track ID.
        
        Args:
            camera_id: Camera identifier
            track_id: Track identifier
            
        Returns:
            Track entity or None if not found
        """
        pass
    
    @abstractmethod
    async def get_active_tracks(
        self,
        camera_id: CameraID
    ) -> Dict[TrackID, Track]:
        """
        Get all currently active tracks for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary of active tracks
        """
        pass
    
    @abstractmethod
    async def get_tracks_by_camera(
        self,
        camera_id: CameraID,
        time_range: Optional[TimeRange] = None,
        include_ended: bool = False
    ) -> Dict[TrackID, Track]:
        """
        Get tracks for specific camera.
        
        Args:
            camera_id: Camera identifier
            time_range: Optional time range filter
            include_ended: Whether to include ended tracks
            
        Returns:
            Dictionary of track entities
        """
        pass
    
    @abstractmethod
    async def get_tracks_by_time_range(
        self,
        time_range: TimeRange,
        camera_ids: Optional[List[CameraID]] = None
    ) -> List[Track]:
        """
        Get tracks within time range across cameras.
        
        Args:
            time_range: Time range for track search
            camera_ids: Optional list of camera filters
            
        Returns:
            List of track entities
        """
        pass
    
    @abstractmethod
    async def mark_track_ended(
        self,
        camera_id: CameraID,
        track_id: TrackID,
        end_time: datetime
    ) -> bool:
        """
        Mark a track as ended.
        
        Args:
            camera_id: Camera identifier
            track_id: Track identifier
            end_time: Track end timestamp
            
        Returns:
            True if update successful
        """
        pass
    
    @abstractmethod
    async def get_track_count(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None,
        active_only: bool = False
    ) -> int:
        """
        Get count of tracks matching criteria.
        
        Args:
            camera_id: Optional camera filter
            time_range: Optional time range filter
            active_only: Count only active tracks
            
        Returns:
            Count of matching tracks
        """
        pass
    
    @abstractmethod
    async def delete_old_tracks(
        self,
        older_than: datetime,
        batch_size: int = 1000
    ) -> int:
        """
        Delete tracks older than specified date.
        
        Args:
            older_than: Delete tracks before this timestamp
            batch_size: Number of records to delete per batch
            
        Returns:
            Number of tracks deleted
        """
        pass
    
    @abstractmethod
    async def get_track_statistics(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None
    ) -> Dict[str, Any]:
        """
        Get tracking statistics for analysis.
        
        Args:
            camera_id: Optional camera filter
            time_range: Optional time range filter
            
        Returns:
            Dictionary containing tracking statistics
        """
        pass
    
    @abstractmethod
    async def get_cross_camera_associations(
        self,
        person_id: str,
        time_range: Optional[TimeRange] = None
    ) -> List[Dict[str, Any]]:
        """
        Get cross-camera track associations for a person.
        
        Args:
            person_id: Person identifier
            time_range: Optional time range filter
            
        Returns:
            List of track associations across cameras
        """
        pass
    
    @abstractmethod
    async def save_cross_camera_association(
        self,
        person_id: str,
        camera_associations: List[Dict[str, Any]]
    ) -> bool:
        """
        Save cross-camera track associations.
        
        Args:
            person_id: Person identifier
            camera_associations: List of camera-track associations
            
        Returns:
            True if save successful
        """
        pass