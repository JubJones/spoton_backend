"""
Tracking use cases for application layer.

Business logic for person tracking operations including track management,
cross-camera association, and trajectory analysis.
Maximum 300 lines per plan.
"""
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.frame_id import FrameID
from app.domain.tracking.entities.track import Track
from app.domain.tracking.value_objects.track_id import TrackID
from app.domain.detection.entities.detection import Detection
from app.domain.tracking.services.tracking_service import TrackingService
# from app.domain.tracking.services.cross_camera_tracking_service import CrossCameraTrackingService

logger = logging.getLogger(__name__)


@dataclass
class TrackingRequest:
    """Request for tracking update."""
    camera_id: CameraID
    frame_id: FrameID
    detections: List[Detection]
    timestamp: datetime


@dataclass
class TrackingResult:
    """Result of tracking operation."""
    camera_id: CameraID
    frame_id: FrameID
    active_tracks: List[Track]
    new_tracks: List[Track]
    ended_tracks: List[Track]
    processing_time_ms: float
    timestamp: datetime


class SingleCameraTrackingUseCase:
    """
    Single camera tracking use case for application layer.
    
    Orchestrates tracking operations within a single camera view
    with business logic for track lifecycle management.
    """
    
    def __init__(self, tracking_service: TrackingService):
        """
        Initialize single camera tracking use case.
        
        Args:
            tracking_service: Domain tracking service
        """
        self.tracking_service = tracking_service
        
        # Tracking statistics
        self._tracking_stats = {
            'frames_processed': 0,
            'tracks_created': 0,
            'tracks_updated': 0,
            'tracks_ended': 0,
            'avg_track_duration_seconds': 0.0
        }
        
        logger.debug("SingleCameraTrackingUseCase initialized")
    
    async def update_tracks_with_detections(
        self,
        request: TrackingRequest
    ) -> TrackingResult:
        """
        Update tracks with new detections.
        
        Args:
            request: Tracking request with detections
            
        Returns:
            Tracking result with updated tracks
        """
        start_time = datetime.utcnow()
        
        try:
            # Get current active tracks for camera
            active_tracks_before = self.tracking_service.get_active_tracks(request.camera_id)
            
            # Update tracks with new detections
            tracking_result = await self.tracking_service.update_tracks(
                camera_id=request.camera_id,
                frame_id=request.frame_id,
                detections=request.detections,
                timestamp=request.timestamp
            )
            
            # Get updated active tracks
            active_tracks_after = self.tracking_service.get_active_tracks(request.camera_id)
            
            # Identify new and ended tracks
            new_tracks = self._identify_new_tracks(active_tracks_before, active_tracks_after)
            ended_tracks = self._identify_ended_tracks(active_tracks_before, active_tracks_after)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_tracking_statistics(new_tracks, ended_tracks)
            
            # Create result
            result = TrackingResult(
                camera_id=request.camera_id,
                frame_id=request.frame_id,
                active_tracks=list(active_tracks_after.values()),
                new_tracks=new_tracks,
                ended_tracks=ended_tracks,
                processing_time_ms=processing_time,
                timestamp=request.timestamp
            )
            
            logger.debug(f"Updated tracks for camera {request.camera_id}: "
                        f"{len(active_tracks_after)} active, {len(new_tracks)} new, {len(ended_tracks)} ended")
            
            return result
            
        except Exception as e:
            logger.error(f"Tracking update failed for camera {request.camera_id}: {e}")
            raise
    
    def get_track_trajectories(
        self,
        camera_id: CameraID,
        track_ids: Optional[List[TrackID]] = None,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[TrackID, List[Dict[str, Any]]]:
        """
        Get track trajectories for analysis.
        
        Args:
            camera_id: Camera identifier
            track_ids: Optional specific track IDs
            time_range: Optional time range filter
            
        Returns:
            Dictionary of track trajectories
        """
        tracks = self.tracking_service.get_tracks_by_camera(camera_id)
        
        # Filter by track IDs if specified
        if track_ids:
            tracks = {tid: track for tid, track in tracks.items() if tid in track_ids}
        
        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            filtered_tracks = {}
            
            for track_id, track in tracks.items():
                if start_time <= track.created_at <= end_time:
                    filtered_tracks[track_id] = track
            
            tracks = filtered_tracks
        
        # Extract trajectories
        trajectories = {}
        for track_id, track in tracks.items():
            trajectory = []
            
            for position in track.position_history:
                trajectory.append({
                    'timestamp': position.timestamp.isoformat(),
                    'x': position.bbox.x,
                    'y': position.bbox.y,
                    'width': position.bbox.width,
                    'height': position.bbox.height,
                    'confidence': position.confidence.value
                })
            
            trajectories[track_id] = trajectory
        
        return trajectories
    
    def analyze_track_behavior(
        self,
        track_id: TrackID,
        camera_id: CameraID
    ) -> Dict[str, Any]:
        """
        Analyze individual track behavior patterns.
        
        Args:
            track_id: Track identifier
            camera_id: Camera identifier
            
        Returns:
            Track behavior analysis
        """
        track = self.tracking_service.get_track(camera_id, track_id)
        if not track:
            return {'error': 'Track not found'}
        
        # Basic behavior metrics
        duration_seconds = (datetime.utcnow() - track.created_at).total_seconds()
        
        # Velocity analysis
        velocity_stats = {
            'avg_speed': 0.0,
            'max_speed': 0.0,
            'is_stationary': False,
            'direction_changes': 0
        }
        
        if track.velocity:
            velocity_stats['avg_speed'] = track.velocity.speed
            velocity_stats['is_stationary'] = not track.velocity.is_moving()
        
        # Position analysis
        position_count = len(track.position_history)
        total_distance = 0.0
        
        if position_count > 1:
            for i in range(1, position_count):
                prev_pos = track.position_history[i-1]
                curr_pos = track.position_history[i]
                
                # Calculate distance moved
                dx = curr_pos.bbox.x - prev_pos.bbox.x
                dy = curr_pos.bbox.y - prev_pos.bbox.y
                distance = (dx ** 2 + dy ** 2) ** 0.5
                total_distance += distance
        
        return {
            'track_id': str(track_id),
            'camera_id': str(camera_id),
            'duration_seconds': duration_seconds,
            'position_updates': position_count,
            'total_distance_pixels': total_distance,
            'velocity_stats': velocity_stats,
            'track_state': track.state.value,
            'confidence_avg': sum(pos.confidence.value for pos in track.position_history) / max(position_count, 1)
        }
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking processing statistics."""
        return self._tracking_stats.copy()
    
    def _identify_new_tracks(
        self,
        tracks_before: Dict[TrackID, Track],
        tracks_after: Dict[TrackID, Track]
    ) -> List[Track]:
        """Identify newly created tracks."""
        before_ids = set(tracks_before.keys())
        after_ids = set(tracks_after.keys())
        new_track_ids = after_ids - before_ids
        
        return [tracks_after[track_id] for track_id in new_track_ids]
    
    def _identify_ended_tracks(
        self,
        tracks_before: Dict[TrackID, Track],
        tracks_after: Dict[TrackID, Track]
    ) -> List[Track]:
        """Identify ended tracks."""
        before_ids = set(tracks_before.keys())
        after_ids = set(tracks_after.keys())
        ended_track_ids = before_ids - after_ids
        
        return [tracks_before[track_id] for track_id in ended_track_ids]
    
    def _update_tracking_statistics(
        self,
        new_tracks: List[Track],
        ended_tracks: List[Track]
    ) -> None:
        """Update tracking statistics."""
        self._tracking_stats['frames_processed'] += 1
        self._tracking_stats['tracks_created'] += len(new_tracks)
        self._tracking_stats['tracks_ended'] += len(ended_tracks)
        
        # Update average track duration for ended tracks
        if ended_tracks:
            total_duration = sum(
                (datetime.utcnow() - track.created_at).total_seconds()
                for track in ended_tracks
            )
            avg_duration = total_duration / len(ended_tracks)
            
            current_avg = self._tracking_stats['avg_track_duration_seconds']
            tracks_ended_count = self._tracking_stats['tracks_ended']
            
            # Update running average
            if tracks_ended_count > len(ended_tracks):
                previous_count = tracks_ended_count - len(ended_tracks)
                self._tracking_stats['avg_track_duration_seconds'] = (
                    (current_avg * previous_count + total_duration) / tracks_ended_count
                )
            else:
                self._tracking_stats['avg_track_duration_seconds'] = avg_duration


class CrossCameraTrackingUseCase:
    """
    Cross-camera tracking use case for application layer.
    
    Orchestrates cross-camera track association and person re-identification
    with business logic for multi-camera coordination.
    """
    
    def __init__(self, cross_camera_service):  # CrossCameraTrackingService
        """
        Initialize cross-camera tracking use case.
        
        Args:
            cross_camera_service: Cross-camera tracking service
        """
        self.cross_camera_service = cross_camera_service
        
        # Cross-camera statistics
        self._cross_camera_stats = {
            'associations_attempted': 0,
            'associations_successful': 0,
            'handoffs_completed': 0,
            'avg_association_confidence': 0.0
        }
        
        logger.debug("CrossCameraTrackingUseCase initialized")
    
    async def associate_cross_camera_tracks(
        self,
        source_camera_id: CameraID,
        source_track_id: TrackID,
        target_cameras: List[CameraID],
        time_window_seconds: int = 30
    ) -> Dict[str, Any]:
        """
        Associate tracks across cameras using re-identification.
        
        Args:
            source_camera_id: Source camera identifier
            source_track_id: Source track identifier
            target_cameras: List of target cameras to search
            time_window_seconds: Time window for association
            
        Returns:
            Association results
        """
        try:
            # Get source track
            source_track = self.cross_camera_service.get_track(source_camera_id, source_track_id)
            if not source_track:
                return {'error': 'Source track not found'}
            
            # Perform cross-camera association
            association_results = await self.cross_camera_service.associate_tracks(
                source_camera_id=source_camera_id,
                source_track_id=source_track_id,
                target_cameras=target_cameras,
                time_window=timedelta(seconds=time_window_seconds)
            )
            
            # Update statistics
            self._update_association_statistics(association_results)
            
            logger.info(f"Cross-camera association completed for track {source_track_id}")
            return association_results
            
        except Exception as e:
            logger.error(f"Cross-camera association failed: {e}")
            return {'error': str(e)}
    
    def get_person_journey(
        self,
        person_id: str,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get complete person journey across cameras.
        
        Args:
            person_id: Person identifier
            time_range: Optional time range filter
            
        Returns:
            Person journey information
        """
        try:
            journey_data = self.cross_camera_service.get_person_journey(
                person_id, time_range
            )
            
            # Enrich journey data with analytics
            if journey_data:
                journey_analytics = self._analyze_journey(journey_data)
                journey_data['analytics'] = journey_analytics
            
            return journey_data
            
        except Exception as e:
            logger.error(f"Failed to get person journey: {e}")
            return {'error': str(e)}
    
    def get_cross_camera_statistics(self) -> Dict[str, Any]:
        """Get cross-camera tracking statistics."""
        return self._cross_camera_stats.copy()
    
    def _analyze_journey(self, journey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze person journey patterns."""
        camera_visits = journey_data.get('camera_sequence', [])
        
        analytics = {
            'total_cameras_visited': len(set(visit['camera_id'] for visit in camera_visits)),
            'total_duration_seconds': 0.0,
            'avg_dwell_time_seconds': 0.0,
            'movement_pattern': 'linear'  # Simplified
        }
        
        if len(camera_visits) > 1:
            start_time = datetime.fromisoformat(camera_visits[0]['entry_time'])
            end_time = datetime.fromisoformat(camera_visits[-1]['exit_time'])
            analytics['total_duration_seconds'] = (end_time - start_time).total_seconds()
        
        return analytics
    
    def _update_association_statistics(self, results: Dict[str, Any]) -> None:
        """Update cross-camera association statistics."""
        self._cross_camera_stats['associations_attempted'] += 1
        
        if results.get('success', False):
            self._cross_camera_stats['associations_successful'] += 1
            
            # Update confidence average
            confidence = results.get('confidence', 0.0)
            current_avg = self._cross_camera_stats['avg_association_confidence']
            successful_count = self._cross_camera_stats['associations_successful']
            
            self._cross_camera_stats['avg_association_confidence'] = (
                (current_avg * (successful_count - 1) + confidence) / successful_count
            )