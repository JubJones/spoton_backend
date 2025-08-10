"""
Tracking domain service providing core tracking business logic.

Encapsulates pure business logic for track lifecycle management,
track state transitions, and tracking quality assessment.
"""
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.detection.entities.detection import Detection
from app.domain.tracking.entities.track import Track, TrackState
from app.domain.tracking.value_objects.track_id import TrackID, GlobalTrackID
from app.domain.tracking.value_objects.velocity import Velocity

logger = logging.getLogger(__name__)


class TrackingService:
    """
    Tracking domain service for core tracking business logic.
    
    Manages track lifecycle, state transitions, and tracking quality
    while maintaining domain integrity. Maximum 400 lines per plan.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        max_time_since_update: int = 5
    ):
        """
        Initialize tracking service.
        
        Args:
            max_age: Maximum age before track deletion
            min_hits: Minimum hits for track confirmation
            max_time_since_update: Max frames without update before marking lost
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_time_since_update = max_time_since_update
        
        self._tracking_stats = {
            'tracks_created': 0,
            'tracks_deleted': 0,
            'tracks_confirmed': 0,
            'tracks_lost': 0,
            'state_transitions': 0
        }
        
        logger.debug(f"TrackingService initialized with max_age={max_age}, min_hits={min_hits}")
    
    def create_track(
        self,
        track_id: TrackID,
        initial_detection: Detection,
        camera_id: CameraID
    ) -> Track:
        """
        Create new track from detection.
        
        Args:
            track_id: Track identifier
            initial_detection: Initial detection
            camera_id: Camera identifier
            
        Returns:
            New Track instance
        """
        track = Track.create(
            track_id=track_id,
            camera_id=camera_id,
            initial_detection=initial_detection,
            state=TrackState.TENTATIVE
        )
        
        self._tracking_stats['tracks_created'] += 1
        logger.debug(f"Created new track: {track}")
        
        return track
    
    def update_track_with_detection(
        self,
        track: Track,
        detection: Detection
    ) -> Track:
        """
        Update track with new detection.
        
        Args:
            track: Track to update
            detection: New detection
            
        Returns:
            Updated Track instance
        """
        if not self._is_detection_compatible(track, detection):
            raise ValueError(f"Detection not compatible with track {track.track_id}")
        
        # Update track
        updated_track = track.update_with_detection(detection)
        
        # Check for state transitions
        new_track = self._evaluate_state_transitions(updated_track)
        
        if new_track.state != track.state:
            self._tracking_stats['state_transitions'] += 1
            logger.debug(f"Track {track.track_id} transitioned from {track.state} to {new_track.state}")
        
        return new_track
    
    def predict_track_position(self, track: Track) -> Track:
        """
        Predict track position for next frame.
        
        Args:
            track: Track to predict
            
        Returns:
            Track with predicted position
        """
        # Mark as missed (increments time_since_update)
        predicted_track = track.mark_missed()
        
        # Evaluate state based on prediction
        evaluated_track = self._evaluate_state_transitions(predicted_track)
        
        if evaluated_track.is_lost and not track.is_lost:
            self._tracking_stats['tracks_lost'] += 1
            logger.debug(f"Track {track.track_id} marked as lost")
        
        return evaluated_track
    
    def should_delete_track(self, track: Track) -> bool:
        """
        Determine if track should be deleted.
        
        Args:
            track: Track to evaluate
            
        Returns:
            True if track should be deleted
        """
        # Delete if too old
        if track.age > self.max_age:
            return True
        
        # Delete if lost for too long
        if track.is_lost and track.time_since_update > self.max_time_since_update * 2:
            return True
        
        # Delete tentative tracks that haven't been confirmed
        if track.is_tentative and track.age > 10 and track.hits < self.min_hits:
            return True
        
        return False
    
    def delete_track(self, track: Track) -> Track:
        """
        Mark track as deleted.
        
        Args:
            track: Track to delete
            
        Returns:
            Deleted Track instance
        """
        deleted_track = track.mark_deleted()
        self._tracking_stats['tracks_deleted'] += 1
        logger.debug(f"Deleted track: {track.track_id}")
        
        return deleted_track
    
    def calculate_track_quality(self, track: Track) -> float:
        """
        Calculate overall track quality score.
        
        Args:
            track: Track to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Hit rate quality (hits vs age)
        hit_rate = track.hits / max(track.age, 1)
        hit_quality = min(1.0, hit_rate)
        
        # Streak quality (recent consistency)
        streak_quality = min(1.0, track.hit_streak / 10.0)
        
        # Recency quality (how recently updated)
        recency_quality = max(0.0, 1.0 - track.time_since_update / 10.0)
        
        # Confidence quality
        confidence_quality = track.current_confidence
        
        # State quality
        state_quality = {
            TrackState.CONFIRMED: 1.0,
            TrackState.ACTIVE: 0.9,
            TrackState.TENTATIVE: 0.5,
            TrackState.LOST: 0.2,
            TrackState.DELETED: 0.0
        }.get(track.state, 0.0)
        
        # Weighted combination
        quality_score = (
            hit_quality * 0.25 +
            streak_quality * 0.20 +
            recency_quality * 0.20 +
            confidence_quality * 0.20 +
            state_quality * 0.15
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def find_track_associations(
        self,
        tracks: List[Track],
        detections: List[Detection],
        distance_threshold: float = 50.0
    ) -> Tuple[List[Tuple[Track, Detection]], List[Track], List[Detection]]:
        """
        Find optimal track-detection associations.
        
        Args:
            tracks: List of tracks
            detections: List of detections
            distance_threshold: Maximum distance for association
            
        Returns:
            Tuple of (associations, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], tracks.copy(), detections.copy()
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(tracks, detections)
        
        # Find optimal assignments using Hungarian algorithm (simplified)
        associations = []
        used_tracks = set()
        used_detections = set()
        
        # Simple greedy assignment for now (could be replaced with Hungarian algorithm)
        for t_idx, track in enumerate(tracks):
            best_detection_idx = None
            best_distance = float('inf')
            
            for d_idx, detection in enumerate(detections):
                if d_idx in used_detections:
                    continue
                
                distance = distance_matrix[t_idx][d_idx]
                if distance < distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_detection_idx = d_idx
            
            if best_detection_idx is not None:
                associations.append((track, detections[best_detection_idx]))
                used_tracks.add(t_idx)
                used_detections.add(best_detection_idx)
        
        # Collect unmatched tracks and detections
        unmatched_tracks = [track for i, track in enumerate(tracks) if i not in used_tracks]
        unmatched_detections = [det for i, det in enumerate(detections) if i not in used_detections]
        
        logger.debug(f"Found {len(associations)} associations, "
                    f"{len(unmatched_tracks)} unmatched tracks, "
                    f"{len(unmatched_detections)} unmatched detections")
        
        return associations, unmatched_tracks, unmatched_detections
    
    def assign_global_id(self, track: Track, global_id: GlobalTrackID) -> Track:
        """
        Assign global ID for cross-camera tracking.
        
        Args:
            track: Track to update
            global_id: Global track identifier
            
        Returns:
            Updated Track with global ID
        """
        updated_track = track.assign_global_id(global_id)
        logger.debug(f"Assigned global ID {global_id.short_id} to track {track.track_id}")
        
        return updated_track
    
    def merge_tracks(self, primary_track: Track, secondary_track: Track) -> Track:
        """
        Merge two tracks (for track re-identification scenarios).
        
        Args:
            primary_track: Primary track to keep
            secondary_track: Secondary track to merge in
            
        Returns:
            Merged Track instance
        """
        # Combine hit counts and select better metadata
        merged_hits = primary_track.hits + secondary_track.hits
        merged_age = max(primary_track.age, secondary_track.age)
        
        # Use the more recent detection data
        if secondary_track.last_updated_at > primary_track.last_updated_at:
            current_bbox = secondary_track.current_bbox
            current_confidence = secondary_track.current_confidence
            last_detection = secondary_track.last_detection
        else:
            current_bbox = primary_track.current_bbox
            current_confidence = primary_track.current_confidence
            last_detection = primary_track.last_detection
        
        # Combine feature histories
        combined_features = primary_track.feature_history + secondary_track.feature_history
        
        # Create merged track (keeping primary track ID)
        merged_track = Track(
            track_id=primary_track.track_id,
            camera_id=primary_track.camera_id,
            state=TrackState.CONFIRMED,  # Merged tracks are confirmed
            current_bbox=current_bbox,
            current_confidence=current_confidence,
            last_detection=last_detection,
            age=merged_age,
            hits=merged_hits,
            hit_streak=max(primary_track.hit_streak, secondary_track.hit_streak),
            time_since_update=0,  # Reset since we just updated
            velocity=primary_track.velocity or secondary_track.velocity,
            global_id=primary_track.global_id or secondary_track.global_id,
            feature_vector=primary_track.feature_vector or secondary_track.feature_vector,
            feature_history=combined_features,
            created_at=min(primary_track.created_at, secondary_track.created_at),
            last_updated_at=datetime.utcnow()
        )
        
        # Copy entity ID from primary track
        merged_track._id = primary_track._id
        merged_track.increment_version()
        
        logger.info(f"Merged tracks {primary_track.track_id} and {secondary_track.track_id}")
        return merged_track
    
    def get_active_tracks(self, tracks: List[Track]) -> List[Track]:
        """Get only active tracks."""
        return [track for track in tracks if track.is_active or track.is_confirmed]
    
    def get_tracks_by_camera(self, tracks: List[Track], camera_id: CameraID) -> List[Track]:
        """Get tracks for specific camera."""
        return [track for track in tracks if track.camera_id == camera_id]
    
    def calculate_tracking_statistics(self, tracks: List[Track]) -> Dict[str, any]:
        """
        Calculate comprehensive tracking statistics.
        
        Args:
            tracks: List of tracks to analyze
            
        Returns:
            Dictionary with tracking statistics
        """
        if not tracks:
            return {
                'total_tracks': 0,
                'active_tracks': 0,
                'confirmed_tracks': 0,
                'lost_tracks': 0,
                'tentative_tracks': 0,
                'average_age': 0.0,
                'average_quality': 0.0,
                'camera_distribution': {},
                'processing_statistics': self._tracking_stats.copy()
            }
        
        # State counts
        state_counts = {
            'active': sum(1 for t in tracks if t.is_active),
            'confirmed': sum(1 for t in tracks if t.is_confirmed),
            'lost': sum(1 for t in tracks if t.is_lost),
            'tentative': sum(1 for t in tracks if t.is_tentative),
            'deleted': sum(1 for t in tracks if t.is_deleted)
        }
        
        # Age statistics
        ages = [track.age for track in tracks if not track.is_deleted]
        average_age = sum(ages) / len(ages) if ages else 0.0
        
        # Quality statistics
        qualities = [self.calculate_track_quality(track) for track in tracks if not track.is_deleted]
        average_quality = sum(qualities) / len(qualities) if qualities else 0.0
        
        # Camera distribution
        camera_counts = {}
        for track in tracks:
            if not track.is_deleted:
                camera_key = str(track.camera_id)
                camera_counts[camera_key] = camera_counts.get(camera_key, 0) + 1
        
        return {
            'total_tracks': len(tracks),
            'active_tracks': state_counts['active'],
            'confirmed_tracks': state_counts['confirmed'],
            'lost_tracks': state_counts['lost'],
            'tentative_tracks': state_counts['tentative'],
            'deleted_tracks': state_counts['deleted'],
            'average_age': average_age,
            'average_quality': average_quality,
            'camera_distribution': camera_counts,
            'processing_statistics': self._tracking_stats.copy()
        }
    
    def _is_detection_compatible(self, track: Track, detection: Detection) -> bool:
        """Check if detection is compatible with track."""
        # Must be from same camera
        if track.camera_id != detection.camera_id:
            return False
        
        # Check class compatibility (if track has class info)
        if track.class_id is not None:
            if track.class_id != detection.detection_class.to_legacy_format():
                return False
        
        return True
    
    def _evaluate_state_transitions(self, track: Track) -> Track:
        """Evaluate and apply state transitions based on track history."""
        current_state = track.state
        new_state = current_state
        
        if track.is_tentative:
            # Tentative -> Confirmed
            if track.hits >= self.min_hits:
                new_state = TrackState.CONFIRMED
                self._tracking_stats['tracks_confirmed'] += 1
            # Tentative -> Lost (if too many misses)
            elif track.time_since_update >= self.max_time_since_update:
                new_state = TrackState.LOST
        
        elif track.is_active or track.is_confirmed:
            # Active/Confirmed -> Lost
            if track.time_since_update >= self.max_time_since_update:
                new_state = TrackState.LOST
        
        elif track.is_lost:
            # Lost -> Active (track recovered)
            if track.time_since_update == 0:  # Just updated
                new_state = TrackState.ACTIVE
        
        # Apply state change if needed
        if new_state != current_state:
            # Create new track with updated state
            updated_track = Track(
                track_id=track.track_id,
                camera_id=track.camera_id,
                state=new_state,
                current_bbox=track.current_bbox,
                current_confidence=track.current_confidence,
                last_detection=track.last_detection,
                age=track.age,
                hits=track.hits,
                hit_streak=track.hit_streak,
                time_since_update=track.time_since_update,
                velocity=track.velocity,
                previous_bbox=track.previous_bbox,
                global_id=track.global_id,
                feature_vector=track.feature_vector,
                feature_history=track.feature_history.copy(),
                created_at=track.created_at,
                last_updated_at=track.last_updated_at,
                class_id=track.class_id,
                metadata=track.metadata.copy()
            )
            
            updated_track._id = track._id
            updated_track.increment_version()
            return updated_track
        
        return track
    
    def _calculate_distance_matrix(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> List[List[float]]:
        """Calculate distance matrix between tracks and detections."""
        matrix = []
        
        for track in tracks:
            track_distances = []
            track_center = track.bbox_center
            
            for detection in detections:
                detection_center = detection.bbox_center
                
                # Euclidean distance between centers
                distance = np.sqrt(
                    (track_center[0] - detection_center[0]) ** 2 +
                    (track_center[1] - detection_center[1]) ** 2
                )
                
                track_distances.append(distance)
            
            matrix.append(track_distances)
        
        return matrix
    
    def get_processing_statistics(self) -> Dict[str, int]:
        """Get tracking processing statistics."""
        return self._tracking_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._tracking_stats = {
            'tracks_created': 0,
            'tracks_deleted': 0,
            'tracks_confirmed': 0,
            'tracks_lost': 0,
            'state_transitions': 0
        }
        logger.debug("Tracking processing statistics reset")