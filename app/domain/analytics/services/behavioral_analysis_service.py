"""
Behavioral analysis domain service for person behavior analysis.

Focused service for analyzing individual person behavior patterns,
replacing the massive 1,440-line behavioral analysis service.
Maximum 400 lines as per refactoring plan.
"""
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict

from app.domain.tracking.entities.track import Track
from app.domain.mapping.value_objects.coordinates import WorldCoordinates
from app.domain.analytics.value_objects.behavioral_metrics import (
    BehavioralFeatures, MovementMetrics, TemporalMetrics, SpatialMetrics,
    MovementPattern, BehavioralAnomalyType, SocialInteractionType
)

logger = logging.getLogger(__name__)


class BehavioralAnalysisService:
    """
    Behavioral analysis domain service.
    
    Analyzes individual person behavior patterns from tracking data
    to extract meaningful behavioral insights and detect anomalies.
    """
    
    def __init__(
        self,
        min_track_duration: float = 5.0,  # Minimum seconds for analysis
        speed_anomaly_threshold: float = 3.0,  # Speed threshold for anomaly
        dwell_time_threshold: float = 1800.0  # Dwell time threshold (30 min)
    ):
        """
        Initialize behavioral analysis service.
        
        Args:
            min_track_duration: Minimum track duration for analysis
            speed_anomaly_threshold: Speed threshold for anomaly detection
            dwell_time_threshold: Dwell time threshold for anomaly detection
        """
        self.min_track_duration = min_track_duration
        self.speed_anomaly_threshold = speed_anomaly_threshold
        self.dwell_time_threshold = dwell_time_threshold
        
        self._analysis_stats = {
            'tracks_analyzed': 0,
            'behaviors_detected': 0,
            'anomalies_found': 0
        }
        
        logger.debug("BehavioralAnalysisService initialized")
    
    def analyze_person_behavior(
        self,
        person_tracks: List[Track],
        world_coordinates: List[WorldCoordinates] = None
    ) -> Optional[BehavioralFeatures]:
        """
        Analyze behavior of a person from their tracking history.
        
        Args:
            person_tracks: List of tracks for the same person
            world_coordinates: Optional world coordinates for each track
            
        Returns:
            BehavioralFeatures if analysis possible, None otherwise
        """
        if not person_tracks:
            return None
        
        # Filter tracks by minimum duration
        valid_tracks = [
            track for track in person_tracks 
            if track.age_seconds >= self.min_track_duration
        ]
        
        if not valid_tracks:
            logger.debug(f"No tracks meet minimum duration {self.min_track_duration}s")
            return None
        
        try:
            # Extract person ID (use global ID or first track ID)
            person_id = (valid_tracks[0].global_id.id if valid_tracks[0].has_global_id 
                        else str(valid_tracks[0].track_id))
            
            # Calculate behavioral metrics
            movement_metrics = self._calculate_movement_metrics(valid_tracks, world_coordinates)
            temporal_metrics = self._calculate_temporal_metrics(valid_tracks)
            spatial_metrics = self._calculate_spatial_metrics(valid_tracks, world_coordinates)
            
            # Analyze social interactions (simplified for now)
            social_interaction = self._analyze_social_interaction(valid_tracks)
            
            # Detect anomalies
            anomalies = self._detect_behavioral_anomalies(
                movement_metrics, temporal_metrics, spatial_metrics
            )
            
            # Create behavioral features
            behavior = BehavioralFeatures.create(
                person_id=person_id,
                movement_metrics=movement_metrics,
                temporal_metrics=temporal_metrics,
                spatial_metrics=spatial_metrics,
                social_interaction_type=social_interaction,
                anomaly_indicators=anomalies
            )
            
            self._analysis_stats['tracks_analyzed'] += len(valid_tracks)
            self._analysis_stats['behaviors_detected'] += 1
            if behavior.has_anomalies:
                self._analysis_stats['anomalies_found'] += 1
            
            logger.debug(f"Analyzed behavior for person {person_id}: {behavior.primary_pattern.value}")
            return behavior
            
        except Exception as e:
            logger.error(f"Failed to analyze person behavior: {e}")
            return None
    
    def classify_movement_pattern(
        self,
        tracks: List[Track],
        world_coordinates: List[WorldCoordinates] = None
    ) -> MovementPattern:
        """
        Classify movement pattern from track sequence.
        
        Args:
            tracks: Sequence of tracks
            world_coordinates: Optional world coordinates
            
        Returns:
            Classified movement pattern
        """
        if not tracks:
            return MovementPattern.STATIONARY
        
        movement_metrics = self._calculate_movement_metrics(tracks, world_coordinates)
        return movement_metrics.get_movement_classification()
    
    def detect_anomalies(
        self,
        behavior: BehavioralFeatures,
        historical_baseline: Optional[Dict[str, float]] = None
    ) -> Set[BehavioralAnomalyType]:
        """
        Detect behavioral anomalies using statistical analysis.
        
        Args:
            behavior: Behavioral features to analyze
            historical_baseline: Optional historical baseline metrics
            
        Returns:
            Set of detected anomaly types
        """
        anomalies = set()
        
        # Speed-based anomalies
        if behavior.movement_metrics.average_speed > self.speed_anomaly_threshold:
            anomalies.add(BehavioralAnomalyType.UNUSUAL_SPEED)
        
        # Dwell time anomalies
        if behavior.temporal_metrics.dwell_time_seconds > self.dwell_time_threshold:
            anomalies.add(BehavioralAnomalyType.UNUSUAL_DWELL_TIME)
        
        # Path efficiency anomalies
        if behavior.movement_metrics.path_efficiency < 0.2:
            anomalies.add(BehavioralAnomalyType.UNUSUAL_PATH)
        
        # Compare against historical baseline if available
        if historical_baseline:
            anomalies.update(self._compare_against_baseline(behavior, historical_baseline))
        
        return anomalies
    
    def analyze_crowd_behavior(
        self,
        all_tracks: List[Track],
        time_window: timedelta = timedelta(minutes=5)
    ) -> Dict[str, any]:
        """
        Analyze crowd-level behavioral patterns.
        
        Args:
            all_tracks: All tracks in the analysis period
            time_window: Time window for crowd analysis
            
        Returns:
            Crowd behavior analysis results
        """
        if not all_tracks:
            return {'crowd_density': 0, 'flow_patterns': [], 'anomalies': []}
        
        # Group tracks by time windows
        time_groups = self._group_tracks_by_time(all_tracks, time_window)
        
        crowd_metrics = {
            'crowd_density': len(all_tracks),
            'average_speed': np.mean([
                track.velocity.speed if track.velocity else 0.0 
                for track in all_tracks
            ]),
            'flow_patterns': [],
            'congestion_areas': [],
            'anomalies': []
        }
        
        # Analyze each time group
        for time_group in time_groups:
            if len(time_group) > 10:  # Crowd threshold
                crowd_metrics['anomalies'].append(BehavioralAnomalyType.CROWD_FORMATION)
        
        return crowd_metrics
    
    def _calculate_movement_metrics(
        self,
        tracks: List[Track],
        world_coordinates: List[WorldCoordinates] = None
    ) -> MovementMetrics:
        """Calculate movement-based metrics."""
        speeds = []
        distances = []
        total_distance = 0.0
        
        for i, track in enumerate(tracks):
            if track.velocity:
                speeds.append(track.velocity.speed)
            
            # Calculate distances if world coordinates available
            if world_coordinates and i > 0:
                distance = world_coordinates[i].distance_to(world_coordinates[i-1])
                distances.append(distance)
                total_distance += distance
        
        if not speeds:
            speeds = [0.0]
        if not distances:
            distances = [0.0]
        
        average_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_variance = np.var(speeds)
        
        # Calculate path efficiency (direct vs actual distance)
        if world_coordinates and len(world_coordinates) > 1:
            direct_distance = world_coordinates[0].distance_to(world_coordinates[-1])
            path_efficiency = direct_distance / max(total_distance, 0.001)
        else:
            path_efficiency = 1.0
        
        return MovementMetrics.create(
            average_speed=average_speed,
            max_speed=max_speed,
            speed_variance=speed_variance,
            total_distance=total_distance,
            path_efficiency=min(1.0, path_efficiency)
        )
    
    def _calculate_temporal_metrics(self, tracks: List[Track]) -> TemporalMetrics:
        """Calculate time-based metrics."""
        if not tracks:
            return TemporalMetrics.create(0.0, 0.0, 0.0, 0.0)
        
        # Sort tracks by creation time
        sorted_tracks = sorted(tracks, key=lambda t: t.created_at)
        
        # Calculate total time span
        total_time = (sorted_tracks[-1].last_updated_at - sorted_tracks[0].created_at).total_seconds()
        
        # Estimate active vs stationary time
        active_time = sum(track.age_seconds for track in tracks if track.is_moving)
        stationary_time = sum(track.age_seconds for track in tracks if not track.is_moving)
        
        # Calculate dwell time (time in environment)
        dwell_time = total_time
        
        return TemporalMetrics.create(
            total_time_seconds=total_time,
            active_time_seconds=active_time,
            stationary_time_seconds=stationary_time,
            dwell_time_seconds=dwell_time
        )
    
    def _calculate_spatial_metrics(
        self,
        tracks: List[Track],
        world_coordinates: List[WorldCoordinates] = None
    ) -> SpatialMetrics:
        """Calculate space-based metrics."""
        visited_zones = set()
        zone_transitions = 0
        boundary_crossings = 0
        area_coverage = 0.0
        
        # Extract spatial information from tracks
        for track in tracks:
            # For now, use camera as zone proxy
            visited_zones.add(str(track.camera_id))
        
        # Calculate area coverage if world coordinates available
        if world_coordinates and len(world_coordinates) > 1:
            x_coords = [coord.x for coord in world_coordinates]
            y_coords = [coord.y for coord in world_coordinates]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            area_coverage = width * height
        
        zone_transitions = max(0, len(visited_zones) - 1)
        
        return SpatialMetrics.create(
            area_coverage_sqm=area_coverage,
            visited_zones=visited_zones,
            zone_transitions=zone_transitions,
            boundary_crossings=boundary_crossings
        )
    
    def _analyze_social_interaction(self, tracks: List[Track]) -> SocialInteractionType:
        """Analyze social interaction type (simplified)."""
        # For now, just return individual - more sophisticated analysis would
        # look at proximity to other tracks, group formation, etc.
        return SocialInteractionType.INDIVIDUAL
    
    def _detect_behavioral_anomalies(
        self,
        movement_metrics: MovementMetrics,
        temporal_metrics: TemporalMetrics,
        spatial_metrics: SpatialMetrics
    ) -> Set[BehavioralAnomalyType]:
        """Detect anomalies from behavioral metrics."""
        anomalies = set()
        
        # Speed anomalies
        if movement_metrics.average_speed > self.speed_anomaly_threshold:
            anomalies.add(BehavioralAnomalyType.UNUSUAL_SPEED)
        
        # Dwell time anomalies
        if temporal_metrics.dwell_time_seconds > self.dwell_time_threshold:
            anomalies.add(BehavioralAnomalyType.UNUSUAL_DWELL_TIME)
        
        # Path anomalies
        if movement_metrics.path_efficiency < 0.2:
            anomalies.add(BehavioralAnomalyType.UNUSUAL_PATH)
        
        return anomalies
    
    def _compare_against_baseline(
        self,
        behavior: BehavioralFeatures,
        baseline: Dict[str, float]
    ) -> Set[BehavioralAnomalyType]:
        """Compare behavior against historical baseline."""
        anomalies = set()
        
        # Compare speed against baseline
        if 'average_speed' in baseline:
            if behavior.movement_metrics.average_speed > baseline['average_speed'] * 2:
                anomalies.add(BehavioralAnomalyType.UNUSUAL_SPEED)
        
        # Compare dwell time against baseline
        if 'dwell_time' in baseline:
            if behavior.temporal_metrics.dwell_time_seconds > baseline['dwell_time'] * 2:
                anomalies.add(BehavioralAnomalyType.UNUSUAL_DWELL_TIME)
        
        return anomalies
    
    def _group_tracks_by_time(
        self,
        tracks: List[Track],
        time_window: timedelta
    ) -> List[List[Track]]:
        """Group tracks by time windows."""
        if not tracks:
            return []
        
        # Sort tracks by creation time
        sorted_tracks = sorted(tracks, key=lambda t: t.created_at)
        
        groups = []
        current_group = [sorted_tracks[0]]
        current_time = sorted_tracks[0].created_at
        
        for track in sorted_tracks[1:]:
            if track.created_at - current_time <= time_window:
                current_group.append(track)
            else:
                groups.append(current_group)
                current_group = [track]
                current_time = track.created_at
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def get_analysis_statistics(self) -> Dict[str, int]:
        """Get behavioral analysis statistics."""
        return self._analysis_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset analysis statistics."""
        self._analysis_stats = {
            'tracks_analyzed': 0,
            'behaviors_detected': 0,
            'anomalies_found': 0
        }
        logger.debug("Behavioral analysis statistics reset")