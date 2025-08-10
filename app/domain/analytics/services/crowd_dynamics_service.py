"""
Crowd dynamics domain service for analyzing group behavior patterns.

Focused service for crowd flow analysis, congestion detection,
and group interaction patterns. Maximum 250 lines per plan.
"""
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict

from app.domain.tracking.entities.track import Track
from app.domain.mapping.value_objects.coordinates import WorldCoordinates
from app.domain.analytics.value_objects.behavioral_metrics import (
    SocialInteractionType, BehavioralAnomalyType
)

logger = logging.getLogger(__name__)


class CrowdDynamicsService:
    """
    Crowd dynamics domain service.
    
    Analyzes crowd-level behavioral patterns including flow analysis,
    density calculations, and group formation detection.
    """
    
    def __init__(
        self,
        crowd_density_threshold: int = 5,  # People per square meter
        group_proximity_threshold: float = 2.0,  # Meters
        congestion_threshold: float = 0.5  # Speed threshold for congestion
    ):
        """
        Initialize crowd dynamics service.
        
        Args:
            crowd_density_threshold: Density threshold for crowd detection
            group_proximity_threshold: Distance threshold for group detection
            congestion_threshold: Speed threshold for congestion detection
        """
        self.crowd_density_threshold = crowd_density_threshold
        self.group_proximity_threshold = group_proximity_threshold
        self.congestion_threshold = congestion_threshold
        
        self._dynamics_stats = {
            'crowds_analyzed': 0,
            'groups_detected': 0,
            'congestion_events': 0
        }
        
        logger.debug("CrowdDynamicsService initialized")
    
    def analyze_crowd_density(
        self,
        tracks: List[Track],
        world_coordinates: List[WorldCoordinates],
        area_bounds: Tuple[float, float, float, float] = None
    ) -> Dict[str, any]:
        """
        Analyze crowd density in a given area.
        
        Args:
            tracks: List of active tracks
            world_coordinates: World coordinates for each track
            area_bounds: Optional area bounds (x_min, y_min, x_max, y_max)
            
        Returns:
            Crowd density analysis results
        """
        if not tracks or not world_coordinates:
            return {'density': 0.0, 'people_count': 0, 'area_sqm': 0.0}
        
        # Calculate area
        if area_bounds:
            x_min, y_min, x_max, y_max = area_bounds
            area_sqm = (x_max - x_min) * (y_max - y_min)
        else:
            # Calculate bounding box from coordinates
            x_coords = [coord.x for coord in world_coordinates]
            y_coords = [coord.y for coord in world_coordinates]
            
            area_sqm = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        
        people_count = len(tracks)
        density = people_count / max(area_sqm, 1.0)  # People per square meter
        
        analysis_result = {
            'density': density,
            'people_count': people_count,
            'area_sqm': area_sqm,
            'is_crowded': density > self.crowd_density_threshold,
            'density_level': self._classify_density_level(density)
        }
        
        self._dynamics_stats['crowds_analyzed'] += 1
        
        logger.debug(f"Crowd density analysis: {density:.2f} people/sqm, {people_count} people")
        return analysis_result
    
    def detect_group_formations(
        self,
        tracks: List[Track],
        world_coordinates: List[WorldCoordinates]
    ) -> List[Dict[str, any]]:
        """
        Detect group formations based on proximity and movement patterns.
        
        Args:
            tracks: List of tracks to analyze
            world_coordinates: World coordinates for each track
            
        Returns:
            List of detected groups with metadata
        """
        if len(tracks) < 2 or len(world_coordinates) != len(tracks):
            return []
        
        # Calculate distance matrix between all people
        distance_matrix = self._calculate_distance_matrix(world_coordinates)
        
        # Use clustering to find groups
        groups = self._find_proximity_groups(tracks, distance_matrix)
        
        # Analyze each group
        group_analyses = []
        for group_indices in groups:
            if len(group_indices) >= 2:  # Minimum group size
                group_tracks = [tracks[i] for i in group_indices]
                group_coords = [world_coordinates[i] for i in group_indices]
                
                group_analysis = self._analyze_group(group_tracks, group_coords)
                group_analyses.append(group_analysis)
                
                self._dynamics_stats['groups_detected'] += 1
        
        return group_analyses
    
    def analyze_flow_patterns(
        self,
        tracks: List[Track],
        world_coordinates: List[WorldCoordinates],
        time_window: timedelta = timedelta(minutes=5)
    ) -> Dict[str, any]:
        """
        Analyze crowd flow patterns and movement directions.
        
        Args:
            tracks: List of tracks with movement data
            world_coordinates: World coordinates for analysis
            time_window: Time window for flow analysis
            
        Returns:
            Flow pattern analysis results
        """
        if not tracks:
            return {'flow_direction': None, 'flow_speed': 0.0, 'flow_consistency': 0.0}
        
        # Extract movement vectors
        movement_vectors = []
        speeds = []
        
        for track in tracks:
            if track.velocity and track.velocity.is_moving():
                # Get movement vector
                velocity_vector = (track.velocity.vx, track.velocity.vy)
                movement_vectors.append(velocity_vector)
                speeds.append(track.velocity.speed)
        
        if not movement_vectors:
            return {'flow_direction': None, 'flow_speed': 0.0, 'flow_consistency': 0.0}
        
        # Calculate dominant flow direction
        avg_vx = np.mean([v[0] for v in movement_vectors])
        avg_vy = np.mean([v[1] for v in movement_vectors])
        flow_direction = np.arctan2(avg_vy, avg_vx)  # Radians
        
        # Calculate flow consistency (how aligned movements are)
        flow_consistency = self._calculate_flow_consistency(movement_vectors)
        
        # Calculate average flow speed
        avg_flow_speed = np.mean(speeds)
        
        flow_analysis = {
            'flow_direction_radians': flow_direction,
            'flow_direction_degrees': np.degrees(flow_direction),
            'flow_speed': avg_flow_speed,
            'flow_consistency': flow_consistency,
            'people_in_flow': len(movement_vectors),
            'is_congested': avg_flow_speed < self.congestion_threshold
        }
        
        if flow_analysis['is_congested']:
            self._dynamics_stats['congestion_events'] += 1
        
        return flow_analysis
    
    def detect_crowd_anomalies(
        self,
        tracks: List[Track],
        world_coordinates: List[WorldCoordinates],
        historical_baseline: Optional[Dict[str, float]] = None
    ) -> Set[BehavioralAnomalyType]:
        """
        Detect crowd-level behavioral anomalies.
        
        Args:
            tracks: Current tracks
            world_coordinates: World coordinates
            historical_baseline: Optional historical baseline for comparison
            
        Returns:
            Set of detected crowd anomalies
        """
        anomalies = set()
        
        if not tracks:
            return anomalies
        
        # Analyze current crowd characteristics
        density_analysis = self.analyze_crowd_density(tracks, world_coordinates)
        flow_analysis = self.analyze_flow_patterns(tracks, world_coordinates)
        group_formations = self.detect_group_formations(tracks, world_coordinates)
        
        # Detect density anomalies
        if density_analysis['is_crowded']:
            anomalies.add(BehavioralAnomalyType.CROWD_FORMATION)
        
        # Detect flow anomalies
        if flow_analysis['is_congested']:
            # Check if this is unusual congestion
            if not historical_baseline or flow_analysis['flow_speed'] < historical_baseline.get('avg_flow_speed', 1.0) * 0.5:
                anomalies.add(BehavioralAnomalyType.UNUSUAL_SPEED)
        
        # Detect unusual group formations
        large_groups = [g for g in group_formations if g['size'] > 10]
        if large_groups:
            anomalies.add(BehavioralAnomalyType.UNUSUAL_GROUP_SIZE)
        
        return anomalies
    
    def _classify_density_level(self, density: float) -> str:
        """Classify density level."""
        if density < 1.0:
            return "sparse"
        elif density < 3.0:
            return "moderate"
        elif density < 5.0:
            return "dense"
        else:
            return "crowded"
    
    def _calculate_distance_matrix(self, coordinates: List[WorldCoordinates]) -> np.ndarray:
        """Calculate distance matrix between all coordinate pairs."""
        n = len(coordinates)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = coordinates[i].distance_to(coordinates[j])
                matrix[i, j] = distance
                matrix[j, i] = distance
        
        return matrix
    
    def _find_proximity_groups(
        self,
        tracks: List[Track],
        distance_matrix: np.ndarray
    ) -> List[List[int]]:
        """Find groups based on proximity using simple clustering."""
        n = len(tracks)
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start new group
            group = [i]
            visited[i] = True
            queue = [i]
            
            # Find all connected nodes within threshold
            while queue:
                current = queue.pop(0)
                
                for j in range(n):
                    if not visited[j] and distance_matrix[current, j] <= self.group_proximity_threshold:
                        visited[j] = True
                        group.append(j)
                        queue.append(j)
            
            if len(group) >= 2:  # Only keep groups with 2+ people
                groups.append(group)
        
        return groups
    
    def _analyze_group(
        self,
        group_tracks: List[Track],
        group_coordinates: List[WorldCoordinates]
    ) -> Dict[str, any]:
        """Analyze a detected group."""
        group_size = len(group_tracks)
        
        # Calculate group center
        center_x = np.mean([coord.x for coord in group_coordinates])
        center_y = np.mean([coord.y for coord in group_coordinates])
        
        # Calculate group spread
        distances_from_center = [
            np.sqrt((coord.x - center_x)**2 + (coord.y - center_y)**2)
            for coord in group_coordinates
        ]
        group_spread = np.mean(distances_from_center)
        
        # Analyze group movement
        group_velocities = [
            track.velocity for track in group_tracks 
            if track.velocity and track.velocity.is_moving()
        ]
        
        if group_velocities:
            avg_speed = np.mean([v.speed for v in group_velocities])
            movement_alignment = self._calculate_group_movement_alignment(group_velocities)
        else:
            avg_speed = 0.0
            movement_alignment = 0.0
        
        # Classify group type
        group_type = self._classify_group_type(group_size, group_spread, movement_alignment)
        
        return {
            'size': group_size,
            'center': {'x': center_x, 'y': center_y},
            'spread': group_spread,
            'average_speed': avg_speed,
            'movement_alignment': movement_alignment,
            'group_type': group_type,
            'track_ids': [str(track.track_id) for track in group_tracks]
        }
    
    def _calculate_flow_consistency(self, movement_vectors: List[Tuple[float, float]]) -> float:
        """Calculate how consistent movement directions are."""
        if len(movement_vectors) < 2:
            return 1.0
        
        # Calculate angles for each movement vector
        angles = [np.arctan2(vy, vx) for vx, vy in movement_vectors]
        
        # Calculate circular variance (measure of angular dispersion)
        cos_sum = sum(np.cos(angle) for angle in angles)
        sin_sum = sum(np.sin(angle) for angle in angles)
        
        resultant_length = np.sqrt(cos_sum**2 + sin_sum**2) / len(angles)
        
        # Convert to consistency score (0 = random, 1 = all aligned)
        return resultant_length
    
    def _calculate_group_movement_alignment(self, velocities) -> float:
        """Calculate how aligned group members' movements are."""
        if len(velocities) < 2:
            return 1.0
        
        # Extract velocity vectors
        vectors = [(v.vx, v.vy) for v in velocities]
        return self._calculate_flow_consistency(vectors)
    
    def _classify_group_type(
        self,
        size: int,
        spread: float,
        alignment: float
    ) -> SocialInteractionType:
        """Classify the type of social group."""
        if size == 2:
            return SocialInteractionType.PAIR
        elif size <= 5:
            return SocialInteractionType.SMALL_GROUP
        elif size <= 10:
            return SocialInteractionType.LARGE_GROUP
        elif spread < 3.0 and alignment > 0.7:
            return SocialInteractionType.QUEUE
        else:
            return SocialInteractionType.CROWD
    
    def get_dynamics_statistics(self) -> Dict[str, int]:
        """Get crowd dynamics statistics."""
        return self._dynamics_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset dynamics statistics."""
        self._dynamics_stats = {
            'crowds_analyzed': 0,
            'groups_detected': 0,
            'congestion_events': 0
        }
        logger.debug("Crowd dynamics statistics reset")