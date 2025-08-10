"""
Behavioral Analysis Service for Movement Pattern Analysis

Advanced behavioral analysis system providing:
- Person movement pattern analysis and classification
- Dwell time calculation and zone interaction metrics
- Crowd flow analysis and congestion detection
- Behavioral anomaly identification and alerting
- Social interaction analysis and group detection
- Predictive behavior modeling
- Long-term behavior trend analysis
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
from scipy.spatial.distance import cdist
from scipy.stats import zscore, pearsonr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from app.services.historical_data_service import (
    HistoricalDataService,
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)
from app.services.movement_path_visualization_service import (
    MovementPathVisualizationService,
    PersonTrajectory,
    TrajectoryPoint
)
from app.domains.mapping.entities.coordinate import Coordinate

logger = logging.getLogger(__name__)


class MovementPattern(Enum):
    """Types of movement patterns."""
    DIRECT_TRANSIT = "direct_transit"
    EXPLORATORY = "exploratory"
    LOITERING = "loitering"
    CIRCULAR = "circular"
    BACK_AND_FORTH = "back_and_forth"
    STATIONARY = "stationary"
    RUSHING = "rushing"
    WANDERING = "wandering"


class BehavioralAnomalyType(Enum):
    """Types of behavioral anomalies."""
    UNUSUAL_SPEED = "unusual_speed"
    UNUSUAL_PATH = "unusual_path"
    UNUSUAL_DWELL_TIME = "unusual_dwell_time"
    RESTRICTED_AREA = "restricted_area"
    CROWD_FORMATION = "crowd_formation"
    COUNTER_FLOW = "counter_flow"
    ABANDONED_OBJECT = "abandoned_object"
    UNUSUAL_GROUP_SIZE = "unusual_group_size"


class SocialInteractionType(Enum):
    """Types of social interactions."""
    INDIVIDUAL = "individual"
    PAIR = "pair"
    SMALL_GROUP = "small_group"  # 3-5 people
    LARGE_GROUP = "large_group"  # 6+ people
    CROWD = "crowd"  # Dense aggregation
    QUEUE = "queue"  # Linear formation


@dataclass
class BehavioralFeatures:
    """Comprehensive behavioral features for a person."""
    global_person_id: str
    
    # Movement characteristics
    average_speed: float
    max_speed: float
    speed_variance: float
    total_distance: float
    path_efficiency: float  # Ratio of direct distance to actual path distance
    
    # Temporal characteristics
    total_time: float  # Total time in environment
    active_time: float  # Time spent moving
    stationary_time: float  # Time spent stationary
    
    # Spatial characteristics
    area_coverage: float  # Area of bounding box of movement
    visited_zones: Set[str]
    zone_transitions: int
    
    # Pattern characteristics
    movement_pattern: MovementPattern
    path_complexity: float  # Measure of path tortuosity
    direction_changes: int
    
    # Interaction characteristics
    proximity_events: int  # Number of times close to others
    group_membership_time: float  # Time spent in groups
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'global_person_id': self.global_person_id,
            'movement': {
                'average_speed': self.average_speed,
                'max_speed': self.max_speed,
                'speed_variance': self.speed_variance,
                'total_distance': self.total_distance,
                'path_efficiency': self.path_efficiency
            },
            'temporal': {
                'total_time': self.total_time,
                'active_time': self.active_time,
                'stationary_time': self.stationary_time
            },
            'spatial': {
                'area_coverage': self.area_coverage,
                'visited_zones': list(self.visited_zones),
                'zone_transitions': self.zone_transitions
            },
            'pattern': {
                'movement_pattern': self.movement_pattern.value,
                'path_complexity': self.path_complexity,
                'direction_changes': self.direction_changes
            },
            'interaction': {
                'proximity_events': self.proximity_events,
                'group_membership_time': self.group_membership_time
            }
        }


@dataclass
class BehavioralAnomaly:
    """Behavioral anomaly detection result."""
    anomaly_id: str
    global_person_id: str
    anomaly_type: BehavioralAnomalyType
    severity: float  # 0.0 - 1.0
    description: str
    timestamp: datetime
    location: Optional[Coordinate] = None
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anomaly_id': self.anomaly_id,
            'global_person_id': self.global_person_id,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'location': {'x': self.location.x, 'y': self.location.y} if self.location else None,
            'supporting_evidence': self.supporting_evidence
        }


@dataclass
class SocialGroup:
    """Social group detection result."""
    group_id: str
    person_ids: Set[str]
    interaction_type: SocialInteractionType
    center_location: Coordinate
    formation_time: datetime
    duration: float  # seconds
    stability_score: float  # How stable the group formation is
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'group_id': self.group_id,
            'person_ids': list(self.person_ids),
            'interaction_type': self.interaction_type.value,
            'center_location': {'x': self.center_location.x, 'y': self.center_location.y},
            'formation_time': self.formation_time.isoformat(),
            'duration': self.duration,
            'stability_score': self.stability_score
        }


@dataclass
class CrowdFlowMetrics:
    """Crowd flow analysis metrics."""
    flow_rate: float  # people per minute
    flow_direction: float  # average direction in radians
    flow_variance: float  # directional variance
    congestion_level: float  # 0.0 - 1.0
    bottleneck_locations: List[Coordinate]
    counter_flow_percentage: float  # Percentage moving against main flow
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'flow_rate': self.flow_rate,
            'flow_direction': self.flow_direction,
            'flow_variance': self.flow_variance,
            'congestion_level': self.congestion_level,
            'bottleneck_locations': [
                {'x': loc.x, 'y': loc.y} for loc in self.bottleneck_locations
            ],
            'counter_flow_percentage': self.counter_flow_percentage
        }


class BehavioralAnalysisService:
    """Comprehensive service for behavioral analysis and pattern recognition."""
    
    def __init__(
        self,
        historical_data_service: HistoricalDataService,
        movement_visualization_service: MovementPathVisualizationService
    ):
        self.historical_service = historical_data_service
        self.visualization_service = movement_visualization_service
        
        # Analysis parameters
        self.proximity_threshold = 2.0  # meters for social interaction detection
        self.group_stability_threshold = 0.7
        self.anomaly_threshold = 2.0  # Z-score threshold
        self.min_dwell_time = 10.0  # seconds
        self.speed_thresholds = {
            'stationary': 0.1,  # m/s
            'walking': 1.5,     # m/s
            'rushing': 3.0      # m/s
        }
        
        # Feature normalization
        self.feature_scaler = StandardScaler()
        self.features_normalized = False
        
        # Behavioral baselines for anomaly detection
        self.behavioral_baselines = {
            'speed': {'mean': 1.2, 'std': 0.5},
            'dwell_time': {'mean': 120.0, 'std': 60.0},
            'path_efficiency': {'mean': 0.7, 'std': 0.2},
            'direction_changes': {'mean': 5.0, 'std': 3.0}
        }
        
        # Analysis cache
        self.behavioral_features_cache: Dict[str, BehavioralFeatures] = {}
        self.group_analysis_cache: Dict[str, List[SocialGroup]] = {}
        
        # Performance metrics
        self.analysis_stats = {
            'behaviors_analyzed': 0,
            'anomalies_detected': 0,
            'groups_identified': 0,
            'avg_processing_time_ms': 0.0
        }
        
        logger.info("BehavioralAnalysisService initialized")
    
    # --- Behavioral Feature Extraction ---
    
    async def extract_behavioral_features(
        self,
        global_person_id: str,
        time_range: TimeRange,
        environment_id: Optional[str] = None
    ) -> BehavioralFeatures:
        """Extract comprehensive behavioral features for a person."""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{global_person_id}_{time_range.start_time.isoformat()}"
            if cache_key in self.behavioral_features_cache:
                return self.behavioral_features_cache[cache_key]
            
            # Get trajectory data
            trajectory = await self.visualization_service.reconstruct_person_trajectory(
                global_person_id, time_range, environment_id
            )
            
            if not trajectory.points or len(trajectory.points) < 2:
                logger.warning(f"Insufficient trajectory data for behavioral analysis: {global_person_id}")
                return self._create_empty_features(global_person_id)
            
            # Extract movement characteristics
            movement_features = self._extract_movement_features(trajectory)
            
            # Extract temporal characteristics
            temporal_features = self._extract_temporal_features(trajectory)
            
            # Extract spatial characteristics
            spatial_features = self._extract_spatial_features(trajectory)
            
            # Extract pattern characteristics
            pattern_features = self._extract_pattern_features(trajectory)
            
            # Extract interaction characteristics (placeholder for now)
            interaction_features = self._extract_interaction_features(trajectory)
            
            # Create comprehensive features object
            features = BehavioralFeatures(
                global_person_id=global_person_id,
                **movement_features,
                **temporal_features,
                **spatial_features,
                **pattern_features,
                **interaction_features
            )
            
            # Cache result
            self.behavioral_features_cache[cache_key] = features
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_analysis_stats('behavior_extraction', processing_time)
            
            logger.debug(f"Extracted behavioral features for {global_person_id}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")
            return self._create_empty_features(global_person_id)
    
    def _extract_movement_features(self, trajectory: PersonTrajectory) -> Dict[str, Any]:
        """Extract movement-related features."""
        try:
            points = trajectory.points
            
            # Calculate speeds between consecutive points
            speeds = []
            distances = []
            
            for i in range(1, len(points)):
                prev_point = points[i-1]
                curr_point = points[i]
                
                # Distance calculation
                dx = curr_point.position.x - prev_point.position.x
                dy = curr_point.position.y - prev_point.position.y
                distance = np.sqrt(dx**2 + dy**2)
                distances.append(distance)
                
                # Speed calculation
                time_diff = (curr_point.timestamp - prev_point.timestamp).total_seconds()
                if time_diff > 0:
                    speed = distance / time_diff
                    speeds.append(speed)
            
            # Direct distance vs actual path distance
            if len(points) >= 2:
                direct_distance = np.sqrt(
                    (points[-1].position.x - points[0].position.x)**2 + 
                    (points[-1].position.y - points[0].position.y)**2
                )
                actual_distance = sum(distances)
                path_efficiency = direct_distance / (actual_distance + 1e-6)
            else:
                path_efficiency = 1.0
            
            return {
                'average_speed': np.mean(speeds) if speeds else 0.0,
                'max_speed': max(speeds) if speeds else 0.0,
                'speed_variance': np.var(speeds) if speeds else 0.0,
                'total_distance': sum(distances),
                'path_efficiency': min(path_efficiency, 1.0)  # Cap at 1.0
            }
            
        except Exception as e:
            logger.error(f"Error extracting movement features: {e}")
            return {
                'average_speed': 0.0,
                'max_speed': 0.0,
                'speed_variance': 0.0,
                'total_distance': 0.0,
                'path_efficiency': 0.0
            }
    
    def _extract_temporal_features(self, trajectory: PersonTrajectory) -> Dict[str, Any]:
        """Extract temporal characteristics."""
        try:
            if not trajectory.start_time or not trajectory.end_time:
                return {
                    'total_time': 0.0,
                    'active_time': 0.0,
                    'stationary_time': 0.0
                }
            
            total_time = (trajectory.end_time - trajectory.start_time).total_seconds()
            
            # Calculate active vs stationary time
            active_time = 0.0
            stationary_time = 0.0
            
            for point in trajectory.points:
                if point.speed is not None:
                    if point.speed > self.speed_thresholds['stationary']:
                        # Estimate time for this segment (simplified)
                        active_time += 1.0  # Approximate 1 second per point
                    else:
                        stationary_time += 1.0
            
            return {
                'total_time': total_time,
                'active_time': active_time,
                'stationary_time': stationary_time
            }
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return {
                'total_time': 0.0,
                'active_time': 0.0,
                'stationary_time': 0.0
            }
    
    def _extract_spatial_features(self, trajectory: PersonTrajectory) -> Dict[str, Any]:
        """Extract spatial characteristics."""
        try:
            points = trajectory.points
            
            if len(points) < 2:
                return {
                    'area_coverage': 0.0,
                    'visited_zones': set(),
                    'zone_transitions': 0
                }
            
            # Calculate bounding box area
            x_coords = [p.position.x for p in points]
            y_coords = [p.position.y for p in points]
            
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            area_coverage = x_range * y_range
            
            # Track visited cameras as proxy for zones
            visited_zones = set(p.camera_id for p in points)
            
            # Count zone transitions (camera changes)
            zone_transitions = 0
            prev_camera = None
            for point in points:
                if prev_camera and prev_camera != point.camera_id:
                    zone_transitions += 1
                prev_camera = point.camera_id
            
            return {
                'area_coverage': area_coverage,
                'visited_zones': visited_zones,
                'zone_transitions': zone_transitions
            }
            
        except Exception as e:
            logger.error(f"Error extracting spatial features: {e}")
            return {
                'area_coverage': 0.0,
                'visited_zones': set(),
                'zone_transitions': 0
            }
    
    def _extract_pattern_features(self, trajectory: PersonTrajectory) -> Dict[str, Any]:
        """Extract movement pattern characteristics."""
        try:
            points = trajectory.points
            
            if len(points) < 3:
                return {
                    'movement_pattern': MovementPattern.STATIONARY,
                    'path_complexity': 0.0,
                    'direction_changes': 0
                }
            
            # Calculate direction changes
            directions = []
            direction_changes = 0
            
            for i in range(1, len(points) - 1):
                prev_point = points[i-1]
                curr_point = points[i]
                next_point = points[i+1]
                
                # Vector from prev to curr
                v1 = (curr_point.position.x - prev_point.position.x,
                      curr_point.position.y - prev_point.position.y)
                
                # Vector from curr to next
                v2 = (next_point.position.x - curr_point.position.x,
                      next_point.position.y - curr_point.position.y)
                
                # Calculate angle between vectors
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    direction1 = np.arctan2(v1[1], v1[0])
                    direction2 = np.arctan2(v2[1], v2[0])
                    
                    directions.append(direction2)
                    
                    # Check for significant direction change (>30 degrees)
                    angle_diff = abs(direction2 - direction1)
                    angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                    
                    if angle_diff > np.pi/6:  # 30 degrees
                        direction_changes += 1
            
            # Path complexity (tortuosity)
            if trajectory.total_distance > 0 and len(points) >= 2:
                direct_distance = np.sqrt(
                    (points[-1].position.x - points[0].position.x)**2 + 
                    (points[-1].position.y - points[0].position.y)**2
                )
                path_complexity = trajectory.total_distance / (direct_distance + 1e-6)
            else:
                path_complexity = 1.0
            
            # Classify movement pattern
            movement_pattern = self._classify_movement_pattern(
                trajectory, direction_changes, path_complexity
            )
            
            return {
                'movement_pattern': movement_pattern,
                'path_complexity': path_complexity,
                'direction_changes': direction_changes
            }
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return {
                'movement_pattern': MovementPattern.STATIONARY,
                'path_complexity': 0.0,
                'direction_changes': 0
            }
    
    def _classify_movement_pattern(
        self,
        trajectory: PersonTrajectory,
        direction_changes: int,
        path_complexity: float
    ) -> MovementPattern:
        """Classify the movement pattern based on features."""
        try:
            avg_speed = trajectory.average_speed
            total_distance = trajectory.total_distance
            
            # Stationary: very low movement
            if avg_speed < self.speed_thresholds['stationary'] or total_distance < 1.0:
                return MovementPattern.STATIONARY
            
            # Rushing: high speed with low complexity
            if avg_speed > self.speed_thresholds['rushing'] and path_complexity < 1.2:
                return MovementPattern.RUSHING
            
            # Direct transit: efficient path with few direction changes
            if trajectory.path_smoothness > 0.8 and direction_changes < 3:
                return MovementPattern.DIRECT_TRANSIT
            
            # Circular: high direction changes with moderate complexity
            if direction_changes > 8 and 1.5 < path_complexity < 3.0:
                return MovementPattern.CIRCULAR
            
            # Back and forth: high complexity with many direction changes
            if path_complexity > 2.0 and direction_changes > 6:
                return MovementPattern.BACK_AND_FORTH
            
            # Loitering: low speed with high complexity
            if avg_speed < self.speed_thresholds['walking'] and path_complexity > 2.0:
                return MovementPattern.LOITERING
            
            # Exploratory: moderate speed with moderate complexity
            if 1.2 < path_complexity < 2.0 and 3 < direction_changes < 8:
                return MovementPattern.EXPLORATORY
            
            # Default to wandering
            return MovementPattern.WANDERING
            
        except Exception as e:
            logger.error(f"Error classifying movement pattern: {e}")
            return MovementPattern.WANDERING
    
    def _extract_interaction_features(self, trajectory: PersonTrajectory) -> Dict[str, Any]:
        """Extract social interaction features."""
        # Placeholder implementation - would require multi-person analysis
        return {
            'proximity_events': 0,
            'group_membership_time': 0.0
        }
    
    def _create_empty_features(self, global_person_id: str) -> BehavioralFeatures:
        """Create empty behavioral features for edge cases."""
        return BehavioralFeatures(
            global_person_id=global_person_id,
            average_speed=0.0,
            max_speed=0.0,
            speed_variance=0.0,
            total_distance=0.0,
            path_efficiency=0.0,
            total_time=0.0,
            active_time=0.0,
            stationary_time=0.0,
            area_coverage=0.0,
            visited_zones=set(),
            zone_transitions=0,
            movement_pattern=MovementPattern.STATIONARY,
            path_complexity=0.0,
            direction_changes=0,
            proximity_events=0,
            group_membership_time=0.0
        )
    
    # --- Anomaly Detection ---
    
    async def detect_behavioral_anomalies(
        self,
        features: BehavioralFeatures,
        context_features: Optional[List[BehavioralFeatures]] = None
    ) -> List[BehavioralAnomaly]:
        """Detect behavioral anomalies for a person."""
        try:
            anomalies = []
            
            # Speed anomalies
            speed_anomalies = self._detect_speed_anomalies(features)
            anomalies.extend(speed_anomalies)
            
            # Path anomalies
            path_anomalies = self._detect_path_anomalies(features)
            anomalies.extend(path_anomalies)
            
            # Temporal anomalies
            temporal_anomalies = self._detect_temporal_anomalies(features)
            anomalies.extend(temporal_anomalies)
            
            # Spatial anomalies
            spatial_anomalies = self._detect_spatial_anomalies(features)
            anomalies.extend(spatial_anomalies)
            
            # Update statistics
            self.analysis_stats['anomalies_detected'] += len(anomalies)
            
            logger.debug(f"Detected {len(anomalies)} behavioral anomalies for {features.global_person_id}")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting behavioral anomalies: {e}")
            return []
    
    def _detect_speed_anomalies(self, features: BehavioralFeatures) -> List[BehavioralAnomaly]:
        """Detect speed-related anomalies."""
        anomalies = []
        
        try:
            # Check for unusual speed
            speed_baseline = self.behavioral_baselines['speed']
            speed_zscore = (features.average_speed - speed_baseline['mean']) / speed_baseline['std']
            
            if abs(speed_zscore) > self.anomaly_threshold:
                severity = min(abs(speed_zscore) / 3.0, 1.0)  # Normalize to 0-1
                
                if speed_zscore > 0:
                    description = f"Unusually high speed: {features.average_speed:.2f} m/s (z-score: {speed_zscore:.2f})"
                else:
                    description = f"Unusually low speed: {features.average_speed:.2f} m/s (z-score: {speed_zscore:.2f})"
                
                anomaly = BehavioralAnomaly(
                    anomaly_id=f"speed_{features.global_person_id}_{int(time.time())}",
                    global_person_id=features.global_person_id,
                    anomaly_type=BehavioralAnomalyType.UNUSUAL_SPEED,
                    severity=severity,
                    description=description,
                    timestamp=datetime.utcnow(),
                    supporting_evidence={
                        'average_speed': features.average_speed,
                        'max_speed': features.max_speed,
                        'z_score': speed_zscore
                    }
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error detecting speed anomalies: {e}")
        
        return anomalies
    
    def _detect_path_anomalies(self, features: BehavioralFeatures) -> List[BehavioralAnomaly]:
        """Detect path-related anomalies."""
        anomalies = []
        
        try:
            # Check for unusual path efficiency
            efficiency_baseline = self.behavioral_baselines['path_efficiency']
            efficiency_zscore = (features.path_efficiency - efficiency_baseline['mean']) / efficiency_baseline['std']
            
            if abs(efficiency_zscore) > self.anomaly_threshold:
                severity = min(abs(efficiency_zscore) / 3.0, 1.0)
                
                description = f"Unusual path efficiency: {features.path_efficiency:.2f} (z-score: {efficiency_zscore:.2f})"
                
                anomaly = BehavioralAnomaly(
                    anomaly_id=f"path_{features.global_person_id}_{int(time.time())}",
                    global_person_id=features.global_person_id,
                    anomaly_type=BehavioralAnomalyType.UNUSUAL_PATH,
                    severity=severity,
                    description=description,
                    timestamp=datetime.utcnow(),
                    supporting_evidence={
                        'path_efficiency': features.path_efficiency,
                        'path_complexity': features.path_complexity,
                        'direction_changes': features.direction_changes,
                        'z_score': efficiency_zscore
                    }
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error detecting path anomalies: {e}")
        
        return anomalies
    
    def _detect_temporal_anomalies(self, features: BehavioralFeatures) -> List[BehavioralAnomaly]:
        """Detect temporal anomalies."""
        anomalies = []
        
        try:
            # Check for unusual dwell time
            if features.total_time > 0:
                dwell_baseline = self.behavioral_baselines['dwell_time']
                dwell_zscore = (features.total_time - dwell_baseline['mean']) / dwell_baseline['std']
                
                if abs(dwell_zscore) > self.anomaly_threshold:
                    severity = min(abs(dwell_zscore) / 3.0, 1.0)
                    
                    if dwell_zscore > 0:
                        description = f"Unusually long dwell time: {features.total_time:.1f} seconds (z-score: {dwell_zscore:.2f})"
                    else:
                        description = f"Unusually short dwell time: {features.total_time:.1f} seconds (z-score: {dwell_zscore:.2f})"
                    
                    anomaly = BehavioralAnomaly(
                        anomaly_id=f"dwell_{features.global_person_id}_{int(time.time())}",
                        global_person_id=features.global_person_id,
                        anomaly_type=BehavioralAnomalyType.UNUSUAL_DWELL_TIME,
                        severity=severity,
                        description=description,
                        timestamp=datetime.utcnow(),
                        supporting_evidence={
                            'total_time': features.total_time,
                            'active_time': features.active_time,
                            'stationary_time': features.stationary_time,
                            'z_score': dwell_zscore
                        }
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error detecting temporal anomalies: {e}")
        
        return anomalies
    
    def _detect_spatial_anomalies(self, features: BehavioralFeatures) -> List[BehavioralAnomaly]:
        """Detect spatial anomalies."""
        anomalies = []
        
        try:
            # Check for excessive zone transitions
            if features.zone_transitions > 10:  # Arbitrary threshold
                severity = min(features.zone_transitions / 20.0, 1.0)
                
                description = f"Excessive zone transitions: {features.zone_transitions}"
                
                anomaly = BehavioralAnomaly(
                    anomaly_id=f"spatial_{features.global_person_id}_{int(time.time())}",
                    global_person_id=features.global_person_id,
                    anomaly_type=BehavioralAnomalyType.UNUSUAL_PATH,
                    severity=severity,
                    description=description,
                    timestamp=datetime.utcnow(),
                    supporting_evidence={
                        'zone_transitions': features.zone_transitions,
                        'visited_zones': list(features.visited_zones),
                        'area_coverage': features.area_coverage
                    }
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error detecting spatial anomalies: {e}")
        
        return anomalies
    
    # --- Social Interaction Analysis ---
    
    async def analyze_social_interactions(
        self,
        time_range: TimeRange,
        environment_id: Optional[str] = None
    ) -> List[SocialGroup]:
        """Analyze social interactions and group formations."""
        try:
            start_time = time.time()
            
            # Check cache
            cache_key = f"{time_range.start_time.isoformat()}_{time_range.end_time.isoformat()}"
            if cache_key in self.group_analysis_cache:
                return self.group_analysis_cache[cache_key]
            
            # Query historical data for the time range
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id
            )
            
            data_points = await self.historical_service.query_historical_data(query_filter)
            
            if len(data_points) < 2:
                return []
            
            # Group data points by time windows
            time_windows = self._create_time_windows(data_points, window_size=30)  # 30-second windows
            
            social_groups = []
            
            # Analyze each time window for group formations
            for window_time, window_points in time_windows.items():
                window_groups = await self._detect_groups_in_window(window_time, window_points)
                social_groups.extend(window_groups)
            
            # Merge and filter groups
            merged_groups = self._merge_temporal_groups(social_groups)
            
            # Cache result
            self.group_analysis_cache[cache_key] = merged_groups
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.analysis_stats['groups_identified'] += len(merged_groups)
            self._update_analysis_stats('social_interaction', processing_time)
            
            logger.info(f"Identified {len(merged_groups)} social groups in {processing_time:.1f}ms")
            return merged_groups
            
        except Exception as e:
            logger.error(f"Error analyzing social interactions: {e}")
            return []
    
    def _create_time_windows(
        self,
        data_points: List[HistoricalDataPoint],
        window_size: int = 30
    ) -> Dict[datetime, List[HistoricalDataPoint]]:
        """Create time windows for group analysis."""
        try:
            time_windows = defaultdict(list)
            
            # Sort data points by timestamp
            sorted_points = sorted(data_points, key=lambda x: x.timestamp)
            
            if not sorted_points:
                return {}
            
            # Create windows
            start_time = sorted_points[0].timestamp
            window_start = start_time
            
            for point in sorted_points:
                # Check if point belongs to current window
                if (point.timestamp - window_start).total_seconds() <= window_size:
                    time_windows[window_start].append(point)
                else:
                    # Start new window
                    window_start = point.timestamp
                    time_windows[window_start].append(point)
            
            return dict(time_windows)
            
        except Exception as e:
            logger.error(f"Error creating time windows: {e}")
            return {}
    
    async def _detect_groups_in_window(
        self,
        window_time: datetime,
        data_points: List[HistoricalDataPoint]
    ) -> List[SocialGroup]:
        """Detect social groups within a time window."""
        try:
            groups = []
            
            # Extract positions for clustering
            positions = []
            person_map = {}
            
            for i, point in enumerate(data_points):
                if point.coordinates:
                    positions.append([point.coordinates.x, point.coordinates.y])
                    person_map[i] = point.global_person_id
            
            if len(positions) < 2:
                return []
            
            positions_array = np.array(positions)
            
            # Apply DBSCAN clustering for group detection
            clustering = DBSCAN(
                eps=self.proximity_threshold,  # Maximum distance for group membership
                min_samples=2  # Minimum group size
            ).fit(positions_array)
            
            # Process clusters
            clusters = defaultdict(list)
            for i, cluster_id in enumerate(clustering.labels_):
                if cluster_id != -1:  # Ignore noise points
                    clusters[cluster_id].append(i)
            
            # Create SocialGroup objects
            for cluster_id, point_indices in clusters.items():
                if len(point_indices) >= 2:
                    person_ids = {person_map[i] for i in point_indices}
                    
                    # Calculate center location
                    cluster_positions = positions_array[point_indices]
                    center_x = np.mean(cluster_positions[:, 0])
                    center_y = np.mean(cluster_positions[:, 1])
                    center_location = Coordinate(x=center_x, y=center_y)
                    
                    # Determine interaction type
                    group_size = len(person_ids)
                    if group_size == 2:
                        interaction_type = SocialInteractionType.PAIR
                    elif group_size <= 5:
                        interaction_type = SocialInteractionType.SMALL_GROUP
                    else:
                        interaction_type = SocialInteractionType.LARGE_GROUP
                    
                    # Calculate stability score (simplified)
                    position_variance = np.mean(np.var(cluster_positions, axis=0))
                    stability_score = 1.0 / (1.0 + position_variance)
                    
                    group = SocialGroup(
                        group_id=f"group_{cluster_id}_{int(window_time.timestamp())}",
                        person_ids=person_ids,
                        interaction_type=interaction_type,
                        center_location=center_location,
                        formation_time=window_time,
                        duration=30.0,  # Window duration
                        stability_score=stability_score
                    )
                    
                    groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error detecting groups in window: {e}")
            return []
    
    def _merge_temporal_groups(self, groups: List[SocialGroup]) -> List[SocialGroup]:
        """Merge groups that persist across multiple time windows."""
        try:
            if not groups:
                return []
            
            # Sort groups by formation time
            sorted_groups = sorted(groups, key=lambda g: g.formation_time)
            
            merged_groups = []
            current_group = sorted_groups[0]
            
            for next_group in sorted_groups[1:]:
                # Check if groups have significant overlap in membership
                overlap = len(current_group.person_ids & next_group.person_ids)
                overlap_ratio = overlap / len(current_group.person_ids | next_group.person_ids)
                
                if overlap_ratio >= 0.5:  # 50% overlap threshold
                    # Merge groups
                    current_group.person_ids.update(next_group.person_ids)
                    current_group.duration += next_group.duration
                    
                    # Update stability score (average)
                    current_group.stability_score = (
                        current_group.stability_score + next_group.stability_score
                    ) / 2.0
                else:
                    # Add current group to results and start new one
                    if current_group.stability_score >= self.group_stability_threshold:
                        merged_groups.append(current_group)
                    current_group = next_group
            
            # Add the last group
            if current_group.stability_score >= self.group_stability_threshold:
                merged_groups.append(current_group)
            
            return merged_groups
            
        except Exception as e:
            logger.error(f"Error merging temporal groups: {e}")
            return groups
    
    # --- Crowd Flow Analysis ---
    
    async def analyze_crowd_flow(
        self,
        time_range: TimeRange,
        environment_id: Optional[str] = None,
        spatial_grid_size: float = 5.0
    ) -> CrowdFlowMetrics:
        """Analyze crowd flow patterns and congestion."""
        try:
            # Query movement data
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id
            )
            
            data_points = await self.historical_service.query_historical_data(query_filter)
            
            if len(data_points) < 10:
                return self._create_empty_flow_metrics()
            
            # Calculate flow vectors for each person
            flow_vectors = self._calculate_flow_vectors(data_points)
            
            if not flow_vectors:
                return self._create_empty_flow_metrics()
            
            # Calculate flow metrics
            flow_rate = self._calculate_flow_rate(data_points, time_range)
            flow_direction = self._calculate_average_flow_direction(flow_vectors)
            flow_variance = self._calculate_flow_variance(flow_vectors, flow_direction)
            congestion_level = self._calculate_congestion_level(data_points, spatial_grid_size)
            bottleneck_locations = self._identify_bottlenecks(data_points, spatial_grid_size)
            counter_flow_percentage = self._calculate_counter_flow(flow_vectors, flow_direction)
            
            return CrowdFlowMetrics(
                flow_rate=flow_rate,
                flow_direction=flow_direction,
                flow_variance=flow_variance,
                congestion_level=congestion_level,
                bottleneck_locations=bottleneck_locations,
                counter_flow_percentage=counter_flow_percentage
            )
            
        except Exception as e:
            logger.error(f"Error analyzing crowd flow: {e}")
            return self._create_empty_flow_metrics()
    
    def _calculate_flow_vectors(self, data_points: List[HistoricalDataPoint]) -> List[Tuple[float, float]]:
        """Calculate flow vectors for movement analysis."""
        try:
            # Group by person
            person_trajectories = defaultdict(list)
            for point in data_points:
                if point.coordinates:
                    person_trajectories[point.global_person_id].append(point)
            
            flow_vectors = []
            
            for person_id, trajectory in person_trajectories.items():
                if len(trajectory) < 2:
                    continue
                
                # Sort by timestamp
                trajectory.sort(key=lambda x: x.timestamp)
                
                # Calculate movement vector for each person
                start_point = trajectory[0]
                end_point = trajectory[-1]
                
                dx = end_point.coordinates.x - start_point.coordinates.x
                dy = end_point.coordinates.y - start_point.coordinates.y
                
                if abs(dx) > 0.1 or abs(dy) > 0.1:  # Filter out minimal movement
                    flow_vectors.append((dx, dy))
            
            return flow_vectors
            
        except Exception as e:
            logger.error(f"Error calculating flow vectors: {e}")
            return []
    
    def _calculate_flow_rate(self, data_points: List[HistoricalDataPoint], time_range: TimeRange) -> float:
        """Calculate flow rate (people per minute)."""
        try:
            unique_persons = len(set(dp.global_person_id for dp in data_points))
            duration_minutes = time_range.duration_hours * 60
            
            return unique_persons / duration_minutes if duration_minutes > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating flow rate: {e}")
            return 0.0
    
    def _calculate_average_flow_direction(self, flow_vectors: List[Tuple[float, float]]) -> float:
        """Calculate average flow direction in radians."""
        try:
            if not flow_vectors:
                return 0.0
            
            # Calculate average direction
            directions = []
            for dx, dy in flow_vectors:
                direction = np.arctan2(dy, dx)
                directions.append(direction)
            
            # Handle circular average
            x_components = [np.cos(d) for d in directions]
            y_components = [np.sin(d) for d in directions]
            
            avg_x = np.mean(x_components)
            avg_y = np.mean(y_components)
            
            return np.arctan2(avg_y, avg_x)
            
        except Exception as e:
            logger.error(f"Error calculating average flow direction: {e}")
            return 0.0
    
    def _calculate_flow_variance(self, flow_vectors: List[Tuple[float, float]], avg_direction: float) -> float:
        """Calculate directional variance in flow."""
        try:
            if not flow_vectors:
                return 0.0
            
            direction_diffs = []
            for dx, dy in flow_vectors:
                direction = np.arctan2(dy, dx)
                diff = abs(direction - avg_direction)
                diff = min(diff, 2*np.pi - diff)  # Circular difference
                direction_diffs.append(diff)
            
            return np.var(direction_diffs)
            
        except Exception as e:
            logger.error(f"Error calculating flow variance: {e}")
            return 0.0
    
    def _calculate_congestion_level(self, data_points: List[HistoricalDataPoint], grid_size: float) -> float:
        """Calculate congestion level based on spatial density."""
        try:
            if not data_points:
                return 0.0
            
            # Create spatial grid
            positions = []
            for point in data_points:
                if point.coordinates:
                    positions.append((point.coordinates.x, point.coordinates.y))
            
            if not positions:
                return 0.0
            
            # Calculate density in each grid cell
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            x_bins = int((x_max - x_min) / grid_size) + 1
            y_bins = int((y_max - y_min) / grid_size) + 1
            
            # Count points in each cell
            grid_counts = np.zeros((y_bins, x_bins))
            
            for x, y in positions:
                x_idx = min(int((x - x_min) / grid_size), x_bins - 1)
                y_idx = min(int((y - y_min) / grid_size), y_bins - 1)
                grid_counts[y_idx, x_idx] += 1
            
            # Calculate congestion as normalized maximum density
            max_density = np.max(grid_counts)
            avg_density = np.mean(grid_counts)
            
            congestion_level = max_density / (avg_density + 1e-6)
            return min(congestion_level / 10.0, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating congestion level: {e}")
            return 0.0
    
    def _identify_bottlenecks(
        self,
        data_points: List[HistoricalDataPoint],
        grid_size: float
    ) -> List[Coordinate]:
        """Identify bottleneck locations."""
        try:
            # Simplified bottleneck detection based on high density areas
            positions = []
            for point in data_points:
                if point.coordinates:
                    positions.append((point.coordinates.x, point.coordinates.y))
            
            if len(positions) < 10:
                return []
            
            # Use KMeans to find density centers
            kmeans = KMeans(n_clusters=min(5, len(positions)//10), random_state=42)
            clusters = kmeans.fit_predict(positions)
            centers = kmeans.cluster_centers_
            
            # Count points in each cluster
            cluster_counts = np.bincount(clusters)
            
            # Identify high-density clusters as bottlenecks
            bottlenecks = []
            threshold = np.mean(cluster_counts) + np.std(cluster_counts)
            
            for i, count in enumerate(cluster_counts):
                if count > threshold:
                    center = centers[i]
                    bottlenecks.append(Coordinate(x=center[0], y=center[1]))
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}")
            return []
    
    def _calculate_counter_flow(self, flow_vectors: List[Tuple[float, float]], avg_direction: float) -> float:
        """Calculate percentage of counter-flow movement."""
        try:
            if not flow_vectors:
                return 0.0
            
            counter_flow_count = 0
            
            for dx, dy in flow_vectors:
                direction = np.arctan2(dy, dx)
                angle_diff = abs(direction - avg_direction)
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                
                # Consider counter-flow if angle difference > 90 degrees
                if angle_diff > np.pi/2:
                    counter_flow_count += 1
            
            return (counter_flow_count / len(flow_vectors)) * 100.0
            
        except Exception as e:
            logger.error(f"Error calculating counter flow: {e}")
            return 0.0
    
    def _create_empty_flow_metrics(self) -> CrowdFlowMetrics:
        """Create empty crowd flow metrics."""
        return CrowdFlowMetrics(
            flow_rate=0.0,
            flow_direction=0.0,
            flow_variance=0.0,
            congestion_level=0.0,
            bottleneck_locations=[],
            counter_flow_percentage=0.0
        )
    
    # --- Batch Analysis ---
    
    async def analyze_behavioral_patterns_batch(
        self,
        time_range: TimeRange,
        environment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive behavioral analysis for multiple persons."""
        try:
            start_time = time.time()
            
            # Get all persons in time range
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id
            )
            
            data_points = await self.historical_service.query_historical_data(query_filter)
            unique_persons = list(set(dp.global_person_id for dp in data_points))
            
            # Extract features for each person
            all_features = []
            all_anomalies = []
            
            for person_id in unique_persons:
                features = await self.extract_behavioral_features(person_id, time_range, environment_id)
                all_features.append(features)
                
                # Detect anomalies
                anomalies = await self.detect_behavioral_anomalies(features)
                all_anomalies.extend(anomalies)
            
            # Analyze social interactions
            social_groups = await self.analyze_social_interactions(time_range, environment_id)
            
            # Analyze crowd flow
            crowd_flow = await self.analyze_crowd_flow(time_range, environment_id)
            
            # Calculate summary statistics
            summary_stats = self._calculate_batch_summary_stats(all_features)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'time_range': {
                    'start': time_range.start_time.isoformat(),
                    'end': time_range.end_time.isoformat(),
                    'duration_hours': time_range.duration_hours
                },
                'persons_analyzed': len(unique_persons),
                'behavioral_features': [f.to_dict() for f in all_features],
                'anomalies': [a.to_dict() for a in all_anomalies],
                'social_groups': [g.to_dict() for g in social_groups],
                'crowd_flow': crowd_flow.to_dict(),
                'summary_statistics': summary_stats,
                'processing_time_ms': processing_time
            }
            
            self.analysis_stats['behaviors_analyzed'] += len(all_features)
            
            logger.info(f"Analyzed {len(unique_persons)} persons in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error in batch behavioral analysis: {e}")
            raise
    
    def _calculate_batch_summary_stats(self, features_list: List[BehavioralFeatures]) -> Dict[str, Any]:
        """Calculate summary statistics for batch analysis."""
        try:
            if not features_list:
                return {}
            
            # Extract values for statistical analysis
            speeds = [f.average_speed for f in features_list]
            distances = [f.total_distance for f in features_list]
            times = [f.total_time for f in features_list]
            complexities = [f.path_complexity for f in features_list]
            
            # Pattern distribution
            pattern_counts = defaultdict(int)
            for features in features_list:
                pattern_counts[features.movement_pattern.value] += 1
            
            return {
                'speed_statistics': {
                    'mean': np.mean(speeds),
                    'std': np.std(speeds),
                    'min': min(speeds),
                    'max': max(speeds),
                    'median': np.median(speeds)
                },
                'distance_statistics': {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': min(distances),
                    'max': max(distances),
                    'median': np.median(distances)
                },
                'time_statistics': {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': min(times),
                    'max': max(times),
                    'median': np.median(times)
                },
                'complexity_statistics': {
                    'mean': np.mean(complexities),
                    'std': np.std(complexities),
                    'min': min(complexities),
                    'max': max(complexities),
                    'median': np.median(complexities)
                },
                'pattern_distribution': dict(pattern_counts),
                'total_analyzed': len(features_list)
            }
            
        except Exception as e:
            logger.error(f"Error calculating batch summary stats: {e}")
            return {}
    
    # --- Utility Methods ---
    
    def _update_analysis_stats(self, operation: str, processing_time_ms: float):
        """Update analysis statistics."""
        # Update average processing time
        current_avg = self.analysis_stats['avg_processing_time_ms']
        total_operations = sum([
            self.analysis_stats['behaviors_analyzed'],
            self.analysis_stats['groups_identified'],
            self.analysis_stats['anomalies_detected']
        ])
        
        if total_operations > 0:
            self.analysis_stats['avg_processing_time_ms'] = (
                (current_avg * (total_operations - 1) + processing_time_ms) / total_operations
            )
    
    def clear_caches(self):
        """Clear all analysis caches."""
        self.behavioral_features_cache.clear()
        self.group_analysis_cache.clear()
        logger.info("Cleared behavioral analysis caches")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'BehavioralAnalysisService',
            'analysis_parameters': {
                'proximity_threshold': self.proximity_threshold,
                'anomaly_threshold': self.anomaly_threshold,
                'min_dwell_time': self.min_dwell_time,
                'speed_thresholds': self.speed_thresholds
            },
            'cache_status': {
                'behavioral_features_cached': len(self.behavioral_features_cache),
                'group_analysis_cached': len(self.group_analysis_cache)
            },
            'behavioral_baselines': self.behavioral_baselines,
            'statistics': self.analysis_stats.copy(),
            'features_normalized': self.features_normalized
        }


# Global service instance
_behavioral_analysis_service: Optional[BehavioralAnalysisService] = None


def get_behavioral_analysis_service() -> Optional[BehavioralAnalysisService]:
    """Get the global behavioral analysis service instance."""
    return _behavioral_analysis_service


def initialize_behavioral_analysis_service(
    historical_data_service: HistoricalDataService,
    movement_visualization_service: MovementPathVisualizationService
) -> BehavioralAnalysisService:
    """Initialize the global behavioral analysis service."""
    global _behavioral_analysis_service
    if _behavioral_analysis_service is None:
        _behavioral_analysis_service = BehavioralAnalysisService(
            historical_data_service, movement_visualization_service
        )
    return _behavioral_analysis_service