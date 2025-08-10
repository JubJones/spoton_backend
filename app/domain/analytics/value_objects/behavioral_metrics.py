"""
Behavioral metrics value objects for person behavior analysis.

Provides type-safe behavioral measurement with validation.
"""
from dataclasses import dataclass
from typing import Set, Dict, List
from enum import Enum

from app.domain.shared.value_objects.base_value_object import BaseValueObject


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


@dataclass(frozen=True)
class MovementMetrics(BaseValueObject):
    """
    Movement metrics value object.
    
    Captures quantitative movement characteristics for behavioral analysis.
    """
    
    average_speed: float  # Average speed in units/second
    max_speed: float      # Maximum observed speed
    speed_variance: float # Variance in speed measurements
    total_distance: float # Total distance traveled
    path_efficiency: float # Ratio of direct to actual path distance
    
    def _validate(self) -> None:
        """Validate movement metrics."""
        if self.average_speed < 0 or self.max_speed < 0:
            raise ValueError("Speeds must be non-negative")
        
        if self.total_distance < 0:
            raise ValueError("Total distance must be non-negative")
        
        if not (0.0 <= self.path_efficiency <= 1.0):
            raise ValueError("Path efficiency must be between 0.0 and 1.0")
        
        if self.max_speed < self.average_speed:
            raise ValueError("Maximum speed cannot be less than average speed")
        
        if self.speed_variance < 0:
            raise ValueError("Speed variance must be non-negative")
    
    @classmethod
    def create(
        cls,
        average_speed: float,
        max_speed: float,
        speed_variance: float,
        total_distance: float,
        path_efficiency: float
    ) -> 'MovementMetrics':
        """Create movement metrics with validation."""
        return cls(
            average_speed=average_speed,
            max_speed=max_speed,
            speed_variance=speed_variance,
            total_distance=total_distance,
            path_efficiency=path_efficiency
        )
    
    @property
    def is_fast_moving(self, threshold: float = 2.0) -> bool:
        """Check if movement is considered fast."""
        return self.average_speed > threshold
    
    @property
    def is_consistent_speed(self, variance_threshold: float = 0.5) -> bool:
        """Check if speed is consistent (low variance)."""
        return self.speed_variance <= variance_threshold
    
    @property
    def is_efficient_path(self, efficiency_threshold: float = 0.8) -> bool:
        """Check if path is efficient (direct)."""
        return self.path_efficiency >= efficiency_threshold
    
    def get_movement_classification(self) -> MovementPattern:
        """Classify movement pattern based on metrics."""
        if self.average_speed < 0.1:
            return MovementPattern.STATIONARY
        elif self.is_efficient_path() and self.is_consistent_speed():
            return MovementPattern.DIRECT_TRANSIT
        elif self.is_fast_moving():
            return MovementPattern.RUSHING
        elif self.path_efficiency < 0.3:
            return MovementPattern.WANDERING
        elif self.path_efficiency < 0.5:
            return MovementPattern.EXPLORATORY
        else:
            return MovementPattern.LOITERING


@dataclass(frozen=True)
class TemporalMetrics(BaseValueObject):
    """
    Temporal behavior metrics value object.
    
    Captures time-based behavioral characteristics.
    """
    
    total_time_seconds: float    # Total time in environment
    active_time_seconds: float   # Time spent moving
    stationary_time_seconds: float # Time spent stationary
    dwell_time_seconds: float    # Time spent in specific zones
    
    def _validate(self) -> None:
        """Validate temporal metrics."""
        if any(t < 0 for t in [self.total_time_seconds, self.active_time_seconds, 
                               self.stationary_time_seconds, self.dwell_time_seconds]):
            raise ValueError("All time values must be non-negative")
        
        if self.active_time_seconds + self.stationary_time_seconds > self.total_time_seconds * 1.01:
            raise ValueError("Active + stationary time cannot exceed total time")
    
    @classmethod
    def create(
        cls,
        total_time_seconds: float,
        active_time_seconds: float,
        stationary_time_seconds: float,
        dwell_time_seconds: float
    ) -> 'TemporalMetrics':
        """Create temporal metrics with validation."""
        return cls(
            total_time_seconds=total_time_seconds,
            active_time_seconds=active_time_seconds,
            stationary_time_seconds=stationary_time_seconds,
            dwell_time_seconds=dwell_time_seconds
        )
    
    @property
    def activity_ratio(self) -> float:
        """Get ratio of active time to total time."""
        return self.active_time_seconds / max(self.total_time_seconds, 0.001)
    
    @property
    def is_highly_active(self, threshold: float = 0.8) -> bool:
        """Check if person is highly active."""
        return self.activity_ratio > threshold
    
    @property
    def is_loitering(self, threshold: float = 0.2) -> bool:
        """Check if person is loitering (low activity)."""
        return self.activity_ratio < threshold
    
    @property
    def average_dwell_time_minutes(self) -> float:
        """Get average dwell time in minutes."""
        return self.dwell_time_seconds / 60.0


@dataclass(frozen=True)
class SpatialMetrics(BaseValueObject):
    """
    Spatial behavior metrics value object.
    
    Captures space-related behavioral characteristics.
    """
    
    area_coverage_sqm: float     # Area covered by movement
    visited_zones: frozenset[str] # Zones visited
    zone_transitions: int        # Number of zone transitions
    boundary_crossings: int      # Number of boundary crossings
    
    def _validate(self) -> None:
        """Validate spatial metrics."""
        if self.area_coverage_sqm < 0:
            raise ValueError("Area coverage must be non-negative")
        
        if self.zone_transitions < 0 or self.boundary_crossings < 0:
            raise ValueError("Counts must be non-negative")
        
        if self.zone_transitions < len(self.visited_zones) - 1:
            # Should have at least (zones-1) transitions to visit all zones
            pass  # This might be valid if person teleported or data is partial
    
    @classmethod
    def create(
        cls,
        area_coverage_sqm: float,
        visited_zones: Set[str],
        zone_transitions: int,
        boundary_crossings: int
    ) -> 'SpatialMetrics':
        """Create spatial metrics with validation."""
        return cls(
            area_coverage_sqm=area_coverage_sqm,
            visited_zones=frozenset(visited_zones),
            zone_transitions=zone_transitions,
            boundary_crossings=boundary_crossings
        )
    
    @property
    def zone_count(self) -> int:
        """Get number of unique zones visited."""
        return len(self.visited_zones)
    
    @property
    def is_wide_ranging(self, area_threshold: float = 100.0) -> bool:
        """Check if person covers a wide area."""
        return self.area_coverage_sqm > area_threshold
    
    @property
    def is_zone_explorer(self, zone_threshold: int = 5) -> bool:
        """Check if person explores many zones."""
        return self.zone_count > zone_threshold
    
    @property
    def transition_rate(self) -> float:
        """Get zone transition rate (transitions per zone)."""
        return self.zone_transitions / max(self.zone_count, 1)


@dataclass(frozen=True)
class BehavioralFeatures(BaseValueObject):
    """
    Comprehensive behavioral features combining all metrics.
    
    Aggregate behavioral analysis for a person across all dimensions.
    """
    
    person_id: str
    movement_metrics: MovementMetrics
    temporal_metrics: TemporalMetrics
    spatial_metrics: SpatialMetrics
    social_interaction_type: SocialInteractionType
    
    # Derived behavioral classifications
    primary_pattern: MovementPattern
    anomaly_indicators: frozenset[BehavioralAnomalyType]
    
    def _validate(self) -> None:
        """Validate behavioral features."""
        if not self.person_id.strip():
            raise ValueError("Person ID cannot be empty")
    
    @classmethod
    def create(
        cls,
        person_id: str,
        movement_metrics: MovementMetrics,
        temporal_metrics: TemporalMetrics,
        spatial_metrics: SpatialMetrics,
        social_interaction_type: SocialInteractionType = SocialInteractionType.INDIVIDUAL,
        anomaly_indicators: Set[BehavioralAnomalyType] = None
    ) -> 'BehavioralFeatures':
        """Create behavioral features with automatic pattern classification."""
        
        # Derive primary movement pattern from metrics
        primary_pattern = movement_metrics.get_movement_classification()
        
        # Detect anomalies based on metrics
        detected_anomalies = set()
        
        # Speed anomalies
        if movement_metrics.average_speed > 5.0:  # Very fast
            detected_anomalies.add(BehavioralAnomalyType.UNUSUAL_SPEED)
        
        # Dwell time anomalies
        if temporal_metrics.dwell_time_seconds > 3600:  # More than 1 hour
            detected_anomalies.add(BehavioralAnomalyType.UNUSUAL_DWELL_TIME)
        
        # Path efficiency anomalies
        if movement_metrics.path_efficiency < 0.1:  # Very inefficient
            detected_anomalies.add(BehavioralAnomalyType.UNUSUAL_PATH)
        
        # Add any externally detected anomalies
        if anomaly_indicators:
            detected_anomalies.update(anomaly_indicators)
        
        return cls(
            person_id=person_id,
            movement_metrics=movement_metrics,
            temporal_metrics=temporal_metrics,
            spatial_metrics=spatial_metrics,
            social_interaction_type=social_interaction_type,
            primary_pattern=primary_pattern,
            anomaly_indicators=frozenset(detected_anomalies)
        )
    
    @property
    def has_anomalies(self) -> bool:
        """Check if any behavioral anomalies detected."""
        return len(self.anomaly_indicators) > 0
    
    @property
    def is_normal_behavior(self) -> bool:
        """Check if behavior appears normal (no anomalies, direct pattern)."""
        return (not self.has_anomalies and 
                self.primary_pattern in {MovementPattern.DIRECT_TRANSIT, MovementPattern.EXPLORATORY})
    
    @property
    def risk_score(self) -> float:
        """
        Calculate behavioral risk score (0-1).
        
        Higher scores indicate more unusual or potentially concerning behavior.
        """
        base_score = 0.0
        
        # Anomaly contribution
        base_score += len(self.anomaly_indicators) * 0.2
        
        # Pattern contribution
        pattern_scores = {
            MovementPattern.DIRECT_TRANSIT: 0.0,
            MovementPattern.EXPLORATORY: 0.1,
            MovementPattern.LOITERING: 0.4,
            MovementPattern.WANDERING: 0.3,
            MovementPattern.STATIONARY: 0.5,
            MovementPattern.RUSHING: 0.2,
            MovementPattern.CIRCULAR: 0.6,
            MovementPattern.BACK_AND_FORTH: 0.7
        }
        base_score += pattern_scores.get(self.primary_pattern, 0.3)
        
        # Social interaction contribution
        if self.social_interaction_type == SocialInteractionType.CROWD:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def get_behavior_summary(self) -> Dict[str, any]:
        """Get comprehensive behavior summary."""
        return {
            'person_id': self.person_id,
            'primary_pattern': self.primary_pattern.value,
            'social_interaction': self.social_interaction_type.value,
            'risk_score': self.risk_score,
            'has_anomalies': self.has_anomalies,
            'anomaly_count': len(self.anomaly_indicators),
            'anomaly_types': [a.value for a in self.anomaly_indicators],
            'movement_summary': {
                'average_speed': self.movement_metrics.average_speed,
                'path_efficiency': self.movement_metrics.path_efficiency,
                'total_distance': self.movement_metrics.total_distance
            },
            'temporal_summary': {
                'total_time_minutes': self.temporal_metrics.total_time_seconds / 60.0,
                'activity_ratio': self.temporal_metrics.activity_ratio,
                'dwell_time_minutes': self.temporal_metrics.average_dwell_time_minutes
            },
            'spatial_summary': {
                'area_coverage': self.spatial_metrics.area_coverage_sqm,
                'zones_visited': self.spatial_metrics.zone_count,
                'zone_transitions': self.spatial_metrics.zone_transitions
            }
        }