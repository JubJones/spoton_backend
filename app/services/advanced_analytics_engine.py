"""
Advanced Analytics Engine for Real-Time and Historical Analytics

Comprehensive analytics system providing:
- Real-time analytics with live person counting and occupancy metrics
- Zone-based analytics with configurable boundaries
- Camera load balancing and performance metrics
- Real-time anomaly detection and alerting
- Statistical analysis and trend identification
- Performance monitoring and optimization
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from app.services.historical_data_service import (
    HistoricalDataService,
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)
from app.services.temporal_query_engine import TemporalQueryEngine, TimeGranularity
from app.infrastructure.cache.tracking_cache import TrackingCache

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of analytics metrics."""
    COUNT = "count"
    RATE = "rate"
    AVERAGE = "average"
    PERCENTAGE = "percentage"
    DISTRIBUTION = "distribution"
    ANOMALY_SCORE = "anomaly_score"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalyticsGranularity(Enum):
    """Analytics time granularity."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


@dataclass
class AnalyticsMetric:
    """Individual analytics metric."""
    name: str
    value: Union[float, int, Dict[str, Any]]
    metric_type: MetricType
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'metric_type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ZoneDefinition:
    """Zone definition for analytics."""
    zone_id: str
    name: str
    boundary_points: List[Tuple[float, float]]  # Polygon vertices
    zone_type: str  # entrance, exit, waiting, restricted
    capacity_limit: Optional[int] = None
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within zone boundary using ray casting algorithm."""
        if len(self.boundary_points) < 3:
            return False
        
        inside = False
        j = len(self.boundary_points) - 1
        
        for i in range(len(self.boundary_points)):
            xi, yi = self.boundary_points[i]
            xj, yj = self.boundary_points[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'zone_id': self.zone_id,
            'name': self.name,
            'boundary_points': self.boundary_points,
            'zone_type': self.zone_type,
            'capacity_limit': self.capacity_limit
        }


@dataclass
class AnalyticsAlert:
    """Analytics alert definition."""
    alert_id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    zone_id: Optional[str] = None
    camera_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'message': self.message,
            'metric_name': self.metric_name,
            'threshold_value': self.threshold_value,
            'current_value': self.current_value,
            'timestamp': self.timestamp.isoformat(),
            'zone_id': self.zone_id,
            'camera_id': self.camera_id
        }


@dataclass
class RealTimeAnalyticsState:
    """Current state of real-time analytics."""
    total_persons: int = 0
    persons_per_camera: Dict[str, int] = field(default_factory=dict)
    zone_occupancy: Dict[str, int] = field(default_factory=dict)
    active_alerts: List[AnalyticsAlert] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_persons': self.total_persons,
            'persons_per_camera': self.persons_per_camera,
            'zone_occupancy': self.zone_occupancy,
            'active_alerts': [alert.to_dict() for alert in self.active_alerts],
            'performance_metrics': self.performance_metrics,
            'last_updated': self.last_updated.isoformat()
        }


class AdvancedAnalyticsEngine:
    """Comprehensive analytics engine for real-time and historical analytics."""
    
    def __init__(
        self,
        historical_data_service: HistoricalDataService,
        temporal_query_engine: TemporalQueryEngine,
        tracking_cache: TrackingCache
    ):
        self.historical_service = historical_data_service
        self.query_engine = temporal_query_engine
        self.cache = tracking_cache
        
        # Real-time analytics state
        self.current_state = RealTimeAnalyticsState()
        
        # Zone definitions
        self.zones: Dict[str, ZoneDefinition] = {}
        self._initialize_default_zones()
        
        # Alert thresholds and rules
        self.alert_thresholds = {
            'high_occupancy': 10,
            'low_occupancy': 1,
            'dwell_time_exceeded': 300,  # 5 minutes
            'crowd_density': 0.8,
            'camera_failure': 30  # seconds without data
        }
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_history = deque(maxlen=1000)
        self.anomaly_trained = False
        
        # Performance monitoring
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Background processing
        self._analytics_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.analytics_stats = {
            'metrics_calculated': 0,
            'alerts_generated': 0,
            'anomalies_detected': 0,
            'zones_monitored': 0,
            'avg_processing_time_ms': 0.0
        }
        
        logger.info("AdvancedAnalyticsEngine initialized")
    
    async def start_analytics_engine(self):
        """Start the analytics engine and background processing."""
        if self._running:
            return
        
        self._running = True
        self._analytics_task = asyncio.create_task(self._analytics_processing_loop())
        
        logger.info("AdvancedAnalyticsEngine started")
    
    async def stop_analytics_engine(self):
        """Stop the analytics engine."""
        self._running = False
        
        if self._analytics_task and not self._analytics_task.done():
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AdvancedAnalyticsEngine stopped")
    
    # --- Real-Time Analytics ---
    
    async def update_real_time_analytics(
        self,
        current_detections: Dict[str, List[HistoricalDataPoint]]
    ):
        """Update real-time analytics with current detection data."""
        try:
            start_time = time.time()
            
            # Update person counts
            await self._update_person_counts(current_detections)
            
            # Update zone occupancy
            await self._update_zone_occupancy(current_detections)
            
            # Calculate performance metrics
            await self._calculate_performance_metrics(current_detections)
            
            # Detect anomalies
            await self._detect_anomalies()
            
            # Generate alerts
            await self._generate_alerts()
            
            # Update timestamp
            self.current_state.last_updated = datetime.utcnow()
            
            # Cache current state
            await self._cache_analytics_state()
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_stats(processing_time)
            
        except Exception as e:
            logger.error(f"Error updating real-time analytics: {e}")
    
    async def _update_person_counts(self, current_detections: Dict[str, List[HistoricalDataPoint]]):
        """Update person count metrics."""
        try:
            # Count persons per camera
            persons_per_camera = {}
            all_person_ids = set()
            
            for camera_id, detections in current_detections.items():
                unique_persons = set()
                for detection in detections:
                    unique_persons.add(detection.global_person_id)
                    all_person_ids.add(detection.global_person_id)
                
                persons_per_camera[camera_id] = len(unique_persons)
            
            self.current_state.persons_per_camera = persons_per_camera
            self.current_state.total_persons = len(all_person_ids)
            
        except Exception as e:
            logger.error(f"Error updating person counts: {e}")
    
    async def _update_zone_occupancy(self, current_detections: Dict[str, List[HistoricalDataPoint]]):
        """Update zone occupancy metrics."""
        try:
            zone_occupancy = defaultdict(set)
            
            for camera_id, detections in current_detections.items():
                for detection in detections:
                    if detection.coordinates:
                        # Check which zones this person is in
                        for zone_id, zone in self.zones.items():
                            if zone.contains_point(detection.coordinates.x, detection.coordinates.y):
                                zone_occupancy[zone_id].add(detection.global_person_id)
            
            # Convert to counts
            self.current_state.zone_occupancy = {
                zone_id: len(person_ids) 
                for zone_id, person_ids in zone_occupancy.items()
            }
            
            # Ensure all zones are represented
            for zone_id in self.zones.keys():
                if zone_id not in self.current_state.zone_occupancy:
                    self.current_state.zone_occupancy[zone_id] = 0
            
        except Exception as e:
            logger.error(f"Error updating zone occupancy: {e}")
    
    async def _calculate_performance_metrics(self, current_detections: Dict[str, List[HistoricalDataPoint]]):
        """Calculate performance metrics."""
        try:
            metrics = {}
            
            # Detection confidence average
            all_confidences = []
            for detections in current_detections.values():
                all_confidences.extend([d.detection.confidence for d in detections])
            
            if all_confidences:
                metrics['avg_detection_confidence'] = np.mean(all_confidences)
                metrics['min_detection_confidence'] = min(all_confidences)
            else:
                metrics['avg_detection_confidence'] = 0.0
                metrics['min_detection_confidence'] = 0.0
            
            # Camera load distribution
            camera_loads = list(self.current_state.persons_per_camera.values())
            if camera_loads:
                metrics['camera_load_std'] = np.std(camera_loads)
                metrics['camera_load_max'] = max(camera_loads)
                metrics['camera_load_balance'] = 1.0 - (np.std(camera_loads) / (np.mean(camera_loads) + 1e-6))
            else:
                metrics['camera_load_std'] = 0.0
                metrics['camera_load_max'] = 0
                metrics['camera_load_balance'] = 1.0
            
            # Crowd density estimation
            if self.current_state.total_persons > 0:
                total_area = sum(self._calculate_zone_area(zone) for zone in self.zones.values())
                metrics['crowd_density'] = self.current_state.total_persons / (total_area + 1e-6)
            else:
                metrics['crowd_density'] = 0.0
            
            self.current_state.performance_metrics = metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def _calculate_zone_area(self, zone: ZoneDefinition) -> float:
        """Calculate zone area using shoelace formula."""
        try:
            if len(zone.boundary_points) < 3:
                return 0.0
            
            area = 0.0
            n = len(zone.boundary_points)
            
            for i in range(n):
                j = (i + 1) % n
                area += zone.boundary_points[i][0] * zone.boundary_points[j][1]
                area -= zone.boundary_points[j][0] * zone.boundary_points[i][1]
            
            return abs(area) / 2.0
            
        except Exception as e:
            logger.error(f"Error calculating zone area: {e}")
            return 0.0
    
    async def _detect_anomalies(self):
        """Detect anomalies in current analytics state."""
        try:
            # Create feature vector from current state
            features = [
                self.current_state.total_persons,
                len(self.current_state.persons_per_camera),
                sum(self.current_state.zone_occupancy.values()),
                self.current_state.performance_metrics.get('crowd_density', 0.0),
                self.current_state.performance_metrics.get('camera_load_std', 0.0)
            ]
            
            # Add to history
            self.anomaly_history.append(features)
            
            # Train anomaly detector if we have enough data
            if len(self.anomaly_history) >= 50 and not self.anomaly_trained:
                self.anomaly_detector.fit(list(self.anomaly_history))
                self.anomaly_trained = True
                logger.info("Anomaly detector trained with historical data")
            
            # Detect anomalies if trained
            if self.anomaly_trained:
                anomaly_score = self.anomaly_detector.decision_function([features])[0]
                is_anomaly = self.anomaly_detector.predict([features])[0] == -1
                
                if is_anomaly:
                    alert = AnalyticsAlert(
                        alert_id=f"anomaly_{int(time.time())}",
                        severity=AlertSeverity.MEDIUM,
                        message=f"Anomaly detected in analytics data (score: {anomaly_score:.2f})",
                        metric_name="anomaly_score",
                        threshold_value=-0.5,
                        current_value=anomaly_score,
                        timestamp=datetime.utcnow()
                    )
                    
                    self.current_state.active_alerts.append(alert)
                    self.analytics_stats['anomalies_detected'] += 1
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
    
    async def _generate_alerts(self):
        """Generate alerts based on current metrics and thresholds."""
        try:
            new_alerts = []
            
            # High occupancy alerts
            for zone_id, occupancy in self.current_state.zone_occupancy.items():
                zone = self.zones.get(zone_id)
                if zone and zone.capacity_limit and occupancy > zone.capacity_limit:
                    alert = AnalyticsAlert(
                        alert_id=f"high_occupancy_{zone_id}_{int(time.time())}",
                        severity=AlertSeverity.HIGH,
                        message=f"High occupancy in zone {zone.name}: {occupancy}/{zone.capacity_limit}",
                        metric_name="zone_occupancy",
                        threshold_value=float(zone.capacity_limit),
                        current_value=float(occupancy),
                        timestamp=datetime.utcnow(),
                        zone_id=zone_id
                    )
                    new_alerts.append(alert)
            
            # Crowd density alerts
            crowd_density = self.current_state.performance_metrics.get('crowd_density', 0.0)
            if crowd_density > self.alert_thresholds['crowd_density']:
                alert = AnalyticsAlert(
                    alert_id=f"high_density_{int(time.time())}",
                    severity=AlertSeverity.MEDIUM,
                    message=f"High crowd density detected: {crowd_density:.2f}",
                    metric_name="crowd_density",
                    threshold_value=self.alert_thresholds['crowd_density'],
                    current_value=crowd_density,
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)
            
            # Camera imbalance alerts
            camera_load_std = self.current_state.performance_metrics.get('camera_load_std', 0.0)
            if camera_load_std > 5.0:  # High standard deviation indicates imbalance
                alert = AnalyticsAlert(
                    alert_id=f"camera_imbalance_{int(time.time())}",
                    severity=AlertSeverity.LOW,
                    message=f"Camera load imbalance detected (std: {camera_load_std:.2f})",
                    metric_name="camera_load_balance",
                    threshold_value=5.0,
                    current_value=camera_load_std,
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)
            
            # Add new alerts and clean up old ones
            self.current_state.active_alerts.extend(new_alerts)
            self._cleanup_old_alerts()
            
            self.analytics_stats['alerts_generated'] += len(new_alerts)
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
    
    def _cleanup_old_alerts(self):
        """Remove old alerts from active list."""
        try:
            current_time = datetime.utcnow()
            active_alerts = []
            
            for alert in self.current_state.active_alerts:
                # Keep alerts for 5 minutes
                if (current_time - alert.timestamp).total_seconds() < 300:
                    active_alerts.append(alert)
            
            self.current_state.active_alerts = active_alerts
            
        except Exception as e:
            logger.error(f"Error cleaning up alerts: {e}")
    
    # --- Historical Analytics ---
    
    async def calculate_historical_metrics(
        self,
        time_range: TimeRange,
        environment_id: Optional[str] = None,
        granularity: AnalyticsGranularity = AnalyticsGranularity.HOUR
    ) -> Dict[str, Any]:
        """Calculate comprehensive historical analytics metrics."""
        try:
            start_time = time.time()
            
            # Query historical data
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id
            )
            
            data_points = await self.historical_service.query_historical_data(query_filter)
            
            if not data_points:
                return {
                    'time_range': {
                        'start': time_range.start_time.isoformat(),
                        'end': time_range.end_time.isoformat()
                    },
                    'total_data_points': 0,
                    'metrics': {}
                }
            
            # Calculate metrics
            metrics = {
                'occupancy_trends': await self._calculate_occupancy_trends(data_points, granularity),
                'movement_patterns': await self._analyze_movement_patterns(data_points),
                'peak_analysis': await self._analyze_peak_periods(data_points, granularity),
                'zone_analytics': await self._calculate_zone_analytics(data_points, granularity),
                'camera_analytics': await self._calculate_camera_analytics(data_points, granularity),
                'behavioral_metrics': await self._calculate_behavioral_metrics(data_points)
            }
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'time_range': {
                    'start': time_range.start_time.isoformat(),
                    'end': time_range.end_time.isoformat(),
                    'duration_hours': time_range.duration_hours
                },
                'total_data_points': len(data_points),
                'unique_persons': len(set(dp.global_person_id for dp in data_points)),
                'cameras_involved': len(set(dp.camera_id for dp in data_points)),
                'metrics': metrics,
                'processing_time_ms': processing_time
            }
            
            logger.info(f"Calculated historical metrics for {len(data_points)} data points in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating historical metrics: {e}")
            raise
    
    async def _calculate_occupancy_trends(
        self,
        data_points: List[HistoricalDataPoint],
        granularity: AnalyticsGranularity
    ) -> Dict[str, Any]:
        """Calculate occupancy trends over time."""
        try:
            # Group data by time intervals
            time_buckets = defaultdict(set)
            
            for data_point in data_points:
                time_key = self._get_time_bucket_key(data_point.timestamp, granularity)
                time_buckets[time_key].add(data_point.global_person_id)
            
            # Calculate trends
            timestamps = sorted(time_buckets.keys())
            occupancy_counts = [len(time_buckets[ts]) for ts in timestamps]
            
            if len(occupancy_counts) > 1:
                # Linear trend analysis
                x = np.arange(len(occupancy_counts))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, occupancy_counts)
                
                trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            else:
                slope, r_value, trend = 0.0, 0.0, "stable"
            
            return {
                'time_series': [
                    {'timestamp': ts, 'occupancy': len(time_buckets[ts])}
                    for ts in timestamps
                ],
                'trend_analysis': {
                    'trend': trend,
                    'slope': slope,
                    'correlation': r_value,
                    'average_occupancy': np.mean(occupancy_counts) if occupancy_counts else 0.0,
                    'max_occupancy': max(occupancy_counts) if occupancy_counts else 0,
                    'min_occupancy': min(occupancy_counts) if occupancy_counts else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating occupancy trends: {e}")
            return {}
    
    async def _analyze_movement_patterns(self, data_points: List[HistoricalDataPoint]) -> Dict[str, Any]:
        """Analyze movement patterns from historical data."""
        try:
            # Group by person
            person_trajectories = defaultdict(list)
            for data_point in data_points:
                if data_point.coordinates:
                    person_trajectories[data_point.global_person_id].append(data_point)
            
            # Calculate movement metrics
            total_distances = []
            speeds = []
            dwell_times = []
            
            for person_id, trajectory in person_trajectories.items():
                trajectory.sort(key=lambda x: x.timestamp)
                
                if len(trajectory) >= 2:
                    # Calculate total distance
                    total_distance = 0.0
                    for i in range(1, len(trajectory)):
                        prev_point = trajectory[i-1]
                        curr_point = trajectory[i]
                        
                        dx = curr_point.coordinates.x - prev_point.coordinates.x
                        dy = curr_point.coordinates.y - prev_point.coordinates.y
                        distance = np.sqrt(dx**2 + dy**2)
                        total_distance += distance
                    
                    total_distances.append(total_distance)
                    
                    # Calculate average speed
                    total_time = (trajectory[-1].timestamp - trajectory[0].timestamp).total_seconds()
                    if total_time > 0:
                        avg_speed = total_distance / total_time
                        speeds.append(avg_speed)
                    
                    # Calculate dwell time (time in environment)
                    dwell_times.append(total_time)
            
            return {
                'distance_statistics': {
                    'mean': np.mean(total_distances) if total_distances else 0.0,
                    'std': np.std(total_distances) if total_distances else 0.0,
                    'median': np.median(total_distances) if total_distances else 0.0
                },
                'speed_statistics': {
                    'mean': np.mean(speeds) if speeds else 0.0,
                    'std': np.std(speeds) if speeds else 0.0,
                    'median': np.median(speeds) if speeds else 0.0
                },
                'dwell_time_statistics': {
                    'mean': np.mean(dwell_times) if dwell_times else 0.0,
                    'std': np.std(dwell_times) if dwell_times else 0.0,
                    'median': np.median(dwell_times) if dwell_times else 0.0
                },
                'trajectory_count': len(person_trajectories)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing movement patterns: {e}")
            return {}
    
    async def _analyze_peak_periods(
        self,
        data_points: List[HistoricalDataPoint],
        granularity: AnalyticsGranularity
    ) -> Dict[str, Any]:
        """Analyze peak occupancy periods."""
        try:
            # Group by time buckets
            time_occupancy = defaultdict(set)
            
            for data_point in data_points:
                time_key = self._get_time_bucket_key(data_point.timestamp, granularity)
                time_occupancy[time_key].add(data_point.global_person_id)
            
            if not time_occupancy:
                return {}
            
            # Find peak periods
            occupancy_counts = [(ts, len(persons)) for ts, persons in time_occupancy.items()]
            occupancy_counts.sort(key=lambda x: x[1], reverse=True)
            
            # Identify peaks (top 10% of periods)
            peak_threshold = int(len(occupancy_counts) * 0.1) or 1
            peak_periods = occupancy_counts[:peak_threshold]
            
            # Analyze peak patterns
            peak_hours = defaultdict(int)
            peak_days = defaultdict(int)
            
            for timestamp_str, count in peak_periods:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    peak_hours[timestamp.hour] += 1
                    peak_days[timestamp.weekday()] += 1
                except:
                    continue
            
            return {
                'peak_periods': [
                    {'timestamp': ts, 'occupancy': count}
                    for ts, count in peak_periods
                ],
                'peak_hour_distribution': dict(peak_hours),
                'peak_day_distribution': dict(peak_days),
                'average_peak_occupancy': np.mean([count for _, count in peak_periods]),
                'peak_threshold': occupancy_counts[peak_threshold-1][1] if len(occupancy_counts) >= peak_threshold else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing peak periods: {e}")
            return {}
    
    async def _calculate_zone_analytics(
        self,
        data_points: List[HistoricalDataPoint],
        granularity: AnalyticsGranularity
    ) -> Dict[str, Any]:
        """Calculate zone-specific analytics."""
        try:
            zone_analytics = {}
            
            for zone_id, zone in self.zones.items():
                zone_data_points = []
                
                # Filter data points within zone
                for data_point in data_points:
                    if data_point.coordinates and zone.contains_point(
                        data_point.coordinates.x, data_point.coordinates.y
                    ):
                        zone_data_points.append(data_point)
                
                if not zone_data_points:
                    zone_analytics[zone_id] = {
                        'total_visits': 0,
                        'unique_persons': 0,
                        'average_dwell_time': 0.0,
                        'utilization_rate': 0.0
                    }
                    continue
                
                # Calculate zone metrics
                unique_persons = set(dp.global_person_id for dp in zone_data_points)
                
                # Calculate dwell times per person
                person_visits = defaultdict(list)
                for dp in zone_data_points:
                    person_visits[dp.global_person_id].append(dp.timestamp)
                
                dwell_times = []
                for person_id, timestamps in person_visits.items():
                    timestamps.sort()
                    if len(timestamps) >= 2:
                        dwell_time = (timestamps[-1] - timestamps[0]).total_seconds()
                        dwell_times.append(dwell_time)
                
                # Calculate utilization rate (proportion of time zone was occupied)
                time_buckets = defaultdict(set)
                for dp in zone_data_points:
                    time_key = self._get_time_bucket_key(dp.timestamp, granularity)
                    time_buckets[time_key].add(dp.global_person_id)
                
                occupied_periods = sum(1 for persons in time_buckets.values() if persons)
                total_periods = len(time_buckets) or 1
                utilization_rate = occupied_periods / total_periods
                
                zone_analytics[zone_id] = {
                    'total_visits': len(zone_data_points),
                    'unique_persons': len(unique_persons),
                    'average_dwell_time': np.mean(dwell_times) if dwell_times else 0.0,
                    'max_dwell_time': max(dwell_times) if dwell_times else 0.0,
                    'utilization_rate': utilization_rate,
                    'zone_info': zone.to_dict()
                }
            
            return zone_analytics
            
        except Exception as e:
            logger.error(f"Error calculating zone analytics: {e}")
            return {}
    
    async def _calculate_camera_analytics(
        self,
        data_points: List[HistoricalDataPoint],
        granularity: AnalyticsGranularity
    ) -> Dict[str, Any]:
        """Calculate camera-specific analytics."""
        try:
            camera_analytics = defaultdict(lambda: {
                'detection_count': 0,
                'unique_persons': set(),
                'avg_confidence': [],
                'time_distribution': defaultdict(int)
            })
            
            # Process data points
            for data_point in data_points:
                camera_id = data_point.camera_id
                camera_analytics[camera_id]['detection_count'] += 1
                camera_analytics[camera_id]['unique_persons'].add(data_point.global_person_id)
                camera_analytics[camera_id]['avg_confidence'].append(data_point.detection.confidence)
                
                time_key = self._get_time_bucket_key(data_point.timestamp, granularity)
                camera_analytics[camera_id]['time_distribution'][time_key] += 1
            
            # Process results
            result = {}
            for camera_id, analytics in camera_analytics.items():
                result[camera_id] = {
                    'detection_count': analytics['detection_count'],
                    'unique_persons': len(analytics['unique_persons']),
                    'average_confidence': np.mean(analytics['avg_confidence']) if analytics['avg_confidence'] else 0.0,
                    'min_confidence': min(analytics['avg_confidence']) if analytics['avg_confidence'] else 0.0,
                    'max_confidence': max(analytics['avg_confidence']) if analytics['avg_confidence'] else 0.0,
                    'activity_distribution': dict(analytics['time_distribution']),
                    'peak_activity_period': max(analytics['time_distribution'], key=analytics['time_distribution'].get) if analytics['time_distribution'] else None
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating camera analytics: {e}")
            return {}
    
    async def _calculate_behavioral_metrics(self, data_points: List[HistoricalDataPoint]) -> Dict[str, Any]:
        """Calculate behavioral analysis metrics."""
        try:
            # Group by person
            person_behaviors = defaultdict(list)
            for data_point in data_points:
                person_behaviors[data_point.global_person_id].append(data_point)
            
            # Analyze behaviors
            behavior_patterns = {
                'frequent_visitors': 0,  # Persons with >5 detections
                'brief_visitors': 0,     # Persons with <3 detections
                'long_stay_visitors': 0,  # Persons with >10 minutes in environment
                'cross_camera_movers': 0  # Persons detected in multiple cameras
            }
            
            for person_id, detections in person_behaviors.items():
                detection_count = len(detections)
                unique_cameras = set(d.camera_id for d in detections)
                
                # Duration calculation
                detections.sort(key=lambda x: x.timestamp)
                if len(detections) >= 2:
                    duration = (detections[-1].timestamp - detections[0].timestamp).total_seconds()
                else:
                    duration = 0
                
                # Classify behavior patterns
                if detection_count > 5:
                    behavior_patterns['frequent_visitors'] += 1
                elif detection_count < 3:
                    behavior_patterns['brief_visitors'] += 1
                
                if duration > 600:  # 10 minutes
                    behavior_patterns['long_stay_visitors'] += 1
                
                if len(unique_cameras) > 1:
                    behavior_patterns['cross_camera_movers'] += 1
            
            total_persons = len(person_behaviors)
            
            return {
                'total_persons_analyzed': total_persons,
                'behavior_patterns': behavior_patterns,
                'behavior_percentages': {
                    pattern: (count / total_persons * 100) if total_persons > 0 else 0.0
                    for pattern, count in behavior_patterns.items()
                },
                'average_detections_per_person': np.mean([len(detections) for detections in person_behaviors.values()]) if person_behaviors else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating behavioral metrics: {e}")
            return {}
    
    def _get_time_bucket_key(self, timestamp: datetime, granularity: AnalyticsGranularity) -> str:
        """Get time bucket key based on granularity."""
        if granularity == AnalyticsGranularity.SECOND:
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
        elif granularity == AnalyticsGranularity.MINUTE:
            return timestamp.strftime("%Y-%m-%d %H:%M")
        elif granularity == AnalyticsGranularity.HOUR:
            return timestamp.strftime("%Y-%m-%d %H:00")
        elif granularity == AnalyticsGranularity.DAY:
            return timestamp.strftime("%Y-%m-%d")
        else:
            return timestamp.strftime("%Y-%m-%d %H:00")
    
    # --- Zone Management ---
    
    def add_zone(self, zone: ZoneDefinition):
        """Add a zone definition for analytics."""
        self.zones[zone.zone_id] = zone
        self.analytics_stats['zones_monitored'] = len(self.zones)
        logger.info(f"Added zone {zone.zone_id}: {zone.name}")
    
    def remove_zone(self, zone_id: str):
        """Remove a zone definition."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.analytics_stats['zones_monitored'] = len(self.zones)
            logger.info(f"Removed zone {zone_id}")
    
    def get_zone(self, zone_id: str) -> Optional[ZoneDefinition]:
        """Get zone definition by ID."""
        return self.zones.get(zone_id)
    
    def list_zones(self) -> List[ZoneDefinition]:
        """List all zone definitions."""
        return list(self.zones.values())
    
    def _initialize_default_zones(self):
        """Initialize default zone definitions."""
        # Example zones - these would be configured based on environment
        default_zones = [
            ZoneDefinition(
                zone_id="entrance",
                name="Entrance Zone",
                boundary_points=[(0, 0), (10, 0), (10, 5), (0, 5)],
                zone_type="entrance",
                capacity_limit=5
            ),
            ZoneDefinition(
                zone_id="main_area",
                name="Main Area",
                boundary_points=[(10, 0), (50, 0), (50, 30), (10, 30)],
                zone_type="main",
                capacity_limit=20
            ),
            ZoneDefinition(
                zone_id="exit",
                name="Exit Zone",
                boundary_points=[(50, 0), (60, 0), (60, 5), (50, 5)],
                zone_type="exit",
                capacity_limit=5
            )
        ]
        
        for zone in default_zones:
            self.zones[zone.zone_id] = zone
        
        self.analytics_stats['zones_monitored'] = len(self.zones)
    
    # --- Background Processing ---
    
    async def _analytics_processing_loop(self):
        """Background analytics processing loop."""
        while self._running:
            try:
                # Update analytics every 30 seconds
                await asyncio.sleep(30)
                
                # Get current tracking data from cache
                current_data = await self._get_current_tracking_data()
                if current_data:
                    await self.update_real_time_analytics(current_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analytics processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _get_current_tracking_data(self) -> Optional[Dict[str, List[HistoricalDataPoint]]]:
        """Get current tracking data from cache."""
        try:
            # This would integrate with the tracking cache to get current data
            # For now, return None as placeholder
            return None
            
        except Exception as e:
            logger.error(f"Error getting current tracking data: {e}")
            return None
    
    async def _cache_analytics_state(self):
        """Cache current analytics state."""
        try:
            cache_key = "analytics_state_current"
            state_data = self.current_state.to_dict()
            
            await self.cache.set_json(cache_key, state_data, ttl=300)  # 5 minutes TTL
            
        except Exception as e:
            logger.error(f"Error caching analytics state: {e}")
    
    # --- Utility Methods ---
    
    def _update_processing_stats(self, processing_time_ms: float):
        """Update processing statistics."""
        self.analytics_stats['metrics_calculated'] += 1
        
        # Update average processing time
        current_avg = self.analytics_stats['avg_processing_time_ms']
        count = self.analytics_stats['metrics_calculated']
        
        self.analytics_stats['avg_processing_time_ms'] = (
            (current_avg * (count - 1) + processing_time_ms) / count
        )
    
    def get_current_analytics_state(self) -> Dict[str, Any]:
        """Get current real-time analytics state."""
        return self.current_state.to_dict()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'AdvancedAnalyticsEngine',
            'running': self._running,
            'zones_configured': len(self.zones),
            'alert_thresholds': self.alert_thresholds,
            'current_state': {
                'total_persons': self.current_state.total_persons,
                'active_alerts': len(self.current_state.active_alerts),
                'last_updated': self.current_state.last_updated.isoformat()
            },
            'statistics': self.analytics_stats.copy(),
            'anomaly_detection': {
                'trained': self.anomaly_trained,
                'history_size': len(self.anomaly_history)
            }
        }


# Global service instance
_advanced_analytics_engine: Optional[AdvancedAnalyticsEngine] = None


def get_advanced_analytics_engine() -> Optional[AdvancedAnalyticsEngine]:
    """Get the global advanced analytics engine instance."""
    return _advanced_analytics_engine


def initialize_advanced_analytics_engine(
    historical_data_service: HistoricalDataService,
    temporal_query_engine: TemporalQueryEngine,
    tracking_cache: TrackingCache
) -> AdvancedAnalyticsEngine:
    """Initialize the global advanced analytics engine."""
    global _advanced_analytics_engine
    if _advanced_analytics_engine is None:
        _advanced_analytics_engine = AdvancedAnalyticsEngine(
            historical_data_service, temporal_query_engine, tracking_cache
        )
    return _advanced_analytics_engine