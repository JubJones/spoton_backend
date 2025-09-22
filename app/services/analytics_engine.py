"""
Advanced analytics engine for person tracking and behavior analysis.

Handles:
- Real-time analytics processing
- Person behavior analysis
- Historical data queries
- Path prediction algorithms
- Anomaly detection
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.infrastructure.cache.tracking_cache import tracking_cache
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.mapping.entities.trajectory import Trajectory
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


@dataclass
class PersonBehaviorProfile:
    """Person behavior analysis profile."""
    person_id: str
    total_detections: int = 0
    total_time_tracked: float = 0.0  # seconds
    average_speed: float = 0.0  # units per second
    dwell_time: float = 0.0  # seconds
    path_complexity: float = 0.0  # path complexity score
    cameras_visited: List[str] = field(default_factory=list)
    frequent_areas: List[Dict[str, Any]] = field(default_factory=list)
    activity_patterns: Dict[str, Any] = field(default_factory=dict)
    anomaly_score: float = 0.0


@dataclass
class AnalyticsReport:
    """Analytics report data structure."""
    report_id: str
    report_type: str
    environment_id: str
    time_range: Dict[str, datetime]
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime
    
    
@dataclass
class RealTimeMetrics:
    """Real-time metrics data structure."""
    timestamp: datetime
    active_persons: int
    detection_rate: float
    average_confidence: float
    camera_loads: Dict[str, int]
    performance_metrics: Dict[str, float]


class AnalyticsEngine:
    """
    Advanced analytics engine for person tracking system.
    
    Features:
    - Real-time analytics processing
    - Person behavior analysis
    - Historical data analysis
    - Path prediction
    - Anomaly detection
    """
    
    def __init__(self):
        self.analytics_cache = {}
        self.behavior_profiles = {}
        self.real_time_metrics = deque(maxlen=1000)
        self.prediction_models = {}
        
        # Configuration
        self.min_tracking_time = 5.0  # minimum seconds for behavior analysis
        self.anomaly_threshold = 0.7  # anomaly detection threshold
        self.clustering_eps = 0.5  # DBSCAN clustering parameter
        self.clustering_min_samples = 3
        
        # Performance tracking
        self.analytics_stats = {
            'total_analyses': 0,
            'behavior_profiles_created': 0,
            'anomalies_detected': 0,
            'predictions_made': 0,
            'reports_generated': 0
        }
        
        logger.info("AnalyticsEngine initialized")
    
    async def initialize(self):
        """Initialize the analytics engine."""
        try:
            # Start real-time analytics processing
            asyncio.create_task(self._real_time_analytics_loop())
            
            # Initialize prediction models
            await self._initialize_prediction_models()
            
            logger.info("AnalyticsEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AnalyticsEngine: {e}")
            raise
    
    # Real-Time Analytics
    async def _real_time_analytics_loop(self):
        """Main real-time analytics processing loop."""
        try:
            while True:
                await self._process_real_time_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds
                
        except asyncio.CancelledError:
            logger.info("Real-time analytics loop cancelled")
        except Exception as e:
            logger.error(f"Error in real-time analytics loop: {e}")
    
    async def _process_real_time_metrics(self):
        """Process real-time metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Get active persons from cache
            active_persons = await tracking_cache.get_active_persons()
            
            # Calculate metrics
            detection_rate = await self._calculate_detection_rate()
            average_confidence = await self._calculate_average_confidence()
            camera_loads = await self._calculate_camera_loads()
            performance_metrics = await self._get_performance_metrics()
            
            # Store metrics
            metrics = RealTimeMetrics(
                timestamp=current_time,
                active_persons=len(active_persons),
                detection_rate=detection_rate,
                average_confidence=average_confidence,
                camera_loads=camera_loads,
                performance_metrics=performance_metrics
            )
            
            self.real_time_metrics.append(metrics)
            
            # Update analytics cache
            self.analytics_cache['real_time_metrics'] = {
                'timestamp': current_time.isoformat(),
                'active_persons': len(active_persons),
                'detection_rate': detection_rate,
                'average_confidence': average_confidence,
                'camera_loads': camera_loads,
                'performance_metrics': performance_metrics
            }
            
            logger.debug(f"Processed real-time metrics: {len(active_persons)} active persons")
            
        except Exception as e:
            logger.error(f"Error processing real-time metrics: {e}")
    
    async def _calculate_detection_rate(self) -> float:
        """Calculate current detection rate."""
        try:
            # Get recent detection statistics
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=1)
            
            stats = await integrated_db_service.get_detection_statistics(
                environment_id="default",
                start_time=start_time,
                end_time=end_time
            )
            
            return stats.get('total_detections', 0) / 60.0  # detections per second
            
        except Exception as e:
            logger.error(f"Error calculating detection rate: {e}")
            return 0.0
    
    async def _calculate_average_confidence(self) -> float:
        """Calculate average detection confidence."""
        try:
            # Get recent detection statistics
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=5)
            
            stats = await integrated_db_service.get_detection_statistics(
                environment_id="default",
                start_time=start_time,
                end_time=end_time
            )
            
            return stats.get('avg_confidence', 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating average confidence: {e}")
            return 0.0
    
    async def _calculate_camera_loads(self) -> Dict[str, int]:
        """Calculate current load per camera."""
        try:
            camera_loads = {}
            
            # Get active persons by camera
            active_persons = await tracking_cache.get_active_persons()
            
            for person in active_persons:
                camera_id = person.last_seen_camera
                camera_loads[camera_id] = camera_loads.get(camera_id, 0) + 1
            
            return camera_loads
            
        except Exception as e:
            logger.error(f"Error calculating camera loads: {e}")
            return {}
    
    async def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        try:
            # Get cache statistics
            cache_stats = await tracking_cache.get_cache_stats()
            
            # Extract relevant performance metrics
            return {
                'cache_hit_rate': cache_stats.get('cache_stats', {}).get('hits', 0) / 
                                 max(1, cache_stats.get('cache_stats', {}).get('hits', 0) + 
                                     cache_stats.get('cache_stats', {}).get('misses', 0)),
                'memory_usage': 0.0,  # Would be implemented based on system monitoring
                'processing_latency': 0.0,  # Would be implemented based on timing metrics
                'error_rate': 0.0  # Would be implemented based on error tracking
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    # Person Behavior Analysis
    async def analyze_person_behavior(
        self,
        person_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Optional[PersonBehaviorProfile]:
        """Analyze person behavior patterns."""
        try:
            self.analytics_stats['total_analyses'] += 1
            
            # Get person trajectory data
            start_time, end_time = time_range or (
                datetime.now(timezone.utc) - timedelta(days=1),
                datetime.now(timezone.utc)
            )
            
            trajectory_data = await integrated_db_service.get_person_trajectory(
                global_person_id=person_id,
                start_time=start_time,
                end_time=end_time
            )
            
            if not trajectory_data:
                return None
            
            # Calculate behavior metrics
            total_time = self._calculate_total_time(trajectory_data)
            
            if total_time < self.min_tracking_time:
                return None
            
            average_speed = self._calculate_average_speed(trajectory_data)
            dwell_time = self._calculate_dwell_time(trajectory_data)
            path_complexity = self._calculate_path_complexity(trajectory_data)
            cameras_visited = self._get_cameras_visited(trajectory_data)
            frequent_areas = await self._identify_frequent_areas(trajectory_data)
            activity_patterns = self._analyze_activity_patterns(trajectory_data)
            anomaly_score = await self._calculate_anomaly_score(trajectory_data)
            
            # Create behavior profile
            profile = PersonBehaviorProfile(
                person_id=person_id,
                total_detections=len(trajectory_data),
                total_time_tracked=total_time,
                average_speed=average_speed,
                dwell_time=dwell_time,
                path_complexity=path_complexity,
                cameras_visited=cameras_visited,
                frequent_areas=frequent_areas,
                activity_patterns=activity_patterns,
                anomaly_score=anomaly_score
            )
            
            # Cache behavior profile
            self.behavior_profiles[person_id] = profile
            self.analytics_stats['behavior_profiles_created'] += 1
            
            if anomaly_score > self.anomaly_threshold:
                self.analytics_stats['anomalies_detected'] += 1
                logger.warning(f"Anomaly detected for person {person_id}: score {anomaly_score}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing person behavior: {e}")
            return None
    
    def _calculate_total_time(self, trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate total tracking time."""
        try:
            if len(trajectory_data) < 2:
                return 0.0
            
            first_timestamp = trajectory_data[0]['timestamp']
            last_timestamp = trajectory_data[-1]['timestamp']
            
            if isinstance(first_timestamp, str):
                first_timestamp = datetime.fromisoformat(first_timestamp.replace('Z', '+00:00'))
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
            
            return (last_timestamp - first_timestamp).total_seconds()
            
        except Exception as e:
            logger.error(f"Error calculating total time: {e}")
            return 0.0
    
    def _calculate_average_speed(self, trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate average movement speed."""
        try:
            if len(trajectory_data) < 2:
                return 0.0
            
            speeds = []
            for i in range(1, len(trajectory_data)):
                prev_point = trajectory_data[i-1]
                curr_point = trajectory_data[i]
                
                # Calculate distance
                dx = curr_point['position_x'] - prev_point['position_x']
                dy = curr_point['position_y'] - prev_point['position_y']
                distance = np.sqrt(dx**2 + dy**2)
                
                # Calculate time difference
                prev_time = prev_point['timestamp']
                curr_time = curr_point['timestamp']
                
                if isinstance(prev_time, str):
                    prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                if isinstance(curr_time, str):
                    curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                
                time_diff = (curr_time - prev_time).total_seconds()
                
                if time_diff > 0:
                    speeds.append(distance / time_diff)
            
            return statistics.mean(speeds) if speeds else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average speed: {e}")
            return 0.0
    
    def _calculate_dwell_time(self, trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate dwell time in specific areas."""
        try:
            if len(trajectory_data) < 2:
                return 0.0
            
            # Simple dwell time calculation based on low movement
            dwell_time = 0.0
            movement_threshold = 5.0  # pixels
            
            for i in range(1, len(trajectory_data)):
                prev_point = trajectory_data[i-1]
                curr_point = trajectory_data[i]
                
                # Calculate distance
                dx = curr_point['position_x'] - prev_point['position_x']
                dy = curr_point['position_y'] - prev_point['position_y']
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < movement_threshold:
                    # Calculate time difference
                    prev_time = prev_point['timestamp']
                    curr_time = curr_point['timestamp']
                    
                    if isinstance(prev_time, str):
                        prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                    if isinstance(curr_time, str):
                        curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                    
                    dwell_time += (curr_time - prev_time).total_seconds()
            
            return dwell_time
            
        except Exception as e:
            logger.error(f"Error calculating dwell time: {e}")
            return 0.0
    
    def _calculate_path_complexity(self, trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate path complexity score."""
        try:
            if len(trajectory_data) < 3:
                return 0.0
            
            # Calculate path complexity based on direction changes
            direction_changes = 0
            prev_direction = None
            
            for i in range(1, len(trajectory_data)):
                prev_point = trajectory_data[i-1]
                curr_point = trajectory_data[i]
                
                # Calculate direction
                dx = curr_point['position_x'] - prev_point['position_x']
                dy = curr_point['position_y'] - prev_point['position_y']
                
                if dx != 0 or dy != 0:
                    direction = np.arctan2(dy, dx)
                    
                    if prev_direction is not None:
                        # Calculate direction change
                        direction_change = abs(direction - prev_direction)
                        if direction_change > np.pi:
                            direction_change = 2 * np.pi - direction_change
                        
                        # Count significant direction changes
                        if direction_change > np.pi / 4:  # 45 degrees
                            direction_changes += 1
                    
                    prev_direction = direction
            
            # Normalize by path length
            return direction_changes / max(1, len(trajectory_data) - 1)
            
        except Exception as e:
            logger.error(f"Error calculating path complexity: {e}")
            return 0.0
    
    def _get_cameras_visited(self, trajectory_data: List[Dict[str, Any]]) -> List[str]:
        """Get list of cameras visited."""
        try:
            cameras = set()
            for point in trajectory_data:
                cameras.add(point['camera_id'])
            return sorted(list(cameras))
            
        except Exception as e:
            logger.error(f"Error getting cameras visited: {e}")
            return []
    
    async def _identify_frequent_areas(self, trajectory_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify frequently visited areas using clustering."""
        try:
            if len(trajectory_data) < self.clustering_min_samples:
                return []
            
            # Extract position data
            positions = []
            for point in trajectory_data:
                positions.append([point['position_x'], point['position_y']])
            
            positions = np.array(positions)
            
            # Normalize positions
            scaler = StandardScaler()
            positions_scaled = scaler.fit_transform(positions)
            
            # Perform clustering
            clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.clustering_min_samples)
            labels = clustering.fit_predict(positions_scaled)
            
            # Identify frequent areas
            frequent_areas = []
            for label in set(labels):
                if label == -1:  # Skip noise points
                    continue
                
                cluster_points = positions[labels == label]
                if len(cluster_points) >= self.clustering_min_samples:
                    center = np.mean(cluster_points, axis=0)
                    frequent_areas.append({
                        'center_x': float(center[0]),
                        'center_y': float(center[1]),
                        'point_count': len(cluster_points),
                        'dwell_time': len(cluster_points) * 0.1  # Approximate dwell time
                    })
            
            return frequent_areas
            
        except Exception as e:
            logger.error(f"Error identifying frequent areas: {e}")
            return []
    
    def _analyze_activity_patterns(self, trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze activity patterns."""
        try:
            patterns = {
                'movement_patterns': [],
                'time_patterns': {},
                'speed_patterns': {}
            }
            
            # Analyze movement patterns
            if len(trajectory_data) >= 2:
                speeds = []
                for i in range(1, len(trajectory_data)):
                    prev_point = trajectory_data[i-1]
                    curr_point = trajectory_data[i]
                    
                    # Calculate speed
                    dx = curr_point['position_x'] - prev_point['position_x']
                    dy = curr_point['position_y'] - prev_point['position_y']
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    prev_time = prev_point['timestamp']
                    curr_time = curr_point['timestamp']
                    
                    if isinstance(prev_time, str):
                        prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                    if isinstance(curr_time, str):
                        curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                    
                    time_diff = (curr_time - prev_time).total_seconds()
                    
                    if time_diff > 0:
                        speeds.append(distance / time_diff)
                
                if speeds:
                    patterns['speed_patterns'] = {
                        'mean': statistics.mean(speeds),
                        'median': statistics.median(speeds),
                        'std': statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
                        'min': min(speeds),
                        'max': max(speeds)
                    }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing activity patterns: {e}")
            return {}
    
    async def _calculate_anomaly_score(self, trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate anomaly score for person behavior."""
        try:
            if len(trajectory_data) < 5:
                return 0.0
            
            anomaly_factors = []
            
            # Speed anomaly
            speeds = []
            for i in range(1, len(trajectory_data)):
                prev_point = trajectory_data[i-1]
                curr_point = trajectory_data[i]
                
                dx = curr_point['position_x'] - prev_point['position_x']
                dy = curr_point['position_y'] - prev_point['position_y']
                distance = np.sqrt(dx**2 + dy**2)
                
                prev_time = prev_point['timestamp']
                curr_time = curr_point['timestamp']
                
                if isinstance(prev_time, str):
                    prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                if isinstance(curr_time, str):
                    curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                
                time_diff = (curr_time - prev_time).total_seconds()
                
                if time_diff > 0:
                    speed = distance / time_diff
                    speeds.append(speed)
            
            if speeds:
                mean_speed = statistics.mean(speeds)
                std_speed = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
                
                # Calculate speed anomaly factor
                if std_speed > 0:
                    speed_anomaly = max(speeds) / (mean_speed + std_speed)
                    anomaly_factors.append(min(speed_anomaly / 3.0, 1.0))  # Normalize to 0-1
            
            # Path complexity anomaly
            path_complexity = self._calculate_path_complexity(trajectory_data)
            if path_complexity > 0.3:  # Threshold for complex paths
                anomaly_factors.append(min(path_complexity / 0.5, 1.0))
            
            # Return average anomaly score
            return statistics.mean(anomaly_factors) if anomaly_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.0
    
    # Path Prediction
    async def predict_person_path(
        self,
        person_id: str,
        current_position: Coordinate,
        prediction_horizon: int = 30  # seconds
    ) -> Optional[List[Dict[str, Any]]]:
        """Predict future path for a person."""
        try:
            self.analytics_stats['predictions_made'] += 1
            
            # Get historical trajectory data
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)
            
            trajectory_data = await integrated_db_service.get_person_trajectory(
                global_person_id=person_id,
                start_time=start_time,
                end_time=end_time
            )
            
            if len(trajectory_data) < 5:
                return None
            
            # Simple linear prediction based on recent movement
            recent_points = trajectory_data[-5:]  # Last 5 points
            
            # Calculate average velocity
            velocities = []
            for i in range(1, len(recent_points)):
                prev_point = recent_points[i-1]
                curr_point = recent_points[i]
                
                dx = curr_point['position_x'] - prev_point['position_x']
                dy = curr_point['position_y'] - prev_point['position_y']
                
                prev_time = prev_point['timestamp']
                curr_time = curr_point['timestamp']
                
                if isinstance(prev_time, str):
                    prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                if isinstance(curr_time, str):
                    curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                
                time_diff = (curr_time - prev_time).total_seconds()
                
                if time_diff > 0:
                    vx = dx / time_diff
                    vy = dy / time_diff
                    velocities.append((vx, vy))
            
            if not velocities:
                return None
            
            # Average velocity
            avg_vx = statistics.mean(v[0] for v in velocities)
            avg_vy = statistics.mean(v[1] for v in velocities)
            
            # Generate predictions
            predictions = []
            current_x = current_position.x
            current_y = current_position.y
            
            for t in range(1, prediction_horizon + 1):
                predicted_x = current_x + avg_vx * t
                predicted_y = current_y + avg_vy * t
                
                predictions.append({
                    'timestamp': (datetime.now(timezone.utc) + timedelta(seconds=t)).isoformat(),
                    'position_x': predicted_x,
                    'position_y': predicted_y,
                    'confidence': max(0.1, 1.0 - (t / prediction_horizon) * 0.8)  # Decreasing confidence
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting person path: {e}")
            return None
    
    async def _initialize_prediction_models(self):
        """Initialize prediction models."""
        try:
            # Initialize simple prediction models
            self.prediction_models = {
                'linear_predictor': {
                    'type': 'linear',
                    'parameters': {
                        'history_window': 5,
                        'max_prediction_horizon': 60
                    }
                }
            }
            
            logger.info("Prediction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing prediction models: {e}")
    
    # Analytics Reports
    async def generate_analytics_report(
        self,
        report_type: str,
        environment_id: str,
        time_range: Tuple[datetime, datetime],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[AnalyticsReport]:
        """Generate comprehensive analytics report."""
        try:
            self.analytics_stats['reports_generated'] += 1
            
            start_time, end_time = time_range
            report_id = f"{report_type}_{environment_id}_{int(datetime.now(timezone.utc).timestamp())}"
            
            # Get base statistics
            detection_stats = await integrated_db_service.get_detection_statistics(
                environment_id=environment_id,
                start_time=start_time,
                end_time=end_time
            )
            
            person_stats = await integrated_db_service.get_person_statistics(
                environment_id=environment_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Generate report based on type
            if report_type == 'summary':
                metrics, insights, recommendations = await self._generate_summary_report(
                    detection_stats, person_stats, parameters
                )
            elif report_type == 'behavior':
                metrics, insights, recommendations = await self._generate_behavior_report(
                    environment_id, start_time, end_time, parameters
                )
            elif report_type == 'performance':
                metrics, insights, recommendations = await self._generate_performance_report(
                    start_time, end_time, parameters
                )
            else:
                logger.warning(f"Unknown report type: {report_type}")
                return None
            
            # Create report
            report = AnalyticsReport(
                report_id=report_id,
                report_type=report_type,
                environment_id=environment_id,
                time_range={'start': start_time, 'end': end_time},
                metrics=metrics,
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.now(timezone.utc)
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return None
    
    async def _generate_summary_report(
        self,
        detection_stats: Dict[str, Any],
        person_stats: Dict[str, Any],
        parameters: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Generate summary report."""
        try:
            metrics = {
                'detection_statistics': detection_stats,
                'person_statistics': person_stats,
                'real_time_metrics': self.analytics_cache.get('real_time_metrics', {})
            }
            
            insights = []
            recommendations = []
            
            # Generate insights
            if detection_stats.get('total_detections', 0) > 0:
                insights.append(f"Total detections: {detection_stats['total_detections']}")
                insights.append(f"Average confidence: {detection_stats.get('avg_confidence', 0):.2f}")
            
            if person_stats.get('total_persons', 0) > 0:
                insights.append(f"Total persons tracked: {person_stats['total_persons']}")
                insights.append(f"Average tracking time: {person_stats.get('avg_tracking_time', 0):.1f}s")
            
            # Generate recommendations
            if detection_stats.get('avg_confidence', 0) < 0.7:
                recommendations.append("Consider adjusting detection thresholds to improve accuracy")
            
            if person_stats.get('avg_tracking_time', 0) < 30:
                recommendations.append("Short tracking times may indicate tracking issues")
            
            return metrics, insights, recommendations
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {}, [], []
    
    async def _generate_behavior_report(
        self,
        environment_id: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Generate behavior analysis report."""
        try:
            # Get behavior profiles
            behavior_profiles = []
            anomalies = []
            
            # This would be implemented to analyze behavior patterns
            # For now, return sample data
            metrics = {
                'behavior_profiles_analyzed': len(behavior_profiles),
                'anomalies_detected': len(anomalies),
                'behavior_patterns': {
                    'average_speed': 0.0,
                    'average_dwell_time': 0.0,
                    'most_visited_areas': []
                }
            }
            
            insights = [
                f"Analyzed {len(behavior_profiles)} behavior profiles",
                f"Detected {len(anomalies)} anomalies"
            ]
            
            recommendations = [
                "Monitor high-anomaly score persons for security concerns",
                "Optimize camera placement based on frequently visited areas"
            ]
            
            return metrics, insights, recommendations
            
        except Exception as e:
            logger.error(f"Error generating behavior report: {e}")
            return {}, [], []
    
    async def _generate_performance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Generate performance analysis report."""
        try:
            # Get performance metrics
            performance_data = list(self.real_time_metrics)
            
            if not performance_data:
                return {}, [], []
            
            # Calculate performance statistics
            detection_rates = [m.detection_rate for m in performance_data]
            confidence_scores = [m.average_confidence for m in performance_data]
            
            metrics = {
                'average_detection_rate': statistics.mean(detection_rates) if detection_rates else 0.0,
                'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0.0,
                'performance_trend': 'stable',  # Would be calculated based on trend analysis
                'system_load': {
                    'average_active_persons': statistics.mean([m.active_persons for m in performance_data]),
                    'peak_active_persons': max([m.active_persons for m in performance_data]) if performance_data else 0
                }
            }
            
            insights = [
                f"Average detection rate: {metrics['average_detection_rate']:.2f} detections/sec",
                f"Average confidence: {metrics['average_confidence']:.2f}",
                f"Peak concurrent persons: {metrics['system_load']['peak_active_persons']}"
            ]
            
            recommendations = [
                "Monitor system performance during peak hours",
                "Consider load balancing for high-traffic periods"
            ]
            
            return metrics, insights, recommendations
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}, [], []
    
    # Utilities
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        return self.analytics_cache.get('real_time_metrics', {})
    
    async def get_behavior_profile(self, person_id: str) -> Optional[PersonBehaviorProfile]:
        """Get cached behavior profile."""
        return self.behavior_profiles.get(person_id)
    
    async def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get analytics engine statistics."""
        return {
            'analytics_stats': self.analytics_stats,
            'cached_profiles': len(self.behavior_profiles),
            'real_time_metrics_count': len(self.real_time_metrics),
            'prediction_models': list(self.prediction_models.keys())
        }
    
    def reset_statistics(self):
        """Reset analytics statistics."""
        self.analytics_stats = {
            'total_analyses': 0,
            'behavior_profiles_created': 0,
            'anomalies_detected': 0,
            'predictions_made': 0,
            'reports_generated': 0
        }
        logger.info("Analytics statistics reset")


# Global analytics engine instance
analytics_engine = AnalyticsEngine()