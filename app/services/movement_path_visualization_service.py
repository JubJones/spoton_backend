"""
Movement Path Visualization and Trajectory Reconstruction Service

Advanced service for visualizing and analyzing person movement patterns:
- Person trajectory reconstruction from historical data points
- Path smoothing and interpolation for continuous visualization
- Multi-person path analysis and comparison capabilities
- Heatmap generation for occupancy and movement analysis
- Interactive path visualization with temporal controls
- Path clustering and pattern recognition
"""

import asyncio
import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from scipy.interpolate import interp1d, splrep, splev
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from app.services.historical_data_service import (
    HistoricalDataService,
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)
from app.services.temporal_query_engine import TemporalQueryEngine, TimeGranularity
from app.domains.mapping.entities.coordinate import Coordinate

logger = logging.getLogger(__name__)


class PathVisualizationMode(Enum):
    """Path visualization modes."""
    LINE = "line"
    DOTS = "dots"
    HEATMAP = "heatmap"
    ANIMATED = "animated"
    CLUSTERED = "clustered"


class InterpolationMethod(Enum):
    """Path interpolation methods."""
    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    KALMAN = "kalman"


@dataclass
class TrajectoryPoint:
    """Individual point in a person's trajectory."""
    timestamp: datetime
    position: Coordinate
    camera_id: str
    confidence: float
    speed: Optional[float] = None
    direction: Optional[float] = None  # Angle in radians
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'position': {'x': self.position.x, 'y': self.position.y},
            'camera_id': self.camera_id,
            'confidence': self.confidence,
            'speed': self.speed,
            'direction': self.direction
        }


@dataclass
class PersonTrajectory:
    """Complete trajectory for a person."""
    global_person_id: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_distance: float = 0.0
    average_speed: float = 0.0
    dwell_zones: List[Dict[str, Any]] = field(default_factory=list)
    path_smoothness: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.points:
            self.start_time = min(p.timestamp for p in self.points)
            self.end_time = max(p.timestamp for p in self.points)
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate trajectory metrics."""
        if len(self.points) < 2:
            return
        
        # Calculate total distance and speeds
        distances = []
        times = []
        
        for i in range(1, len(self.points)):
            prev_point = self.points[i-1]
            curr_point = self.points[i]
            
            # Distance calculation
            dx = curr_point.position.x - prev_point.position.x
            dy = curr_point.position.y - prev_point.position.y
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
            
            # Time calculation
            time_diff = (curr_point.timestamp - prev_point.timestamp).total_seconds()
            times.append(time_diff)
            
            # Speed calculation
            if time_diff > 0:
                speed = distance / time_diff
                curr_point.speed = speed
                
                # Direction calculation
                curr_point.direction = np.arctan2(dy, dx)
        
        self.total_distance = sum(distances)
        
        # Average speed (excluding stationary periods)
        valid_speeds = [p.speed for p in self.points if p.speed and p.speed > 0.1]
        self.average_speed = np.mean(valid_speeds) if valid_speeds else 0.0
        
        # Path smoothness (inverse of direction changes)
        direction_changes = []
        for i in range(2, len(self.points)):
            if self.points[i].direction is not None and self.points[i-1].direction is not None:
                angle_diff = abs(self.points[i].direction - self.points[i-1].direction)
                # Normalize to [0, π]
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                direction_changes.append(angle_diff)
        
        if direction_changes:
            avg_direction_change = np.mean(direction_changes)
            self.path_smoothness = max(0.0, 1.0 - (avg_direction_change / np.pi))
        else:
            self.path_smoothness = 1.0
    
    def get_position_at_time(self, timestamp: datetime) -> Optional[Coordinate]:
        """Get interpolated position at specific timestamp."""
        if not self.points or timestamp < self.start_time or timestamp > self.end_time:
            return None
        
        # Find surrounding points
        before_point = None
        after_point = None
        
        for point in self.points:
            if point.timestamp <= timestamp:
                before_point = point
            elif point.timestamp > timestamp and after_point is None:
                after_point = point
                break
        
        if before_point is None:
            return self.points[0].position
        if after_point is None:
            return self.points[-1].position
        
        # Linear interpolation
        time_ratio = (timestamp - before_point.timestamp).total_seconds() / \
                    (after_point.timestamp - before_point.timestamp).total_seconds()
        
        x = before_point.position.x + time_ratio * (after_point.position.x - before_point.position.x)
        y = before_point.position.y + time_ratio * (after_point.position.y - before_point.position.y)
        
        return Coordinate(x=x, y=y)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'global_person_id': self.global_person_id,
            'points': [point.to_dict() for point in self.points],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_distance': self.total_distance,
            'average_speed': self.average_speed,
            'dwell_zones': self.dwell_zones,
            'path_smoothness': self.path_smoothness
        }


@dataclass
class HeatmapData:
    """Heatmap data for occupancy visualization."""
    width: int
    height: int
    resolution: float  # pixels per meter
    data: np.ndarray
    extent: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    
    def to_image(self, colormap: str = 'hot') -> np.ndarray:
        """Convert heatmap data to image."""
        # Normalize data to 0-255
        normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)
        
        # Apply colormap
        colormap_func = plt.get_cmap(colormap)
        colored = colormap_func(normalized / 255.0)
        
        # Convert to RGB
        image = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return image


@dataclass
class PathCluster:
    """Cluster of similar movement paths."""
    cluster_id: int
    trajectories: List[PersonTrajectory]
    center_path: List[Coordinate]
    similarity_score: float
    common_pattern: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'trajectory_count': len(self.trajectories),
            'person_ids': [t.global_person_id for t in self.trajectories],
            'center_path': [{'x': c.x, 'y': c.y} for c in self.center_path],
            'similarity_score': self.similarity_score,
            'common_pattern': self.common_pattern
        }


class MovementPathVisualizationService:
    """Comprehensive service for movement path visualization and analysis."""
    
    def __init__(
        self,
        historical_data_service: HistoricalDataService,
        temporal_query_engine: TemporalQueryEngine,
        output_path: Optional[str] = None
    ):
        self.historical_service = historical_data_service
        self.query_engine = temporal_query_engine
        
        # Output configuration
        self.output_path = Path(output_path or "data/visualizations")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Trajectory cache
        self.trajectory_cache: Dict[str, PersonTrajectory] = {}
        self.heatmap_cache: Dict[str, HeatmapData] = {}
        
        # Visualization settings
        self.default_resolution = 10.0  # pixels per meter
        self.default_smoothing_factor = 0.1
        self.min_trajectory_points = 3
        
        # Performance metrics
        self.processing_stats = {
            'trajectories_processed': 0,
            'heatmaps_generated': 0,
            'paths_clustered': 0,
            'avg_processing_time_ms': 0.0
        }
        
        logger.info("MovementPathVisualizationService initialized")
    
    # --- Trajectory Reconstruction ---
    
    async def reconstruct_person_trajectory(
        self,
        global_person_id: str,
        time_range: TimeRange,
        environment_id: Optional[str] = None,
        interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR
    ) -> PersonTrajectory:
        """Reconstruct complete trajectory for a specific person."""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{global_person_id}_{time_range.start_time.isoformat()}_{time_range.end_time.isoformat()}"
            if cache_key in self.trajectory_cache:
                return self.trajectory_cache[cache_key]
            
            # Query historical data
            data_points = await self.historical_service.get_person_trajectory(
                global_person_id, time_range, environment_id
            )
            
            if len(data_points) < self.min_trajectory_points:
                logger.warning(f"Insufficient data points for trajectory reconstruction: {len(data_points)}")
                return PersonTrajectory(global_person_id=global_person_id)
            
            # Convert to trajectory points
            trajectory_points = []
            for data_point in data_points:
                if data_point.coordinates:
                    point = TrajectoryPoint(
                        timestamp=data_point.timestamp,
                        position=data_point.coordinates,
                        camera_id=data_point.camera_id,
                        confidence=data_point.detection.confidence
                    )
                    trajectory_points.append(point)
            
            # Sort by timestamp
            trajectory_points.sort(key=lambda p: p.timestamp)
            
            # Apply interpolation and smoothing
            if interpolation_method != InterpolationMethod.LINEAR:
                trajectory_points = await self._smooth_trajectory(trajectory_points, interpolation_method)
            
            # Create trajectory
            trajectory = PersonTrajectory(
                global_person_id=global_person_id,
                points=trajectory_points
            )
            
            # Detect dwell zones
            trajectory.dwell_zones = await self._detect_dwell_zones(trajectory_points)
            
            # Cache result
            self.trajectory_cache[cache_key] = trajectory
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics('trajectory_reconstruction', processing_time)
            
            logger.info(f"Reconstructed trajectory for {global_person_id}: {len(trajectory_points)} points")
            return trajectory
            
        except Exception as e:
            logger.error(f"Error reconstructing trajectory: {e}")
            raise
    
    async def _smooth_trajectory(
        self,
        points: List[TrajectoryPoint],
        method: InterpolationMethod
    ) -> List[TrajectoryPoint]:
        """Apply smoothing to trajectory points."""
        try:
            if len(points) < 3:
                return points
            
            # Extract coordinates and timestamps
            timestamps = [p.timestamp.timestamp() for p in points]
            x_coords = [p.position.x for p in points]
            y_coords = [p.position.y for p in points]
            
            # Apply smoothing based on method
            if method == InterpolationMethod.CUBIC:
                # Cubic interpolation
                f_x = interp1d(timestamps, x_coords, kind='cubic', fill_value='extrapolate')
                f_y = interp1d(timestamps, y_coords, kind='cubic', fill_value='extrapolate')
                
                # Generate new points
                new_timestamps = np.linspace(timestamps[0], timestamps[-1], len(points))
                new_x = f_x(new_timestamps)
                new_y = f_y(new_timestamps)
                
            elif method == InterpolationMethod.SPLINE:
                # Spline interpolation
                tck_x = splrep(timestamps, x_coords, s=self.default_smoothing_factor)
                tck_y = splrep(timestamps, y_coords, s=self.default_smoothing_factor)
                
                new_timestamps = np.linspace(timestamps[0], timestamps[-1], len(points))
                new_x = splev(new_timestamps, tck_x)
                new_y = splev(new_timestamps, tck_y)
                
            elif method == InterpolationMethod.KALMAN:
                # Simple Kalman-like filtering
                new_x, new_y = await self._kalman_filter(x_coords, y_coords)
                new_timestamps = timestamps
                
            else:
                return points
            
            # Create smoothed points
            smoothed_points = []
            for i, (ts, x, y) in enumerate(zip(new_timestamps, new_x, new_y)):
                original_point = points[min(i, len(points) - 1)]
                smoothed_point = TrajectoryPoint(
                    timestamp=datetime.fromtimestamp(ts),
                    position=Coordinate(x=float(x), y=float(y)),
                    camera_id=original_point.camera_id,
                    confidence=original_point.confidence
                )
                smoothed_points.append(smoothed_point)
            
            return smoothed_points
            
        except Exception as e:
            logger.error(f"Error smoothing trajectory: {e}")
            return points
    
    async def _kalman_filter(self, x_coords: List[float], y_coords: List[float]) -> Tuple[List[float], List[float]]:
        """Apply simple Kalman-like filtering to coordinates."""
        try:
            if len(x_coords) < 3:
                return x_coords, y_coords
            
            # Simple exponential smoothing (Kalman-like)
            alpha = 0.3  # Smoothing factor
            
            filtered_x = [x_coords[0]]
            filtered_y = [y_coords[0]]
            
            for i in range(1, len(x_coords)):
                # Exponential smoothing
                smooth_x = alpha * x_coords[i] + (1 - alpha) * filtered_x[-1]
                smooth_y = alpha * y_coords[i] + (1 - alpha) * filtered_y[-1]
                
                filtered_x.append(smooth_x)
                filtered_y.append(smooth_y)
            
            return filtered_x, filtered_y
            
        except Exception as e:
            logger.error(f"Error in Kalman filtering: {e}")
            return x_coords, y_coords
    
    async def _detect_dwell_zones(self, points: List[TrajectoryPoint]) -> List[Dict[str, Any]]:
        """Detect areas where person spent significant time (dwell zones)."""
        try:
            if len(points) < 5:
                return []
            
            dwell_zones = []
            speed_threshold = 0.2  # meters per second
            min_dwell_time = 10.0  # seconds
            
            current_zone_start = None
            zone_positions = []
            
            for i, point in enumerate(points):
                if point.speed is not None and point.speed < speed_threshold:
                    if current_zone_start is None:
                        current_zone_start = point.timestamp
                        zone_positions = []
                    
                    zone_positions.append(point.position)
                else:
                    if current_zone_start is not None:
                        dwell_time = (point.timestamp - current_zone_start).total_seconds()
                        
                        if dwell_time >= min_dwell_time:
                            # Calculate center of dwell zone
                            center_x = np.mean([pos.x for pos in zone_positions])
                            center_y = np.mean([pos.y for pos in zone_positions])
                            
                            # Calculate radius
                            distances = [np.sqrt((pos.x - center_x)**2 + (pos.y - center_y)**2) 
                                       for pos in zone_positions]
                            radius = np.mean(distances)
                            
                            dwell_zone = {
                                'center': {'x': center_x, 'y': center_y},
                                'radius': radius,
                                'start_time': current_zone_start.isoformat(),
                                'duration_seconds': dwell_time,
                                'point_count': len(zone_positions)
                            }
                            
                            dwell_zones.append(dwell_zone)
                        
                        current_zone_start = None
                        zone_positions = []
            
            return dwell_zones
            
        except Exception as e:
            logger.error(f"Error detecting dwell zones: {e}")
            return []
    
    # --- Multi-Person Path Analysis ---
    
    async def analyze_multiple_trajectories(
        self,
        time_range: TimeRange,
        environment_id: Optional[str] = None,
        person_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze multiple person trajectories for patterns and interactions."""
        try:
            start_time = time.time()
            
            # Query for person trajectories
            if person_ids:
                trajectories = []
                for person_id in person_ids:
                    trajectory = await self.reconstruct_person_trajectory(
                        person_id, time_range, environment_id
                    )
                    if trajectory.points:
                        trajectories.append(trajectory)
            else:
                # Get all persons in time range
                query_filter = HistoricalQueryFilter(
                    time_range=time_range,
                    environment_id=environment_id
                )
                
                data_points = await self.historical_service.query_historical_data(query_filter)
                
                # Group by person
                person_data = defaultdict(list)
                for point in data_points:
                    person_data[point.global_person_id].append(point)
                
                # Build trajectories
                trajectories = []
                for person_id, points in person_data.items():
                    if len(points) >= self.min_trajectory_points:
                        trajectory = await self.reconstruct_person_trajectory(
                            person_id, time_range, environment_id
                        )
                        if trajectory.points:
                            trajectories.append(trajectory)
            
            # Analyze trajectories
            analysis = {
                'trajectory_count': len(trajectories),
                'time_range': {
                    'start': time_range.start_time.isoformat(),
                    'end': time_range.end_time.isoformat(),
                    'duration_hours': time_range.duration_hours
                },
                'movement_statistics': await self._calculate_movement_statistics(trajectories),
                'path_intersections': await self._find_path_intersections(trajectories),
                'common_routes': await self._identify_common_routes(trajectories),
                'temporal_patterns': await self._analyze_temporal_patterns(trajectories),
                'spatial_clustering': await self._cluster_trajectories(trajectories)
            }
            
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics('multi_trajectory_analysis', processing_time)
            
            logger.info(f"Analyzed {len(trajectories)} trajectories in {processing_time:.1f}ms")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing multiple trajectories: {e}")
            raise
    
    async def _calculate_movement_statistics(self, trajectories: List[PersonTrajectory]) -> Dict[str, Any]:
        """Calculate comprehensive movement statistics."""
        try:
            if not trajectories:
                return {}
            
            # Basic statistics
            distances = [t.total_distance for t in trajectories if t.total_distance > 0]
            speeds = [t.average_speed for t in trajectories if t.average_speed > 0]
            smoothness = [t.path_smoothness for t in trajectories]
            
            # Duration statistics
            durations = []
            for t in trajectories:
                if t.start_time and t.end_time:
                    duration = (t.end_time - t.start_time).total_seconds()
                    durations.append(duration)
            
            return {
                'distance_statistics': {
                    'mean': np.mean(distances) if distances else 0.0,
                    'std': np.std(distances) if distances else 0.0,
                    'min': min(distances) if distances else 0.0,
                    'max': max(distances) if distances else 0.0,
                    'median': np.median(distances) if distances else 0.0
                },
                'speed_statistics': {
                    'mean': np.mean(speeds) if speeds else 0.0,
                    'std': np.std(speeds) if speeds else 0.0,
                    'min': min(speeds) if speeds else 0.0,
                    'max': max(speeds) if speeds else 0.0,
                    'median': np.median(speeds) if speeds else 0.0
                },
                'duration_statistics': {
                    'mean': np.mean(durations) if durations else 0.0,
                    'std': np.std(durations) if durations else 0.0,
                    'min': min(durations) if durations else 0.0,
                    'max': max(durations) if durations else 0.0,
                    'median': np.median(durations) if durations else 0.0
                },
                'path_smoothness': {
                    'mean': np.mean(smoothness) if smoothness else 0.0,
                    'std': np.std(smoothness) if smoothness else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating movement statistics: {e}")
            return {}
    
    async def _find_path_intersections(self, trajectories: List[PersonTrajectory]) -> List[Dict[str, Any]]:
        """Find intersections between different person paths."""
        try:
            intersections = []
            intersection_threshold = 2.0  # meters
            
            for i, traj1 in enumerate(trajectories):
                for j, traj2 in enumerate(trajectories[i+1:], i+1):
                    # Find spatial intersections
                    for point1 in traj1.points:
                        for point2 in traj2.points:
                            distance = np.sqrt(
                                (point1.position.x - point2.position.x)**2 + 
                                (point1.position.y - point2.position.y)**2
                            )
                            
                            if distance <= intersection_threshold:
                                # Check temporal proximity (within 30 seconds)
                                time_diff = abs((point1.timestamp - point2.timestamp).total_seconds())
                                
                                intersection = {
                                    'person_1': traj1.global_person_id,
                                    'person_2': traj2.global_person_id,
                                    'location': {
                                        'x': (point1.position.x + point2.position.x) / 2,
                                        'y': (point1.position.y + point2.position.y) / 2
                                    },
                                    'timestamp_1': point1.timestamp.isoformat(),
                                    'timestamp_2': point2.timestamp.isoformat(),
                                    'time_difference_seconds': time_diff,
                                    'spatial_distance': distance,
                                    'intersection_type': 'near_encounter' if time_diff <= 30 else 'path_crossing'
                                }
                                
                                intersections.append(intersection)
            
            return intersections[:100]  # Limit to avoid memory issues
            
        except Exception as e:
            logger.error(f"Error finding path intersections: {e}")
            return []
    
    async def _identify_common_routes(self, trajectories: List[PersonTrajectory]) -> List[Dict[str, Any]]:
        """Identify common movement routes between trajectories."""
        try:
            # Simplified route identification based on start/end points
            routes = defaultdict(list)
            route_threshold = 5.0  # meters
            
            for trajectory in trajectories:
                if len(trajectory.points) < 2:
                    continue
                
                start_point = trajectory.points[0].position
                end_point = trajectory.points[-1].position
                
                # Create route signature
                route_key = f"{int(start_point.x/route_threshold)}_{int(start_point.y/route_threshold)}_" \
                           f"{int(end_point.x/route_threshold)}_{int(end_point.y/route_threshold)}"
                
                routes[route_key].append(trajectory)
            
            # Filter routes with multiple trajectories
            common_routes = []
            for route_key, route_trajectories in routes.items():
                if len(route_trajectories) >= 2:
                    # Calculate average route
                    start_positions = [t.points[0].position for t in route_trajectories]
                    end_positions = [t.points[-1].position for t in route_trajectories]
                    
                    avg_start_x = np.mean([p.x for p in start_positions])
                    avg_start_y = np.mean([p.y for p in start_positions])
                    avg_end_x = np.mean([p.x for p in end_positions])
                    avg_end_y = np.mean([p.y for p in end_positions])
                    
                    common_route = {
                        'route_id': route_key,
                        'usage_count': len(route_trajectories),
                        'person_ids': [t.global_person_id for t in route_trajectories],
                        'average_start': {'x': avg_start_x, 'y': avg_start_y},
                        'average_end': {'x': avg_end_x, 'y': avg_end_y},
                        'average_distance': np.mean([t.total_distance for t in route_trajectories]),
                        'average_duration': np.mean([
                            (t.end_time - t.start_time).total_seconds() 
                            for t in route_trajectories 
                            if t.start_time and t.end_time
                        ])
                    }
                    
                    common_routes.append(common_route)
            
            # Sort by usage count
            common_routes.sort(key=lambda r: r['usage_count'], reverse=True)
            
            return common_routes[:20]  # Top 20 common routes
            
        except Exception as e:
            logger.error(f"Error identifying common routes: {e}")
            return []
    
    async def _analyze_temporal_patterns(self, trajectories: List[PersonTrajectory]) -> Dict[str, Any]:
        """Analyze temporal patterns in movement."""
        try:
            hourly_activity = defaultdict(int)
            daily_patterns = defaultdict(list)
            
            for trajectory in trajectories:
                if not trajectory.start_time:
                    continue
                
                # Hourly activity
                hour = trajectory.start_time.hour
                hourly_activity[hour] += 1
                
                # Daily patterns
                day_of_week = trajectory.start_time.weekday()
                daily_patterns[day_of_week].append({
                    'start_hour': trajectory.start_time.hour,
                    'duration': (trajectory.end_time - trajectory.start_time).total_seconds() 
                              if trajectory.end_time else 0,
                    'distance': trajectory.total_distance
                })
            
            return {
                'hourly_activity': dict(hourly_activity),
                'peak_hour': max(hourly_activity, key=hourly_activity.get) if hourly_activity else 0,
                'daily_patterns': {
                    day: {
                        'activity_count': len(patterns),
                        'avg_start_hour': np.mean([p['start_hour'] for p in patterns]),
                        'avg_duration': np.mean([p['duration'] for p in patterns]),
                        'avg_distance': np.mean([p['distance'] for p in patterns])
                    }
                    for day, patterns in daily_patterns.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {}
    
    async def _cluster_trajectories(self, trajectories: List[PersonTrajectory]) -> List[PathCluster]:
        """Cluster trajectories based on spatial similarity."""
        try:
            if len(trajectories) < 2:
                return []
            
            # Create feature vectors for trajectories (simplified)
            features = []
            valid_trajectories = []
            
            for trajectory in trajectories:
                if len(trajectory.points) >= 3:
                    # Use start point, end point, and path length as features
                    start = trajectory.points[0].position
                    end = trajectory.points[-1].position
                    
                    feature_vector = [
                        start.x, start.y,
                        end.x, end.y,
                        trajectory.total_distance,
                        trajectory.average_speed
                    ]
                    
                    features.append(feature_vector)
                    valid_trajectories.append(trajectory)
            
            if len(features) < 2:
                return []
            
            # Apply DBSCAN clustering
            features_array = np.array(features)
            clustering = DBSCAN(eps=10.0, min_samples=2).fit(features_array)
            
            # Group trajectories by cluster
            clusters = defaultdict(list)
            for i, cluster_id in enumerate(clustering.labels_):
                if cluster_id != -1:  # Ignore noise points
                    clusters[cluster_id].append(valid_trajectories[i])
            
            # Create PathCluster objects
            path_clusters = []
            for cluster_id, cluster_trajectories in clusters.items():
                if len(cluster_trajectories) >= 2:
                    # Calculate center path (simplified)
                    center_path = await self._calculate_center_path(cluster_trajectories)
                    
                    # Calculate similarity score (simplified)
                    similarity_score = 1.0 / (1.0 + np.std([t.total_distance for t in cluster_trajectories]))
                    
                    common_pattern = {
                        'avg_distance': np.mean([t.total_distance for t in cluster_trajectories]),
                        'avg_speed': np.mean([t.average_speed for t in cluster_trajectories]),
                        'avg_smoothness': np.mean([t.path_smoothness for t in cluster_trajectories])
                    }
                    
                    cluster = PathCluster(
                        cluster_id=cluster_id,
                        trajectories=cluster_trajectories,
                        center_path=center_path,
                        similarity_score=similarity_score,
                        common_pattern=common_pattern
                    )
                    
                    path_clusters.append(cluster)
            
            return path_clusters
            
        except Exception as e:
            logger.error(f"Error clustering trajectories: {e}")
            return []
    
    async def _calculate_center_path(self, trajectories: List[PersonTrajectory]) -> List[Coordinate]:
        """Calculate the center path for a cluster of trajectories."""
        try:
            if not trajectories:
                return []
            
            # Find common number of points (use minimum)
            min_points = min(len(t.points) for t in trajectories)
            
            if min_points < 2:
                return []
            
            # Interpolate all trajectories to have the same number of points
            center_points = []
            
            for i in range(min_points):
                x_coords = []
                y_coords = []
                
                for trajectory in trajectories:
                    if i < len(trajectory.points):
                        point = trajectory.points[i]
                        x_coords.append(point.position.x)
                        y_coords.append(point.position.y)
                
                # Calculate centroid
                if x_coords and y_coords:
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    center_points.append(Coordinate(x=center_x, y=center_y))
            
            return center_points
            
        except Exception as e:
            logger.error(f"Error calculating center path: {e}")
            return []
    
    # --- Heatmap Generation ---
    
    async def generate_occupancy_heatmap(
        self,
        time_range: TimeRange,
        environment_id: Optional[str] = None,
        resolution: float = 10.0,
        extent: Optional[Tuple[float, float, float, float]] = None
    ) -> HeatmapData:
        """Generate occupancy heatmap for the given time range."""
        try:
            start_time = time.time()
            
            # Check cache
            cache_key = f"heatmap_{time_range.start_time.isoformat()}_{time_range.end_time.isoformat()}_{resolution}"
            if cache_key in self.heatmap_cache:
                return self.heatmap_cache[cache_key]
            
            # Query historical data
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id
            )
            
            data_points = await self.historical_service.query_historical_data(query_filter)
            
            if not data_points:
                logger.warning("No data points found for heatmap generation")
                return HeatmapData(
                    width=100, height=100, resolution=resolution,
                    data=np.zeros((100, 100)),
                    extent=(0, 100, 0, 100)
                )
            
            # Extract coordinates
            coordinates = []
            for point in data_points:
                if point.coordinates:
                    coordinates.append((point.coordinates.x, point.coordinates.y))
            
            if not coordinates:
                logger.warning("No valid coordinates found for heatmap generation")
                return HeatmapData(
                    width=100, height=100, resolution=resolution,
                    data=np.zeros((100, 100)),
                    extent=(0, 100, 0, 100)
                )
            
            # Calculate extent if not provided
            if extent is None:
                x_coords = [c[0] for c in coordinates]
                y_coords = [c[1] for c in coordinates]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # Add padding
                padding = 5.0
                extent = (min_x - padding, max_x + padding, min_y - padding, max_y + padding)
            
            # Calculate grid dimensions
            width = int((extent[1] - extent[0]) * resolution)
            height = int((extent[3] - extent[2]) * resolution)
            
            # Create heatmap grid
            heatmap_data = np.zeros((height, width))
            
            # Fill heatmap
            for x, y in coordinates:
                # Convert to grid coordinates
                grid_x = int((x - extent[0]) * resolution)
                grid_y = int((y - extent[2]) * resolution)
                
                # Check bounds
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    heatmap_data[grid_y, grid_x] += 1
            
            # Apply Gaussian smoothing
            from scipy.ndimage import gaussian_filter
            heatmap_data = gaussian_filter(heatmap_data, sigma=2.0)
            
            heatmap = HeatmapData(
                width=width,
                height=height,
                resolution=resolution,
                data=heatmap_data,
                extent=extent
            )
            
            # Cache result
            self.heatmap_cache[cache_key] = heatmap
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics('heatmap_generation', processing_time)
            
            logger.info(f"Generated heatmap ({width}x{height}) in {processing_time:.1f}ms")
            return heatmap
            
        except Exception as e:
            logger.error(f"Error generating occupancy heatmap: {e}")
            raise
    
    # --- Visualization Export ---
    
    async def export_trajectory_visualization(
        self,
        trajectory: PersonTrajectory,
        output_format: str = "png",
        mode: PathVisualizationMode = PathVisualizationMode.LINE,
        background_image: Optional[str] = None
    ) -> str:
        """Export trajectory visualization to file."""
        try:
            if not trajectory.points:
                raise ValueError("Trajectory has no points to visualize")
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Extract coordinates
            x_coords = [p.position.x for p in trajectory.points]
            y_coords = [p.position.y for p in trajectory.points]
            
            # Plot based on mode
            if mode == PathVisualizationMode.LINE:
                plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
                plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
                plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
                
            elif mode == PathVisualizationMode.DOTS:
                colors = plt.cm.viridis(np.linspace(0, 1, len(x_coords)))
                plt.scatter(x_coords, y_coords, c=colors, s=20, alpha=0.7)
                
            # Add dwell zones
            for dwell_zone in trajectory.dwell_zones:
                circle = plt.Circle(
                    (dwell_zone['center']['x'], dwell_zone['center']['y']),
                    dwell_zone['radius'],
                    fill=False,
                    color='red',
                    linestyle='--',
                    alpha=0.5
                )
                plt.gca().add_patch(circle)
            
            # Formatting
            plt.title(f"Trajectory for Person {trajectory.global_person_id}")
            plt.xlabel("X Coordinate (m)")
            plt.ylabel("Y Coordinate (m)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.axis('equal')
            
            # Save file
            filename = f"trajectory_{trajectory.global_person_id}_{int(time.time())}.{output_format}"
            filepath = self.output_path / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Exported trajectory visualization: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting trajectory visualization: {e}")
            raise
    
    async def export_heatmap_visualization(
        self,
        heatmap: HeatmapData,
        output_format: str = "png",
        colormap: str = "hot"
    ) -> str:
        """Export heatmap visualization to file."""
        try:
            # Convert to image
            image = heatmap.to_image(colormap)
            
            # Create PIL image
            pil_image = Image.fromarray(image)
            
            # Save file
            filename = f"heatmap_{int(time.time())}.{output_format}"
            filepath = self.output_path / filename
            pil_image.save(filepath)
            
            logger.info(f"Exported heatmap visualization: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting heatmap visualization: {e}")
            raise
    
    # --- Utility Methods ---
    
    def _update_processing_metrics(self, operation: str, processing_time_ms: float):
        """Update processing performance metrics."""
        if operation == 'trajectory_reconstruction':
            self.processing_stats['trajectories_processed'] += 1
        elif operation == 'heatmap_generation':
            self.processing_stats['heatmaps_generated'] += 1
        elif operation == 'path_clustering':
            self.processing_stats['paths_clustered'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['avg_processing_time_ms']
        total_operations = sum([
            self.processing_stats['trajectories_processed'],
            self.processing_stats['heatmaps_generated'],
            self.processing_stats['paths_clustered']
        ])
        
        if total_operations > 0:
            self.processing_stats['avg_processing_time_ms'] = (
                (current_avg * (total_operations - 1) + processing_time_ms) / total_operations
            )
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        self.trajectory_cache.clear()
        self.heatmap_cache.clear()
        logger.info("Cleared visualization caches")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'MovementPathVisualizationService',
            'output_path': str(self.output_path),
            'cache_status': {
                'trajectory_cache_size': len(self.trajectory_cache),
                'heatmap_cache_size': len(self.heatmap_cache)
            },
            'processing_statistics': self.processing_stats.copy(),
            'configuration': {
                'default_resolution': self.default_resolution,
                'default_smoothing_factor': self.default_smoothing_factor,
                'min_trajectory_points': self.min_trajectory_points
            }
        }


# Global service instance
_movement_path_visualization_service: Optional[MovementPathVisualizationService] = None


def get_movement_path_visualization_service() -> Optional[MovementPathVisualizationService]:
    """Get the global movement path visualization service instance."""
    return _movement_path_visualization_service


def initialize_movement_path_visualization_service(
    historical_data_service: HistoricalDataService,
    temporal_query_engine: TemporalQueryEngine,
    output_path: Optional[str] = None
) -> MovementPathVisualizationService:
    """Initialize the global movement path visualization service."""
    global _movement_path_visualization_service
    if _movement_path_visualization_service is None:
        _movement_path_visualization_service = MovementPathVisualizationService(
            historical_data_service, temporal_query_engine, output_path
        )
    return _movement_path_visualization_service