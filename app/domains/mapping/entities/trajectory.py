"""
Trajectory entity for person movement tracking.

Contains trajectory objects for tracking person paths across time and space.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import math

from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem
from app.shared.types import CameraID

@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    
    coordinate: Coordinate
    velocity: Optional[Tuple[float, float]] = None
    acceleration: Optional[Tuple[float, float]] = None
    
    # Metadata
    camera_id: Optional[CameraID] = None
    detection_id: Optional[str] = None
    interpolated: bool = False
    
    @property
    def speed(self) -> Optional[float]:
        """Get speed magnitude."""
        if not self.velocity:
            return None
        
        vx, vy = self.velocity
        return math.sqrt(vx*vx + vy*vy)
    
    @property
    def direction(self) -> Optional[float]:
        """Get direction angle in radians."""
        if not self.velocity:
            return None
        
        vx, vy = self.velocity
        return math.atan2(vy, vx)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "coordinate": self.coordinate.to_dict(),
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "camera_id": self.camera_id,
            "detection_id": self.detection_id,
            "interpolated": self.interpolated,
            "speed": self.speed,
            "direction": self.direction
        }

@dataclass
class Trajectory:
    """Person trajectory object."""
    
    global_id: str
    path_points: List[TrajectoryPoint] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    cameras_traversed: List[CameraID] = field(default_factory=list)
    
    # Quality metrics
    smoothness_score: float = 0.0
    completeness_score: float = 0.0
    confidence_score: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate trajectory."""
        if not self.global_id:
            raise ValueError("Global ID cannot be empty")
        
        if not 0 <= self.smoothness_score <= 1:
            raise ValueError("Smoothness score must be between 0 and 1")
        
        if not 0 <= self.completeness_score <= 1:
            raise ValueError("Completeness score must be between 0 and 1")
        
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
    
    def add_point(self, point: TrajectoryPoint) -> 'Trajectory':
        """Add a point to the trajectory."""
        new_points = self.path_points.copy()
        new_points.append(point)
        
        # Sort by timestamp
        new_points.sort(key=lambda p: p.coordinate.timestamp)
        
        # Update timing
        start_time = self.start_time or point.coordinate.timestamp
        end_time = point.coordinate.timestamp
        
        # Update cameras traversed
        new_cameras = self.cameras_traversed.copy()
        if point.camera_id and point.camera_id not in new_cameras:
            new_cameras.append(point.camera_id)
        
        return Trajectory(
            global_id=self.global_id,
            path_points=new_points,
            start_time=start_time,
            end_time=end_time,
            cameras_traversed=new_cameras,
            smoothness_score=self.smoothness_score,
            completeness_score=self.completeness_score,
            confidence_score=self.confidence_score,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def add_points(self, points: List[TrajectoryPoint]) -> 'Trajectory':
        """Add multiple points to the trajectory."""
        result = self
        for point in points:
            result = result.add_point(point)
        return result
    
    def get_point_at_time(self, timestamp: datetime) -> Optional[TrajectoryPoint]:
        """Get trajectory point at specific time."""
        for point in self.path_points:
            if point.coordinate.timestamp == timestamp:
                return point
        return None
    
    def get_points_in_time_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[TrajectoryPoint]:
        """Get points within time range."""
        return [
            point for point in self.path_points
            if start_time <= point.coordinate.timestamp <= end_time
        ]
    
    def get_points_for_camera(self, camera_id: CameraID) -> List[TrajectoryPoint]:
        """Get points from specific camera."""
        return [
            point for point in self.path_points
            if point.camera_id == camera_id
        ]
    
    @property
    def point_count(self) -> int:
        """Get number of points in trajectory."""
        return len(self.path_points)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get trajectory duration."""
        if not self.start_time or not self.end_time:
            return None
        
        return self.end_time - self.start_time
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get trajectory duration in seconds."""
        duration = self.duration
        return duration.total_seconds() if duration else None
    
    @property
    def total_distance(self) -> float:
        """Calculate total distance traveled."""
        if len(self.path_points) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self.path_points)):
            prev_point = self.path_points[i-1]
            curr_point = self.path_points[i]
            
            try:
                distance = prev_point.coordinate.distance_to(curr_point.coordinate)
                total += distance
            except ValueError:
                # Skip if coordinates are in different systems
                continue
        
        return total
    
    @property
    def average_speed(self) -> Optional[float]:
        """Calculate average speed."""
        duration = self.duration_seconds
        if not duration or duration == 0:
            return None
        
        return self.total_distance / duration
    
    @property
    def max_speed(self) -> Optional[float]:
        """Get maximum speed from trajectory points."""
        speeds = [point.speed for point in self.path_points if point.speed is not None]
        return max(speeds) if speeds else None
    
    @property
    def camera_count(self) -> int:
        """Get number of cameras this trajectory traversed."""
        return len(self.cameras_traversed)
    
    @property
    def interpolated_point_count(self) -> int:
        """Get number of interpolated points."""
        return sum(1 for point in self.path_points if point.interpolated)
    
    def calculate_smoothness(self) -> float:
        """Calculate trajectory smoothness score."""
        if len(self.path_points) < 3:
            return 1.0
        
        # Calculate direction changes
        direction_changes = []
        
        for i in range(1, len(self.path_points) - 1):
            prev_point = self.path_points[i-1]
            curr_point = self.path_points[i]
            next_point = self.path_points[i+1]
            
            # Calculate direction vectors
            dir1 = (
                curr_point.coordinate.x - prev_point.coordinate.x,
                curr_point.coordinate.y - prev_point.coordinate.y
            )
            dir2 = (
                next_point.coordinate.x - curr_point.coordinate.x,
                next_point.coordinate.y - curr_point.coordinate.y
            )
            
            # Calculate angle change
            if dir1[0] != 0 or dir1[1] != 0:
                angle1 = math.atan2(dir1[1], dir1[0])
                angle2 = math.atan2(dir2[1], dir2[0])
                
                angle_change = abs(angle2 - angle1)
                if angle_change > math.pi:
                    angle_change = 2 * math.pi - angle_change
                
                direction_changes.append(angle_change)
        
        if not direction_changes:
            return 1.0
        
        # Calculate smoothness (lower angle changes = higher smoothness)
        avg_angle_change = sum(direction_changes) / len(direction_changes)
        smoothness = 1.0 - (avg_angle_change / math.pi)
        
        return max(0.0, min(1.0, smoothness))
    
    def calculate_completeness(self, expected_fps: float = 30.0) -> float:
        """Calculate trajectory completeness score."""
        if not self.duration_seconds:
            return 0.0
        
        expected_points = int(self.duration_seconds * expected_fps)
        if expected_points == 0:
            return 1.0
        
        actual_points = len(self.path_points)
        completeness = min(1.0, actual_points / expected_points)
        
        return completeness
    
    def interpolate_missing_points(self, target_fps: float = 30.0) -> 'Trajectory':
        """Interpolate missing points for smoother trajectory."""
        if len(self.path_points) < 2:
            return self
        
        interpolated_points = []
        
        for i in range(len(self.path_points) - 1):
            current_point = self.path_points[i]
            next_point = self.path_points[i + 1]
            
            # Add current point
            interpolated_points.append(current_point)
            
            # Calculate time gap
            time_gap = (next_point.coordinate.timestamp - current_point.coordinate.timestamp).total_seconds()
            
            # Determine if interpolation is needed
            expected_points = int(time_gap * target_fps)
            if expected_points > 1:
                # Interpolate points
                for j in range(1, expected_points):
                    t = j / expected_points
                    
                    # Linear interpolation
                    interp_x = current_point.coordinate.x + t * (next_point.coordinate.x - current_point.coordinate.x)
                    interp_y = current_point.coordinate.y + t * (next_point.coordinate.y - current_point.coordinate.y)
                    
                    interp_time = current_point.coordinate.timestamp + timedelta(seconds=t * time_gap)
                    
                    interp_coord = Coordinate(
                        x=interp_x,
                        y=interp_y,
                        coordinate_system=current_point.coordinate.coordinate_system,
                        timestamp=interp_time,
                        confidence=min(current_point.coordinate.confidence, next_point.coordinate.confidence)
                    )
                    
                    interp_point = TrajectoryPoint(
                        coordinate=interp_coord,
                        interpolated=True
                    )
                    
                    interpolated_points.append(interp_point)
        
        # Add last point
        if self.path_points:
            interpolated_points.append(self.path_points[-1])
        
        return Trajectory(
            global_id=self.global_id,
            path_points=interpolated_points,
            start_time=self.start_time,
            end_time=self.end_time,
            cameras_traversed=self.cameras_traversed.copy(),
            smoothness_score=self.smoothness_score,
            completeness_score=self.completeness_score,
            confidence_score=self.confidence_score,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def smooth_trajectory(self, window_size: int = 5) -> 'Trajectory':
        """Apply smoothing filter to trajectory."""
        if len(self.path_points) < window_size:
            return self
        
        smoothed_points = []
        half_window = window_size // 2
        
        for i in range(len(self.path_points)):
            if i < half_window or i >= len(self.path_points) - half_window:
                # Keep original points at boundaries
                smoothed_points.append(self.path_points[i])
            else:
                # Apply smoothing
                start_idx = i - half_window
                end_idx = i + half_window + 1
                
                window_points = self.path_points[start_idx:end_idx]
                
                # Calculate weighted average
                avg_x = sum(p.coordinate.x for p in window_points) / len(window_points)
                avg_y = sum(p.coordinate.y for p in window_points) / len(window_points)
                
                original_point = self.path_points[i]
                smoothed_coord = Coordinate(
                    x=avg_x,
                    y=avg_y,
                    coordinate_system=original_point.coordinate.coordinate_system,
                    timestamp=original_point.coordinate.timestamp,
                    confidence=original_point.coordinate.confidence,
                    camera_id=original_point.coordinate.camera_id,
                    frame_index=original_point.coordinate.frame_index
                )
                
                smoothed_point = TrajectoryPoint(
                    coordinate=smoothed_coord,
                    camera_id=original_point.camera_id,
                    detection_id=original_point.detection_id,
                    interpolated=original_point.interpolated
                )
                
                smoothed_points.append(smoothed_point)
        
        return Trajectory(
            global_id=self.global_id,
            path_points=smoothed_points,
            start_time=self.start_time,
            end_time=self.end_time,
            cameras_traversed=self.cameras_traversed.copy(),
            smoothness_score=self.calculate_smoothness(),
            completeness_score=self.completeness_score,
            confidence_score=self.confidence_score,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def calculate_velocities(self) -> 'Trajectory':
        """Calculate velocities for all trajectory points."""
        if len(self.path_points) < 2:
            return self
        
        updated_points = []
        
        for i in range(len(self.path_points)):
            point = self.path_points[i]
            
            if i == 0:
                # First point: use forward difference
                next_point = self.path_points[i + 1]
                dt = (next_point.coordinate.timestamp - point.coordinate.timestamp).total_seconds()
                
                if dt > 0:
                    vx = (next_point.coordinate.x - point.coordinate.x) / dt
                    vy = (next_point.coordinate.y - point.coordinate.y) / dt
                    velocity = (vx, vy)
                else:
                    velocity = (0.0, 0.0)
                    
            elif i == len(self.path_points) - 1:
                # Last point: use backward difference
                prev_point = self.path_points[i - 1]
                dt = (point.coordinate.timestamp - prev_point.coordinate.timestamp).total_seconds()
                
                if dt > 0:
                    vx = (point.coordinate.x - prev_point.coordinate.x) / dt
                    vy = (point.coordinate.y - prev_point.coordinate.y) / dt
                    velocity = (vx, vy)
                else:
                    velocity = (0.0, 0.0)
                    
            else:
                # Middle points: use central difference
                prev_point = self.path_points[i - 1]
                next_point = self.path_points[i + 1]
                dt = (next_point.coordinate.timestamp - prev_point.coordinate.timestamp).total_seconds()
                
                if dt > 0:
                    vx = (next_point.coordinate.x - prev_point.coordinate.x) / dt
                    vy = (next_point.coordinate.y - prev_point.coordinate.y) / dt
                    velocity = (vx, vy)
                else:
                    velocity = (0.0, 0.0)
            
            updated_point = TrajectoryPoint(
                coordinate=point.coordinate,
                velocity=velocity,
                acceleration=point.acceleration,
                camera_id=point.camera_id,
                detection_id=point.detection_id,
                interpolated=point.interpolated
            )
            
            updated_points.append(updated_point)
        
        return Trajectory(
            global_id=self.global_id,
            path_points=updated_points,
            start_time=self.start_time,
            end_time=self.end_time,
            cameras_traversed=self.cameras_traversed.copy(),
            smoothness_score=self.smoothness_score,
            completeness_score=self.completeness_score,
            confidence_score=self.confidence_score,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "global_id": self.global_id,
            "path_points": [point.to_dict() for point in self.path_points],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "cameras_traversed": self.cameras_traversed,
            "smoothness_score": self.smoothness_score,
            "completeness_score": self.completeness_score,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "point_count": self.point_count,
            "duration_seconds": self.duration_seconds,
            "total_distance": self.total_distance,
            "average_speed": self.average_speed,
            "max_speed": self.max_speed,
            "camera_count": self.camera_count,
            "interpolated_point_count": self.interpolated_point_count
        }