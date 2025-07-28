"""
Trajectory builder service for person trajectory construction and optimization.

Provides business logic for:
- Building trajectories from detection sequences
- Trajectory smoothing and interpolation
- Multi-camera trajectory fusion
- Trajectory quality assessment
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem
from app.domains.mapping.entities.trajectory import Trajectory, TrajectoryPoint
from app.domains.detection.entities.detection import Detection
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.models.coordinate_transformer import CoordinateTransformer, TransformationResult
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class TrajectoryBuilder:
    """
    Service for building and optimizing person trajectories.
    
    Features:
    - Build trajectories from detection sequences
    - Multi-camera trajectory fusion
    - Trajectory smoothing and interpolation
    - Quality assessment and validation
    - Gap filling and outlier removal
    """
    
    def __init__(
        self,
        coordinate_transformer: CoordinateTransformer,
        min_trajectory_length: int = 3,
        max_gap_duration: float = 2.0,  # seconds
        smoothing_window: int = 5,
        outlier_threshold: float = 2.0  # standard deviations
    ):
        """
        Initialize trajectory builder.
        
        Args:
            coordinate_transformer: Coordinate transformer for spatial conversions
            min_trajectory_length: Minimum number of points for valid trajectory
            max_gap_duration: Maximum gap duration to interpolate (seconds)
            smoothing_window: Window size for trajectory smoothing
            outlier_threshold: Threshold for outlier detection (std devs)
        """
        self.coordinate_transformer = coordinate_transformer
        self.min_trajectory_length = min_trajectory_length
        self.max_gap_duration = max_gap_duration
        self.smoothing_window = smoothing_window
        self.outlier_threshold = outlier_threshold
        
        # Performance tracking
        self.builder_stats = {
            "trajectories_built": 0,
            "trajectories_smoothed": 0,
            "trajectories_interpolated": 0,
            "outliers_removed": 0,
            "gaps_filled": 0,
            "multi_camera_fusions": 0
        }
        
        logger.info("TrajectoryBuilder initialized")
    
    async def build_trajectory_from_detections(
        self,
        person_identity: PersonIdentity,
        detections: List[Detection],
        target_coordinate_system: CoordinateSystem = CoordinateSystem.MAP
    ) -> Optional[Trajectory]:
        """
        Build trajectory from person detections.
        
        Args:
            person_identity: Person identity information
            detections: List of person detections
            target_coordinate_system: Target coordinate system for trajectory
            
        Returns:
            Built trajectory or None if failed
        """
        try:
            if not detections or len(detections) < self.min_trajectory_length:
                logger.warning(f"Insufficient detections for trajectory: {len(detections)}")
                return None
            
            # Sort detections by timestamp
            sorted_detections = sorted(detections, key=lambda d: d.timestamp)
            
            # Transform detection coordinates to target system
            trajectory_points = []
            
            for detection in sorted_detections:
                # Create coordinate from detection
                image_coord = Coordinate(
                    x=detection.bbox.center_x,
                    y=detection.bbox.center_y,
                    coordinate_system=CoordinateSystem.IMAGE,
                    timestamp=detection.timestamp,
                    camera_id=detection.camera_id,
                    frame_index=detection.frame_index,
                    confidence=detection.confidence
                )
                
                # Transform to target coordinate system
                transformation_result = self.coordinate_transformer.transform_coordinate(
                    image_coord,
                    target_coordinate_system,
                    detection.camera_id
                )
                
                if transformation_result.success and transformation_result.transformed_coordinate:
                    # Create trajectory point
                    trajectory_point = TrajectoryPoint(
                        coordinate=transformation_result.transformed_coordinate,
                        camera_id=detection.camera_id,
                        detection_id=detection.id,
                        interpolated=False,
                        velocity=None,  # Will be calculated later
                        acceleration=None
                    )
                    
                    trajectory_points.append(trajectory_point)
                else:
                    logger.warning(f"Failed to transform detection {detection.id}: {transformation_result.error_message}")
            
            if len(trajectory_points) < self.min_trajectory_length:
                logger.warning(f"Insufficient transformed points for trajectory: {len(trajectory_points)}")
                return None
            
            # Create initial trajectory
            trajectory = Trajectory(
                global_id=person_identity.global_id,
                path_points=trajectory_points,
                start_time=sorted_detections[0].timestamp,
                end_time=sorted_detections[-1].timestamp,
                cameras_traversed=list(set(d.camera_id for d in sorted_detections))
            )
            
            # Calculate initial metrics
            trajectory = self._calculate_trajectory_metrics(trajectory)
            
            self.builder_stats["trajectories_built"] += 1
            
            logger.info(f"Built trajectory for {person_identity.global_id} with {len(trajectory_points)} points")
            return trajectory
            
        except Exception as e:
            logger.error(f"Error building trajectory for {person_identity.global_id}: {e}")
            return None
    
    async def build_multi_camera_trajectory(
        self,
        person_identity: PersonIdentity,
        detections_by_camera: Dict[CameraID, List[Detection]],
        target_coordinate_system: CoordinateSystem = CoordinateSystem.MAP
    ) -> Optional[Trajectory]:
        """
        Build trajectory from multi-camera detections.
        
        Args:
            person_identity: Person identity information
            detections_by_camera: Detections grouped by camera
            target_coordinate_system: Target coordinate system
            
        Returns:
            Fused multi-camera trajectory or None if failed
        """
        try:
            # Build individual trajectories for each camera
            camera_trajectories = []
            
            for camera_id, detections in detections_by_camera.items():
                if len(detections) >= self.min_trajectory_length:
                    trajectory = await self.build_trajectory_from_detections(
                        person_identity,
                        detections,
                        target_coordinate_system
                    )
                    
                    if trajectory:
                        camera_trajectories.append(trajectory)
            
            if not camera_trajectories:
                logger.warning(f"No valid camera trajectories for {person_identity.global_id}")
                return None
            
            # Fuse trajectories into single multi-camera trajectory
            fused_trajectory = self._fuse_camera_trajectories(camera_trajectories)
            
            if fused_trajectory:
                self.builder_stats["multi_camera_fusions"] += 1
                logger.info(f"Fused multi-camera trajectory for {person_identity.global_id}")
            
            return fused_trajectory
            
        except Exception as e:
            logger.error(f"Error building multi-camera trajectory for {person_identity.global_id}: {e}")
            return None
    
    def _fuse_camera_trajectories(self, camera_trajectories: List[Trajectory]) -> Optional[Trajectory]:
        """Fuse multiple camera trajectories into single trajectory."""
        try:
            if not camera_trajectories:
                return None
            
            if len(camera_trajectories) == 1:
                return camera_trajectories[0]
            
            # Combine all trajectory points
            all_points = []
            all_cameras = set()
            
            for trajectory in camera_trajectories:
                all_points.extend(trajectory.path_points)
                all_cameras.update(trajectory.cameras_traversed)
            
            # Sort by timestamp
            all_points.sort(key=lambda p: p.coordinate.timestamp)
            
            # Create fused trajectory
            fused_trajectory = Trajectory(
                global_id=camera_trajectories[0].global_id,
                path_points=all_points,
                start_time=min(t.start_time for t in camera_trajectories),
                end_time=max(t.end_time for t in camera_trajectories),
                cameras_traversed=list(all_cameras)
            )
            
            # Calculate metrics for fused trajectory
            fused_trajectory = self._calculate_trajectory_metrics(fused_trajectory)
            
            return fused_trajectory
            
        except Exception as e:
            logger.error(f"Error fusing camera trajectories: {e}")
            return None
    
    def _calculate_trajectory_metrics(self, trajectory: Trajectory) -> Trajectory:
        """Calculate trajectory quality metrics."""
        try:
            # Calculate velocities
            trajectory_with_velocities = self._calculate_velocities(trajectory)
            
            # Calculate quality scores
            smoothness_score = self._calculate_smoothness_score(trajectory_with_velocities)
            completeness_score = self._calculate_completeness_score(trajectory_with_velocities)
            confidence_score = self._calculate_confidence_score(trajectory_with_velocities)
            
            # Update trajectory with metrics
            updated_trajectory = Trajectory(
                global_id=trajectory.global_id,
                path_points=trajectory_with_velocities.path_points,
                start_time=trajectory.start_time,
                end_time=trajectory.end_time,
                cameras_traversed=trajectory.cameras_traversed,
                smoothness_score=smoothness_score,
                completeness_score=completeness_score,
                confidence_score=confidence_score,
                created_at=trajectory.created_at,
                last_updated=datetime.now(timezone.utc)
            )
            
            return updated_trajectory
            
        except Exception as e:
            logger.error(f"Error calculating trajectory metrics: {e}")
            return trajectory
    
    def _calculate_velocities(self, trajectory: Trajectory) -> Trajectory:
        """Calculate velocities for trajectory points."""
        try:
            if len(trajectory.path_points) < 2:
                return trajectory
            
            updated_points = []
            
            for i, point in enumerate(trajectory.path_points):
                velocity = None
                acceleration = None
                
                if i > 0:
                    prev_point = trajectory.path_points[i-1]
                    
                    # Calculate velocity
                    dx = point.coordinate.x - prev_point.coordinate.x
                    dy = point.coordinate.y - prev_point.coordinate.y
                    dt = (point.coordinate.timestamp - prev_point.coordinate.timestamp).total_seconds()
                    
                    if dt > 0:
                        velocity = np.sqrt(dx**2 + dy**2) / dt
                
                if i > 1:
                    prev_point = trajectory.path_points[i-1]
                    prev_prev_point = trajectory.path_points[i-2]
                    
                    # Calculate acceleration
                    if prev_point.velocity is not None and point.velocity is not None:
                        dv = point.velocity - prev_point.velocity
                        dt = (point.coordinate.timestamp - prev_point.coordinate.timestamp).total_seconds()
                        
                        if dt > 0:
                            acceleration = dv / dt
                
                # Update point with velocity and acceleration
                updated_point = TrajectoryPoint(
                    coordinate=point.coordinate,
                    camera_id=point.camera_id,
                    detection_id=point.detection_id,
                    interpolated=point.interpolated,
                    velocity=velocity,
                    acceleration=acceleration
                )
                
                updated_points.append(updated_point)
            
            # Update trajectory with velocity-calculated points
            return Trajectory(
                global_id=trajectory.global_id,
                path_points=updated_points,
                start_time=trajectory.start_time,
                end_time=trajectory.end_time,
                cameras_traversed=trajectory.cameras_traversed,
                smoothness_score=trajectory.smoothness_score,
                completeness_score=trajectory.completeness_score,
                confidence_score=trajectory.confidence_score,
                created_at=trajectory.created_at,
                last_updated=trajectory.last_updated
            )
            
        except Exception as e:
            logger.error(f"Error calculating velocities: {e}")
            return trajectory
    
    def _calculate_smoothness_score(self, trajectory: Trajectory) -> float:
        """Calculate trajectory smoothness score."""
        try:
            if len(trajectory.path_points) < 3:
                return 1.0
            
            # Calculate velocity variations
            velocities = [p.velocity for p in trajectory.path_points if p.velocity is not None]
            
            if len(velocities) < 2:
                return 1.0
            
            # Calculate velocity variance
            velocity_variance = np.var(velocities)
            
            # Convert to smoothness score (0-1, higher is smoother)
            # Using exponential decay for variance
            smoothness_score = np.exp(-velocity_variance / 10.0)
            
            return max(0.0, min(1.0, smoothness_score))
            
        except Exception as e:
            logger.error(f"Error calculating smoothness score: {e}")
            return 0.5
    
    def _calculate_completeness_score(self, trajectory: Trajectory) -> float:
        """Calculate trajectory completeness score."""
        try:
            if not trajectory.path_points:
                return 0.0
            
            # Calculate time span
            time_span = (trajectory.end_time - trajectory.start_time).total_seconds()
            
            if time_span <= 0:
                return 1.0
            
            # Calculate average detection rate
            detection_rate = len(trajectory.path_points) / time_span
            
            # Expected detection rate (frames per second)
            expected_rate = 30.0  # Assume 30 FPS
            
            # Calculate completeness as ratio of actual to expected
            completeness_score = min(1.0, detection_rate / expected_rate)
            
            return max(0.0, completeness_score)
            
        except Exception as e:
            logger.error(f"Error calculating completeness score: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, trajectory: Trajectory) -> float:
        """Calculate trajectory confidence score."""
        try:
            if not trajectory.path_points:
                return 0.0
            
            # Calculate average coordinate confidence
            confidences = [p.coordinate.confidence for p in trajectory.path_points if p.coordinate.confidence is not None]
            
            if not confidences:
                return 0.5
            
            average_confidence = np.mean(confidences)
            
            # Weight by trajectory length (longer trajectories are more confident)
            length_weight = min(1.0, len(trajectory.path_points) / 10.0)
            
            # Weight by camera coverage (more cameras = more confident)
            camera_weight = min(1.0, len(trajectory.cameras_traversed) / 4.0)
            
            # Combined confidence score
            confidence_score = average_confidence * length_weight * camera_weight
            
            return max(0.0, min(1.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    async def smooth_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Apply smoothing to trajectory."""
        try:
            if len(trajectory.path_points) < self.smoothing_window:
                return trajectory
            
            # Extract coordinates
            x_coords = [p.coordinate.x for p in trajectory.path_points]
            y_coords = [p.coordinate.y for p in trajectory.path_points]
            
            # Apply Gaussian smoothing
            smoothed_x = gaussian_filter1d(x_coords, sigma=1.0)
            smoothed_y = gaussian_filter1d(y_coords, sigma=1.0)
            
            # Create smoothed points
            smoothed_points = []
            
            for i, point in enumerate(trajectory.path_points):
                smoothed_coord = Coordinate(
                    x=float(smoothed_x[i]),
                    y=float(smoothed_y[i]),
                    coordinate_system=point.coordinate.coordinate_system,
                    timestamp=point.coordinate.timestamp,
                    camera_id=point.coordinate.camera_id,
                    frame_index=point.coordinate.frame_index,
                    confidence=point.coordinate.confidence
                )
                
                smoothed_point = TrajectoryPoint(
                    coordinate=smoothed_coord,
                    camera_id=point.camera_id,
                    detection_id=point.detection_id,
                    interpolated=point.interpolated,
                    velocity=point.velocity,
                    acceleration=point.acceleration
                )
                
                smoothed_points.append(smoothed_point)
            
            # Create smoothed trajectory
            smoothed_trajectory = Trajectory(
                global_id=trajectory.global_id,
                path_points=smoothed_points,
                start_time=trajectory.start_time,
                end_time=trajectory.end_time,
                cameras_traversed=trajectory.cameras_traversed,
                smoothness_score=trajectory.smoothness_score,
                completeness_score=trajectory.completeness_score,
                confidence_score=trajectory.confidence_score,
                created_at=trajectory.created_at,
                last_updated=datetime.now(timezone.utc)
            )
            
            # Recalculate metrics
            smoothed_trajectory = self._calculate_trajectory_metrics(smoothed_trajectory)
            
            self.builder_stats["trajectories_smoothed"] += 1
            
            return smoothed_trajectory
            
        except Exception as e:
            logger.error(f"Error smoothing trajectory: {e}")
            return trajectory
    
    async def interpolate_trajectory_gaps(self, trajectory: Trajectory) -> Trajectory:
        """Fill gaps in trajectory through interpolation."""
        try:
            if len(trajectory.path_points) < 2:
                return trajectory
            
            # Find gaps that need interpolation
            gaps = self._find_trajectory_gaps(trajectory)
            
            if not gaps:
                return trajectory
            
            # Interpolate each gap
            interpolated_points = []
            point_index = 0
            
            for gap in gaps:
                # Add points before gap
                while point_index < gap["start_index"]:
                    interpolated_points.append(trajectory.path_points[point_index])
                    point_index += 1
                
                # Interpolate gap
                gap_points = self._interpolate_gap(
                    trajectory.path_points[gap["start_index"]],
                    trajectory.path_points[gap["end_index"]],
                    gap["duration"]
                )
                
                interpolated_points.extend(gap_points)
                
                # Skip to after gap
                point_index = gap["end_index"]
            
            # Add remaining points
            while point_index < len(trajectory.path_points):
                interpolated_points.append(trajectory.path_points[point_index])
                point_index += 1
            
            # Create interpolated trajectory
            interpolated_trajectory = Trajectory(
                global_id=trajectory.global_id,
                path_points=interpolated_points,
                start_time=trajectory.start_time,
                end_time=trajectory.end_time,
                cameras_traversed=trajectory.cameras_traversed,
                smoothness_score=trajectory.smoothness_score,
                completeness_score=trajectory.completeness_score,
                confidence_score=trajectory.confidence_score,
                created_at=trajectory.created_at,
                last_updated=datetime.now(timezone.utc)
            )
            
            # Recalculate metrics
            interpolated_trajectory = self._calculate_trajectory_metrics(interpolated_trajectory)
            
            self.builder_stats["trajectories_interpolated"] += 1
            self.builder_stats["gaps_filled"] += len(gaps)
            
            return interpolated_trajectory
            
        except Exception as e:
            logger.error(f"Error interpolating trajectory gaps: {e}")
            return trajectory
    
    def _find_trajectory_gaps(self, trajectory: Trajectory) -> List[Dict[str, Any]]:
        """Find gaps in trajectory that need interpolation."""
        gaps = []
        
        try:
            for i in range(len(trajectory.path_points) - 1):
                current_point = trajectory.path_points[i]
                next_point = trajectory.path_points[i + 1]
                
                # Calculate time gap
                time_gap = (next_point.coordinate.timestamp - current_point.coordinate.timestamp).total_seconds()
                
                # If gap is longer than threshold, mark for interpolation
                if time_gap > self.max_gap_duration:
                    gaps.append({
                        "start_index": i,
                        "end_index": i + 1,
                        "duration": time_gap
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error finding trajectory gaps: {e}")
            return []
    
    def _interpolate_gap(
        self, 
        start_point: TrajectoryPoint, 
        end_point: TrajectoryPoint, 
        gap_duration: float
    ) -> List[TrajectoryPoint]:
        """Interpolate points between two trajectory points."""
        try:
            # Calculate number of points to interpolate
            # Assume 30 FPS for interpolation
            num_points = int(gap_duration * 30)
            
            # Limit interpolation to reasonable number
            num_points = min(num_points, 100)
            
            if num_points <= 0:
                return []
            
            interpolated_points = []
            
            # Linear interpolation
            for i in range(1, num_points + 1):
                t = i / (num_points + 1)
                
                # Interpolate coordinates
                x = start_point.coordinate.x + t * (end_point.coordinate.x - start_point.coordinate.x)
                y = start_point.coordinate.y + t * (end_point.coordinate.y - start_point.coordinate.y)
                
                # Interpolate timestamp
                time_offset = timedelta(seconds=t * gap_duration)
                timestamp = start_point.coordinate.timestamp + time_offset
                
                # Create interpolated coordinate
                interpolated_coord = Coordinate(
                    x=x,
                    y=y,
                    coordinate_system=start_point.coordinate.coordinate_system,
                    timestamp=timestamp,
                    camera_id=start_point.coordinate.camera_id,
                    frame_index=None,  # No frame for interpolated points
                    confidence=min(start_point.coordinate.confidence or 0.5, end_point.coordinate.confidence or 0.5)
                )
                
                # Create interpolated point
                interpolated_point = TrajectoryPoint(
                    coordinate=interpolated_coord,
                    camera_id=start_point.camera_id,
                    detection_id=None,  # No detection for interpolated points
                    interpolated=True,
                    velocity=None,  # Will be calculated later
                    acceleration=None
                )
                
                interpolated_points.append(interpolated_point)
            
            return interpolated_points
            
        except Exception as e:
            logger.error(f"Error interpolating gap: {e}")
            return []
    
    def remove_trajectory_outliers(self, trajectory: Trajectory) -> Trajectory:
        """Remove outlier points from trajectory."""
        try:
            if len(trajectory.path_points) < 5:
                return trajectory
            
            # Calculate position changes
            position_changes = []
            
            for i in range(1, len(trajectory.path_points)):
                current = trajectory.path_points[i]
                previous = trajectory.path_points[i-1]
                
                dx = current.coordinate.x - previous.coordinate.x
                dy = current.coordinate.y - previous.coordinate.y
                distance = np.sqrt(dx**2 + dy**2)
                
                position_changes.append(distance)
            
            # Calculate statistics
            mean_change = np.mean(position_changes)
            std_change = np.std(position_changes)
            
            # Find outliers
            outlier_indices = []
            
            for i, change in enumerate(position_changes):
                if abs(change - mean_change) > self.outlier_threshold * std_change:
                    outlier_indices.append(i + 1)  # +1 because we calculated changes from index 1
            
            # Remove outliers
            filtered_points = []
            
            for i, point in enumerate(trajectory.path_points):
                if i not in outlier_indices:
                    filtered_points.append(point)
            
            # Create filtered trajectory
            filtered_trajectory = Trajectory(
                global_id=trajectory.global_id,
                path_points=filtered_points,
                start_time=trajectory.start_time,
                end_time=trajectory.end_time,
                cameras_traversed=trajectory.cameras_traversed,
                smoothness_score=trajectory.smoothness_score,
                completeness_score=trajectory.completeness_score,
                confidence_score=trajectory.confidence_score,
                created_at=trajectory.created_at,
                last_updated=datetime.now(timezone.utc)
            )
            
            # Recalculate metrics
            filtered_trajectory = self._calculate_trajectory_metrics(filtered_trajectory)
            
            self.builder_stats["outliers_removed"] += len(outlier_indices)
            
            return filtered_trajectory
            
        except Exception as e:
            logger.error(f"Error removing trajectory outliers: {e}")
            return trajectory
    
    def get_builder_stats(self) -> Dict[str, Any]:
        """Get trajectory builder statistics."""
        return {
            **self.builder_stats,
            "min_trajectory_length": self.min_trajectory_length,
            "max_gap_duration": self.max_gap_duration,
            "smoothing_window": self.smoothing_window,
            "outlier_threshold": self.outlier_threshold
        }
    
    def reset_stats(self):
        """Reset builder statistics."""
        self.builder_stats = {
            "trajectories_built": 0,
            "trajectories_smoothed": 0,
            "trajectories_interpolated": 0,
            "outliers_removed": 0,
            "gaps_filled": 0,
            "multi_camera_fusions": 0
        }
        logger.info("Trajectory builder statistics reset")
    
    def cleanup(self):
        """Clean up builder resources."""
        # No resources to clean up currently
        logger.info("TrajectoryBuilder cleaned up")