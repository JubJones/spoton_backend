"""
Mapping service for unified spatial coordinate transformation.

Provides business logic for:
- Coordinate transformation between systems
- Homography matrix management
- Camera calibration handling
- Spatial mapping operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np

from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem, CoordinateTransformation
from app.domains.mapping.entities.camera_view import CameraView, CameraViewManager
from app.domains.mapping.entities.trajectory import Trajectory, TrajectoryPoint
from app.domains.detection.entities.detection import Detection, DetectionBatch
from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class MappingService:
    """Service for spatial mapping and coordinate transformation."""
    
    def __init__(self):
        self.camera_view_manager = CameraViewManager()
        self.homography_matrices: Dict[CameraID, np.ndarray] = {}
        self.transformation_cache: Dict[Tuple[CameraID, str], CoordinateTransformation] = {}
        self.mapping_stats = {
            "total_transformations": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("MappingService initialized")
    
    def load_camera_homography(self, camera_id: CameraID, homography_matrix: np.ndarray):
        """Load homography matrix for a camera."""
        try:
            if homography_matrix.shape != (3, 3):
                raise ValueError("Homography matrix must be 3x3")
            
            self.homography_matrices[camera_id] = homography_matrix
            
            # Create coordinate transformation
            transformation = CoordinateTransformation(
                source_system=CoordinateSystem.IMAGE,
                target_system=CoordinateSystem.MAP,
                transformation_matrix=homography_matrix.tolist(),
                camera_id=camera_id,
                calibration_date=datetime.now(timezone.utc)
            )
            
            # Cache the transformation
            cache_key = (camera_id, "image_to_map")
            self.transformation_cache[cache_key] = transformation
            
            logger.info(f"Loaded homography matrix for camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Error loading homography matrix for camera {camera_id}: {e}")
            raise
    
    def register_camera_view(self, camera_view: CameraView):
        """Register a camera view for mapping."""
        try:
            self.camera_view_manager.add_camera_view(camera_view)
            
            # Cache the homography transformation
            cache_key = (camera_view.camera_id, "image_to_map")
            self.transformation_cache[cache_key] = camera_view.homography_transformation
            
            logger.info(f"Registered camera view for {camera_view.camera_id}")
            
        except Exception as e:
            logger.error(f"Error registering camera view for {camera_view.camera_id}: {e}")
            raise
    
    async def transform_detections_to_map(
        self, 
        detection_batch: DetectionBatch
    ) -> Dict[str, Any]:
        """Transform detection coordinates to map coordinates."""
        try:
            transformed_detections = []
            transformation_results = {
                "successful": 0,
                "failed": 0,
                "cameras_processed": set()
            }
            
            for detection in detection_batch.detections:
                try:
                    # Get detection center point
                    center_x = detection.bbox.center_x
                    center_y = detection.bbox.center_y
                    
                    # Create coordinate object
                    image_coord = Coordinate(
                        x=center_x,
                        y=center_y,
                        coordinate_system=CoordinateSystem.IMAGE,
                        timestamp=detection.timestamp,
                        camera_id=detection.camera_id,
                        frame_index=detection.frame_index
                    )
                    
                    # Transform to map coordinates
                    map_coord = await self._transform_coordinate(
                        image_coord, 
                        detection.camera_id, 
                        CoordinateSystem.MAP
                    )
                    
                    if map_coord:
                        # Create new detection with map coordinates
                        detection_dict = {
                            "detection": detection,
                            "map_coordinate": map_coord,
                            "transformation_success": True
                        }
                        
                        transformed_detections.append(detection_dict)
                        transformation_results["successful"] += 1
                        transformation_results["cameras_processed"].add(detection.camera_id)
                    else:
                        # Keep original detection without map coordinates
                        detection_dict = {
                            "detection": detection,
                            "map_coordinate": None,
                            "transformation_success": False
                        }
                        
                        transformed_detections.append(detection_dict)
                        transformation_results["failed"] += 1
                        
                except Exception as e:
                    logger.warning(f"Error transforming detection {detection.id}: {e}")
                    transformation_results["failed"] += 1
            
            self.mapping_stats["total_transformations"] += len(detection_batch.detections)
            self.mapping_stats["successful_transformations"] += transformation_results["successful"]
            self.mapping_stats["failed_transformations"] += transformation_results["failed"]
            
            logger.info(
                f"Transformed {transformation_results['successful']}/{len(detection_batch.detections)} "
                f"detections from {len(transformation_results['cameras_processed'])} cameras"
            )
            
            return {
                "transformed_detections": transformed_detections,
                "results": transformation_results,
                "processing_time": detection_batch.processing_time
            }
            
        except Exception as e:
            logger.error(f"Error transforming detection batch: {e}")
            raise
    
    async def _transform_coordinate(
        self,
        coordinate: Coordinate,
        camera_id: CameraID,
        target_system: CoordinateSystem
    ) -> Optional[Coordinate]:
        """Transform a coordinate to target system."""
        try:
            # Check cache first
            cache_key = (camera_id, f"{coordinate.coordinate_system.value}_to_{target_system.value}")
            
            if cache_key in self.transformation_cache:
                transformation = self.transformation_cache[cache_key]
                self.mapping_stats["cache_hits"] += 1
                
                return transformation.transform_coordinate(coordinate)
            else:
                self.mapping_stats["cache_misses"] += 1
                
                # Try to create transformation
                transformation = await self._create_transformation(
                    camera_id,
                    coordinate.coordinate_system,
                    target_system
                )
                
                if transformation:
                    # Cache the transformation
                    self.transformation_cache[cache_key] = transformation
                    return transformation.transform_coordinate(coordinate)
                else:
                    return None
                    
        except Exception as e:
            logger.warning(f"Error transforming coordinate for camera {camera_id}: {e}")
            return None
    
    async def _create_transformation(
        self,
        camera_id: CameraID,
        source_system: CoordinateSystem,
        target_system: CoordinateSystem
    ) -> Optional[CoordinateTransformation]:
        """Create coordinate transformation for camera."""
        try:
            # Get camera view
            camera_view = self.camera_view_manager.get_camera_view(camera_id)
            if not camera_view:
                # Check if we have homography matrix directly
                if camera_id in self.homography_matrices:
                    matrix = self.homography_matrices[camera_id]
                    
                    return CoordinateTransformation(
                        source_system=source_system,
                        target_system=target_system,
                        transformation_matrix=matrix.tolist(),
                        camera_id=camera_id,
                        calibration_date=datetime.now(timezone.utc)
                    )
                else:
                    logger.warning(f"No homography data available for camera {camera_id}")
                    return None
            
            # Use camera view transformation
            if (source_system == CoordinateSystem.IMAGE and 
                target_system == CoordinateSystem.MAP):
                return camera_view.homography_transformation
            
            # For other transformations, we'd need to implement inverse or chained transformations
            logger.warning(f"Unsupported transformation: {source_system} -> {target_system}")
            return None
            
        except Exception as e:
            logger.error(f"Error creating transformation for camera {camera_id}: {e}")
            return None
    
    def build_trajectory_from_detections(
        self,
        global_id: str,
        detections: List[Detection]
    ) -> Optional[Trajectory]:
        """Build trajectory from a sequence of detections."""
        try:
            if not detections:
                return None
            
            # Sort detections by timestamp
            sorted_detections = sorted(detections, key=lambda d: d.timestamp)
            
            # Create trajectory points
            trajectory_points = []
            
            for detection in sorted_detections:
                # Get detection center
                center_x = detection.bbox.center_x
                center_y = detection.bbox.center_y
                
                # Create image coordinate
                image_coord = Coordinate(
                    x=center_x,
                    y=center_y,
                    coordinate_system=CoordinateSystem.IMAGE,
                    timestamp=detection.timestamp,
                    camera_id=detection.camera_id,
                    frame_index=detection.frame_index
                )
                
                # Try to transform to map coordinates
                map_coord = None
                try:
                    # Use synchronous version for now
                    transformation = self._get_cached_transformation(detection.camera_id)
                    if transformation:
                        map_coord = transformation.transform_coordinate(image_coord)
                except Exception as e:
                    logger.warning(f"Error transforming detection {detection.id} to map: {e}")
                
                # Create trajectory point (use map coordinate if available, otherwise image)
                coord_to_use = map_coord if map_coord else image_coord
                
                trajectory_point = TrajectoryPoint(
                    coordinate=coord_to_use,
                    camera_id=detection.camera_id,
                    detection_id=detection.id,
                    interpolated=False
                )
                
                trajectory_points.append(trajectory_point)
            
            # Create trajectory
            trajectory = Trajectory(
                global_id=global_id,
                path_points=trajectory_points,
                start_time=sorted_detections[0].timestamp,
                end_time=sorted_detections[-1].timestamp,
                cameras_traversed=list(set(d.camera_id for d in sorted_detections))
            )
            
            # Calculate quality metrics
            trajectory = trajectory.calculate_velocities()
            
            updated_trajectory = Trajectory(
                global_id=trajectory.global_id,
                path_points=trajectory.path_points,
                start_time=trajectory.start_time,
                end_time=trajectory.end_time,
                cameras_traversed=trajectory.cameras_traversed,
                smoothness_score=trajectory.calculate_smoothness(),
                completeness_score=trajectory.calculate_completeness(),
                confidence_score=sum(d.confidence for d in sorted_detections) / len(sorted_detections),
                created_at=trajectory.created_at,
                last_updated=datetime.now(timezone.utc)
            )
            
            logger.info(f"Built trajectory for {global_id} with {len(trajectory_points)} points")
            return updated_trajectory
            
        except Exception as e:
            logger.error(f"Error building trajectory for {global_id}: {e}")
            return None
    
    def _get_cached_transformation(self, camera_id: CameraID) -> Optional[CoordinateTransformation]:
        """Get cached transformation for camera."""
        cache_key = (camera_id, "image_to_map")
        return self.transformation_cache.get(cache_key)
    
    def get_camera_overlaps(self) -> List[Tuple[CameraID, CameraID, float]]:
        """Get overlapping regions between cameras."""
        overlaps = []
        
        try:
            camera_views = self.camera_view_manager.get_all_camera_views()
            
            for i, camera_view1 in enumerate(camera_views):
                for camera_view2 in camera_views[i+1:]:
                    overlap_region = camera_view1.get_overlapping_region(camera_view2)
                    
                    if overlap_region:
                        overlap_area = overlap_region.area
                        overlaps.append((camera_view1.camera_id, camera_view2.camera_id, overlap_area))
            
            return overlaps
            
        except Exception as e:
            logger.error(f"Error calculating camera overlaps: {e}")
            return []
    
    def validate_transformations(self) -> Dict[CameraID, bool]:
        """Validate all camera transformations."""
        results = {}
        
        for camera_id in self.homography_matrices:
            try:
                # Test transformation with a sample point
                test_coord = Coordinate(
                    x=100.0,
                    y=100.0,
                    coordinate_system=CoordinateSystem.IMAGE,
                    timestamp=datetime.now(timezone.utc),
                    camera_id=camera_id
                )
                
                transformation = self._get_cached_transformation(camera_id)
                if transformation:
                    transformed = transformation.transform_coordinate(test_coord)
                    results[camera_id] = transformed is not None
                else:
                    results[camera_id] = False
                    
            except Exception as e:
                logger.warning(f"Validation failed for camera {camera_id}: {e}")
                results[camera_id] = False
        
        return results
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get mapping service statistics."""
        return {
            **self.mapping_stats,
            "registered_cameras": len(self.camera_view_manager.camera_views),
            "cached_transformations": len(self.transformation_cache),
            "homography_matrices": len(self.homography_matrices),
            "total_coverage_area": self.camera_view_manager.get_total_coverage_area()
        }
    
    def get_camera_info(self, camera_id: CameraID) -> Optional[Dict[str, Any]]:
        """Get information about a specific camera."""
        camera_view = self.camera_view_manager.get_camera_view(camera_id)
        
        if not camera_view:
            return None
        
        # Get overlapping cameras
        overlapping_cameras = self.camera_view_manager.get_overlapping_cameras(camera_id)
        
        # Check transformation validity
        transformation_valid = camera_id in self.validate_transformations()
        
        return {
            "camera_view": camera_view.to_dict(),
            "overlapping_cameras": [
                {"camera_id": cam_id, "overlap_area": region.area}
                for cam_id, region in overlapping_cameras
            ],
            "transformation_valid": transformation_valid,
            "has_homography": camera_id in self.homography_matrices
        }
    
    def reset_stats(self):
        """Reset mapping statistics."""
        self.mapping_stats = {
            "total_transformations": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("Mapping statistics reset")
    
    def clear_cache(self):
        """Clear transformation cache."""
        self.transformation_cache.clear()
        logger.info("Transformation cache cleared")
    
    async def optimize_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Optimize trajectory by smoothing and interpolation."""
        try:
            # First, interpolate missing points
            interpolated = trajectory.interpolate_missing_points(target_fps=30.0)
            
            # Then smooth the trajectory
            smoothed = interpolated.smooth_trajectory(window_size=5)
            
            # Recalculate velocities
            optimized = smoothed.calculate_velocities()
            
            logger.info(f"Optimized trajectory {trajectory.global_id}")
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing trajectory {trajectory.global_id}: {e}")
            return trajectory