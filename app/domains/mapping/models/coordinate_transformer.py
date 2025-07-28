"""
Coordinate transformer for multi-camera coordinate system conversion.

Handles transformations between different coordinate systems including:
- Image coordinates to world coordinates
- World coordinates to map coordinates
- Cross-camera coordinate transformations
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem
from app.domains.mapping.entities.camera_view import CameraView
from app.domains.mapping.models.homography_model import HomographyModel
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class TransformationMode(Enum):
    """Available transformation modes."""
    DIRECT = "direct"  # Direct homography transformation
    CHAIN = "chain"    # Chained transformations through intermediate systems
    OPTIMIZE = "optimize"  # Optimized transformation with caching


@dataclass
class TransformationPath:
    """Represents a path for coordinate transformation."""
    source_system: CoordinateSystem
    target_system: CoordinateSystem
    intermediate_systems: List[CoordinateSystem] = field(default_factory=list)
    transformation_matrices: List[np.ndarray] = field(default_factory=list)
    total_error_estimate: float = 0.0
    path_confidence: float = 1.0


@dataclass
class TransformationResult:
    """Result of coordinate transformation."""
    transformed_coordinate: Optional[Coordinate]
    transformation_path: TransformationPath
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    confidence_score: float = 1.0


class CoordinateTransformer:
    """
    Advanced coordinate transformer for multi-camera systems.
    
    Features:
    - Multi-system coordinate transformations
    - Homography-based transformations
    - Chained transformations through intermediate systems
    - Transformation caching and optimization
    - Error estimation and confidence scoring
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_size: int = 1000,
        default_mode: TransformationMode = TransformationMode.DIRECT
    ):
        """
        Initialize coordinate transformer.
        
        Args:
            enable_caching: Whether to enable transformation caching
            cache_size: Maximum number of cached transformations
            default_mode: Default transformation mode
        """
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.default_mode = default_mode
        
        # Homography models for each camera
        self.homography_models: Dict[CameraID, HomographyModel] = {}
        
        # Transformation cache
        self.transformation_cache: Dict[str, TransformationResult] = {}
        
        # Transformation paths
        self.transformation_paths: Dict[Tuple[CoordinateSystem, CoordinateSystem], TransformationPath] = {}
        
        # Performance tracking
        self.transformer_stats = {
            "total_transformations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "direct_transformations": 0,
            "chained_transformations": 0
        }
        
        logger.info("CoordinateTransformer initialized")
    
    def register_camera_homography(
        self, 
        camera_id: CameraID, 
        homography_matrix: np.ndarray,
        enable_gpu: bool = True
    ) -> bool:
        """
        Register homography model for a camera.
        
        Args:
            camera_id: Camera identifier
            homography_matrix: 3x3 homography matrix
            enable_gpu: Whether to enable GPU acceleration
            
        Returns:
            True if registration successful
        """
        try:
            # Create homography model
            homography_model = HomographyModel(
                camera_id=camera_id,
                enable_gpu=enable_gpu
            )
            
            # Load homography matrix
            if homography_model.load_homography_matrix(homography_matrix):
                self.homography_models[camera_id] = homography_model
                
                # Update transformation paths
                self._update_transformation_paths(camera_id)
                
                logger.info(f"Registered homography model for camera {camera_id}")
                return True
            else:
                logger.error(f"Failed to load homography matrix for camera {camera_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering camera homography for {camera_id}: {e}")
            return False
    
    def _update_transformation_paths(self, camera_id: CameraID):
        """Update available transformation paths for a camera."""
        try:
            # Add direct image-to-map transformation path
            image_to_map_path = TransformationPath(
                source_system=CoordinateSystem.IMAGE,
                target_system=CoordinateSystem.MAP,
                intermediate_systems=[],
                transformation_matrices=[self.homography_models[camera_id].get_homography_matrix()],
                total_error_estimate=0.01,  # Estimated transformation error
                path_confidence=0.95
            )
            
            self.transformation_paths[(CoordinateSystem.IMAGE, CoordinateSystem.MAP)] = image_to_map_path
            
            # Add reverse map-to-image transformation path
            map_to_image_path = TransformationPath(
                source_system=CoordinateSystem.MAP,
                target_system=CoordinateSystem.IMAGE,
                intermediate_systems=[],
                transformation_matrices=[self.homography_models[camera_id].get_inverse_homography_matrix()],
                total_error_estimate=0.01,
                path_confidence=0.95
            )
            
            self.transformation_paths[(CoordinateSystem.MAP, CoordinateSystem.IMAGE)] = map_to_image_path
            
        except Exception as e:
            logger.error(f"Error updating transformation paths for {camera_id}: {e}")
    
    def transform_coordinate(
        self, 
        coordinate: Coordinate, 
        target_system: CoordinateSystem,
        camera_id: Optional[CameraID] = None,
        mode: Optional[TransformationMode] = None
    ) -> TransformationResult:
        """
        Transform coordinate to target system.
        
        Args:
            coordinate: Source coordinate
            target_system: Target coordinate system
            camera_id: Camera ID for transformation (uses coordinate's camera_id if None)
            mode: Transformation mode (uses default if None)
            
        Returns:
            Transformation result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Use coordinate's camera_id if not provided
            if camera_id is None:
                camera_id = coordinate.camera_id
            
            if camera_id is None:
                return TransformationResult(
                    transformed_coordinate=None,
                    transformation_path=TransformationPath(
                        source_system=coordinate.coordinate_system,
                        target_system=target_system
                    ),
                    success=False,
                    error_message="No camera ID provided"
                )
            
            # Check cache first
            if self.enable_caching:
                cache_key = self._generate_cache_key(coordinate, target_system, camera_id)
                if cache_key in self.transformation_cache:
                    cached_result = self.transformation_cache[cache_key]
                    self.transformer_stats["cache_hits"] += 1
                    return cached_result
                else:
                    self.transformer_stats["cache_misses"] += 1
            
            # Use provided mode or default
            transformation_mode = mode or self.default_mode
            
            # Perform transformation
            if transformation_mode == TransformationMode.DIRECT:
                result = self._direct_transform(coordinate, target_system, camera_id)
            elif transformation_mode == TransformationMode.CHAIN:
                result = self._chain_transform(coordinate, target_system, camera_id)
            elif transformation_mode == TransformationMode.OPTIMIZE:
                result = self._optimize_transform(coordinate, target_system, camera_id)
            else:
                result = TransformationResult(
                    transformed_coordinate=None,
                    transformation_path=TransformationPath(
                        source_system=coordinate.coordinate_system,
                        target_system=target_system
                    ),
                    success=False,
                    error_message=f"Unknown transformation mode: {transformation_mode}"
                )
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time = processing_time
            
            # Update statistics
            self.transformer_stats["total_transformations"] += 1
            if result.success:
                self.transformer_stats["successful_transformations"] += 1
            else:
                self.transformer_stats["failed_transformations"] += 1
            
            # Cache result
            if self.enable_caching and result.success:
                self._cache_result(coordinate, target_system, camera_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transforming coordinate: {e}")
            return TransformationResult(
                transformed_coordinate=None,
                transformation_path=TransformationPath(
                    source_system=coordinate.coordinate_system,
                    target_system=target_system
                ),
                success=False,
                error_message=f"Transformation error: {e}"
            )
    
    def _direct_transform(
        self, 
        coordinate: Coordinate, 
        target_system: CoordinateSystem,
        camera_id: CameraID
    ) -> TransformationResult:
        """Direct transformation using homography model."""
        try:
            # Get homography model for camera
            if camera_id not in self.homography_models:
                return TransformationResult(
                    transformed_coordinate=None,
                    transformation_path=TransformationPath(
                        source_system=coordinate.coordinate_system,
                        target_system=target_system
                    ),
                    success=False,
                    error_message=f"No homography model for camera {camera_id}"
                )
            
            homography_model = self.homography_models[camera_id]
            
            # Transform coordinate
            transformed_coord = homography_model.transform_coordinate(coordinate, target_system)
            
            if transformed_coord is None:
                return TransformationResult(
                    transformed_coordinate=None,
                    transformation_path=TransformationPath(
                        source_system=coordinate.coordinate_system,
                        target_system=target_system
                    ),
                    success=False,
                    error_message="Direct transformation failed"
                )
            
            # Get transformation path
            path_key = (coordinate.coordinate_system, target_system)
            transformation_path = self.transformation_paths.get(
                path_key,
                TransformationPath(
                    source_system=coordinate.coordinate_system,
                    target_system=target_system
                )
            )
            
            self.transformer_stats["direct_transformations"] += 1
            
            return TransformationResult(
                transformed_coordinate=transformed_coord,
                transformation_path=transformation_path,
                success=True,
                confidence_score=transformation_path.path_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in direct transformation: {e}")
            return TransformationResult(
                transformed_coordinate=None,
                transformation_path=TransformationPath(
                    source_system=coordinate.coordinate_system,
                    target_system=target_system
                ),
                success=False,
                error_message=f"Direct transformation error: {e}"
            )
    
    def _chain_transform(
        self, 
        coordinate: Coordinate, 
        target_system: CoordinateSystem,
        camera_id: CameraID
    ) -> TransformationResult:
        """Chained transformation through intermediate systems."""
        try:
            # For now, use direct transformation
            # In future, this could implement multi-step transformations
            # e.g., Image -> World -> Map
            
            self.transformer_stats["chained_transformations"] += 1
            return self._direct_transform(coordinate, target_system, camera_id)
            
        except Exception as e:
            logger.error(f"Error in chained transformation: {e}")
            return TransformationResult(
                transformed_coordinate=None,
                transformation_path=TransformationPath(
                    source_system=coordinate.coordinate_system,
                    target_system=target_system
                ),
                success=False,
                error_message=f"Chained transformation error: {e}"
            )
    
    def _optimize_transform(
        self, 
        coordinate: Coordinate, 
        target_system: CoordinateSystem,
        camera_id: CameraID
    ) -> TransformationResult:
        """Optimized transformation with best path selection."""
        try:
            # For now, use direct transformation
            # In future, this could select the best transformation path
            # based on accuracy, speed, and other factors
            
            return self._direct_transform(coordinate, target_system, camera_id)
            
        except Exception as e:
            logger.error(f"Error in optimized transformation: {e}")
            return TransformationResult(
                transformed_coordinate=None,
                transformation_path=TransformationPath(
                    source_system=coordinate.coordinate_system,
                    target_system=target_system
                ),
                success=False,
                error_message=f"Optimized transformation error: {e}"
            )
    
    def transform_coordinates_batch(
        self,
        coordinates: List[Coordinate],
        target_system: CoordinateSystem,
        camera_id: Optional[CameraID] = None,
        mode: Optional[TransformationMode] = None
    ) -> List[TransformationResult]:
        """
        Transform multiple coordinates in batch.
        
        Args:
            coordinates: List of source coordinates
            target_system: Target coordinate system
            camera_id: Camera ID for transformation
            mode: Transformation mode
            
        Returns:
            List of transformation results
        """
        try:
            results = []
            
            # Group coordinates by camera_id for batch processing
            camera_groups: Dict[CameraID, List[Tuple[int, Coordinate]]] = {}
            
            for i, coord in enumerate(coordinates):
                coord_camera_id = camera_id or coord.camera_id
                if coord_camera_id not in camera_groups:
                    camera_groups[coord_camera_id] = []
                camera_groups[coord_camera_id].append((i, coord))
            
            # Initialize results array
            results = [None] * len(coordinates)
            
            # Process each camera group
            for cam_id, coord_group in camera_groups.items():
                if cam_id in self.homography_models:
                    homography_model = self.homography_models[cam_id]
                    
                    # Extract coordinates for batch processing
                    batch_coords = [coord for _, coord in coord_group]
                    
                    # Transform batch
                    transformed_coords = homography_model.transform_coordinates_batch(
                        batch_coords, target_system
                    )
                    
                    # Create results
                    for (orig_idx, orig_coord), transformed_coord in zip(coord_group, transformed_coords):
                        if transformed_coord is not None:
                            results[orig_idx] = TransformationResult(
                                transformed_coordinate=transformed_coord,
                                transformation_path=TransformationPath(
                                    source_system=orig_coord.coordinate_system,
                                    target_system=target_system
                                ),
                                success=True,
                                confidence_score=0.95
                            )
                        else:
                            results[orig_idx] = TransformationResult(
                                transformed_coordinate=None,
                                transformation_path=TransformationPath(
                                    source_system=orig_coord.coordinate_system,
                                    target_system=target_system
                                ),
                                success=False,
                                error_message="Batch transformation failed"
                            )
                else:
                    # No homography model for this camera
                    for orig_idx, orig_coord in coord_group:
                        results[orig_idx] = TransformationResult(
                            transformed_coordinate=None,
                            transformation_path=TransformationPath(
                                source_system=orig_coord.coordinate_system,
                                target_system=target_system
                            ),
                            success=False,
                            error_message=f"No homography model for camera {cam_id}"
                        )
            
            # Update statistics
            self.transformer_stats["total_transformations"] += len(coordinates)
            successful = sum(1 for r in results if r and r.success)
            failed = len(coordinates) - successful
            self.transformer_stats["successful_transformations"] += successful
            self.transformer_stats["failed_transformations"] += failed
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch transformation: {e}")
            return [
                TransformationResult(
                    transformed_coordinate=None,
                    transformation_path=TransformationPath(
                        source_system=coord.coordinate_system,
                        target_system=target_system
                    ),
                    success=False,
                    error_message=f"Batch transformation error: {e}"
                )
                for coord in coordinates
            ]
    
    def _generate_cache_key(
        self, 
        coordinate: Coordinate, 
        target_system: CoordinateSystem, 
        camera_id: CameraID
    ) -> str:
        """Generate cache key for transformation."""
        return f"{camera_id}:{coordinate.coordinate_system.value}:{target_system.value}:{coordinate.x:.2f}:{coordinate.y:.2f}"
    
    def _cache_result(
        self, 
        coordinate: Coordinate, 
        target_system: CoordinateSystem, 
        camera_id: CameraID,
        result: TransformationResult
    ):
        """Cache transformation result."""
        try:
            cache_key = self._generate_cache_key(coordinate, target_system, camera_id)
            
            # Implement LRU eviction if cache is full
            if len(self.transformation_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO for now)
                oldest_key = next(iter(self.transformation_cache))
                del self.transformation_cache[oldest_key]
            
            self.transformation_cache[cache_key] = result
            
        except Exception as e:
            logger.warning(f"Error caching transformation result: {e}")
    
    def get_available_transformations(self, camera_id: CameraID) -> List[Tuple[CoordinateSystem, CoordinateSystem]]:
        """Get available transformations for a camera."""
        if camera_id not in self.homography_models:
            return []
        
        return [
            (CoordinateSystem.IMAGE, CoordinateSystem.MAP),
            (CoordinateSystem.MAP, CoordinateSystem.IMAGE)
        ]
    
    def validate_camera_transformation(self, camera_id: CameraID) -> Dict[str, Any]:
        """Validate camera transformation setup."""
        if camera_id not in self.homography_models:
            return {
                "valid": False,
                "error": f"No homography model for camera {camera_id}"
            }
        
        homography_model = self.homography_models[camera_id]
        validation_result = homography_model.get_validation_result()
        
        if validation_result is None:
            return {
                "valid": False,
                "error": "No validation result available"
            }
        
        return {
            "valid": validation_result.is_valid,
            "error": validation_result.error_message,
            "condition_number": validation_result.condition_number,
            "determinant": validation_result.determinant,
            "test_transformation_success": validation_result.test_transformation_success
        }
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        return {
            **self.transformer_stats,
            "registered_cameras": len(self.homography_models),
            "cached_transformations": len(self.transformation_cache),
            "available_paths": len(self.transformation_paths),
            "cache_hit_rate": (
                self.transformer_stats["cache_hits"] / 
                max(1, self.transformer_stats["cache_hits"] + self.transformer_stats["cache_misses"])
            ),
            "success_rate": (
                self.transformer_stats["successful_transformations"] / 
                max(1, self.transformer_stats["total_transformations"])
            )
        }
    
    def clear_cache(self):
        """Clear transformation cache."""
        self.transformation_cache.clear()
        logger.info("Transformation cache cleared")
    
    def reset_stats(self):
        """Reset transformation statistics."""
        self.transformer_stats = {
            "total_transformations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "direct_transformations": 0,
            "chained_transformations": 0
        }
        logger.info("Transformation statistics reset")
    
    def cleanup(self):
        """Clean up transformer resources."""
        # Clean up all homography models
        for camera_id, homography_model in self.homography_models.items():
            homography_model.cleanup()
        
        self.homography_models.clear()
        self.transformation_cache.clear()
        self.transformation_paths.clear()
        
        logger.info("CoordinateTransformer cleaned up")