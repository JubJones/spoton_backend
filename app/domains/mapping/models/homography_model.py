"""
Homography transformation model for coordinate conversion.

Provides homography matrix operations for transforming coordinates between
camera image space and unified map space.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import cv2

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem
from app.domains.mapping.entities.camera_view import CameraView
from app.infrastructure.gpu import get_gpu_manager
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


@dataclass
class HomographyValidationResult:
    """Result of homography matrix validation."""
    is_valid: bool
    error_message: Optional[str] = None
    condition_number: Optional[float] = None
    determinant: Optional[float] = None
    test_transformation_success: bool = False


class HomographyModel:
    """
    Homography transformation model for camera coordinate mapping.
    
    Features:
    - Homography matrix validation and loading
    - Coordinate transformation between image and map spaces
    - GPU acceleration for batch transformations
    - Inverse transformation support
    - Transformation quality assessment
    """
    
    def __init__(
        self,
        camera_id: CameraID,
        enable_gpu: bool = True,
        validation_threshold: float = 1000.0
    ):
        """
        Initialize homography model.
        
        Args:
            camera_id: Camera identifier
            enable_gpu: Whether to use GPU acceleration
            validation_threshold: Condition number threshold for validation
        """
        self.camera_id = camera_id
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.validation_threshold = validation_threshold
        
        # Homography matrices
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        
        # GPU arrays (if enabled)
        self.gpu_homography: Optional[Any] = None
        self.gpu_inverse_homography: Optional[Any] = None
        
        # Validation results
        self.validation_result: Optional[HomographyValidationResult] = None
        
        # Performance tracking
        self.transformation_stats = {
            "total_transformations": 0,
            "batch_transformations": 0,
            "gpu_transformations": 0,
            "cpu_transformations": 0,
            "validation_checks": 0
        }
        
        # GPU manager
        self.gpu_manager = get_gpu_manager()
        
        logger.info(f"HomographyModel initialized for camera {camera_id}, GPU: {self.enable_gpu}")
    
    def load_homography_matrix(self, homography_matrix: np.ndarray) -> bool:
        """
        Load and validate homography matrix.
        
        Args:
            homography_matrix: 3x3 homography matrix
            
        Returns:
            True if matrix is valid and loaded successfully
        """
        try:
            # Validate matrix shape
            if homography_matrix.shape != (3, 3):
                raise ValueError(f"Homography matrix must be 3x3, got {homography_matrix.shape}")
            
            # Validate matrix content
            validation_result = self._validate_homography_matrix(homography_matrix)
            self.validation_result = validation_result
            
            if not validation_result.is_valid:
                logger.error(f"Invalid homography matrix for camera {self.camera_id}: {validation_result.error_message}")
                return False
            
            # Store matrices
            self.homography_matrix = homography_matrix.copy()
            self.inverse_homography = self._compute_inverse_homography(homography_matrix)
            
            if self.inverse_homography is None:
                logger.error(f"Failed to compute inverse homography for camera {self.camera_id}")
                return False
            
            # Load to GPU if enabled
            if self.enable_gpu:
                self._load_to_gpu()
            
            logger.info(f"Homography matrix loaded successfully for camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading homography matrix for camera {self.camera_id}: {e}")
            return False
    
    def _validate_homography_matrix(self, matrix: np.ndarray) -> HomographyValidationResult:
        """Validate homography matrix properties."""
        try:
            self.transformation_stats["validation_checks"] += 1
            
            # Check for NaN or infinite values
            if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                return HomographyValidationResult(
                    is_valid=False,
                    error_message="Matrix contains NaN or infinite values"
                )
            
            # Check determinant (should not be zero)
            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:
                return HomographyValidationResult(
                    is_valid=False,
                    error_message="Matrix is singular (determinant near zero)",
                    determinant=det
                )
            
            # Check condition number (should not be too large)
            cond_number = np.linalg.cond(matrix)
            if cond_number > self.validation_threshold:
                return HomographyValidationResult(
                    is_valid=False,
                    error_message=f"Matrix is ill-conditioned (condition number: {cond_number:.2f})",
                    condition_number=cond_number,
                    determinant=det
                )
            
            # Test transformation with sample points
            test_success = self._test_transformation(matrix)
            
            return HomographyValidationResult(
                is_valid=True,
                condition_number=cond_number,
                determinant=det,
                test_transformation_success=test_success
            )
            
        except Exception as e:
            return HomographyValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}"
            )
    
    def _test_transformation(self, matrix: np.ndarray) -> bool:
        """Test homography transformation with sample points."""
        try:
            # Test points in image space
            test_points = np.array([
                [100, 100],
                [200, 200],
                [300, 150],
                [150, 300]
            ], dtype=np.float32)
            
            # Transform points
            transformed = cv2.perspectiveTransform(
                test_points.reshape(-1, 1, 2), 
                matrix
            )
            
            # Check if transformed points are reasonable
            if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
                return False
            
            # Check if points are within reasonable bounds
            # (adjust these bounds based on your map coordinate system)
            if np.any(np.abs(transformed) > 10000):  # Arbitrary large bound
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Transformation test failed: {e}")
            return False
    
    def _compute_inverse_homography(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        """Compute inverse homography matrix."""
        try:
            inverse = np.linalg.inv(matrix)
            
            # Validate inverse
            if np.any(np.isnan(inverse)) or np.any(np.isinf(inverse)):
                return None
            
            # Test that H * H^-1 = I
            identity_test = np.dot(matrix, inverse)
            expected_identity = np.eye(3)
            
            if not np.allclose(identity_test, expected_identity, atol=1e-6):
                logger.warning(f"Inverse homography validation failed for camera {self.camera_id}")
                return None
            
            return inverse
            
        except Exception as e:
            logger.error(f"Error computing inverse homography: {e}")
            return None
    
    def _load_to_gpu(self):
        """Load homography matrices to GPU."""
        if not self.enable_gpu or not CUPY_AVAILABLE:
            return
        
        try:
            # Move to GPU
            self.gpu_homography = cp.asarray(self.homography_matrix)
            self.gpu_inverse_homography = cp.asarray(self.inverse_homography)
            
            logger.debug(f"Homography matrices loaded to GPU for camera {self.camera_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load homography to GPU: {e}")
            self.enable_gpu = False
    
    def transform_coordinate(
        self, 
        coordinate: Coordinate, 
        target_system: CoordinateSystem
    ) -> Optional[Coordinate]:
        """
        Transform coordinate between systems.
        
        Args:
            coordinate: Input coordinate
            target_system: Target coordinate system
            
        Returns:
            Transformed coordinate or None if transformation failed
        """
        if self.homography_matrix is None:
            logger.error(f"No homography matrix loaded for camera {self.camera_id}")
            return None
        
        try:
            # Determine transformation direction
            if (coordinate.coordinate_system == CoordinateSystem.IMAGE and 
                target_system == CoordinateSystem.MAP):
                # Image to map transformation
                transformed_point = self._transform_point_to_map(coordinate.x, coordinate.y)
            elif (coordinate.coordinate_system == CoordinateSystem.MAP and 
                  target_system == CoordinateSystem.IMAGE):
                # Map to image transformation
                transformed_point = self._transform_point_to_image(coordinate.x, coordinate.y)
            else:
                logger.error(f"Unsupported coordinate transformation: {coordinate.coordinate_system} -> {target_system}")
                return None
            
            if transformed_point is None:
                return None
            
            # Create transformed coordinate
            transformed_coord = Coordinate(
                x=transformed_point[0],
                y=transformed_point[1],
                coordinate_system=target_system,
                timestamp=coordinate.timestamp,
                camera_id=self.camera_id,
                frame_index=coordinate.frame_index,
                confidence=coordinate.confidence * 0.95  # Slight confidence reduction due to transformation
            )
            
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["cpu_transformations"] += 1
            
            return transformed_coord
            
        except Exception as e:
            logger.error(f"Error transforming coordinate: {e}")
            return None
    
    def transform_coordinates_batch(
        self, 
        coordinates: List[Coordinate], 
        target_system: CoordinateSystem
    ) -> List[Optional[Coordinate]]:
        """
        Transform multiple coordinates in batch.
        
        Args:
            coordinates: List of input coordinates
            target_system: Target coordinate system
            
        Returns:
            List of transformed coordinates (None for failed transformations)
        """
        if not coordinates:
            return []
        
        if self.homography_matrix is None:
            logger.error(f"No homography matrix loaded for camera {self.camera_id}")
            return [None] * len(coordinates)
        
        try:
            results = []
            
            # Group coordinates by transformation direction
            image_to_map = []
            map_to_image = []
            image_to_map_indices = []
            map_to_image_indices = []
            
            for i, coord in enumerate(coordinates):
                if (coord.coordinate_system == CoordinateSystem.IMAGE and 
                    target_system == CoordinateSystem.MAP):
                    image_to_map.append((coord.x, coord.y))
                    image_to_map_indices.append(i)
                elif (coord.coordinate_system == CoordinateSystem.MAP and 
                      target_system == CoordinateSystem.IMAGE):
                    map_to_image.append((coord.x, coord.y))
                    map_to_image_indices.append(i)
                else:
                    logger.warning(f"Unsupported transformation at index {i}")
            
            # Initialize results array
            results = [None] * len(coordinates)
            
            # Process image to map transformations
            if image_to_map:
                transformed_points = self._transform_points_to_map_batch(image_to_map)
                
                for idx, (orig_idx, transformed_point) in enumerate(zip(image_to_map_indices, transformed_points)):
                    if transformed_point is not None:
                        original_coord = coordinates[orig_idx]
                        results[orig_idx] = Coordinate(
                            x=transformed_point[0],
                            y=transformed_point[1],
                            coordinate_system=target_system,
                            timestamp=original_coord.timestamp,
                            camera_id=self.camera_id,
                            frame_index=original_coord.frame_index,
                            confidence=original_coord.confidence * 0.95
                        )
            
            # Process map to image transformations
            if map_to_image:
                transformed_points = self._transform_points_to_image_batch(map_to_image)
                
                for idx, (orig_idx, transformed_point) in enumerate(zip(map_to_image_indices, transformed_points)):
                    if transformed_point is not None:
                        original_coord = coordinates[orig_idx]
                        results[orig_idx] = Coordinate(
                            x=transformed_point[0],
                            y=transformed_point[1],
                            coordinate_system=target_system,
                            timestamp=original_coord.timestamp,
                            camera_id=self.camera_id,
                            frame_index=original_coord.frame_index,
                            confidence=original_coord.confidence * 0.95
                        )
            
            # Update statistics
            self.transformation_stats["total_transformations"] += len(coordinates)
            self.transformation_stats["batch_transformations"] += 1
            
            if self.enable_gpu:
                self.transformation_stats["gpu_transformations"] += len(coordinates)
            else:
                self.transformation_stats["cpu_transformations"] += len(coordinates)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch coordinate transformation: {e}")
            return [None] * len(coordinates)
    
    def _transform_point_to_map(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Transform single point from image to map coordinates."""
        try:
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(
                point.reshape(-1, 1, 2), 
                self.homography_matrix
            )
            
            result = transformed.reshape(-1, 2)[0]
            
            # Validate result
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return None
            
            return float(result[0]), float(result[1])
            
        except Exception as e:
            logger.warning(f"Error transforming point to map: {e}")
            return None
    
    def _transform_point_to_image(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Transform single point from map to image coordinates."""
        try:
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(
                point.reshape(-1, 1, 2), 
                self.inverse_homography
            )
            
            result = transformed.reshape(-1, 2)[0]
            
            # Validate result
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return None
            
            return float(result[0]), float(result[1])
            
        except Exception as e:
            logger.warning(f"Error transforming point to image: {e}")
            return None
    
    def _transform_points_to_map_batch(self, points: List[Tuple[float, float]]) -> List[Optional[Tuple[float, float]]]:
        """Transform multiple points from image to map coordinates."""
        try:
            if not points:
                return []
            
            # Convert to numpy array
            points_array = np.array(points, dtype=np.float32)
            
            # Use GPU if available
            if self.enable_gpu and self.gpu_homography is not None:
                return self._gpu_transform_points(points_array, self.gpu_homography)
            else:
                return self._cpu_transform_points(points_array, self.homography_matrix)
                
        except Exception as e:
            logger.error(f"Error in batch transformation to map: {e}")
            return [None] * len(points)
    
    def _transform_points_to_image_batch(self, points: List[Tuple[float, float]]) -> List[Optional[Tuple[float, float]]]:
        """Transform multiple points from map to image coordinates."""
        try:
            if not points:
                return []
            
            # Convert to numpy array
            points_array = np.array(points, dtype=np.float32)
            
            # Use GPU if available
            if self.enable_gpu and self.gpu_inverse_homography is not None:
                return self._gpu_transform_points(points_array, self.gpu_inverse_homography)
            else:
                return self._cpu_transform_points(points_array, self.inverse_homography)
                
        except Exception as e:
            logger.error(f"Error in batch transformation to image: {e}")
            return [None] * len(points)
    
    def _cpu_transform_points(self, points: np.ndarray, matrix: np.ndarray) -> List[Optional[Tuple[float, float]]]:
        """CPU-based batch point transformation."""
        try:
            transformed = cv2.perspectiveTransform(
                points.reshape(-1, 1, 2), 
                matrix
            )
            
            results = []
            for point in transformed.reshape(-1, 2):
                if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                    results.append(None)
                else:
                    results.append((float(point[0]), float(point[1])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in CPU point transformation: {e}")
            return [None] * len(points)
    
    def _gpu_transform_points(self, points: np.ndarray, gpu_matrix: Any) -> List[Optional[Tuple[float, float]]]:
        """GPU-based batch point transformation."""
        try:
            # Move points to GPU
            gpu_points = cp.asarray(points)
            
            # Add homogeneous coordinate
            ones = cp.ones((gpu_points.shape[0], 1))
            homogeneous_points = cp.hstack([gpu_points, ones])
            
            # Transform points
            transformed_homogeneous = cp.dot(homogeneous_points, gpu_matrix.T)
            
            # Convert back to Cartesian coordinates
            transformed_points = transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3]
            
            # Move back to CPU
            cpu_points = cp.asnumpy(transformed_points)
            
            # Validate and convert results
            results = []
            for point in cpu_points:
                if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                    results.append(None)
                else:
                    results.append((float(point[0]), float(point[1])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in GPU point transformation: {e}")
            # Fallback to CPU
            return self._cpu_transform_points(points, cp.asnumpy(gpu_matrix))
    
    def get_transformation_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Get transformation bounds for validation."""
        if self.homography_matrix is None:
            return None
        
        try:
            # Test with corner points and some interior points
            test_points = [
                (0, 0),
                (640, 0),
                (640, 480),
                (0, 480),
                (320, 240),
                (160, 120),
                (480, 360)
            ]
            
            transformed_points = self._transform_points_to_map_batch(test_points)
            
            valid_points = [p for p in transformed_points if p is not None]
            
            if not valid_points:
                return None
            
            x_coords = [p[0] for p in valid_points]
            y_coords = [p[1] for p in valid_points]
            
            return {
                "x_bounds": (min(x_coords), max(x_coords)),
                "y_bounds": (min(y_coords), max(y_coords))
            }
            
        except Exception as e:
            logger.error(f"Error computing transformation bounds: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if homography matrix is loaded."""
        return self.homography_matrix is not None
    
    def get_validation_result(self) -> Optional[HomographyValidationResult]:
        """Get validation result."""
        return self.validation_result
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        return {
            **self.transformation_stats,
            "camera_id": self.camera_id,
            "gpu_enabled": self.enable_gpu,
            "is_loaded": self.is_loaded(),
            "validation_result": self.validation_result.__dict__ if self.validation_result else None
        }
    
    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """Get homography matrix."""
        return self.homography_matrix.copy() if self.homography_matrix is not None else None
    
    def get_inverse_homography_matrix(self) -> Optional[np.ndarray]:
        """Get inverse homography matrix."""
        return self.inverse_homography.copy() if self.inverse_homography is not None else None
    
    def reset_stats(self):
        """Reset transformation statistics."""
        self.transformation_stats = {
            "total_transformations": 0,
            "batch_transformations": 0,
            "gpu_transformations": 0,
            "cpu_transformations": 0,
            "validation_checks": 0
        }
        logger.info(f"Transformation stats reset for camera {self.camera_id}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.enable_gpu and CUPY_AVAILABLE:
            if self.gpu_homography is not None:
                del self.gpu_homography
            if self.gpu_inverse_homography is not None:
                del self.gpu_inverse_homography
        
        self.homography_matrix = None
        self.inverse_homography = None
        self.validation_result = None
        
        logger.info(f"HomographyModel cleaned up for camera {self.camera_id}")