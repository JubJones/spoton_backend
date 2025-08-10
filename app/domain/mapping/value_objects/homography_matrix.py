"""
Homography matrix value object for coordinate transformations.

Provides type-safe homography matrix handling with validation and operations.
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from app.domain.shared.value_objects.base_value_object import BaseValueObject


@dataclass(frozen=True)
class HomographyMatrix(BaseValueObject):
    """
    Homography matrix value object.
    
    Represents a 3x3 homography matrix for perspective transformations
    between image and world coordinate systems.
    """
    
    matrix: np.ndarray  # 3x3 homography matrix
    source_points: Optional[List[Tuple[float, float]]] = None  # Source calibration points
    target_points: Optional[List[Tuple[float, float]]] = None  # Target calibration points
    calibration_error: float = 0.0  # RMS error from calibration
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Make matrix immutable
        self.matrix.flags.writeable = False
        super().__post_init__()
    
    def _validate(self) -> None:
        """Validate homography matrix."""
        if not isinstance(self.matrix, np.ndarray):
            raise ValueError("Matrix must be a numpy array")
        
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Matrix must be 3x3, got shape {self.matrix.shape}")
        
        if not np.isfinite(self.matrix).all():
            raise ValueError("Matrix contains invalid values (NaN or infinite)")
        
        # Check if matrix is singular (determinant near zero)
        det = np.linalg.det(self.matrix)
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular (determinant near zero)")
        
        if self.calibration_error < 0:
            raise ValueError("Calibration error must be non-negative")
        
        # Validate point correspondence if provided
        if self.source_points is not None and self.target_points is not None:
            if len(self.source_points) != len(self.target_points):
                raise ValueError("Source and target points must have same length")
            if len(self.source_points) < 4:
                raise ValueError("At least 4 point pairs required for homography")
    
    @classmethod
    def create(
        cls,
        matrix: np.ndarray,
        source_points: Optional[List[Tuple[float, float]]] = None,
        target_points: Optional[List[Tuple[float, float]]] = None,
        calibration_error: float = 0.0
    ) -> 'HomographyMatrix':
        """
        Create homography matrix with validation.
        
        Args:
            matrix: 3x3 homography matrix
            source_points: Source calibration points
            target_points: Target calibration points
            calibration_error: Calibration RMS error
            
        Returns:
            HomographyMatrix instance
        """
        matrix_copy = matrix.copy()
        return cls(
            matrix=matrix_copy,
            source_points=source_points,
            target_points=target_points,
            calibration_error=calibration_error
        )
    
    @classmethod
    def from_point_pairs(
        cls,
        source_points: List[Tuple[float, float]],
        target_points: List[Tuple[float, float]]
    ) -> 'HomographyMatrix':
        """
        Create homography matrix from point correspondences.
        
        Args:
            source_points: Source points (image coordinates)
            target_points: Target points (world coordinates)
            
        Returns:
            HomographyMatrix instance computed from point pairs
        """
        if len(source_points) != len(target_points):
            raise ValueError("Source and target points must have same length")
        if len(source_points) < 4:
            raise ValueError("At least 4 point pairs required for homography")
        
        # Convert to numpy arrays
        src_pts = np.array(source_points, dtype=np.float32)
        dst_pts = np.array(target_points, dtype=np.float32)
        
        # Compute homography using OpenCV-style algorithm
        try:
            import cv2
            matrix, mask = cv2.findHomography(
                src_pts, dst_pts, 
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            # Calculate reprojection error
            if mask is not None:
                inliers = np.sum(mask)
                error = cls._calculate_reprojection_error(matrix, src_pts, dst_pts)
            else:
                error = 0.0
            
        except ImportError:
            # Fallback to DLT algorithm if OpenCV not available
            matrix = cls._compute_dlt_homography(src_pts, dst_pts)
            error = cls._calculate_reprojection_error(matrix, src_pts, dst_pts)
        
        return cls.create(
            matrix=matrix,
            source_points=source_points,
            target_points=target_points,
            calibration_error=error
        )
    
    @classmethod
    def identity(cls) -> 'HomographyMatrix':
        """Create identity homography matrix."""
        return cls.create(matrix=np.eye(3, dtype=np.float64))
    
    @classmethod
    def from_array(cls, matrix_array: List[List[float]]) -> 'HomographyMatrix':
        """
        Create from 2D list/array.
        
        Args:
            matrix_array: 3x3 matrix as nested list
            
        Returns:
            HomographyMatrix instance
        """
        matrix = np.array(matrix_array, dtype=np.float64)
        return cls.create(matrix=matrix)
    
    @property
    def is_identity(self) -> bool:
        """Check if matrix is identity."""
        return np.allclose(self.matrix, np.eye(3))
    
    @property
    def determinant(self) -> float:
        """Get matrix determinant."""
        return np.linalg.det(self.matrix)
    
    @property
    def is_invertible(self) -> bool:
        """Check if matrix is invertible."""
        return abs(self.determinant) > 1e-10
    
    @property
    def condition_number(self) -> float:
        """Get matrix condition number (measure of numerical stability)."""
        return np.linalg.cond(self.matrix)
    
    @property
    def is_well_conditioned(self, threshold: float = 1e6) -> bool:
        """Check if matrix is well-conditioned."""
        return self.condition_number < threshold
    
    @property
    def calibration_quality(self) -> str:
        """Get qualitative assessment of calibration quality."""
        if self.calibration_error < 1.0:
            return "excellent"
        elif self.calibration_error < 3.0:
            return "good"
        elif self.calibration_error < 10.0:
            return "fair"
        else:
            return "poor"
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single point using the homography.
        
        Args:
            point: Input point (x, y)
            
        Returns:
            Transformed point (x', y')
        """
        x, y = point
        homogeneous_point = np.array([x, y, 1.0], dtype=np.float64)
        
        transformed = self.matrix @ homogeneous_point
        
        # Convert back from homogeneous coordinates
        if abs(transformed[2]) < 1e-10:
            raise ValueError("Point projects to infinity")
        
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])
    
    def transform_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Transform multiple points.
        
        Args:
            points: List of input points
            
        Returns:
            List of transformed points
        """
        return [self.transform_point(point) for point in points]
    
    def inverse(self) -> 'HomographyMatrix':
        """
        Get inverse homography matrix.
        
        Returns:
            Inverse HomographyMatrix
            
        Raises:
            ValueError: If matrix is not invertible
        """
        if not self.is_invertible:
            raise ValueError("Matrix is not invertible")
        
        inv_matrix = np.linalg.inv(self.matrix)
        
        # Swap source/target points for inverse
        inv_source = self.target_points if self.target_points else None
        inv_target = self.source_points if self.source_points else None
        
        return HomographyMatrix.create(
            matrix=inv_matrix,
            source_points=inv_source,
            target_points=inv_target,
            calibration_error=self.calibration_error
        )
    
    def compose_with(self, other: 'HomographyMatrix') -> 'HomographyMatrix':
        """
        Compose with another homography (matrix multiplication).
        
        Args:
            other: Other homography matrix
            
        Returns:
            Composed HomographyMatrix
        """
        composed_matrix = self.matrix @ other.matrix
        
        # Error propagation (simple addition - more sophisticated methods possible)
        combined_error = self.calibration_error + other.calibration_error
        
        return HomographyMatrix.create(
            matrix=composed_matrix,
            calibration_error=combined_error
        )
    
    def decompose(self) -> dict:
        """
        Decompose homography into rotation, translation, and scale components.
        
        Returns:
            Dictionary with decomposition components
        """
        # SVD decomposition for analysis
        u, s, vh = np.linalg.svd(self.matrix[:2, :2])
        
        # Extract approximate rotation angle
        rotation_matrix = u @ vh
        rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Extract approximate scale
        scale_x = s[0]
        scale_y = s[1]
        
        # Extract translation from last column (approximate)
        translation_x = self.matrix[0, 2]
        translation_y = self.matrix[1, 2]
        
        return {
            'rotation_angle_rad': rotation_angle,
            'rotation_angle_deg': np.degrees(rotation_angle),
            'scale_x': scale_x,
            'scale_y': scale_y,
            'translation_x': translation_x,
            'translation_y': translation_y,
            'determinant': self.determinant,
            'condition_number': self.condition_number
        }
    
    @staticmethod
    def _compute_dlt_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        Compute homography using Direct Linear Transform (DLT) algorithm.
        
        Fallback implementation when OpenCV is not available.
        """
        n = len(src_pts)
        if n < 4:
            raise ValueError("At least 4 points required for DLT")
        
        # Build the A matrix for DLT
        A = np.zeros((2 * n, 9))
        
        for i in range(n):
            x, y = src_pts[i]
            X, Y = dst_pts[i]
            
            A[2 * i] = [-x, -y, -1, 0, 0, 0, x * X, y * X, X]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * Y, y * Y, Y]
        
        # Solve using SVD
        _, _, vh = np.linalg.svd(A)
        h = vh[-1]  # Last row of V^T
        
        return h.reshape(3, 3)
    
    @staticmethod
    def _calculate_reprojection_error(
        matrix: np.ndarray, 
        src_pts: np.ndarray, 
        dst_pts: np.ndarray
    ) -> float:
        """Calculate RMS reprojection error."""
        if matrix is None:
            return float('inf')
        
        # Transform source points
        src_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
        transformed = (matrix @ src_homogeneous.T).T
        
        # Convert from homogeneous coordinates
        transformed_2d = transformed[:, :2] / transformed[:, 2:3]
        
        # Calculate RMS error
        errors = np.linalg.norm(transformed_2d - dst_pts, axis=1)
        return np.sqrt(np.mean(errors ** 2))
    
    def to_list(self) -> List[List[float]]:
        """Convert matrix to nested list."""
        return self.matrix.tolist()
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'matrix': self.to_list(),
            'source_points': self.source_points,
            'target_points': self.target_points,
            'calibration_error': self.calibration_error,
            'calibration_quality': self.calibration_quality,
            'determinant': self.determinant,
            'condition_number': self.condition_number,
            'is_well_conditioned': self.is_well_conditioned,
            'decomposition': self.decompose()
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"HomographyMatrix(det={self.determinant:.3f}, "
                f"error={self.calibration_error:.2f}, "
                f"quality={self.calibration_quality})")