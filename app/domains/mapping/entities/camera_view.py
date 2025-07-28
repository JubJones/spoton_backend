"""
Camera view entity for mapping configuration.

Contains camera view specifications and calibration data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from app.domains.mapping.entities.coordinate import CoordinateTransformation, CoordinateSystem, BoundingRegion
from app.shared.types import CameraID

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float
    distortion_coefficients: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate camera intrinsics."""
        if self.focal_length_x <= 0 or self.focal_length_y <= 0:
            raise ValueError("Focal lengths must be positive")
        
        if self.principal_point_x < 0 or self.principal_point_y < 0:
            raise ValueError("Principal point coordinates must be non-negative")
    
    @property
    def camera_matrix(self) -> List[List[float]]:
        """Get camera matrix in standard format."""
        return [
            [self.focal_length_x, 0, self.principal_point_x],
            [0, self.focal_length_y, self.principal_point_y],
            [0, 0, 1]
        ]
    
    @property
    def has_distortion(self) -> bool:
        """Check if camera has distortion parameters."""
        return len(self.distortion_coefficients) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "focal_length_x": self.focal_length_x,
            "focal_length_y": self.focal_length_y,
            "principal_point_x": self.principal_point_x,
            "principal_point_y": self.principal_point_y,
            "distortion_coefficients": self.distortion_coefficients,
            "has_distortion": self.has_distortion
        }

@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (pose)."""
    
    rotation_matrix: List[List[float]]
    translation_vector: List[float]
    
    def __post_init__(self):
        """Validate camera extrinsics."""
        # Validate rotation matrix
        if len(self.rotation_matrix) != 3:
            raise ValueError("Rotation matrix must be 3x3")
        
        for row in self.rotation_matrix:
            if len(row) != 3:
                raise ValueError("Rotation matrix must be 3x3")
        
        # Validate translation vector
        if len(self.translation_vector) != 3:
            raise ValueError("Translation vector must have 3 elements")
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Get camera position in world coordinates."""
        return tuple(self.translation_vector)
    
    @property
    def transformation_matrix(self) -> List[List[float]]:
        """Get 4x4 transformation matrix."""
        matrix = [
            [self.rotation_matrix[0][0], self.rotation_matrix[0][1], self.rotation_matrix[0][2], self.translation_vector[0]],
            [self.rotation_matrix[1][0], self.rotation_matrix[1][1], self.rotation_matrix[1][2], self.translation_vector[1]],
            [self.rotation_matrix[2][0], self.rotation_matrix[2][1], self.rotation_matrix[2][2], self.translation_vector[2]],
            [0, 0, 0, 1]
        ]
        return matrix
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rotation_matrix": self.rotation_matrix,
            "translation_vector": self.translation_vector,
            "position": self.position
        }

@dataclass
class CameraView:
    """Camera view entity with calibration and mapping information."""
    
    camera_id: CameraID
    resolution: Tuple[int, int]
    homography_transformation: CoordinateTransformation
    
    # Optional calibration data
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
    
    # View properties
    field_of_view: Optional[Tuple[float, float]] = None  # (horizontal, vertical) in degrees
    view_region: Optional[BoundingRegion] = None
    
    # Metadata
    calibration_date: Optional[datetime] = None
    calibration_accuracy: Optional[float] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate camera view."""
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError("Resolution must be positive")
        
        if self.calibration_accuracy is not None:
            if not 0 <= self.calibration_accuracy <= 1:
                raise ValueError("Calibration accuracy must be between 0 and 1")
        
        if self.field_of_view is not None:
            h_fov, v_fov = self.field_of_view
            if not (0 < h_fov <= 180 and 0 < v_fov <= 180):
                raise ValueError("Field of view must be between 0 and 180 degrees")
    
    @property
    def width(self) -> int:
        """Get camera image width."""
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        """Get camera image height."""
        return self.resolution[1]
    
    @property
    def aspect_ratio(self) -> float:
        """Get camera aspect ratio."""
        return self.width / self.height
    
    @property
    def has_full_calibration(self) -> bool:
        """Check if camera has complete calibration data."""
        return self.intrinsics is not None and self.extrinsics is not None
    
    @property
    def is_calibrated(self) -> bool:
        """Check if camera has at least basic calibration."""
        return self.homography_transformation.is_valid()
    
    def transform_to_map(self, image_coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform image coordinates to map coordinates."""
        from app.domains.mapping.entities.coordinate import Coordinate
        
        results = []
        
        for x, y in image_coordinates:
            # Create coordinate object
            coord = Coordinate(
                x=x,
                y=y,
                coordinate_system=CoordinateSystem.IMAGE,
                timestamp=datetime.now()
            )
            
            try:
                # Transform using homography
                map_coord = self.homography_transformation.transform_coordinate(coord)
                results.append((map_coord.x, map_coord.y))
            except Exception:
                # If transformation fails, append original coordinates
                results.append((x, y))
        
        return results
    
    def transform_from_map(self, map_coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform map coordinates to image coordinates."""
        from app.domains.mapping.entities.coordinate import Coordinate
        
        # Create inverse transformation
        inverse_transformation = CoordinateTransformation(
            source_system=CoordinateSystem.MAP,
            target_system=CoordinateSystem.IMAGE,
            transformation_matrix=self._invert_matrix(self.homography_transformation.transformation_matrix)
        )
        
        results = []
        
        for x, y in map_coordinates:
            # Create coordinate object
            coord = Coordinate(
                x=x,
                y=y,
                coordinate_system=CoordinateSystem.MAP,
                timestamp=datetime.now()
            )
            
            try:
                # Transform using inverse homography
                image_coord = inverse_transformation.transform_coordinate(coord)
                results.append((image_coord.x, image_coord.y))
            except Exception:
                # If transformation fails, append original coordinates
                results.append((x, y))
        
        return results
    
    def _invert_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        """Invert transformation matrix."""
        try:
            # Convert to numpy for easier inversion
            np_matrix = np.array(matrix)
            inverted = np.linalg.inv(np_matrix)
            return inverted.tolist()
        except Exception:
            # Return identity matrix if inversion fails
            size = len(matrix)
            identity = [[0.0] * size for _ in range(size)]
            for i in range(size):
                identity[i][i] = 1.0
            return identity
    
    def is_point_in_view(self, image_x: float, image_y: float) -> bool:
        """Check if point is within camera view."""
        if self.view_region:
            from app.domains.mapping.entities.coordinate import Coordinate
            
            coord = Coordinate(
                x=image_x,
                y=image_y,
                coordinate_system=CoordinateSystem.IMAGE,
                timestamp=datetime.now()
            )
            
            return self.view_region.contains(coord)
        else:
            # Use image boundaries
            return 0 <= image_x <= self.width and 0 <= image_y <= self.height
    
    def calculate_coverage_area(self) -> Optional[float]:
        """Calculate coverage area in map coordinates."""
        if not self.view_region:
            return None
        
        # Transform view region to map coordinates
        corners = [
            (self.view_region.min_coord.x, self.view_region.min_coord.y),
            (self.view_region.max_coord.x, self.view_region.min_coord.y),
            (self.view_region.max_coord.x, self.view_region.max_coord.y),
            (self.view_region.min_coord.x, self.view_region.max_coord.y)
        ]
        
        map_corners = self.transform_to_map(corners)
        
        # Calculate area using shoelace formula
        area = 0.0
        n = len(map_corners)
        
        for i in range(n):
            j = (i + 1) % n
            area += map_corners[i][0] * map_corners[j][1]
            area -= map_corners[j][0] * map_corners[i][1]
        
        return abs(area) / 2.0
    
    def get_overlapping_region(self, other: 'CameraView') -> Optional[BoundingRegion]:
        """Get overlapping region with another camera view."""
        if not self.view_region or not other.view_region:
            return None
        
        # Check if regions intersect
        if not self.view_region.intersects(other.view_region):
            return None
        
        # Calculate intersection
        from app.domains.mapping.entities.coordinate import Coordinate
        
        min_x = max(self.view_region.min_coord.x, other.view_region.min_coord.x)
        min_y = max(self.view_region.min_coord.y, other.view_region.min_coord.y)
        max_x = min(self.view_region.max_coord.x, other.view_region.max_coord.x)
        max_y = min(self.view_region.max_coord.y, other.view_region.max_coord.y)
        
        if min_x >= max_x or min_y >= max_y:
            return None
        
        min_coord = Coordinate(
            x=min_x,
            y=min_y,
            coordinate_system=self.view_region.min_coord.coordinate_system,
            timestamp=datetime.now()
        )
        
        max_coord = Coordinate(
            x=max_x,
            y=max_y,
            coordinate_system=self.view_region.max_coord.coordinate_system,
            timestamp=datetime.now()
        )
        
        return BoundingRegion(min_coord=min_coord, max_coord=max_coord)
    
    def update_calibration(
        self,
        homography_transformation: Optional[CoordinateTransformation] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
        calibration_date: Optional[datetime] = None,
        calibration_accuracy: Optional[float] = None
    ) -> 'CameraView':
        """Update camera calibration data."""
        return CameraView(
            camera_id=self.camera_id,
            resolution=self.resolution,
            homography_transformation=homography_transformation or self.homography_transformation,
            intrinsics=intrinsics or self.intrinsics,
            extrinsics=extrinsics or self.extrinsics,
            field_of_view=self.field_of_view,
            view_region=self.view_region,
            calibration_date=calibration_date or self.calibration_date,
            calibration_accuracy=calibration_accuracy or self.calibration_accuracy,
            notes=self.notes
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "camera_id": self.camera_id,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "homography_transformation": self.homography_transformation.to_dict(),
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
            "extrinsics": self.extrinsics.to_dict() if self.extrinsics else None,
            "field_of_view": self.field_of_view,
            "view_region": self.view_region.to_dict() if self.view_region else None,
            "calibration_date": self.calibration_date.isoformat() if self.calibration_date else None,
            "calibration_accuracy": self.calibration_accuracy,
            "notes": self.notes,
            "has_full_calibration": self.has_full_calibration,
            "is_calibrated": self.is_calibrated,
            "coverage_area": self.calculate_coverage_area()
        }

@dataclass
class CameraViewManager:
    """Manager for multiple camera views."""
    
    camera_views: Dict[CameraID, CameraView] = field(default_factory=dict)
    
    def add_camera_view(self, camera_view: CameraView):
        """Add a camera view."""
        self.camera_views[camera_view.camera_id] = camera_view
    
    def get_camera_view(self, camera_id: CameraID) -> Optional[CameraView]:
        """Get camera view by ID."""
        return self.camera_views.get(camera_id)
    
    def get_all_camera_views(self) -> List[CameraView]:
        """Get all camera views."""
        return list(self.camera_views.values())
    
    def get_overlapping_cameras(self, camera_id: CameraID) -> List[Tuple[CameraID, BoundingRegion]]:
        """Get cameras that overlap with specified camera."""
        camera_view = self.camera_views.get(camera_id)
        if not camera_view:
            return []
        
        overlapping = []
        
        for other_id, other_view in self.camera_views.items():
            if other_id == camera_id:
                continue
            
            overlap_region = camera_view.get_overlapping_region(other_view)
            if overlap_region:
                overlapping.append((other_id, overlap_region))
        
        return overlapping
    
    def get_total_coverage_area(self) -> float:
        """Get total coverage area of all cameras."""
        total_area = 0.0
        
        for camera_view in self.camera_views.values():
            area = camera_view.calculate_coverage_area()
            if area:
                total_area += area
        
        return total_area
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "camera_views": {
                camera_id: view.to_dict() 
                for camera_id, view in self.camera_views.items()
            },
            "total_cameras": len(self.camera_views),
            "total_coverage_area": self.get_total_coverage_area()
        }