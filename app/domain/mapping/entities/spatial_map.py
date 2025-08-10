"""
Spatial map entity for managing coordinate transformations and spatial relationships.

Provides domain logic for coordinate system mappings and spatial operations.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from app.domain.shared.entities.base_entity import BaseEntity
from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.mapping.value_objects.coordinates import (
    ImageCoordinates, WorldCoordinates, CoordinateTransformation
)
from app.domain.mapping.value_objects.homography_matrix import HomographyMatrix


@dataclass
class CameraView(BaseEntity):
    """
    Camera view entity representing a single camera's spatial mapping.
    
    Contains the homography matrix and calibration data for transforming
    between image coordinates and world coordinates for a specific camera.
    """
    
    camera_id: CameraID
    homography: HomographyMatrix
    image_width: int
    image_height: int
    
    # Camera physical properties
    position: Optional[WorldCoordinates] = None
    orientation_degrees: float = 0.0  # Camera orientation in degrees
    field_of_view_degrees: float = 60.0  # Horizontal field of view
    
    # Calibration metadata
    calibration_date: datetime = field(default_factory=datetime.utcnow)
    calibration_method: str = "manual"
    calibration_points_count: int = 4
    is_active: bool = True
    
    # Quality metrics
    transformation_accuracy: float = 1.0  # Overall transformation quality
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize entity after dataclass creation."""
        super().__init__()  # Initialize BaseEntity
        self._validate_camera_view()
    
    def _validate_camera_view(self) -> None:
        """Validate camera view data."""
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        
        if not (0.0 <= self.transformation_accuracy <= 1.0):
            raise ValueError("Transformation accuracy must be between 0.0 and 1.0")
        
        if not (0.0 <= self.field_of_view_degrees <= 180.0):
            raise ValueError("Field of view must be between 0 and 180 degrees")
        
        if self.calibration_points_count < 4:
            raise ValueError("At least 4 calibration points required")
    
    @classmethod
    def create(
        cls,
        camera_id: CameraID,
        homography: HomographyMatrix,
        image_width: int,
        image_height: int,
        **kwargs
    ) -> 'CameraView':
        """
        Create CameraView with validation.
        
        Args:
            camera_id: Camera identifier
            homography: Homography transformation matrix
            image_width: Image width in pixels
            image_height: Image height in pixels
            **kwargs: Additional optional parameters
            
        Returns:
            CameraView instance
        """
        return cls(
            camera_id=camera_id,
            homography=homography,
            image_width=image_width,
            image_height=image_height,
            **kwargs
        )
    
    @property
    def aspect_ratio(self) -> float:
        """Get image aspect ratio."""
        return self.image_width / self.image_height
    
    @property
    def calibration_age_days(self) -> float:
        """Get calibration age in days."""
        return (datetime.utcnow() - self.calibration_date).days
    
    @property
    def is_calibration_recent(self, max_days: int = 30) -> bool:
        """Check if calibration is recent."""
        return self.calibration_age_days <= max_days
    
    @property
    def quality_score(self) -> float:
        """
        Get overall quality score combining multiple factors.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Combine transformation accuracy, homography condition, and calibration recency
        homography_quality = 1.0 if self.homography.is_well_conditioned else 0.5
        calibration_recency = max(0.0, 1.0 - self.calibration_age_days / 365.0)  # Decay over a year
        
        return (
            self.transformation_accuracy * 0.5 +
            homography_quality * 0.3 +
            calibration_recency * 0.2
        )
    
    def transform_to_world(self, image_coords: ImageCoordinates) -> CoordinateTransformation:
        """
        Transform image coordinates to world coordinates.
        
        Args:
            image_coords: Image coordinates to transform
            
        Returns:
            CoordinateTransformation with world coordinates
        """
        # Apply homography transformation
        world_point = self.homography.transform_point(image_coords.to_tuple())
        world_coords = WorldCoordinates.from_tuple(world_point)
        
        return CoordinateTransformation.create(
            from_coords=image_coords,
            to_coords=world_coords,
            accuracy=self.transformation_accuracy,
            method="homography"
        )
    
    def transform_to_image(self, world_coords: WorldCoordinates) -> CoordinateTransformation:
        """
        Transform world coordinates to image coordinates.
        
        Args:
            world_coords: World coordinates to transform
            
        Returns:
            CoordinateTransformation with image coordinates
        """
        # Apply inverse homography transformation
        inverse_homography = self.homography.inverse()
        image_point = inverse_homography.transform_point(world_coords.to_tuple_2d())
        
        image_coords = ImageCoordinates.create(
            x=image_point[0],
            y=image_point[1],
            image_width=self.image_width,
            image_height=self.image_height
        )
        
        return CoordinateTransformation.create(
            from_coords=image_coords,
            to_coords=world_coords,
            accuracy=self.transformation_accuracy,
            method="inverse_homography"
        )
    
    def is_point_in_view(self, world_coords: WorldCoordinates) -> bool:
        """
        Check if a world coordinate point is visible in this camera view.
        
        Args:
            world_coords: World coordinates to check
            
        Returns:
            True if point is in camera view
        """
        try:
            transformation = self.transform_to_image(world_coords)
            image_coords = transformation.from_coords
            return image_coords.is_within_bounds
        except Exception:
            return False
    
    def get_coverage_area(self) -> List[WorldCoordinates]:
        """
        Get approximate world coordinates coverage area.
        
        Returns:
            List of WorldCoordinates defining the coverage polygon
        """
        # Transform image corners to world coordinates
        corners = [
            (0, 0),  # Top-left
            (self.image_width, 0),  # Top-right
            (self.image_width, self.image_height),  # Bottom-right
            (0, self.image_height)  # Bottom-left
        ]
        
        world_corners = []
        for corner in corners:
            try:
                image_coords = ImageCoordinates.create(
                    x=corner[0], y=corner[1],
                    image_width=self.image_width,
                    image_height=self.image_height
                )
                transformation = self.transform_to_world(image_coords)
                world_corners.append(transformation.to_coords)
            except Exception:
                # Handle edge cases where transformation fails
                pass
        
        return world_corners
    
    def update_homography(self, new_homography: HomographyMatrix) -> 'CameraView':
        """
        Update homography matrix.
        
        Args:
            new_homography: New homography matrix
            
        Returns:
            Updated CameraView instance
        """
        new_view = CameraView(
            camera_id=self.camera_id,
            homography=new_homography,
            image_width=self.image_width,
            image_height=self.image_height,
            position=self.position,
            orientation_degrees=self.orientation_degrees,
            field_of_view_degrees=self.field_of_view_degrees,
            calibration_date=datetime.utcnow(),  # Update calibration date
            calibration_method=self.calibration_method,
            calibration_points_count=self.calibration_points_count,
            is_active=self.is_active,
            transformation_accuracy=new_homography.calibration_error,
            metadata=self.metadata.copy()
        )
        
        new_view._id = self._id  # Keep same entity ID
        new_view.increment_version()
        return new_view
    
    def deactivate(self) -> 'CameraView':
        """
        Deactivate camera view.
        
        Returns:
            Deactivated CameraView instance
        """
        new_view = CameraView(
            camera_id=self.camera_id,
            homography=self.homography,
            image_width=self.image_width,
            image_height=self.image_height,
            position=self.position,
            orientation_degrees=self.orientation_degrees,
            field_of_view_degrees=self.field_of_view_degrees,
            calibration_date=self.calibration_date,
            calibration_method=self.calibration_method,
            calibration_points_count=self.calibration_points_count,
            is_active=False,
            transformation_accuracy=self.transformation_accuracy,
            metadata=self.metadata.copy()
        )
        
        new_view._id = self._id  # Keep same entity ID
        new_view.increment_version()
        return new_view
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': str(self.id),
            'camera_id': str(self.camera_id),
            'homography': self.homography.to_dict(),
            'image_width': self.image_width,
            'image_height': self.image_height,
            'aspect_ratio': self.aspect_ratio,
            'position': self.position.to_dict() if self.position else None,
            'orientation_degrees': self.orientation_degrees,
            'field_of_view_degrees': self.field_of_view_degrees,
            'calibration_date': self.calibration_date.isoformat(),
            'calibration_method': self.calibration_method,
            'calibration_points_count': self.calibration_points_count,
            'calibration_age_days': self.calibration_age_days,
            'is_active': self.is_active,
            'transformation_accuracy': self.transformation_accuracy,
            'quality_score': self.quality_score,
            'version': self.version,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"CameraView({self.camera_id}, "
                f"{self.image_width}x{self.image_height}, "
                f"quality={self.quality_score:.2f})")


@dataclass
class SpatialMap(BaseEntity):
    """
    Spatial map aggregate root managing multiple camera views and their relationships.
    
    Coordinates spatial operations across multiple cameras and provides
    unified spatial mapping capabilities.
    """
    
    name: str
    camera_views: Dict[str, CameraView] = field(default_factory=dict)
    
    # Map metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated_at: datetime = field(default_factory=datetime.utcnow)
    coordinate_system: str = "meters"  # World coordinate units
    origin_description: str = "Map origin"
    
    # Quality metrics
    overall_accuracy: float = 1.0
    coverage_area: Optional[List[WorldCoordinates]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize entity after dataclass creation."""
        super().__init__()  # Initialize BaseEntity
        self._validate_spatial_map()
    
    def _validate_spatial_map(self) -> None:
        """Validate spatial map data."""
        if not self.name.strip():
            raise ValueError("Spatial map name cannot be empty")
        
        if not (0.0 <= self.overall_accuracy <= 1.0):
            raise ValueError("Overall accuracy must be between 0.0 and 1.0")
        
        valid_coordinate_systems = {"meters", "feet", "pixels", "normalized"}
        if self.coordinate_system not in valid_coordinate_systems:
            raise ValueError(f"Invalid coordinate system. Must be one of: {valid_coordinate_systems}")
    
    @classmethod
    def create(cls, name: str, coordinate_system: str = "meters", **kwargs) -> 'SpatialMap':
        """
        Create SpatialMap with validation.
        
        Args:
            name: Map name
            coordinate_system: World coordinate system
            **kwargs: Additional optional parameters
            
        Returns:
            SpatialMap instance
        """
        return cls(name=name, coordinate_system=coordinate_system, **kwargs)
    
    @property
    def camera_count(self) -> int:
        """Get number of cameras in the map."""
        return len(self.camera_views)
    
    @property
    def active_camera_count(self) -> int:
        """Get number of active cameras."""
        return sum(1 for view in self.camera_views.values() if view.is_active)
    
    @property
    def has_cameras(self) -> bool:
        """Check if map has any cameras."""
        return self.camera_count > 0
    
    @property
    def average_quality(self) -> float:
        """Get average quality score across all cameras."""
        if not self.camera_views:
            return 0.0
        
        active_views = [view for view in self.camera_views.values() if view.is_active]
        if not active_views:
            return 0.0
        
        return sum(view.quality_score for view in active_views) / len(active_views)
    
    def add_camera_view(self, camera_view: CameraView) -> 'SpatialMap':
        """
        Add camera view to the spatial map.
        
        Args:
            camera_view: CameraView to add
            
        Returns:
            Updated SpatialMap instance
        """
        new_camera_views = self.camera_views.copy()
        new_camera_views[str(camera_view.camera_id)] = camera_view
        
        new_map = SpatialMap(
            name=self.name,
            camera_views=new_camera_views,
            created_at=self.created_at,
            last_updated_at=datetime.utcnow(),
            coordinate_system=self.coordinate_system,
            origin_description=self.origin_description,
            overall_accuracy=self._calculate_overall_accuracy(new_camera_views),
            coverage_area=self.coverage_area,
            metadata=self.metadata.copy()
        )
        
        new_map._id = self._id  # Keep same entity ID
        new_map.increment_version()
        return new_map
    
    def remove_camera_view(self, camera_id: CameraID) -> 'SpatialMap':
        """
        Remove camera view from the spatial map.
        
        Args:
            camera_id: Camera ID to remove
            
        Returns:
            Updated SpatialMap instance
        """
        new_camera_views = self.camera_views.copy()
        camera_key = str(camera_id)
        
        if camera_key not in new_camera_views:
            raise ValueError(f"Camera {camera_id} not found in spatial map")
        
        del new_camera_views[camera_key]
        
        new_map = SpatialMap(
            name=self.name,
            camera_views=new_camera_views,
            created_at=self.created_at,
            last_updated_at=datetime.utcnow(),
            coordinate_system=self.coordinate_system,
            origin_description=self.origin_description,
            overall_accuracy=self._calculate_overall_accuracy(new_camera_views),
            coverage_area=self.coverage_area,
            metadata=self.metadata.copy()
        )
        
        new_map._id = self._id  # Keep same entity ID
        new_map.increment_version()
        return new_map
    
    def get_camera_view(self, camera_id: CameraID) -> Optional[CameraView]:
        """
        Get camera view by ID.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            CameraView if found, None otherwise
        """
        return self.camera_views.get(str(camera_id))
    
    def has_camera_view(self, camera_id: CameraID) -> bool:
        """
        Check if camera view exists in map.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            True if camera view exists
        """
        return str(camera_id) in self.camera_views
    
    def transform_between_cameras(
        self, 
        point: ImageCoordinates, 
        from_camera: CameraID, 
        to_camera: CameraID
    ) -> Optional[ImageCoordinates]:
        """
        Transform point from one camera view to another via world coordinates.
        
        Args:
            point: Image coordinates in source camera
            from_camera: Source camera ID
            to_camera: Target camera ID
            
        Returns:
            Image coordinates in target camera, None if transformation fails
        """
        from_view = self.get_camera_view(from_camera)
        to_view = self.get_camera_view(to_camera)
        
        if from_view is None or to_view is None:
            return None
        
        try:
            # Transform to world coordinates
            world_transformation = from_view.transform_to_world(point)
            world_coords = world_transformation.to_coords
            
            # Transform to target camera
            image_transformation = to_view.transform_to_image(world_coords)
            return image_transformation.from_coords
            
        except Exception:
            return None
    
    def find_cameras_seeing_point(self, world_point: WorldCoordinates) -> List[CameraView]:
        """
        Find all cameras that can see a given world point.
        
        Args:
            world_point: World coordinates to check
            
        Returns:
            List of CameraView instances that can see the point
        """
        visible_cameras = []
        
        for camera_view in self.camera_views.values():
            if camera_view.is_active and camera_view.is_point_in_view(world_point):
                visible_cameras.append(camera_view)
        
        return visible_cameras
    
    def get_optimal_camera_for_point(self, world_point: WorldCoordinates) -> Optional[CameraView]:
        """
        Get the optimal camera for viewing a specific world point.
        
        Args:
            world_point: World coordinates
            
        Returns:
            Optimal CameraView or None if no camera can see the point
        """
        visible_cameras = self.find_cameras_seeing_point(world_point)
        
        if not visible_cameras:
            return None
        
        # Choose camera with highest quality score
        return max(visible_cameras, key=lambda cam: cam.quality_score)
    
    def _calculate_overall_accuracy(self, camera_views: Dict[str, CameraView]) -> float:
        """Calculate overall accuracy from camera views."""
        if not camera_views:
            return 0.0
        
        active_views = [view for view in camera_views.values() if view.is_active]
        if not active_views:
            return 0.0
        
        # Use weighted average based on quality scores
        total_quality = sum(view.quality_score for view in active_views)
        if total_quality == 0:
            return 0.0
        
        weighted_accuracy = sum(
            view.transformation_accuracy * view.quality_score
            for view in active_views
        ) / total_quality
        
        return weighted_accuracy
    
    def calculate_coverage_area(self) -> List[WorldCoordinates]:
        """
        Calculate overall coverage area combining all active cameras.
        
        Returns:
            List of WorldCoordinates defining the total coverage area
        """
        all_coverage_points = []
        
        for camera_view in self.camera_views.values():
            if camera_view.is_active:
                coverage = camera_view.get_coverage_area()
                all_coverage_points.extend(coverage)
        
        # In a real implementation, you would compute the union of polygons
        # For now, return all points (simplified)
        return all_coverage_points
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'camera_count': self.camera_count,
            'active_camera_count': self.active_camera_count,
            'camera_views': {
                camera_id: view.to_dict() 
                for camera_id, view in self.camera_views.items()
            },
            'created_at': self.created_at.isoformat(),
            'last_updated_at': self.last_updated_at.isoformat(),
            'coordinate_system': self.coordinate_system,
            'origin_description': self.origin_description,
            'overall_accuracy': self.overall_accuracy,
            'average_quality': self.average_quality,
            'coverage_area': [coord.to_dict() for coord in self.coverage_area] if self.coverage_area else None,
            'version': self.version,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"SpatialMap({self.name}, "
                f"{self.active_camera_count}/{self.camera_count} cameras, "
                f"quality={self.average_quality:.2f})")