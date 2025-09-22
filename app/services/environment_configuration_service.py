"""
Environment Configuration Service

Comprehensive environment management system providing:
- Campus and Factory environment definitions with metadata
- Camera configuration per environment with calibration data
- Zone and layout management for each environment
- Environment-specific calibration and homography data
- Multi-tenant environment isolation and access control
- Environment switching with state management
- Configuration validation and schema management
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.infrastructure.cache.tracking_cache import TrackingCache
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Types of environments."""
    CAMPUS = "campus"
    FACTORY = "factory"
    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    CUSTOM = "custom"


class CameraType(Enum):
    """Types of cameras."""
    FIXED = "fixed"
    PTZ = "ptz"
    DOME = "dome"
    BULLET = "bullet"
    FISHEYE = "fisheye"


class ZoneType(Enum):
    """Types of zones in environments."""
    ENTRANCE = "entrance"
    EXIT = "exit"
    WAITING = "waiting"
    RESTRICTED = "restricted"
    MAIN_AREA = "main_area"
    CORRIDOR = "corridor"
    QUEUE = "queue"
    PARKING = "parking"


@dataclass
class CameraConfiguration:
    """Camera configuration for an environment."""
    camera_id: str
    name: str
    camera_type: CameraType
    position: Coordinate  # Physical position in environment
    resolution: Tuple[int, int]  # (width, height)
    field_of_view: float  # degrees
    orientation: float  # rotation angle in degrees
    
    # Technical specifications
    frame_rate: float
    exposure_settings: Dict[str, Any]
    focus_settings: Dict[str, Any]
    
    # Calibration data
    intrinsic_matrix: Optional[List[List[float]]] = None
    distortion_coefficients: Optional[List[float]] = None
    homography_matrix: Optional[List[List[float]]] = None
    
    # Status and metadata
    is_active: bool = True
    last_calibrated: Optional[datetime] = None
    calibration_accuracy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'camera_type': self.camera_type.value,
            'position': {'x': self.position.x, 'y': self.position.y},
            'resolution': list(self.resolution),
            'field_of_view': self.field_of_view,
            'orientation': self.orientation,
            'frame_rate': self.frame_rate,
            'exposure_settings': self.exposure_settings,
            'focus_settings': self.focus_settings,
            'intrinsic_matrix': self.intrinsic_matrix,
            'distortion_coefficients': self.distortion_coefficients,
            'homography_matrix': self.homography_matrix,
            'is_active': self.is_active,
            'last_calibrated': self.last_calibrated.isoformat() if self.last_calibrated else None,
            'calibration_accuracy': self.calibration_accuracy,
            'metadata': self.metadata
        }


@dataclass
class ZoneDefinition:
    """Zone definition within an environment."""
    zone_id: str
    name: str
    zone_type: ZoneType
    boundary_points: List[Coordinate]  # Polygon vertices
    
    # Zone properties
    capacity_limit: Optional[int] = None
    access_restrictions: List[str] = field(default_factory=list)
    monitoring_enabled: bool = True
    
    # Alert thresholds
    occupancy_threshold: Optional[int] = None
    dwell_time_threshold: Optional[float] = None  # seconds
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def contains_point(self, coordinate: Coordinate) -> bool:
        """Check if coordinate is within zone boundary using ray casting."""
        if len(self.boundary_points) < 3:
            return False
        
        x, y = coordinate.x, coordinate.y
        inside = False
        j = len(self.boundary_points) - 1
        
        for i in range(len(self.boundary_points)):
            xi, yi = self.boundary_points[i].x, self.boundary_points[i].y
            xj, yj = self.boundary_points[j].x, self.boundary_points[j].y
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def get_area(self) -> float:
        """Calculate zone area using shoelace formula."""
        if len(self.boundary_points) < 3:
            return 0.0
        
        area = 0.0
        n = len(self.boundary_points)
        
        for i in range(n):
            j = (i + 1) % n
            area += self.boundary_points[i].x * self.boundary_points[j].y
            area -= self.boundary_points[j].x * self.boundary_points[i].y
        
        return abs(area) / 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'zone_id': self.zone_id,
            'name': self.name,
            'zone_type': self.zone_type.value,
            'boundary_points': [{'x': p.x, 'y': p.y} for p in self.boundary_points],
            'capacity_limit': self.capacity_limit,
            'access_restrictions': self.access_restrictions,
            'monitoring_enabled': self.monitoring_enabled,
            'occupancy_threshold': self.occupancy_threshold,
            'dwell_time_threshold': self.dwell_time_threshold,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'area': self.get_area(),
            'metadata': self.metadata
        }


@dataclass
class EnvironmentLayout:
    """Physical layout configuration for an environment."""
    layout_id: str
    name: str
    
    # Physical dimensions
    bounds: Tuple[Coordinate, Coordinate]  # (min_coord, max_coord)
    scale_meters_per_pixel: float
    
    # Reference points for coordinate system
    reference_points: List[Tuple[Coordinate, str]] = field(default_factory=list)  # (coordinate, description)
    
    # Layout elements
    walls: List[Tuple[Coordinate, Coordinate]] = field(default_factory=list)  # Wall segments
    doors: List[Tuple[Coordinate, str]] = field(default_factory=list)  # (position, name)
    landmarks: List[Tuple[Coordinate, str]] = field(default_factory=list)  # (position, description)
    
    # Visual representation
    floor_plan_image_path: Optional[str] = None
    overlay_image_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'layout_id': self.layout_id,
            'name': self.name,
            'bounds': {
                'min': {'x': self.bounds[0].x, 'y': self.bounds[0].y},
                'max': {'x': self.bounds[1].x, 'y': self.bounds[1].y}
            },
            'scale_meters_per_pixel': self.scale_meters_per_pixel,
            'reference_points': [
                {'coordinate': {'x': coord.x, 'y': coord.y}, 'description': desc}
                for coord, desc in self.reference_points
            ],
            'walls': [
                {'start': {'x': start.x, 'y': start.y}, 'end': {'x': end.x, 'y': end.y}}
                for start, end in self.walls
            ],
            'doors': [
                {'position': {'x': pos.x, 'y': pos.y}, 'name': name}
                for pos, name in self.doors
            ],
            'landmarks': [
                {'position': {'x': pos.x, 'y': pos.y}, 'description': desc}
                for pos, desc in self.landmarks
            ],
            'floor_plan_image_path': self.floor_plan_image_path,
            'overlay_image_path': self.overlay_image_path
        }


@dataclass
class EnvironmentConfiguration:
    """Complete environment configuration."""
    environment_id: str
    name: str
    environment_type: EnvironmentType
    description: str
    
    # Configuration components
    cameras: Dict[str, CameraConfiguration] = field(default_factory=dict)
    zones: Dict[str, ZoneDefinition] = field(default_factory=dict)
    layout: Optional[EnvironmentLayout] = None
    
    # Environment settings
    timezone: str = "UTC"
    operating_hours: Dict[str, Any] = field(default_factory=dict)  # Start/end times per day
    capacity_limits: Dict[str, int] = field(default_factory=dict)  # Global and zone limits
    
    # Data settings
    data_retention_days: int = 90
    recording_enabled: bool = True
    analytics_enabled: bool = True
    
    # Access control
    user_permissions: Dict[str, List[str]] = field(default_factory=dict)  # user_id -> permissions
    api_keys: List[str] = field(default_factory=list)
    
    # Status and metadata
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_active_cameras(self) -> List[CameraConfiguration]:
        """Get list of active cameras."""
        return [camera for camera in self.cameras.values() if camera.is_active]
    
    def get_zones_by_type(self, zone_type: ZoneType) -> List[ZoneDefinition]:
        """Get zones of specific type."""
        return [zone for zone in self.zones.values() if zone.zone_type == zone_type]
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate environment configuration and return any issues."""
        issues = defaultdict(list)
        
        # Check cameras
        if not self.cameras:
            issues['cameras'].append("No cameras configured")
        else:
            for camera_id, camera in self.cameras.items():
                if not camera.homography_matrix:
                    issues['cameras'].append(f"Camera {camera_id} missing homography matrix")
                
                if not camera.last_calibrated:
                    issues['cameras'].append(f"Camera {camera_id} never calibrated")
                elif (datetime.utcnow() - camera.last_calibrated).days > 30:
                    issues['cameras'].append(f"Camera {camera_id} calibration is outdated")
        
        # Check zones
        if not self.zones:
            issues['zones'].append("No zones configured")
        else:
            for zone_id, zone in self.zones.items():
                if len(zone.boundary_points) < 3:
                    issues['zones'].append(f"Zone {zone_id} has insufficient boundary points")
                
                if zone.get_area() == 0:
                    issues['zones'].append(f"Zone {zone_id} has zero area")
        
        # Check layout
        if not self.layout:
            issues['layout'].append("No layout configuration")
        
        return dict(issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'environment_id': self.environment_id,
            'name': self.name,
            'environment_type': self.environment_type.value,
            'description': self.description,
            'cameras': {cid: camera.to_dict() for cid, camera in self.cameras.items()},
            'zones': {zid: zone.to_dict() for zid, zone in self.zones.items()},
            'layout': self.layout.to_dict() if self.layout else None,
            'timezone': self.timezone,
            'operating_hours': self.operating_hours,
            'capacity_limits': self.capacity_limits,
            'data_retention_days': self.data_retention_days,
            'recording_enabled': self.recording_enabled,
            'analytics_enabled': self.analytics_enabled,
            'user_permissions': self.user_permissions,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'version': self.version,
            'validation_issues': self.validate_configuration(),
            'metadata': self.metadata
        }


class EnvironmentConfigurationService:
    """Comprehensive service for environment configuration management."""
    
    def __init__(
        self,
        tracking_cache: TrackingCache,
        tracking_repository: TrackingRepository,
        config_path: Optional[str] = None
    ):
        self.cache = tracking_cache
        self.repository = tracking_repository
        
        # Configuration storage
        self.config_path = Path(config_path or "config/environments")
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        # Environment configurations
        self.environments: Dict[str, EnvironmentConfiguration] = {}
        
        # Default configurations
        self.default_camera_settings = {
            'frame_rate': 25.0,
            'exposure_settings': {'auto': True, 'value': 0},
            'focus_settings': {'auto': True, 'distance': 0}
        }
        
        # Performance tracking
        self.config_stats = {
            'environments_loaded': 0,
            'configurations_validated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_load_time_ms': 0.0
        }
        
        # Initialize default environments
        asyncio.create_task(self._initialize_default_environments())
        
        logger.info("EnvironmentConfigurationService initialized")
    
    def _create_coordinate(self, x: float, y: float, coordinate_system: CoordinateSystem = CoordinateSystem.MAP) -> Coordinate:
        """Helper method to create Coordinate objects with required arguments."""
        return Coordinate(
            x=x,
            y=y,
            coordinate_system=coordinate_system,
            timestamp=datetime.utcnow()
        )
    
    # --- Environment Management ---
    
    async def create_environment(
        self,
        environment_id: str,
        name: str,
        environment_type: EnvironmentType,
        description: str = "",
        **kwargs
    ) -> EnvironmentConfiguration:
        """Create new environment configuration."""
        try:
            if environment_id in self.environments:
                raise ValueError(f"Environment {environment_id} already exists")
            
            # Create environment configuration
            environment = EnvironmentConfiguration(
                environment_id=environment_id,
                name=name,
                environment_type=environment_type,
                description=description,
                **kwargs
            )
            
            # Store configuration
            self.environments[environment_id] = environment
            
            # Persist to storage
            await self._save_environment_config(environment)
            
            # Cache configuration
            await self._cache_environment_config(environment)
            
            logger.info(f"Created environment {environment_id}: {name}")
            return environment
            
        except Exception as e:
            logger.error(f"Error creating environment: {e}")
            raise
    
    async def get_environment(self, environment_id: str) -> Optional[EnvironmentConfiguration]:
        """Get environment configuration by ID."""
        try:
            start_time = time.time()
            
            # Check memory cache first
            if environment_id in self.environments:
                self.config_stats['cache_hits'] += 1
                return self.environments[environment_id]
            
            # Check Redis cache
            cached_config = await self._get_cached_environment_config(environment_id)
            if cached_config:
                self.environments[environment_id] = cached_config
                self.config_stats['cache_hits'] += 1
                return cached_config
            
            # Load from storage
            environment = await self._load_environment_config(environment_id)
            if environment:
                self.environments[environment_id] = environment
                await self._cache_environment_config(environment)
            
            # Update metrics
            load_time = (time.time() - start_time) * 1000
            self._update_load_time_metrics(load_time)
            self.config_stats['cache_misses'] += 1
            
            return environment
            
        except Exception as e:
            logger.error(f"Error getting environment {environment_id}: {e}")
            return None
    
    async def list_environments(
        self,
        environment_type: Optional[EnvironmentType] = None,
        active_only: bool = True
    ) -> List[EnvironmentConfiguration]:
        """List all environment configurations."""
        try:
            # Ensure all environments are loaded
            await self._load_all_environments()
            
            environments = list(self.environments.values())
            
            # Filter by type
            if environment_type:
                environments = [env for env in environments if env.environment_type == environment_type]
            
            # Filter by active status
            if active_only:
                environments = [env for env in environments if env.is_active]
            
            # Sort by name
            environments.sort(key=lambda env: env.name)
            
            return environments
            
        except Exception as e:
            logger.error(f"Error listing environments: {e}")
            return []
    
    async def update_environment(
        self,
        environment_id: str,
        **updates
    ) -> Optional[EnvironmentConfiguration]:
        """Update environment configuration."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment:
                return None
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(environment, key):
                    setattr(environment, key, value)
            
            # Update timestamp
            environment.last_updated = datetime.utcnow()
            
            # Persist changes
            await self._save_environment_config(environment)
            await self._cache_environment_config(environment)
            
            logger.info(f"Updated environment {environment_id}")
            return environment
            
        except Exception as e:
            logger.error(f"Error updating environment: {e}")
            return None
    
    async def delete_environment(self, environment_id: str) -> bool:
        """Delete environment configuration."""
        try:
            if environment_id not in self.environments:
                return False
            
            # Remove from memory
            del self.environments[environment_id]
            
            # Remove from storage
            config_file = self.config_path / f"{environment_id}.json"
            if config_file.exists():
                config_file.unlink()
            
            # Remove from cache
            cache_key = f"environment_config_{environment_id}"
            await self.cache.delete(cache_key)
            
            logger.info(f"Deleted environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting environment: {e}")
            return False
    
    # --- Camera Management ---
    
    async def add_camera(
        self,
        environment_id: str,
        camera: CameraConfiguration
    ) -> bool:
        """Add camera to environment."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment:
                return False
            
            environment.cameras[camera.camera_id] = camera
            environment.last_updated = datetime.utcnow()
            
            await self._save_environment_config(environment)
            await self._cache_environment_config(environment)
            
            logger.info(f"Added camera {camera.camera_id} to environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return False
    
    async def update_camera(
        self,
        environment_id: str,
        camera_id: str,
        **updates
    ) -> bool:
        """Update camera configuration."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment or camera_id not in environment.cameras:
                return False
            
            camera = environment.cameras[camera_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(camera, key):
                    setattr(camera, key, value)
            
            environment.last_updated = datetime.utcnow()
            
            await self._save_environment_config(environment)
            await self._cache_environment_config(environment)
            
            logger.info(f"Updated camera {camera_id} in environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating camera: {e}")
            return False
    
    async def remove_camera(self, environment_id: str, camera_id: str) -> bool:
        """Remove camera from environment."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment or camera_id not in environment.cameras:
                return False
            
            del environment.cameras[camera_id]
            environment.last_updated = datetime.utcnow()
            
            await self._save_environment_config(environment)
            await self._cache_environment_config(environment)
            
            logger.info(f"Removed camera {camera_id} from environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing camera: {e}")
            return False
    
    # --- Zone Management ---
    
    async def add_zone(
        self,
        environment_id: str,
        zone: ZoneDefinition
    ) -> bool:
        """Add zone to environment."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment:
                return False
            
            environment.zones[zone.zone_id] = zone
            environment.last_updated = datetime.utcnow()
            
            await self._save_environment_config(environment)
            await self._cache_environment_config(environment)
            
            logger.info(f"Added zone {zone.zone_id} to environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding zone: {e}")
            return False
    
    async def update_zone(
        self,
        environment_id: str,
        zone_id: str,
        **updates
    ) -> bool:
        """Update zone configuration."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment or zone_id not in environment.zones:
                return False
            
            zone = environment.zones[zone_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(zone, key):
                    if key == 'boundary_points' and isinstance(value, list):
                        # Convert dict coordinates to Coordinate objects
                        zone.boundary_points = [
                            self._create_coordinate(x=p['x'], y=p['y']) if isinstance(p, dict) else p
                            for p in value
                        ]
                    else:
                        setattr(zone, key, value)
            
            environment.last_updated = datetime.utcnow()
            
            await self._save_environment_config(environment)
            await self._cache_environment_config(environment)
            
            logger.info(f"Updated zone {zone_id} in environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating zone: {e}")
            return False
    
    async def remove_zone(self, environment_id: str, zone_id: str) -> bool:
        """Remove zone from environment."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment or zone_id not in environment.zones:
                return False
            
            del environment.zones[zone_id]
            environment.last_updated = datetime.utcnow()
            
            await self._save_environment_config(environment)
            await self._cache_environment_config(environment)
            
            logger.info(f"Removed zone {zone_id} from environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing zone: {e}")
            return False
    
    # --- Configuration Validation ---
    
    async def validate_environment(self, environment_id: str) -> Dict[str, Any]:
        """Validate environment configuration."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment:
                return {'error': 'Environment not found'}
            
            validation_result = environment.validate_configuration()
            self.config_stats['configurations_validated'] += 1
            
            return {
                'environment_id': environment_id,
                'validation_passed': len(validation_result) == 0,
                'issues': validation_result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating environment: {e}")
            return {'error': str(e)}
    
    async def validate_all_environments(self) -> Dict[str, Dict[str, Any]]:
        """Validate all environment configurations."""
        try:
            await self._load_all_environments()
            
            validation_results = {}
            for environment_id in self.environments.keys():
                validation_results[environment_id] = await self.validate_environment(environment_id)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating all environments: {e}")
            return {'error': str(e)}
    
    # --- Data Access Methods ---
    
    async def get_available_date_ranges(
        self,
        environment_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get available data date ranges per environment."""
        try:
            if environment_id:
                environments = [environment_id]
            else:
                env_configs = await self.list_environments()
                environments = [env.environment_id for env in env_configs]
            
            date_ranges = {}
            
            for env_id in environments:
                try:
                    # Query database for available data ranges
                    env_range = await self.repository.get_available_date_ranges(env_id)
                    date_ranges[env_id] = env_range
                except Exception as e:
                    logger.warning(f"Could not get date ranges for {env_id}: {e}")
                    date_ranges[env_id] = {
                        'earliest_date': None,
                        'latest_date': None,
                        'total_days': 0,
                        'has_data': False
                    }
            
            return date_ranges
            
        except Exception as e:
            logger.error(f"Error getting available date ranges: {e}")
            return {}
    
    async def get_environment_metadata(self, environment_id: str) -> Dict[str, Any]:
        """Get comprehensive environment metadata."""
        try:
            environment = await self.get_environment(environment_id)
            if not environment:
                return {}
            
            # Get date ranges
            date_ranges = await self.get_available_date_ranges(environment_id)
            
            # Calculate statistics
            active_cameras = environment.get_active_cameras()
            zones_by_type = defaultdict(int)
            total_zone_area = 0.0
            
            for zone in environment.zones.values():
                zones_by_type[zone.zone_type.value] += 1
                total_zone_area += zone.get_area()
            
            return {
                'environment_id': environment_id,
                'name': environment.name,
                'type': environment.environment_type.value,
                'description': environment.description,
                'is_active': environment.is_active,
                'cameras': {
                    'total': len(environment.cameras),
                    'active': len(active_cameras),
                    'by_type': defaultdict(int)
                },
                'zones': {
                    'total': len(environment.zones),
                    'by_type': dict(zones_by_type),
                    'total_area': total_zone_area
                },
                'data_availability': date_ranges.get(environment_id, {}),
                'settings': {
                    'timezone': environment.timezone,
                    'data_retention_days': environment.data_retention_days,
                    'recording_enabled': environment.recording_enabled,
                    'analytics_enabled': environment.analytics_enabled
                },
                'validation': environment.validate_configuration(),
                'last_updated': environment.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting environment metadata: {e}")
            return {}
    
    # --- Storage and Caching ---
    
    async def _save_environment_config(self, environment: EnvironmentConfiguration):
        """Save environment configuration to storage."""
        try:
            config_file = self.config_path / f"{environment.environment_id}.json"
            
            # Convert to serializable format
            config_data = environment.to_dict()
            
            # Write to file
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving environment config: {e}")
    
    async def _load_environment_config(self, environment_id: str) -> Optional[EnvironmentConfiguration]:
        """Load environment configuration from storage."""
        try:
            config_file = self.config_path / f"{environment_id}.json"
            
            if not config_file.exists():
                return None
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert back to objects
            environment = self._dict_to_environment(config_data)
            return environment
            
        except Exception as e:
            logger.error(f"Error loading environment config: {e}")
            return None
    
    async def _load_all_environments(self):
        """Load all environment configurations from storage."""
        try:
            config_files = list(self.config_path.glob("*.json"))
            
            for config_file in config_files:
                environment_id = config_file.stem
                if environment_id not in self.environments:
                    environment = await self._load_environment_config(environment_id)
                    if environment:
                        self.environments[environment_id] = environment
            
            self.config_stats['environments_loaded'] = len(self.environments)
            
        except Exception as e:
            logger.error(f"Error loading all environments: {e}")
    
    async def _cache_environment_config(self, environment: EnvironmentConfiguration):
        """Cache environment configuration in Redis."""
        try:
            cache_key = f"environment_config_{environment.environment_id}"
            config_data = environment.to_dict()
            
            await self.cache.set_json(cache_key, config_data, ttl=3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Error caching environment config: {e}")
    
    async def _get_cached_environment_config(self, environment_id: str) -> Optional[EnvironmentConfiguration]:
        """Get environment configuration from cache."""
        try:
            cache_key = f"environment_config_{environment_id}"
            config_data = await self.cache.get_json(cache_key)
            
            if config_data:
                return self._dict_to_environment(config_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached environment config: {e}")
            return None
    
    def _dict_to_environment(self, config_data: Dict[str, Any]) -> EnvironmentConfiguration:
        """Convert dictionary to EnvironmentConfiguration object."""
        # Convert cameras
        cameras = {}
        for camera_id, camera_data in config_data.get('cameras', {}).items():
            cameras[camera_id] = self._dict_to_camera(camera_data)
        
        # Convert zones
        zones = {}
        for zone_id, zone_data in config_data.get('zones', {}).items():
            zones[zone_id] = self._dict_to_zone(zone_data)
        
        # Convert layout
        layout = None
        if config_data.get('layout'):
            layout = self._dict_to_layout(config_data['layout'])
        
        # Create environment
        environment = EnvironmentConfiguration(
            environment_id=config_data['environment_id'],
            name=config_data['name'],
            environment_type=EnvironmentType(config_data['environment_type']),
            description=config_data.get('description', ''),
            cameras=cameras,
            zones=zones,
            layout=layout,
            timezone=config_data.get('timezone', 'UTC'),
            operating_hours=config_data.get('operating_hours', {}),
            capacity_limits=config_data.get('capacity_limits', {}),
            data_retention_days=config_data.get('data_retention_days', 90),
            recording_enabled=config_data.get('recording_enabled', True),
            analytics_enabled=config_data.get('analytics_enabled', True),
            user_permissions=config_data.get('user_permissions', {}),
            api_keys=config_data.get('api_keys', []),
            is_active=config_data.get('is_active', True),
            created_at=datetime.fromisoformat(config_data.get('created_at', datetime.utcnow().isoformat())),
            last_updated=datetime.fromisoformat(config_data.get('last_updated', datetime.utcnow().isoformat())),
            version=config_data.get('version', '1.0'),
            metadata=config_data.get('metadata', {})
        )
        
        return environment
    
    def _dict_to_camera(self, camera_data: Dict[str, Any]) -> CameraConfiguration:
        """Convert dictionary to CameraConfiguration object."""
        position = self._create_coordinate(
            x=camera_data['position']['x'],
            y=camera_data['position']['y']
        )
        
        last_calibrated = None
        if camera_data.get('last_calibrated'):
            last_calibrated = datetime.fromisoformat(camera_data['last_calibrated'])
        
        return CameraConfiguration(
            camera_id=camera_data['camera_id'],
            name=camera_data['name'],
            camera_type=CameraType(camera_data['camera_type']),
            position=position,
            resolution=tuple(camera_data['resolution']),
            field_of_view=camera_data['field_of_view'],
            orientation=camera_data['orientation'],
            frame_rate=camera_data['frame_rate'],
            exposure_settings=camera_data['exposure_settings'],
            focus_settings=camera_data['focus_settings'],
            intrinsic_matrix=camera_data.get('intrinsic_matrix'),
            distortion_coefficients=camera_data.get('distortion_coefficients'),
            homography_matrix=camera_data.get('homography_matrix'),
            is_active=camera_data.get('is_active', True),
            last_calibrated=last_calibrated,
            calibration_accuracy=camera_data.get('calibration_accuracy'),
            metadata=camera_data.get('metadata', {})
        )
    
    def _dict_to_zone(self, zone_data: Dict[str, Any]) -> ZoneDefinition:
        """Convert dictionary to ZoneDefinition object."""
        boundary_points = [
            self._create_coordinate(x=point['x'], y=point['y'])
            for point in zone_data['boundary_points']
        ]
        
        created_at = datetime.fromisoformat(zone_data.get('created_at', datetime.utcnow().isoformat()))
        
        return ZoneDefinition(
            zone_id=zone_data['zone_id'],
            name=zone_data['name'],
            zone_type=ZoneType(zone_data['zone_type']),
            boundary_points=boundary_points,
            capacity_limit=zone_data.get('capacity_limit'),
            access_restrictions=zone_data.get('access_restrictions', []),
            monitoring_enabled=zone_data.get('monitoring_enabled', True),
            occupancy_threshold=zone_data.get('occupancy_threshold'),
            dwell_time_threshold=zone_data.get('dwell_time_threshold'),
            description=zone_data.get('description', ''),
            created_at=created_at,
            metadata=zone_data.get('metadata', {})
        )
    
    def _dict_to_layout(self, layout_data: Dict[str, Any]) -> EnvironmentLayout:
        """Convert dictionary to EnvironmentLayout object."""
        bounds = (
            self._create_coordinate(x=layout_data['bounds']['min']['x'], y=layout_data['bounds']['min']['y']),
            self._create_coordinate(x=layout_data['bounds']['max']['x'], y=layout_data['bounds']['max']['y'])
        )
        
        reference_points = [
            (self._create_coordinate(x=rp['coordinate']['x'], y=rp['coordinate']['y']), rp['description'])
            for rp in layout_data.get('reference_points', [])
        ]
        
        walls = [
            (self._create_coordinate(x=wall['start']['x'], y=wall['start']['y']),
             self._create_coordinate(x=wall['end']['x'], y=wall['end']['y']))
            for wall in layout_data.get('walls', [])
        ]
        
        doors = [
            (self._create_coordinate(x=door['position']['x'], y=door['position']['y']), door['name'])
            for door in layout_data.get('doors', [])
        ]
        
        landmarks = [
            (self._create_coordinate(x=lm['position']['x'], y=lm['position']['y']), lm['description'])
            for lm in layout_data.get('landmarks', [])
        ]
        
        return EnvironmentLayout(
            layout_id=layout_data['layout_id'],
            name=layout_data['name'],
            bounds=bounds,
            scale_meters_per_pixel=layout_data['scale_meters_per_pixel'],
            reference_points=reference_points,
            walls=walls,
            doors=doors,
            landmarks=landmarks,
            floor_plan_image_path=layout_data.get('floor_plan_image_path'),
            overlay_image_path=layout_data.get('overlay_image_path')
        )
    
    # --- Default Environments Initialization ---
    
    async def _initialize_default_environments(self):
        """Initialize default environment configurations."""
        try:
            # Campus environment
            if "campus" not in self.environments:
                await self._create_campus_environment()
            
            # Factory environment
            if "factory" not in self.environments:
                await self._create_factory_environment()
            
        except Exception as e:
            logger.error(f"Error initializing default environments: {e}")
    
    async def _create_campus_environment(self):
        """Create default campus environment."""
        try:
            # Create cameras
            cameras = {}
            
            camera_configs = [
                ("c09", "Campus Entry Camera", (5.0, 2.0)),
                ("c12", "Campus Main Area Camera", (25.0, 15.0)),
                ("c13", "Campus Corridor Camera", (45.0, 10.0)),
                ("c16", "Campus Exit Camera", (65.0, 2.0))
            ]
            
            for cam_id, cam_name, (x, y) in camera_configs:
                cameras[cam_id] = CameraConfiguration(
                    camera_id=cam_id,
                    name=cam_name,
                    camera_type=CameraType.FIXED,
                    position=self._create_coordinate(x=x, y=y),
                    resolution=(1920, 1080),
                    field_of_view=70.0,
                    orientation=0.0,
                    **self.default_camera_settings,
                    last_calibrated=datetime.utcnow() - timedelta(days=1)
                )
            
            # Create zones
            zones = {}
            
            zone_configs = [
                ("entrance", "Campus Entrance", ZoneType.ENTRANCE, [(0, 0), (10, 0), (10, 5), (0, 5)], 8),
                ("main_area", "Campus Main Area", ZoneType.MAIN_AREA, [(10, 0), (60, 0), (60, 30), (10, 30)], 50),
                ("corridor", "Campus Corridor", ZoneType.CORRIDOR, [(60, 5), (70, 5), (70, 15), (60, 15)], 15),
                ("exit", "Campus Exit", ZoneType.EXIT, [(60, 0), (70, 0), (70, 5), (60, 5)], 8)
            ]
            
            for zone_id, zone_name, zone_type, boundary, capacity in zone_configs:
                zones[zone_id] = ZoneDefinition(
                    zone_id=zone_id,
                    name=zone_name,
                    zone_type=zone_type,
                    boundary_points=[self._create_coordinate(x=x, y=y) for x, y in boundary],
                    capacity_limit=capacity,
                    monitoring_enabled=True,
                    occupancy_threshold=int(capacity * 0.8),
                    description=f"Default {zone_name.lower()} zone for campus environment"
                )
            
            # Create layout
            layout = EnvironmentLayout(
                layout_id="campus_layout",
                name="Campus Layout",
                bounds=(self._create_coordinate(x=0, y=0), self._create_coordinate(x=70, y=30)),
                scale_meters_per_pixel=0.1,
                reference_points=[
                    (self._create_coordinate(x=0, y=0), "Origin point"),
                    (self._create_coordinate(x=35, y=15), "Center point")
                ]
            )
            
            # Create environment
            campus_env = EnvironmentConfiguration(
                environment_id="campus",
                name="Campus Environment",
                environment_type=EnvironmentType.CAMPUS,
                description="Default campus environment with multiple zones and cameras",
                cameras=cameras,
                zones=zones,
                layout=layout,
                timezone="UTC",
                operating_hours={
                    "monday": {"start": "06:00", "end": "22:00"},
                    "tuesday": {"start": "06:00", "end": "22:00"},
                    "wednesday": {"start": "06:00", "end": "22:00"},
                    "thursday": {"start": "06:00", "end": "22:00"},
                    "friday": {"start": "06:00", "end": "22:00"},
                    "saturday": {"start": "08:00", "end": "18:00"},
                    "sunday": {"start": "10:00", "end": "16:00"}
                },
                capacity_limits={"total": 100, "entrance": 8, "main_area": 50},
                data_retention_days=180,  # Longer retention for campus
                analytics_enabled=True
            )
            
            self.environments["campus"] = campus_env
            await self._save_environment_config(campus_env)
            
            logger.info("Created default campus environment")
            
        except Exception as e:
            logger.error(f"Error creating campus environment: {e}")
    
    async def _create_factory_environment(self):
        """Create default factory environment."""
        try:
            # Create cameras
            cameras = {}
            
            camera_configs = [
                ("f01", "Factory Line 1 Camera", (10.0, 5.0)),
                ("f02", "Factory Line 2 Camera", (30.0, 5.0)),
                ("f03", "Factory QC Camera", (50.0, 5.0)),
                ("f04", "Factory Exit Camera", (70.0, 2.0))
            ]
            
            for cam_id, cam_name, (x, y) in camera_configs:
                cameras[cam_id] = CameraConfiguration(
                    camera_id=cam_id,
                    name=cam_name,
                    camera_type=CameraType.FIXED,
                    position=self._create_coordinate(x=x, y=y),
                    resolution=(1920, 1080),
                    field_of_view=60.0,
                    orientation=0.0,
                    **self.default_camera_settings,
                    last_calibrated=datetime.utcnow() - timedelta(days=2)
                )
            
            # Create zones
            zones = {}
            
            zone_configs = [
                ("production_line_1", "Production Line 1", ZoneType.MAIN_AREA, [(5, 0), (25, 0), (25, 10), (5, 10)], 12),
                ("production_line_2", "Production Line 2", ZoneType.MAIN_AREA, [(25, 0), (45, 0), (45, 10), (25, 10)], 12),
                ("quality_control", "Quality Control", ZoneType.RESTRICTED, [(45, 0), (65, 0), (65, 10), (45, 10)], 5),
                ("factory_exit", "Factory Exit", ZoneType.EXIT, [(65, 0), (75, 0), (75, 5), (65, 5)], 6)
            ]
            
            for zone_id, zone_name, zone_type, boundary, capacity in zone_configs:
                zones[zone_id] = ZoneDefinition(
                    zone_id=zone_id,
                    name=zone_name,
                    zone_type=zone_type,
                    boundary_points=[self._create_coordinate(x=x, y=y) for x, y in boundary],
                    capacity_limit=capacity,
                    monitoring_enabled=True,
                    occupancy_threshold=int(capacity * 0.9),  # Tighter control in factory
                    description=f"Default {zone_name.lower()} zone for factory environment"
                )
            
            # Create layout
            layout = EnvironmentLayout(
                layout_id="factory_layout",
                name="Factory Floor Layout",
                bounds=(self._create_coordinate(x=0, y=0), self._create_coordinate(x=75, y=10)),
                scale_meters_per_pixel=0.1,
                reference_points=[
                    (self._create_coordinate(x=0, y=0), "Factory entrance"),
                    (self._create_coordinate(x=37.5, y=5), "Factory center")
                ]
            )
            
            # Create environment
            factory_env = EnvironmentConfiguration(
                environment_id="factory",
                name="Factory Environment",
                environment_type=EnvironmentType.FACTORY,
                description="Default factory environment with production lines and quality control",
                cameras=cameras,
                zones=zones,
                layout=layout,
                timezone="UTC",
                operating_hours={
                    "monday": {"start": "06:00", "end": "18:00"},
                    "tuesday": {"start": "06:00", "end": "18:00"},
                    "wednesday": {"start": "06:00", "end": "18:00"},
                    "thursday": {"start": "06:00", "end": "18:00"},
                    "friday": {"start": "06:00", "end": "18:00"}
                },
                capacity_limits={"total": 40, "production_line_1": 12, "production_line_2": 12},
                data_retention_days=60,  # Shorter retention for factory
                analytics_enabled=True
            )
            
            self.environments["factory"] = factory_env
            await self._save_environment_config(factory_env)
            
            logger.info("Created default factory environment")
            
        except Exception as e:
            logger.error(f"Error creating factory environment: {e}")
    
    # --- Utility Methods ---
    
    def _update_load_time_metrics(self, load_time_ms: float):
        """Update load time performance metrics."""
        current_avg = self.config_stats['avg_load_time_ms']
        total_loads = self.config_stats['cache_misses']
        
        if total_loads > 0:
            self.config_stats['avg_load_time_ms'] = (
                (current_avg * (total_loads - 1) + load_time_ms) / total_loads
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'EnvironmentConfigurationService',
            'environments_loaded': len(self.environments),
            'config_path': str(self.config_path),
            'statistics': self.config_stats.copy(),
            'environments': {
                env_id: {
                    'name': env.name,
                    'type': env.environment_type.value,
                    'active': env.is_active,
                    'cameras': len(env.cameras),
                    'zones': len(env.zones),
                    'last_updated': env.last_updated.isoformat()
                }
                for env_id, env in self.environments.items()
            }
        }


# Global service instance
_environment_configuration_service: Optional[EnvironmentConfigurationService] = None


def get_environment_configuration_service() -> Optional[EnvironmentConfigurationService]:
    """Get the global environment configuration service instance."""
    return _environment_configuration_service


def initialize_environment_configuration_service(
    tracking_cache: TrackingCache,
    tracking_repository: TrackingRepository,
    config_path: Optional[str] = None
) -> EnvironmentConfigurationService:
    """Initialize the global environment configuration service."""
    global _environment_configuration_service
    if _environment_configuration_service is None:
        _environment_configuration_service = EnvironmentConfigurationService(
            tracking_cache, tracking_repository, config_path
        )
    return _environment_configuration_service