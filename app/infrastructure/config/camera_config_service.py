"""
Camera configuration service for infrastructure layer.

Handles camera-specific configuration management, calibration data,
and hardware specifications. Maximum 200 lines per plan.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json
from pathlib import Path

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.mapping.value_objects.homography_matrix import HomographyMatrix
from app.domain.mapping.value_objects.coordinates import WorldCoordinates

logger = logging.getLogger(__name__)


class CameraConfigService:
    """
    Camera configuration infrastructure service.
    
    Manages camera configurations, calibration data, and hardware settings.
    """
    
    def __init__(self, config_file_path: Optional[Path] = None):
        """
        Initialize camera configuration service.
        
        Args:
            config_file_path: Optional path to configuration file
        """
        self.config_file_path = config_file_path or Path("config/cameras.json")
        self._camera_configs = {}
        self._homography_cache = {}
        
        self._load_camera_configurations()
        logger.debug("CameraConfigService initialized")
    
    def get_camera_config(self, camera_id: CameraID) -> Optional[Dict[str, Any]]:
        """
        Get configuration for specific camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Camera configuration dictionary or None
        """
        camera_key = str(camera_id)
        return self._camera_configs.get(camera_key)
    
    def get_all_camera_configs(self) -> List[Dict[str, Any]]:
        """Get configurations for all cameras."""
        return list(self._camera_configs.values())
    
    def update_camera_config(
        self,
        camera_id: CameraID,
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update camera configuration.
        
        Args:
            camera_id: Camera identifier
            config_updates: Configuration updates to apply
            
        Returns:
            True if update successful
        """
        camera_key = str(camera_id)
        
        if camera_key not in self._camera_configs:
            logger.error(f"Camera {camera_id} not found in configuration")
            return False
        
        try:
            # Update configuration
            self._camera_configs[camera_key].update(config_updates)
            
            # Clear homography cache for this camera
            self._homography_cache.pop(camera_key, None)
            
            # Save to file
            self._save_camera_configurations()
            
            logger.info(f"Updated configuration for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update camera {camera_id} configuration: {e}")
            return False
    
    def get_camera_homography(self, camera_id: CameraID) -> Optional[HomographyMatrix]:
        """
        Get homography matrix for camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            HomographyMatrix or None if not available
        """
        camera_key = str(camera_id)
        
        # Check cache first
        if camera_key in self._homography_cache:
            return self._homography_cache[camera_key]
        
        # Load from configuration
        camera_config = self.get_camera_config(camera_id)
        if not camera_config:
            return None
        
        homography_data = camera_config.get('homography_matrix')
        if not homography_data:
            return None
        
        try:
            homography_matrix = HomographyMatrix.from_array(homography_data)
            self._homography_cache[camera_key] = homography_matrix
            return homography_matrix
            
        except Exception as e:
            logger.error(f"Failed to load homography for camera {camera_id}: {e}")
            return None
    
    def update_camera_homography(
        self,
        camera_id: CameraID,
        homography_matrix: HomographyMatrix
    ) -> bool:
        """
        Update homography matrix for camera.
        
        Args:
            camera_id: Camera identifier
            homography_matrix: New homography matrix
            
        Returns:
            True if update successful
        """
        try:
            # Update configuration
            config_updates = {
                'homography_matrix': homography_matrix.to_list(),
                'calibration_accuracy': homography_matrix.calibration_error,
                'last_calibrated': datetime.utcnow().isoformat()
            }
            
            success = self.update_camera_config(camera_id, config_updates)
            
            if success:
                # Update cache
                self._homography_cache[str(camera_id)] = homography_matrix
                logger.info(f"Updated homography for camera {camera_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update homography for camera {camera_id}: {e}")
            return False
    
    def validate_all_cameras(self) -> Dict[str, Any]:
        """
        Validate all camera configurations.
        
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'cameras_checked': 0
        }
        
        for camera_key, config in self._camera_configs.items():
            validation_result['cameras_checked'] += 1
            
            # Validate required fields
            required_fields = ['camera_id', 'name', 'resolution', 'frame_rate']
            for field in required_fields:
                if field not in config:
                    validation_result['errors'].append(
                        f"Camera {camera_key}: Missing required field '{field}'"
                    )
                    validation_result['is_valid'] = False
            
            # Validate resolution
            resolution = config.get('resolution')
            if resolution and len(resolution) == 2:
                width, height = resolution
                if width < 640 or height < 480:
                    validation_result['warnings'].append(
                        f"Camera {camera_key}: Low resolution {width}x{height}"
                    )
            
            # Validate frame rate
            frame_rate = config.get('frame_rate', 0)
            if frame_rate < 1 or frame_rate > 60:
                validation_result['warnings'].append(
                    f"Camera {camera_key}: Unusual frame rate {frame_rate} fps"
                )
            
            # Validate homography if present
            homography_data = config.get('homography_matrix')
            if homography_data:
                try:
                    HomographyMatrix.from_array(homography_data)
                except Exception as e:
                    validation_result['errors'].append(
                        f"Camera {camera_key}: Invalid homography matrix - {e}"
                    )
                    validation_result['is_valid'] = False
        
        return validation_result
    
    def reload_configurations(self) -> bool:
        """
        Reload camera configurations from file.
        
        Returns:
            True if reload successful
        """
        try:
            self._camera_configs.clear()
            self._homography_cache.clear()
            self._load_camera_configurations()
            
            logger.info("Camera configurations reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload camera configurations: {e}")
            return False
    
    def _load_camera_configurations(self) -> None:
        """Load camera configurations from file."""
        try:
            if not self.config_file_path.exists():
                logger.warning(f"Camera config file not found: {self.config_file_path}")
                self._create_default_configuration()
                return
            
            with open(self.config_file_path, 'r') as f:
                config_data = json.load(f)
            
            # Process camera configurations
            for camera_data in config_data.get('cameras', []):
                camera_id = camera_data.get('camera_id')
                if camera_id:
                    self._camera_configs[camera_id] = camera_data
            
            logger.info(f"Loaded {len(self._camera_configs)} camera configurations")
            
        except Exception as e:
            logger.error(f"Failed to load camera configurations: {e}")
            self._create_default_configuration()
    
    def _save_camera_configurations(self) -> None:
        """Save camera configurations to file."""
        try:
            # Ensure directory exists
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'cameras': list(self._camera_configs.values()),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            with open(self.config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.debug(f"Saved camera configurations to {self.config_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save camera configurations: {e}")
    
    def _create_default_configuration(self) -> None:
        """Create default camera configuration."""
        default_cameras = [
            {
                'camera_id': 'camera1',
                'name': 'Camera 1',
                'type': 'fixed',
                'resolution': [1920, 1080],
                'frame_rate': 30.0,
                'field_of_view': 60.0,
                'orientation': 0.0,
                'is_active': True,
                'position': {'x': 0.0, 'y': 0.0},
                'last_calibrated': None
            }
        ]
        
        for camera_data in default_cameras:
            camera_id = camera_data['camera_id']
            self._camera_configs[camera_id] = camera_data
        
        # Save default configuration
        self._save_camera_configurations()
        
        logger.info("Created default camera configuration")