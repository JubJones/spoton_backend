"""
Configuration manager application service.

Handles application-level configuration management and validation.
Coordinates between different configuration domains.
Maximum 200 lines as per refactoring plan.
"""
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from app.domain.shared.value_objects.camera_id import CameraID
from app.infrastructure.config.camera_config_service import CameraConfigService
from app.infrastructure.config.ai_model_config_service import AIModelConfigService
from app.infrastructure.config.deployment_config_service import DeploymentConfigService

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Application-level configuration manager.
    
    Orchestrates configuration from different sources and domains
    while maintaining consistency and validation.
    """
    
    def __init__(
        self,
        camera_config_service: CameraConfigService,
        ai_config_service: AIModelConfigService,
        deployment_config_service: DeploymentConfigService
    ):
        """
        Initialize configuration manager.
        
        Args:
            camera_config_service: Camera configuration service
            ai_config_service: AI model configuration service
            deployment_config_service: Deployment configuration service
        """
        self.camera_config_service = camera_config_service
        self.ai_config_service = ai_config_service
        self.deployment_config_service = deployment_config_service
        
        self._config_cache = {}
        self._last_reload = datetime.utcnow()
        
        logger.debug("ConfigurationManager initialized")
    
    def get_application_config(self) -> Dict[str, Any]:
        """
        Get complete application configuration.
        
        Returns:
            Dictionary with all application configuration
        """
        config = {
            'cameras': self.camera_config_service.get_all_camera_configs(),
            'ai_models': self.ai_config_service.get_ai_model_configs(),
            'deployment': self.deployment_config_service.get_deployment_config(),
            'environment': self._get_environment_config(),
            'processing': self._get_processing_config(),
            'last_updated': self._last_reload.isoformat()
        }
        
        return config
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate entire application configuration.
        
        Returns:
            Validation results with any errors or warnings
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'components_validated': []
        }
        
        # Validate camera configurations
        camera_validation = self.camera_config_service.validate_all_cameras()
        if not camera_validation['is_valid']:
            validation_results['is_valid'] = False
            validation_results['errors'].extend(camera_validation['errors'])
        
        validation_results['warnings'].extend(camera_validation.get('warnings', []))
        validation_results['components_validated'].append('cameras')
        
        # Validate AI model configurations
        ai_validation = self.ai_config_service.validate_ai_configurations()
        if not ai_validation['is_valid']:
            validation_results['is_valid'] = False
            validation_results['errors'].extend(ai_validation['errors'])
        
        validation_results['warnings'].extend(ai_validation.get('warnings', []))
        validation_results['components_validated'].append('ai_models')
        
        # Cross-component validation
        cross_validation = self._validate_cross_component_consistency()
        if not cross_validation['is_valid']:
            validation_results['is_valid'] = False
            validation_results['errors'].extend(cross_validation['errors'])
        
        validation_results['warnings'].extend(cross_validation.get('warnings', []))
        validation_results['components_validated'].append('cross_component')
        
        logger.info(f"Configuration validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        return validation_results
    
    def reload_configuration(self) -> bool:
        """
        Reload configuration from all sources.
        
        Returns:
            True if reload successful
        """
        try:
            # Clear cache
            self._config_cache.clear()
            
            # Reload all configuration services
            self.camera_config_service.reload_configurations()
            self.ai_config_service.reload_configurations()
            self.deployment_config_service.reload_configurations()
            
            # Update reload timestamp
            self._last_reload = datetime.utcnow()
            
            # Validate after reload
            validation = self.validate_configuration()
            if not validation['is_valid']:
                logger.error(f"Configuration reload validation failed: {validation['errors']}")
                return False
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_camera_configuration(self, camera_id: CameraID) -> Optional[Dict[str, Any]]:
        """
        Get configuration for specific camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Camera configuration or None if not found
        """
        return self.camera_config_service.get_camera_config(camera_id)
    
    def update_camera_configuration(
        self,
        camera_id: CameraID,
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update camera configuration.
        
        Args:
            camera_id: Camera identifier
            config_updates: Configuration updates
            
        Returns:
            True if update successful
        """
        try:
            success = self.camera_config_service.update_camera_config(camera_id, config_updates)
            
            if success:
                # Clear relevant cache entries
                cache_key = f"camera_{camera_id}"
                self._config_cache.pop(cache_key, None)
                
                logger.info(f"Updated configuration for camera {camera_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update camera {camera_id} configuration: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system configuration status.
        
        Returns:
            System status information
        """
        return {
            'configuration_loaded': bool(self._config_cache or self._last_reload),
            'last_reload': self._last_reload.isoformat(),
            'camera_count': len(self.camera_config_service.get_all_camera_configs()),
            'ai_models_configured': len(self.ai_config_service.get_ai_model_configs()),
            'deployment_environment': self.deployment_config_service.get_current_environment().value,
            'validation_status': self.validate_configuration()['is_valid']
        }
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        return {
            'name': 'default_environment',
            'type': 'factory',
            'timezone': 'UTC',
            'units': 'metric'
        }
    
    def _get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return {
            'target_fps': 10,
            'detection_threshold': 0.5,
            'tracking_threshold': 0.7,
            'max_age': 30,
            'frame_buffer_size': 100
        }
    
    def _validate_cross_component_consistency(self) -> Dict[str, Any]:
        """Validate consistency across different configuration components."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check that all cameras have required AI model configurations
        camera_configs = self.camera_config_service.get_all_camera_configs()
        ai_configs = self.ai_config_service.get_ai_model_configs()
        
        if camera_configs and not ai_configs.get('detector'):
            validation_result['errors'].append("Cameras configured but no detector model specified")
            validation_result['is_valid'] = False
        
        # Check resolution compatibility
        for camera_config in camera_configs:
            resolution = camera_config.get('resolution', [])
            if resolution and (resolution[0] < 640 or resolution[1] < 480):
                validation_result['warnings'].append(
                    f"Camera {camera_config.get('camera_id')} has low resolution: {resolution}"
                )
        
        return validation_result