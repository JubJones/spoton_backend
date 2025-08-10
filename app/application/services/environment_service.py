"""
Environment service for application layer.

Lightweight service focusing on environment-specific business logic.
Replaces the mega environment_configuration_service.py with focused responsibilities.
Maximum 300 lines per plan.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from app.domain.shared.value_objects.camera_id import CameraID
from app.infrastructure.config.camera_config_service import CameraConfigService
from app.infrastructure.config.ai_model_config_service import AIModelConfigService
from app.infrastructure.config.deployment_config_service import DeploymentConfigService
from app.application.services.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Types of environments."""
    CAMPUS = "campus"
    FACTORY = "factory"
    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"


@dataclass
class EnvironmentInfo:
    """Environment information summary."""
    environment_id: str
    name: str
    environment_type: EnvironmentType
    description: str
    camera_count: int
    is_active: bool
    last_updated: datetime


class EnvironmentService:
    """
    Environment service for application layer.
    
    Provides environment-specific business logic and coordination
    between configuration services. Focused on environment management
    rather than complex configuration handling.
    """
    
    def __init__(self, configuration_manager: ConfigurationManager):
        """
        Initialize environment service.
        
        Args:
            configuration_manager: Configuration manager for orchestration
        """
        self.config_manager = configuration_manager
        
        # Environment definitions
        self._environment_definitions = {
            'campus': {
                'name': 'Campus Environment',
                'type': EnvironmentType.CAMPUS,
                'description': 'Multi-zone campus with entrance/exit monitoring',
                'default_cameras': ['c09', 'c12', 'c13', 'c16']
            },
            'factory': {
                'name': 'Factory Environment', 
                'type': EnvironmentType.FACTORY,
                'description': 'Production facility with line monitoring',
                'default_cameras': ['f01', 'f02', 'f03', 'f04']
            }
        }
        
        logger.debug("EnvironmentService initialized")
    
    def get_available_environments(self) -> List[EnvironmentInfo]:
        """
        Get list of available environments.
        
        Returns:
            List of environment information
        """
        environments = []
        
        for env_id, env_def in self._environment_definitions.items():
            # Get camera configurations to determine if environment is ready
            camera_configs = self.config_manager.camera_config_service.get_all_camera_configs()
            available_cameras = [
                cam for cam in camera_configs 
                if cam.get('camera_id') in env_def['default_cameras']
            ]
            
            environments.append(EnvironmentInfo(
                environment_id=env_id,
                name=env_def['name'],
                environment_type=env_def['type'],
                description=env_def['description'],
                camera_count=len(available_cameras),
                is_active=len(available_cameras) > 0,
                last_updated=datetime.utcnow()
            ))
        
        return environments
    
    def get_environment_details(self, environment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed environment information.
        
        Args:
            environment_id: Environment identifier
            
        Returns:
            Environment details or None if not found
        """
        if environment_id not in self._environment_definitions:
            return None
        
        env_def = self._environment_definitions[environment_id]
        
        # Get camera configurations for this environment
        all_camera_configs = self.config_manager.camera_config_service.get_all_camera_configs()
        env_cameras = []
        
        for camera_config in all_camera_configs:
            if camera_config.get('camera_id') in env_def['default_cameras']:
                env_cameras.append(camera_config)
        
        # Get AI model configurations
        ai_models = self.config_manager.ai_config_service.get_ai_model_configs()
        
        # Get deployment configuration
        deployment_config = self.config_manager.deployment_config_service.get_deployment_config()
        
        return {
            'environment_id': environment_id,
            'name': env_def['name'],
            'type': env_def['type'].value,
            'description': env_def['description'],
            'cameras': env_cameras,
            'ai_models': ai_models,
            'deployment': {
                'environment': deployment_config.get('environment', 'unknown'),
                'resource_limits': deployment_config.get('resource_limits', {})
            },
            'status': {
                'is_active': len(env_cameras) > 0,
                'camera_count': len(env_cameras),
                'models_configured': len(ai_models),
                'ready_for_processing': len(env_cameras) > 0 and len(ai_models) > 0
            },
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def validate_environment(self, environment_id: str) -> Dict[str, Any]:
        """
        Validate environment readiness.
        
        Args:
            environment_id: Environment to validate
            
        Returns:
            Validation results
        """
        if environment_id not in self._environment_definitions:
            return {
                'is_valid': False,
                'errors': [f'Unknown environment: {environment_id}'],
                'warnings': []
            }
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        env_def = self._environment_definitions[environment_id]
        
        # Validate camera availability
        all_camera_configs = self.config_manager.camera_config_service.get_all_camera_configs()
        configured_cameras = [cam.get('camera_id') for cam in all_camera_configs]
        
        missing_cameras = []
        for expected_camera in env_def['default_cameras']:
            if expected_camera not in configured_cameras:
                missing_cameras.append(expected_camera)
        
        if missing_cameras:
            validation_result['errors'].append(
                f"Missing camera configurations: {', '.join(missing_cameras)}"
            )
            validation_result['is_valid'] = False
        
        # Validate AI models
        ai_models = self.config_manager.ai_config_service.get_ai_model_configs()
        required_models = ['detector', 'tracker']
        
        missing_models = []
        for model_type in required_models:
            if model_type not in ai_models:
                missing_models.append(model_type)
        
        if missing_models:
            validation_result['errors'].append(
                f"Missing AI model configurations: {', '.join(missing_models)}"
            )
            validation_result['is_valid'] = False
        
        # Validate deployment configuration
        deployment_validation = self.config_manager.deployment_config_service.validate_deployment_configuration()
        if not deployment_validation['is_valid']:
            validation_result['errors'].extend(deployment_validation['errors'])
            validation_result['is_valid'] = False
        
        validation_result['warnings'].extend(deployment_validation.get('warnings', []))
        
        return validation_result
    
    def get_environment_cameras(self, environment_id: str) -> List[Dict[str, Any]]:
        """
        Get cameras for specific environment.
        
        Args:
            environment_id: Environment identifier
            
        Returns:
            List of camera configurations
        """
        if environment_id not in self._environment_definitions:
            return []
        
        env_def = self._environment_definitions[environment_id]
        all_camera_configs = self.config_manager.camera_config_service.get_all_camera_configs()
        
        env_cameras = []
        for camera_config in all_camera_configs:
            if camera_config.get('camera_id') in env_def['default_cameras']:
                env_cameras.append(camera_config)
        
        return env_cameras
    
    def setup_environment(self, environment_id: str) -> Dict[str, Any]:
        """
        Setup environment with default configurations.
        
        Args:
            environment_id: Environment to setup
            
        Returns:
            Setup results
        """
        if environment_id not in self._environment_definitions:
            return {
                'success': False,
                'error': f'Unknown environment: {environment_id}'
            }
        
        try:
            env_def = self._environment_definitions[environment_id]
            
            # Setup default camera configurations if missing
            existing_cameras = self.get_environment_cameras(environment_id)
            existing_camera_ids = {cam.get('camera_id') for cam in existing_cameras}
            
            cameras_created = 0
            for camera_id in env_def['default_cameras']:
                if camera_id not in existing_camera_ids:
                    # Create default camera configuration
                    default_camera_config = self._create_default_camera_config(
                        camera_id, environment_id
                    )
                    
                    success = self.config_manager.camera_config_service.update_camera_config(
                        CameraID(camera_id), default_camera_config
                    )
                    
                    if success:
                        cameras_created += 1
            
            # Validate final environment state
            validation = self.validate_environment(environment_id)
            
            return {
                'success': validation['is_valid'],
                'environment_id': environment_id,
                'cameras_created': cameras_created,
                'validation': validation
            }
            
        except Exception as e:
            logger.error(f"Error setting up environment {environment_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_environment_status(self, environment_id: str) -> Dict[str, Any]:
        """
        Get current environment status.
        
        Args:
            environment_id: Environment identifier
            
        Returns:
            Environment status information
        """
        if environment_id not in self._environment_definitions:
            return {'error': f'Unknown environment: {environment_id}'}
        
        validation = self.validate_environment(environment_id)
        cameras = self.get_environment_cameras(environment_id)
        
        return {
            'environment_id': environment_id,
            'is_ready': validation['is_valid'],
            'camera_count': len(cameras),
            'active_cameras': len([cam for cam in cameras if cam.get('is_active', True)]),
            'validation': validation,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _create_default_camera_config(
        self, 
        camera_id: str, 
        environment_id: str
    ) -> Dict[str, Any]:
        """Create default camera configuration."""
        return {
            'camera_id': camera_id,
            'name': f'{environment_id.title()} Camera {camera_id}',
            'type': 'fixed',
            'resolution': [1920, 1080],
            'frame_rate': 30.0,
            'field_of_view': 70.0,
            'orientation': 0.0,
            'is_active': True,
            'position': {'x': 0.0, 'y': 0.0},
            'last_calibrated': None,
            'created_at': datetime.utcnow().isoformat()
        }
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get environment service statistics."""
        environments = self.get_available_environments()
        
        return {
            'service_name': 'EnvironmentService',
            'total_environments': len(self._environment_definitions),
            'active_environments': len([env for env in environments if env.is_active]),
            'environment_types': list(set(env.environment_type.value for env in environments)),
            'configuration_status': self.config_manager.get_system_status()
        }