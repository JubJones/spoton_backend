"""
Deployment configuration service for infrastructure layer.

Handles deployment-specific settings, environment variables, and system configuration.
Maximum 200 lines per plan.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import os
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Types of deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentConfigService:
    """
    Deployment configuration infrastructure service.
    
    Manages environment-specific settings, resource limits, and deployment parameters.
    """
    
    def __init__(self, config_file_path: Optional[Path] = None):
        """
        Initialize deployment configuration service.
        
        Args:
            config_file_path: Optional path to configuration file
        """
        self.config_file_path = config_file_path or Path("config/deployment.json")
        self._deployment_configs = {}
        self._current_environment = self._detect_environment()
        
        self._load_deployment_configurations()
        logger.debug("DeploymentConfigService initialized")
    
    def get_current_environment(self) -> EnvironmentType:
        """Get current deployment environment."""
        return self._current_environment
    
    def get_deployment_config(self, environment: Optional[EnvironmentType] = None) -> Dict[str, Any]:
        """
        Get deployment configuration for environment.
        
        Args:
            environment: Target environment (defaults to current)
            
        Returns:
            Deployment configuration dictionary
        """
        env_key = (environment or self._current_environment).value
        return self._deployment_configs.get(env_key, {})
    
    def get_all_deployment_configs(self) -> Dict[str, Any]:
        """Get all deployment configurations."""
        return self._deployment_configs.copy()
    
    def update_deployment_config(
        self,
        environment: EnvironmentType,
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update deployment configuration for environment.
        
        Args:
            environment: Target environment
            config_updates: Configuration updates to apply
            
        Returns:
            True if update successful
        """
        env_key = environment.value
        
        try:
            # Update or create configuration
            if env_key not in self._deployment_configs:
                self._deployment_configs[env_key] = {}
            
            self._deployment_configs[env_key].update(config_updates)
            
            # Save to file
            self._save_deployment_configurations()
            
            logger.info(f"Updated deployment configuration for {environment.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update deployment config for {environment.value}: {e}")
            return False
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits for current environment."""
        config = self.get_deployment_config()
        return config.get('resource_limits', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for current environment."""
        config = self.get_deployment_config()
        return config.get('database', {})
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration for current environment."""
        config = self.get_deployment_config()
        return config.get('redis', {})
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration for current environment."""
        config = self.get_deployment_config()
        return config.get('s3', {})
    
    def validate_deployment_configuration(self, environment: Optional[EnvironmentType] = None) -> Dict[str, Any]:
        """
        Validate deployment configuration.
        
        Args:
            environment: Environment to validate (defaults to current)
            
        Returns:
            Validation results
        """
        env = environment or self._current_environment
        config = self.get_deployment_config(env)
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'environment': env.value
        }
        
        # Validate required sections
        required_sections = ['database', 'redis', 's3', 'resource_limits']
        for section in required_sections:
            if section not in config:
                validation_result['errors'].append(f"Missing required section: {section}")
                validation_result['is_valid'] = False
        
        # Validate database configuration
        db_config = config.get('database', {})
        if db_config:
            required_db_fields = ['host', 'port', 'database', 'user']
            for field in required_db_fields:
                if field not in db_config:
                    validation_result['errors'].append(f"Database: Missing required field '{field}'")
                    validation_result['is_valid'] = False
        
        # Validate Redis configuration
        redis_config = config.get('redis', {})
        if redis_config:
            if 'host' not in redis_config or 'port' not in redis_config:
                validation_result['errors'].append("Redis: Missing host or port")
                validation_result['is_valid'] = False
        
        # Validate S3 configuration
        s3_config = config.get('s3', {})
        if s3_config:
            required_s3_fields = ['endpoint_url', 'bucket_name', 'access_key_id']
            for field in required_s3_fields:
                if field not in s3_config:
                    validation_result['errors'].append(f"S3: Missing required field '{field}'")
                    validation_result['is_valid'] = False
        
        # Validate resource limits
        resource_limits = config.get('resource_limits', {})
        if resource_limits:
            memory_limit = resource_limits.get('memory_mb')
            if memory_limit and memory_limit < 1024:
                validation_result['warnings'].append(f"Memory limit {memory_limit}MB may be too low")
        
        return validation_result
    
    def reload_configurations(self) -> bool:
        """
        Reload deployment configurations from file.
        
        Returns:
            True if reload successful
        """
        try:
            self._deployment_configs.clear()
            self._load_deployment_configurations()
            
            logger.info("Deployment configurations reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload deployment configurations: {e}")
            return False
    
    def _detect_environment(self) -> EnvironmentType:
        """Detect current deployment environment."""
        env_var = os.getenv('ENVIRONMENT', 'development').lower()
        
        try:
            return EnvironmentType(env_var)
        except ValueError:
            logger.warning(f"Unknown environment '{env_var}', defaulting to development")
            return EnvironmentType.DEVELOPMENT
    
    def _load_deployment_configurations(self) -> None:
        """Load deployment configurations from file."""
        try:
            if not self.config_file_path.exists():
                logger.warning(f"Deployment config file not found: {self.config_file_path}")
                self._create_default_configuration()
                return
            
            with open(self.config_file_path, 'r') as f:
                self._deployment_configs = json.load(f)
            
            logger.info(f"Loaded {len(self._deployment_configs)} deployment configurations")
            
        except Exception as e:
            logger.error(f"Failed to load deployment configurations: {e}")
            self._create_default_configuration()
    
    def _save_deployment_configurations(self) -> None:
        """Save deployment configurations to file."""
        try:
            # Ensure directory exists
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                **self._deployment_configs,
                'last_updated': datetime.utcnow().isoformat(),
                'current_environment': self._current_environment.value
            }
            
            with open(self.config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.debug(f"Saved deployment configurations to {self.config_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save deployment configurations: {e}")
    
    def _create_default_configuration(self) -> None:
        """Create default deployment configurations."""
        self._deployment_configs = {
            'development': {
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'spoton_dev',
                    'user': 'spoton_user',
                    'password': 'dev_password',
                    'pool_size': 5
                },
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'max_connections': 10
                },
                's3': {
                    'endpoint_url': 'http://localhost:9000',
                    'bucket_name': 'spoton-dev',
                    'access_key_id': 'dev_key',
                    'secret_access_key': 'dev_secret'
                },
                'resource_limits': {
                    'memory_mb': 2048,
                    'cpu_cores': 2,
                    'max_workers': 4
                },
                'logging': {
                    'level': 'DEBUG',
                    'format': 'detailed'
                }
            },
            'production': {
                'database': {
                    'host': 'db.spoton.prod',
                    'port': 5432,
                    'database': 'spoton_prod',
                    'user': 'spoton_user',
                    'pool_size': 20,
                    'ssl_mode': 'require'
                },
                'redis': {
                    'host': 'redis.spoton.prod',
                    'port': 6379,
                    'db': 0,
                    'max_connections': 100,
                    'ssl': True
                },
                's3': {
                    'endpoint_url': 'https://s3.amazonaws.com',
                    'bucket_name': 'spoton-production',
                    'region': 'us-west-2'
                },
                'resource_limits': {
                    'memory_mb': 8192,
                    'cpu_cores': 8,
                    'max_workers': 16
                },
                'logging': {
                    'level': 'INFO',
                    'format': 'json'
                }
            }
        }
        
        # Save default configuration
        self._save_deployment_configurations()
        
        logger.info("Created default deployment configurations")