"""
Consolidated core configuration system.

Clean architecture implementation with proper separation of concerns,
environment-specific configuration, and comprehensive validation.
Following Phase 6: Configuration Consolidation requirements.
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from enum import Enum
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConfigValidationError(Exception):
    """Configuration validation errors."""
    pass


class BaseConfigModel(BaseModel):
    """Base configuration model with validation support."""
    
    class Config:
        extra = "ignore"
        validate_assignment = True
        use_enum_values = True
    
    def validate_config(self) -> None:
        """Override in subclasses for custom validation."""
        pass


class CoreApplicationConfig(BaseConfigModel):
    """
    Core application configuration (≤200 lines as per plan).
    
    Contains only essential application-level settings that are
    environment-independent and domain-agnostic.
    """
    
    # Application Identity
    app_name: str = Field(
        default="SpotOn Backend", 
        description="Application name for logging and monitoring"
    )
    app_version: str = Field(
        default="1.0.0", 
        description="Application version"
    )
    
    # Environment Configuration
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Current application environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # API Configuration
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="API version 1 route prefix"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # Logging Configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application log level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Security Configuration
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Application secret key"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # Monitoring and Health
    health_check_timeout: int = Field(
        default=30,
        description="Health check timeout in seconds"
    )
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    
    # Resource Limits
    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent processing tasks"
    )
    request_timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Request timeout in seconds"
    )
    
    # Local Storage Paths (Core application directories)
    local_storage_path: str = Field(
        default="./storage",
        description="Base local storage directory"
    )
    temp_storage_path: str = Field(
        default="./temp",
        description="Temporary storage directory"
    )
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            try:
                return EnvironmentType(v.lower())
            except ValueError:
                valid_envs = [e.value for e in EnvironmentType]
                raise ValueError(f"Invalid environment '{v}'. Must be one of: {valid_envs}")
        return v
    
    @validator('log_level', pre=True)
    def validate_log_level(cls, v):
        """Validate log level setting."""
        if isinstance(v, str):
            try:
                return LogLevel(v.upper())
            except ValueError:
                valid_levels = [l.value for l in LogLevel]
                raise ValueError(f"Invalid log level '{v}'. Must be one of: {valid_levels}")
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v, values):
        """Validate secret key for production."""
        environment = values.get('environment', EnvironmentType.DEVELOPMENT)
        if environment == EnvironmentType.PRODUCTION and v == "dev-secret-key-change-in-production":
            raise ValueError("Production environment requires a secure secret key")
        return v
    
    def validate_config(self) -> None:
        """Validate configuration consistency."""
        # Ensure directories exist or can be created
        for path_field in ['local_storage_path', 'temp_storage_path']:
            path_value = getattr(self, path_field)
            path_obj = Path(path_value)
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Validated directory path: {path_obj}")
            except Exception as e:
                raise ConfigValidationError(f"Cannot create directory {path_obj}: {e}")
        
        # Environment-specific validations
        if self.environment == EnvironmentType.PRODUCTION:
            if self.debug:
                logger.warning("Debug mode is enabled in production environment")
            
            if self.log_level == LogLevel.DEBUG:
                logger.warning("Debug log level in production may impact performance")


class ConsolidatedSettings(BaseSettings):
    """
    Consolidated settings manager.
    
    Orchestrates configuration loading from multiple sources with proper
    precedence order: environment variables > config files > defaults.
    """
    
    # Define core configuration field
    core: CoreApplicationConfig = Field(default_factory=CoreApplicationConfig)
    
    def __init__(self, **kwargs):
        """Initialize consolidated settings with configuration validation."""
        super().__init__(**kwargs)
        
        # Configure logging based on core settings
        self._configure_logging()
        
        # Load infrastructure configurations (lazy-loaded)
        self._ai_config = None
        self._database_config = None
        self._storage_config = None
        self._camera_config = None
        
        logger.info(f"Consolidated configuration loaded for environment: {self.core.environment.value}")
    
    def _configure_logging(self) -> None:
        """Configure logging based on core settings."""
        logging.basicConfig(
            level=getattr(logging, self.core.log_level.value),
            format=self.core.log_format
        )
        
        # Set application logger level
        app_logger = logging.getLogger("app")
        app_logger.setLevel(getattr(logging, self.core.log_level.value))
    
    @property
    def ai_config(self):
        """Lazy-loaded AI configuration."""
        if self._ai_config is None:
            from app.infrastructure.config.ai_model_settings import AIModelSettings
            self._ai_config = AIModelSettings()
        return self._ai_config
    
    @property
    def database_config(self):
        """Lazy-loaded database configuration."""
        if self._database_config is None:
            from app.infrastructure.config.database_settings import DatabaseSettings
            self._database_config = DatabaseSettings()
        return self._database_config
    
    @property
    def storage_config(self):
        """Lazy-loaded storage configuration."""
        if self._storage_config is None:
            from app.infrastructure.config.storage_settings import StorageSettings
            self._storage_config = StorageSettings()
        return self._storage_config
    
    @property
    def camera_config(self):
        """Lazy-loaded camera configuration."""
        if self._camera_config is None:
            from app.infrastructure.config.camera_settings import CameraSettings
            self._camera_config = CameraSettings()
        return self._camera_config
    
    def validate_all_configurations(self) -> Dict[str, bool]:
        """
        Validate all loaded configurations.
        
        Returns:
            Dictionary mapping configuration type to validation status
        """
        validation_results = {}
        
        try:
            self.core.validate_config()
            validation_results['core'] = True
        except Exception as e:
            logger.error(f"Core configuration validation failed: {e}")
            validation_results['core'] = False
        
        # Validate infrastructure configurations if loaded
        config_types = [
            ('ai_config', 'AI'),
            ('database_config', 'Database'),
            ('storage_config', 'Storage'),
            ('camera_config', 'Camera')
        ]
        
        for attr_name, config_name in config_types:
            private_attr = f"_{attr_name}"
            if getattr(self, private_attr) is not None:
                try:
                    config = getattr(self, attr_name)
                    if hasattr(config, 'validate_config'):
                        config.validate_config()
                    validation_results[attr_name] = True
                except Exception as e:
                    logger.error(f"{config_name} configuration validation failed: {e}")
                    validation_results[attr_name] = False
        
        return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration summary.
        
        Returns:
            Dictionary containing configuration overview
        """
        return {
            'environment': self.core.environment.value,
            'app_name': self.core.app_name,
            'app_version': self.core.app_version,
            'debug_mode': self.core.debug,
            'log_level': self.core.log_level.value,
            'loaded_configs': {
                'core': True,
                'ai_config': self._ai_config is not None,
                'database_config': self._database_config is not None,
                'storage_config': self._storage_config is not None,
                'camera_config': self._camera_config is not None
            },
            'validation_timestamp': datetime.utcnow().isoformat()
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Environment-specific configuration factories
def create_development_config() -> ConsolidatedSettings:
    """
    Create development environment configuration.
    
    Optimized for local development with debug features enabled,
    relaxed security settings, and detailed logging.
    """
    os.environ.setdefault('ENVIRONMENT', 'development')
    os.environ.setdefault('DEBUG', 'true')
    os.environ.setdefault('LOG_LEVEL', 'DEBUG')
    
    # Development-specific defaults
    os.environ.setdefault('SECRET_KEY', 'dev-secret-key-change-in-production')
    os.environ.setdefault('CORS_ORIGINS', '["http://localhost:3000","http://localhost:5173","http://127.0.0.1:3000"]')
    os.environ.setdefault('METRICS_ENABLED', 'true')
    
    # Database settings for development
    os.environ.setdefault('DB_POSTGRESQL_HOST', 'localhost')
    os.environ.setdefault('DB_POSTGRESQL_PORT', '5432')
    os.environ.setdefault('DB_REDIS_HOST', 'localhost')
    os.environ.setdefault('DB_REDIS_PORT', '6379')
    
    # AI model settings for development
    os.environ.setdefault('AI_DETECTOR_USE_AMP', 'false')  # Disable AMP for debugging
    os.environ.setdefault('AI_TRACKER_HALF_PRECISION', 'false')  # Full precision for accuracy
    os.environ.setdefault('AI_TARGET_FPS', '10')  # Lower FPS for development
    
    # Storage settings for development
    os.environ.setdefault('STORAGE_STORAGE_BACKEND', 'local')
    os.environ.setdefault('STORAGE_ENABLE_CACHING', 'true')
    os.environ.setdefault('STORAGE_LOCAL_MAX_STORAGE_SIZE_GB', '50')
    
    return ConsolidatedSettings()


def create_production_config() -> ConsolidatedSettings:
    """
    Create production environment configuration.
    
    Optimized for production deployment with security hardening,
    performance optimization, and comprehensive monitoring.
    """
    os.environ.setdefault('ENVIRONMENT', 'production')
    os.environ.setdefault('DEBUG', 'false')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    # Production security settings
    os.environ.setdefault('CORS_ORIGINS', '[]')  # Restrict CORS in production
    os.environ.setdefault('METRICS_ENABLED', 'true')
    os.environ.setdefault('HEALTH_CHECK_TIMEOUT', '10')  # Faster health checks
    
    # Database settings for production
    os.environ.setdefault('DB_POSTGRESQL_POOL_SIZE', '20')
    os.environ.setdefault('DB_POSTGRESQL_MAX_OVERFLOW', '40')
    os.environ.setdefault('DB_POSTGRESQL_SSL_MODE', 'require')
    os.environ.setdefault('DB_REDIS_CONNECTION_POOL_SIZE', '100')
    
    # AI model settings for production
    os.environ.setdefault('AI_DETECTOR_USE_AMP', 'true')  # Enable AMP for performance
    os.environ.setdefault('AI_TRACKER_HALF_PRECISION', 'true')  # Half precision for speed
    os.environ.setdefault('AI_TARGET_FPS', '23')  # Full target FPS
    
    # Storage settings for production
    os.environ.setdefault('STORAGE_STORAGE_BACKEND', 'hybrid')
    os.environ.setdefault('STORAGE_ENABLE_CACHING', 'true')
    os.environ.setdefault('STORAGE_ENABLE_METRICS', 'true')
    os.environ.setdefault('STORAGE_LOCAL_MAX_STORAGE_SIZE_GB', '200')
    
    return ConsolidatedSettings()


def create_testing_config() -> ConsolidatedSettings:
    """
    Create testing environment configuration.
    
    Optimized for automated testing with fast execution,
    isolated resources, and comprehensive validation.
    """
    os.environ.setdefault('ENVIRONMENT', 'testing')
    os.environ.setdefault('DEBUG', 'true')  # Enable debug for test diagnostics
    os.environ.setdefault('LOG_LEVEL', 'WARNING')  # Reduce log noise in tests
    
    # Testing-specific settings
    os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-only')
    os.environ.setdefault('CORS_ORIGINS', '["http://localhost"]')
    os.environ.setdefault('METRICS_ENABLED', 'false')  # Disable metrics for testing
    os.environ.setdefault('MAX_CONCURRENT_TASKS', '5')  # Limit concurrency
    
    # Test database settings
    os.environ.setdefault('DB_POSTGRESQL_DATABASE', 'spotondb_test')
    os.environ.setdefault('DB_POSTGRESQL_POOL_SIZE', '5')
    os.environ.setdefault('DB_REDIS_DATABASE', '1')  # Use different Redis DB
    os.environ.setdefault('DB_REDIS_CONNECTION_POOL_SIZE', '10')
    
    # AI model settings for testing
    os.environ.setdefault('AI_DETECTOR_USE_AMP', 'false')
    os.environ.setdefault('AI_TARGET_FPS', '5')  # Very low FPS for fast tests
    os.environ.setdefault('AI_DETECTOR_CONFIDENCE_THRESHOLD', '0.8')  # Higher threshold
    
    # Storage settings for testing
    os.environ.setdefault('STORAGE_STORAGE_BACKEND', 'local')
    os.environ.setdefault('STORAGE_LOCAL_BASE_STORAGE_PATH', './test_storage')
    os.environ.setdefault('STORAGE_LOCAL_MAX_STORAGE_SIZE_GB', '10')
    os.environ.setdefault('STORAGE_EXPORTS_EXPORT_EXPIRY_HOURS', '1')  # Quick cleanup
    
    return ConsolidatedSettings()


def create_staging_config() -> ConsolidatedSettings:
    """
    Create staging environment configuration.
    
    Production-like configuration for pre-production testing
    and validation with enhanced monitoring and debugging.
    """
    os.environ.setdefault('ENVIRONMENT', 'production')  # Use production validation
    os.environ.setdefault('DEBUG', 'false')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    # Staging-specific settings
    os.environ.setdefault('SECRET_KEY', 'staging-secret-key-change-for-production')
    os.environ.setdefault('CORS_ORIGINS', '["https://staging.spoton.example.com"]')
    os.environ.setdefault('METRICS_ENABLED', 'true')
    
    # Database settings for staging
    os.environ.setdefault('DB_POSTGRESQL_DATABASE', 'spotondb_staging')
    os.environ.setdefault('DB_POSTGRESQL_POOL_SIZE', '15')
    os.environ.setdefault('DB_REDIS_DATABASE', '2')
    os.environ.setdefault('DB_REDIS_CONNECTION_POOL_SIZE', '50')
    
    # AI model settings for staging
    os.environ.setdefault('AI_DETECTOR_USE_AMP', 'true')
    os.environ.setdefault('AI_TRACKER_HALF_PRECISION', 'true')
    os.environ.setdefault('AI_TARGET_FPS', '20')  # Slightly reduced for stability
    
    # Storage settings for staging
    os.environ.setdefault('STORAGE_STORAGE_BACKEND', 'hybrid')
    os.environ.setdefault('STORAGE_ENABLE_METRICS', 'true')
    os.environ.setdefault('STORAGE_LOCAL_MAX_STORAGE_SIZE_GB', '100')
    
    return ConsolidatedSettings()


# Configuration factory function
def get_settings() -> ConsolidatedSettings:
    """
    Get appropriate settings based on environment.
    
    Supports development, testing, staging, and production environments
    with environment-specific optimizations and security settings.
    
    Returns:
        Configured ConsolidatedSettings instance
    """
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return create_production_config()
    elif env == 'staging':
        return create_staging_config()
    elif env in ['testing', 'test']:
        return create_testing_config()
    else:
        return create_development_config()


# Environment validation utility
def validate_environment_config() -> Dict[str, Any]:
    """
    Validate current environment configuration.
    
    Returns:
        Dictionary containing validation results and recommendations
    """
    current_env = os.getenv('ENVIRONMENT', 'development').lower()
    settings = get_settings()
    
    validation_result = {
        'environment': current_env,
        'is_valid': True,
        'warnings': [],
        'recommendations': [],
        'configuration_summary': settings.get_configuration_summary()
    }
    
    # Environment-specific validation
    if current_env == 'production':
        # Production security checks
        if settings.core.debug:
            validation_result['warnings'].append("Debug mode enabled in production")
        
        if settings.core.secret_key.startswith(('dev-', 'test-', 'staging-')):
            validation_result['warnings'].append("Non-production secret key detected in production")
            validation_result['is_valid'] = False
        
        if settings.core.cors_origins == ["*"]:
            validation_result['warnings'].append("Wildcard CORS origins in production is insecure")
        
        validation_result['recommendations'].extend([
            "Ensure all secrets are properly configured via environment variables",
            "Enable SSL/TLS in production database connections",
            "Configure monitoring and alerting systems",
            "Review and restrict CORS origins to specific domains"
        ])
    
    elif current_env == 'development':
        validation_result['recommendations'].extend([
            "Use Docker Compose for consistent local development environment",
            "Consider enabling debug logging for troubleshooting",
            "Test with production-like data volumes periodically"
        ])
    
    elif current_env in ['testing', 'test']:
        validation_result['recommendations'].extend([
            "Use isolated test databases to prevent data pollution",
            "Configure shorter timeouts for faster test execution",
            "Consider using test fixtures for consistent test data"
        ])
    
    return validation_result


# Global consolidated settings instance
consolidated_settings = get_settings()

# Backward compatibility alias - Legacy Settings wrapper
def _create_backward_compatible_settings():
    """Create backward compatible settings wrapper on demand."""
    try:
        from app.core.config_compatibility import LegacySettingsWrapper
        return LegacySettingsWrapper(consolidated_settings)
    except ImportError:
        logger.warning("Backward compatibility module not available, using consolidated settings directly")
        return consolidated_settings

# Backward compatibility alias for existing code
settings = _create_backward_compatible_settings()