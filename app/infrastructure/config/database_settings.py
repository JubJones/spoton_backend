"""
Database infrastructure configuration settings.

Clean infrastructure configuration for database connections following Phase 6:
Configuration consolidation requirements. Handles PostgreSQL and Redis settings
with proper validation and environment management.
"""
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import logging
from urllib.parse import quote_plus
import os

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    TIMESCALEDB = "timescaledb"


class RedisMode(Enum):
    """Redis deployment modes."""
    STANDALONE = "standalone"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


class PostgreSQLSettings(BaseModel):
    """PostgreSQL/TimescaleDB configuration settings."""
    
    user: str = Field(
        default="spoton_user",
        description="Database username"
    )
    password: str = Field(
        default="spoton_password",
        description="Database password"
    )
    host: str = Field(
        default="localhost",
        description="Database host"
    )
    port: int = Field(
        default=5432,
        ge=1024,
        le=65535,
        description="Database port"
    )
    database: str = Field(
        default="spotondb",
        min_length=1,
        description="Database name"
    )
    database_type: DatabaseType = Field(
        default=DatabaseType.TIMESCALEDB,
        description="Database type (PostgreSQL or TimescaleDB)"
    )
    
    # Connection pool settings
    pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum pool overflow connections"
    )
    pool_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Connection pool timeout (seconds)"
    )
    pool_recycle: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Connection recycle time (seconds)"
    )
    
    # Query settings
    statement_timeout: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="SQL statement timeout (seconds)"
    )
    connect_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Database connection timeout (seconds)"
    )
    
    # SSL settings
    ssl_mode: str = Field(
        default="prefer",
        description="SSL mode (disable, allow, prefer, require)"
    )
    ssl_cert: Optional[str] = Field(
        default=None,
        description="SSL certificate file path"
    )
    ssl_key: Optional[str] = Field(
        default=None,
        description="SSL private key file path"
    )
    ssl_ca: Optional[str] = Field(
        default=None,
        description="SSL certificate authority file path"
    )
    
    @validator('ssl_mode')
    def validate_ssl_mode(cls, v):
        """Validate SSL mode setting."""
        valid_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        if v not in valid_modes:
            raise ValueError(f"Invalid SSL mode '{v}'. Must be one of: {valid_modes}")
        return v
    
    @property
    def database_url(self) -> str:
        """Generate database connection URL."""
        # URL-encode password to handle special characters
        encoded_password = quote_plus(self.password)
        
        base_url = f"postgresql://{self.user}:{encoded_password}@{self.host}:{self.port}/{self.database}"
        
        # Add connection parameters
        params = []
        if self.ssl_mode != "prefer":
            params.append(f"sslmode={self.ssl_mode}")
        if self.connect_timeout != 10:
            params.append(f"connect_timeout={self.connect_timeout}")
        
        if params:
            base_url += "?" + "&".join(params)
        
        return base_url
    
    @property
    def async_database_url(self) -> str:
        """Generate async database connection URL."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    def validate_postgresql_settings(self) -> None:
        """Validate PostgreSQL-specific settings."""
        # Check SSL certificate files exist if specified
        ssl_files = [
            ("ssl_cert", self.ssl_cert),
            ("ssl_key", self.ssl_key), 
            ("ssl_ca", self.ssl_ca)
        ]
        
        for name, path in ssl_files:
            if path and not Path(path).exists():
                logger.warning(f"SSL file not found: {name} = {path}")
        
        # Validate pool settings
        if self.max_overflow < self.pool_size // 2:
            logger.info(f"Consider increasing max_overflow ({self.max_overflow}) relative to pool_size ({self.pool_size})")


class RedisSettings(BaseModel):
    """Redis cache configuration settings."""
    
    host: str = Field(
        default="localhost",
        description="Redis host"
    )
    port: int = Field(
        default=6379,
        ge=1024,
        le=65535,
        description="Redis port"
    )
    database: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    password: Optional[str] = Field(
        default=None,
        description="Redis password (if authentication enabled)"
    )
    
    # Connection settings
    mode: RedisMode = Field(
        default=RedisMode.STANDALONE,
        description="Redis deployment mode"
    )
    connection_pool_size: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Redis connection pool size"
    )
    connection_timeout: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Redis connection timeout (seconds)"
    )
    socket_keepalive: bool = Field(
        default=True,
        description="Enable TCP keepalive"
    )
    health_check_interval: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Health check interval (seconds)"
    )
    
    # Performance settings
    max_memory_policy: str = Field(
        default="allkeys-lru",
        description="Memory eviction policy"
    )
    default_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default TTL for cache entries (seconds)"
    )
    
    # Clustering settings (for cluster mode)
    cluster_nodes: Optional[List[str]] = Field(
        default=None,
        description="Cluster node addresses (host:port)"
    )
    
    @validator('max_memory_policy')
    def validate_memory_policy(cls, v):
        """Validate Redis memory eviction policy."""
        valid_policies = [
            "noeviction", "allkeys-lru", "volatile-lru", 
            "allkeys-random", "volatile-random", "volatile-ttl"
        ]
        if v not in valid_policies:
            raise ValueError(f"Invalid memory policy '{v}'. Must be one of: {valid_policies}")
        return v
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        if self.password:
            auth = f":{self.password}@"
        else:
            auth = ""
        
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"
    
    def validate_redis_settings(self) -> None:
        """Validate Redis-specific settings."""
        if self.mode == RedisMode.CLUSTER and not self.cluster_nodes:
            logger.warning("Cluster mode specified but no cluster nodes configured")
        
        if self.connection_pool_size < 10:
            logger.warning(f"Small connection pool size ({self.connection_pool_size}) may limit performance")


class DatabaseSettings(BaseSettings):
    """
    Consolidated database infrastructure configuration.
    
    Handles PostgreSQL and Redis settings with proper validation and environment management.
    Follows Phase 6 configuration consolidation requirements.
    """
    
    # Database components
    postgresql: PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    
    # Migration settings
    run_migrations_on_startup: bool = Field(
        default=False,
        description="Automatically run migrations on startup"
    )
    migration_timeout: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Migration timeout (seconds)"
    )
    
    # Backup settings
    backup_enabled: bool = Field(
        default=True,
        description="Enable automatic database backups"
    )
    backup_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Backup interval (hours)"
    )
    backup_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Backup retention period (days)"
    )
    
    # Health check settings
    health_check_enabled: bool = Field(
        default=True,
        description="Enable database health checks"
    )
    health_check_interval: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Health check interval (seconds)"
    )
    
    def __init__(self, **kwargs):
        """Initialize database settings with validation."""
        super().__init__(**kwargs)
        self._validate_all_settings()
        logger.debug("DatabaseSettings initialized successfully")
    
    def validate_config(self) -> None:
        """Validate complete database configuration."""
        self._validate_all_settings()
    
    def _validate_all_settings(self) -> None:
        """Validate all database settings."""
        try:
            self.postgresql.validate_postgresql_settings()
            self.redis.validate_redis_settings()
            
            # Cross-component validation
            if self.postgresql.pool_size + self.redis.connection_pool_size > 100:
                logger.info("Total database connections > 100, monitor resource usage")
            
        except Exception as e:
            logger.error(f"Database settings validation failed: {e}")
            raise
    
    def get_database_urls(self) -> Dict[str, str]:
        """
        Get all database connection URLs.
        
        Returns:
            Dictionary mapping database type to connection URL
        """
        return {
            'postgresql_sync': self.postgresql.database_url,
            'postgresql_async': self.postgresql.async_database_url,
            'redis': self.redis.redis_url
        }
    
    def test_connections(self) -> Dict[str, bool]:
        """
        Test database connections (placeholder for actual implementation).
        
        Returns:
            Dictionary mapping database type to connection status
        """
        # This would contain actual connection testing logic
        return {
            'postgresql': True,  # Placeholder
            'redis': True       # Placeholder
        }
    
    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive database configuration summary.
        
        Returns:
            Dictionary containing configuration overview
        """
        return {
            'postgresql': {
                'host': self.postgresql.host,
                'port': self.postgresql.port,
                'database': self.postgresql.database,
                'type': self.postgresql.database_type.value,
                'pool_size': self.postgresql.pool_size,
                'ssl_mode': self.postgresql.ssl_mode
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'database': self.redis.database,
                'mode': self.redis.mode.value,
                'pool_size': self.redis.connection_pool_size,
                'auth_enabled': self.redis.password is not None
            },
            'operations': {
                'migrations_on_startup': self.run_migrations_on_startup,
                'backup_enabled': self.backup_enabled,
                'health_checks_enabled': self.health_check_enabled
            }
        }
    
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        use_enum_values = True


# Default database settings factory
def get_database_settings() -> DatabaseSettings:
    """
    Get database settings with environment-specific defaults.
    
    Returns:
        Configured DatabaseSettings instance
    """
    return DatabaseSettings()