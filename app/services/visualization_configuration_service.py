"""
Visualization Configuration Management Service

Manages all visualization-related configurations including:
- Overlay settings and visual preferences
- Camera-specific visualization configs
- User-specific display preferences
- Environment-based configuration templates
- Real-time configuration updates
- Performance optimization settings
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json

from app.domains.visualization.entities.overlay_config import OverlayConfig
from app.infrastructure.cache.tracking_cache import TrackingCache
from app.api.v1.visualization_schemas import (
    OverlayType,
    VisualizationMode,
    FocusTrackMode
)

logger = logging.getLogger(__name__)


@dataclass
class CameraVisualizationConfig:
    """Configuration for camera-specific visualization settings."""
    camera_id: str
    
    # Overlay settings
    overlay_config: OverlayConfig = field(default_factory=OverlayConfig)
    
    # Display settings
    frame_quality: int = 85
    frame_rate_limit: float = 30.0
    enable_compression: bool = True
    
    # Visual enhancements
    auto_brightness: bool = True
    contrast_adjustment: float = 1.0
    color_enhancement: bool = False
    
    # Focus tracking
    focus_highlight_color: str = "#FF0000"
    focus_highlight_thickness: int = 3
    focus_follow_enabled: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_duration_seconds: int = 30
    adaptive_quality: bool = True
    
    # Last updated
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        config_dict = asdict(self)
        config_dict['last_modified'] = self.last_modified.isoformat()
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraVisualizationConfig':
        """Create from dictionary."""
        # Handle datetime parsing
        if 'last_modified' in data and isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        
        # Handle overlay_config
        if 'overlay_config' in data and isinstance(data['overlay_config'], dict):
            overlay_data = data['overlay_config']
            data['overlay_config'] = OverlayConfig(**overlay_data)
        
        return cls(**data)


@dataclass
class EnvironmentVisualizationConfig:
    """Configuration for environment-wide visualization settings."""
    environment_id: str
    
    # Global display settings
    visualization_mode: VisualizationMode = VisualizationMode.LIVE
    sync_all_cameras: bool = True
    global_frame_rate: float = 25.0
    
    # Multi-camera layout
    default_layout: str = "grid"
    camera_grid_size: tuple = (2, 2)
    enable_camera_switching: bool = True
    
    # Map visualization
    show_unified_map: bool = True
    map_overlay_opacity: float = 0.8
    show_trajectories: bool = True
    trajectory_length: int = 50
    
    # Analytics display
    show_person_count: bool = True
    show_zone_occupancy: bool = True
    show_performance_metrics: bool = False
    
    # Focus tracking
    default_focus_mode: FocusTrackMode = FocusTrackMode.SINGLE_PERSON
    cross_camera_focus: bool = True
    auto_focus_timeout_seconds: int = 300
    
    # Quality settings
    adaptive_quality_enabled: bool = True
    min_quality: int = 65
    max_quality: int = 95
    
    # Camera configurations
    camera_configs: Dict[str, CameraVisualizationConfig] = field(default_factory=dict)
    
    # Last updated
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    def get_camera_config(self, camera_id: str) -> CameraVisualizationConfig:
        """Get or create camera configuration."""
        if camera_id not in self.camera_configs:
            self.camera_configs[camera_id] = CameraVisualizationConfig(
                camera_id=camera_id
            )
            self.last_modified = datetime.utcnow()
        
        return self.camera_configs[camera_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        config_dict = asdict(self)
        config_dict['last_modified'] = self.last_modified.isoformat()
        
        # Convert enum values
        config_dict['visualization_mode'] = self.visualization_mode.value
        config_dict['default_focus_mode'] = self.default_focus_mode.value
        
        # Convert camera configs
        config_dict['camera_configs'] = {
            camera_id: camera_config.to_dict()
            for camera_id, camera_config in self.camera_configs.items()
        }
        
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentVisualizationConfig':
        """Create from dictionary."""
        # Handle datetime parsing
        if 'last_modified' in data and isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        
        # Handle enum values
        if 'visualization_mode' in data:
            data['visualization_mode'] = VisualizationMode(data['visualization_mode'])
        
        if 'default_focus_mode' in data:
            data['default_focus_mode'] = FocusTrackMode(data['default_focus_mode'])
        
        # Handle camera configs
        if 'camera_configs' in data and isinstance(data['camera_configs'], dict):
            camera_configs = {}
            for camera_id, camera_data in data['camera_configs'].items():
                camera_configs[camera_id] = CameraVisualizationConfig.from_dict(camera_data)
            data['camera_configs'] = camera_configs
        
        return cls(**data)


@dataclass
class UserVisualizationPreferences:
    """User-specific visualization preferences."""
    user_id: str
    
    # Personal preferences
    preferred_quality: int = 85
    preferred_frame_rate: float = 25.0
    enable_sound_alerts: bool = False
    
    # Display preferences
    dark_mode: bool = False
    color_scheme: str = "default"
    font_size: str = "medium"
    
    # Feature preferences
    auto_focus_enabled: bool = True
    show_person_ids: bool = True
    show_confidence_scores: bool = False
    show_trajectories: bool = True
    
    # Notification preferences
    alert_on_new_person: bool = False
    alert_on_zone_breach: bool = True
    alert_sound_enabled: bool = False
    
    # Performance preferences
    prefer_quality_over_speed: bool = True
    enable_predictive_loading: bool = True
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Last updated
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    def get_environment_override(
        self,
        environment_id: str,
        setting_name: str,
        default_value: Any = None
    ) -> Any:
        """Get environment-specific override for a setting."""
        env_overrides = self.environment_overrides.get(environment_id, {})
        return env_overrides.get(setting_name, default_value)
    
    def set_environment_override(
        self,
        environment_id: str,
        setting_name: str,
        value: Any
    ):
        """Set environment-specific override for a setting."""
        if environment_id not in self.environment_overrides:
            self.environment_overrides[environment_id] = {}
        
        self.environment_overrides[environment_id][setting_name] = value
        self.last_modified = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        config_dict = asdict(self)
        config_dict['last_modified'] = self.last_modified.isoformat()
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserVisualizationPreferences':
        """Create from dictionary."""
        # Handle datetime parsing
        if 'last_modified' in data and isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        
        return cls(**data)


class VisualizationConfigurationService:
    """Service for managing visualization configurations and preferences."""
    
    def __init__(self, cache: TrackingCache):
        self.cache = cache
        
        # In-memory configuration storage
        self.environment_configs: Dict[str, EnvironmentVisualizationConfig] = {}
        self.user_preferences: Dict[str, UserVisualizationPreferences] = {}
        
        # Configuration templates
        self.config_templates = self._load_default_templates()
        
        # Change tracking
        self.config_change_listeners: Set[callable] = set()
        
        # Performance metrics
        self.config_access_count = 0
        self.cache_hit_count = 0
        
        logger.info("VisualizationConfigurationService initialized")
    
    # --- Environment Configuration Management ---
    
    async def get_environment_config(
        self,
        environment_id: str
    ) -> EnvironmentVisualizationConfig:
        """Get environment visualization configuration."""
        try:
            # Check in-memory cache first
            if environment_id in self.environment_configs:
                config = self.environment_configs[environment_id]
                self.config_access_count += 1
                return config
            
            # Try to load from persistent cache
            cached_config = await self._load_environment_config_from_cache(environment_id)
            if cached_config:
                self.environment_configs[environment_id] = cached_config
                self.cache_hit_count += 1
                return cached_config
            
            # Create default configuration
            default_config = self._create_default_environment_config(environment_id)
            self.environment_configs[environment_id] = default_config
            
            # Cache the default configuration
            await self._save_environment_config_to_cache(environment_id, default_config)
            
            logger.info(f"Created default environment configuration for {environment_id}")
            return default_config
            
        except Exception as e:
            logger.error(f"Error getting environment config: {e}")
            # Return minimal default config
            return EnvironmentVisualizationConfig(environment_id=environment_id)
    
    async def update_environment_config(
        self,
        environment_id: str,
        config_updates: Dict[str, Any]
    ) -> EnvironmentVisualizationConfig:
        """Update environment configuration with new settings."""
        try:
            # Get current configuration
            current_config = await self.get_environment_config(environment_id)
            
            # Apply updates
            for key, value in config_updates.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Update timestamp
            current_config.last_modified = datetime.utcnow()
            
            # Save updated configuration
            await self._save_environment_config_to_cache(environment_id, current_config)
            
            # Notify listeners
            await self._notify_config_change(environment_id, "environment", config_updates)
            
            logger.info(f"Updated environment configuration for {environment_id}")
            return current_config
            
        except Exception as e:
            logger.error(f"Error updating environment config: {e}")
            raise
    
    async def get_camera_config(
        self,
        environment_id: str,
        camera_id: str
    ) -> CameraVisualizationConfig:
        """Get camera-specific visualization configuration."""
        try:
            env_config = await self.get_environment_config(environment_id)
            return env_config.get_camera_config(camera_id)
            
        except Exception as e:
            logger.error(f"Error getting camera config: {e}")
            return CameraVisualizationConfig(camera_id=camera_id)
    
    async def update_camera_config(
        self,
        environment_id: str,
        camera_id: str,
        config_updates: Dict[str, Any]
    ) -> CameraVisualizationConfig:
        """Update camera-specific configuration."""
        try:
            # Get environment config
            env_config = await self.get_environment_config(environment_id)
            
            # Get camera config
            camera_config = env_config.get_camera_config(camera_id)
            
            # Apply updates
            for key, value in config_updates.items():
                if key == "overlay_config" and isinstance(value, dict):
                    # Update overlay config
                    overlay_config = camera_config.overlay_config
                    for overlay_key, overlay_value in value.items():
                        if hasattr(overlay_config, overlay_key):
                            setattr(overlay_config, overlay_key, overlay_value)
                elif hasattr(camera_config, key):
                    setattr(camera_config, key, value)
                else:
                    logger.warning(f"Unknown camera configuration key: {key}")
            
            # Update timestamps
            camera_config.last_modified = datetime.utcnow()
            env_config.last_modified = datetime.utcnow()
            
            # Save configuration
            await self._save_environment_config_to_cache(environment_id, env_config)
            
            # Notify listeners
            await self._notify_config_change(
                environment_id, "camera", config_updates, camera_id=camera_id
            )
            
            logger.info(f"Updated camera configuration for {environment_id}:{camera_id}")
            return camera_config
            
        except Exception as e:
            logger.error(f"Error updating camera config: {e}")
            raise
    
    # --- User Preferences Management ---
    
    async def get_user_preferences(self, user_id: str) -> UserVisualizationPreferences:
        """Get user visualization preferences."""
        try:
            # Check in-memory cache first
            if user_id in self.user_preferences:
                preferences = self.user_preferences[user_id]
                self.config_access_count += 1
                return preferences
            
            # Try to load from persistent cache
            cached_preferences = await self._load_user_preferences_from_cache(user_id)
            if cached_preferences:
                self.user_preferences[user_id] = cached_preferences
                self.cache_hit_count += 1
                return cached_preferences
            
            # Create default preferences
            default_preferences = self._create_default_user_preferences(user_id)
            self.user_preferences[user_id] = default_preferences
            
            # Cache the default preferences
            await self._save_user_preferences_to_cache(user_id, default_preferences)
            
            logger.info(f"Created default user preferences for {user_id}")
            return default_preferences
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            # Return minimal default preferences
            return UserVisualizationPreferences(user_id=user_id)
    
    async def update_user_preferences(
        self,
        user_id: str,
        preference_updates: Dict[str, Any]
    ) -> UserVisualizationPreferences:
        """Update user visualization preferences."""
        try:
            # Get current preferences
            current_preferences = await self.get_user_preferences(user_id)
            
            # Apply updates
            for key, value in preference_updates.items():
                if hasattr(current_preferences, key):
                    setattr(current_preferences, key, value)
                else:
                    logger.warning(f"Unknown preference key: {key}")
            
            # Update timestamp
            current_preferences.last_modified = datetime.utcnow()
            
            # Save updated preferences
            await self._save_user_preferences_to_cache(user_id, current_preferences)
            
            # Notify listeners
            await self._notify_config_change(user_id, "user_preferences", preference_updates)
            
            logger.info(f"Updated user preferences for {user_id}")
            return current_preferences
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            raise
    
    # --- Configuration Templates ---
    
    def get_config_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration template by name."""
        return self.config_templates.get(template_name)
    
    def list_config_templates(self) -> List[str]:
        """List available configuration templates."""
        return list(self.config_templates.keys())
    
    async def apply_config_template(
        self,
        environment_id: str,
        template_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> EnvironmentVisualizationConfig:
        """Apply configuration template to environment."""
        try:
            template = self.get_config_template(template_name)
            if not template:
                raise ValueError(f"Template '{template_name}' not found")
            
            # Create configuration from template
            config_data = template.copy()
            config_data['environment_id'] = environment_id
            
            # Apply overrides
            if overrides:
                config_data.update(overrides)
            
            # Create configuration object
            config = EnvironmentVisualizationConfig.from_dict(config_data)
            
            # Save configuration
            self.environment_configs[environment_id] = config
            await self._save_environment_config_to_cache(environment_id, config)
            
            # Notify listeners
            await self._notify_config_change(
                environment_id, "template_applied", {"template": template_name}
            )
            
            logger.info(f"Applied template '{template_name}' to environment {environment_id}")
            return config
            
        except Exception as e:
            logger.error(f"Error applying config template: {e}")
            raise
    
    # --- Real-time Configuration Updates ---
    
    def add_config_change_listener(self, listener: callable):
        """Add listener for configuration changes."""
        self.config_change_listeners.add(listener)
    
    def remove_config_change_listener(self, listener: callable):
        """Remove configuration change listener."""
        self.config_change_listeners.discard(listener)
    
    async def _notify_config_change(
        self,
        entity_id: str,
        change_type: str,
        changes: Dict[str, Any],
        **kwargs
    ):
        """Notify all listeners of configuration changes."""
        try:
            change_event = {
                'entity_id': entity_id,
                'change_type': change_type,
                'changes': changes,
                'timestamp': datetime.utcnow().isoformat(),
                **kwargs
            }
            
            # Notify all listeners
            for listener in self.config_change_listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(change_event)
                    else:
                        listener(change_event)
                except Exception as e:
                    logger.error(f"Error in config change listener: {e}")
            
        except Exception as e:
            logger.error(f"Error notifying config change: {e}")
    
    # --- Cache Management ---
    
    async def _load_environment_config_from_cache(
        self,
        environment_id: str
    ) -> Optional[EnvironmentVisualizationConfig]:
        """Load environment config from cache."""
        try:
            cache_key = f"env_config_{environment_id}"
            cached_data = await self.cache.get_json(cache_key)
            
            if cached_data:
                return EnvironmentVisualizationConfig.from_dict(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading environment config from cache: {e}")
            return None
    
    async def _save_environment_config_to_cache(
        self,
        environment_id: str,
        config: EnvironmentVisualizationConfig
    ):
        """Save environment config to cache."""
        try:
            cache_key = f"env_config_{environment_id}"
            config_data = config.to_dict()
            
            # Cache for 24 hours
            await self.cache.set_json(cache_key, config_data, ttl=86400)
            
        except Exception as e:
            logger.error(f"Error saving environment config to cache: {e}")
    
    async def _load_user_preferences_from_cache(
        self,
        user_id: str
    ) -> Optional[UserVisualizationPreferences]:
        """Load user preferences from cache."""
        try:
            cache_key = f"user_prefs_{user_id}"
            cached_data = await self.cache.get_json(cache_key)
            
            if cached_data:
                return UserVisualizationPreferences.from_dict(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading user preferences from cache: {e}")
            return None
    
    async def _save_user_preferences_to_cache(
        self,
        user_id: str,
        preferences: UserVisualizationPreferences
    ):
        """Save user preferences to cache."""
        try:
            cache_key = f"user_prefs_{user_id}"
            preferences_data = preferences.to_dict()
            
            # Cache for 7 days
            await self.cache.set_json(cache_key, preferences_data, ttl=604800)
            
        except Exception as e:
            logger.error(f"Error saving user preferences to cache: {e}")
    
    # --- Default Configuration Creation ---
    
    def _create_default_environment_config(
        self,
        environment_id: str
    ) -> EnvironmentVisualizationConfig:
        """Create default environment configuration."""
        return EnvironmentVisualizationConfig(
            environment_id=environment_id,
            visualization_mode=VisualizationMode.LIVE,
            sync_all_cameras=True,
            global_frame_rate=25.0,
            default_layout="grid",
            camera_grid_size=(2, 2),
            show_unified_map=True,
            show_person_count=True,
            show_zone_occupancy=True,
            adaptive_quality_enabled=True,
            min_quality=70,
            max_quality=95
        )
    
    def _create_default_user_preferences(self, user_id: str) -> UserVisualizationPreferences:
        """Create default user preferences."""
        return UserVisualizationPreferences(
            user_id=user_id,
            preferred_quality=85,
            preferred_frame_rate=25.0,
            dark_mode=False,
            color_scheme="default",
            auto_focus_enabled=True,
            show_person_ids=True,
            show_trajectories=True,
            prefer_quality_over_speed=True
        )
    
    def _load_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load default configuration templates."""
        templates = {
            "campus_default": {
                "visualization_mode": "live",
                "sync_all_cameras": True,
                "global_frame_rate": 25.0,
                "default_layout": "grid",
                "camera_grid_size": [2, 2],
                "show_unified_map": True,
                "show_person_count": True,
                "show_zone_occupancy": True,
                "adaptive_quality_enabled": True,
                "min_quality": 75,
                "max_quality": 95
            },
            "factory_default": {
                "visualization_mode": "live",
                "sync_all_cameras": True,
                "global_frame_rate": 30.0,
                "default_layout": "grid",
                "camera_grid_size": [3, 2],
                "show_unified_map": True,
                "show_person_count": True,
                "show_zone_occupancy": True,
                "show_performance_metrics": True,
                "adaptive_quality_enabled": True,
                "min_quality": 80,
                "max_quality": 95
            },
            "security_monitoring": {
                "visualization_mode": "live",
                "sync_all_cameras": False,
                "global_frame_rate": 30.0,
                "default_layout": "single",
                "show_unified_map": True,
                "show_person_count": True,
                "show_zone_occupancy": True,
                "show_performance_metrics": False,
                "cross_camera_focus": True,
                "adaptive_quality_enabled": False,
                "min_quality": 90,
                "max_quality": 95
            },
            "high_performance": {
                "visualization_mode": "live",
                "sync_all_cameras": True,
                "global_frame_rate": 60.0,
                "adaptive_quality_enabled": True,
                "min_quality": 60,
                "max_quality": 85,
                "show_performance_metrics": True
            }
        }
        
        return templates
    
    # --- Service Status and Management ---
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and statistics."""
        cache_hit_rate = (
            self.cache_hit_count / max(1, self.config_access_count) * 100
        )
        
        return {
            "service_name": "VisualizationConfigurationService",
            "environment_configs_loaded": len(self.environment_configs),
            "user_preferences_loaded": len(self.user_preferences),
            "available_templates": len(self.config_templates),
            "config_access_count": self.config_access_count,
            "cache_hit_rate_percent": cache_hit_rate,
            "active_listeners": len(self.config_change_listeners),
            "template_names": list(self.config_templates.keys())
        }
    
    async def reload_configurations(self):
        """Reload all configurations from cache."""
        try:
            # Clear in-memory configurations
            old_env_count = len(self.environment_configs)
            old_user_count = len(self.user_preferences)
            
            self.environment_configs.clear()
            self.user_preferences.clear()
            
            logger.info(
                f"Reloaded configurations: cleared {old_env_count} environment configs "
                f"and {old_user_count} user preferences"
            )
            
        except Exception as e:
            logger.error(f"Error reloading configurations: {e}")
    
    async def cleanup_expired_configurations(self):
        """Clean up expired configurations from cache."""
        try:
            # This would implement cache cleanup logic
            # For now, just log the action
            logger.info("Cleaned up expired configurations")
            
        except Exception as e:
            logger.error(f"Error cleaning up configurations: {e}")


# Global service instance
_config_service: Optional[VisualizationConfigurationService] = None


def get_config_service() -> Optional[VisualizationConfigurationService]:
    """Get the global configuration service instance."""
    return _config_service


def initialize_config_service(cache: TrackingCache) -> VisualizationConfigurationService:
    """Initialize the global configuration service."""
    global _config_service
    if _config_service is None:
        _config_service = VisualizationConfigurationService(cache)
    return _config_service