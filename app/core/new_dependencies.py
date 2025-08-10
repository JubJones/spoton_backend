"""
Enhanced dependency injection container for the refactored architecture.

This module introduces modern DI patterns while maintaining compatibility
with the existing dependencies.py. Gradually migrate to this system.
"""
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Type, TypeVar, Generic, Protocol, Any
import logging

from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)

# Type definitions for dependency injection
ServiceType = TypeVar('ServiceType')
ImplementationType = TypeVar('ImplementationType')


class ServiceContainer:
    """
    Dependency injection container for managing service lifecycles.
    
    Provides registration and resolution of services with different lifecycles:
    - Singleton: Single instance per application
    - Scoped: Single instance per request
    - Transient: New instance every time
    """
    
    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._scoped: Dict[Type, Any] = {}
        self._transient_factories: Dict[Type, callable] = {}
        self._interfaces: Dict[Type, Type] = {}
    
    def register_singleton(self, interface: Type[ServiceType], implementation: ServiceType) -> None:
        """Register a singleton service instance."""
        self._singletons[interface] = implementation
        logger.debug(f"Registered singleton: {interface.__name__} -> {type(implementation).__name__}")
    
    def register_scoped(self, interface: Type[ServiceType], factory: callable) -> None:
        """Register a scoped service factory."""
        self._scoped[interface] = factory
        logger.debug(f"Registered scoped service: {interface.__name__}")
    
    def register_transient(self, interface: Type[ServiceType], factory: callable) -> None:
        """Register a transient service factory."""
        self._transient_factories[interface] = factory
        logger.debug(f"Registered transient service: {interface.__name__}")
    
    def register_interface(self, interface: Type, implementation: Type) -> None:
        """Register interface to implementation mapping."""
        self._interfaces[interface] = implementation
        logger.debug(f"Registered interface: {interface.__name__} -> {implementation.__name__}")
    
    def get(self, service_type: Type[ServiceType]) -> ServiceType:
        """
        Resolve a service instance.
        
        Args:
            service_type: The service interface or implementation type
            
        Returns:
            Service instance
            
        Raises:
            HTTPException: If service cannot be resolved
        """
        try:
            # Check singletons first
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            # Check if we have an interface mapping
            if service_type in self._interfaces:
                impl_type = self._interfaces[service_type]
                if impl_type in self._singletons:
                    return self._singletons[impl_type]
            
            # Check transient services
            if service_type in self._transient_factories:
                return self._transient_factories[service_type]()
            
            # If nothing found, raise exception
            raise KeyError(f"Service not registered: {service_type.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to resolve service {service_type.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service unavailable: {service_type.__name__}"
            )
    
    def has_service(self, service_type: Type) -> bool:
        """Check if a service is registered."""
        return (
            service_type in self._singletons or
            service_type in self._scoped or
            service_type in self._transient_factories or
            service_type in self._interfaces
        )


# Global service container instance
_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """Get the global service container instance."""
    return _container


class ServiceFactory(Protocol):
    """Protocol for service factories."""
    
    def __call__(self, container: ServiceContainer) -> Any:
        """Create service instance using container."""
        ...


class BaseServiceFactory(ABC):
    """Base class for service factories with dependency resolution."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
    
    @abstractmethod
    def create(self) -> Any:
        """Create service instance."""
        pass


# AI Model Factory Pattern
class AIModelFactory:
    """
    Factory for creating AI model instances.
    
    Centralizes AI model creation and configuration while maintaining
    compatibility with existing model loading.
    """
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self._model_registry: Dict[str, Type] = {}
    
    def register_model(self, model_type: str, model_class: Type) -> None:
        """Register a model implementation."""
        self._model_registry[model_type] = model_class
        logger.debug(f"Registered AI model: {model_type} -> {model_class.__name__}")
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        """
        Create AI model instance.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model configuration parameters
            
        Returns:
            Configured model instance
        """
        if model_type not in self._model_registry:
            available = ", ".join(self._model_registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        
        model_class = self._model_registry[model_type]
        return model_class(**kwargs)
    
    def get_available_models(self) -> list[str]:
        """Get list of available model types."""
        return list(self._model_registry.keys())


# Repository Factory Pattern
class RepositoryFactory:
    """Factory for creating repository instances with proper DI."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
    
    def create_repository(self, repository_type: Type[ServiceType]) -> ServiceType:
        """Create repository instance with dependencies."""
        return self.container.get(repository_type)


# FastAPI dependency functions
def get_service_container() -> ServiceContainer:
    """FastAPI dependency to get service container."""
    return _container


def get_ai_model_factory(
    container: ServiceContainer = get_service_container()
) -> AIModelFactory:
    """FastAPI dependency to get AI model factory."""
    return AIModelFactory(container)


def get_repository_factory(
    container: ServiceContainer = get_service_container()
) -> RepositoryFactory:
    """FastAPI dependency to get repository factory."""
    return RepositoryFactory(container)


# Service registration helper functions
def configure_services(container: ServiceContainer) -> None:
    """
    Configure services in the container.
    
    This function will be called during application startup to register
    all services with their appropriate lifecycles.
    """
    # AI Model Factory setup
    ai_factory = AIModelFactory(container)
    container.register_singleton(AIModelFactory, ai_factory)
    
    # Repository Factory setup
    repo_factory = RepositoryFactory(container)
    container.register_singleton(RepositoryFactory, repo_factory)
    
    logger.info("New dependency injection container configured")


# Compatibility bridge with existing dependencies
class LegacyServiceBridge:
    """
    Bridge to integrate legacy services with new DI container.
    
    Provides backward compatibility while migrating to new architecture.
    """
    
    def __init__(self, container: ServiceContainer):
        self.container = container
    
    def register_legacy_service(self, service_type: Type, service_instance: Any) -> None:
        """Register a legacy service instance."""
        self.container.register_singleton(service_type, service_instance)
    
    def migrate_from_request_state(self, request: Request) -> None:
        """
        Migrate services from FastAPI request.app.state to container.
        
        This allows gradual migration from the old state-based approach
        to the new container-based approach.
        """
        # Migrate compute device if available
        if hasattr(request.app.state, 'compute_device'):
            # Would register device with appropriate interface
            pass
        
        # Migrate other services as needed during transition
        logger.debug("Legacy services migrated to new container")