"""
Repository interfaces for the application layer.

These interfaces define contracts for data access without coupling
the application to specific infrastructure implementations.
"""
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
from uuid import UUID

from app.domain.shared.entities.base_entity import BaseEntity

# Generic type for entities
EntityType = TypeVar('EntityType', bound=BaseEntity)


class BaseRepository(ABC, Generic[EntityType]):
    """
    Base repository interface for entity persistence.
    
    Provides CRUD operations for domain entities while maintaining
    the Repository pattern to abstract data access concerns.
    """
    
    @abstractmethod
    async def save(self, entity: EntityType) -> EntityType:
        """
        Save entity to persistence layer.
        
        Args:
            entity: Entity to save
            
        Returns:
            Saved entity with updated metadata
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, entity_id: UUID) -> Optional[EntityType]:
        """
        Find entity by its unique identifier.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_all(self) -> List[EntityType]:
        """
        Find all entities of this type.
        
        Returns:
            List of all entities
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            True if exists, False otherwise
        """
        pass


class UnitOfWork(ABC):
    """
    Unit of Work pattern interface for transaction management.
    
    Maintains a list of objects affected by a business transaction
    and coordinates writing out changes and resolving concurrency issues.
    """
    
    @abstractmethod
    async def begin(self) -> None:
        """Begin a new transaction."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit current transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback current transaction."""
        pass
    
    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        await self.begin()
        return self
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()