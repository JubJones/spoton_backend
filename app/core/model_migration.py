"""
Model migration helpers for transitioning from legacy models to unified domain models.

Provides backward compatibility during the refactoring process while ensuring
zero-downtime migration from old BoundingBox and Detection models.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging
from datetime import datetime

from app.domain.shared.value_objects.bounding_box import BoundingBox
from app.models.base_models import BoundingBox as LegacyBoundingBox
from app.models.base_models import Detection as LegacyDetection
from app.models.base_models import TrackedObject as LegacyTrackedObject

logger = logging.getLogger(__name__)


class ModelMigrationHelper:
    """
    Helper class for migrating between legacy and unified domain models.
    
    Provides conversion methods to maintain backward compatibility while
    migrating to the new unified domain model architecture.
    """
    
    @staticmethod
    def migrate_legacy_bbox_to_unified(legacy_bbox: LegacyBoundingBox) -> BoundingBox:
        """
        Convert legacy BoundingBox to unified BoundingBox.
        
        Args:
            legacy_bbox: Legacy BoundingBox instance (x1, y1, x2, y2 format)
            
        Returns:
            Unified BoundingBox instance
        """
        try:
            return BoundingBox.from_coordinates(
                x1=legacy_bbox.x1,
                y1=legacy_bbox.y1,
                x2=legacy_bbox.x2,
                y2=legacy_bbox.y2,
                normalized=False
            )
        except Exception as e:
            logger.error(f"Failed to migrate legacy BoundingBox: {e}")
            raise ValueError(f"BoundingBox migration failed: {e}")
    
    @staticmethod
    def migrate_unified_bbox_to_legacy(unified_bbox: BoundingBox) -> LegacyBoundingBox:
        """
        Convert unified BoundingBox to legacy BoundingBox.
        
        Args:
            unified_bbox: Unified BoundingBox instance
            
        Returns:
            Legacy BoundingBox instance
        """
        try:
            x1, y1, x2, y2 = unified_bbox.to_coordinates()
            return LegacyBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except Exception as e:
            logger.error(f"Failed to migrate unified BoundingBox to legacy: {e}")
            raise ValueError(f"BoundingBox migration to legacy failed: {e}")
    
    @staticmethod
    def migrate_detection_to_legacy(
        unified_bbox: BoundingBox,
        confidence: float,
        class_id: Any,
        class_name: Optional[str] = None
    ) -> LegacyDetection:
        """
        Create legacy Detection from unified components.
        
        Args:
            unified_bbox: Unified BoundingBox
            confidence: Detection confidence
            class_id: Class identifier
            class_name: Optional class name
            
        Returns:
            Legacy Detection instance
        """
        try:
            legacy_bbox = ModelMigrationHelper.migrate_unified_bbox_to_legacy(unified_bbox)
            return LegacyDetection(
                bbox=legacy_bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            )
        except Exception as e:
            logger.error(f"Failed to create legacy Detection: {e}")
            raise ValueError(f"Detection migration to legacy failed: {e}")
    
    @staticmethod
    def migrate_tracked_object_to_legacy(
        unified_bbox: BoundingBox,
        track_id: int,
        confidence: Optional[float] = None,
        class_id: Optional[Any] = None,
        global_id: Optional[int] = None,
        feature_embedding: Optional[np.ndarray] = None,
        state: Optional[str] = None,
        age: Optional[int] = None
    ) -> LegacyTrackedObject:
        """
        Create legacy TrackedObject from unified components.
        
        Args:
            unified_bbox: Unified BoundingBox
            track_id: Track identifier
            confidence: Optional confidence
            class_id: Optional class identifier
            global_id: Optional global identifier
            feature_embedding: Optional feature vector
            state: Optional state
            age: Optional age
            
        Returns:
            Legacy TrackedObject instance
        """
        try:
            legacy_bbox = ModelMigrationHelper.migrate_unified_bbox_to_legacy(unified_bbox)
            return LegacyTrackedObject(
                track_id=track_id,
                bbox=legacy_bbox,
                confidence=confidence,
                class_id=class_id,
                global_id=global_id,
                feature_embedding=feature_embedding,
                state=state,
                age=age
            )
        except Exception as e:
            logger.error(f"Failed to create legacy TrackedObject: {e}")
            raise ValueError(f"TrackedObject migration to legacy failed: {e}")
    
    @staticmethod
    def extract_bbox_from_legacy_detection(legacy_detection: LegacyDetection) -> BoundingBox:
        """
        Extract unified BoundingBox from legacy Detection.
        
        Args:
            legacy_detection: Legacy Detection instance
            
        Returns:
            Unified BoundingBox
        """
        return ModelMigrationHelper.migrate_legacy_bbox_to_unified(legacy_detection.bbox)
    
    @staticmethod
    def extract_bbox_from_legacy_tracked_object(legacy_tracked: LegacyTrackedObject) -> BoundingBox:
        """
        Extract unified BoundingBox from legacy TrackedObject.
        
        Args:
            legacy_tracked: Legacy TrackedObject instance
            
        Returns:
            Unified BoundingBox
        """
        return ModelMigrationHelper.migrate_legacy_bbox_to_unified(legacy_tracked.bbox)


class LegacyBoundingBoxWrapper:
    """
    Wrapper class that provides legacy BoundingBox interface while using unified model internally.
    
    This allows gradual migration by providing backward compatibility for existing code
    that expects the legacy BoundingBox interface.
    """
    
    def __init__(self, unified_bbox: BoundingBox):
        """
        Initialize wrapper with unified BoundingBox.
        
        Args:
            unified_bbox: Unified BoundingBox instance
        """
        self._bbox = unified_bbox
    
    # Legacy property interface
    @property
    def x1(self) -> float:
        """Left edge coordinate."""
        return self._bbox.x1
    
    @property
    def y1(self) -> float:
        """Top edge coordinate."""
        return self._bbox.y1
    
    @property
    def x2(self) -> float:
        """Right edge coordinate."""
        return self._bbox.x2
    
    @property
    def y2(self) -> float:
        """Bottom edge coordinate."""
        return self._bbox.y2
    
    # Legacy method interface
    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        return self._bbox.to_xywh()
    
    def to_list(self) -> List[float]:
        """Convert to list format."""
        return self._bbox.to_list()
    
    # Access to unified model
    @property
    def unified(self) -> BoundingBox:
        """Get the underlying unified BoundingBox."""
        return self._bbox
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LegacyBoundingBoxWrapper({self._bbox})"


class CompatibilityService:
    """
    Service for managing compatibility between legacy and unified models.
    
    Provides centralized logic for handling model compatibility during migration.
    """
    
    def __init__(self):
        self._migration_stats = {
            'bbox_migrations': 0,
            'detection_migrations': 0,
            'tracked_object_migrations': 0,
            'errors': 0
        }
    
    def create_legacy_compatible_bbox(self, unified_bbox: BoundingBox) -> LegacyBoundingBoxWrapper:
        """
        Create legacy-compatible bounding box wrapper.
        
        Args:
            unified_bbox: Unified BoundingBox
            
        Returns:
            Legacy-compatible wrapper
        """
        try:
            self._migration_stats['bbox_migrations'] += 1
            return LegacyBoundingBoxWrapper(unified_bbox)
        except Exception as e:
            self._migration_stats['errors'] += 1
            logger.error(f"Failed to create legacy-compatible bbox: {e}")
            raise
    
    def batch_migrate_bboxes(self, legacy_bboxes: List[LegacyBoundingBox]) -> List[BoundingBox]:
        """
        Batch migrate legacy bounding boxes to unified format.
        
        Args:
            legacy_bboxes: List of legacy BoundingBox instances
            
        Returns:
            List of unified BoundingBox instances
        """
        unified_bboxes = []
        errors = []
        
        for i, legacy_bbox in enumerate(legacy_bboxes):
            try:
                unified_bbox = ModelMigrationHelper.migrate_legacy_bbox_to_unified(legacy_bbox)
                unified_bboxes.append(unified_bbox)
                self._migration_stats['bbox_migrations'] += 1
            except Exception as e:
                errors.append((i, str(e)))
                self._migration_stats['errors'] += 1
                logger.warning(f"Failed to migrate bbox at index {i}: {e}")
        
        if errors:
            logger.warning(f"Batch migration completed with {len(errors)} errors")
        
        return unified_bboxes
    
    def get_migration_stats(self) -> Dict[str, int]:
        """Get migration statistics."""
        return self._migration_stats.copy()
    
    def reset_migration_stats(self) -> None:
        """Reset migration statistics."""
        self._migration_stats = {
            'bbox_migrations': 0,
            'detection_migrations': 0,
            'tracked_object_migrations': 0,
            'errors': 0
        }


# Global compatibility service instance
_compatibility_service = CompatibilityService()


def get_compatibility_service() -> CompatibilityService:
    """Get the global compatibility service instance."""
    return _compatibility_service


# Convenience functions for common migrations
def migrate_bbox(legacy_bbox: LegacyBoundingBox) -> BoundingBox:
    """Convenience function to migrate legacy bbox."""
    return ModelMigrationHelper.migrate_legacy_bbox_to_unified(legacy_bbox)


def create_legacy_bbox(unified_bbox: BoundingBox) -> LegacyBoundingBox:
    """Convenience function to create legacy bbox."""
    return ModelMigrationHelper.migrate_unified_bbox_to_legacy(unified_bbox)


def wrap_for_legacy(unified_bbox: BoundingBox) -> LegacyBoundingBoxWrapper:
    """Convenience function to wrap unified bbox for legacy compatibility."""
    return _compatibility_service.create_legacy_compatible_bbox(unified_bbox)