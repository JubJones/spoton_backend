"""Geometric utilities for space-based re-identification."""

from .bottom_point_extractor import BottomPointExtractor, ImagePoint
from .world_plane_transformer import WorldPlaneTransformer, WorldPoint
from .roi_calculator import ROICalculator, SearchROI, ROIShape
from .geometric_matcher import GeometricMatcher, MatchResult, MatchType
from .metrics_collector import MetricsCollector, GeometricMatchingMetrics

__all__ = [
    "BottomPointExtractor",
    "ImagePoint",
    "WorldPlaneTransformer",
    "WorldPoint",
    "ROICalculator",
    "SearchROI",
    "ROIShape",
    "GeometricMatcher",
    "MatchResult",
    "MatchType",
    "MetricsCollector",
    "GeometricMatchingMetrics",
]
