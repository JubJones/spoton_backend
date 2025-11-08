"""
Metrics collector for geometric matching (Phase 5).
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .geometric_matcher import MatchResult, MatchType


@dataclass
class GeometricMatchingMetrics:
    """Aggregated metrics for geometric matching system."""

    total_matches: int = 0
    exact_matches: int = 0
    closest_matches: int = 0
    no_matches: int = 0

    overall_success_rate: float = 0.0
    exact_match_rate: float = 0.0
    closest_match_rate: float = 0.0
    no_match_rate: float = 0.0

    avg_distance: float = 0.0
    min_distance: float = 0.0
    max_distance: float = 0.0
    median_distance: float = 0.0

    avg_confidence: float = 0.0
    high_confidence_matches: int = 0

    avg_roi_radius: float = 0.0
    avg_candidates_per_roi: float = 0.0

    avg_processing_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0

    transformation_success_rate: float = 0.0
    avg_transformation_quality: float = 0.0

    avg_reprojection_error_px: float = 0.0
    max_reprojection_error_px: float = 0.0
    reprojection_samples: int = 0
    reprojection_events: int = 0
    reprojection_missing_actual: int = 0

    camera_pair_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_matches": self.total_matches,
            "exact_matches": self.exact_matches,
            "closest_matches": self.closest_matches,
            "no_matches": self.no_matches,
            "overall_success_rate": self.overall_success_rate,
            "exact_match_rate": self.exact_match_rate,
            "closest_match_rate": self.closest_match_rate,
            "no_match_rate": self.no_match_rate,
            "avg_distance": self.avg_distance,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "median_distance": self.median_distance,
            "avg_confidence": self.avg_confidence,
            "high_confidence_matches": self.high_confidence_matches,
            "avg_roi_radius": self.avg_roi_radius,
            "avg_candidates_per_roi": self.avg_candidates_per_roi,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "total_processing_time_ms": self.total_processing_time_ms,
            "transformation_success_rate": self.transformation_success_rate,
            "avg_transformation_quality": self.avg_transformation_quality,
            "avg_reprojection_error_px": self.avg_reprojection_error_px,
            "max_reprojection_error_px": self.max_reprojection_error_px,
            "reprojection_samples": self.reprojection_samples,
            "reprojection_events": self.reprojection_events,
            "reprojection_missing_actual": self.reprojection_missing_actual,
            "camera_pair_stats": dict(self.camera_pair_stats),
        }


class MetricsCollector:
    """Collect and analyze metrics for geometric matching system."""

    def __init__(self, high_confidence_threshold: float = 0.8) -> None:
        self.high_confidence_threshold = high_confidence_threshold
        self.logger = logging.getLogger(__name__)
        self._records: List[Dict[str, Any]] = []
        self.camera_pair_counts: Dict[str, int] = {}
        self._reprojection_errors: List[float] = []
        self._reprojection_events: int = 0
        self._reprojection_missing_actual: int = 0

    def record_match(
        self,
        match_result: MatchResult,
        processing_time_ms: float,
        transformation_quality: Optional[float],
    ) -> None:
        """Record a single match attempt."""
        record = {
            "match_result": match_result,
            "processing_time_ms": processing_time_ms,
            "transformation_quality": transformation_quality,
            "timestamp": match_result.timestamp,
            "roi_radius": match_result.roi_radius,
            "candidates_in_roi": match_result.candidates_in_roi,
        }
        self._records.append(record)

        pair_key = f"{match_result.source_camera}→{match_result.dest_camera}"
        self.camera_pair_counts[pair_key] = self.camera_pair_counts.get(pair_key, 0) + 1

    def get_metrics(self, time_window_seconds: Optional[float] = None) -> GeometricMatchingMetrics:
        """Calculate aggregated metrics."""
        if not self._records:
            return GeometricMatchingMetrics()

        if time_window_seconds:
            cutoff = time.time() - time_window_seconds
            records = [r for r in self._records if r["timestamp"] >= cutoff]
        else:
            records = list(self._records)

        if not records:
            return GeometricMatchingMetrics()

        match_results = [r["match_result"] for r in records]
        processing_times = [r["processing_time_ms"] for r in records]
        transformation_qualities = [
            r["transformation_quality"] for r in records if r["transformation_quality"] is not None
        ]
        roi_radii = [r["roi_radius"] for r in records]
        candidates_counts = [r["candidates_in_roi"] for r in records]

        total = len(match_results)
        exact = sum(1 for m in match_results if m.match_type == MatchType.EXACT_MATCH)
        closest = sum(1 for m in match_results if m.match_type == MatchType.CLOSEST_MATCH)
        no_match = sum(1 for m in match_results if m.match_type == MatchType.NO_MATCH)
        successful = exact + closest

        overall_success_rate = successful / total if total > 0 else 0.0
        exact_rate = exact / total if total > 0 else 0.0
        closest_rate = closest / total if total > 0 else 0.0
        no_match_rate = no_match / total if total > 0 else 0.0

        distances = [
            m.spatial_distance
            for m in match_results
            if m.is_successful() and m.spatial_distance is not None
        ]

        if distances:
            avg_dist = statistics.mean(distances)
            min_dist = min(distances)
            max_dist = max(distances)
            median_dist = statistics.median(distances)
        else:
            avg_dist = min_dist = max_dist = median_dist = 0.0

        confidences = [m.confidence for m in match_results]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        high_confidence_matches = sum(1 for c in confidences if c >= self.high_confidence_threshold)

        avg_roi_radius = statistics.mean(roi_radii) if roi_radii else 0.0
        avg_candidates = statistics.mean(candidates_counts) if candidates_counts else 0.0

        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        total_processing_time = sum(processing_times)

        if transformation_qualities:
            avg_trans_quality = statistics.mean(transformation_qualities)
            transformation_success_rate = sum(
                1 for q in transformation_qualities if q >= 0.5
            ) / len(transformation_qualities)
        else:
            avg_trans_quality = 0.0
            transformation_success_rate = 0.0

        if self._reprojection_errors:
            avg_reprojection_error = statistics.mean(self._reprojection_errors)
            max_reprojection_error = max(self._reprojection_errors)
            reprojection_samples = len(self._reprojection_errors)
        else:
            avg_reprojection_error = 0.0
            max_reprojection_error = 0.0
            reprojection_samples = 0

        filtered_pairs: Dict[str, int] = {}
        for m in match_results:
            key = f"{m.source_camera}→{m.dest_camera}"
            filtered_pairs[key] = filtered_pairs.get(key, 0) + 1

        return GeometricMatchingMetrics(
            total_matches=total,
            exact_matches=exact,
            closest_matches=closest,
            no_matches=no_match,
            overall_success_rate=overall_success_rate,
            exact_match_rate=exact_rate,
            closest_match_rate=closest_rate,
            no_match_rate=no_match_rate,
            avg_distance=avg_dist,
            min_distance=min_dist,
            max_distance=max_dist,
            median_distance=median_dist,
            avg_confidence=avg_confidence,
            high_confidence_matches=high_confidence_matches,
            avg_roi_radius=avg_roi_radius,
            avg_candidates_per_roi=avg_candidates,
            avg_processing_time_ms=avg_processing_time,
            total_processing_time_ms=total_processing_time,
            transformation_success_rate=transformation_success_rate,
            avg_transformation_quality=avg_trans_quality,
            avg_reprojection_error_px=avg_reprojection_error,
            max_reprojection_error_px=max_reprojection_error,
            reprojection_samples=reprojection_samples,
            reprojection_events=self._reprojection_events,
            reprojection_missing_actual=self._reprojection_missing_actual,
            camera_pair_stats=filtered_pairs,
        )

    def record_reprojection_error(self, error_px: float) -> None:
        """Record pixel-space reprojection error for debugging."""
        if error_px >= 0:
            self._reprojection_errors.append(float(error_px))

    def record_reprojection_event(self, has_actual: bool) -> None:
        """Track when a projected point was rendered and whether a destination detection existed."""
        self._reprojection_events += 1
        if not has_actual:
            self._reprojection_missing_actual += 1

    def get_camera_pair_performance(self, source_camera: str, dest_camera: str) -> Dict[str, Any]:
        """Get performance metrics for specific camera pair."""
        pair_matches = [
            r["match_result"]
            for r in self._records
            if r["match_result"].source_camera == source_camera
            and r["match_result"].dest_camera == dest_camera
        ]

        if not pair_matches:
            return {"error": "No data for this camera pair"}

        total = len(pair_matches)
        successful = sum(1 for m in pair_matches if m.is_successful())

        return {
            "camera_pair": f"{source_camera}→{dest_camera}",
            "total_attempts": total,
            "successful_matches": successful,
            "success_rate": successful / total if total else 0.0,
            "avg_confidence": statistics.mean([m.confidence for m in pair_matches]) if pair_matches else 0.0,
            "avg_candidates_in_roi": statistics.mean([m.candidates_in_roi for m in pair_matches]) if pair_matches else 0.0,
        }

    def print_summary(self) -> None:
        """Print human-readable metrics summary."""
        metrics = self.get_metrics()

        self.logger.info("=" * 60)
        self.logger.info("Geometric Matching System - Performance Summary")
        self.logger.info("=" * 60)

        self.logger.info("Total Matches: %s", metrics.total_matches)
        self.logger.info("  - Exact Matches: %s (%.1f%%)", metrics.exact_matches, metrics.exact_match_rate * 100)
        self.logger.info("  - Closest Matches: %s (%.1f%%)", metrics.closest_matches, metrics.closest_match_rate * 100)
        self.logger.info("  - No Matches: %s (%.1f%%)", metrics.no_matches, metrics.no_match_rate * 100)

        self.logger.info("\nSuccess Rate: %.1f%%", metrics.overall_success_rate * 100)
        self.logger.info("High Confidence Matches: %s", metrics.high_confidence_matches)

        self.logger.info("\nSpatial Accuracy:")
        self.logger.info("  - Average Distance: %.2fm", metrics.avg_distance)
        self.logger.info("  - Median Distance: %.2fm", metrics.median_distance)
        self.logger.info("  - Min/Max Distance: %.2fm / %.2fm", metrics.min_distance, metrics.max_distance)

        self.logger.info("\nAverage Confidence: %.2f", metrics.avg_confidence)
        self.logger.info("Average ROI Radius: %.2fm", metrics.avg_roi_radius)
        self.logger.info("Average Candidates per ROI: %.1f", metrics.avg_candidates_per_roi)

        self.logger.info("\nPerformance:")
        self.logger.info("  - Avg Processing Time: %.1fms", metrics.avg_processing_time_ms)
        self.logger.info("  - Transformation Success Rate: %.1f%%", metrics.transformation_success_rate * 100)

        self.logger.info("\nReprojection Debug:")
        self.logger.info(
            "  - Events: %d (missing actual detection: %d)",
            metrics.reprojection_events,
            metrics.reprojection_missing_actual,
        )
        self.logger.info("  - Samples: %d", metrics.reprojection_samples)
        self.logger.info("  - Avg Pixel Error: %.1fpx", metrics.avg_reprojection_error_px)
        self.logger.info("  - Max Pixel Error: %.1fpx", metrics.max_reprojection_error_px)

        self.logger.info("=" * 60)
