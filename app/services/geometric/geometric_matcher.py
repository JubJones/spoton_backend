"""
Pure geometric matcher (Phase 4) selecting matches based on world-plane proximity.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from app.services.geometric.world_plane_transformer import WorldPoint
from app.services.geometric.roi_calculator import SearchROI


class MatchType(Enum):
    """Types of geometric matching results."""

    EXACT_MATCH = "exact_match"
    CLOSEST_MATCH = "closest_match"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class PersonCandidate:
    """Candidate person detection in destination camera."""

    person_id: int
    world_location: Tuple[float, float]
    camera_id: str
    timestamp: float


@dataclass
class MatchResult:
    """Result of geometric matching attempt."""

    source_camera: str
    dest_camera: str
    source_person_id: int
    matched_person_id: Optional[int]
    match_type: MatchType
    confidence: float
    spatial_distance: Optional[float]
    candidates_in_roi: int
    timestamp: float
    roi_radius: float

    def is_successful(self) -> bool:
        return self.match_type != MatchType.NO_MATCH


class GeometricMatcher:
    """Pure geometric matching using world-plane spatial reasoning."""

    def __init__(
        self,
        exact_match_confidence: float = 0.95,
        closest_match_confidence: float = 0.70,
        distance_penalty_factor: float = 0.1,
    ) -> None:
        self.exact_match_confidence = exact_match_confidence
        self.closest_match_confidence = closest_match_confidence
        self.distance_penalty_factor = distance_penalty_factor

        self.logger = logging.getLogger(__name__)
        self.match_count = 0
        self.match_type_counts: Dict[MatchType, int] = {match_type: 0 for match_type in MatchType}

    def match_person(
        self,
        source_world_point: WorldPoint,
        destination_candidates: List[WorldPoint],
        search_roi: SearchROI,
    ) -> MatchResult:
        """Match source person to destination camera candidates using geometry."""
        candidates_in_roi: List[PersonCandidate] = []
        for candidate in destination_candidates:
            if search_roi.contains_point((candidate.x, candidate.y)):
                candidates_in_roi.append(
                    PersonCandidate(
                        person_id=candidate.person_id,
                        world_location=(candidate.x, candidate.y),
                        camera_id=candidate.camera_id,
                        timestamp=candidate.timestamp,
                    )
                )

        num_candidates = len(candidates_in_roi)

        if num_candidates == 0:
            result = self._create_no_match_result(source_world_point, search_roi, num_candidates)
        elif num_candidates == 1:
            result = self._create_exact_match_result(source_world_point, candidates_in_roi[0], search_roi, num_candidates)
        else:
            result = self._create_closest_match_result(source_world_point, candidates_in_roi, search_roi, num_candidates)

        self.match_count += 1
        self.match_type_counts[result.match_type] += 1
        return result

    def _create_no_match_result(
        self,
        source_point: WorldPoint,
        roi: SearchROI,
        num_candidates: int,
    ) -> MatchResult:
        self.logger.debug(
            "No geometric match %s→%s for track %s (candidates=%s)",
            roi.source_camera,
            roi.dest_camera,
            source_point.person_id,
            num_candidates,
        )

        return MatchResult(
            source_camera=source_point.camera_id,
            dest_camera=roi.dest_camera or "",
            source_person_id=source_point.person_id,
            matched_person_id=None,
            match_type=MatchType.NO_MATCH,
            confidence=0.0,
            spatial_distance=None,
            candidates_in_roi=num_candidates,
            timestamp=source_point.timestamp,
            roi_radius=roi.radius,
        )

    def _create_exact_match_result(
        self,
        source_point: WorldPoint,
        matched_candidate: PersonCandidate,
        roi: SearchROI,
        num_candidates: int,
    ) -> MatchResult:
        distance = self._euclidean_distance((source_point.x, source_point.y), matched_candidate.world_location)

        confidence = self.exact_match_confidence - (distance * self.distance_penalty_factor)
        confidence = max(0.0, min(1.0, confidence))

        self.logger.debug(
            "Exact geometric match %s→%s source=%s dest=%s distance=%.3fm confidence=%.2f",
            roi.source_camera,
            roi.dest_camera,
            source_point.person_id,
            matched_candidate.person_id,
            distance,
            confidence,
        )

        return MatchResult(
            source_camera=source_point.camera_id,
            dest_camera=roi.dest_camera or matched_candidate.camera_id,
            source_person_id=source_point.person_id,
            matched_person_id=matched_candidate.person_id,
            match_type=MatchType.EXACT_MATCH,
            confidence=confidence,
            spatial_distance=distance,
            candidates_in_roi=num_candidates,
            timestamp=source_point.timestamp,
            roi_radius=roi.radius,
        )

    def _create_closest_match_result(
        self,
        source_point: WorldPoint,
        candidates: List[PersonCandidate],
        roi: SearchROI,
        num_candidates: int,
    ) -> MatchResult:
        source_location = (source_point.x, source_point.y)
        closest_candidate: Optional[PersonCandidate] = None
        closest_distance = float("inf")

        for candidate in candidates:
            distance = self._euclidean_distance(source_location, candidate.world_location)
            if distance < closest_distance:
                closest_distance = distance
                closest_candidate = candidate

        confidence = self.closest_match_confidence - (closest_distance * self.distance_penalty_factor)
        confidence = max(0.0, min(1.0, confidence))

        self.logger.debug(
            "Closest geometric match %s→%s source=%s dest=%s distance=%.3fm candidates=%d confidence=%.2f",
            roi.source_camera,
            roi.dest_camera,
            source_point.person_id,
            closest_candidate.person_id if closest_candidate else None,
            closest_distance,
            num_candidates,
            confidence,
        )

        return MatchResult(
            source_camera=source_point.camera_id,
            dest_camera=roi.dest_camera or (closest_candidate.camera_id if closest_candidate else ""),
            source_person_id=source_point.person_id,
            matched_person_id=closest_candidate.person_id if closest_candidate else None,
            match_type=MatchType.CLOSEST_MATCH,
            confidence=confidence,
            spatial_distance=closest_distance if closest_candidate else None,
            candidates_in_roi=num_candidates,
            timestamp=source_point.timestamp,
            roi_radius=roi.radius,
        )

    @staticmethod
    def _euclidean_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        return math.sqrt(dx**2 + dy**2)

    def get_statistics(self) -> Dict[str, float]:
        success_count = self.match_type_counts[MatchType.EXACT_MATCH] + self.match_type_counts[MatchType.CLOSEST_MATCH]
        return {
            "total_matches": self.match_count,
            "exact_matches": self.match_type_counts[MatchType.EXACT_MATCH],
            "closest_matches": self.match_type_counts[MatchType.CLOSEST_MATCH],
            "no_matches": self.match_type_counts[MatchType.NO_MATCH],
            "success_rate": success_count / self.match_count if self.match_count > 0 else 0.0,
        }
