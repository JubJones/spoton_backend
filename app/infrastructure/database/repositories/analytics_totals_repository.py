"""Repository and helper dataclasses for analytics totals aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.infrastructure.database.models.tracking_models import AnalyticsTotals, AnalyticsUptimeDaily

GLOBAL_CAMERA_KEY = "__all__"
DEFAULT_BUCKET_SECONDS = 3600


def _floor_to_bucket(ts: datetime, bucket_size_seconds: int) -> datetime:
    """Floor timestamp to the configured bucket size."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    epoch_seconds = int(ts.timestamp())
    bucket_seconds = (epoch_seconds // bucket_size_seconds) * bucket_size_seconds
    return datetime.fromtimestamp(bucket_seconds, tz=timezone.utc)


@dataclass
class WindowTotals:
    detections: int
    unique_entities: int
    confidence_sum: float
    confidence_samples: int

    @property
    def average_confidence(self) -> float:
        return (self.confidence_sum / self.confidence_samples) if self.confidence_samples else 0.0


@dataclass
class CameraTotals(WindowTotals):
    camera_id: str


@dataclass
class TimeBucket:
    bucket_start: datetime
    detections: int
    average_confidence: float


@dataclass
class DailyUptime:
    day: date
    uptime_percent: float


class AnalyticsTotalsRepository:
    """Repository encapsulating analytics totals persistence and queries."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def increment_detection_totals(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        event_time: datetime,
        detections: int,
        unique_entities: int,
        confidence_sum: float,
        confidence_samples: int,
        bucket_size_seconds: int = DEFAULT_BUCKET_SECONDS,
    ) -> None:
        """Increment aggregated detection totals for the given bucket."""
        if detections == 0 and confidence_samples == 0:
            return

        camera_key = camera_id or GLOBAL_CAMERA_KEY
        bucket_start = _floor_to_bucket(event_time, bucket_size_seconds)

        stmt = insert(AnalyticsTotals).values(
            bucket_start=bucket_start,
            bucket_size_seconds=bucket_size_seconds,
            environment_id=environment_id,
            camera_id=camera_key,
            detections=detections,
            unique_entities=unique_entities,
            confidence_sum=confidence_sum,
            confidence_samples=confidence_samples,
            updated_at=datetime.now(timezone.utc),
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=[
                AnalyticsTotals.bucket_start,
                AnalyticsTotals.bucket_size_seconds,
                AnalyticsTotals.environment_id,
                AnalyticsTotals.camera_id,
            ],
            set_={
                'detections': AnalyticsTotals.detections + detections,
                'unique_entities': AnalyticsTotals.unique_entities + unique_entities,
                'confidence_sum': AnalyticsTotals.confidence_sum + confidence_sum,
                'confidence_samples': AnalyticsTotals.confidence_samples + confidence_samples,
                'updated_at': datetime.now(timezone.utc),
            }
        )

        self.db.execute(stmt)
        self.db.commit()

    def upsert_uptime_snapshot(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        day: date,
        uptime_percent: float,
        samples: int = 1,
    ) -> None:
        camera_key = camera_id or GLOBAL_CAMERA_KEY
        stmt = insert(AnalyticsUptimeDaily).values(
            day=day,
            environment_id=environment_id,
            camera_id=camera_key,
            uptime_percent=uptime_percent,
            samples=samples,
            updated_at=datetime.now(timezone.utc),
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=[
                AnalyticsUptimeDaily.day,
                AnalyticsUptimeDaily.environment_id,
                AnalyticsUptimeDaily.camera_id,
            ],
            set_={
                'uptime_percent': (
                    (AnalyticsUptimeDaily.uptime_percent * AnalyticsUptimeDaily.samples) + (uptime_percent * samples)
                ) / (AnalyticsUptimeDaily.samples + samples),
                'samples': AnalyticsUptimeDaily.samples + samples,
                'updated_at': datetime.now(timezone.utc),
            }
        )

        self.db.execute(stmt)
        self.db.commit()

    def fetch_window_totals(
        self,
        *,
        environment_id: str,
        start: datetime,
        end: datetime,
        bucket_size_seconds: int = DEFAULT_BUCKET_SECONDS,
    ) -> WindowTotals:
        bucket_start = _floor_to_bucket(start, bucket_size_seconds)
        bucket_end = _floor_to_bucket(end, bucket_size_seconds)

        query = (
            self.db.query(
                func.coalesce(func.sum(AnalyticsTotals.detections), 0),
                func.coalesce(func.sum(AnalyticsTotals.unique_entities), 0),
                func.coalesce(func.sum(AnalyticsTotals.confidence_sum), 0.0),
                func.coalesce(func.sum(AnalyticsTotals.confidence_samples), 0),
            )
            .filter(AnalyticsTotals.environment_id == environment_id)
            .filter(AnalyticsTotals.camera_id == GLOBAL_CAMERA_KEY)
            .filter(AnalyticsTotals.bucket_size_seconds == bucket_size_seconds)
            .filter(AnalyticsTotals.bucket_start.between(bucket_start, bucket_end))
        )

        detections, unique_entities, confidence_sum, confidence_samples = query.one()
        return WindowTotals(
            detections=int(detections or 0),
            unique_entities=int(unique_entities or 0),
            confidence_sum=float(confidence_sum or 0.0),
            confidence_samples=int(confidence_samples or 0),
        )

    def fetch_camera_breakdown(
        self,
        *,
        environment_id: str,
        start: datetime,
        end: datetime,
        bucket_size_seconds: int = DEFAULT_BUCKET_SECONDS,
    ) -> List[CameraTotals]:
        bucket_start = _floor_to_bucket(start, bucket_size_seconds)
        bucket_end = _floor_to_bucket(end, bucket_size_seconds)

        results = (
            self.db.query(
                AnalyticsTotals.camera_id,
                func.coalesce(func.sum(AnalyticsTotals.detections), 0),
                func.coalesce(func.sum(AnalyticsTotals.unique_entities), 0),
                func.coalesce(func.sum(AnalyticsTotals.confidence_sum), 0.0),
                func.coalesce(func.sum(AnalyticsTotals.confidence_samples), 0),
            )
            .filter(AnalyticsTotals.environment_id == environment_id)
            .filter(AnalyticsTotals.camera_id != GLOBAL_CAMERA_KEY)
            .filter(AnalyticsTotals.bucket_size_seconds == bucket_size_seconds)
            .filter(AnalyticsTotals.bucket_start.between(bucket_start, bucket_end))
            .group_by(AnalyticsTotals.camera_id)
            .order_by(AnalyticsTotals.camera_id.asc())
            .all()
        )

        cameras: List[CameraTotals] = []
        for camera_id, detections, unique_entities, confidence_sum, confidence_samples in results:
            totals = CameraTotals(
                camera_id=camera_id,
                detections=int(detections or 0),
                unique_entities=int(unique_entities or 0),
                confidence_sum=float(confidence_sum or 0.0),
                confidence_samples=int(confidence_samples or 0),
            )
            cameras.append(totals)
        return cameras

    def fetch_bucketed_series(
        self,
        *,
        environment_id: str,
        start: datetime,
        end: datetime,
        bucket_size_seconds: int = DEFAULT_BUCKET_SECONDS,
    ) -> List[TimeBucket]:
        bucket_start = _floor_to_bucket(start, bucket_size_seconds)
        bucket_end = _floor_to_bucket(end, bucket_size_seconds)

        rows = (
            self.db.query(
                AnalyticsTotals.bucket_start,
                func.coalesce(func.sum(AnalyticsTotals.detections), 0),
                func.coalesce(func.sum(AnalyticsTotals.confidence_sum), 0.0),
                func.coalesce(func.sum(AnalyticsTotals.confidence_samples), 0),
            )
            .filter(AnalyticsTotals.environment_id == environment_id)
            .filter(AnalyticsTotals.camera_id == GLOBAL_CAMERA_KEY)
            .filter(AnalyticsTotals.bucket_size_seconds == bucket_size_seconds)
            .filter(AnalyticsTotals.bucket_start.between(bucket_start, bucket_end))
            .group_by(AnalyticsTotals.bucket_start)
            .order_by(AnalyticsTotals.bucket_start.asc())
            .all()
        )

        buckets: List[TimeBucket] = []
        for ts, detections, confidence_sum, confidence_samples in rows:
            avg_conf = (confidence_sum / confidence_samples) if confidence_samples else 0.0
            buckets.append(
                TimeBucket(
                    bucket_start=ts,
                    detections=int(detections or 0),
                    average_confidence=float(avg_conf),
                )
            )
        return buckets

    def fetch_uptime_series(
        self,
        *,
        environment_id: str,
        days: int,
        camera_id: Optional[str] = None,
    ) -> List[DailyUptime]:
        target_day = date.today()
        day_start = target_day - timedelta(days=days - 1)
        camera_key = camera_id or GLOBAL_CAMERA_KEY

        rows = (
            self.db.query(
                AnalyticsUptimeDaily.day,
                AnalyticsUptimeDaily.uptime_percent,
            )
            .filter(AnalyticsUptimeDaily.environment_id == environment_id)
            .filter(AnalyticsUptimeDaily.camera_id == camera_key)
            .filter(AnalyticsUptimeDaily.day.between(day_start, target_day))
            .order_by(AnalyticsUptimeDaily.day.asc())
            .all()
        )

        return [DailyUptime(day=day_, uptime_percent=float(uptime or 0.0)) for day_, uptime in rows]
