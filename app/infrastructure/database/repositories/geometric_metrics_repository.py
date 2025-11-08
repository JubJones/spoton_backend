"""Repository for storing geometric pipeline metrics."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.infrastructure.database.models.tracking_models import GeometricMetricsEvent


class GeometricMetricsRepository:
    """Encapsulates persistence for geometric metrics events."""

    def __init__(self, db: Session):
        self.db = db

    def insert_metric(
        self,
        *,
        event_timestamp: datetime,
        environment_id: str,
        camera_id: Optional[str],
        extraction_total_attempts: int,
        extraction_validation_failures: int,
        extraction_success_rate: float,
        transformation_total_attempts: Optional[int],
        transformation_validation_failures: Optional[int],
        transformation_success_rate: Optional[float],
        roi_total_created: int,
    ) -> None:
        event = GeometricMetricsEvent(
            event_timestamp=event_timestamp,
            environment_id=environment_id,
            camera_id=camera_id,
            extraction_total_attempts=extraction_total_attempts,
            extraction_validation_failures=extraction_validation_failures,
            extraction_success_rate=extraction_success_rate,
            transformation_total_attempts=transformation_total_attempts,
            transformation_validation_failures=transformation_validation_failures,
            transformation_success_rate=transformation_success_rate,
            roi_total_created=roi_total_created,
        )
        self.db.add(event)
        self.db.commit()
