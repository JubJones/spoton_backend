"""Tracing utilities for analytics aggregation producers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.infrastructure.integration.database_integration_service import database_integration_service

logger = logging.getLogger(__name__)


class AnalyticsEventTracer:
    """Helper emitting structured logs whenever analytics aggregates are updated."""

    async def record_detection_batch(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        detections: int,
        unique_entities: int,
        average_confidence: Optional[float],
        trace_id: Optional[str] = None,
        bucket_seconds: int = 3600,
        timestamp: Optional[datetime] = None,
    ) -> None:
        event_ts = timestamp or datetime.now(timezone.utc)
        await database_integration_service.record_detection_batch(
            environment_id=environment_id,
            camera_id=camera_id,
            detections=detections,
            unique_entities=unique_entities,
            average_confidence=average_confidence,
            event_timestamp=event_ts,
            bucket_size_seconds=bucket_seconds,
        )

        logger.info(
            json.dumps(
                {
                    "event": "analytics_detection_batch",
                    "environment_id": environment_id,
                    "camera_id": camera_id,
                    "detections": detections,
                    "unique_entities": unique_entities,
                    "average_confidence": average_confidence,
                    "bucket_seconds": bucket_seconds,
                    "timestamp": event_ts.isoformat(),
                    "trace_id": trace_id,
                }
            )
        )

    async def record_camera_uptime(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        uptime_percent: float,
        samples: int = 1,
        day: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        await database_integration_service.record_uptime_snapshot(
            environment_id=environment_id,
            camera_id=camera_id,
            uptime_percent=uptime_percent,
            snapshot_day=day,
            samples=samples,
        )

        logger.info(
            json.dumps(
                {
                    "event": "analytics_camera_uptime",
                    "environment_id": environment_id,
                    "camera_id": camera_id,
                    "uptime_percent": uptime_percent,
                    "samples": samples,
                    "date": (day or datetime.now(timezone.utc)).date().isoformat(),
                    "trace_id": trace_id,
                }
            )
        )


analytics_event_tracer = AnalyticsEventTracer()
"""Singleton tracer instance used across the application."""
