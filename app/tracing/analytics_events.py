"""Tracing utilities for analytics aggregation producers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any

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

    async def record_geometric_metrics(
        self,
        *,
        environment_id: str,
        camera_id: Optional[str],
        extraction_stats: Dict[str, float],
        transformation_stats: Optional[Dict[str, float]],
        roi_stats: Optional[Dict[str, float]],
        matcher_stats: Optional[Dict[str, float]] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record geometric pipeline metrics for observability."""
        event_ts = timestamp or datetime.now(timezone.utc)

        extraction_total = int(extraction_stats.get("total_attempts", 0))
        extraction_failures = int(extraction_stats.get("validation_failures", 0))
        extraction_success_rate = float(extraction_stats.get("success_rate", 0.0))

        transformation_total = int(transformation_stats.get("total_transformations", 0)) if transformation_stats else None
        transformation_failures = int(transformation_stats.get("validation_failures", 0)) if transformation_stats else None
        transformation_success_rate = float(transformation_stats.get("success_rate", 0.0)) if transformation_stats else None
        roi_total = int(roi_stats.get("total_rois_created", 0)) if roi_stats else 0

        await database_integration_service.record_geometric_metrics(
            environment_id=environment_id,
            camera_id=camera_id,
            extraction_total_attempts=extraction_total,
            extraction_validation_failures=extraction_failures,
            extraction_success_rate=extraction_success_rate,
            transformation_total_attempts=transformation_total,
            transformation_validation_failures=transformation_failures,
            transformation_success_rate=transformation_success_rate,
            roi_total_created=roi_total,
            event_timestamp=event_ts,
        )

        payload = {
            "event": "geometric_metrics",
            "environment_id": environment_id,
            "camera_id": camera_id,
            "extraction": extraction_stats,
            "transformation": transformation_stats,
            "roi": roi_stats,
            "matching": matcher_stats,
            "metrics_summary": metrics_summary,
            "timestamp": event_ts.isoformat(),
            "trace_id": trace_id,
        }
        # logger.info(json.dumps(payload))


analytics_event_tracer = AnalyticsEventTracer()
"""Singleton tracer instance used across the application."""
