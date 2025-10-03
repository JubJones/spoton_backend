#!/usr/bin/env python3
"""Database bootstrap utility with demo analytics seeding.

This script now performs three responsibilities:

1. Ensure all database tables/indexes exist (respecting ``DB_ENABLED``).
2. Seed a small set of analytics/demo data so dashboards have immediate content.
3. Prime the tracking cache (Redis) with a few active persons when available.

Usage::

    python scripts/setup_db.py

The demo seed is idempotent – if detection events already exist for the
``factory`` environment the seed step is skipped so live data can accumulate
without duplication.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List

from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.infrastructure.database.base import setup_database, get_session_factory
from app.infrastructure.database.models.tracking_models import DetectionEvent, PersonIdentity
from app.infrastructure.database.repositories.analytics_totals_repository import (
    AnalyticsTotalsRepository,
    DEFAULT_BUCKET_SECONDS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup_db")


def seed_demo_database_data() -> None:
    """Populate TimescaleDB with representative analytics and detection data."""

    session_factory = get_session_factory()
    if session_factory is None:
        logger.info("Database session factory unavailable; skipping demo seed.")
        return

    session = session_factory()
    try:
        repo = AnalyticsTotalsRepository(session)
        now_bucket = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        bucket_seconds = DEFAULT_BUCKET_SECONDS

        environments: Dict[str, Dict[str, float]] = {
            "factory": {
                "c09": 0.32,
                "c12": 0.26,
                "c13": 0.18,
                "c16": 0.24,
            },
            "default": {
                "cam-a": 0.34,
                "cam-b": 0.28,
                "cam-c": 0.22,
                "cam-d": 0.16,
            },
        }

        now_precise = datetime.now(timezone.utc)

        for environment_id, camera_profiles in environments.items():
            existing_detection = (
                session.query(DetectionEvent)
                .filter(DetectionEvent.environment_id == environment_id)
                .limit(1)
                .first()
            )
            if existing_detection is not None:
                logger.info("%s detection events already present; demo seed skipped for this environment.", environment_id)
                continue

            # Seed eight hours of aggregated analytics totals (global + camera breakdowns)
            for hours_back in range(0, 8):
                bucket_time = now_bucket - timedelta(hours=hours_back)
                base_detections = max(24, 96 - hours_back * 8)
                unique_entities = max(12, int(base_detections * 0.72))
                avg_confidence = 0.84 + (hours_back % 4) * 0.01

                repo.increment_detection_totals(
                    environment_id=environment_id,
                    camera_id=None,
                    event_time=bucket_time,
                    detections=base_detections,
                    unique_entities=unique_entities,
                    confidence_sum=base_detections * avg_confidence,
                    confidence_samples=base_detections,
                    bucket_size_seconds=bucket_seconds,
                )

                for index, (camera_id, weight) in enumerate(camera_profiles.items()):
                    camera_detections = max(6, int(base_detections * weight))
                    repo.increment_detection_totals(
                        environment_id=environment_id,
                        camera_id=camera_id,
                        event_time=bucket_time,
                        detections=camera_detections,
                        unique_entities=max(3, int(camera_detections * 0.68)),
                        confidence_sum=camera_detections * (avg_confidence + 0.01 * (index - 1)),
                        confidence_samples=camera_detections,
                        bucket_size_seconds=bucket_seconds,
                    )

            today = date.today()
            for day_offset in range(0, 7):
                day = today - timedelta(days=day_offset)
                uptime_base = max(92.0, 98.2 - day_offset * 0.6)

                repo.upsert_uptime_snapshot(
                    environment_id=environment_id,
                    camera_id=None,
                    day=day,
                    uptime_percent=uptime_base,
                )

                for index, camera_id in enumerate(camera_profiles.keys()):
                    repo.upsert_uptime_snapshot(
                        environment_id=environment_id,
                        camera_id=camera_id,
                        day=day,
                        uptime_percent=max(88.0, uptime_base - 1.2 + index * 0.35),
                    )

            # Seed recent detection events (last 15 minutes per camera)
            detection_rows: List[DetectionEvent] = []
            for minute_offset in range(0, 15):
                event_time = now_precise - timedelta(minutes=minute_offset)
                for index, camera_id in enumerate(camera_profiles.keys()):
                    confidence = max(0.6, min(0.98, 0.79 + 0.03 * (index % 3) - minute_offset * 0.002))
                    detection_rows.append(
                        DetectionEvent(
                            timestamp=event_time,
                            camera_id=camera_id,
                            environment_id=environment_id,
                            frame_number=minute_offset * 20 + index,
                            bbox_x1=0.12 + index * 0.02,
                            bbox_y1=0.18,
                            bbox_x2=0.42 + index * 0.015,
                            bbox_y2=0.58,
                            confidence=confidence,
                            object_class="person",
                            session_id=f"demo-session-{environment_id}",
                        )
                    )

            if detection_rows:
                session.bulk_save_objects(detection_rows)
                session.commit()
                logger.info(
                    "Inserted %s demo detection events for environment %s.",
                    len(detection_rows),
                    environment_id,
                )

            # Seed active person identities (supports /real-time/active-persons fallback)
            base_people = [
                {
                    "global_person_id": f"demo-person-{environment_id}-1",
                    "cameras": list(camera_profiles.keys())[:2],
                    "first_camera": list(camera_profiles.keys())[0],
                    "last_camera": list(camera_profiles.keys())[1],
                    "confidence": 0.92,
                    "detections": 36,
                    "tracking_time": 420.0,
                    "last_seen_offset_min": 1,
                },
                {
                    "global_person_id": f"demo-person-{environment_id}-2",
                    "cameras": [list(camera_profiles.keys())[2]],
                    "first_camera": list(camera_profiles.keys())[2],
                    "last_camera": list(camera_profiles.keys())[2],
                    "confidence": 0.88,
                    "detections": 24,
                    "tracking_time": 310.0,
                    "last_seen_offset_min": 2,
                },
                {
                    "global_person_id": f"demo-person-{environment_id}-3",
                    "cameras": list(camera_profiles.keys())[1:3],
                    "first_camera": list(camera_profiles.keys())[1],
                    "last_camera": list(camera_profiles.keys())[2],
                    "confidence": 0.9,
                    "detections": 29,
                    "tracking_time": 365.0,
                    "last_seen_offset_min": 3,
                },
            ]

            for person in base_people:
                person_obj = PersonIdentity(
                    global_person_id=person["global_person_id"],
                    first_seen_at=now_precise - timedelta(minutes=person["last_seen_offset_min"] + 25),
                    last_seen_at=now_precise - timedelta(minutes=person["last_seen_offset_min"]),
                    first_seen_camera=person["first_camera"],
                    last_seen_camera=person["last_camera"],
                    cameras_seen=person["cameras"],
                    environment_id=environment_id,
                    is_active=True,
                    confidence=person["confidence"],
                    total_detections=person["detections"],
                    total_tracking_time=person["tracking_time"],
                    identity_metadata={"seed": "demo"},
                )
                session.add(person_obj)

            session.commit()
            logger.info(
                "Seeded %s demo person identities for environment %s.",
                len(base_people),
                environment_id,
            )

    except SQLAlchemyError as exc:  # pragma: no cover - defensive logging
        session.rollback()
        logger.exception("Failed to seed demo analytics data: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        session.rollback()
        logger.exception("Unexpected error while seeding demo analytics data: %s", exc)
    finally:
        session.close()


async def seed_tracking_cache_with_demo_persons() -> None:
    """Populate Redis cache with a few active person entries if available."""

    try:
        from app.infrastructure.cache.tracking_cache import tracking_cache
        from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem

        await tracking_cache.initialize()

        now = datetime.now(timezone.utc)
        cache_people = [
            {
                "global_id": "demo-person-1",
                "camera_id": "c09",
                "confidence": 0.93,
                "track_id": "track-demo-1",
                "position": (12.4, 7.6),
            },
            {
                "global_id": "demo-person-2",
                "camera_id": "c13",
                "confidence": 0.87,
                "track_id": "track-demo-2",
                "position": (32.1, 11.3),
            },
            {
                "global_id": "demo-person-3",
                "camera_id": "c16",
                "confidence": 0.9,
                "track_id": "track-demo-3",
                "position": (22.8, 15.9),
            },
        ]

        for item in cache_people:
            coordinate = Coordinate(
                x=item["position"][0],
                y=item["position"][1],
                coordinate_system=CoordinateSystem.MAP,
                timestamp=now,
                confidence=0.9,
                camera_id=item["camera_id"],
            )
            await tracking_cache.cache_person_state(
                person_identity={
                    "global_id": item["global_id"],
                    "confidence": item["confidence"],
                    "track_id": item["track_id"],
                    "is_active": True,
                },
                camera_id=item["camera_id"],
                position=coordinate,
                trajectory=None,
            )

        logger.info("Seeded tracking cache with %s demo active persons.", len(cache_people))

    except Exception as exc:  # pragma: no cover - cache is optional; log and continue
        logger.warning("Skipping tracking cache seed: %s", exc)


async def main() -> None:
    if not getattr(settings, "DB_ENABLED", True):
        logger.info("DB_ENABLED=false; skipping database setup and seed.")
        return

    await setup_database()
    seed_demo_database_data()
    await seed_tracking_cache_with_demo_persons()


if __name__ == "__main__":
    asyncio.run(main())
