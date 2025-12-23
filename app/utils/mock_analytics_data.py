"""
Mock data generator for analytics endpoints.

Provides pre-populated demo data that gets merged with live data to ensure
the analytics dashboard always has meaningful content to display.
"""

import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List


# Camera IDs per environment
ENVIRONMENT_CAMERAS = {
    "campus": ["c01", "c02", "c03", "c05"],
    "factory": ["c09", "c12", "c13", "c16"],
}

# All cameras across all environments
ALL_CAMERAS = ENVIRONMENT_CAMERAS["campus"] + ENVIRONMENT_CAMERAS["factory"]


def _generate_random_detections_series(hours: int = 24, bucket_minutes: int = 15) -> List[Dict[str, Any]]:
    """Generate realistic detection counts over time buckets."""
    now = datetime.now(timezone.utc)
    buckets = []
    
    for i in range(hours * (60 // bucket_minutes)):
        bucket_time = now - timedelta(minutes=i * bucket_minutes)
        # Simulate higher traffic during business hours (9-18)
        hour = bucket_time.hour
        if 9 <= hour <= 18:
            base_detections = random.randint(80, 200)
        elif 6 <= hour <= 22:
            base_detections = random.randint(30, 100)
        else:
            base_detections = random.randint(5, 30)
        
        buckets.append({
            "timestamp": bucket_time.isoformat(),
            "detections": base_detections,
        })
    
    return list(reversed(buckets))


def _generate_confidence_trend(hours: int = 24, bucket_minutes: int = 15) -> List[Dict[str, Any]]:
    """Generate confidence trend data."""
    now = datetime.now(timezone.utc)
    buckets = []
    
    for i in range(hours * (60 // bucket_minutes)):
        bucket_time = now - timedelta(minutes=i * bucket_minutes)
        # Confidence typically between 75-95%
        confidence = random.uniform(78, 96)
        
        buckets.append({
            "timestamp": bucket_time.isoformat(),
            "confidence_percent": round(confidence, 2),
        })
    
    return list(reversed(buckets))


def _generate_uptime_trend(days: int = 7) -> List[Dict[str, Any]]:
    """Generate uptime trend data."""
    today = datetime.now(timezone.utc).date()
    trend = []
    
    for i in range(days):
        day = today - timedelta(days=days - 1 - i)
        # Uptime typically 95-100%
        uptime = random.uniform(96, 100)
        trend.append({
            "date": str(day),
            "uptime_percent": round(uptime, 2),
        })
    
    return trend


def _generate_camera_stats(camera_ids: List[str], window_hours: int = 24) -> List[Dict[str, Any]]:
    """Generate per-camera statistics scaled by time window."""
    cameras = []
    # Scale detections by window size
    base_per_hour = random.randint(20, 100)
    
    for camera_id in camera_ids:
        detections = base_per_hour * window_hours + random.randint(-50, 200)
        cameras.append({
            "camera_id": camera_id,
            "detections": max(0, detections),
            "unique_entities": max(1, detections // random.randint(5, 15)),
            "average_confidence_percent": round(random.uniform(80, 95), 2),
            "uptime_percent": round(random.uniform(97, 100), 2),
        })
    return cameras


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

def get_mock_dashboard_data(
    environment_id: str = "default",
    window_hours: int = 24,
    uptime_days: int = 7,
) -> Dict[str, Any]:
    """
    Generate comprehensive mock dashboard data.
    This will be merged with real data - real data takes priority.
    
    Supports time windows: 1h, 6h, 24h, and 7d (168h).
    Includes ALL cameras from ALL environments.
    """
    # Use all cameras from all environments
    mock_camera_ids = ALL_CAMERAS
    
    # Adjust bucket size based on window
    if window_hours <= 1:
        bucket_minutes = 5  # 12 buckets for 1h
    elif window_hours <= 6:
        bucket_minutes = 15  # 24 buckets for 6h
    elif window_hours <= 24:
        bucket_minutes = 30  # 48 buckets for 24h
    else:
        bucket_minutes = 60  # 168 buckets for 7d
    
    # Scale detection totals based on window size
    base_detections_per_hour = random.randint(200, 500)
    total_detections = base_detections_per_hour * window_hours
    
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_detections": total_detections,
            "average_confidence_percent": round(random.uniform(82, 94), 2),
            "system_uptime_percent": round(random.uniform(98, 99.9), 2),
            "uptime_delta_percent": round(random.uniform(-0.5, 1.5), 2),
        },
        "cameras": _generate_camera_stats(mock_camera_ids, window_hours),
        "charts": {
            "detections_per_bucket": _generate_random_detections_series(window_hours, bucket_minutes),
            "average_confidence_trend": _generate_confidence_trend(window_hours, bucket_minutes),
            "uptime_trend": _generate_uptime_trend(uptime_days),
        },
        "notes": [
            "Mock data shown for demonstration purposes.",
            f"Time window: {window_hours}h, Uptime history: {uptime_days}d",
            "Real data will appear once detection processing starts.",
        ],
        "_is_mock": True,
        "_window_hours": window_hours,
        "_uptime_days": uptime_days,
    }


def get_mock_system_statistics() -> Dict[str, Any]:
    """
    Generate mock system statistics.
    """
    now = datetime.now(timezone.utc)
    uptime_seconds = random.randint(3600 * 24, 3600 * 24 * 7)  # 1-7 days
    
    return {
        "analytics_engine": {
            "total_behavior_analyses": random.randint(100, 500),
            "total_path_predictions": random.randint(50, 200),
            "cached_behavior_profiles": random.randint(20, 80),
            "reports_generated": random.randint(5, 30),
            "total_queries_processed": random.randint(500, 2000),
            "average_query_time_ms": round(random.uniform(15, 50), 2),
            "cache_hit_rate": round(random.uniform(0.75, 0.95), 4),
            "last_analysis_timestamp": (now - timedelta(minutes=random.randint(1, 30))).isoformat(),
        },
        "database_service": {
            "status": "healthy",
            "redis_status": "connected",
            "postgres_status": "connected",
            "total_cached_persons": random.randint(10, 100),
            "total_active_sessions": random.randint(1, 5),
            "sync_operations": random.randint(100, 1000),
            "sync_failures": random.randint(0, 5),
            "last_sync_time": (now - timedelta(seconds=random.randint(1, 60))).isoformat(),
            "db_writes": random.randint(5000, 50000),
            "redis_writes": random.randint(10000, 100000),
        },
        "system_uptime": f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m",
        "last_updated": now.isoformat(),
        "_is_mock": True,
    }


def merge_with_real_data(mock_data: Dict[str, Any], real_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge mock data with real data. Real data takes priority.
    For lists (like cameras, chart buckets), real data replaces mock if non-empty.
    """
    if not real_data:
        return mock_data
    
    result = mock_data.copy()
    
    for key, value in real_data.items():
        if value is None:
            continue
        
        if isinstance(value, dict):
            if key in result and isinstance(result[key], dict):
                result[key] = merge_with_real_data(result[key], value)
            else:
                result[key] = value
        elif isinstance(value, list):
            # For lists, use real data if it has items
            if value:
                result[key] = value
        elif isinstance(value, (int, float)):
            # For numeric values, use real if non-zero
            if value != 0:
                result[key] = value
        else:
            result[key] = value
    
    # Mark as merged if we had real data
    if real_data.get("summary", {}).get("total_detections", 0) > 0:
        result["_is_mock"] = False
        result["notes"] = real_data.get("notes", ["Analytics values are aggregated from persisted detection and uptime events."])
    
    return result
