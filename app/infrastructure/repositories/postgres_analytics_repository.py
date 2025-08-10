"""
PostgreSQL implementation of analytics repository.

Concrete implementation of analytics repository using TimescaleDB
for time-series analytics data with optimized aggregations.
"""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import logging
import asyncpg
import json

from app.application.repositories.analytics_repository import AnalyticsRepository
from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.time_range import TimeRange

logger = logging.getLogger(__name__)


class PostgreSQLAnalyticsRepository(AnalyticsRepository):
    """
    PostgreSQL implementation of analytics repository.
    
    Uses TimescaleDB for optimized time-series storage and aggregation
    of analytics data with proper partitioning and compression.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize PostgreSQL analytics repository.
        
        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
        logger.debug("PostgreSQLAnalyticsRepository initialized")
    
    async def save_analytics_session(
        self,
        session_id: str,
        session_data: Dict[str, Any]
    ) -> bool:
        """Save analytics session data."""
        query = """
        INSERT INTO analytics_sessions (
            session_id, session_data, created_at, updated_at
        ) VALUES ($1, $2, NOW(), NOW())
        ON CONFLICT (session_id) 
        DO UPDATE SET session_data = $2, updated_at = NOW()
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    session_id,
                    json.dumps(session_data)
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to save analytics session: {e}")
            return False
    
    async def get_analytics_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get analytics session by ID."""
        query = """
        SELECT session_data FROM analytics_sessions 
        WHERE session_id = $1
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, session_id)
                
                if result:
                    return json.loads(result)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get analytics session: {e}")
            return None
    
    async def save_behavioral_metrics(
        self,
        camera_id: CameraID,
        timestamp: datetime,
        metrics: Dict[str, Any]
    ) -> bool:
        """Save behavioral metrics data."""
        query = """
        INSERT INTO behavioral_metrics (
            camera_id, timestamp, metrics_data, created_at
        ) VALUES ($1, $2, $3, NOW())
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    str(camera_id),
                    timestamp,
                    json.dumps(metrics)
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to save behavioral metrics: {e}")
            return False
    
    async def get_behavioral_metrics(
        self,
        camera_id: CameraID,
        time_range: TimeRange,
        metric_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get behavioral metrics for analysis."""
        base_query = """
        SELECT camera_id, timestamp, metrics_data
        FROM behavioral_metrics
        WHERE camera_id = $1 
        AND timestamp >= $2 
        AND timestamp <= $3
        """
        
        params = [str(camera_id), time_range.start_time, time_range.end_time]
        
        # Note: metric_types filtering would require JSONB operations
        # This is a simplified version
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(base_query + " ORDER BY timestamp", *params)
                
                metrics = []
                for row in rows:
                    metric_data = {
                        'camera_id': row['camera_id'],
                        'timestamp': row['timestamp'].isoformat(),
                        'metrics': json.loads(row['metrics_data'])
                    }
                    
                    # Filter by metric types if specified
                    if metric_types:
                        filtered_metrics = {
                            k: v for k, v in metric_data['metrics'].items()
                            if k in metric_types
                        }
                        metric_data['metrics'] = filtered_metrics
                    
                    metrics.append(metric_data)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get behavioral metrics: {e}")
            return []
    
    async def save_crowd_analytics(
        self,
        camera_id: CameraID,
        timestamp: datetime,
        crowd_data: Dict[str, Any]
    ) -> bool:
        """Save crowd analytics data."""
        query = """
        INSERT INTO crowd_analytics (
            camera_id, timestamp, crowd_data, density, flow_rate, created_at
        ) VALUES ($1, $2, $3, $4, $5, NOW())
        """
        
        try:
            # Extract specific metrics for indexing
            density = crowd_data.get('density', 0.0)
            flow_rate = crowd_data.get('flow_rate', 0.0)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    str(camera_id),
                    timestamp,
                    json.dumps(crowd_data),
                    density,
                    flow_rate
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to save crowd analytics: {e}")
            return False
    
    async def get_crowd_analytics(
        self,
        camera_id: CameraID,
        time_range: TimeRange,
        aggregation_interval: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get crowd analytics data."""
        if aggregation_interval:
            # Use TimescaleDB time_bucket for aggregation
            interval_map = {
                'hourly': '1 hour',
                'daily': '1 day',
                'weekly': '1 week'
            }
            interval = interval_map.get(aggregation_interval, '1 hour')
            
            query = f"""
            SELECT 
                time_bucket('{interval}', timestamp) as time_bucket,
                AVG(density) as avg_density,
                MAX(density) as max_density,
                AVG(flow_rate) as avg_flow_rate,
                COUNT(*) as data_points
            FROM crowd_analytics
            WHERE camera_id = $1 
            AND timestamp >= $2 
            AND timestamp <= $3
            GROUP BY time_bucket
            ORDER BY time_bucket
            """
        else:
            query = """
            SELECT camera_id, timestamp, crowd_data, density, flow_rate
            FROM crowd_analytics
            WHERE camera_id = $1 
            AND timestamp >= $2 
            AND timestamp <= $3
            ORDER BY timestamp
            """
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    query,
                    str(camera_id),
                    time_range.start_time,
                    time_range.end_time
                )
                
                analytics = []
                for row in rows:
                    if aggregation_interval:
                        # Aggregated data
                        analytics.append({
                            'time_bucket': row['time_bucket'].isoformat(),
                            'avg_density': float(row['avg_density']),
                            'max_density': float(row['max_density']),
                            'avg_flow_rate': float(row['avg_flow_rate']),
                            'data_points': row['data_points']
                        })
                    else:
                        # Raw data
                        analytics.append({
                            'camera_id': row['camera_id'],
                            'timestamp': row['timestamp'].isoformat(),
                            'crowd_data': json.loads(row['crowd_data']),
                            'density': float(row['density']),
                            'flow_rate': float(row['flow_rate'])
                        })
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get crowd analytics: {e}")
            return []
    
    async def save_anomaly_detection(
        self,
        camera_id: CameraID,
        timestamp: datetime,
        anomaly_data: Dict[str, Any]
    ) -> bool:
        """Save anomaly detection results."""
        query = """
        INSERT INTO anomaly_detections (
            camera_id, timestamp, anomaly_type, severity, anomaly_data, created_at
        ) VALUES ($1, $2, $3, $4, $5, NOW())
        """
        
        try:
            # Extract specific fields for indexing
            anomaly_type = anomaly_data.get('type', 'unknown')
            severity = anomaly_data.get('severity', 0.0)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    str(camera_id),
                    timestamp,
                    anomaly_type,
                    severity,
                    json.dumps(anomaly_data)
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to save anomaly detection: {e}")
            return False
    
    async def get_anomalies(
        self,
        camera_ids: Optional[List[CameraID]] = None,
        time_range: Optional[TimeRange] = None,
        anomaly_types: Optional[List[str]] = None,
        severity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get anomaly detection results."""
        base_query = """
        SELECT camera_id, timestamp, anomaly_type, severity, anomaly_data
        FROM anomaly_detections
        WHERE 1=1
        """
        
        params = []
        param_count = 0
        
        if camera_ids:
            param_count += 1
            camera_id_strs = [str(cid) for cid in camera_ids]
            base_query += f" AND camera_id = ANY(${param_count})"
            params.append(camera_id_strs)
        
        if time_range:
            param_count += 1
            base_query += f" AND timestamp >= ${param_count}"
            params.append(time_range.start_time)
            
            param_count += 1
            base_query += f" AND timestamp <= ${param_count}"
            params.append(time_range.end_time)
        
        if anomaly_types:
            param_count += 1
            base_query += f" AND anomaly_type = ANY(${param_count})"
            params.append(anomaly_types)
        
        if severity_threshold is not None:
            param_count += 1
            base_query += f" AND severity >= ${param_count}"
            params.append(severity_threshold)
        
        base_query += " ORDER BY timestamp DESC"
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(base_query, *params)
                
                anomalies = []
                for row in rows:
                    anomalies.append({
                        'camera_id': row['camera_id'],
                        'timestamp': row['timestamp'].isoformat(),
                        'anomaly_type': row['anomaly_type'],
                        'severity': float(row['severity']),
                        'anomaly_data': json.loads(row['anomaly_data'])
                    })
                
                return anomalies
                
        except Exception as e:
            logger.error(f"Failed to get anomalies: {e}")
            return []
    
    async def save_performance_metrics(
        self,
        component: str,
        metrics: Dict[str, Union[int, float, str]],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Save system performance metrics."""
        query = """
        INSERT INTO performance_metrics (
            component, timestamp, metrics_data, created_at
        ) VALUES ($1, $2, $3, NOW())
        """
        
        try:
            metric_timestamp = timestamp or datetime.utcnow()
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    component,
                    metric_timestamp,
                    json.dumps(metrics)
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
            return False
    
    async def get_performance_metrics(
        self,
        component: Optional[str] = None,
        time_range: Optional[TimeRange] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get system performance metrics."""
        base_query = """
        SELECT component, timestamp, metrics_data
        FROM performance_metrics
        WHERE 1=1
        """
        
        params = []
        param_count = 0
        
        if component:
            param_count += 1
            base_query += f" AND component = ${param_count}"
            params.append(component)
        
        if time_range:
            param_count += 1
            base_query += f" AND timestamp >= ${param_count}"
            params.append(time_range.start_time)
            
            param_count += 1
            base_query += f" AND timestamp <= ${param_count}"
            params.append(time_range.end_time)
        
        base_query += " ORDER BY timestamp DESC"
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(base_query, *params)
                
                metrics = []
                for row in rows:
                    metric_data = {
                        'component': row['component'],
                        'timestamp': row['timestamp'].isoformat(),
                        'metrics': json.loads(row['metrics_data'])
                    }
                    
                    # Filter by metric names if specified
                    if metric_names:
                        filtered_metrics = {
                            k: v for k, v in metric_data['metrics'].items()
                            if k in metric_names
                        }
                        metric_data['metrics'] = filtered_metrics
                    
                    metrics.append(metric_data)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return []
    
    async def get_analytics_summary(
        self,
        camera_ids: Optional[List[CameraID]] = None,
        time_range: Optional[TimeRange] = None,
        summary_type: str = "daily"
    ) -> Dict[str, Any]:
        """Get analytics summary data."""
        # This would be a complex aggregation query
        # Simplified implementation for demonstration
        try:
            summary = {
                'summary_type': summary_type,
                'total_behavioral_metrics': 0,
                'total_crowd_analytics': 0,
                'total_anomalies': 0,
                'avg_density': 0.0,
                'cameras_analyzed': 0
            }
            
            # Count different analytics types
            count_queries = [
                ("behavioral_metrics", "total_behavioral_metrics"),
                ("crowd_analytics", "total_crowd_analytics"),
                ("anomaly_detections", "total_anomalies")
            ]
            
            async with self.db_pool.acquire() as conn:
                for table, field in count_queries:
                    query = f"SELECT COUNT(*) FROM {table} WHERE 1=1"
                    params = []
                    param_count = 0
                    
                    if time_range:
                        param_count += 1
                        query += f" AND timestamp >= ${param_count}"
                        params.append(time_range.start_time)
                        
                        param_count += 1
                        query += f" AND timestamp <= ${param_count}"
                        params.append(time_range.end_time)
                    
                    count = await conn.fetchval(query, *params)
                    summary[field] = count or 0
                
                # Get average density
                density_query = """
                SELECT AVG(density) FROM crowd_analytics WHERE 1=1
                """
                if time_range:
                    density_query += " AND timestamp >= $1 AND timestamp <= $2"
                    avg_density = await conn.fetchval(
                        density_query, time_range.start_time, time_range.end_time
                    )
                else:
                    avg_density = await conn.fetchval(density_query)
                
                summary['avg_density'] = float(avg_density) if avg_density else 0.0
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
    
    async def delete_old_analytics(
        self,
        older_than: datetime,
        data_types: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """Delete old analytics data."""
        tables_to_clean = data_types or [
            'behavioral_metrics',
            'crowd_analytics', 
            'anomaly_detections',
            'performance_metrics'
        ]
        
        deletion_counts = {}
        
        try:
            async with self.db_pool.acquire() as conn:
                for table in tables_to_clean:
                    delete_query = f"""
                    DELETE FROM {table}
                    WHERE id IN (
                        SELECT id FROM {table}
                        WHERE timestamp < $1
                        ORDER BY timestamp
                        LIMIT $2
                    )
                    """
                    
                    total_deleted = 0
                    while True:
                        result = await conn.execute(delete_query, older_than, batch_size)
                        deleted_count = int(result.split()[-1])
                        
                        if deleted_count == 0:
                            break
                        
                        total_deleted += deleted_count
                        
                    deletion_counts[table] = total_deleted
                    logger.debug(f"Deleted {total_deleted} records from {table}")
                
                return deletion_counts
                
        except Exception as e:
            logger.error(f"Failed to delete old analytics: {e}")
            return {}
    
    async def get_available_date_ranges(
        self,
        camera_id: Optional[CameraID] = None,
        data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get available data date ranges."""
        tables_to_check = [data_type] if data_type else [
            'behavioral_metrics',
            'crowd_analytics',
            'anomaly_detections'
        ]
        
        try:
            date_ranges = {}
            
            async with self.db_pool.acquire() as conn:
                for table in tables_to_check:
                    query = f"""
                    SELECT 
                        MIN(timestamp) as earliest_date,
                        MAX(timestamp) as latest_date,
                        COUNT(*) as total_records
                    FROM {table}
                    WHERE 1=1
                    """
                    
                    params = []
                    if camera_id and table != 'performance_metrics':
                        query += " AND camera_id = $1"
                        params.append(str(camera_id))
                    
                    result = await conn.fetchrow(query, *params)
                    
                    if result:
                        date_ranges[table] = {
                            'earliest_date': result['earliest_date'].isoformat() if result['earliest_date'] else None,
                            'latest_date': result['latest_date'].isoformat() if result['latest_date'] else None,
                            'total_records': result['total_records'],
                            'has_data': bool(result['total_records'])
                        }
            
            return date_ranges
            
        except Exception as e:
            logger.error(f"Failed to get available date ranges: {e}")
            return {}