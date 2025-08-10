"""
Historical Data Storage & Retrieval Service

Comprehensive service for managing historical tracking data including:
- Time-based data storage and indexing
- Efficient temporal queries and filtering
- Historical data archival and retention
- Person trajectory reconstruction
- Movement path analysis
- Data compression and optimization
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import gzip
import pickle
from pathlib import Path

from app.infrastructure.cache.tracking_cache import TrackingCache
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.domains.detection.entities.detection import Detection
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import Coordinate

logger = logging.getLogger(__name__)


@dataclass
class HistoricalDataPoint:
    """Individual historical tracking data point."""
    timestamp: datetime
    global_person_id: str
    camera_id: str
    detection: Detection
    coordinates: Optional[Coordinate] = None
    person_identity: Optional[PersonIdentity] = None
    environment_id: str = "default"
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'global_person_id': self.global_person_id,
            'camera_id': self.camera_id,
            'environment_id': self.environment_id,
            'session_id': self.session_id,
            'detection': {
                'bbox': self.detection.bbox,
                'confidence': self.detection.confidence,
                'track_id': self.detection.track_id,
                'class_id': getattr(self.detection, 'class_id', 0)
            },
            'coordinates': {
                'x': self.coordinates.x,
                'y': self.coordinates.y,
                'confidence': getattr(self.coordinates, 'confidence', 1.0)
            } if self.coordinates else None,
            'person_identity': {
                'person_id': self.person_identity.person_id,
                'confidence': self.person_identity.confidence,
                'global_id': getattr(self.person_identity, 'global_id', None)
            } if self.person_identity else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalDataPoint':
        """Create from dictionary."""
        # Reconstruct detection
        detection = Detection(
            bbox=data['detection']['bbox'],
            confidence=data['detection']['confidence'],
            track_id=data['detection']['track_id'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
        
        # Reconstruct coordinates
        coordinates = None
        if data.get('coordinates'):
            coord_data = data['coordinates']
            coordinates = Coordinate(
                x=coord_data['x'],
                y=coord_data['y']
            )
        
        # Reconstruct person identity
        person_identity = None
        if data.get('person_identity'):
            identity_data = data['person_identity']
            person_identity = PersonIdentity(
                person_id=identity_data['person_id'],
                confidence=identity_data['confidence'],
                feature_vector=None  # Not stored in historical data
            )
        
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            global_person_id=data['global_person_id'],
            camera_id=data['camera_id'],
            environment_id=data.get('environment_id', 'default'),
            session_id=data.get('session_id'),
            detection=detection,
            coordinates=coordinates,
            person_identity=person_identity
        )


@dataclass
class TimeRange:
    """Time range specification for queries."""
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return self.duration_seconds / 3600.0
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within range."""
        return self.start_time <= timestamp <= self.end_time
    
    def overlap(self, other: 'TimeRange') -> Optional['TimeRange']:
        """Get overlap with another time range."""
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        
        if overlap_start < overlap_end:
            return TimeRange(overlap_start, overlap_end)
        return None


@dataclass
class HistoricalQueryFilter:
    """Filter specifications for historical data queries."""
    time_range: TimeRange
    environment_id: Optional[str] = None
    global_person_ids: Optional[List[str]] = None
    camera_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    limit: Optional[int] = None
    offset: Optional[int] = 0
    
    def matches(self, data_point: HistoricalDataPoint) -> bool:
        """Check if data point matches filter criteria."""
        # Time range check
        if not self.time_range.contains(data_point.timestamp):
            return False
        
        # Environment filter
        if self.environment_id and data_point.environment_id != self.environment_id:
            return False
        
        # Person ID filter
        if (self.global_person_ids and 
            data_point.global_person_id not in self.global_person_ids):
            return False
        
        # Camera filter
        if self.camera_ids and data_point.camera_id not in self.camera_ids:
            return False
        
        # Session filter
        if (self.session_ids and 
            data_point.session_id not in self.session_ids):
            return False
        
        # Confidence filter
        if (self.min_confidence and 
            data_point.detection.confidence < self.min_confidence):
            return False
        
        return True


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""
    environment_id: str
    retention_days: int = 90
    archive_days: int = 30
    compress_after_days: int = 7
    max_storage_gb: float = 100.0
    
    def should_archive(self, timestamp: datetime) -> bool:
        """Check if data should be archived."""
        age_days = (datetime.utcnow() - timestamp).days
        return age_days >= self.archive_days
    
    def should_compress(self, timestamp: datetime) -> bool:
        """Check if data should be compressed."""
        age_days = (datetime.utcnow() - timestamp).days
        return age_days >= self.compress_after_days
    
    def should_delete(self, timestamp: datetime) -> bool:
        """Check if data should be deleted."""
        age_days = (datetime.utcnow() - timestamp).days
        return age_days >= self.retention_days


class HistoricalDataService:
    """Comprehensive service for historical data management."""
    
    def __init__(
        self,
        tracking_cache: TrackingCache,
        tracking_repository: TrackingRepository,
        storage_path: Optional[str] = None
    ):
        self.cache = tracking_cache
        self.repository = tracking_repository
        
        # Storage configuration
        self.storage_path = Path(storage_path or "data/historical")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory data buffers (for performance)
        self.data_buffer: deque = deque(maxlen=10000)  # Recent data buffer
        self.index_cache: Dict[str, List[str]] = {}  # Index cache
        
        # Data retention policies
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self._initialize_default_policies()
        
        # Background processing
        self._storage_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance metrics
        self.storage_stats = {
            'total_stored': 0,
            'compressed_files': 0,
            'archived_files': 0,
            'query_count': 0,
            'avg_query_time_ms': 0.0
        }
        
        logger.info("HistoricalDataService initialized")
    
    async def start_service(self):
        """Start background processing tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._storage_task = asyncio.create_task(self._background_storage())
        self._cleanup_task = asyncio.create_task(self._background_cleanup())
        
        logger.info("HistoricalDataService background tasks started")
    
    async def stop_service(self):
        """Stop background processing tasks."""
        self._running = False
        
        # Cancel and wait for tasks
        for task in [self._storage_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush remaining data
        await self._flush_buffer()
        
        logger.info("HistoricalDataService stopped")
    
    # --- Data Storage Methods ---
    
    async def store_tracking_data(
        self,
        global_person_id: str,
        camera_id: str,
        detection: Detection,
        coordinates: Optional[Coordinate] = None,
        person_identity: Optional[PersonIdentity] = None,
        environment_id: str = "default",
        session_id: Optional[str] = None
    ):
        """Store individual tracking data point."""
        try:
            # Create historical data point
            data_point = HistoricalDataPoint(
                timestamp=detection.timestamp,
                global_person_id=global_person_id,
                camera_id=camera_id,
                detection=detection,
                coordinates=coordinates,
                person_identity=person_identity,
                environment_id=environment_id,
                session_id=session_id
            )
            
            # Add to buffer for batch processing
            self.data_buffer.append(data_point)
            
            # Store in cache for recent data access
            await self._cache_recent_data(data_point)
            
            # Update storage statistics
            self.storage_stats['total_stored'] += 1
            
            logger.debug(f"Stored tracking data for person {global_person_id} in camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Error storing tracking data: {e}")
            raise
    
    async def store_batch_data(
        self,
        data_points: List[HistoricalDataPoint]
    ):
        """Store batch of tracking data points."""
        try:
            # Add to buffer
            self.data_buffer.extend(data_points)
            
            # Cache recent data
            for data_point in data_points[-100:]:  # Cache last 100 points
                await self._cache_recent_data(data_point)
            
            # Update statistics
            self.storage_stats['total_stored'] += len(data_points)
            
            logger.info(f"Stored batch of {len(data_points)} historical data points")
            
        except Exception as e:
            logger.error(f"Error storing batch data: {e}")
            raise
    
    async def _cache_recent_data(self, data_point: HistoricalDataPoint):
        """Cache recent data for fast access."""
        try:
            cache_key = f"historical_{data_point.environment_id}_{data_point.global_person_id}"
            
            # Get existing cached data
            cached_data = await self.cache.get_json(cache_key) or []
            
            # Add new data point
            cached_data.append(data_point.to_dict())
            
            # Keep only recent data (last 100 points per person)
            if len(cached_data) > 100:
                cached_data = cached_data[-100:]
            
            # Store back in cache with 1 hour TTL
            await self.cache.set_json(cache_key, cached_data, ttl=3600)
            
        except Exception as e:
            logger.error(f"Error caching recent data: {e}")
    
    # --- Data Retrieval Methods ---
    
    async def query_historical_data(
        self,
        query_filter: HistoricalQueryFilter
    ) -> List[HistoricalDataPoint]:
        """Query historical data with filters."""
        start_time = time.time()
        
        try:
            # Check cache for recent data first
            cached_results = await self._query_cached_data(query_filter)
            
            # Query database for older data
            db_results = await self._query_database_data(query_filter)
            
            # Combine and deduplicate results
            all_results = cached_results + db_results
            unique_results = self._deduplicate_results(all_results)
            
            # Sort by timestamp
            unique_results.sort(key=lambda x: x.timestamp)
            
            # Apply limit and offset
            if query_filter.offset:
                unique_results = unique_results[query_filter.offset:]
            
            if query_filter.limit:
                unique_results = unique_results[:query_filter.limit]
            
            # Update performance metrics
            query_time = (time.time() - start_time) * 1000
            self._update_query_metrics(query_time)
            
            logger.info(
                f"Historical data query returned {len(unique_results)} results "
                f"in {query_time:.1f}ms"
            )
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error querying historical data: {e}")
            raise
    
    async def _query_cached_data(
        self,
        query_filter: HistoricalQueryFilter
    ) -> List[HistoricalDataPoint]:
        """Query cached recent data."""
        try:
            results = []
            
            # Determine cache keys to check
            cache_keys = []
            
            if query_filter.global_person_ids:
                for person_id in query_filter.global_person_ids:
                    env_id = query_filter.environment_id or "default"
                    cache_keys.append(f"historical_{env_id}_{person_id}")
            else:
                # Get all cache keys for environment (this is expensive, avoid if possible)
                env_id = query_filter.environment_id or "default"
                pattern = f"historical_{env_id}_*"
                cache_keys = await self.cache.get_keys_pattern(pattern)
            
            # Query each cache key
            for cache_key in cache_keys:
                cached_data = await self.cache.get_json(cache_key)
                
                if cached_data:
                    for data_dict in cached_data:
                        try:
                            data_point = HistoricalDataPoint.from_dict(data_dict)
                            
                            if query_filter.matches(data_point):
                                results.append(data_point)
                                
                        except Exception as e:
                            logger.warning(f"Error parsing cached data: {e}")
                            continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying cached data: {e}")
            return []
    
    async def _query_database_data(
        self,
        query_filter: HistoricalQueryFilter
    ) -> List[HistoricalDataPoint]:
        """Query database for historical data."""
        try:
            # Convert filter to database query parameters
            db_params = {
                'start_time': query_filter.time_range.start_time,
                'end_time': query_filter.time_range.end_time,
                'environment_id': query_filter.environment_id,
                'person_ids': query_filter.global_person_ids,
                'camera_ids': query_filter.camera_ids,
                'session_ids': query_filter.session_ids,
                'min_confidence': query_filter.min_confidence,
                'limit': query_filter.limit,
                'offset': query_filter.offset
            }
            
            # Query database
            db_data = await self.repository.get_historical_tracking_data(**db_params)
            
            # Convert to HistoricalDataPoint objects
            results = []
            for record in db_data:
                try:
                    data_point = self._convert_db_record_to_data_point(record)
                    results.append(data_point)
                except Exception as e:
                    logger.warning(f"Error converting database record: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying database data: {e}")
            return []
    
    def _convert_db_record_to_data_point(self, record: Dict[str, Any]) -> HistoricalDataPoint:
        """Convert database record to HistoricalDataPoint."""
        # Create Detection object
        detection = Detection(
            bbox=record['bbox'],
            confidence=record['confidence'],
            track_id=record['track_id'],
            timestamp=record['timestamp']
        )
        
        # Create Coordinate object if available
        coordinates = None
        if record.get('map_x') is not None and record.get('map_y') is not None:
            coordinates = Coordinate(
                x=record['map_x'],
                y=record['map_y']
            )
        
        # Create PersonIdentity object if available
        person_identity = None
        if record.get('person_id') and record.get('identity_confidence'):
            person_identity = PersonIdentity(
                person_id=record['person_id'],
                confidence=record['identity_confidence'],
                feature_vector=None
            )
        
        return HistoricalDataPoint(
            timestamp=record['timestamp'],
            global_person_id=record['global_person_id'],
            camera_id=record['camera_id'],
            environment_id=record.get('environment_id', 'default'),
            session_id=record.get('session_id'),
            detection=detection,
            coordinates=coordinates,
            person_identity=person_identity
        )
    
    def _deduplicate_results(
        self,
        results: List[HistoricalDataPoint]
    ) -> List[HistoricalDataPoint]:
        """Remove duplicate data points."""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create unique key based on timestamp, person_id, and camera_id
            key = (
                result.timestamp.isoformat(),
                result.global_person_id,
                result.camera_id
            )
            
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
    
    # --- Specialized Query Methods ---
    
    async def get_person_trajectory(
        self,
        global_person_id: str,
        time_range: TimeRange,
        environment_id: Optional[str] = None
    ) -> List[HistoricalDataPoint]:
        """Get complete trajectory for a specific person."""
        try:
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id,
                global_person_ids=[global_person_id]
            )
            
            trajectory = await self.query_historical_data(query_filter)
            
            logger.info(
                f"Retrieved trajectory for person {global_person_id}: "
                f"{len(trajectory)} data points"
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error getting person trajectory: {e}")
            raise
    
    async def get_camera_activity(
        self,
        camera_id: str,
        time_range: TimeRange,
        environment_id: Optional[str] = None
    ) -> List[HistoricalDataPoint]:
        """Get all activity for a specific camera."""
        try:
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id,
                camera_ids=[camera_id]
            )
            
            activity = await self.query_historical_data(query_filter)
            
            logger.info(
                f"Retrieved camera activity for {camera_id}: "
                f"{len(activity)} data points"
            )
            
            return activity
            
        except Exception as e:
            logger.error(f"Error getting camera activity: {e}")
            raise
    
    async def get_environment_summary(
        self,
        environment_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Get summary statistics for an environment."""
        try:
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id
            )
            
            data = await self.query_historical_data(query_filter)
            
            # Calculate summary statistics
            unique_persons = set(d.global_person_id for d in data)
            unique_cameras = set(d.camera_id for d in data)
            
            person_activity = defaultdict(int)
            camera_activity = defaultdict(int)
            
            for data_point in data:
                person_activity[data_point.global_person_id] += 1
                camera_activity[data_point.camera_id] += 1
            
            summary = {
                'environment_id': environment_id,
                'time_range': {
                    'start': time_range.start_time.isoformat(),
                    'end': time_range.end_time.isoformat(),
                    'duration_hours': time_range.duration_hours
                },
                'total_detections': len(data),
                'unique_persons': len(unique_persons),
                'unique_cameras': len(unique_cameras),
                'persons_per_camera': dict(camera_activity),
                'detections_per_person': dict(person_activity),
                'most_active_person': max(person_activity, key=person_activity.get) if person_activity else None,
                'most_active_camera': max(camera_activity, key=camera_activity.get) if camera_activity else None
            }
            
            logger.info(f"Generated environment summary for {environment_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting environment summary: {e}")
            raise
    
    # --- Data Management Methods ---
    
    def set_retention_policy(
        self,
        environment_id: str,
        policy: DataRetentionPolicy
    ):
        """Set data retention policy for environment."""
        self.retention_policies[environment_id] = policy
        logger.info(f"Set retention policy for environment {environment_id}")
    
    async def get_available_date_ranges(
        self,
        environment_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get available date ranges for historical data."""
        try:
            date_ranges = await self.repository.get_available_date_ranges(environment_id)
            
            logger.info(f"Retrieved available date ranges for environment {environment_id}")
            return date_ranges
            
        except Exception as e:
            logger.error(f"Error getting available date ranges: {e}")
            raise
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics and metrics."""
        try:
            # Get database statistics
            db_stats = await self.repository.get_storage_statistics()
            
            # Get file system statistics
            fs_stats = self._get_filesystem_statistics()
            
            # Combine with service statistics
            combined_stats = {
                'service_stats': self.storage_stats.copy(),
                'database_stats': db_stats,
                'filesystem_stats': fs_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            return {'error': str(e)}
    
    def _get_filesystem_statistics(self) -> Dict[str, Any]:
        """Get file system storage statistics."""
        try:
            total_size = 0
            file_count = 0
            compressed_count = 0
            
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
                    
                    if file_path.suffix == '.gz':
                        compressed_count += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'compressed_files': compressed_count,
                'storage_path': str(self.storage_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting filesystem statistics: {e}")
            return {'error': str(e)}
    
    # --- Background Processing ---
    
    async def _background_storage(self):
        """Background task for storing buffered data."""
        while self._running:
            try:
                # Process buffer every 30 seconds or when it's getting full
                await asyncio.sleep(30)
                
                if len(self.data_buffer) > 0:
                    await self._flush_buffer()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background storage: {e}")
                await asyncio.sleep(60)
    
    async def _background_cleanup(self):
        """Background task for data cleanup and archival."""
        while self._running:
            try:
                # Run cleanup every 6 hours
                await asyncio.sleep(6 * 3600)
                
                await self._cleanup_expired_data()
                await self._compress_old_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _flush_buffer(self):
        """Flush data buffer to persistent storage."""
        try:
            if not self.data_buffer:
                return
            
            # Get current buffer content
            buffer_data = list(self.data_buffer)
            self.data_buffer.clear()
            
            # Convert to database records
            db_records = []
            for data_point in buffer_data:
                try:
                    record = self._convert_data_point_to_db_record(data_point)
                    db_records.append(record)
                except Exception as e:
                    logger.warning(f"Error converting data point to db record: {e}")
                    continue
            
            # Store in database
            if db_records:
                await self.repository.store_historical_batch(db_records)
                logger.info(f"Flushed {len(db_records)} records to database")
            
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
    
    def _convert_data_point_to_db_record(self, data_point: HistoricalDataPoint) -> Dict[str, Any]:
        """Convert HistoricalDataPoint to database record format."""
        record = {
            'timestamp': data_point.timestamp,
            'global_person_id': data_point.global_person_id,
            'camera_id': data_point.camera_id,
            'environment_id': data_point.environment_id,
            'session_id': data_point.session_id,
            'bbox': data_point.detection.bbox,
            'confidence': data_point.detection.confidence,
            'track_id': data_point.detection.track_id,
            'class_id': getattr(data_point.detection, 'class_id', 0)
        }
        
        # Add coordinates if available
        if data_point.coordinates:
            record['map_x'] = data_point.coordinates.x
            record['map_y'] = data_point.coordinates.y
            record['coordinate_confidence'] = getattr(data_point.coordinates, 'confidence', 1.0)
        
        # Add person identity if available
        if data_point.person_identity:
            record['person_id'] = data_point.person_identity.person_id
            record['identity_confidence'] = data_point.person_identity.confidence
            record['global_id'] = getattr(data_point.person_identity, 'global_id', None)
        
        return record
    
    async def _cleanup_expired_data(self):
        """Clean up expired data based on retention policies."""
        try:
            for env_id, policy in self.retention_policies.items():
                # Get expired data
                cutoff_time = datetime.utcnow() - timedelta(days=policy.retention_days)
                
                # Delete from database
                deleted_count = await self.repository.delete_data_before(
                    cutoff_time, env_id
                )
                
                if deleted_count > 0:
                    logger.info(
                        f"Deleted {deleted_count} expired records for environment {env_id}"
                    )
            
        except Exception as e:
            logger.error(f"Error cleaning up expired data: {e}")
    
    async def _compress_old_data(self):
        """Compress old data files."""
        try:
            for env_id, policy in self.retention_policies.items():
                cutoff_time = datetime.utcnow() - timedelta(days=policy.compress_after_days)
                
                # Find files older than compression threshold
                env_path = self.storage_path / env_id
                if not env_path.exists():
                    continue
                
                for file_path in env_path.glob("*.json"):
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_mtime < cutoff_time and not str(file_path).endswith('.gz'):
                        await self._compress_file(file_path)
            
        except Exception as e:
            logger.error(f"Error compressing old data: {e}")
    
    async def _compress_file(self, file_path: Path):
        """Compress a single file."""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            file_path.unlink()
            
            self.storage_stats['compressed_files'] += 1
            logger.debug(f"Compressed file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error compressing file {file_path}: {e}")
    
    # --- Utility Methods ---
    
    def _initialize_default_policies(self):
        """Initialize default retention policies."""
        # Default policy
        self.retention_policies['default'] = DataRetentionPolicy(
            environment_id='default',
            retention_days=90,
            archive_days=30,
            compress_after_days=7,
            max_storage_gb=50.0
        )
        
        # Campus environment (longer retention)
        self.retention_policies['campus'] = DataRetentionPolicy(
            environment_id='campus',
            retention_days=180,
            archive_days=60,
            compress_after_days=14,
            max_storage_gb=100.0
        )
        
        # Factory environment (shorter retention, more compression)
        self.retention_policies['factory'] = DataRetentionPolicy(
            environment_id='factory',
            retention_days=60,
            archive_days=15,
            compress_after_days=3,
            max_storage_gb=30.0
        )
    
    def _update_query_metrics(self, query_time_ms: float):
        """Update query performance metrics."""
        self.storage_stats['query_count'] += 1
        
        # Update running average
        current_avg = self.storage_stats['avg_query_time_ms']
        query_count = self.storage_stats['query_count']
        
        self.storage_stats['avg_query_time_ms'] = (
            (current_avg * (query_count - 1) + query_time_ms) / query_count
        )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'HistoricalDataService',
            'running': self._running,
            'buffer_size': len(self.data_buffer),
            'storage_path': str(self.storage_path),
            'retention_policies': len(self.retention_policies),
            'statistics': self.storage_stats.copy(),
            'background_tasks': {
                'storage_task_running': self._storage_task and not self._storage_task.done(),
                'cleanup_task_running': self._cleanup_task and not self._cleanup_task.done()
            }
        }


# Global service instance
_historical_data_service: Optional[HistoricalDataService] = None


def get_historical_data_service() -> Optional[HistoricalDataService]:
    """Get the global historical data service instance."""
    return _historical_data_service


def initialize_historical_data_service(
    tracking_cache: TrackingCache,
    tracking_repository: TrackingRepository,
    storage_path: Optional[str] = None
) -> HistoricalDataService:
    """Initialize the global historical data service."""
    global _historical_data_service
    if _historical_data_service is None:
        _historical_data_service = HistoricalDataService(
            tracking_cache, tracking_repository, storage_path
        )
    return _historical_data_service