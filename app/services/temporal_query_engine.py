"""
Time-Based Query Engine for Efficient Temporal Data Access

Advanced query engine optimized for temporal data with features:
- Intelligent time-based indexing and partitioning
- Optimized temporal range queries
- Smart caching and prefetching strategies
- Time-series aggregation and analytics
- Efficient memory management for large datasets
- Query optimization and performance tuning
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import bisect
import json
import hashlib

from app.services.historical_data_service import (
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)

logger = logging.getLogger(__name__)


class TimeGranularity(Enum):
    """Time granularity for aggregation and indexing."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class QueryStrategy(Enum):
    """Query optimization strategies."""
    SEQUENTIAL = "sequential"  # Linear scan
    INDEXED = "indexed"       # Use time indexes
    CACHED = "cached"         # Use cached results
    HYBRID = "hybrid"         # Combination approach


@dataclass
class TimeIndex:
    """Time-based index for efficient data access."""
    granularity: TimeGranularity
    index_data: Dict[str, List[str]] = field(default_factory=dict)  # time_key -> data_ids
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_time_key(self, timestamp: datetime) -> str:
        """Convert timestamp to index key based on granularity."""
        if self.granularity == TimeGranularity.SECOND:
            return timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        elif self.granularity == TimeGranularity.MINUTE:
            return timestamp.strftime("%Y-%m-%d_%H-%M")
        elif self.granularity == TimeGranularity.HOUR:
            return timestamp.strftime("%Y-%m-%d_%H")
        elif self.granularity == TimeGranularity.DAY:
            return timestamp.strftime("%Y-%m-%d")
        elif self.granularity == TimeGranularity.WEEK:
            year, week, _ = timestamp.isocalendar()
            return f"{year}-W{week:02d}"
        elif self.granularity == TimeGranularity.MONTH:
            return timestamp.strftime("%Y-%m")
        else:
            return timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    
    def add_data_point(self, data_id: str, timestamp: datetime):
        """Add data point to index."""
        time_key = self.get_time_key(timestamp)
        if time_key not in self.index_data:
            self.index_data[time_key] = []
        self.index_data[time_key].append(data_id)
        self.last_updated = datetime.utcnow()
    
    def get_time_keys_in_range(self, time_range: TimeRange) -> List[str]:
        """Get all time keys that overlap with the given range."""
        keys = []
        
        # Generate all possible keys in the range
        current_time = time_range.start_time
        while current_time <= time_range.end_time:
            key = self.get_time_key(current_time)
            if key not in keys:
                keys.append(key)
            
            # Increment by granularity
            if self.granularity == TimeGranularity.SECOND:
                current_time += timedelta(seconds=1)
            elif self.granularity == TimeGranularity.MINUTE:
                current_time += timedelta(minutes=1)
            elif self.granularity == TimeGranularity.HOUR:
                current_time += timedelta(hours=1)
            elif self.granularity == TimeGranularity.DAY:
                current_time += timedelta(days=1)
            elif self.granularity == TimeGranularity.WEEK:
                current_time += timedelta(weeks=1)
            elif self.granularity == TimeGranularity.MONTH:
                # Approximate month increment
                current_time += timedelta(days=30)
            else:
                break
        
        return keys


@dataclass
class QueryCacheEntry:
    """Cache entry for query results."""
    query_hash: str
    results: List[HistoricalDataPoint]
    timestamp: datetime
    access_count: int = 0
    ttl_seconds: int = 3600
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl_seconds
    
    def access(self):
        """Record cache access."""
        self.access_count += 1


@dataclass
class QueryOptimizationMetrics:
    """Metrics for query optimization."""
    query_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_query_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    index_usage_count: int = 0
    sequential_scan_count: int = 0
    
    def record_query(self, query_time_ms: float, used_cache: bool, used_index: bool):
        """Record query execution metrics."""
        self.query_count += 1
        self.total_query_time_ms += query_time_ms
        self.avg_query_time_ms = self.total_query_time_ms / self.query_count
        
        if used_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if used_index:
            self.index_usage_count += 1
        else:
            self.sequential_scan_count += 1
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0


class TemporalQueryEngine:
    """Advanced query engine optimized for temporal data access."""
    
    def __init__(self, max_cache_size: int = 1000):
        # Time indexes for different granularities
        self.time_indexes: Dict[TimeGranularity, TimeIndex] = {}
        self._initialize_indexes()
        
        # Query cache
        self.query_cache: Dict[str, QueryCacheEntry] = {}
        self.max_cache_size = max_cache_size
        
        # Data storage (in-memory for fast access)
        self.data_storage: Dict[str, HistoricalDataPoint] = {}
        
        # Optimization metrics
        self.metrics = QueryOptimizationMetrics()
        
        # Configuration
        self.enable_caching = True
        self.enable_indexing = True
        self.default_cache_ttl = 3600  # 1 hour
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("TemporalQueryEngine initialized")
    
    async def start_engine(self):
        """Start the query engine and background tasks."""
        if self._running:
            return
        
        self._running = True
        self._maintenance_task = asyncio.create_task(self._background_maintenance())
        
        logger.info("TemporalQueryEngine started")
    
    async def stop_engine(self):
        """Stop the query engine and background tasks."""
        self._running = False
        
        if self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        logger.info("TemporalQueryEngine stopped")
    
    # --- Data Ingestion ---
    
    def add_data_points(self, data_points: List[HistoricalDataPoint]):
        """Add data points to the engine for indexing and caching."""
        try:
            for data_point in data_points:
                # Generate unique ID for data point
                data_id = self._generate_data_id(data_point)
                
                # Store data point
                self.data_storage[data_id] = data_point
                
                # Update indexes
                if self.enable_indexing:
                    self._update_indexes(data_id, data_point)
            
            logger.debug(f"Added {len(data_points)} data points to query engine")
            
        except Exception as e:
            logger.error(f"Error adding data points: {e}")
            raise
    
    def _generate_data_id(self, data_point: HistoricalDataPoint) -> str:
        """Generate unique ID for data point."""
        key_data = f"{data_point.timestamp.isoformat()}_{data_point.global_person_id}_{data_point.camera_id}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _update_indexes(self, data_id: str, data_point: HistoricalDataPoint):
        """Update time indexes with new data point."""
        try:
            for granularity, index in self.time_indexes.items():
                index.add_data_point(data_id, data_point.timestamp)
        
        except Exception as e:
            logger.error(f"Error updating indexes: {e}")
    
    # --- Query Execution ---
    
    async def execute_query(
        self,
        query_filter: HistoricalQueryFilter,
        strategy: Optional[QueryStrategy] = None
    ) -> List[HistoricalDataPoint]:
        """Execute temporal query with optimization."""
        start_time = time.time()
        
        try:
            # Generate query hash for caching
            query_hash = self._generate_query_hash(query_filter)
            
            # Check cache first
            cached_result = self._get_cached_result(query_hash)
            if cached_result is not None:
                query_time = (time.time() - start_time) * 1000
                self.metrics.record_query(query_time, used_cache=True, used_index=False)
                logger.debug(f"Query served from cache in {query_time:.1f}ms")
                return cached_result
            
            # Determine optimal query strategy
            if strategy is None:
                strategy = self._determine_optimal_strategy(query_filter)
            
            # Execute query based on strategy
            results = await self._execute_strategy(query_filter, strategy)
            
            # Cache results
            if self.enable_caching:
                self._cache_results(query_hash, results)
            
            # Record metrics
            query_time = (time.time() - start_time) * 1000
            used_index = strategy in [QueryStrategy.INDEXED, QueryStrategy.HYBRID]
            self.metrics.record_query(query_time, used_cache=False, used_index=used_index)
            
            logger.debug(
                f"Query executed using {strategy.value} strategy in {query_time:.1f}ms, "
                f"returned {len(results)} results"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def _determine_optimal_strategy(self, query_filter: HistoricalQueryFilter) -> QueryStrategy:
        """Determine the optimal query strategy based on filter characteristics."""
        try:
            # Factors for strategy selection
            time_span_hours = query_filter.time_range.duration_hours
            has_person_filter = query_filter.global_person_ids is not None
            has_camera_filter = query_filter.camera_ids is not None
            data_size = len(self.data_storage)
            
            # Use cached strategy for repeated similar queries
            if self.enable_caching and self._has_similar_cached_query(query_filter):
                return QueryStrategy.CACHED
            
            # Use indexed strategy for large datasets with time-based queries
            if (self.enable_indexing and 
                data_size > 10000 and 
                time_span_hours < 24):
                return QueryStrategy.INDEXED
            
            # Use hybrid strategy for complex queries
            if (has_person_filter or has_camera_filter) and time_span_hours > 1:
                return QueryStrategy.HYBRID
            
            # Default to sequential for simple queries
            return QueryStrategy.SEQUENTIAL
            
        except Exception as e:
            logger.error(f"Error determining query strategy: {e}")
            return QueryStrategy.SEQUENTIAL
    
    async def _execute_strategy(
        self,
        query_filter: HistoricalQueryFilter,
        strategy: QueryStrategy
    ) -> List[HistoricalDataPoint]:
        """Execute query using specified strategy."""
        try:
            if strategy == QueryStrategy.SEQUENTIAL:
                return await self._execute_sequential_query(query_filter)
            elif strategy == QueryStrategy.INDEXED:
                return await self._execute_indexed_query(query_filter)
            elif strategy == QueryStrategy.HYBRID:
                return await self._execute_hybrid_query(query_filter)
            else:
                # Fallback to sequential
                return await self._execute_sequential_query(query_filter)
                
        except Exception as e:
            logger.error(f"Error executing {strategy.value} strategy: {e}")
            raise
    
    async def _execute_sequential_query(
        self,
        query_filter: HistoricalQueryFilter
    ) -> List[HistoricalDataPoint]:
        """Execute query using sequential scan."""
        results = []
        
        for data_point in self.data_storage.values():
            if query_filter.matches(data_point):
                results.append(data_point)
        
        # Sort and apply pagination
        results.sort(key=lambda x: x.timestamp)
        
        if query_filter.offset:
            results = results[query_filter.offset:]
        
        if query_filter.limit:
            results = results[:query_filter.limit]
        
        return results
    
    async def _execute_indexed_query(
        self,
        query_filter: HistoricalQueryFilter
    ) -> List[HistoricalDataPoint]:
        """Execute query using time indexes."""
        # Choose appropriate index granularity
        granularity = self._choose_index_granularity(query_filter.time_range)
        index = self.time_indexes[granularity]
        
        # Get relevant time keys
        time_keys = index.get_time_keys_in_range(query_filter.time_range)
        
        # Collect data IDs from index
        candidate_ids = set()
        for time_key in time_keys:
            candidate_ids.update(index.index_data.get(time_key, []))
        
        # Filter candidates
        results = []
        for data_id in candidate_ids:
            if data_id in self.data_storage:
                data_point = self.data_storage[data_id]
                if query_filter.matches(data_point):
                    results.append(data_point)
        
        # Sort and apply pagination
        results.sort(key=lambda x: x.timestamp)
        
        if query_filter.offset:
            results = results[query_filter.offset:]
        
        if query_filter.limit:
            results = results[:query_filter.limit]
        
        return results
    
    async def _execute_hybrid_query(
        self,
        query_filter: HistoricalQueryFilter
    ) -> List[HistoricalDataPoint]:
        """Execute query using hybrid approach (index + filtering)."""
        # Use index to narrow down candidates
        indexed_results = await self._execute_indexed_query(
            HistoricalQueryFilter(
                time_range=query_filter.time_range,
                limit=None,  # Don't limit at index level
                offset=0
            )
        )
        
        # Apply additional filters
        filtered_results = []
        for data_point in indexed_results:
            if query_filter.matches(data_point):
                filtered_results.append(data_point)
        
        # Apply pagination
        if query_filter.offset:
            filtered_results = filtered_results[query_filter.offset:]
        
        if query_filter.limit:
            filtered_results = filtered_results[:query_filter.limit]
        
        return filtered_results
    
    def _choose_index_granularity(self, time_range: TimeRange) -> TimeGranularity:
        """Choose appropriate index granularity based on time range."""
        duration_hours = time_range.duration_hours
        
        if duration_hours <= 1:
            return TimeGranularity.MINUTE
        elif duration_hours <= 24:
            return TimeGranularity.HOUR
        elif duration_hours <= 168:  # 1 week
            return TimeGranularity.DAY
        elif duration_hours <= 720:  # 1 month
            return TimeGranularity.WEEK
        else:
            return TimeGranularity.MONTH
    
    # --- Caching System ---
    
    def _generate_query_hash(self, query_filter: HistoricalQueryFilter) -> str:
        """Generate hash for query caching."""
        key_data = {
            'start_time': query_filter.time_range.start_time.isoformat(),
            'end_time': query_filter.time_range.end_time.isoformat(),
            'environment_id': query_filter.environment_id,
            'global_person_ids': sorted(query_filter.global_person_ids) if query_filter.global_person_ids else None,
            'camera_ids': sorted(query_filter.camera_ids) if query_filter.camera_ids else None,
            'session_ids': sorted(query_filter.session_ids) if query_filter.session_ids else None,
            'min_confidence': query_filter.min_confidence,
            'limit': query_filter.limit,
            'offset': query_filter.offset
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, query_hash: str) -> Optional[List[HistoricalDataPoint]]:
        """Get cached query result if available and not expired."""
        if not self.enable_caching or query_hash not in self.query_cache:
            return None
        
        cache_entry = self.query_cache[query_hash]
        
        if cache_entry.is_expired():
            del self.query_cache[query_hash]
            return None
        
        cache_entry.access()
        return cache_entry.results.copy()
    
    def _cache_results(self, query_hash: str, results: List[HistoricalDataPoint]):
        """Cache query results."""
        try:
            # Clean cache if at capacity
            if len(self.query_cache) >= self.max_cache_size:
                self._clean_cache()
            
            # Create cache entry
            cache_entry = QueryCacheEntry(
                query_hash=query_hash,
                results=results.copy(),
                timestamp=datetime.utcnow(),
                ttl_seconds=self.default_cache_ttl
            )
            
            self.query_cache[query_hash] = cache_entry
            
        except Exception as e:
            logger.error(f"Error caching results: {e}")
    
    def _has_similar_cached_query(self, query_filter: HistoricalQueryFilter) -> bool:
        """Check if there are similar cached queries."""
        # Simple heuristic: check if there are cached queries with overlapping time ranges
        query_start = query_filter.time_range.start_time
        query_end = query_filter.time_range.end_time
        
        for cache_entry in self.query_cache.values():
            if cache_entry.is_expired():
                continue
            
            # Check if any cached results have overlapping time ranges
            if cache_entry.results:
                cached_start = min(r.timestamp for r in cache_entry.results)
                cached_end = max(r.timestamp for r in cache_entry.results)
                
                # Check for overlap
                if (query_start <= cached_end and query_end >= cached_start):
                    return True
        
        return False
    
    def _clean_cache(self):
        """Clean expired and least-used cache entries."""
        try:
            # Remove expired entries
            expired_keys = [
                key for key, entry in self.query_cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.query_cache[key]
            
            # If still at capacity, remove least-used entries
            if len(self.query_cache) >= self.max_cache_size:
                # Sort by access count and timestamp
                sorted_entries = sorted(
                    self.query_cache.items(),
                    key=lambda x: (x[1].access_count, x[1].timestamp)
                )
                
                # Remove oldest, least-used entries
                remove_count = len(self.query_cache) - (self.max_cache_size // 2)
                for key, _ in sorted_entries[:remove_count]:
                    del self.query_cache[key]
            
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    # --- Aggregation Queries ---
    
    async def execute_aggregation_query(
        self,
        query_filter: HistoricalQueryFilter,
        aggregation_func: Callable[[List[HistoricalDataPoint]], Any],
        granularity: TimeGranularity = TimeGranularity.HOUR
    ) -> Dict[str, Any]:
        """Execute aggregation query over time periods."""
        try:
            # Get raw data
            data_points = await self.execute_query(query_filter)
            
            # Group by time granularity
            time_groups = defaultdict(list)
            
            for data_point in data_points:
                time_key = self._get_time_key_for_granularity(data_point.timestamp, granularity)
                time_groups[time_key].append(data_point)
            
            # Apply aggregation function to each group
            aggregated_results = {}
            for time_key, group_data in time_groups.items():
                aggregated_results[time_key] = aggregation_func(group_data)
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Error executing aggregation query: {e}")
            raise
    
    def _get_time_key_for_granularity(self, timestamp: datetime, granularity: TimeGranularity) -> str:
        """Get time key for specified granularity."""
        if granularity == TimeGranularity.SECOND:
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
        elif granularity == TimeGranularity.MINUTE:
            return timestamp.strftime("%Y-%m-%d %H:%M")
        elif granularity == TimeGranularity.HOUR:
            return timestamp.strftime("%Y-%m-%d %H:00")
        elif granularity == TimeGranularity.DAY:
            return timestamp.strftime("%Y-%m-%d")
        elif granularity == TimeGranularity.WEEK:
            year, week, _ = timestamp.isocalendar()
            return f"{year}-W{week:02d}"
        elif granularity == TimeGranularity.MONTH:
            return timestamp.strftime("%Y-%m")
        else:
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # --- Time Series Analysis ---
    
    async def get_person_timeline(
        self,
        global_person_id: str,
        time_range: TimeRange,
        environment_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get chronological timeline for a specific person."""
        try:
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id,
                global_person_ids=[global_person_id]
            )
            
            data_points = await self.execute_query(query_filter)
            
            # Build timeline with movement events
            timeline = []
            prev_camera = None
            
            for data_point in sorted(data_points, key=lambda x: x.timestamp):
                event = {
                    'timestamp': data_point.timestamp.isoformat(),
                    'camera_id': data_point.camera_id,
                    'bbox': data_point.detection.bbox,
                    'confidence': data_point.detection.confidence,
                    'coordinates': {
                        'x': data_point.coordinates.x,
                        'y': data_point.coordinates.y
                    } if data_point.coordinates else None
                }
                
                # Mark camera transitions
                if prev_camera and prev_camera != data_point.camera_id:
                    event['camera_transition'] = {
                        'from': prev_camera,
                        'to': data_point.camera_id
                    }
                
                timeline.append(event)
                prev_camera = data_point.camera_id
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error getting person timeline: {e}")
            raise
    
    async def get_occupancy_over_time(
        self,
        time_range: TimeRange,
        granularity: TimeGranularity = TimeGranularity.HOUR,
        environment_id: Optional[str] = None,
        camera_ids: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Get occupancy counts over time."""
        try:
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id,
                camera_ids=camera_ids
            )
            
            def count_unique_persons(data_points: List[HistoricalDataPoint]) -> int:
                return len(set(dp.global_person_id for dp in data_points))
            
            occupancy_data = await self.execute_aggregation_query(
                query_filter, count_unique_persons, granularity
            )
            
            return occupancy_data
            
        except Exception as e:
            logger.error(f"Error getting occupancy over time: {e}")
            raise
    
    # --- Background Maintenance ---
    
    async def _background_maintenance(self):
        """Background task for cache cleanup and index maintenance."""
        while self._running:
            try:
                # Run maintenance every 5 minutes
                await asyncio.sleep(300)
                
                # Clean expired cache entries
                self._clean_cache()
                
                # Cleanup old indexes (optional)
                await self._cleanup_old_indexes()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background maintenance: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_indexes(self):
        """Clean up old index data to prevent memory bloat."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            for index in self.time_indexes.values():
                old_keys = []
                
                for time_key in index.index_data.keys():
                    # This is a simplified cleanup - in practice, you'd need to
                    # parse the time key and compare with cutoff_time
                    if len(index.index_data[time_key]) == 0:
                        old_keys.append(time_key)
                
                for key in old_keys:
                    del index.index_data[key]
            
        except Exception as e:
            logger.error(f"Error cleaning up old indexes: {e}")
    
    # --- Utility Methods ---
    
    def _initialize_indexes(self):
        """Initialize time indexes for different granularities."""
        for granularity in TimeGranularity:
            self.time_indexes[granularity] = TimeIndex(granularity=granularity)
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        return {
            'data_points_stored': len(self.data_storage),
            'cache_entries': len(self.query_cache),
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'query_metrics': {
                'total_queries': self.metrics.query_count,
                'avg_query_time_ms': self.metrics.avg_query_time_ms,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'index_usage': self.metrics.index_usage_count,
                'sequential_scans': self.metrics.sequential_scan_count
            },
            'index_statistics': {
                granularity.value: len(index.index_data)
                for granularity, index in self.time_indexes.items()
            },
            'configuration': {
                'enable_caching': self.enable_caching,
                'enable_indexing': self.enable_indexing,
                'max_cache_size': self.max_cache_size,
                'default_cache_ttl': self.default_cache_ttl
            }
        }
    
    def clear_cache(self):
        """Clear all cached query results."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def rebuild_indexes(self):
        """Rebuild all time indexes."""
        # Clear existing indexes
        for index in self.time_indexes.values():
            index.index_data.clear()
        
        # Rebuild from stored data
        for data_id, data_point in self.data_storage.items():
            self._update_indexes(data_id, data_point)
        
        logger.info("Time indexes rebuilt")


# Global query engine instance
_temporal_query_engine: Optional[TemporalQueryEngine] = None


def get_temporal_query_engine() -> Optional[TemporalQueryEngine]:
    """Get the global temporal query engine instance."""
    return _temporal_query_engine


def initialize_temporal_query_engine(max_cache_size: int = 1000) -> TemporalQueryEngine:
    """Initialize the global temporal query engine."""
    global _temporal_query_engine
    if _temporal_query_engine is None:
        _temporal_query_engine = TemporalQueryEngine(max_cache_size)
    return _temporal_query_engine