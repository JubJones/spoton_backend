"""
Date/Time Range Management System

Comprehensive temporal data management system providing:
- Available data range queries per environment with intelligent optimization
- Efficient temporal data indexing and retrieval strategies
- Time zone handling and conversion with DST support
- Data availability validation and error handling
- Temporal data caching and performance optimization
- Smart date range suggestions and user experience enhancements
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import pytz
from bisect import bisect_left, bisect_right
import calendar

from app.services.historical_data_service import (
    HistoricalDataService,
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)
from app.services.environment_configuration_service import (
    EnvironmentConfigurationService,
    EnvironmentConfiguration
)
from app.infrastructure.cache.tracking_cache import TrackingCache

logger = logging.getLogger(__name__)


class DataGranularity(Enum):
    """Data availability granularity levels."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"          # >95% coverage
    GOOD = "good"          # 80-95% coverage
    PARTIAL = "partial"    # 50-80% coverage
    SPARSE = "sparse"      # 20-50% coverage
    MINIMAL = "minimal"    # <20% coverage


class TimezoneSuggestion(Enum):
    """Smart timezone suggestion types."""
    DETECTED = "detected"      # Auto-detected from data
    USER_PREFERRED = "user_preferred"
    ENVIRONMENT_DEFAULT = "environment_default"
    SYSTEM_DEFAULT = "system_default"


@dataclass
class TimeRangeAvailability:
    """Detailed availability information for a time range."""
    start_time: datetime
    end_time: datetime
    total_duration_hours: float
    data_coverage_percentage: float
    data_quality: DataQuality
    gap_count: int
    largest_gap_hours: float
    data_points_estimate: int
    environments_available: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_duration_hours': self.total_duration_hours,
            'data_coverage_percentage': self.data_coverage_percentage,
            'data_quality': self.data_quality.value,
            'gap_count': self.gap_count,
            'largest_gap_hours': self.largest_gap_hours,
            'data_points_estimate': self.data_points_estimate,
            'environments_available': self.environments_available
        }


@dataclass
class DataGap:
    """Represents a gap in data availability."""
    start_time: datetime
    end_time: datetime
    duration_hours: float
    gap_type: str  # maintenance, outage, no_data, processing_error
    affected_cameras: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_hours': self.duration_hours,
            'gap_type': self.gap_type,
            'affected_cameras': self.affected_cameras,
            'reason': self.reason
        }


@dataclass
class SmartDateSuggestion:
    """Smart date range suggestion with metadata."""
    suggestion_id: str
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    expected_data_quality: DataQuality
    reason: str
    confidence_score: float  # 0.0 - 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'suggestion_id': self.suggestion_id,
            'name': self.name,
            'description': self.description,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'expected_data_quality': self.expected_data_quality.value,
            'reason': self.reason,
            'confidence_score': self.confidence_score
        }


@dataclass
class TimeRangeValidationResult:
    """Result of time range validation."""
    is_valid: bool
    start_time: datetime
    end_time: datetime
    adjusted_start: Optional[datetime] = None
    adjusted_end: Optional[datetime] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'adjusted_start': self.adjusted_start.isoformat() if self.adjusted_start else None,
            'adjusted_end': self.adjusted_end.isoformat() if self.adjusted_end else None,
            'warnings': self.warnings,
            'errors': self.errors,
            'suggestions': self.suggestions
        }


class DateTimeRangeManager:
    """Comprehensive service for date/time range management and temporal data operations."""
    
    def __init__(
        self,
        historical_data_service: HistoricalDataService,
        environment_config_service: EnvironmentConfigurationService,
        tracking_cache: TrackingCache
    ):
        self.historical_service = historical_data_service
        self.environment_service = environment_config_service
        self.cache = tracking_cache
        
        # Data availability index (environment_id -> sorted list of (timestamp, data_present))
        self.data_availability_index: Dict[str, List[Tuple[datetime, bool]]] = {}
        
        # Time zone management
        self.supported_timezones = [
            'UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific',
            'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Rome',
            'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Mumbai', 'Australia/Sydney'
        ]
        
        # Data quality thresholds
        self.quality_thresholds = {
            DataQuality.HIGH: 0.95,
            DataQuality.GOOD: 0.80,
            DataQuality.PARTIAL: 0.50,
            DataQuality.SPARSE: 0.20,
            DataQuality.MINIMAL: 0.0
        }
        
        # Caching configuration
        self.cache_ttl = {
            'availability_index': 3600,      # 1 hour
            'date_ranges': 1800,             # 30 minutes
            'validation_results': 300,        # 5 minutes
            'smart_suggestions': 600          # 10 minutes
        }
        
        # Performance metrics
        self.performance_stats = {
            'range_queries': 0,
            'validation_requests': 0,
            'suggestion_generations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_query_time_ms': 0.0
        }
        
        logger.info("DateTimeRangeManager initialized")
    
    # --- Data Availability Index Management ---
    
    async def build_availability_index(self, environment_id: str, force_rebuild: bool = False) -> bool:
        """Build or update data availability index for an environment."""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"availability_index_{environment_id}"
            if not force_rebuild:
                cached_index = await self.cache.get_json(cache_key)
                if cached_index:
                    self.data_availability_index[environment_id] = [
                        (datetime.fromisoformat(ts), present) 
                        for ts, present in cached_index
                    ]
                    self.performance_stats['cache_hits'] += 1
                    return True
            
            self.performance_stats['cache_misses'] += 1
            
            # Get environment date ranges
            date_ranges = await self.environment_service.get_available_date_ranges(environment_id)
            env_range = date_ranges.get(environment_id, {})
            
            if not env_range.get('has_data', False):
                self.data_availability_index[environment_id] = []
                return True
            
            # Parse date range
            earliest_date = datetime.fromisoformat(env_range['earliest_date'])
            latest_date = datetime.fromisoformat(env_range['latest_date'])
            
            # Build index by sampling data at regular intervals
            availability_points = []
            current_time = earliest_date
            sample_interval = timedelta(hours=1)  # Sample every hour
            
            while current_time <= latest_date:
                # Sample data availability at this time point
                sample_end = min(current_time + sample_interval, latest_date)
                time_range = TimeRange(start_time=current_time, end_time=sample_end)
                
                query_filter = HistoricalQueryFilter(
                    time_range=time_range,
                    environment_id=environment_id
                )
                
                try:
                    # Quick sample query
                    sample_data = await self.historical_service.query_historical_data(query_filter, limit=1)
                    has_data = len(sample_data) > 0
                except Exception as e:
                    logger.debug(f"Sample query failed for {current_time}: {e}")
                    has_data = False
                
                availability_points.append((current_time, has_data))
                current_time += sample_interval
            
            # Store in index
            self.data_availability_index[environment_id] = availability_points
            
            # Cache the index
            cache_data = [
                (ts.isoformat(), present) 
                for ts, present in availability_points
            ]
            await self.cache.set_json(cache_key, cache_data, ttl=self.cache_ttl['availability_index'])
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Built availability index for {environment_id} with {len(availability_points)} points in {processing_time:.1f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building availability index for {environment_id}: {e}")
            return False
    
    async def get_data_availability(
        self,
        environment_id: str,
        time_range: TimeRange,
        granularity: DataGranularity = DataGranularity.HOUR
    ) -> TimeRangeAvailability:
        """Get detailed data availability information for a time range."""
        try:
            start_time = time.time()
            
            # Ensure availability index exists
            if environment_id not in self.data_availability_index:
                await self.build_availability_index(environment_id)
            
            availability_points = self.data_availability_index.get(environment_id, [])
            
            if not availability_points:
                return TimeRangeAvailability(
                    start_time=time_range.start_time,
                    end_time=time_range.end_time,
                    total_duration_hours=time_range.duration_hours,
                    data_coverage_percentage=0.0,
                    data_quality=DataQuality.MINIMAL,
                    gap_count=1,
                    largest_gap_hours=time_range.duration_hours,
                    data_points_estimate=0,
                    environments_available=[environment_id] if environment_id else []
                )
            
            # Find relevant availability points
            start_idx = bisect_left(availability_points, (time_range.start_time, False))
            end_idx = bisect_right(availability_points, (time_range.end_time, True))
            
            relevant_points = availability_points[start_idx:end_idx]
            
            if not relevant_points:
                # No data points in range
                return TimeRangeAvailability(
                    start_time=time_range.start_time,
                    end_time=time_range.end_time,
                    total_duration_hours=time_range.duration_hours,
                    data_coverage_percentage=0.0,
                    data_quality=DataQuality.MINIMAL,
                    gap_count=1,
                    largest_gap_hours=time_range.duration_hours,
                    data_points_estimate=0,
                    environments_available=[]
                )
            
            # Calculate coverage and gaps
            coverage_stats = self._calculate_coverage_stats(relevant_points, time_range)
            
            # Determine data quality
            data_quality = self._determine_data_quality(coverage_stats['coverage_percentage'])
            
            # Estimate data points
            data_points_estimate = int(coverage_stats['coverage_percentage'] / 100.0 * time_range.duration_hours * 10)  # ~10 points per hour estimate
            
            result = TimeRangeAvailability(
                start_time=time_range.start_time,
                end_time=time_range.end_time,
                total_duration_hours=time_range.duration_hours,
                data_coverage_percentage=coverage_stats['coverage_percentage'],
                data_quality=data_quality,
                gap_count=coverage_stats['gap_count'],
                largest_gap_hours=coverage_stats['largest_gap_hours'],
                data_points_estimate=data_points_estimate,
                environments_available=[environment_id] if coverage_stats['coverage_percentage'] > 0 else []
            )
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['range_queries'] += 1
            self._update_avg_processing_time(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting data availability: {e}")
            # Return empty availability on error
            return TimeRangeAvailability(
                start_time=time_range.start_time,
                end_time=time_range.end_time,
                total_duration_hours=time_range.duration_hours,
                data_coverage_percentage=0.0,
                data_quality=DataQuality.MINIMAL,
                gap_count=1,
                largest_gap_hours=time_range.duration_hours,
                data_points_estimate=0,
                environments_available=[]
            )
    
    def _calculate_coverage_stats(
        self,
        availability_points: List[Tuple[datetime, bool]],
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate coverage statistics from availability points."""
        try:
            if not availability_points:
                return {
                    'coverage_percentage': 0.0,
                    'gap_count': 1,
                    'largest_gap_hours': time_range.duration_hours
                }
            
            # Calculate time with data vs without data
            total_time_with_data = 0.0
            gap_count = 0
            current_gap_start = None
            gaps = []
            
            prev_time = time_range.start_time
            
            for timestamp, has_data in availability_points:
                time_diff = (timestamp - prev_time).total_seconds() / 3600.0  # Convert to hours
                
                if has_data:
                    total_time_with_data += time_diff
                    
                    # End current gap if one was active
                    if current_gap_start is not None:
                        gap_duration = (timestamp - current_gap_start).total_seconds() / 3600.0
                        gaps.append(gap_duration)
                        current_gap_start = None
                else:
                    # Start new gap if not already in one
                    if current_gap_start is None:
                        current_gap_start = prev_time
                        gap_count += 1
                
                prev_time = timestamp
            
            # Handle final period
            if prev_time < time_range.end_time:
                time_diff = (time_range.end_time - prev_time).total_seconds() / 3600.0
                
                if current_gap_start is not None:
                    gap_duration = (time_range.end_time - current_gap_start).total_seconds() / 3600.0
                    gaps.append(gap_duration)
                else:
                    total_time_with_data += time_diff
            
            # Calculate final statistics
            coverage_percentage = (total_time_with_data / time_range.duration_hours) * 100.0
            largest_gap_hours = max(gaps) if gaps else 0.0
            
            return {
                'coverage_percentage': min(100.0, max(0.0, coverage_percentage)),
                'gap_count': gap_count,
                'largest_gap_hours': largest_gap_hours
            }
            
        except Exception as e:
            logger.error(f"Error calculating coverage stats: {e}")
            return {
                'coverage_percentage': 0.0,
                'gap_count': 1,
                'largest_gap_hours': time_range.duration_hours
            }
    
    def _determine_data_quality(self, coverage_percentage: float) -> DataQuality:
        """Determine data quality based on coverage percentage."""
        if coverage_percentage >= self.quality_thresholds[DataQuality.HIGH] * 100:
            return DataQuality.HIGH
        elif coverage_percentage >= self.quality_thresholds[DataQuality.GOOD] * 100:
            return DataQuality.GOOD
        elif coverage_percentage >= self.quality_thresholds[DataQuality.PARTIAL] * 100:
            return DataQuality.PARTIAL
        elif coverage_percentage >= self.quality_thresholds[DataQuality.SPARSE] * 100:
            return DataQuality.SPARSE
        else:
            return DataQuality.MINIMAL
    
    # --- Time Range Validation ---
    
    async def validate_time_range(
        self,
        time_range: TimeRange,
        environment_id: Optional[str] = None,
        user_timezone: Optional[str] = None
    ) -> TimeRangeValidationResult:
        """Validate and potentially adjust a time range request."""
        try:
            start_time = time.time()
            
            errors = []
            warnings = []
            suggestions = []
            adjusted_start = None
            adjusted_end = None
            
            # Basic validation
            if time_range.start_time >= time_range.end_time:
                errors.append("Start time must be before end time")
                return TimeRangeValidationResult(
                    is_valid=False,
                    start_time=time_range.start_time,
                    end_time=time_range.end_time,
                    errors=errors
                )
            
            # Duration validation
            duration_hours = time_range.duration_hours
            
            if duration_hours > 24 * 30:  # More than 30 days
                warnings.append("Large time range may result in slow query performance")
                suggestions.append("Consider breaking into smaller ranges for better performance")
            
            if duration_hours < 0.1:  # Less than 6 minutes
                warnings.append("Very short time range may have limited data")
                suggestions.append("Consider extending the time range for more meaningful analysis")
            
            # Future date validation
            now = datetime.utcnow()
            if time_range.start_time > now:
                errors.append("Start time cannot be in the future")
            
            if time_range.end_time > now:
                warnings.append("End time is in the future, adjusting to current time")
                adjusted_end = now
            
            # Environment-specific validation
            if environment_id:
                env_availability = await self.get_data_availability(environment_id, time_range)
                
                if env_availability.data_coverage_percentage < 10:
                    warnings.append(f"Low data availability ({env_availability.data_coverage_percentage:.1f}%) in selected time range")
                    
                    # Suggest better time ranges
                    better_ranges = await self.suggest_date_ranges(environment_id, duration_hours)
                    if better_ranges:
                        suggestions.append(f"Consider alternative time ranges with better data quality")
                
                # Check if we can adjust the range for better data coverage
                if env_availability.data_coverage_percentage < 50:
                    adjusted_range = await self._suggest_adjusted_range(environment_id, time_range)
                    if adjusted_range:
                        adjusted_start = adjusted_range.start_time
                        adjusted_end = adjusted_range.end_time
                        suggestions.append("Adjusted time range for better data coverage")
            
            # Time zone validation
            if user_timezone and user_timezone not in self.supported_timezones:
                warnings.append(f"Unsupported timezone '{user_timezone}', using UTC")
            
            # Determine if range is valid
            is_valid = len(errors) == 0
            
            result = TimeRangeValidationResult(
                is_valid=is_valid,
                start_time=time_range.start_time,
                end_time=time_range.end_time,
                adjusted_start=adjusted_start,
                adjusted_end=adjusted_end,
                warnings=warnings,
                errors=errors,
                suggestions=suggestions
            )
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['validation_requests'] += 1
            self._update_avg_processing_time(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating time range: {e}")
            return TimeRangeValidationResult(
                is_valid=False,
                start_time=time_range.start_time,
                end_time=time_range.end_time,
                errors=[f"Validation error: {str(e)}"]
            )
    
    async def _suggest_adjusted_range(
        self,
        environment_id: str,
        original_range: TimeRange
    ) -> Optional[TimeRange]:
        """Suggest an adjusted time range with better data coverage."""
        try:
            # Look for nearby time periods with better data coverage
            duration_hours = original_range.duration_hours
            
            # Try periods before and after the original range
            test_ranges = []
            
            # 1 hour before
            test_ranges.append(TimeRange(
                start_time=original_range.start_time - timedelta(hours=1),
                end_time=original_range.end_time - timedelta(hours=1)
            ))
            
            # 1 hour after
            test_ranges.append(TimeRange(
                start_time=original_range.start_time + timedelta(hours=1),
                end_time=original_range.end_time + timedelta(hours=1)
            ))
            
            # Previous day same time
            test_ranges.append(TimeRange(
                start_time=original_range.start_time - timedelta(days=1),
                end_time=original_range.end_time - timedelta(days=1)
            ))
            
            best_range = None
            best_coverage = 0.0
            
            for test_range in test_ranges:
                if test_range.end_time > datetime.utcnow():
                    continue
                
                availability = await self.get_data_availability(environment_id, test_range)
                if availability.data_coverage_percentage > best_coverage:
                    best_coverage = availability.data_coverage_percentage
                    best_range = test_range
            
            # Only suggest if significantly better
            if best_coverage > 70.0:  # At least 70% coverage
                return best_range
            
            return None
            
        except Exception as e:
            logger.error(f"Error suggesting adjusted range: {e}")
            return None
    
    # --- Smart Date Suggestions ---
    
    async def suggest_date_ranges(
        self,
        environment_id: str,
        desired_duration_hours: float = 24.0,
        limit: int = 5
    ) -> List[SmartDateSuggestion]:
        """Generate smart date range suggestions based on data quality and patterns."""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"smart_suggestions_{environment_id}_{desired_duration_hours}_{limit}"
            cached_suggestions = await self.cache.get_json(cache_key)
            if cached_suggestions:
                self.performance_stats['cache_hits'] += 1
                return [SmartDateSuggestion(**suggestion) for suggestion in cached_suggestions]
            
            self.performance_stats['cache_misses'] += 1
            
            suggestions = []
            
            # Get environment date ranges
            date_ranges = await self.environment_service.get_available_date_ranges(environment_id)
            env_range = date_ranges.get(environment_id, {})
            
            if not env_range.get('has_data', False):
                return suggestions
            
            latest_date = datetime.fromisoformat(env_range['latest_date'])
            earliest_date = datetime.fromisoformat(env_range['earliest_date'])
            
            # Suggestion 1: Most recent data with good quality
            recent_end = min(latest_date, datetime.utcnow())
            recent_start = recent_end - timedelta(hours=desired_duration_hours)
            
            if recent_start >= earliest_date:
                recent_range = TimeRange(start_time=recent_start, end_time=recent_end)
                recent_availability = await self.get_data_availability(environment_id, recent_range)
                
                suggestions.append(SmartDateSuggestion(
                    suggestion_id="recent_data",
                    name="Most Recent Data",
                    description=f"Latest {desired_duration_hours} hours of data",
                    start_time=recent_start,
                    end_time=recent_end,
                    expected_data_quality=recent_availability.data_quality,
                    reason="Latest available data with real-time insights",
                    confidence_score=0.9
                ))
            
            # Suggestion 2: Peak activity period (simplified - would use actual analysis)
            peak_suggestion = await self._suggest_peak_activity_period(
                environment_id, earliest_date, latest_date, desired_duration_hours
            )
            if peak_suggestion:
                suggestions.append(peak_suggestion)
            
            # Suggestion 3: High quality data period
            high_quality_suggestion = await self._suggest_high_quality_period(
                environment_id, earliest_date, latest_date, desired_duration_hours
            )
            if high_quality_suggestion:
                suggestions.append(high_quality_suggestion)
            
            # Suggestion 4: Previous week same time (if applicable)
            if desired_duration_hours <= 24:
                week_ago_end = recent_end - timedelta(days=7)
                week_ago_start = week_ago_end - timedelta(hours=desired_duration_hours)
                
                if week_ago_start >= earliest_date:
                    week_ago_range = TimeRange(start_time=week_ago_start, end_time=week_ago_end)
                    week_ago_availability = await self.get_data_availability(environment_id, week_ago_range)
                    
                    suggestions.append(SmartDateSuggestion(
                        suggestion_id="previous_week",
                        name="Previous Week Same Time",
                        description="Same time period from one week ago",
                        start_time=week_ago_start,
                        end_time=week_ago_end,
                        expected_data_quality=week_ago_availability.data_quality,
                        reason="Compare with similar time period for trend analysis",
                        confidence_score=0.7
                    ))
            
            # Suggestion 5: Full day with best coverage (for shorter requests)
            if desired_duration_hours < 12:
                full_day_suggestion = await self._suggest_best_full_day(
                    environment_id, earliest_date, latest_date
                )
                if full_day_suggestion:
                    suggestions.append(full_day_suggestion)
            
            # Sort by confidence score and limit results
            suggestions.sort(key=lambda s: s.confidence_score, reverse=True)
            suggestions = suggestions[:limit]
            
            # Cache results
            suggestions_data = [suggestion.to_dict() for suggestion in suggestions]
            await self.cache.set_json(cache_key, suggestions_data, ttl=self.cache_ttl['smart_suggestions'])
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['suggestion_generations'] += 1
            self._update_avg_processing_time(processing_time)
            
            logger.debug(f"Generated {len(suggestions)} date suggestions for {environment_id} in {processing_time:.1f}ms")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting date ranges: {e}")
            return []
    
    async def _suggest_peak_activity_period(
        self,
        environment_id: str,
        earliest_date: datetime,
        latest_date: datetime,
        duration_hours: float
    ) -> Optional[SmartDateSuggestion]:
        """Suggest time period with peak activity (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In production, this would analyze historical data to find actual peak periods
            
            # Assume peak activity is typically during business hours (9 AM - 5 PM)
            # Find a recent business day
            test_date = latest_date.replace(hour=12, minute=0, second=0, microsecond=0)  # Noon
            
            # Go back to find a weekday
            while test_date.weekday() > 4:  # Not Monday-Friday
                test_date -= timedelta(days=1)
            
            if test_date < earliest_date:
                return None
            
            peak_start = test_date.replace(hour=9)  # 9 AM
            peak_end = peak_start + timedelta(hours=duration_hours)
            
            if peak_end > latest_date:
                return None
            
            peak_range = TimeRange(start_time=peak_start, end_time=peak_end)
            peak_availability = await self.get_data_availability(environment_id, peak_range)
            
            return SmartDateSuggestion(
                suggestion_id="peak_activity",
                name="Peak Activity Period",
                description="Time period with typically high activity",
                start_time=peak_start,
                end_time=peak_end,
                expected_data_quality=peak_availability.data_quality,
                reason="Business hours typically show more activity and interactions",
                confidence_score=0.8
            )
            
        except Exception as e:
            logger.error(f"Error suggesting peak activity period: {e}")
            return None
    
    async def _suggest_high_quality_period(
        self,
        environment_id: str,
        earliest_date: datetime,
        latest_date: datetime,
        duration_hours: float
    ) -> Optional[SmartDateSuggestion]:
        """Suggest time period with high data quality."""
        try:
            # Sample several time periods to find one with high quality
            best_range = None
            best_quality = DataQuality.MINIMAL
            best_coverage = 0.0
            
            # Test 10 different time periods
            total_days = (latest_date - earliest_date).days
            if total_days < 1:
                return None
            
            test_count = min(10, total_days)
            
            for i in range(test_count):
                # Distribute test points across available time range
                days_offset = i * (total_days / test_count)
                test_start = earliest_date + timedelta(days=days_offset)
                test_end = test_start + timedelta(hours=duration_hours)
                
                if test_end > latest_date:
                    continue
                
                test_range = TimeRange(start_time=test_start, end_time=test_end)
                availability = await self.get_data_availability(environment_id, test_range)
                
                # Prefer higher coverage and better quality
                if (availability.data_quality.value > best_quality.value or
                    (availability.data_quality == best_quality and availability.data_coverage_percentage > best_coverage)):
                    best_range = test_range
                    best_quality = availability.data_quality
                    best_coverage = availability.data_coverage_percentage
            
            if best_range and best_coverage > 80.0:  # Only suggest if good coverage
                return SmartDateSuggestion(
                    suggestion_id="high_quality",
                    name="High Quality Data",
                    description="Time period with excellent data coverage",
                    start_time=best_range.start_time,
                    end_time=best_range.end_time,
                    expected_data_quality=best_quality,
                    reason="Selected for optimal data quality and coverage",
                    confidence_score=0.85
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error suggesting high quality period: {e}")
            return None
    
    async def _suggest_best_full_day(
        self,
        environment_id: str,
        earliest_date: datetime,
        latest_date: datetime
    ) -> Optional[SmartDateSuggestion]:
        """Suggest the best full day period."""
        try:
            # Find a recent full day with good data coverage
            test_date = latest_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            while test_date >= earliest_date:
                day_start = test_date
                day_end = day_start + timedelta(days=1)
                
                if day_end > latest_date:
                    test_date -= timedelta(days=1)
                    continue
                
                day_range = TimeRange(start_time=day_start, end_time=day_end)
                availability = await self.get_data_availability(environment_id, day_range)
                
                if availability.data_coverage_percentage > 60.0:  # Good coverage
                    return SmartDateSuggestion(
                        suggestion_id="full_day",
                        name="Complete Day Analysis",
                        description="Full 24-hour period with good data coverage",
                        start_time=day_start,
                        end_time=day_end,
                        expected_data_quality=availability.data_quality,
                        reason="Complete daily cycle for comprehensive analysis",
                        confidence_score=0.75
                    )
                
                test_date -= timedelta(days=1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error suggesting best full day: {e}")
            return None
    
    # --- Time Zone Handling ---
    
    def convert_time_range(
        self,
        time_range: TimeRange,
        from_timezone: str,
        to_timezone: str
    ) -> TimeRange:
        """Convert time range between time zones."""
        try:
            from_tz = pytz.timezone(from_timezone)
            to_tz = pytz.timezone(to_timezone)
            
            # Localize and convert start time
            start_localized = from_tz.localize(time_range.start_time.replace(tzinfo=None))
            start_converted = start_localized.astimezone(to_tz)
            
            # Localize and convert end time  
            end_localized = from_tz.localize(time_range.end_time.replace(tzinfo=None))
            end_converted = end_localized.astimezone(to_tz)
            
            return TimeRange(
                start_time=start_converted.replace(tzinfo=None),
                end_time=end_converted.replace(tzinfo=None)
            )
            
        except Exception as e:
            logger.error(f"Error converting time range: {e}")
            return time_range  # Return original on error
    
    def suggest_timezone(
        self,
        environment_id: Optional[str] = None,
        user_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Suggest appropriate timezone for user."""
        try:
            suggestions = []
            
            # User preferred timezone
            if user_preference and user_preference in self.supported_timezones:
                suggestions.append({
                    'timezone': user_preference,
                    'type': TimezoneSuggestion.USER_PREFERRED.value,
                    'confidence': 1.0,
                    'reason': 'User preferred timezone'
                })
            
            # Environment default (if available)
            # This would be configured per environment in production
            environment_timezones = {
                'campus': 'US/Eastern',
                'factory': 'US/Central'
            }
            
            if environment_id and environment_id in environment_timezones:
                env_tz = environment_timezones[environment_id]
                suggestions.append({
                    'timezone': env_tz,
                    'type': TimezoneSuggestion.ENVIRONMENT_DEFAULT.value,
                    'confidence': 0.8,
                    'reason': f'Default timezone for {environment_id} environment'
                })
            
            # System default
            suggestions.append({
                'timezone': 'UTC',
                'type': TimezoneSuggestion.SYSTEM_DEFAULT.value,
                'confidence': 0.6,
                'reason': 'Universal Coordinated Time (recommended for data analysis)'
            })
            
            return {
                'suggested_timezone': suggestions[0]['timezone'] if suggestions else 'UTC',
                'all_suggestions': suggestions,
                'supported_timezones': self.supported_timezones
            }
            
        except Exception as e:
            logger.error(f"Error suggesting timezone: {e}")
            return {
                'suggested_timezone': 'UTC',
                'all_suggestions': [],
                'supported_timezones': self.supported_timezones
            }
    
    # --- Utility Methods ---
    
    def _update_avg_processing_time(self, processing_time_ms: float):
        """Update average processing time metric."""
        current_avg = self.performance_stats['avg_query_time_ms']
        total_queries = (self.performance_stats['range_queries'] + 
                        self.performance_stats['validation_requests'] + 
                        self.performance_stats['suggestion_generations'])
        
        if total_queries > 0:
            self.performance_stats['avg_query_time_ms'] = (
                (current_avg * (total_queries - 1) + processing_time_ms) / total_queries
            )
    
    async def clear_caches(self, environment_id: Optional[str] = None):
        """Clear cached data for environment or all environments."""
        try:
            if environment_id:
                # Clear specific environment caches
                cache_keys = [
                    f"availability_index_{environment_id}",
                    f"smart_suggestions_{environment_id}_*",
                    f"env_metrics_{environment_id}"
                ]
                
                for key_pattern in cache_keys:
                    if '*' in key_pattern:
                        # This is a simplified implementation
                        # In production, you'd use pattern matching to delete multiple keys
                        pass
                    else:
                        await self.cache.delete(key_pattern)
                
                # Clear in-memory index
                if environment_id in self.data_availability_index:
                    del self.data_availability_index[environment_id]
                
                logger.info(f"Cleared caches for environment {environment_id}")
            else:
                # Clear all caches
                self.data_availability_index.clear()
                logger.info("Cleared all date/time range caches")
                
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'DateTimeRangeManager',
            'environments_indexed': len(self.data_availability_index),
            'supported_timezones_count': len(self.supported_timezones),
            'cache_hit_rate': (
                self.performance_stats['cache_hits'] / 
                max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) * 100
            ),
            'performance_stats': self.performance_stats.copy(),
            'quality_thresholds': {q.value: threshold for q, threshold in self.quality_thresholds.items()},
            'cache_ttl_settings': self.cache_ttl
        }


# Global service instance
_datetime_range_manager: Optional[DateTimeRangeManager] = None


def get_datetime_range_manager() -> Optional[DateTimeRangeManager]:
    """Get the global date/time range manager instance."""
    return _datetime_range_manager


def initialize_datetime_range_manager(
    historical_data_service: HistoricalDataService,
    environment_config_service: EnvironmentConfigurationService,
    tracking_cache: TrackingCache
) -> DateTimeRangeManager:
    """Initialize the global date/time range manager."""
    global _datetime_range_manager
    if _datetime_range_manager is None:
        _datetime_range_manager = DateTimeRangeManager(
            historical_data_service,
            environment_config_service,
            tracking_cache
        )
    return _datetime_range_manager