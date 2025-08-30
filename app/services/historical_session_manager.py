"""
Historical Session Management System

Comprehensive historical session management system providing:
- Time-based session creation and management with temporal context
- Historical data preprocessing for efficient access and analysis
- Temporal data caching and optimization strategies
- Session state persistence across user interactions
- Multi-user concurrent historical analysis support
- Advanced session analytics and performance monitoring
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

from app.services.historical_data_service import (
    HistoricalDataService,
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)
from app.services.datetime_range_manager import (
    DateTimeRangeManager,
    TimeRangeValidationResult,
    DataQuality
)
from app.services.environment_configuration_service import (
    EnvironmentConfigurationService
)
from app.services.temporal_query_engine import (
    TemporalQueryEngine,
    TimeGranularity
)
from app.infrastructure.cache.tracking_cache import TrackingCache

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Historical session status states."""
    INITIALIZING = "initializing"
    PREPROCESSING = "preprocessing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"


class SessionType(Enum):
    """Types of historical sessions."""
    ANALYSIS = "analysis"        # General historical analysis
    PLAYBACK = "playback"        # Video playback with data
    COMPARISON = "comparison"    # Multi-time period comparison
    EXPORT = "export"           # Data export session
    INVESTIGATION = "investigation"  # Detailed investigation session


class PreprocessingStage(Enum):
    """Stages of data preprocessing."""
    DATA_VALIDATION = "data_validation"
    TEMPORAL_INDEXING = "temporal_indexing"
    FEATURE_EXTRACTION = "feature_extraction"
    TRAJECTORY_BUILDING = "trajectory_building"
    ANALYTICS_PREPARATION = "analytics_preparation"
    CACHING_OPTIMIZATION = "caching_optimization"
    COMPLETION = "completion"


@dataclass
class SessionConfiguration:
    """Configuration parameters for a historical session."""
    session_id: str
    session_type: SessionType
    environment_id: str
    time_range: TimeRange
    user_id: str
    
    # Processing options
    enable_preprocessing: bool = True
    cache_results: bool = True
    preload_trajectories: bool = False
    enable_analytics: bool = True
    
    # Performance options
    batch_size: int = 1000
    max_concurrent_queries: int = 3
    memory_limit_mb: int = 512
    
    # Session preferences
    timezone: str = "UTC"
    data_quality_threshold: DataQuality = DataQuality.GOOD
    
    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'session_type': self.session_type.value,
            'environment_id': self.environment_id,
            'time_range': {
                'start': self.time_range.start_time.isoformat(),
                'end': self.time_range.end_time.isoformat()
            },
            'user_id': self.user_id,
            'enable_preprocessing': self.enable_preprocessing,
            'cache_results': self.cache_results,
            'preload_trajectories': self.preload_trajectories,
            'enable_analytics': self.enable_analytics,
            'batch_size': self.batch_size,
            'max_concurrent_queries': self.max_concurrent_queries,
            'memory_limit_mb': self.memory_limit_mb,
            'timezone': self.timezone,
            'data_quality_threshold': self.data_quality_threshold.value,
            'custom_parameters': self.custom_parameters
        }


@dataclass
class PreprocessingProgress:
    """Progress tracking for data preprocessing."""
    current_stage: PreprocessingStage
    stage_progress: float  # 0.0 - 1.0
    overall_progress: float  # 0.0 - 1.0
    
    # Stage details
    stages_completed: List[PreprocessingStage] = field(default_factory=list)
    current_stage_message: str = ""
    estimated_completion_time: Optional[datetime] = None
    
    # Data processing stats
    data_points_processed: int = 0
    total_data_points: int = 0
    trajectories_built: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'current_stage': self.current_stage.value,
            'stage_progress': self.stage_progress,
            'overall_progress': self.overall_progress,
            'stages_completed': [stage.value for stage in self.stages_completed],
            'current_stage_message': self.current_stage_message,
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'data_points_processed': self.data_points_processed,
            'total_data_points': self.total_data_points,
            'trajectories_built': self.trajectories_built
        }


@dataclass
class SessionState:
    """Current state of a historical session."""
    session_id: str
    status: SessionStatus
    configuration: SessionConfiguration
    
    # Timestamps
    created_at: datetime
    last_activity: datetime
    preprocessing_started_at: Optional[datetime] = None
    ready_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Processing state
    preprocessing_progress: Optional[PreprocessingProgress] = None
    error_message: Optional[str] = None
    
    # Data state
    data_points_available: int = 0
    trajectories_available: int = 0
    cached_data_size_mb: float = 0.0
    
    # Performance metrics
    total_query_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Session data
    session_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'status': self.status.value,
            'configuration': self.configuration.to_dict(),
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'preprocessing_started_at': self.preprocessing_started_at.isoformat() if self.preprocessing_started_at else None,
            'ready_at': self.ready_at.isoformat() if self.ready_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'preprocessing_progress': self.preprocessing_progress.to_dict() if self.preprocessing_progress else None,
            'error_message': self.error_message,
            'data_points_available': self.data_points_available,
            'trajectories_available': self.trajectories_available,
            'cached_data_size_mb': self.cached_data_size_mb,
            'total_query_time_ms': self.total_query_time_ms,
            'cache_hit_rate': self.cache_hit_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'session_data': self.session_data
        }


@dataclass
class SessionQuery:
    """Represents a query executed within a session."""
    query_id: str
    session_id: str
    query_type: str
    parameters: Dict[str, Any]
    executed_at: datetime
    execution_time_ms: float
    result_size: int
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query_id': self.query_id,
            'session_id': self.session_id,
            'query_type': self.query_type,
            'parameters': self.parameters,
            'executed_at': self.executed_at.isoformat(),
            'execution_time_ms': self.execution_time_ms,
            'result_size': self.result_size,
            'cache_hit': self.cache_hit
        }


class HistoricalSessionManager:
    """Comprehensive service for managing historical analysis sessions."""
    
    def __init__(
        self,
        historical_data_service: HistoricalDataService,
        datetime_range_manager: DateTimeRangeManager,
        environment_config_service: EnvironmentConfigurationService,
        temporal_query_engine: TemporalQueryEngine,
        tracking_cache: TrackingCache
    ):
        self.historical_service = historical_data_service
        self.range_manager = datetime_range_manager
        self.environment_service = environment_config_service
        self.query_engine = temporal_query_engine
        self.cache = tracking_cache
        
        # Session management
        self.active_sessions: Dict[str, SessionState] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        self.preprocessing_tasks: Dict[str, asyncio.Task] = {}
        
        # Session configuration
        self.default_session_ttl = timedelta(hours=4)  # 4 hours default TTL
        self.max_concurrent_sessions_per_user = 5
        self.max_total_active_sessions = 50
        
        # Performance configuration
        self.batch_processing_size = 1000
        self.max_memory_per_session_mb = 512
        self.cache_cleanup_interval = 300  # 5 minutes
        
        # Background processing
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance metrics
        self.performance_stats = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'preprocessing_tasks': 0,
            'total_query_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'avg_session_duration_minutes': 0.0,
            'memory_usage_mb': 0.0
        }
        
        logger.info("HistoricalSessionManager initialized")
    
    async def start_manager(self):
        """Start the historical session manager."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
        
        logger.info("HistoricalSessionManager started")
    
    async def stop_manager(self):
        """Stop the historical session manager."""
        self._running = False
        
        # Cancel all preprocessing tasks
        for task in list(self.preprocessing_tasks.values()):
            if not task.done():
                task.cancel()
        
        # Wait for cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all sessions
        await self._cleanup_all_sessions()
        
        logger.info("HistoricalSessionManager stopped")
    
    # --- Session Creation and Management ---
    
    async def create_session(
        self,
        environment_id: str,
        time_range: TimeRange,
        user_id: str,
        session_type: SessionType = SessionType.ANALYSIS,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> SessionState:
        """Create a new historical analysis session."""
        try:
            start_time = time.time()
            
            # Check user session limits
            user_session_count = len(self.user_sessions.get(user_id, set()))
            if user_session_count >= self.max_concurrent_sessions_per_user:
                raise ValueError(f"User {user_id} has reached maximum concurrent sessions limit ({self.max_concurrent_sessions_per_user})")
            
            # Check total session limits
            if len(self.active_sessions) >= self.max_total_active_sessions:
                raise ValueError(f"System has reached maximum concurrent sessions limit ({self.max_total_active_sessions})")
            
            # Validate time range
            validation_result = await self.range_manager.validate_time_range(time_range, environment_id)
            if not validation_result.is_valid:
                raise ValueError(f"Invalid time range: {', '.join(validation_result.errors)}")
            
            # Use adjusted time range if suggested
            final_time_range = TimeRange(
                start_time=validation_result.adjusted_start or time_range.start_time,
                end_time=validation_result.adjusted_end or time_range.end_time
            )
            
            # Generate session ID
            session_data = f"{user_id}_{environment_id}_{datetime.utcnow().isoformat()}_{uuid.uuid4()}"
            session_id = hashlib.sha256(session_data.encode()).hexdigest()[:16]
            
            # Create session configuration
            config = SessionConfiguration(
                session_id=session_id,
                session_type=session_type,
                environment_id=environment_id,
                time_range=final_time_range,
                user_id=user_id
            )
            
            # Apply configuration overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Create session state
            now = datetime.utcnow()
            session_state = SessionState(
                session_id=session_id,
                status=SessionStatus.INITIALIZING,
                configuration=config,
                created_at=now,
                last_activity=now,
                expires_at=now + self.default_session_ttl
            )
            
            # Store session
            self.active_sessions[session_id] = session_state
            self.user_sessions[user_id].add(session_id)
            
            # Cache session state
            await self._cache_session_state(session_state)
            
            # Start preprocessing if enabled
            if config.enable_preprocessing:
                session_state.status = SessionStatus.PREPROCESSING
                session_state.preprocessing_started_at = now
                
                # Start preprocessing task
                preprocessing_task = asyncio.create_task(
                    self._preprocess_session_data(session_state)
                )
                self.preprocessing_tasks[session_id] = preprocessing_task
            else:
                session_state.status = SessionStatus.READY
                session_state.ready_at = now
            
            # Update metrics
            self.performance_stats['sessions_created'] += 1
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Created historical session {session_id} for user {user_id} in {processing_time:.1f}ms")
            
            return session_state
            
        except Exception as e:
            logger.error(f"Error creating historical session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID with activity tracking."""
        try:
            # Check memory first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_activity = datetime.utcnow()
                return session
            
            # Check cache
            session_data = await self.cache.get_json(f"historical_session_{session_id}")
            if session_data:
                session = self._deserialize_session_state(session_data)
                self.active_sessions[session_id] = session
                self.user_sessions[session.configuration.user_id].add(session_id)
                session.last_activity = datetime.utcnow()
                return session
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    async def update_session_state(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session state with new data."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Apply updates to session state
            for key, value in updates.items():
                if key == 'session_data':
                    session.session_data.update(value)
                elif hasattr(session, key):
                    setattr(session, key, value)
            
            session.last_activity = datetime.utcnow()
            
            # Update cache
            await self._cache_session_state(session)
            
            logger.debug(f"Updated session state for {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating session state: {e}")
            return False
    
    async def end_session(self, session_id: str) -> bool:
        """End a session and cleanup resources."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            # Cancel preprocessing task if running
            if session_id in self.preprocessing_tasks:
                task = self.preprocessing_tasks[session_id]
                if not task.done():
                    task.cancel()
                del self.preprocessing_tasks[session_id]
            
            # Mark as completed
            session.status = SessionStatus.COMPLETED
            session.last_activity = datetime.utcnow()
            
            # Update metrics
            session_duration = (session.last_activity - session.created_at).total_seconds() / 60.0
            self._update_session_duration_metric(session_duration)
            self.performance_stats['sessions_completed'] += 1
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            self.user_sessions[session.configuration.user_id].discard(session_id)
            
            # Clean up session-specific cache
            await self._cleanup_session_cache(session_id)
            
            logger.info(f"Ended session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False
    
    # --- Data Preprocessing ---
    
    async def _preprocess_session_data(self, session: SessionState):
        """Preprocess historical data for efficient session access."""
        try:
            config = session.configuration
            
            # Initialize preprocessing progress
            progress = PreprocessingProgress(
                current_stage=PreprocessingStage.DATA_VALIDATION,
                stage_progress=0.0,
                overall_progress=0.0
            )
            session.preprocessing_progress = progress
            
            # Stage 1: Data Validation
            await self._preprocessing_stage_data_validation(session)
            
            # Stage 2: Temporal Indexing
            await self._preprocessing_stage_temporal_indexing(session)
            
            # Stage 3: Feature Extraction (if needed)
            if config.enable_analytics:
                await self._preprocessing_stage_feature_extraction(session)
            
            # Stage 4: Trajectory Building (if requested)
            if config.preload_trajectories:
                await self._preprocessing_stage_trajectory_building(session)
            
            # Stage 5: Analytics Preparation
            if config.enable_analytics:
                await self._preprocessing_stage_analytics_preparation(session)
            
            # Stage 6: Caching Optimization
            if config.cache_results:
                await self._preprocessing_stage_caching_optimization(session)
            
            # Stage 7: Completion
            await self._preprocessing_stage_completion(session)
            
            # Mark session as ready
            session.status = SessionStatus.READY
            session.ready_at = datetime.utcnow()
            
            # Update cache
            await self._cache_session_state(session)
            
            self.performance_stats['preprocessing_tasks'] += 1
            logger.info(f"Completed preprocessing for session {session.session_id}")
            
        except asyncio.CancelledError:
            session.status = SessionStatus.ERROR
            session.error_message = "Preprocessing was cancelled"
            logger.info(f"Preprocessing cancelled for session {session.session_id}")
            raise
        except Exception as e:
            session.status = SessionStatus.ERROR
            session.error_message = f"Preprocessing failed: {str(e)}"
            logger.error(f"Error preprocessing session {session.session_id}: {e}")
            await self._cache_session_state(session)
    
    async def _preprocessing_stage_data_validation(self, session: SessionState):
        """Stage 1: Validate data availability and quality."""
        try:
            progress = session.preprocessing_progress
            progress.current_stage = PreprocessingStage.DATA_VALIDATION
            progress.current_stage_message = "Validating data availability and quality"
            
            config = session.configuration
            
            # Check data availability
            availability = await self.range_manager.get_data_availability(
                config.environment_id, config.time_range
            )
            
            # Validate data quality meets threshold
            if availability.data_quality.value < config.data_quality_threshold.value:
                logger.warning(f"Data quality ({availability.data_quality.value}) below threshold ({config.data_quality_threshold.value}) for session {session.session_id}")
            
            # Estimate total data points
            progress.total_data_points = availability.data_points_estimate
            session.data_points_available = availability.data_points_estimate
            
            # Mark stage complete
            progress.stages_completed.append(PreprocessingStage.DATA_VALIDATION)
            progress.stage_progress = 1.0
            progress.overall_progress = 1.0 / 7.0  # 7 total stages
            
            await self._cache_session_state(session)
            
        except Exception as e:
            logger.error(f"Error in data validation stage: {e}")
            raise
    
    async def _preprocessing_stage_temporal_indexing(self, session: SessionState):
        """Stage 2: Build temporal index for efficient queries."""
        try:
            progress = session.preprocessing_progress
            progress.current_stage = PreprocessingStage.TEMPORAL_INDEXING
            progress.current_stage_message = "Building temporal index for efficient queries"
            progress.stage_progress = 0.0
            
            config = session.configuration
            
            # Query data in batches to build temporal index
            current_time = config.time_range.start_time
            batch_duration = timedelta(hours=1)  # Process 1 hour at a time
            
            total_batches = int(config.time_range.duration_hours)
            processed_batches = 0
            
            temporal_index = []
            
            while current_time < config.time_range.end_time:
                batch_end = min(current_time + batch_duration, config.time_range.end_time)
                batch_range = TimeRange(start_time=current_time, end_time=batch_end)
                
                query_filter = HistoricalQueryFilter(
                    time_range=batch_range,
                    environment_id=config.environment_id
                )
                
                # Sample data for indexing
                batch_data = await self.historical_service.query_historical_data(
                    query_filter, limit=100
                )
                
                # Add to temporal index
                if batch_data:
                    temporal_index.append({
                        'time_bucket': current_time.isoformat(),
                        'data_points': len(batch_data),
                        'unique_persons': len(set(dp.global_person_id for dp in batch_data)),
                        'cameras': list(set(dp.camera_id for dp in batch_data))
                    })
                
                current_time = batch_end
                processed_batches += 1
                
                # Update progress
                progress.stage_progress = processed_batches / max(1, total_batches)
                
                if processed_batches % 5 == 0:  # Update cache every 5 batches
                    await self._cache_session_state(session)
            
            # Store temporal index in session data
            session.session_data['temporal_index'] = temporal_index
            
            # Mark stage complete
            progress.stages_completed.append(PreprocessingStage.TEMPORAL_INDEXING)
            progress.stage_progress = 1.0
            progress.overall_progress = 2.0 / 7.0
            
            await self._cache_session_state(session)
            
        except Exception as e:
            logger.error(f"Error in temporal indexing stage: {e}")
            raise
    
    async def _preprocessing_stage_feature_extraction(self, session: SessionState):
        """Stage 3: Extract features for analytics."""
        try:
            progress = session.preprocessing_progress
            progress.current_stage = PreprocessingStage.FEATURE_EXTRACTION
            progress.current_stage_message = "Extracting features for analytics"
            progress.stage_progress = 0.0
            
            # Simplified feature extraction - in production this would be more comprehensive
            config = session.configuration
            
            # Extract basic features from temporal index
            temporal_index = session.session_data.get('temporal_index', [])
            
            if temporal_index:
                features = {
                    'peak_activity_hour': max(temporal_index, key=lambda x: x['data_points'])['time_bucket'],
                    'total_unique_persons': sum(bucket['unique_persons'] for bucket in temporal_index),
                    'active_cameras': list(set(
                        camera for bucket in temporal_index for camera in bucket['cameras']
                    )),
                    'activity_distribution': [bucket['data_points'] for bucket in temporal_index]
                }
                
                session.session_data['extracted_features'] = features
            
            # Mark stage complete
            progress.stages_completed.append(PreprocessingStage.FEATURE_EXTRACTION)
            progress.stage_progress = 1.0
            progress.overall_progress = 3.0 / 7.0
            
            await self._cache_session_state(session)
            
        except Exception as e:
            logger.error(f"Error in feature extraction stage: {e}")
            raise
    
    async def _preprocessing_stage_trajectory_building(self, session: SessionState):
        """Stage 4: Build person trajectories."""
        try:
            progress = session.preprocessing_progress
            progress.current_stage = PreprocessingStage.TRAJECTORY_BUILDING
            progress.current_stage_message = "Building person trajectories"
            progress.stage_progress = 0.0
            
            config = session.configuration
            
            # Query all data for trajectory building
            query_filter = HistoricalQueryFilter(
                time_range=config.time_range,
                environment_id=config.environment_id
            )
            
            # Process in batches to manage memory
            trajectories = defaultdict(list)
            offset = 0
            
            while True:
                batch_data = await self.historical_service.query_historical_data(
                    query_filter, limit=config.batch_size, offset=offset
                )
                
                if not batch_data:
                    break
                
                # Group by person ID
                for data_point in batch_data:
                    trajectories[data_point.global_person_id].append({
                        'timestamp': data_point.timestamp.isoformat(),
                        'camera_id': data_point.camera_id,
                        'coordinates': {
                            'x': data_point.coordinates.x,
                            'y': data_point.coordinates.y
                        } if data_point.coordinates else None
                    })
                
                offset += len(batch_data)
                progress.data_points_processed = offset
                progress.stage_progress = min(1.0, offset / max(1, progress.total_data_points))
                
                if offset % (config.batch_size * 5) == 0:  # Update cache every 5 batches
                    await self._cache_session_state(session)
            
            # Store trajectories
            session.session_data['trajectories'] = dict(trajectories)
            session.trajectories_available = len(trajectories)
            progress.trajectories_built = len(trajectories)
            
            # Mark stage complete
            progress.stages_completed.append(PreprocessingStage.TRAJECTORY_BUILDING)
            progress.stage_progress = 1.0
            progress.overall_progress = 4.0 / 7.0
            
            await self._cache_session_state(session)
            
        except Exception as e:
            logger.error(f"Error in trajectory building stage: {e}")
            raise
    
    async def _preprocessing_stage_analytics_preparation(self, session: SessionState):
        """Stage 5: Prepare analytics data structures."""
        try:
            progress = session.preprocessing_progress
            progress.current_stage = PreprocessingStage.ANALYTICS_PREPARATION
            progress.current_stage_message = "Preparing analytics data structures"
            progress.stage_progress = 0.0
            
            # Prepare analytics summary
            temporal_index = session.session_data.get('temporal_index', [])
            features = session.session_data.get('extracted_features', {})
            
            analytics_summary = {
                'session_overview': {
                    'total_time_buckets': len(temporal_index),
                    'peak_activity': features.get('peak_activity_hour'),
                    'total_unique_persons': features.get('total_unique_persons', 0),
                    'active_cameras': features.get('active_cameras', [])
                },
                'prepared_at': datetime.utcnow().isoformat()
            }
            
            session.session_data['analytics_summary'] = analytics_summary
            
            # Mark stage complete
            progress.stages_completed.append(PreprocessingStage.ANALYTICS_PREPARATION)
            progress.stage_progress = 1.0
            progress.overall_progress = 5.0 / 7.0
            
            await self._cache_session_state(session)
            
        except Exception as e:
            logger.error(f"Error in analytics preparation stage: {e}")
            raise
    
    async def _preprocessing_stage_caching_optimization(self, session: SessionState):
        """Stage 6: Optimize caching for session data."""
        try:
            progress = session.preprocessing_progress
            progress.current_stage = PreprocessingStage.CACHING_OPTIMIZATION
            progress.current_stage_message = "Optimizing data caching"
            progress.stage_progress = 0.0
            
            config = session.configuration
            
            # Cache commonly accessed data
            cache_keys = []
            
            # Cache temporal index
            temporal_cache_key = f"session_temporal_{session.session_id}"
            await self.cache.set_json(
                temporal_cache_key,
                session.session_data.get('temporal_index', []),
                ttl=int(self.default_session_ttl.total_seconds())
            )
            cache_keys.append(temporal_cache_key)
            
            # Cache features
            if 'extracted_features' in session.session_data:
                features_cache_key = f"session_features_{session.session_id}"
                await self.cache.set_json(
                    features_cache_key,
                    session.session_data['extracted_features'],
                    ttl=int(self.default_session_ttl.total_seconds())
                )
                cache_keys.append(features_cache_key)
            
            # Cache trajectories (if not too large)
            if 'trajectories' in session.session_data:
                trajectories_size = len(json.dumps(session.session_data['trajectories']).encode('utf-8'))
                session.cached_data_size_mb = trajectories_size / (1024 * 1024)
                
                if session.cached_data_size_mb < config.memory_limit_mb / 2:  # Cache if under half memory limit
                    trajectories_cache_key = f"session_trajectories_{session.session_id}"
                    await self.cache.set_json(
                        trajectories_cache_key,
                        session.session_data['trajectories'],
                        ttl=int(self.default_session_ttl.total_seconds())
                    )
                    cache_keys.append(trajectories_cache_key)
            
            # Store cache keys for cleanup
            session.session_data['cache_keys'] = cache_keys
            
            # Mark stage complete
            progress.stages_completed.append(PreprocessingStage.CACHING_OPTIMIZATION)
            progress.stage_progress = 1.0
            progress.overall_progress = 6.0 / 7.0
            
            await self._cache_session_state(session)
            
        except Exception as e:
            logger.error(f"Error in caching optimization stage: {e}")
            raise
    
    async def _preprocessing_stage_completion(self, session: SessionState):
        """Stage 7: Complete preprocessing."""
        try:
            progress = session.preprocessing_progress
            progress.current_stage = PreprocessingStage.COMPLETION
            progress.current_stage_message = "Finalizing session preparation"
            progress.stage_progress = 0.0
            
            # Final session statistics
            session.session_data['preprocessing_stats'] = {
                'completed_at': datetime.utcnow().isoformat(),
                'stages_completed': len(progress.stages_completed),
                'data_points_processed': progress.data_points_processed,
                'trajectories_built': progress.trajectories_built,
                'cached_data_size_mb': session.cached_data_size_mb
            }
            
            # Mark stage complete
            progress.stages_completed.append(PreprocessingStage.COMPLETION)
            progress.stage_progress = 1.0
            progress.overall_progress = 1.0
            progress.current_stage_message = "Preprocessing completed successfully"
            
            await self._cache_session_state(session)
            
        except Exception as e:
            logger.error(f"Error in completion stage: {e}")
            raise
    
    # --- Session Queries ---
    
    async def execute_session_query(
        self,
        session_id: str,
        query_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a query within a session context."""
        try:
            start_time = time.time()
            
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            if session.status not in [SessionStatus.READY, SessionStatus.ACTIVE]:
                raise ValueError(f"Session {session_id} is not ready for queries (status: {session.status.value})")
            
            # Generate query ID
            query_id = f"{session_id}_{query_type}_{uuid.uuid4().hex[:8]}"
            
            # Execute query based on type
            result = {}
            cache_hit = False
            
            if query_type == "temporal_summary":
                result, cache_hit = await self._execute_temporal_summary_query(session, parameters)
            elif query_type == "person_trajectories":
                result, cache_hit = await self._execute_trajectories_query(session, parameters)
            elif query_type == "analytics_data":
                result, cache_hit = await self._execute_analytics_query(session, parameters)
            elif query_type == "raw_data":
                result, cache_hit = await self._execute_raw_data_query(session, parameters)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            # Record query execution
            execution_time = (time.time() - start_time) * 1000
            
            query_record = SessionQuery(
                query_id=query_id,
                session_id=session_id,
                query_type=query_type,
                parameters=parameters,
                executed_at=datetime.utcnow(),
                execution_time_ms=execution_time,
                result_size=len(json.dumps(result).encode('utf-8')),
                cache_hit=cache_hit
            )
            
            # Update session metrics
            session.total_query_time_ms += execution_time
            if cache_hit:
                session.cache_hit_rate = (session.cache_hit_rate + 1.0) / 2.0
            else:
                session.cache_hit_rate = session.cache_hit_rate * 0.9
            
            # Update global metrics
            self.performance_stats['total_query_time_ms'] += execution_time
            
            logger.debug(f"Executed query {query_id} in {execution_time:.1f}ms (cache_hit: {cache_hit})")
            
            return {
                'query_id': query_id,
                'result': result,
                'execution_time_ms': execution_time,
                'cache_hit': cache_hit,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing session query: {e}")
            raise
    
    async def _execute_temporal_summary_query(
        self,
        session: SessionState,
        parameters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """Execute temporal summary query."""
        try:
            # Check cache first
            cache_key = f"temporal_summary_{session.session_id}"
            cached_result = await self.cache.get_json(cache_key)
            if cached_result:
                return cached_result, True
            
            # Get temporal index from session data
            temporal_index = session.session_data.get('temporal_index', [])
            
            if not temporal_index:
                return {'error': 'No temporal index available'}, False
            
            # Build summary
            result = {
                'time_buckets': len(temporal_index),
                'total_data_points': sum(bucket['data_points'] for bucket in temporal_index),
                'total_unique_persons': sum(bucket['unique_persons'] for bucket in temporal_index),
                'activity_timeline': temporal_index,
                'peak_activity': max(temporal_index, key=lambda x: x['data_points']) if temporal_index else None
            }
            
            # Cache result
            await self.cache.set_json(cache_key, result, ttl=1800)  # 30 minutes
            
            return result, False
            
        except Exception as e:
            logger.error(f"Error executing temporal summary query: {e}")
            return {'error': str(e)}, False
    
    async def _execute_trajectories_query(
        self,
        session: SessionState,
        parameters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """Execute person trajectories query."""
        try:
            # Check if trajectories are available in session
            trajectories = session.session_data.get('trajectories')
            
            if trajectories is None:
                # Try to load from cache
                cache_key = f"session_trajectories_{session.session_id}"
                trajectories = await self.cache.get_json(cache_key)
                
                if trajectories:
                    return {'trajectories': trajectories}, True
                else:
                    return {'error': 'Trajectories not available in this session'}, False
            
            # Filter trajectories based on parameters
            person_id = parameters.get('person_id')
            limit = parameters.get('limit', 100)
            
            if person_id:
                filtered_trajectories = {person_id: trajectories.get(person_id, [])}
            else:
                # Return limited set of trajectories
                filtered_trajectories = dict(list(trajectories.items())[:limit])
            
            result = {
                'trajectories': filtered_trajectories,
                'total_persons': len(trajectories),
                'returned_persons': len(filtered_trajectories)
            }
            
            return result, False
            
        except Exception as e:
            logger.error(f"Error executing trajectories query: {e}")
            return {'error': str(e)}, False
    
    async def _execute_analytics_query(
        self,
        session: SessionState,
        parameters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """Execute analytics data query."""
        try:
            # Get analytics summary from session
            analytics_summary = session.session_data.get('analytics_summary')
            
            if not analytics_summary:
                return {'error': 'Analytics not available in this session'}, False
            
            # Get extracted features
            features = session.session_data.get('extracted_features', {})
            
            result = {
                'session_overview': analytics_summary.get('session_overview', {}),
                'features': features,
                'data_quality': {
                    'data_points_available': session.data_points_available,
                    'trajectories_available': session.trajectories_available,
                    'cached_data_size_mb': session.cached_data_size_mb
                }
            }
            
            return result, False
            
        except Exception as e:
            logger.error(f"Error executing analytics query: {e}")
            return {'error': str(e)}, False
    
    async def _execute_raw_data_query(
        self,
        session: SessionState,
        parameters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """Execute raw data query."""
        try:
            config = session.configuration
            
            # Extract query parameters
            start_time = parameters.get('start_time')
            end_time = parameters.get('end_time')
            limit = parameters.get('limit', 1000)
            offset = parameters.get('offset', 0)
            
            # Use session time range if not specified
            if not start_time:
                start_time = config.time_range.start_time
            else:
                start_time = datetime.fromisoformat(start_time)
            
            if not end_time:
                end_time = config.time_range.end_time
            else:
                end_time = datetime.fromisoformat(end_time)
            
            # Query raw data
            query_filter = HistoricalQueryFilter(
                time_range=TimeRange(start_time=start_time, end_time=end_time),
                environment_id=config.environment_id
            )
            
            raw_data = await self.historical_service.query_historical_data(
                query_filter, limit=limit, offset=offset
            )
            
            # Convert to serializable format
            serialized_data = []
            for data_point in raw_data:
                serialized_data.append({
                    'timestamp': data_point.timestamp.isoformat(),
                    'global_person_id': data_point.global_person_id,
                    'camera_id': data_point.camera_id,
                    'coordinates': {
                        'x': data_point.coordinates.x,
                        'y': data_point.coordinates.y
                    } if data_point.coordinates else None,
                    'detection': {
                        'confidence': data_point.detection.confidence,
                        'bbox': list(data_point.detection.bbox.to_xyxy())
                    }
                })
            
            result = {
                'data': serialized_data,
                'count': len(serialized_data),
                'offset': offset,
                'limit': limit,
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                }
            }
            
            return result, False
            
        except Exception as e:
            logger.error(f"Error executing raw data query: {e}")
            return {'error': str(e)}, False
    
    # --- Utility Methods ---
    
    async def _cache_session_state(self, session: SessionState):
        """Cache session state."""
        try:
            cache_key = f"historical_session_{session.session_id}"
            await self.cache.set_json(
                cache_key,
                session.to_dict(),
                ttl=int(self.default_session_ttl.total_seconds())
            )
            
        except Exception as e:
            logger.error(f"Error caching session state: {e}")
    
    def _deserialize_session_state(self, session_data: Dict[str, Any]) -> SessionState:
        """Deserialize session state from cached data."""
        try:
            # Reconstruct configuration
            config_data = session_data['configuration']
            time_range = TimeRange(
                start_time=datetime.fromisoformat(config_data['time_range']['start']),
                end_time=datetime.fromisoformat(config_data['time_range']['end'])
            )
            
            config = SessionConfiguration(
                session_id=config_data['session_id'],
                session_type=SessionType(config_data['session_type']),
                environment_id=config_data['environment_id'],
                time_range=time_range,
                user_id=config_data['user_id']
            )
            
            # Apply other config fields
            for key, value in config_data.items():
                if hasattr(config, key) and key not in ['session_id', 'session_type', 'environment_id', 'time_range', 'user_id']:
                    setattr(config, key, value)
            
            # Reconstruct preprocessing progress
            preprocessing_progress = None
            if session_data.get('preprocessing_progress'):
                pp_data = session_data['preprocessing_progress']
                preprocessing_progress = PreprocessingProgress(
                    current_stage=PreprocessingStage(pp_data['current_stage']),
                    stage_progress=pp_data['stage_progress'],
                    overall_progress=pp_data['overall_progress']
                )
                
                # Apply other progress fields
                for key, value in pp_data.items():
                    if hasattr(preprocessing_progress, key) and key not in ['current_stage']:
                        if key == 'stages_completed':
                            preprocessing_progress.stages_completed = [PreprocessingStage(s) for s in value]
                        elif key == 'estimated_completion_time' and value:
                            preprocessing_progress.estimated_completion_time = datetime.fromisoformat(value)
                        else:
                            setattr(preprocessing_progress, key, value)
            
            # Reconstruct session state
            session = SessionState(
                session_id=session_data['session_id'],
                status=SessionStatus(session_data['status']),
                configuration=config,
                created_at=datetime.fromisoformat(session_data['created_at']),
                last_activity=datetime.fromisoformat(session_data['last_activity']),
                preprocessing_progress=preprocessing_progress
            )
            
            # Apply other session fields
            for key, value in session_data.items():
                if hasattr(session, key) and key not in ['session_id', 'status', 'configuration', 'created_at', 'last_activity', 'preprocessing_progress']:
                    if key in ['preprocessing_started_at', 'ready_at', 'expires_at'] and value:
                        setattr(session, key, datetime.fromisoformat(value))
                    else:
                        setattr(session, key, value)
            
            return session
            
        except Exception as e:
            logger.error(f"Error deserializing session state: {e}")
            raise
    
    async def _cleanup_session_cache(self, session_id: str):
        """Clean up cache data for a session."""
        try:
            session = self.active_sessions.get(session_id)
            if session:
                cache_keys = session.session_data.get('cache_keys', [])
                for cache_key in cache_keys:
                    await self.cache.delete(cache_key)
            
            # Remove session state cache
            await self.cache.delete(f"historical_session_{session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session cache: {e}")
    
    async def _cleanup_all_sessions(self):
        """Clean up all active sessions."""
        try:
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.end_session(session_id)
                
        except Exception as e:
            logger.error(f"Error cleaning up all sessions: {e}")
    
    def _update_session_duration_metric(self, duration_minutes: float):
        """Update average session duration metric."""
        current_avg = self.performance_stats['avg_session_duration_minutes']
        sessions_completed = self.performance_stats['sessions_completed']
        
        if sessions_completed > 0:
            self.performance_stats['avg_session_duration_minutes'] = (
                (current_avg * (sessions_completed - 1) + duration_minutes) / sessions_completed
            )
    
    # --- Background Cleanup ---
    
    async def _background_cleanup_loop(self):
        """Background loop for session cleanup and maintenance."""
        while self._running:
            try:
                await asyncio.sleep(self.cache_cleanup_interval)
                
                # Clean up expired sessions
                await self._cleanup_expired_sessions()
                
                # Update memory usage metrics
                self._update_memory_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            now = datetime.utcnow()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session.expires_at and now > session.expires_at:
                    expired_sessions.append(session_id)
                elif (now - session.last_activity).total_seconds() > self.default_session_ttl.total_seconds():
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = self.active_sessions.get(session_id)
                if session:
                    session.status = SessionStatus.EXPIRED
                await self.end_session(session_id)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
    
    def _update_memory_metrics(self):
        """Update memory usage metrics."""
        try:
            total_memory = sum(session.memory_usage_mb for session in self.active_sessions.values())
            self.performance_stats['memory_usage_mb'] = total_memory
            
        except Exception as e:
            logger.error(f"Error updating memory metrics: {e}")
    
    # --- Public API Methods ---
    
    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """List all sessions for a user."""
        try:
            user_session_ids = self.user_sessions.get(user_id, set())
            sessions = []
            
            for session_id in user_session_ids:
                session = self.active_sessions.get(session_id)
                if session:
                    sessions.append({
                        'session_id': session.session_id,
                        'status': session.status.value,
                        'session_type': session.configuration.session_type.value,
                        'environment_id': session.configuration.environment_id,
                        'created_at': session.created_at.isoformat(),
                        'last_activity': session.last_activity.isoformat(),
                        'expires_at': session.expires_at.isoformat() if session.expires_at else None
                    })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error listing user sessions: {e}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'HistoricalSessionManager',
            'running': self._running,
            'active_sessions': len(self.active_sessions),
            'preprocessing_tasks': len(self.preprocessing_tasks),
            'performance_stats': self.performance_stats.copy(),
            'configuration': {
                'max_concurrent_sessions_per_user': self.max_concurrent_sessions_per_user,
                'max_total_active_sessions': self.max_total_active_sessions,
                'default_session_ttl_hours': self.default_session_ttl.total_seconds() / 3600,
                'batch_processing_size': self.batch_processing_size
            }
        }


# Global service instance
_historical_session_manager: Optional[HistoricalSessionManager] = None


def get_historical_session_manager() -> Optional[HistoricalSessionManager]:
    """Get the global historical session manager instance."""
    return _historical_session_manager


def initialize_historical_session_manager(
    historical_data_service: HistoricalDataService,
    datetime_range_manager: DateTimeRangeManager,
    environment_config_service: EnvironmentConfigurationService,
    temporal_query_engine: TemporalQueryEngine,
    tracking_cache: TrackingCache
) -> HistoricalSessionManager:
    """Initialize the global historical session manager."""
    global _historical_session_manager
    if _historical_session_manager is None:
        _historical_session_manager = HistoricalSessionManager(
            historical_data_service,
            datetime_range_manager,
            environment_config_service,
            temporal_query_engine,
            tracking_cache
        )
    return _historical_session_manager