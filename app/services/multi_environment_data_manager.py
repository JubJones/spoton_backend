"""
Multi-Environment Data Management System

Comprehensive multi-environment data management system providing:
- Environment-specific data isolation and access control
- Cross-environment analytics and comparison capabilities
- Environment switching with state management and session persistence
- Environment-specific user permissions and security
- Data federation across multiple environments
- Performance monitoring and resource allocation per environment
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from app.services.environment_configuration_service import (
    EnvironmentConfigurationService,
    EnvironmentConfiguration,
    EnvironmentType
)
from app.services.historical_data_service import (
    HistoricalDataService,
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)
from app.services.advanced_analytics_engine import (
    AdvancedAnalyticsEngine,
    AnalyticsGranularity,
    RealTimeAnalyticsState
)
from app.infrastructure.cache.tracking_cache import TrackingCache

logger = logging.getLogger(__name__)


class DataAccessLevel(Enum):
    """Data access permission levels."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    CROSS_ENVIRONMENT = "cross_environment"


class EnvironmentStatus(Enum):
    """Environment operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    RESTRICTED = "restricted"


@dataclass
class EnvironmentDataMetrics:
    """Data metrics for an environment."""
    environment_id: str
    total_data_points: int
    unique_persons: int
    date_range_days: int
    storage_size_mb: float
    active_cameras: int
    zones_configured: int
    last_data_timestamp: Optional[datetime] = None
    data_quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'environment_id': self.environment_id,
            'total_data_points': self.total_data_points,
            'unique_persons': self.unique_persons,
            'date_range_days': self.date_range_days,
            'storage_size_mb': self.storage_size_mb,
            'active_cameras': self.active_cameras,
            'zones_configured': self.zones_configured,
            'last_data_timestamp': self.last_data_timestamp.isoformat() if self.last_data_timestamp else None,
            'data_quality_score': self.data_quality_score
        }


@dataclass
class UserEnvironmentSession:
    """User session for environment access."""
    session_id: str
    user_id: str
    environment_id: str
    access_level: DataAccessLevel
    start_time: datetime
    last_activity: datetime
    session_data: Dict[str, Any] = field(default_factory=dict)
    permissions: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'environment_id': self.environment_id,
            'access_level': self.access_level.value,
            'start_time': self.start_time.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'session_data': self.session_data,
            'permissions': list(self.permissions)
        }


@dataclass
class CrossEnvironmentQuery:
    """Query definition for cross-environment operations."""
    query_id: str
    environment_ids: List[str]
    time_range: TimeRange
    query_type: str  # comparison, aggregation, correlation
    parameters: Dict[str, Any] = field(default_factory=dict)
    requester_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query_id': self.query_id,
            'environment_ids': self.environment_ids,
            'time_range': {
                'start': self.time_range.start_time.isoformat(),
                'end': self.time_range.end_time.isoformat()
            },
            'query_type': self.query_type,
            'parameters': self.parameters,
            'requester_id': self.requester_id
        }


class MultiEnvironmentDataManager:
    """Comprehensive service for managing data across multiple environments."""
    
    def __init__(
        self,
        environment_config_service: EnvironmentConfigurationService,
        historical_data_service: HistoricalDataService,
        analytics_engine: AdvancedAnalyticsEngine,
        tracking_cache: TrackingCache
    ):
        self.environment_service = environment_config_service
        self.historical_service = historical_data_service
        self.analytics_engine = analytics_engine
        self.cache = tracking_cache
        
        # Environment management
        self.environment_states: Dict[str, EnvironmentStatus] = {}
        self.environment_metrics: Dict[str, EnvironmentDataMetrics] = {}
        
        # Session management
        self.active_sessions: Dict[str, UserEnvironmentSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)  # user_id -> session_ids
        
        # Data isolation and access control
        self.access_policies: Dict[str, Dict[str, DataAccessLevel]] = {}  # environment_id -> user_id -> access_level
        self.environment_permissions: Dict[str, Set[str]] = defaultdict(set)  # environment_id -> permissions
        
        # Cross-environment operations
        self.active_cross_queries: Dict[str, CrossEnvironmentQuery] = {}
        self.cross_environment_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'environments_monitored': 0,
            'active_sessions': 0,
            'cross_environment_queries': 0,
            'data_access_requests': 0,
            'avg_query_time_ms': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Background processing
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("MultiEnvironmentDataManager initialized")
    
    async def start_manager(self):
        """Start the multi-environment data manager."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize environment states
        await self._initialize_environments()
        
        # Start background monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("MultiEnvironmentDataManager started")
    
    async def stop_manager(self):
        """Stop the multi-environment data manager."""
        self._running = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MultiEnvironmentDataManager stopped")
    
    # --- Environment Management ---
    
    async def _initialize_environments(self):
        """Initialize all environments and their states."""
        try:
            environments = await self.environment_service.list_environments()
            
            for environment in environments:
                # Initialize environment state
                if environment.is_active:
                    self.environment_states[environment.environment_id] = EnvironmentStatus.ACTIVE
                else:
                    self.environment_states[environment.environment_id] = EnvironmentStatus.INACTIVE
                
                # Initialize metrics
                await self._update_environment_metrics(environment.environment_id)
                
                # Initialize default permissions
                self.environment_permissions[environment.environment_id] = {
                    'read', 'analyze', 'export'
                }
            
            self.performance_metrics['environments_monitored'] = len(environments)
            logger.info(f"Initialized {len(environments)} environments")
            
        except Exception as e:
            logger.error(f"Error initializing environments: {e}")
    
    async def _update_environment_metrics(self, environment_id: str):
        """Update metrics for a specific environment."""
        try:
            start_time = time.time()
            
            # Get environment configuration
            environment = await self.environment_service.get_environment(environment_id)
            if not environment:
                return
            
            # Get data availability
            date_ranges = await self.environment_service.get_available_date_ranges(environment_id)
            env_range = date_ranges.get(environment_id, {})
            
            # Query recent data for metrics
            if env_range.get('has_data', False):
                # Get sample data for metrics calculation
                recent_time = datetime.utcnow() - timedelta(hours=24)
                time_range = TimeRange(start_time=recent_time, end_time=datetime.utcnow())
                
                query_filter = HistoricalQueryFilter(
                    time_range=time_range,
                    environment_id=environment_id
                )
                
                try:
                    sample_data = await self.historical_service.query_historical_data(query_filter, limit=1000)
                    
                    # Calculate metrics
                    total_points = len(sample_data) * 24  # Rough estimate for full day
                    unique_persons = len(set(dp.global_person_id for dp in sample_data))
                    
                    last_timestamp = None
                    if sample_data:
                        last_timestamp = max(dp.timestamp for dp in sample_data)
                    
                    # Estimate storage size (rough calculation)
                    storage_size = total_points * 0.5  # Assume ~0.5KB per data point
                    
                    # Data quality score (simplified)
                    quality_score = min(1.0, len(sample_data) / 100.0)  # Based on data density
                    
                except Exception as e:
                    logger.warning(f"Could not calculate data metrics for {environment_id}: {e}")
                    total_points = 0
                    unique_persons = 0
                    last_timestamp = None
                    storage_size = 0.0
                    quality_score = 0.0
            else:
                total_points = 0
                unique_persons = 0
                last_timestamp = None
                storage_size = 0.0
                quality_score = 0.0
            
            # Create metrics object
            metrics = EnvironmentDataMetrics(
                environment_id=environment_id,
                total_data_points=total_points,
                unique_persons=unique_persons,
                date_range_days=env_range.get('total_days', 0),
                storage_size_mb=storage_size,
                active_cameras=len(environment.get_active_cameras()),
                zones_configured=len(environment.zones),
                last_data_timestamp=last_timestamp,
                data_quality_score=quality_score
            )
            
            self.environment_metrics[environment_id] = metrics
            
            # Cache metrics
            await self._cache_environment_metrics(environment_id, metrics)
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Updated metrics for environment {environment_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error updating environment metrics for {environment_id}: {e}")
    
    async def get_environment_status(self, environment_id: str) -> Dict[str, Any]:
        """Get comprehensive status for an environment."""
        try:
            environment = await self.environment_service.get_environment(environment_id)
            if not environment:
                raise ValueError(f"Environment {environment_id} not found")
            
            status = self.environment_states.get(environment_id, EnvironmentStatus.INACTIVE)
            metrics = self.environment_metrics.get(environment_id)
            
            # Get active sessions for this environment
            env_sessions = [
                session for session in self.active_sessions.values()
                if session.environment_id == environment_id
            ]
            
            return {
                'environment_id': environment_id,
                'name': environment.name,
                'type': environment.environment_type.value,
                'status': status.value,
                'is_active': environment.is_active,
                'metrics': metrics.to_dict() if metrics else None,
                'active_sessions': len(env_sessions),
                'permissions': list(self.environment_permissions.get(environment_id, set())),
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting environment status: {e}")
            raise
    
    async def set_environment_status(
        self,
        environment_id: str,
        status: EnvironmentStatus,
        reason: Optional[str] = None
    ) -> bool:
        """Set operational status for an environment."""
        try:
            environment = await self.environment_service.get_environment(environment_id)
            if not environment:
                return False
            
            old_status = self.environment_states.get(environment_id, EnvironmentStatus.INACTIVE)
            self.environment_states[environment_id] = status
            
            # Handle status change side effects
            if status == EnvironmentStatus.MAINTENANCE:
                await self._handle_maintenance_mode(environment_id)
            elif status == EnvironmentStatus.RESTRICTED:
                await self._handle_restricted_mode(environment_id)
            elif status == EnvironmentStatus.INACTIVE:
                await self._handle_inactive_mode(environment_id)
            
            logger.info(f"Environment {environment_id} status changed from {old_status.value} to {status.value}")
            
            # Cache status change
            status_data = {
                'environment_id': environment_id,
                'status': status.value,
                'changed_at': datetime.utcnow().isoformat(),
                'reason': reason
            }
            
            await self.cache.set_json(f"env_status_{environment_id}", status_data, ttl=3600)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting environment status: {e}")
            return False
    
    # --- Session Management ---
    
    async def create_user_session(
        self,
        user_id: str,
        environment_id: str,
        access_level: DataAccessLevel,
        permissions: Optional[Set[str]] = None
    ) -> UserEnvironmentSession:
        """Create a new user session for environment access."""
        try:
            # Verify environment exists and is accessible
            environment = await self.environment_service.get_environment(environment_id)
            if not environment:
                raise ValueError(f"Environment {environment_id} not found")
            
            env_status = self.environment_states.get(environment_id, EnvironmentStatus.INACTIVE)
            if env_status in [EnvironmentStatus.INACTIVE, EnvironmentStatus.MAINTENANCE]:
                raise ValueError(f"Environment {environment_id} is not available (status: {env_status.value})")
            
            # Check user permissions
            if not await self._check_user_access(user_id, environment_id, access_level):
                raise PermissionError(f"User {user_id} does not have {access_level.value} access to environment {environment_id}")
            
            # Generate session ID
            session_data = f"{user_id}_{environment_id}_{datetime.utcnow().isoformat()}"
            session_id = hashlib.sha256(session_data.encode()).hexdigest()[:16]
            
            # Create session
            session = UserEnvironmentSession(
                session_id=session_id,
                user_id=user_id,
                environment_id=environment_id,
                access_level=access_level,
                start_time=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                permissions=permissions or set()
            )
            
            # Store session
            self.active_sessions[session_id] = session
            self.user_sessions[user_id].append(session_id)
            
            # Update metrics
            self.performance_metrics['active_sessions'] = len(self.active_sessions)
            
            # Cache session
            await self.cache.set_json(f"session_{session_id}", session.to_dict(), ttl=3600)
            
            logger.info(f"Created session {session_id} for user {user_id} in environment {environment_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating user session: {e}")
            raise
    
    async def get_user_session(self, session_id: str) -> Optional[UserEnvironmentSession]:
        """Get user session by ID."""
        try:
            # Check memory first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_activity = datetime.utcnow()
                return session
            
            # Check cache
            session_data = await self.cache.get_json(f"session_{session_id}")
            if session_data:
                session = UserEnvironmentSession(
                    session_id=session_data['session_id'],
                    user_id=session_data['user_id'],
                    environment_id=session_data['environment_id'],
                    access_level=DataAccessLevel(session_data['access_level']),
                    start_time=datetime.fromisoformat(session_data['start_time']),
                    last_activity=datetime.fromisoformat(session_data['last_activity']),
                    session_data=session_data.get('session_data', {}),
                    permissions=set(session_data.get('permissions', []))
                )
                
                # Restore to memory
                self.active_sessions[session_id] = session
                session.last_activity = datetime.utcnow()
                
                return session
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user session: {e}")
            return None
    
    async def switch_environment(
        self,
        session_id: str,
        new_environment_id: str,
        preserve_state: bool = True
    ) -> UserEnvironmentSession:
        """Switch user session to a different environment."""
        try:
            # Get current session
            current_session = await self.get_user_session(session_id)
            if not current_session:
                raise ValueError(f"Session {session_id} not found")
            
            # Preserve current session state if requested
            preserved_state = {}
            if preserve_state:
                preserved_state = current_session.session_data.copy()
            
            # End current session
            await self.end_user_session(session_id)
            
            # Create new session for new environment
            new_session = await self.create_user_session(
                user_id=current_session.user_id,
                environment_id=new_environment_id,
                access_level=current_session.access_level,
                permissions=current_session.permissions
            )
            
            # Restore preserved state
            if preserve_state:
                new_session.session_data.update(preserved_state)
                new_session.session_data['previous_environment'] = current_session.environment_id
                new_session.session_data['switched_at'] = datetime.utcnow().isoformat()
            
            logger.info(f"Switched session from environment {current_session.environment_id} to {new_environment_id}")
            return new_session
            
        except Exception as e:
            logger.error(f"Error switching environment: {e}")
            raise
    
    async def end_user_session(self, session_id: str) -> bool:
        """End a user session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            # Remove from user sessions
            if session.user_id in self.user_sessions:
                self.user_sessions[session.user_id] = [
                    sid for sid in self.user_sessions[session.user_id] if sid != session_id
                ]
                if not self.user_sessions[session.user_id]:
                    del self.user_sessions[session.user_id]
            
            # Update metrics
            self.performance_metrics['active_sessions'] = len(self.active_sessions)
            
            # Remove from cache
            await self.cache.delete(f"session_{session_id}")
            
            logger.info(f"Ended session {session_id} for user {session.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending user session: {e}")
            return False
    
    # --- Data Access Control ---
    
    async def _check_user_access(
        self,
        user_id: str,
        environment_id: str,
        required_access: DataAccessLevel
    ) -> bool:
        """Check if user has required access level for environment."""
        try:
            # Get user access level for environment
            env_policies = self.access_policies.get(environment_id, {})
            user_access = env_policies.get(user_id, DataAccessLevel.READ_ONLY)
            
            # Define access hierarchy
            access_hierarchy = {
                DataAccessLevel.READ_ONLY: 0,
                DataAccessLevel.READ_WRITE: 1,
                DataAccessLevel.ADMIN: 2,
                DataAccessLevel.CROSS_ENVIRONMENT: 3
            }
            
            user_level = access_hierarchy.get(user_access, 0)
            required_level = access_hierarchy.get(required_access, 0)
            
            return user_level >= required_level
            
        except Exception as e:
            logger.error(f"Error checking user access: {e}")
            return False
    
    async def set_user_access(
        self,
        user_id: str,
        environment_id: str,
        access_level: DataAccessLevel
    ) -> bool:
        """Set user access level for an environment."""
        try:
            if environment_id not in self.access_policies:
                self.access_policies[environment_id] = {}
            
            self.access_policies[environment_id][user_id] = access_level
            
            # Cache access policy
            policy_data = {
                'user_id': user_id,
                'environment_id': environment_id,
                'access_level': access_level.value,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            await self.cache.set_json(f"access_{environment_id}_{user_id}", policy_data, ttl=86400)
            
            logger.info(f"Set {access_level.value} access for user {user_id} in environment {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting user access: {e}")
            return False
    
    # --- Cross-Environment Operations ---
    
    async def execute_cross_environment_query(
        self,
        query: CrossEnvironmentQuery
    ) -> Dict[str, Any]:
        """Execute a query across multiple environments."""
        try:
            start_time = time.time()
            
            # Validate environments
            valid_environments = []
            for env_id in query.environment_ids:
                environment = await self.environment_service.get_environment(env_id)
                if environment and self.environment_states.get(env_id) == EnvironmentStatus.ACTIVE:
                    valid_environments.append(env_id)
            
            if not valid_environments:
                raise ValueError("No valid environments found for cross-environment query")
            
            # Store active query
            self.active_cross_queries[query.query_id] = query
            self.performance_metrics['cross_environment_queries'] += 1
            
            # Execute query based on type
            if query.query_type == "comparison":
                result = await self._execute_comparison_query(valid_environments, query.time_range, query.parameters)
            elif query.query_type == "aggregation":
                result = await self._execute_aggregation_query(valid_environments, query.time_range, query.parameters)
            elif query.query_type == "correlation":
                result = await self._execute_correlation_query(valid_environments, query.time_range, query.parameters)
            else:
                raise ValueError(f"Unsupported query type: {query.query_type}")
            
            # Add metadata
            processing_time = (time.time() - start_time) * 1000
            result.update({
                'query_id': query.query_id,
                'environments_queried': valid_environments,
                'query_type': query.query_type,
                'processing_time_ms': processing_time,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Cache result
            cache_key = f"cross_query_{query.query_id}"
            await self.cache.set_json(cache_key, result, ttl=1800)  # 30 minutes TTL
            
            # Clean up active query
            if query.query_id in self.active_cross_queries:
                del self.active_cross_queries[query.query_id]
            
            logger.info(f"Completed cross-environment query {query.query_id} in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error executing cross-environment query: {e}")
            # Clean up on error
            if query.query_id in self.active_cross_queries:
                del self.active_cross_queries[query.query_id]
            raise
    
    async def _execute_comparison_query(
        self,
        environment_ids: List[str],
        time_range: TimeRange,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute comparison query across environments."""
        try:
            comparison_results = {}
            
            # Query each environment
            for env_id in environment_ids:
                query_filter = HistoricalQueryFilter(
                    time_range=time_range,
                    environment_id=env_id
                )
                
                # Get analytics for environment
                analytics_result = await self.analytics_engine.calculate_historical_metrics(
                    time_range, env_id, AnalyticsGranularity.HOUR
                )
                
                comparison_results[env_id] = {
                    'environment_id': env_id,
                    'total_persons': analytics_result.get('unique_persons', 0),
                    'data_points': analytics_result.get('total_data_points', 0),
                    'cameras_active': analytics_result.get('cameras_involved', 0),
                    'metrics': analytics_result.get('metrics', {})
                }
            
            # Calculate comparison metrics
            comparison_metrics = self._calculate_comparison_metrics(comparison_results)
            
            return {
                'type': 'comparison',
                'environment_results': comparison_results,
                'comparison_metrics': comparison_metrics
            }
            
        except Exception as e:
            logger.error(f"Error executing comparison query: {e}")
            return {'type': 'comparison', 'error': str(e)}
    
    async def _execute_aggregation_query(
        self,
        environment_ids: List[str],
        time_range: TimeRange,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute aggregation query across environments."""
        try:
            aggregated_data = {
                'total_persons': 0,
                'total_data_points': 0,
                'total_cameras': 0,
                'environment_count': len(environment_ids),
                'combined_metrics': defaultdict(list)
            }
            
            environment_contributions = {}
            
            # Aggregate data from all environments
            for env_id in environment_ids:
                analytics_result = await self.analytics_engine.calculate_historical_metrics(
                    time_range, env_id, AnalyticsGranularity.HOUR
                )
                
                env_contribution = {
                    'persons': analytics_result.get('unique_persons', 0),
                    'data_points': analytics_result.get('total_data_points', 0),
                    'cameras': analytics_result.get('cameras_involved', 0)
                }
                
                environment_contributions[env_id] = env_contribution
                
                # Add to totals
                aggregated_data['total_persons'] += env_contribution['persons']
                aggregated_data['total_data_points'] += env_contribution['data_points']
                aggregated_data['total_cameras'] += env_contribution['cameras']
                
                # Aggregate metrics
                metrics = analytics_result.get('metrics', {})
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and 'occupancy_trends' in metric_data:
                        # Aggregate occupancy trends
                        trend_data = metric_data.get('trend_analysis', {})
                        if 'average_occupancy' in trend_data:
                            aggregated_data['combined_metrics']['average_occupancy'].append(
                                trend_data['average_occupancy']
                            )
            
            # Calculate final aggregated metrics
            final_metrics = {}
            for metric_name, values in aggregated_data['combined_metrics'].items():
                if values:
                    final_metrics[metric_name] = {
                        'sum': sum(values),
                        'average': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values)
                    }
            
            return {
                'type': 'aggregation',
                'aggregated_data': aggregated_data,
                'environment_contributions': environment_contributions,
                'final_metrics': final_metrics
            }
            
        except Exception as e:
            logger.error(f"Error executing aggregation query: {e}")
            return {'type': 'aggregation', 'error': str(e)}
    
    async def _execute_correlation_query(
        self,
        environment_ids: List[str],
        time_range: TimeRange,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute correlation query across environments."""
        try:
            # This is a simplified correlation analysis
            # In production, this would perform sophisticated correlation analysis
            
            environment_time_series = {}
            
            # Get time series data for each environment
            for env_id in environment_ids:
                analytics_result = await self.analytics_engine.calculate_historical_metrics(
                    time_range, env_id, AnalyticsGranularity.HOUR
                )
                
                # Extract time series from occupancy trends
                metrics = analytics_result.get('metrics', {})
                occupancy_trends = metrics.get('occupancy_trends', {})
                time_series = occupancy_trends.get('time_series', [])
                
                if time_series:
                    environment_time_series[env_id] = [
                        entry.get('occupancy', 0) for entry in time_series
                    ]
            
            # Calculate correlations (simplified)
            correlations = {}
            env_list = list(environment_time_series.keys())
            
            for i in range(len(env_list)):
                for j in range(i + 1, len(env_list)):
                    env_a, env_b = env_list[i], env_list[j]
                    
                    if env_a in environment_time_series and env_b in environment_time_series:
                        series_a = environment_time_series[env_a]
                        series_b = environment_time_series[env_b]
                        
                        # Simple correlation calculation (Pearson)
                        if len(series_a) == len(series_b) and len(series_a) > 1:
                            try:
                                import numpy as np
                                correlation = np.corrcoef(series_a, series_b)[0, 1]
                                correlations[f"{env_a}-{env_b}"] = correlation
                            except Exception:
                                correlations[f"{env_a}-{env_b}"] = 0.0
            
            return {
                'type': 'correlation',
                'environment_time_series': environment_time_series,
                'correlations': correlations,
                'correlation_strength': self._analyze_correlation_strength(correlations)
            }
            
        except Exception as e:
            logger.error(f"Error executing correlation query: {e}")
            return {'type': 'correlation', 'error': str(e)}
    
    def _calculate_comparison_metrics(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparison metrics across environments."""
        try:
            if not comparison_results:
                return {}
            
            # Extract values for comparison
            person_counts = [result['total_persons'] for result in comparison_results.values()]
            data_point_counts = [result['data_points'] for result in comparison_results.values()]
            camera_counts = [result['cameras_active'] for result in comparison_results.values()]
            
            import numpy as np
            
            return {
                'person_count_stats': {
                    'max': max(person_counts),
                    'min': min(person_counts),
                    'average': np.mean(person_counts),
                    'std': np.std(person_counts),
                    'total': sum(person_counts)
                },
                'data_volume_stats': {
                    'max': max(data_point_counts),
                    'min': min(data_point_counts),
                    'average': np.mean(data_point_counts),
                    'total': sum(data_point_counts)
                },
                'camera_utilization': {
                    'max': max(camera_counts),
                    'min': min(camera_counts),
                    'average': np.mean(camera_counts),
                    'total': sum(camera_counts)
                },
                'environment_rankings': {
                    'by_person_count': sorted(
                        comparison_results.keys(),
                        key=lambda env: comparison_results[env]['total_persons'],
                        reverse=True
                    ),
                    'by_activity': sorted(
                        comparison_results.keys(),
                        key=lambda env: comparison_results[env]['data_points'],
                        reverse=True
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating comparison metrics: {e}")
            return {}
    
    def _analyze_correlation_strength(self, correlations: Dict[str, float]) -> Dict[str, Any]:
        """Analyze strength of correlations."""
        try:
            if not correlations:
                return {}
            
            strong_correlations = {k: v for k, v in correlations.items() if abs(v) > 0.7}
            moderate_correlations = {k: v for k, v in correlations.items() if 0.3 < abs(v) <= 0.7}
            weak_correlations = {k: v for k, v in correlations.items() if abs(v) <= 0.3}
            
            return {
                'strong_correlations': strong_correlations,
                'moderate_correlations': moderate_correlations,
                'weak_correlations': weak_correlations,
                'average_correlation': sum(correlations.values()) / len(correlations) if correlations else 0.0,
                'max_correlation': max(correlations.values()) if correlations else 0.0,
                'min_correlation': min(correlations.values()) if correlations else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlation strength: {e}")
            return {}
    
    # --- Utility Methods ---
    
    async def _handle_maintenance_mode(self, environment_id: str):
        """Handle environment entering maintenance mode."""
        try:
            # End all active sessions for this environment
            env_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if session.environment_id == environment_id
            ]
            
            for session_id in env_sessions:
                await self.end_user_session(session_id)
            
            logger.info(f"Environment {environment_id} entered maintenance mode, ended {len(env_sessions)} sessions")
            
        except Exception as e:
            logger.error(f"Error handling maintenance mode for {environment_id}: {e}")
    
    async def _handle_restricted_mode(self, environment_id: str):
        """Handle environment entering restricted mode."""
        try:
            # End sessions for users without admin access
            env_sessions = [
                (session_id, session) for session_id, session in self.active_sessions.items()
                if session.environment_id == environment_id
            ]
            
            ended_sessions = 0
            for session_id, session in env_sessions:
                if session.access_level not in [DataAccessLevel.ADMIN, DataAccessLevel.CROSS_ENVIRONMENT]:
                    await self.end_user_session(session_id)
                    ended_sessions += 1
            
            logger.info(f"Environment {environment_id} entered restricted mode, ended {ended_sessions} non-admin sessions")
            
        except Exception as e:
            logger.error(f"Error handling restricted mode for {environment_id}: {e}")
    
    async def _handle_inactive_mode(self, environment_id: str):
        """Handle environment becoming inactive."""
        try:
            # End all sessions for this environment
            env_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if session.environment_id == environment_id
            ]
            
            for session_id in env_sessions:
                await self.end_user_session(session_id)
            
            logger.info(f"Environment {environment_id} became inactive, ended {len(env_sessions)} sessions")
            
        except Exception as e:
            logger.error(f"Error handling inactive mode for {environment_id}: {e}")
    
    async def _cache_environment_metrics(self, environment_id: str, metrics: EnvironmentDataMetrics):
        """Cache environment metrics."""
        try:
            cache_key = f"env_metrics_{environment_id}"
            await self.cache.set_json(cache_key, metrics.to_dict(), ttl=900)  # 15 minutes TTL
            
        except Exception as e:
            logger.error(f"Error caching environment metrics: {e}")
    
    # --- Background Monitoring ---
    
    async def _monitoring_loop(self):
        """Background monitoring loop for multi-environment management."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Update environment metrics
                for environment_id in list(self.environment_states.keys()):
                    await self._update_environment_metrics(environment_id)
                
                # Clean up expired sessions
                await self._cleanup_expired_sessions()
                
                # Update performance metrics
                self._update_performance_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired user sessions."""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                # Sessions expire after 1 hour of inactivity
                if (current_time - session.last_activity).total_seconds() > 3600:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self.end_user_session(session_id)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            self.performance_metrics['environments_monitored'] = len(self.environment_states)
            self.performance_metrics['active_sessions'] = len(self.active_sessions)
            self.performance_metrics['cross_environment_queries'] = len(self.active_cross_queries)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    # --- Status and Health ---
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'MultiEnvironmentDataManager',
            'running': self._running,
            'performance_metrics': self.performance_metrics.copy(),
            'environment_states': {env_id: status.value for env_id, status in self.environment_states.items()},
            'active_sessions_count': len(self.active_sessions),
            'active_cross_queries': len(self.active_cross_queries),
            'environments_managed': len(self.environment_states)
        }


# Global service instance
_multi_environment_data_manager: Optional[MultiEnvironmentDataManager] = None


def get_multi_environment_data_manager() -> Optional[MultiEnvironmentDataManager]:
    """Get the global multi-environment data manager instance."""
    return _multi_environment_data_manager


def initialize_multi_environment_data_manager(
    environment_config_service: EnvironmentConfigurationService,
    historical_data_service: HistoricalDataService,
    analytics_engine: AdvancedAnalyticsEngine,
    tracking_cache: TrackingCache
) -> MultiEnvironmentDataManager:
    """Initialize the global multi-environment data manager."""
    global _multi_environment_data_manager
    if _multi_environment_data_manager is None:
        _multi_environment_data_manager = MultiEnvironmentDataManager(
            environment_config_service,
            historical_data_service,
            analytics_engine,
            tracking_cache
        )
    return _multi_environment_data_manager