"""
Historical analysis orchestrator for application layer.

Coordinates historical sessions, preprocessing, and data access.
Replaces the mega historical_session_manager.py with focused orchestration.
Maximum 400 lines per plan.
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio
import uuid
from collections import defaultdict

from app.domain.shared.value_objects.time_range import TimeRange
from app.domain.shared.value_objects.user_id import UserID
from app.domain.analytics.services.historical_session_service import (
    HistoricalSessionService, SessionType, SessionStatus, SessionConfiguration, SessionState
)
from app.domain.analytics.services.preprocessing_service import (
    PreprocessingService, PreprocessingProgress, PreprocessingStage
)
from app.application.services.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)


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


class HistoricalAnalysisOrchestrator:
    """
    Historical analysis orchestrator for application layer.
    
    Coordinates session management, preprocessing, and data access
    for historical analysis operations. Provides high-level orchestration
    while delegating specific responsibilities to domain services.
    """
    
    def __init__(
        self,
        session_service: HistoricalSessionService,
        preprocessing_service: PreprocessingService,
        configuration_manager: ConfigurationManager
    ):
        """
        Initialize historical analysis orchestrator.
        
        Args:
            session_service: Session management domain service
            preprocessing_service: Data preprocessing domain service
            configuration_manager: Configuration management service
        """
        self.session_service = session_service
        self.preprocessing_service = preprocessing_service
        self.config_manager = configuration_manager
        
        # Query tracking
        self.session_queries: Dict[str, List[SessionQuery]] = defaultdict(list)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance metrics
        self.orchestrator_stats = {
            'total_queries_executed': 0,
            'total_query_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'avg_session_duration_minutes': 0.0
        }
        
        logger.debug("HistoricalAnalysisOrchestrator initialized")
    
    async def start_orchestrator(self) -> None:
        """Start the historical analysis orchestrator."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
        
        logger.info("Historical analysis orchestrator started")
    
    async def stop_orchestrator(self) -> None:
        """Stop the historical analysis orchestrator."""
        self._running = False
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Historical analysis orchestrator stopped")
    
    async def create_analysis_session(
        self,
        user_id: UserID,
        session_type: SessionType,
        environment_id: str,
        start_time: datetime,
        end_time: datetime,
        **config_options
    ) -> Dict[str, Any]:
        """
        Create new historical analysis session.
        
        Args:
            user_id: User creating the session
            session_type: Type of analysis session
            environment_id: Environment identifier
            start_time: Analysis start time
            end_time: Analysis end time
            **config_options: Additional configuration options
            
        Returns:
            Session creation result with session details
        """
        try:
            # Validate environment
            env_validation = await self._validate_environment(environment_id)
            if not env_validation['is_valid']:
                return {
                    'success': False,
                    'error': 'Environment validation failed',
                    'details': env_validation
                }
            
            # Create time range
            time_range = TimeRange(start_time=start_time, end_time=end_time)
            
            # Create session
            session_state = self.session_service.create_session(
                user_id=user_id,
                session_type=session_type,
                environment_id=environment_id,
                time_range=time_range,
                **config_options
            )
            
            # Start preprocessing if enabled
            preprocessing_task_id = None
            if session_state.configuration.enable_preprocessing:
                preprocessing_task_id = await self._start_session_preprocessing(session_state)
            else:
                # Mark session as ready if no preprocessing needed
                self.session_service.update_session_status(
                    session_state.session_id,
                    SessionStatus.READY
                )
            
            return {
                'success': True,
                'session_id': session_state.session_id,
                'session_status': session_state.status.value,
                'preprocessing_task_id': preprocessing_task_id,
                'expires_at': session_state.expires_at.isoformat(),
                'configuration': session_state.configuration.__dict__
            }
            
        except Exception as e:
            logger.error(f"Failed to create analysis session: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed session status.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session status information
        """
        session = self.session_service.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        # Get preprocessing progress if applicable
        preprocessing_progress = None
        if session.status == SessionStatus.PREPROCESSING:
            # Find preprocessing task for this session
            for task_id, task in self.preprocessing_service.active_tasks.items():
                if task.session_id == session_id:
                    preprocessing_progress = self.preprocessing_service.get_preprocessing_progress(task_id)
                    break
        
        # Get session query history
        query_history = self.session_queries.get(session_id, [])
        
        return {
            'session_id': session_id,
            'status': session.status.value,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'expires_at': session.expires_at.isoformat(),
            'environment_id': session.configuration.environment_id,
            'time_range': {
                'start': session.configuration.time_range.start_time.isoformat(),
                'end': session.configuration.time_range.end_time.isoformat()
            },
            'preprocessing_progress': preprocessing_progress.to_dict() if preprocessing_progress else None,
            'data_points_available': session.data_points_available,
            'trajectories_available': session.trajectories_available,
            'cached_data_size_mb': session.cached_data_size_mb,
            'total_queries': len(query_history),
            'cache_hit_rate': session.cache_hit_rate,
            'memory_usage_mb': session.memory_usage_mb,
            'can_process_queries': session.can_process_queries()
        }
    
    async def execute_session_query(
        self,
        session_id: str,
        query_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute query within a session.
        
        Args:
            session_id: Session identifier
            query_type: Type of query to execute
            parameters: Query parameters
            
        Returns:
            Query execution result
        """
        session = self.session_service.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if not session.can_process_queries():
            return {'error': f'Session not ready for queries. Status: {session.status.value}'}
        
        try:
            start_time = datetime.utcnow()
            
            # Execute query (placeholder implementation)
            result_data = await self._execute_query_logic(session, query_type, parameters)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create query record
            query_record = SessionQuery(
                query_id=str(uuid.uuid4()),
                session_id=session_id,
                query_type=query_type,
                parameters=parameters,
                executed_at=start_time,
                execution_time_ms=execution_time,
                result_size=len(str(result_data)),
                cache_hit=result_data.get('_cache_hit', False)
            )
            
            # Store query record
            self.session_queries[session_id].append(query_record)
            
            # Update session metrics
            session.total_query_time_ms += execution_time
            session.update_activity()
            
            # Update orchestrator metrics
            self.orchestrator_stats['total_queries_executed'] += 1
            self.orchestrator_stats['total_query_time_ms'] += execution_time
            
            return {
                'success': True,
                'query_id': query_record.query_id,
                'execution_time_ms': execution_time,
                'result_size': query_record.result_size,
                'cache_hit': query_record.cache_hit,
                'data': result_data
            }
            
        except Exception as e:
            logger.error(f"Query execution failed for session {session_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def extend_session(self, session_id: str, additional_hours: int = 2) -> Dict[str, Any]:
        """
        Extend session expiration time.
        
        Args:
            session_id: Session identifier
            additional_hours: Hours to extend
            
        Returns:
            Extension result
        """
        success = self.session_service.extend_session(session_id, additional_hours)
        
        if success:
            session = self.session_service.get_session(session_id)
            return {
                'success': True,
                'new_expiration': session.expires_at.isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'Session not found or extension failed'
            }
    
    async def complete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Complete and clean up session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Completion result
        """
        success = self.session_service.complete_session(session_id)
        
        if success:
            # Clean up query history
            self.session_queries.pop(session_id, None)
            
            return {'success': True}
        else:
            return {
                'success': False,
                'error': 'Session not found or completion failed'
            }
    
    def get_user_sessions(self, user_id: UserID) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user session summaries
        """
        sessions = self.session_service.get_user_sessions(user_id)
        
        session_summaries = []
        for session in sessions:
            query_count = len(self.session_queries.get(session.session_id, []))
            
            session_summaries.append({
                'session_id': session.session_id,
                'session_type': session.configuration.session_type.value,
                'environment_id': session.configuration.environment_id,
                'status': session.status.value,
                'created_at': session.created_at.isoformat(),
                'expires_at': session.expires_at.isoformat(),
                'total_queries': query_count,
                'can_process_queries': session.can_process_queries()
            })
        
        return session_summaries
    
    def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics."""
        session_stats = self.session_service.get_session_statistics()
        preprocessing_stats = self.preprocessing_service.get_preprocessing_statistics()
        
        # Calculate combined metrics
        total_queries = self.orchestrator_stats['total_queries_executed']
        avg_query_time = (
            self.orchestrator_stats['total_query_time_ms'] / max(total_queries, 1)
        )
        
        return {
            'orchestrator': {
                **self.orchestrator_stats,
                'avg_query_time_ms': avg_query_time,
                'total_active_sessions': session_stats['active_sessions']
            },
            'sessions': session_stats,
            'preprocessing': preprocessing_stats
        }
    
    async def _start_session_preprocessing(self, session_state: SessionState) -> str:
        """Start preprocessing for a session."""
        # Update session status
        self.session_service.update_session_status(
            session_state.session_id,
            SessionStatus.PREPROCESSING
        )
        
        # Create progress callback
        def progress_callback(progress: PreprocessingProgress):
            session_state.data_points_available = progress.data_points_processed
            session_state.trajectories_available = progress.trajectories_built
        
        # Create completion callback
        def completion_callback(success: bool, error_message: Optional[str]):
            if success:
                self.session_service.update_session_status(
                    session_state.session_id,
                    SessionStatus.READY
                )
            else:
                self.session_service.update_session_status(
                    session_state.session_id,
                    SessionStatus.ERROR,
                    error_message
                )
        
        # Start preprocessing
        task_id = await self.preprocessing_service.start_preprocessing(
            session_id=session_state.session_id,
            environment_id=session_state.configuration.environment_id,
            time_range=session_state.configuration.time_range,
            batch_size=session_state.configuration.batch_size,
            progress_callback=progress_callback,
            completion_callback=completion_callback
        )
        
        return task_id
    
    async def _validate_environment(self, environment_id: str) -> Dict[str, Any]:
        """Validate environment for analysis."""
        # Use configuration manager to validate environment
        try:
            camera_configs = self.config_manager.camera_config_service.get_all_camera_configs()
            ai_configs = self.config_manager.ai_config_service.get_ai_model_configs()
            
            # Check if environment has required configurations
            env_cameras = [cam for cam in camera_configs if cam.get('environment_id') == environment_id]
            
            return {
                'is_valid': len(env_cameras) > 0 and len(ai_configs) > 0,
                'cameras_available': len(env_cameras),
                'models_configured': len(ai_configs)
            }
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            return {
                'is_valid': False,
                'error': str(e)
            }
    
    async def _execute_query_logic(
        self,
        session: SessionState,
        query_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the actual query logic (placeholder implementation)."""
        # This is a placeholder - in real implementation, this would
        # delegate to appropriate data services based on query type
        
        await asyncio.sleep(0.1)  # Simulate query processing
        
        return {
            'query_type': query_type,
            'parameters': parameters,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {'placeholder': 'result'},
            '_cache_hit': False  # Placeholder cache status
        }
    
    async def _background_cleanup_loop(self):
        """Background task for cleaning up expired sessions and resources."""
        while self._running:
            try:
                # Clean up expired sessions
                expired_count = self.session_service.cleanup_expired_sessions()
                
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired sessions")
                
                # Wait before next cleanup
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying