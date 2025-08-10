"""
Historical session domain service for managing analysis sessions.

Focused service for session lifecycle management, state tracking,
and session-specific business logic. Maximum 350 lines per plan.
"""
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

from app.domain.shared.entities.base_entity import BaseEntity
from app.domain.shared.value_objects.time_range import TimeRange
from app.domain.shared.value_objects.user_id import UserID

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
    ANALYSIS = "analysis"
    PLAYBACK = "playback"
    COMPARISON = "comparison"
    EXPORT = "export"
    INVESTIGATION = "investigation"


@dataclass(frozen=True)
class SessionConfiguration:
    """Configuration parameters for a historical session."""
    session_type: SessionType
    environment_id: str
    time_range: TimeRange
    user_id: UserID
    
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
    custom_parameters: Dict[str, any] = field(default_factory=dict)


@dataclass
class SessionState(BaseEntity):
    """Session state entity."""
    session_id: str
    configuration: SessionConfiguration
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    
    # Processing state
    preprocessing_started_at: Optional[datetime] = None
    ready_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Data metrics
    data_points_available: int = 0
    trajectories_available: int = 0
    cached_data_size_mb: float = 0.0
    
    # Performance metrics
    total_query_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Session data
    session_data: Dict[str, any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status in {SessionStatus.READY, SessionStatus.ACTIVE, SessionStatus.PREPROCESSING}
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at
    
    def can_process_queries(self) -> bool:
        """Check if session can process queries."""
        return self.status == SessionStatus.READY or self.status == SessionStatus.ACTIVE
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def mark_ready(self) -> None:
        """Mark session as ready for queries."""
        self.status = SessionStatus.READY
        self.ready_at = datetime.utcnow()
        self.update_activity()
    
    def mark_error(self, error_message: str) -> None:
        """Mark session as error state."""
        self.status = SessionStatus.ERROR
        self.error_message = error_message
        self.update_activity()


class HistoricalSessionService:
    """
    Historical session domain service.
    
    Manages session lifecycle, state tracking, and session-specific business logic
    for historical analysis operations.
    """
    
    def __init__(self, default_ttl_hours: int = 4, max_sessions_per_user: int = 5):
        """
        Initialize historical session service.
        
        Args:
            default_ttl_hours: Default session time-to-live in hours
            max_sessions_per_user: Maximum concurrent sessions per user
        """
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.max_sessions_per_user = max_sessions_per_user
        
        # Session storage
        self.active_sessions: Dict[str, SessionState] = {}
        self.user_sessions: Dict[str, Set[str]] = {}
        
        # Session statistics
        self._session_stats = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'sessions_expired': 0,
            'sessions_errored': 0
        }
        
        logger.debug("HistoricalSessionService initialized")
    
    def create_session(
        self,
        user_id: UserID,
        session_type: SessionType,
        environment_id: str,
        time_range: TimeRange,
        **config_options
    ) -> SessionState:
        """
        Create new historical session.
        
        Args:
            user_id: User creating the session
            session_type: Type of session
            environment_id: Environment identifier
            time_range: Time range for analysis
            **config_options: Additional configuration options
            
        Returns:
            Created session state
            
        Raises:
            ValueError: If user has too many active sessions
        """
        # Check user session limits
        user_session_count = len(self.user_sessions.get(str(user_id), set()))
        if user_session_count >= self.max_sessions_per_user:
            raise ValueError(f"User {user_id} has reached maximum session limit ({self.max_sessions_per_user})")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create configuration
        configuration = SessionConfiguration(
            session_type=session_type,
            environment_id=environment_id,
            time_range=time_range,
            user_id=user_id,
            **config_options
        )
        
        # Calculate expiration
        expires_at = datetime.utcnow() + self.default_ttl
        
        # Create session state
        session_state = SessionState(
            session_id=session_id,
            configuration=configuration,
            status=SessionStatus.INITIALIZING,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=expires_at
        )
        
        # Store session
        self.active_sessions[session_id] = session_state
        
        # Track user sessions
        user_key = str(user_id)
        if user_key not in self.user_sessions:
            self.user_sessions[user_key] = set()
        self.user_sessions[user_key].add(session_id)
        
        self._session_stats['sessions_created'] += 1
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_state
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state or None if not found
        """
        session = self.active_sessions.get(session_id)
        
        if session and session.is_expired():
            self._expire_session(session_id)
            return None
        
        return session
    
    def get_user_sessions(self, user_id: UserID) -> List[SessionState]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user sessions
        """
        user_key = str(user_id)
        session_ids = self.user_sessions.get(user_key, set())
        
        sessions = []
        expired_sessions = []
        
        for session_id in session_ids:
            session = self.active_sessions.get(session_id)
            if session:
                if session.is_expired():
                    expired_sessions.append(session_id)
                else:
                    sessions.append(session)
        
        # Clean up expired sessions
        for session_id in expired_sessions:
            self._expire_session(session_id)
        
        return sessions
    
    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update session status.
        
        Args:
            session_id: Session identifier
            status: New status
            error_message: Optional error message
            
        Returns:
            True if update successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.status = status
        session.update_activity()
        
        if status == SessionStatus.ERROR and error_message:
            session.mark_error(error_message)
            self._session_stats['sessions_errored'] += 1
        elif status == SessionStatus.READY:
            session.mark_ready()
        elif status == SessionStatus.COMPLETED:
            self._session_stats['sessions_completed'] += 1
        
        logger.debug(f"Updated session {session_id} status to {status.value}")
        return True
    
    def extend_session(self, session_id: str, additional_hours: int = 2) -> bool:
        """
        Extend session expiration time.
        
        Args:
            session_id: Session identifier
            additional_hours: Hours to extend
            
        Returns:
            True if extension successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.expires_at += timedelta(hours=additional_hours)
        session.update_activity()
        
        logger.info(f"Extended session {session_id} by {additional_hours} hours")
        return True
    
    def complete_session(self, session_id: str) -> bool:
        """
        Mark session as completed.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if completion successful
        """
        return self.update_session_status(session_id, SessionStatus.COMPLETED)
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if session.expires_at <= current_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._expire_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_session_statistics(self) -> Dict[str, any]:
        """Get session service statistics."""
        active_count = len(self.active_sessions)
        user_count = len(self.user_sessions)
        
        return {
            **self._session_stats,
            'active_sessions': active_count,
            'active_users': user_count,
            'avg_sessions_per_user': active_count / max(user_count, 1)
        }
    
    def _expire_session(self, session_id: str) -> None:
        """Expire a session and clean up resources."""
        session = self.active_sessions.pop(session_id, None)
        if not session:
            return
        
        # Remove from user sessions
        user_key = str(session.configuration.user_id)
        if user_key in self.user_sessions:
            self.user_sessions[user_key].discard(session_id)
            if not self.user_sessions[user_key]:
                del self.user_sessions[user_key]
        
        self._session_stats['sessions_expired'] += 1
        
        logger.debug(f"Expired session {session_id}")
    
    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.active_sessions)
    
    def get_user_session_count(self, user_id: UserID) -> int:
        """Get number of active sessions for user."""
        user_key = str(user_id)
        return len(self.user_sessions.get(user_key, set()))