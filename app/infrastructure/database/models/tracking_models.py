"""
TimescaleDB models for tracking data storage.

Handles:
- Person tracking events
- Detection events
- Trajectory data
- Analytics aggregations
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    JSON,
    Index,
    ForeignKey,
    BigInteger,
    Date,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid

from app.infrastructure.database.base import Base


class TrackingEvent(Base):
    """
    Time-series table for tracking events.
    
    Stores all person tracking events with TimescaleDB partitioning.
    """
    __tablename__ = 'tracking_events'
    
    # Primary key and partitioning
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Person identification
    global_person_id = Column(String(100), nullable=False, index=True)
    track_id = Column(String(100), nullable=True, index=True)
    
    # Camera and location
    camera_id = Column(String(50), nullable=False, index=True)
    environment_id = Column(String(50), nullable=False, index=True)
    
    # Position data
    position_x = Column(Float, nullable=True)
    position_y = Column(Float, nullable=True)
    world_x = Column(Float, nullable=True)  # World coordinates
    world_y = Column(Float, nullable=True)
    
    # Bounding box
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    
    # Confidence and quality
    detection_confidence = Column(Float, nullable=True)
    reid_confidence = Column(Float, nullable=True)
    
    # Event type and metadata
    event_type = Column(String(50), nullable=False)  # 'detection', 'entry', 'exit', 'transition'
    session_id = Column(String(100), nullable=True, index=True)
    
    # Additional data
    event_metadata = Column(JSONB, nullable=True)
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_tracking_events_timestamp', 'timestamp'),
        Index('idx_tracking_events_person_time', 'global_person_id', 'timestamp'),
        Index('idx_tracking_events_camera_time', 'camera_id', 'timestamp'),
        Index('idx_tracking_events_session_time', 'session_id', 'timestamp'),
        Index('idx_tracking_events_env_time', 'environment_id', 'timestamp'),
    )


class DetectionEvent(Base):
    """
    Time-series table for detection events.
    
    Stores all person detection events from all cameras.
    """
    __tablename__ = 'detection_events'
    
    # Primary key and partitioning
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Detection data
    camera_id = Column(String(50), nullable=False, index=True)
    environment_id = Column(String(50), nullable=False, index=True)
    frame_number = Column(Integer, nullable=True)
    
    # Bounding box
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # Detection confidence
    confidence = Column(Float, nullable=False)
    
    # Object class
    object_class = Column(String(50), nullable=False, default='person')
    
    # Associated tracking event
    tracking_event_id = Column(UUID(as_uuid=True), ForeignKey('tracking_events.id'), nullable=True)
    
    # Session context
    session_id = Column(String(100), nullable=True, index=True)
    
    # Additional metadata
    detection_metadata = Column(JSONB, nullable=True)
    
    # Relationship
    tracking_event = relationship("TrackingEvent", back_populates="detection_events")
    
    # Indexes
    __table_args__ = (
        Index('idx_detection_events_timestamp', 'timestamp'),
        Index('idx_detection_events_camera_time', 'camera_id', 'timestamp'),
        Index('idx_detection_events_session_time', 'session_id', 'timestamp'),
        Index('idx_detection_events_confidence', 'confidence'),
    )


class PersonTrajectory(Base):
    """
    Time-series table for person trajectory data.
    
    Stores trajectory points for person movement analysis.
    """
    __tablename__ = 'person_trajectories'
    
    # Primary key and partitioning
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Person identification
    global_person_id = Column(String(100), nullable=False, index=True)
    
    # Camera and environment
    camera_id = Column(String(50), nullable=False, index=True)
    environment_id = Column(String(50), nullable=False, index=True)
    
    # Trajectory data
    sequence_number = Column(Integer, nullable=False)
    position_x = Column(Float, nullable=False)
    position_y = Column(Float, nullable=False)
    world_x = Column(Float, nullable=True)
    world_y = Column(Float, nullable=True)
    
    # Movement data
    velocity_x = Column(Float, nullable=True)
    velocity_y = Column(Float, nullable=True)
    acceleration_x = Column(Float, nullable=True)
    acceleration_y = Column(Float, nullable=True)
    
    # Quality metrics
    confidence = Column(Float, nullable=True)
    smoothed = Column(Boolean, default=False)
    
    # Session context
    session_id = Column(String(100), nullable=True, index=True)
    
    # Additional data
    trajectory_metadata = Column(JSONB, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_person_trajectories_timestamp', 'timestamp'),
        Index('idx_person_trajectories_person_time', 'global_person_id', 'timestamp'),
        Index('idx_person_trajectories_camera_time', 'camera_id', 'timestamp'),
        Index('idx_person_trajectories_sequence', 'global_person_id', 'sequence_number'),
    )


class PersonIdentity(Base):
    """
    Table for person identity management.
    
    Stores person identities and their associations across cameras.
    """
    __tablename__ = 'person_identities'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Person identity
    global_person_id = Column(String(100), nullable=False, unique=True, index=True)
    
    # Creation and updates
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # First and last seen
    first_seen_at = Column(DateTime(timezone=True), nullable=False)
    last_seen_at = Column(DateTime(timezone=True), nullable=False)
    
    # Camera associations
    first_seen_camera = Column(String(50), nullable=False)
    last_seen_camera = Column(String(50), nullable=False)
    cameras_seen = Column(JSONB, nullable=True)  # List of camera IDs
    
    # Environment context
    environment_id = Column(String(50), nullable=False, index=True)
    
    # Identity status
    is_active = Column(Boolean, default=True)
    confidence = Column(Float, nullable=True)
    
    # Feature data
    primary_embedding = Column(JSONB, nullable=True)  # Primary embedding
    embedding_history = Column(JSONB, nullable=True)  # Historical embeddings
    
    # Statistics
    total_detections = Column(Integer, default=0)
    total_tracking_time = Column(Float, default=0.0)  # Total time tracked (seconds)
    
    # Additional metadata
    identity_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    tracking_events = relationship(
        "TrackingEvent", 
        back_populates="person_identity",
        foreign_keys="TrackingEvent.global_person_id",
        primaryjoin="PersonIdentity.global_person_id == TrackingEvent.global_person_id"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_person_identities_global_id', 'global_person_id'),
        Index('idx_person_identities_environment', 'environment_id'),
        Index('idx_person_identities_last_seen', 'last_seen_at'),
        Index('idx_person_identities_active', 'is_active'),
    )


class AnalyticsAggregation(Base):
    """
    Table for pre-computed analytics aggregations.
    
    Stores hourly, daily, and weekly aggregations for performance.
    """
    __tablename__ = 'analytics_aggregations'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time window
    time_bucket = Column(DateTime(timezone=True), nullable=False, index=True)
    aggregation_type = Column(String(20), nullable=False)  # 'hourly', 'daily', 'weekly'
    
    # Scope
    camera_id = Column(String(50), nullable=True, index=True)
    environment_id = Column(String(50), nullable=False, index=True)
    
    # Person counts
    total_persons = Column(Integer, default=0)
    unique_persons = Column(Integer, default=0)
    active_persons = Column(Integer, default=0)
    
    # Detection counts
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float, nullable=True)
    
    # Movement data
    total_entries = Column(Integer, default=0)
    total_exits = Column(Integer, default=0)
    total_transitions = Column(Integer, default=0)
    
    # Time metrics
    avg_dwell_time = Column(Float, nullable=True)
    total_tracking_time = Column(Float, default=0.0)
    
    # Quality metrics
    avg_reid_confidence = Column(Float, nullable=True)
    successful_tracks = Column(Integer, default=0)
    failed_tracks = Column(Integer, default=0)
    
    # Additional aggregated data
    aggregated_data = Column(JSONB, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_analytics_aggregations_time_bucket', 'time_bucket'),
        Index('idx_analytics_aggregations_type_camera', 'aggregation_type', 'camera_id'),
        Index('idx_analytics_aggregations_environment', 'environment_id'),
        Index('idx_analytics_aggregations_composite', 'aggregation_type', 'environment_id', 'time_bucket'),
    )


class SessionRecord(Base):
    """
    Table for tracking session records.
    
    Stores session metadata and statistics.
    """
    __tablename__ = 'session_records'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Session identification
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    
    # User and environment
    user_id = Column(String(100), nullable=True, index=True)
    environment_id = Column(String(50), nullable=False, index=True)
    
    # Session timing
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Float, nullable=True)  # Duration in seconds
    
    # Camera configuration
    camera_ids = Column(JSONB, nullable=False)  # List of camera IDs
    
    # Session statistics
    total_persons_tracked = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    total_events = Column(Integer, default=0)
    
    # Quality metrics
    avg_detection_confidence = Column(Float, nullable=True)
    avg_reid_confidence = Column(Float, nullable=True)
    
    # Session status
    status = Column(String(20), nullable=False, default='active')  # 'active', 'completed', 'interrupted'
    
    # Configuration and settings
    settings = Column(JSONB, nullable=True)
    
    # Additional metadata
    session_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_session_records_session_id', 'session_id'),
        Index('idx_session_records_user', 'user_id'),
        Index('idx_session_records_environment', 'environment_id'),
        Index('idx_session_records_start_time', 'start_time'),
        Index('idx_session_records_status', 'status'),
    )


# Add relationships
TrackingEvent.detection_events = relationship("DetectionEvent", back_populates="tracking_event")
TrackingEvent.person_identity = relationship(
    "PersonIdentity", 
    back_populates="tracking_events",
    foreign_keys=[TrackingEvent.global_person_id],
    primaryjoin="TrackingEvent.global_person_id == PersonIdentity.global_person_id"
)


class AnalyticsTotals(Base):
    """Aggregated detection metrics stored per time bucket."""

    __tablename__ = 'analytics_totals'

    bucket_start = Column(DateTime(timezone=True), primary_key=True)
    bucket_size_seconds = Column(Integer, primary_key=True)
    environment_id = Column(String(50), primary_key=True)
    camera_id = Column(String(50), primary_key=True, default='__all__')

    detections = Column(BigInteger, nullable=False, default=0)
    unique_entities = Column(BigInteger, nullable=False, default=0)
    confidence_sum = Column(Float, nullable=False, default=0.0)
    confidence_samples = Column(BigInteger, nullable=False, default=0)
    uptime_percent = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_analytics_totals_bucket', 'bucket_start'),
        Index('idx_analytics_totals_env', 'environment_id'),
        Index('idx_analytics_totals_camera_bucket', 'environment_id', 'camera_id', 'bucket_start'),
    )


class AnalyticsUptimeDaily(Base):
    """Daily uptime snapshot metrics for cameras and environments."""

    __tablename__ = 'analytics_uptime_daily'

    day = Column(Date, primary_key=True)
    environment_id = Column(String(50), primary_key=True)
    camera_id = Column(String(50), primary_key=True, default='__all__')
    uptime_percent = Column(Float, nullable=False)
    samples = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_analytics_uptime_daily_day_env', 'day', 'environment_id'),
        Index('idx_analytics_uptime_daily_camera', 'camera_id'),
    )
