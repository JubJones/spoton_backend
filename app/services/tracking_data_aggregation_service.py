"""
Comprehensive Tracking Data Aggregation Service

Centralizes and manages all tracking data for frontend visualization including:
- Multi-camera tracking data synchronization
- Person identity aggregation across cameras
- Real-time data streaming coordination
- Historical data retrieval and caching
- Cross-session data continuity
- Performance optimization and indexing
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

from app.domains.detection.entities.detection import Detection
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import Coordinate
from app.infrastructure.cache.tracking_cache import TrackingCache
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.api.v1.visualization_schemas import (
    MultiCameraVisualizationUpdate,
    EnhancedPersonTrack,
    PersonDetailedInfo,
    PersonMovementMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class PersonSession:
    """Represents a person's complete tracking session across cameras."""
    global_person_id: str
    first_detected: datetime
    last_seen: datetime
    total_detections: int = 0
    cameras_visited: Set[str] = field(default_factory=set)
    detection_history: List[Dict[str, Any]] = field(default_factory=list)
    movement_history: List[Coordinate] = field(default_factory=list)
    identity_confidence: float = 1.0
    session_active: bool = True
    last_activity_time: datetime = field(default_factory=datetime.utcnow)
    
    def add_detection(
        self,
        detection: Detection,
        camera_id: str,
        map_coordinate: Optional[Coordinate] = None
    ):
        """Add a new detection to the person's session."""
        self.last_seen = detection.timestamp
        self.last_activity_time = datetime.utcnow()
        self.total_detections += 1
        self.cameras_visited.add(camera_id)
        
        # Add to detection history
        detection_record = {
            'timestamp': detection.timestamp,
            'camera_id': camera_id,
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'track_id': detection.track_id
        }
        
        if map_coordinate:
            detection_record['map_coordinate'] = {
                'x': map_coordinate.x,
                'y': map_coordinate.y,
                'confidence': getattr(map_coordinate, 'confidence', 1.0)
            }
            self.movement_history.append(map_coordinate)
        
        self.detection_history.append(detection_record)
        
        # Keep only recent history (configurable limit)
        if len(self.detection_history) > 1000:
            self.detection_history.pop(0)
        
        if len(self.movement_history) > 1000:
            self.movement_history.pop(0)
    
    def calculate_movement_metrics(self) -> PersonMovementMetrics:
        """Calculate comprehensive movement metrics."""
        if len(self.movement_history) < 2:
            return PersonMovementMetrics()
        
        # Calculate total distance
        total_distance = 0.0
        speeds = []
        direction_changes = 0
        last_direction = None
        
        for i in range(1, len(self.movement_history)):
            curr_pos = self.movement_history[i]
            prev_pos = self.movement_history[i-1]
            
            # Calculate distance
            distance = ((curr_pos.x - prev_pos.x) ** 2 + 
                       (curr_pos.y - prev_pos.y) ** 2) ** 0.5
            total_distance += distance
            
            # Calculate speed if we have timestamps
            if (i < len(self.detection_history) and 
                i-1 < len(self.detection_history)):
                time_diff = (self.detection_history[i]['timestamp'] - 
                           self.detection_history[i-1]['timestamp']).total_seconds()
                if time_diff > 0:
                    speed = distance / time_diff
                    speeds.append(speed)
            
            # Calculate direction changes
            if prev_pos.x != curr_pos.x or prev_pos.y != curr_pos.y:
                direction = np.arctan2(
                    curr_pos.y - prev_pos.y,
                    curr_pos.x - prev_pos.x
                )
                if last_direction is not None:
                    angle_diff = abs(direction - last_direction)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    if angle_diff > np.pi / 4:  # 45 degrees threshold
                        direction_changes += 1
                last_direction = direction
        
        # Calculate metrics
        average_speed = sum(speeds) / len(speeds) if speeds else 0.0
        duration = (self.last_seen - self.first_detected).total_seconds()
        dwell_time = duration - sum(speeds) if speeds else duration
        
        # Calculate trajectory smoothness (0 = very jagged, 1 = very smooth)
        if len(self.movement_history) > 2:
            # Simple smoothness metric based on direction changes
            max_changes = len(self.movement_history) - 2
            trajectory_smoothness = 1.0 - (direction_changes / max(1, max_changes))
        else:
            trajectory_smoothness = 1.0
        
        return PersonMovementMetrics(
            total_distance=total_distance,
            average_speed=average_speed,
            dwell_time=max(0.0, dwell_time),
            direction_changes=direction_changes,
            trajectory_smoothness=max(0.0, min(1.0, trajectory_smoothness))
        )
    
    def is_session_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired due to inactivity."""
        time_since_activity = datetime.utcnow() - self.last_activity_time
        return time_since_activity > timedelta(minutes=timeout_minutes)


@dataclass
class AggregationMetrics:
    """Metrics for tracking aggregation performance."""
    persons_tracked: int = 0
    total_detections: int = 0
    cross_camera_handoffs: int = 0
    session_merges: int = 0
    active_sessions: int = 0
    expired_sessions: int = 0
    processing_time_ms: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'persons_tracked': self.persons_tracked,
            'total_detections': self.total_detections,
            'cross_camera_handoffs': self.cross_camera_handoffs,
            'session_merges': self.session_merges,
            'active_sessions': self.active_sessions,
            'expired_sessions': self.expired_sessions,
            'processing_time_ms': self.processing_time_ms,
            'last_update': self.last_update.isoformat()
        }


class TrackingDataAggregationService:
    """Comprehensive service for aggregating and managing tracking data."""
    
    def __init__(
        self,
        tracking_cache: TrackingCache,
        tracking_repository: TrackingRepository
    ):
        self.cache = tracking_cache
        self.repository = tracking_repository
        
        # Active person sessions by task_id
        self.active_sessions: Dict[str, Dict[str, PersonSession]] = defaultdict(dict)
        
        # Cross-camera identity mapping
        self.identity_mapping: Dict[str, Dict[str, str]] = defaultdict(dict)  # task_id -> {local_id: global_id}
        
        # Metrics and performance tracking
        self.metrics = AggregationMetrics()
        self.performance_history = deque(maxlen=1000)
        
        # Configuration
        self.session_timeout_minutes = 30
        self.max_sessions_per_task = 1000
        self.identity_confidence_threshold = 0.7
        self.cross_camera_merge_threshold = 0.8
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("TrackingDataAggregationService initialized")
    
    async def start_service(self):
        """Start background cleanup and maintenance tasks."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        logger.info("Started tracking data aggregation service")
    
    async def stop_service(self):
        """Stop background tasks and cleanup."""
        self._running = False
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped tracking data aggregation service")
    
    # --- Core Aggregation Methods ---
    
    async def aggregate_tracking_data(
        self,
        task_id: str,
        camera_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate tracking data from multiple cameras into unified format.
        
        Args:
            task_id: Unique identifier for the tracking task
            camera_data: Dict mapping camera_id to frame data with detections
        
        Returns:
            Aggregated tracking data with cross-camera person identities
        """
        start_time = time.time()
        
        try:
            aggregated_data = {
                'task_id': task_id,
                'timestamp': datetime.utcnow(),
                'cameras': {},
                'global_persons': {},
                'session_stats': {}
            }
            
            # Process each camera's data
            for camera_id, frame_data in camera_data.items():
                camera_result = await self._process_camera_data(
                    task_id, camera_id, frame_data
                )
                aggregated_data['cameras'][camera_id] = camera_result
            
            # Perform cross-camera identity resolution
            await self._resolve_cross_camera_identities(task_id, aggregated_data)
            
            # Generate global person summaries
            aggregated_data['global_persons'] = await self._generate_person_summaries(task_id)
            
            # Update session statistics
            aggregated_data['session_stats'] = self._get_session_statistics(task_id)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time)
            
            logger.debug(
                f"Aggregated tracking data for task {task_id}: "
                f"{len(aggregated_data['global_persons'])} persons, "
                f"{processing_time:.1f}ms"
            )
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error aggregating tracking data: {e}")
            raise
    
    async def _process_camera_data(
        self,
        task_id: str,
        camera_id: str,
        frame_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process individual camera data and update person sessions."""
        try:
            detections = frame_data.get('detections', [])
            person_identities = frame_data.get('person_identities', [])
            map_coordinates = frame_data.get('map_coordinates', [])
            
            processed_tracks = []
            
            # Process each detection
            for i, detection in enumerate(detections):
                # Get corresponding identity and coordinate
                person_identity = person_identities[i] if i < len(person_identities) else None
                map_coordinate = map_coordinates[i] if i < len(map_coordinates) else None
                
                # Determine global person ID
                global_id = await self._resolve_person_identity(
                    task_id, camera_id, detection, person_identity
                )
                
                # Update or create person session
                await self._update_person_session(
                    task_id, global_id, detection, camera_id, map_coordinate
                )
                
                # Create processed track
                processed_track = {
                    'detection': detection,
                    'global_person_id': global_id,
                    'camera_id': camera_id,
                    'map_coordinate': map_coordinate,
                    'identity_confidence': (
                        person_identity.confidence if person_identity else 1.0
                    )
                }
                processed_tracks.append(processed_track)
            
            return {
                'camera_id': camera_id,
                'frame_timestamp': frame_data.get('timestamp', datetime.utcnow()),
                'processed_tracks': processed_tracks,
                'detection_count': len(detections)
            }
            
        except Exception as e:
            logger.error(f"Error processing camera data for {camera_id}: {e}")
            raise
    
    async def _resolve_person_identity(
        self,
        task_id: str,
        camera_id: str,
        detection: Detection,
        person_identity: Optional[PersonIdentity]
    ) -> str:
        """Resolve the global person ID for a detection."""
        try:
            # Check if we have a person identity with global ID
            if person_identity and hasattr(person_identity, 'global_id'):
                global_id = person_identity.global_id
                
                # Store identity mapping
                local_key = f"{camera_id}_{detection.track_id}"
                self.identity_mapping[task_id][local_key] = global_id
                
                return global_id
            
            # Check identity mapping cache
            local_key = f"{camera_id}_{detection.track_id}"
            if local_key in self.identity_mapping[task_id]:
                return self.identity_mapping[task_id][local_key]
            
            # Generate new global ID
            global_id = f"person_{task_id}_{int(time.time())}_{detection.track_id}"
            self.identity_mapping[task_id][local_key] = global_id
            
            return global_id
            
        except Exception as e:
            logger.error(f"Error resolving person identity: {e}")
            # Fallback to basic ID generation
            return f"person_{camera_id}_{detection.track_id}_{int(time.time())}"
    
    async def _update_person_session(
        self,
        task_id: str,
        global_person_id: str,
        detection: Detection,
        camera_id: str,
        map_coordinate: Optional[Coordinate]
    ):
        """Update or create a person session with new detection."""
        try:
            # Get or create session
            if global_person_id not in self.active_sessions[task_id]:
                session = PersonSession(
                    global_person_id=global_person_id,
                    first_detected=detection.timestamp,
                    last_seen=detection.timestamp
                )
                self.active_sessions[task_id][global_person_id] = session
            else:
                session = self.active_sessions[task_id][global_person_id]
            
            # Add detection to session
            session.add_detection(detection, camera_id, map_coordinate)
            
            # Cache updated session data
            await self._cache_person_session(task_id, session)
            
        except Exception as e:
            logger.error(f"Error updating person session: {e}")
    
    async def _resolve_cross_camera_identities(
        self,
        task_id: str,
        aggregated_data: Dict[str, Any]
    ):
        """Resolve person identities across cameras and merge if necessary."""
        try:
            # Get all person sessions for this task
            sessions = self.active_sessions.get(task_id, {})
            
            # Look for potential merges based on appearance similarity and timing
            merge_candidates = []
            
            for person_id1, session1 in sessions.items():
                for person_id2, session2 in sessions.items():
                    if person_id1 >= person_id2:  # Avoid duplicate comparisons
                        continue
                    
                    # Check if sessions could be the same person
                    if await self._should_merge_sessions(session1, session2):
                        merge_candidates.append((person_id1, person_id2))
            
            # Perform merges
            for person_id1, person_id2 in merge_candidates:
                await self._merge_person_sessions(task_id, person_id1, person_id2)
                self.metrics.session_merges += 1
            
        except Exception as e:
            logger.error(f"Error resolving cross-camera identities: {e}")
    
    async def _should_merge_sessions(
        self,
        session1: PersonSession,
        session2: PersonSession
    ) -> bool:
        """Determine if two sessions should be merged as the same person."""
        try:
            # Time overlap check - sessions should have some temporal relationship
            time_gap = abs((session1.last_seen - session2.first_detected).total_seconds())
            if time_gap > 300:  # 5 minutes threshold
                return False
            
            # Camera transition check - should be reasonable camera transitions
            shared_cameras = session1.cameras_visited.intersection(session2.cameras_visited)
            if len(shared_cameras) > 0:
                # Same person shouldn't be in same camera simultaneously
                return False
            
            # Movement pattern similarity (simplified)
            if (len(session1.movement_history) > 0 and 
                len(session2.movement_history) > 0):
                
                # Check if movement directions are consistent
                last_pos1 = session1.movement_history[-1]
                first_pos2 = session2.movement_history[0]
                
                # Simple distance check
                distance = ((last_pos1.x - first_pos2.x) ** 2 + 
                           (last_pos1.y - first_pos2.y) ** 2) ** 0.5
                
                # Should be within reasonable movement distance
                max_distance = 50.0  # Configurable threshold
                if distance > max_distance:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking session merge: {e}")
            return False
    
    async def _merge_person_sessions(
        self,
        task_id: str,
        primary_person_id: str,
        secondary_person_id: str
    ):
        """Merge two person sessions into one."""
        try:
            sessions = self.active_sessions[task_id]
            
            if (primary_person_id not in sessions or 
                secondary_person_id not in sessions):
                return
            
            primary_session = sessions[primary_person_id]
            secondary_session = sessions[secondary_person_id]
            
            # Merge data into primary session
            primary_session.detection_history.extend(secondary_session.detection_history)
            primary_session.movement_history.extend(secondary_session.movement_history)
            primary_session.cameras_visited.update(secondary_session.cameras_visited)
            primary_session.total_detections += secondary_session.total_detections
            
            # Update timestamps
            primary_session.first_detected = min(
                primary_session.first_detected,
                secondary_session.first_detected
            )
            primary_session.last_seen = max(
                primary_session.last_seen,
                secondary_session.last_seen
            )
            
            # Sort histories by timestamp
            primary_session.detection_history.sort(key=lambda x: x['timestamp'])
            # Movement history sorting would need coordinate timestamps
            
            # Update identity mapping
            for local_key, global_id in self.identity_mapping[task_id].items():
                if global_id == secondary_person_id:
                    self.identity_mapping[task_id][local_key] = primary_person_id
            
            # Remove secondary session
            del sessions[secondary_person_id]
            
            # Cache updated primary session
            await self._cache_person_session(task_id, primary_session)
            
            logger.debug(
                f"Merged person sessions: {secondary_person_id} -> {primary_person_id}"
            )
            
        except Exception as e:
            logger.error(f"Error merging person sessions: {e}")
    
    # --- Data Retrieval Methods ---
    
    async def get_person_detailed_info(
        self,
        task_id: str,
        global_person_id: str
    ) -> Optional[PersonDetailedInfo]:
        """Get comprehensive detailed information for a person."""
        try:
            # Check active sessions first
            if (task_id in self.active_sessions and 
                global_person_id in self.active_sessions[task_id]):
                
                session = self.active_sessions[task_id][global_person_id]
                return await self._create_person_detailed_info(session)
            
            # Check cache
            cached_session = await self._get_cached_person_session(task_id, global_person_id)
            if cached_session:
                return await self._create_person_detailed_info(cached_session)
            
            # Check database
            person_data = await self.repository.get_person_tracking_data(global_person_id)
            if person_data:
                # Convert database data to PersonDetailedInfo format
                return await self._convert_db_data_to_detailed_info(person_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting person detailed info: {e}")
            return None
    
    async def _create_person_detailed_info(
        self,
        session: PersonSession
    ) -> PersonDetailedInfo:
        """Create PersonDetailedInfo from session data."""
        try:
            # Get current position
            current_position = None
            current_bbox = None
            current_camera = None
            
            if session.detection_history:
                latest_detection = session.detection_history[-1]
                current_camera = latest_detection['camera_id']
                current_bbox = latest_detection.get('bbox')
            
            if session.movement_history:
                latest_coord = session.movement_history[-1]
                current_position = latest_coord
            
            # Build position history
            position_history = []
            for detection in session.detection_history[-50:]:  # Last 50 positions
                position_record = {
                    'timestamp': detection['timestamp'].isoformat(),
                    'camera': detection['camera_id'],
                    'bbox': detection['bbox']
                }
                if 'map_coordinate' in detection:
                    position_record['map_coords'] = [
                        detection['map_coordinate']['x'],
                        detection['map_coordinate']['y']
                    ]
                position_history.append(position_record)
            
            # Build camera transitions
            camera_transitions = []
            last_camera = None
            for detection in session.detection_history:
                current_cam = detection['camera_id']
                if last_camera and last_camera != current_cam:
                    camera_transitions.append({
                        'from_camera': last_camera,
                        'to_camera': current_cam,
                        'timestamp': detection['timestamp'].isoformat(),
                        'confidence': detection.get('confidence', 1.0)
                    })
                last_camera = current_cam
            
            # Calculate movement metrics
            movement_metrics = session.calculate_movement_metrics()
            
            # Build trajectory path
            trajectory_path = []
            for coord in session.movement_history[-100:]:  # Last 100 coordinates
                trajectory_path.append(coord)
            
            return PersonDetailedInfo(
                global_id=session.global_person_id,
                first_detected=session.first_detected,
                last_seen=session.last_seen,
                tracking_duration=(session.last_seen - session.first_detected).total_seconds(),
                current_camera=current_camera or "unknown",
                current_position=current_position,
                current_bbox=current_bbox,
                position_history=position_history,
                camera_transitions=camera_transitions,
                movement_metrics=movement_metrics,
                behavior_patterns={
                    'cameras_visited': list(session.cameras_visited),
                    'total_detections': session.total_detections,
                    'session_duration': (session.last_seen - session.first_detected).total_seconds()
                },
                trajectory_path=trajectory_path
            )
            
        except Exception as e:
            logger.error(f"Error creating person detailed info: {e}")
            return PersonDetailedInfo(
                global_id=session.global_person_id,
                first_detected=session.first_detected,
                last_seen=session.last_seen,
                tracking_duration=0.0,
                current_camera="unknown",
                movement_metrics=PersonMovementMetrics()
            )
    
    async def get_active_persons(self, task_id: str) -> List[Dict[str, Any]]:
        """Get list of currently active persons for a task."""
        try:
            active_persons = []
            sessions = self.active_sessions.get(task_id, {})
            
            for person_id, session in sessions.items():
                if session.session_active and not session.is_session_expired():
                    person_info = {
                        'global_person_id': person_id,
                        'first_detected': session.first_detected.isoformat(),
                        'last_seen': session.last_seen.isoformat(),
                        'total_detections': session.total_detections,
                        'cameras_visited': list(session.cameras_visited),
                        'tracking_duration': (session.last_seen - session.first_detected).total_seconds(),
                        'identity_confidence': session.identity_confidence
                    }
                    active_persons.append(person_info)
            
            return active_persons
            
        except Exception as e:
            logger.error(f"Error getting active persons: {e}")
            return []
    
    # --- Cache Management ---
    
    async def _cache_person_session(self, task_id: str, session: PersonSession):
        """Cache person session data."""
        try:
            cache_key = f"session_{task_id}_{session.global_person_id}"
            session_data = {
                'global_person_id': session.global_person_id,
                'first_detected': session.first_detected.isoformat(),
                'last_seen': session.last_seen.isoformat(),
                'total_detections': session.total_detections,
                'cameras_visited': list(session.cameras_visited),
                'identity_confidence': session.identity_confidence,
                'session_active': session.session_active,
                'detection_count': len(session.detection_history),
                'movement_count': len(session.movement_history)
            }
            
            await self.cache.set_person_data(cache_key, session_data, ttl=3600)
            
        except Exception as e:
            logger.error(f"Error caching person session: {e}")
    
    async def _get_cached_person_session(
        self,
        task_id: str,
        global_person_id: str
    ) -> Optional[PersonSession]:
        """Retrieve cached person session."""
        try:
            cache_key = f"session_{task_id}_{global_person_id}"
            cached_data = await self.cache.get_person_data(cache_key)
            
            if not cached_data:
                return None
            
            # Note: This is a simplified version - full reconstruction would need
            # to retrieve detailed detection and movement history
            session = PersonSession(
                global_person_id=cached_data['global_person_id'],
                first_detected=datetime.fromisoformat(cached_data['first_detected']),
                last_seen=datetime.fromisoformat(cached_data['last_seen'])
            )
            session.total_detections = cached_data.get('total_detections', 0)
            session.cameras_visited = set(cached_data.get('cameras_visited', []))
            session.identity_confidence = cached_data.get('identity_confidence', 1.0)
            session.session_active = cached_data.get('session_active', True)
            
            return session
            
        except Exception as e:
            logger.error(f"Error getting cached person session: {e}")
            return None
    
    # --- Background Tasks ---
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                expired_count = 0
                
                for task_id in list(self.active_sessions.keys()):
                    sessions = self.active_sessions[task_id]
                    expired_sessions = []
                    
                    for person_id, session in sessions.items():
                        if session.is_session_expired(self.session_timeout_minutes):
                            expired_sessions.append(person_id)
                    
                    # Archive expired sessions
                    for person_id in expired_sessions:
                        await self._archive_session(task_id, sessions[person_id])
                        del sessions[person_id]
                        expired_count += 1
                    
                    # Clean up empty task sessions
                    if not sessions:
                        del self.active_sessions[task_id]
                        if task_id in self.identity_mapping:
                            del self.identity_mapping[task_id]
                
                if expired_count > 0:
                    self.metrics.expired_sessions += expired_count
                    logger.debug(f"Cleaned up {expired_count} expired sessions")
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _archive_session(self, task_id: str, session: PersonSession):
        """Archive expired session to database."""
        try:
            # Convert session to database format and store
            session_data = {
                'task_id': task_id,
                'global_person_id': session.global_person_id,
                'first_detected': session.first_detected,
                'last_seen': session.last_seen,
                'total_detections': session.total_detections,
                'cameras_visited': list(session.cameras_visited),
                'detection_history': session.detection_history[-100:],  # Keep last 100
                'movement_history': [
                    {'x': coord.x, 'y': coord.y} 
                    for coord in session.movement_history[-100:]
                ]
            }
            
            await self.repository.store_person_session(session_data)
            
        except Exception as e:
            logger.error(f"Error archiving session: {e}")
    
    # --- Utility Methods ---
    
    def _update_metrics(self, processing_time_ms: float):
        """Update aggregation metrics."""
        self.metrics.processing_time_ms = processing_time_ms
        self.metrics.last_update = datetime.utcnow()
        
        # Update counters
        total_sessions = sum(
            len(sessions) for sessions in self.active_sessions.values()
        )
        self.metrics.active_sessions = total_sessions
        
        # Add to performance history
        self.performance_history.append({
            'timestamp': datetime.utcnow(),
            'processing_time_ms': processing_time_ms,
            'active_sessions': total_sessions
        })
    
    def _get_session_statistics(self, task_id: str) -> Dict[str, Any]:
        """Get statistics for a specific task's sessions."""
        sessions = self.active_sessions.get(task_id, {})
        
        if not sessions:
            return {
                'total_persons': 0,
                'active_persons': 0,
                'total_detections': 0,
                'cameras_covered': 0,
                'average_tracking_duration': 0.0
            }
        
        total_detections = sum(s.total_detections for s in sessions.values())
        all_cameras = set()
        for session in sessions.values():
            all_cameras.update(session.cameras_visited)
        
        durations = [
            (s.last_seen - s.first_detected).total_seconds()
            for s in sessions.values()
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            'total_persons': len(sessions),
            'active_persons': len([s for s in sessions.values() if s.session_active]),
            'total_detections': total_detections,
            'cameras_covered': len(all_cameras),
            'average_tracking_duration': avg_duration
        }
    
    async def _generate_person_summaries(self, task_id: str) -> Dict[str, Dict[str, Any]]:
        """Generate summaries of all persons in task."""
        try:
            summaries = {}
            sessions = self.active_sessions.get(task_id, {})
            
            for person_id, session in sessions.items():
                movement_metrics = session.calculate_movement_metrics()
                
                summaries[person_id] = {
                    'global_person_id': person_id,
                    'tracking_status': 'active' if session.session_active else 'inactive',
                    'first_detected': session.first_detected.isoformat(),
                    'last_seen': session.last_seen.isoformat(),
                    'total_detections': session.total_detections,
                    'cameras_visited': list(session.cameras_visited),
                    'movement_metrics': movement_metrics.dict(),
                    'identity_confidence': session.identity_confidence
                }
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error generating person summaries: {e}")
            return {}
    
    # --- Service Status and Management ---
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        total_sessions = sum(len(sessions) for sessions in self.active_sessions.values())
        total_identities = sum(len(mapping) for mapping in self.identity_mapping.values())
        
        return {
            'service_running': self._running,
            'active_tasks': len(self.active_sessions),
            'total_active_sessions': total_sessions,
            'total_identity_mappings': total_identities,
            'metrics': self.metrics.to_dict(),
            'configuration': {
                'session_timeout_minutes': self.session_timeout_minutes,
                'max_sessions_per_task': self.max_sessions_per_task,
                'identity_confidence_threshold': self.identity_confidence_threshold,
                'cross_camera_merge_threshold': self.cross_camera_merge_threshold
            },
            'performance_samples': len(self.performance_history)
        }
    
    async def cleanup_task_data(self, task_id: str):
        """Clean up all data for a completed task."""
        try:
            # Archive active sessions
            if task_id in self.active_sessions:
                for session in self.active_sessions[task_id].values():
                    await self._archive_session(task_id, session)
                del self.active_sessions[task_id]
            
            # Clear identity mapping
            if task_id in self.identity_mapping:
                del self.identity_mapping[task_id]
            
            # Clear cached data
            await self._clear_task_cache(task_id)
            
            logger.info(f"Cleaned up aggregation data for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up task data: {e}")
    
    async def _clear_task_cache(self, task_id: str):
        """Clear all cached data for a task."""
        try:
            # This would clear all cache entries for the task
            # Implementation depends on cache structure
            cache_pattern = f"session_{task_id}_*"
            await self.cache.clear_pattern(cache_pattern)
            
        except Exception as e:
            logger.error(f"Error clearing task cache: {e}")


# Global service instance
_aggregation_service: Optional[TrackingDataAggregationService] = None


def get_aggregation_service() -> Optional[TrackingDataAggregationService]:
    """Get the global aggregation service instance."""
    return _aggregation_service


def initialize_aggregation_service(
    tracking_cache: TrackingCache,
    tracking_repository: TrackingRepository
) -> TrackingDataAggregationService:
    """Initialize the global aggregation service."""
    global _aggregation_service
    if _aggregation_service is None:
        _aggregation_service = TrackingDataAggregationService(
            tracking_cache, tracking_repository
        )
    return _aggregation_service