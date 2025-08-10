"""
Historical Playback Infrastructure Service

Comprehensive playback system for historical video frames and tracking data:
- Synchronized playback of multi-camera video feeds with tracking overlays
- Temporal navigation and seeking capabilities
- Variable playback speeds and frame stepping
- Timeline synchronization across cameras
- Historical data reconstruction for specific time points
- Efficient frame caching and streaming
"""

import asyncio
import logging
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncGenerator
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
import threading
from pathlib import Path

from app.services.historical_data_service import (
    HistoricalDataService,
    HistoricalDataPoint,
    TimeRange,
    HistoricalQueryFilter
)
from app.services.temporal_query_engine import TemporalQueryEngine, TimeGranularity
from app.domains.detection.entities.detection import Detection
from app.domains.mapping.entities.coordinate import Coordinate

logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """Playback state enumeration."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"
    BUFFERING = "buffering"


class PlaybackSpeed(Enum):
    """Playback speed options."""
    QUARTER = 0.25
    HALF = 0.5
    NORMAL = 1.0
    DOUBLE = 2.0
    QUAD = 4.0
    OCTO = 8.0


@dataclass
class PlaybackFrame:
    """Single frame in playback sequence."""
    timestamp: datetime
    frame_index: int
    camera_frames: Dict[str, bytes] = field(default_factory=dict)  # camera_id -> frame_data
    tracking_data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # camera_id -> tracks
    overlays_rendered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'frame_index': self.frame_index,
            'camera_frames': {
                camera_id: base64.b64encode(frame_data).decode()
                for camera_id, frame_data in self.camera_frames.items()
            },
            'tracking_data': self.tracking_data,
            'overlays_rendered': self.overlays_rendered
        }


@dataclass
class PlaybackSession:
    """Playback session configuration and state."""
    session_id: str
    environment_id: str
    time_range: TimeRange
    camera_ids: List[str]
    
    # Playback control
    state: PlaybackState = PlaybackState.STOPPED
    speed: PlaybackSpeed = PlaybackSpeed.NORMAL
    current_timestamp: Optional[datetime] = None
    current_frame_index: int = 0
    
    # Configuration
    target_fps: float = 25.0
    enable_overlays: bool = True
    frame_quality: int = 85
    
    # Buffer management
    buffer_frames: deque = field(default_factory=lambda: deque(maxlen=100))
    preload_seconds: int = 30
    
    # Statistics
    frames_played: int = 0
    seek_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def duration_seconds(self) -> float:
        """Get total playback duration."""
        return self.time_range.duration_seconds
    
    @property
    def progress_ratio(self) -> float:
        """Get playback progress as ratio (0.0 to 1.0)."""
        if not self.current_timestamp or self.duration_seconds == 0:
            return 0.0
        
        elapsed = (self.current_timestamp - self.time_range.start_time).total_seconds()
        return max(0.0, min(1.0, elapsed / self.duration_seconds))
    
    def get_timestamp_at_ratio(self, ratio: float) -> datetime:
        """Get timestamp at specific progress ratio."""
        ratio = max(0.0, min(1.0, ratio))
        elapsed_seconds = ratio * self.duration_seconds
        return self.time_range.start_time + timedelta(seconds=elapsed_seconds)


class PlaybackService:
    """Comprehensive service for historical video playback."""
    
    def __init__(
        self,
        historical_data_service: HistoricalDataService,
        temporal_query_engine: TemporalQueryEngine,
        video_storage_path: Optional[str] = None
    ):
        self.historical_service = historical_data_service
        self.query_engine = temporal_query_engine
        
        # Video storage configuration
        self.video_storage_path = Path(video_storage_path or "data/videos")
        self.video_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Active playback sessions
        self.active_sessions: Dict[str, PlaybackSession] = {}
        
        # Frame cache for efficient access
        self.frame_cache: Dict[str, PlaybackFrame] = {}
        self.max_cache_frames = 1000
        
        # Background processing
        self._playback_tasks: Dict[str, asyncio.Task] = {}
        self._preload_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
        # Performance metrics
        self.playback_stats = {
            'active_sessions': 0,
            'frames_served': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_frame_generation_ms': 0.0
        }
        
        logger.info("PlaybackService initialized")
    
    async def start_service(self):
        """Start the playback service."""
        if self._running:
            return
        
        self._running = True
        logger.info("PlaybackService started")
    
    async def stop_service(self):
        """Stop the playback service and cleanup."""
        self._running = False
        
        # Stop all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.stop_playback_session(session_id)
        
        logger.info("PlaybackService stopped")
    
    # --- Session Management ---
    
    async def create_playback_session(
        self,
        environment_id: str,
        start_time: datetime,
        end_time: datetime,
        camera_ids: List[str],
        target_fps: float = 25.0,
        enable_overlays: bool = True,
        frame_quality: int = 85
    ) -> str:
        """Create new playback session."""
        try:
            # Generate session ID
            session_id = f"playback_{int(time.time())}_{len(self.active_sessions)}"
            
            # Validate time range
            time_range = TimeRange(start_time, end_time)
            
            # Create session
            session = PlaybackSession(
                session_id=session_id,
                environment_id=environment_id,
                time_range=time_range,
                camera_ids=camera_ids,
                target_fps=target_fps,
                enable_overlays=enable_overlays,
                frame_quality=frame_quality
            )
            
            # Store session
            self.active_sessions[session_id] = session
            
            # Start preloading frames
            await self._start_frame_preloading(session_id)
            
            # Update statistics
            self.playback_stats['active_sessions'] = len(self.active_sessions)
            
            logger.info(
                f"Created playback session {session_id} for environment {environment_id} "
                f"from {start_time} to {end_time}"
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating playback session: {e}")
            raise
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get playback session information."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            return {
                'session_id': session_id,
                'environment_id': session.environment_id,
                'time_range': {
                    'start': session.time_range.start_time.isoformat(),
                    'end': session.time_range.end_time.isoformat(),
                    'duration_seconds': session.duration_seconds
                },
                'camera_ids': session.camera_ids,
                'state': session.state.value,
                'speed': session.speed.value,
                'current_timestamp': session.current_timestamp.isoformat() if session.current_timestamp else None,
                'current_frame_index': session.current_frame_index,
                'progress_ratio': session.progress_ratio,
                'configuration': {
                    'target_fps': session.target_fps,
                    'enable_overlays': session.enable_overlays,
                    'frame_quality': session.frame_quality
                },
                'statistics': {
                    'frames_played': session.frames_played,
                    'seek_count': session.seek_count,
                    'buffer_frames': len(session.buffer_frames),
                    'session_duration': (datetime.utcnow() - session.created_at).total_seconds()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            raise
    
    async def stop_playback_session(self, session_id: str):
        """Stop and cleanup playback session."""
        try:
            if session_id not in self.active_sessions:
                return
            
            # Stop playback task
            if session_id in self._playback_tasks:
                self._playback_tasks[session_id].cancel()
                try:
                    await self._playback_tasks[session_id]
                except asyncio.CancelledError:
                    pass
                del self._playback_tasks[session_id]
            
            # Stop preload task
            if session_id in self._preload_tasks:
                self._preload_tasks[session_id].cancel()
                try:
                    await self._preload_tasks[session_id]
                except asyncio.CancelledError:
                    pass
                del self._preload_tasks[session_id]
            
            # Remove session
            del self.active_sessions[session_id]
            
            # Update statistics
            self.playback_stats['active_sessions'] = len(self.active_sessions)
            
            logger.info(f"Stopped playback session {session_id}")
            
        except Exception as e:
            logger.error(f"Error stopping playback session: {e}")
    
    # --- Playback Control ---
    
    async def start_playback(self, session_id: str):
        """Start playback for session."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            if session.state == PlaybackState.PLAYING:
                return
            
            # Set initial timestamp if not set
            if session.current_timestamp is None:
                session.current_timestamp = session.time_range.start_time
                session.current_frame_index = 0
            
            # Update state
            session.state = PlaybackState.PLAYING
            session.last_activity = datetime.utcnow()
            
            # Start playback task
            if session_id in self._playback_tasks:
                self._playback_tasks[session_id].cancel()
            
            self._playback_tasks[session_id] = asyncio.create_task(
                self._playback_loop(session_id)
            )
            
            logger.info(f"Started playback for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            raise
    
    async def pause_playback(self, session_id: str):
        """Pause playback for session."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            if session.state != PlaybackState.PLAYING:
                return
            
            # Update state
            session.state = PlaybackState.PAUSED
            session.last_activity = datetime.utcnow()
            
            # Stop playback task
            if session_id in self._playback_tasks:
                self._playback_tasks[session_id].cancel()
                del self._playback_tasks[session_id]
            
            logger.info(f"Paused playback for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error pausing playback: {e}")
            raise
    
    async def seek_to_timestamp(self, session_id: str, timestamp: datetime):
        """Seek to specific timestamp in playback."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Validate timestamp
            if not session.time_range.contains(timestamp):
                raise ValueError("Timestamp outside playback range")
            
            # Update session state
            was_playing = session.state == PlaybackState.PLAYING
            session.state = PlaybackState.SEEKING
            session.current_timestamp = timestamp
            session.seek_count += 1
            session.last_activity = datetime.utcnow()
            
            # Calculate new frame index
            elapsed_seconds = (timestamp - session.time_range.start_time).total_seconds()
            session.current_frame_index = int(elapsed_seconds * session.target_fps)
            
            # Clear buffer and reload
            session.buffer_frames.clear()
            await self._preload_frames_around_timestamp(session_id, timestamp)
            
            # Resume if was playing
            if was_playing:
                session.state = PlaybackState.PLAYING
                if session_id in self._playback_tasks:
                    self._playback_tasks[session_id].cancel()
                
                self._playback_tasks[session_id] = asyncio.create_task(
                    self._playback_loop(session_id)
                )
            else:
                session.state = PlaybackState.PAUSED
            
            logger.info(f"Seeked session {session_id} to {timestamp}")
            
        except Exception as e:
            logger.error(f"Error seeking playback: {e}")
            raise
    
    async def seek_to_ratio(self, session_id: str, ratio: float):
        """Seek to specific progress ratio (0.0 to 1.0)."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            target_timestamp = session.get_timestamp_at_ratio(ratio)
            
            await self.seek_to_timestamp(session_id, target_timestamp)
            
        except Exception as e:
            logger.error(f"Error seeking to ratio: {e}")
            raise
    
    async def set_playback_speed(self, session_id: str, speed: float):
        """Set playback speed."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Find closest speed enum value
            speed_options = [s.value for s in PlaybackSpeed]
            closest_speed = min(speed_options, key=lambda x: abs(x - speed))
            
            session.speed = PlaybackSpeed(closest_speed)
            session.last_activity = datetime.utcnow()
            
            logger.info(f"Set playback speed for session {session_id} to {closest_speed}x")
            
        except Exception as e:
            logger.error(f"Error setting playback speed: {e}")
            raise
    
    # --- Frame Generation and Streaming ---
    
    async def get_current_frame(self, session_id: str) -> Optional[PlaybackFrame]:
        """Get current frame for session."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            if session.current_timestamp is None:
                return None
            
            # Check buffer first
            for buffered_frame in session.buffer_frames:
                if abs((buffered_frame.timestamp - session.current_timestamp).total_seconds()) < 0.1:
                    return buffered_frame
            
            # Generate frame if not in buffer
            frame = await self._generate_frame_at_timestamp(
                session_id, session.current_timestamp
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error getting current frame: {e}")
            return None
    
    async def get_frame_at_timestamp(
        self,
        session_id: str,
        timestamp: datetime
    ) -> Optional[PlaybackFrame]:
        """Get frame at specific timestamp."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            if not session.time_range.contains(timestamp):
                return None
            
            return await self._generate_frame_at_timestamp(session_id, timestamp)
            
        except Exception as e:
            logger.error(f"Error getting frame at timestamp: {e}")
            return None
    
    async def stream_playback_frames(
        self,
        session_id: str
    ) -> AsyncGenerator[PlaybackFrame, None]:
        """Stream playback frames as async generator."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            while (session.state == PlaybackState.PLAYING and
                   session.current_timestamp and
                   session.current_timestamp <= session.time_range.end_time):
                
                # Get current frame
                frame = await self.get_current_frame(session_id)
                if frame:
                    yield frame
                
                # Advance to next frame
                frame_interval = 1.0 / session.target_fps / session.speed.value
                await asyncio.sleep(frame_interval)
                
                # Update current timestamp
                session.current_timestamp += timedelta(seconds=1.0 / session.target_fps)
                session.current_frame_index += 1
                session.frames_played += 1
                
        except Exception as e:
            logger.error(f"Error streaming playback frames: {e}")
    
    async def _generate_frame_at_timestamp(
        self,
        session_id: str,
        timestamp: datetime
    ) -> Optional[PlaybackFrame]:
        """Generate complete frame at specific timestamp."""
        start_time = time.time()
        
        try:
            session = self.active_sessions[session_id]
            
            # Create cache key
            cache_key = f"{session_id}_{timestamp.isoformat()}"
            
            # Check frame cache
            if cache_key in self.frame_cache:
                self.playback_stats['cache_hits'] += 1
                return self.frame_cache[cache_key]
            
            # Query historical data around timestamp
            query_filter = HistoricalQueryFilter(
                time_range=TimeRange(
                    timestamp - timedelta(seconds=1),
                    timestamp + timedelta(seconds=1)
                ),
                environment_id=session.environment_id,
                camera_ids=session.camera_ids
            )
            
            historical_data = await self.historical_service.query_historical_data(query_filter)
            
            # Find closest data points for each camera
            camera_data = defaultdict(list)
            for data_point in historical_data:
                time_diff = abs((data_point.timestamp - timestamp).total_seconds())
                if time_diff <= 0.5:  # Within 0.5 seconds
                    camera_data[data_point.camera_id].append((time_diff, data_point))
            
            # Sort by time difference and take closest
            for camera_id in camera_data:
                camera_data[camera_id].sort(key=lambda x: x[0])
            
            # Generate frame data
            frame = PlaybackFrame(
                timestamp=timestamp,
                frame_index=session.current_frame_index
            )
            
            # Process each camera
            for camera_id in session.camera_ids:
                camera_frame_data, camera_tracks = await self._generate_camera_frame(
                    session, camera_id, timestamp, camera_data.get(camera_id, [])
                )
                
                if camera_frame_data is not None:
                    frame.camera_frames[camera_id] = camera_frame_data
                    frame.tracking_data[camera_id] = camera_tracks
            
            frame.overlays_rendered = session.enable_overlays
            
            # Cache the frame
            if len(self.frame_cache) >= self.max_cache_frames:
                self._clean_frame_cache()
            
            self.frame_cache[cache_key] = frame
            
            # Update statistics
            generation_time = (time.time() - start_time) * 1000
            self._update_frame_generation_metrics(generation_time)
            self.playback_stats['cache_misses'] += 1
            self.playback_stats['frames_served'] += 1
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating frame at timestamp: {e}")
            return None
    
    async def _generate_camera_frame(
        self,
        session: PlaybackSession,
        camera_id: str,
        timestamp: datetime,
        camera_data_points: List[Tuple[float, HistoricalDataPoint]]
    ) -> Tuple[Optional[bytes], List[Dict[str, Any]]]:
        """Generate frame data and tracks for specific camera."""
        try:
            # Get base frame image (this would normally come from video storage)
            base_frame = await self._get_base_frame(
                session.environment_id, camera_id, timestamp
            )
            
            if base_frame is None:
                return None, []
            
            # Extract tracking data
            tracks = []
            if camera_data_points:
                for _, data_point in camera_data_points:
                    track_data = {
                        'global_person_id': data_point.global_person_id,
                        'track_id': data_point.detection.track_id,
                        'bbox_xyxy': data_point.detection.bbox,
                        'confidence': data_point.detection.confidence,
                        'timestamp': data_point.timestamp.isoformat()
                    }
                    
                    if data_point.coordinates:
                        track_data['map_coords'] = [data_point.coordinates.x, data_point.coordinates.y]
                    
                    tracks.append(track_data)
            
            # Render overlays if enabled
            if session.enable_overlays and tracks:
                base_frame = await self._render_overlays(base_frame, tracks)
            
            # Encode frame
            frame_data = await self._encode_frame(base_frame, session.frame_quality)
            
            return frame_data, tracks
            
        except Exception as e:
            logger.error(f"Error generating camera frame: {e}")
            return None, []
    
    async def _get_base_frame(
        self,
        environment_id: str,
        camera_id: str,
        timestamp: datetime
    ) -> Optional[np.ndarray]:
        """Get base frame image from video storage."""
        try:
            # This is a placeholder implementation
            # In practice, you would:
            # 1. Look up the video file for the given timestamp
            # 2. Extract the frame at the specific time
            # 3. Return the frame as numpy array
            
            # For now, create a placeholder frame
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Add timestamp overlay
            cv2.putText(
                frame,
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # Add camera ID
            cv2.putText(
                frame,
                f"Camera: {camera_id}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error getting base frame: {e}")
            return None
    
    async def _render_overlays(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Render tracking overlays on frame."""
        try:
            overlay_frame = frame.copy()
            
            for track in tracks:
                bbox = track.get('bbox_xyxy', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox
                    
                    # Draw bounding box
                    cv2.rectangle(
                        overlay_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),  # Green
                        2
                    )
                    
                    # Draw person ID
                    person_id = track.get('global_person_id', 'Unknown')
                    cv2.putText(
                        overlay_frame,
                        str(person_id)[-8:],  # Last 8 characters
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Draw confidence
                    confidence = track.get('confidence', 0.0)
                    cv2.putText(
                        overlay_frame,
                        f"{confidence:.2f}",
                        (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Error rendering overlays: {e}")
            return frame
    
    async def _encode_frame(self, frame: np.ndarray, quality: int) -> bytes:
        """Encode frame to JPEG bytes."""
        try:
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
            
            if success:
                return encoded_frame.tobytes()
            else:
                raise Exception("Frame encoding failed")
            
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            raise
    
    # --- Background Processing ---
    
    async def _playback_loop(self, session_id: str):
        """Background playback loop for session."""
        try:
            session = self.active_sessions[session_id]
            
            while (session.state == PlaybackState.PLAYING and
                   session.current_timestamp and
                   session.current_timestamp <= session.time_range.end_time):
                
                # Calculate frame interval based on speed
                frame_interval = 1.0 / session.target_fps / session.speed.value
                
                # Advance timestamp
                session.current_timestamp += timedelta(seconds=1.0 / session.target_fps)
                session.current_frame_index += 1
                session.frames_played += 1
                session.last_activity = datetime.utcnow()
                
                # Wait for next frame
                await asyncio.sleep(frame_interval)
            
            # End of playback reached
            if session.current_timestamp and session.current_timestamp > session.time_range.end_time:
                session.state = PlaybackState.STOPPED
                logger.info(f"Playback completed for session {session_id}")
            
        except asyncio.CancelledError:
            logger.debug(f"Playback loop cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Error in playback loop: {e}")
            session.state = PlaybackState.STOPPED
    
    async def _start_frame_preloading(self, session_id: str):
        """Start frame preloading task for session."""
        try:
            if session_id in self._preload_tasks:
                self._preload_tasks[session_id].cancel()
            
            self._preload_tasks[session_id] = asyncio.create_task(
                self._preload_frames(session_id)
            )
            
        except Exception as e:
            logger.error(f"Error starting frame preloading: {e}")
    
    async def _preload_frames(self, session_id: str):
        """Preload frames for smooth playback."""
        try:
            session = self.active_sessions[session_id]
            
            # Start preloading from beginning of time range
            current_preload_time = session.time_range.start_time
            
            while (current_preload_time <= session.time_range.end_time and
                   session_id in self.active_sessions):
                
                # Check if we need more frames in buffer
                if len(session.buffer_frames) < 50:  # Keep 50 frames buffered
                    frame = await self._generate_frame_at_timestamp(session_id, current_preload_time)
                    
                    if frame:
                        session.buffer_frames.append(frame)
                    
                    # Advance preload time
                    current_preload_time += timedelta(seconds=1.0 / session.target_fps)
                
                # Don't preload too aggressively
                await asyncio.sleep(0.1)
            
        except asyncio.CancelledError:
            logger.debug(f"Frame preloading cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Error preloading frames: {e}")
    
    async def _preload_frames_around_timestamp(self, session_id: str, timestamp: datetime):
        """Preload frames around specific timestamp for seeking."""
        try:
            session = self.active_sessions[session_id]
            
            # Preload frames ±5 seconds around target timestamp
            start_preload = timestamp - timedelta(seconds=5)
            end_preload = timestamp + timedelta(seconds=5)
            
            # Ensure within session range
            start_preload = max(start_preload, session.time_range.start_time)
            end_preload = min(end_preload, session.time_range.end_time)
            
            current_time = start_preload
            
            while current_time <= end_preload:
                frame = await self._generate_frame_at_timestamp(session_id, current_time)
                
                if frame and len(session.buffer_frames) < session.buffer_frames.maxlen:
                    session.buffer_frames.append(frame)
                
                current_time += timedelta(seconds=1.0 / session.target_fps)
            
        except Exception as e:
            logger.error(f"Error preloading frames around timestamp: {e}")
    
    # --- Utility Methods ---
    
    def _clean_frame_cache(self):
        """Clean frame cache to prevent memory bloat."""
        try:
            # Remove oldest 25% of cached frames
            cache_items = list(self.frame_cache.items())
            remove_count = len(cache_items) // 4
            
            for key, _ in cache_items[:remove_count]:
                del self.frame_cache[key]
            
        except Exception as e:
            logger.error(f"Error cleaning frame cache: {e}")
    
    def _update_frame_generation_metrics(self, generation_time_ms: float):
        """Update frame generation performance metrics."""
        current_avg = self.playback_stats['avg_frame_generation_ms']
        frames_served = self.playback_stats['frames_served']
        
        if frames_served > 0:
            self.playback_stats['avg_frame_generation_ms'] = (
                (current_avg * (frames_served - 1) + generation_time_ms) / frames_served
            )
        else:
            self.playback_stats['avg_frame_generation_ms'] = generation_time_ms
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'service_name': 'PlaybackService',
            'running': self._running,
            'active_sessions': len(self.active_sessions),
            'playback_tasks': len(self._playback_tasks),
            'preload_tasks': len(self._preload_tasks),
            'frame_cache_size': len(self.frame_cache),
            'video_storage_path': str(self.video_storage_path),
            'statistics': self.playback_stats.copy(),
            'session_details': {
                session_id: {
                    'state': session.state.value,
                    'speed': session.speed.value,
                    'progress': session.progress_ratio,
                    'frames_played': session.frames_played
                }
                for session_id, session in self.active_sessions.items()
            }
        }
    
    async def cleanup_expired_sessions(self):
        """Clean up expired or inactive sessions."""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                # Consider session expired if inactive for more than 1 hour
                if (current_time - session.last_activity).total_seconds() > 3600:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self.stop_playback_session(session_id)
                logger.info(f"Cleaned up expired session {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")


# Global service instance
_playback_service: Optional[PlaybackService] = None


def get_playback_service() -> Optional[PlaybackService]:
    """Get the global playback service instance."""
    return _playback_service


def initialize_playback_service(
    historical_data_service: HistoricalDataService,
    temporal_query_engine: TemporalQueryEngine,
    video_storage_path: Optional[str] = None
) -> PlaybackService:
    """Initialize the global playback service."""
    global _playback_service
    if _playback_service is None:
        _playback_service = PlaybackService(
            historical_data_service, temporal_query_engine, video_storage_path
        )
    return _playback_service