"""
Frontend Visualization Data Schemas

Comprehensive data structures for frontend integration including:
- Enhanced tracking updates with visualization data
- Focus tracking schemas
- Real-time analytics schemas
- Multi-camera synchronization schemas
- Interactive control schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime
from enum import Enum
import uuid


# --- Enums for Type Safety ---

class VisualizationMode(str, Enum):
    """Visualization display modes."""
    LIVE = "live"
    PLAYBACK = "playback"
    ANALYSIS = "analysis"


class FocusTrackMode(str, Enum):
    """Focus tracking modes."""
    SINGLE_PERSON = "single_person"
    MULTI_PERSON = "multi_person"
    CAMERA_FOCUS = "camera_focus"


class OverlayType(str, Enum):
    """Types of overlays for visualization."""
    BOUNDING_BOX = "bounding_box"
    PERSON_ID = "person_id"
    CONFIDENCE = "confidence"
    TRAJECTORY = "trajectory"
    ZONE_HIGHLIGHT = "zone_highlight"


class PlaybackState(str, Enum):
    """Playback control states."""
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    SEEKING = "seeking"


class AnalyticsType(str, Enum):
    """Types of real-time analytics."""
    OCCUPANCY = "occupancy"
    MOVEMENT = "movement"
    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"


# --- Core Visualization Data Structures ---

class BoundingBoxData(BaseModel):
    """Enhanced bounding box with visualization data."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    
    # Visualization properties
    color: Optional[str] = Field(None, description="Hex color for bounding box")
    thickness: int = Field(2, description="Line thickness for drawing")
    style: str = Field("solid", description="Line style (solid, dashed, dotted)")
    highlight: bool = Field(False, description="Whether this box is highlighted")
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height


class MapCoordinates(BaseModel):
    """Map coordinate system data."""
    x: float = Field(..., description="X coordinate on unified map")
    y: float = Field(..., description="Y coordinate on unified map")
    z: Optional[float] = Field(None, description="Z coordinate if 3D mapping")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Mapping confidence")
    
    # Trajectory data
    velocity_x: Optional[float] = Field(None, description="X velocity component")
    velocity_y: Optional[float] = Field(None, description="Y velocity component")
    speed: Optional[float] = Field(None, description="Overall speed")
    direction: Optional[float] = Field(None, description="Movement direction in degrees")


class PersonAppearanceData(BaseModel):
    """Person appearance and visual data."""
    cropped_image_base64: Optional[str] = Field(None, description="Base64 cropped person image")
    image_format: str = Field("jpeg", description="Image format")
    image_quality: int = Field(95, ge=1, le=100, description="Image quality")
    
    # Appearance features
    dominant_colors: Optional[List[str]] = Field(None, description="Dominant clothing colors")
    height_estimate: Optional[float] = Field(None, description="Estimated height in pixels")
    appearance_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Appearance data confidence")


class PersonMovementMetrics(BaseModel):
    """Person movement analysis metrics."""
    total_distance: float = Field(0.0, description="Total distance traveled")
    average_speed: float = Field(0.0, description="Average movement speed")
    dwell_time: float = Field(0.0, description="Time spent in current location")
    direction_changes: int = Field(0, description="Number of direction changes")
    trajectory_smoothness: float = Field(1.0, ge=0.0, le=1.0, description="Trajectory smoothness score")


class EnhancedPersonTrack(BaseModel):
    """Enhanced person tracking data with visualization features."""
    track_id: int
    global_id: str = Field(..., description="Global person identifier")
    
    # Detection data
    bbox: BoundingBoxData
    map_coords: Optional[MapCoordinates] = None
    
    # Timing data
    detection_time: datetime
    last_seen_time: datetime
    tracking_duration: float = Field(0.0, description="Duration in seconds")
    
    # State data
    is_focused: bool = Field(False, description="Whether this person is focused")
    is_active: bool = Field(True, description="Whether this person is currently active")
    current_camera: str = Field(..., description="Camera currently tracking this person")
    
    # Visual data
    appearance: Optional[PersonAppearanceData] = None
    movement_metrics: Optional[PersonMovementMetrics] = None
    
    # Cross-camera data
    cameras_seen: List[str] = Field(default_factory=list, description="List of cameras that have seen this person")
    handoff_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Cross-camera handoff confidence")


class CameraVisualizationData(BaseModel):
    """Enhanced camera data with visualization features."""
    camera_id: str
    
    # Frame data
    image_source: str = Field(..., description="Source identifier for frame")
    frame_image_base64: Optional[str] = Field(None, description="Base64 encoded frame")
    frame_width: int = Field(..., gt=0, description="Frame width in pixels")
    frame_height: int = Field(..., gt=0, description="Frame height in pixels")
    frame_timestamp: datetime
    
    # Tracking data
    tracks: List[EnhancedPersonTrack] = Field(default_factory=list)
    person_count: int = Field(0, description="Current person count in this camera")
    
    # Performance data
    processing_time_ms: float = Field(0.0, description="Frame processing time")
    fps: float = Field(0.0, description="Current frames per second")
    quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Frame quality score")
    
    # Overlay configuration
    active_overlays: List[OverlayType] = Field(default_factory=list)
    overlay_opacity: float = Field(0.8, ge=0.0, le=1.0, description="Overlay opacity")


# --- Focus Tracking Schemas ---

class PersonDetailedInfo(BaseModel):
    """Comprehensive person information for focus tracking."""
    global_id: str
    first_detected: datetime
    last_seen: datetime
    tracking_duration: float
    
    # Current state
    current_camera: str
    current_position: Optional[MapCoordinates] = None
    current_bbox: Optional[BoundingBoxData] = None
    
    # Movement history
    position_history: List[Dict[str, Any]] = Field(default_factory=list)
    camera_transitions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Analytics
    movement_metrics: PersonMovementMetrics
    behavior_patterns: Dict[str, Any] = Field(default_factory=dict)
    
    # Visual data
    appearance_data: Optional[PersonAppearanceData] = None
    trajectory_path: List[MapCoordinates] = Field(default_factory=list)


class FocusTrackState(BaseModel):
    """Focus tracking state management."""
    task_id: str
    focused_person_id: Optional[str] = None
    focus_mode: FocusTrackMode = FocusTrackMode.SINGLE_PERSON
    focus_start_time: Optional[datetime] = None
    
    # Focus configuration
    highlight_color: str = Field("#FF0000", description="Highlight color for focused person")
    cross_camera_sync: bool = Field(True, description="Sync focus across all cameras")
    show_trajectory: bool = Field(True, description="Show movement trajectory")
    auto_follow: bool = Field(True, description="Auto-follow person across cameras")
    
    # Current focus data
    person_details: Optional[PersonDetailedInfo] = None


class FocusUpdateMessage(BaseModel):
    """WebSocket message for focus track updates."""
    type: str = Field("focus_update", const=True)
    payload: Dict[str, Any] = Field(
        ...,
        description="Focus update payload with person details and state"
    )


# --- Real-Time Analytics Schemas ---

class OccupancyMetrics(BaseModel):
    """Real-time occupancy analytics."""
    total_persons: int = Field(0, description="Total persons currently tracked")
    persons_per_camera: Dict[str, int] = Field(default_factory=dict)
    zone_occupancy: Dict[str, int] = Field(default_factory=dict)
    occupancy_trend: str = Field("stable", description="Trend: increasing, decreasing, stable")
    peak_occupancy: int = Field(0, description="Peak occupancy in current session")


class MovementMetrics(BaseModel):
    """Real-time movement analytics."""
    average_speed: float = Field(0.0, description="Average movement speed across all persons")
    movement_density: float = Field(0.0, description="Movement density score")
    congestion_areas: List[Dict[str, Any]] = Field(default_factory=list)
    flow_patterns: Dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """System performance metrics for analytics."""
    avg_processing_time_ms: float = Field(0.0)
    total_frames_processed: int = Field(0)
    frames_per_second: float = Field(0.0)
    memory_usage_mb: float = Field(0.0)
    gpu_utilization: float = Field(0.0)
    active_connections: int = Field(0)


class LiveAnalyticsData(BaseModel):
    """Comprehensive live analytics data."""
    environment_id: str
    timestamp: datetime
    
    # Metrics
    occupancy: OccupancyMetrics
    movement: MovementMetrics
    performance: PerformanceMetrics
    
    # Alerts
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Trends
    trend_data: Dict[str, Any] = Field(default_factory=dict)


# --- Playback Control Schemas ---

class PlaybackControls(BaseModel):
    """Playback control state and configuration."""
    state: PlaybackState = PlaybackState.STOPPED
    current_position: float = Field(0.0, ge=0.0, description="Current position (0.0 to 1.0)")
    playback_speed: float = Field(1.0, gt=0.0, description="Playback speed multiplier")
    loop_enabled: bool = Field(False, description="Whether to loop playback")
    
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    total_duration: float = Field(0.0, description="Total duration in seconds")
    
    # Capabilities
    can_seek: bool = Field(True, description="Whether seeking is available")
    can_change_speed: bool = Field(True, description="Whether speed change is available")
    available_speeds: List[float] = Field([0.25, 0.5, 1.0, 2.0, 4.0], description="Available playback speeds")


# --- Enhanced WebSocket Message Schemas ---

class EnhancedTrackingUpdate(BaseModel):
    """Enhanced tracking update with full visualization data."""
    type: str = Field("tracking_update", const=True)
    payload: Dict[str, Any] = Field(
        ...,
        description="Enhanced tracking payload with visualization data"
    )
    
    # Message metadata
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    compression_enabled: bool = Field(False)
    binary_data_included: bool = Field(False)


class MultiCameraVisualizationUpdate(BaseModel):
    """Multi-camera synchronized visualization update."""
    global_frame_index: int
    scene_id: str
    timestamp_processed_utc: datetime
    
    # Camera data
    cameras: Dict[str, CameraVisualizationData] = Field(default_factory=dict)
    
    # Global state
    total_person_count: int = Field(0)
    focused_person_id: Optional[str] = None
    active_cameras: List[str] = Field(default_factory=list)
    
    # Performance data
    total_processing_time_ms: float = Field(0.0)
    synchronization_offset_ms: float = Field(0.0)


class AnalyticsUpdateMessage(BaseModel):
    """Real-time analytics update message."""
    type: str = Field("analytics_update", const=True)
    payload: LiveAnalyticsData
    
    # Message metadata
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PlaybackFrameMessage(BaseModel):
    """Historical playback frame message."""
    type: str = Field("playback_frame", const=True)
    payload: Dict[str, Any] = Field(
        ...,
        description="Playback frame data with controls"
    )
    
    # Playback metadata
    playback_controls: PlaybackControls
    historical_data: bool = Field(True)


# --- API Response Schemas ---

class CameraConfigurationResponse(BaseModel):
    """Camera configuration for frontend."""
    camera_id: str
    name: str
    location: Optional[str] = None
    resolution: Tuple[int, int]
    position: Optional[MapCoordinates] = None
    capabilities: List[str] = Field(default_factory=list)
    calibration_status: str = Field("calibrated")


class EnvironmentConfigurationResponse(BaseModel):
    """Environment configuration for frontend."""
    environment_id: str
    name: str
    description: Optional[str] = None
    
    # Cameras
    cameras: List[CameraConfigurationResponse] = Field(default_factory=list)
    
    # Zones and layout
    zones: List[Dict[str, Any]] = Field(default_factory=list)
    map_bounds: Optional[Dict[str, float]] = None
    
    # Data availability
    available_date_ranges: List[Dict[str, str]] = Field(default_factory=list)
    total_data_hours: float = Field(0.0)


class PersonJourneyResponse(BaseModel):
    """Person journey analysis response."""
    global_person_id: str
    journey_start: datetime
    journey_end: datetime
    total_duration: float
    
    # Movement data
    total_distance: float
    average_speed: float
    cameras_visited: List[str] = Field(default_factory=list)
    zones_visited: List[str] = Field(default_factory=list)
    
    # Trajectory
    trajectory_points: List[MapCoordinates] = Field(default_factory=list)
    camera_transitions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Behavior analysis
    dwell_times: Dict[str, float] = Field(default_factory=dict)
    movement_patterns: Dict[str, Any] = Field(default_factory=dict)


class ExportJobResponse(BaseModel):
    """Data export job response."""
    job_id: str = Field(..., description="Export job identifier")
    status: str = Field(..., description="Export status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Export progress")
    
    # Job details
    export_type: str = Field(..., description="Type of export")
    file_format: str = Field(..., description="Output file format")
    estimated_size_mb: Optional[float] = None
    estimated_completion: Optional[datetime] = None
    
    # Download info
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


# --- Validation and Helper Functions ---

def validate_bbox_coordinates(bbox: BoundingBoxData) -> bool:
    """Validate bounding box coordinates are logical."""
    return (bbox.x1 < bbox.x2 and 
            bbox.y1 < bbox.y2 and
            all(coord >= 0 for coord in [bbox.x1, bbox.y1, bbox.x2, bbox.y2]))


def create_tracking_update_message(
    frame_data: MultiCameraVisualizationUpdate,
    focus_state: Optional[FocusTrackState] = None
) -> Dict[str, Any]:
    """Create a complete tracking update message for WebSocket transmission."""
    
    # Build camera payloads
    camera_payloads = {}
    for camera_id, camera_data in frame_data.cameras.items():
        camera_payloads[camera_id] = {
            "image_source": camera_data.image_source,
            "frame_image_base64": camera_data.frame_image_base64,
            "cropped_persons": {
                track.global_id: track.appearance.cropped_image_base64
                for track in camera_data.tracks
                if track.appearance and track.appearance.cropped_image_base64
            },
            "tracks": [
                {
                    "track_id": track.track_id,
                    "global_id": track.global_id,
                    "bbox_xyxy": [track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2],
                    "confidence": track.bbox.confidence,
                    "map_coords": [track.map_coords.x, track.map_coords.y] if track.map_coords else [0.0, 0.0],
                    "is_focused": track.is_focused,
                    "detection_time": track.detection_time.isoformat(),
                    "tracking_duration": track.tracking_duration
                }
                for track in camera_data.tracks
            ]
        }
    
    return {
        "type": "tracking_update",
        "payload": {
            "global_frame_index": frame_data.global_frame_index,
            "scene_id": frame_data.scene_id,
            "timestamp_processed_utc": frame_data.timestamp_processed_utc.isoformat(),
            "cameras": camera_payloads,
            "person_count_per_camera": {
                camera_id: camera_data.person_count
                for camera_id, camera_data in frame_data.cameras.items()
            },
            "focus_person_id": frame_data.focused_person_id,
            "total_processing_time_ms": frame_data.total_processing_time_ms,
            "synchronization_offset_ms": frame_data.synchronization_offset_ms
        }
    }