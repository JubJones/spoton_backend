from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union # Added Union
from datetime import datetime
import uuid

# --- Request Schemas ---

class ProcessingTaskStartRequest(BaseModel):
    """Request body to start a new processing task."""
    environment_id: str = Field(..., description="Identifier for the environment (e.g., 'campus', 'factory') to process.")

# --- Response Schemas ---

class ProcessingTaskCreateResponse(BaseModel):
    """Response after successfully queuing a processing task."""
    task_id: uuid.UUID
    message: str = "Processing task initiated."
    status_url: str # Relative path for status check
    websocket_url: str # Relative path for WebSocket connection

class TaskStatusResponse(BaseModel):
    """Response containing the current status of a task."""
    task_id: uuid.UUID
    status: str = Field(..., description="Current status (e.g., QUEUED, INITIALIZING, DOWNLOADING, EXTRACTING, PROCESSING, COMPLETED, FAILED)")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall task progress estimate (0.0 to 1.0)")
    current_step: Optional[str] = Field(None, description="Description of the current step being performed.")
    details: Optional[str] = Field(None, description="Additional details or error message.")

# --- WebSocket Message Schemas ---

class TrackedPersonData(BaseModel):
    """Data for a single tracked person in a frame for WebSocket output."""
    track_id: int
    global_id: Optional[str] = Field(None, description="Globally unique ID assigned by Re-ID (null if not identified).")
    bbox_xyxy: List[float] = Field(..., description="Bounding box in image coordinates [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Detection confidence score.")
    class_id: Optional[int] = Field(None, description="Class ID of the detection (e.g., 1 for person).")
    map_coords: Optional[List[float]] = Field(None, description="Projected [X, Y] coordinates on the map. Null if not available.")

class CameraTracksData(BaseModel):
    """Holds tracking data for a single camera in a frame for WebSocket output."""
    image_source: str = Field(..., description="Filename of the image source for this camera (e.g., '000000.jpg').")
    tracks: List[TrackedPersonData] = Field(default_factory=list)

class WebSocketTrackingMessagePayload(BaseModel):
    """Payload for the 'tracking_update' WebSocket message, matching JSON structure."""
    frame_index: int = Field(..., description="0-indexed frame number for the scene.")
    scene_id: str = Field(..., description="Identifier for the scene (e.g., 's10').")
    timestamp_processed_utc: str = Field(..., description="ISO UTC timestamp of when the frame was processed.")
    cameras: Dict[str, CameraTracksData] # Keyed by CameraID string (e.g., "c09")


# --- New Schemas for Media Availability Notification ---
class MediaURLEntry(BaseModel):
    """Represents a single media URL entry for a camera."""
    camera_id: str = Field(..., description="The camera ID this media URL pertains to.")
    sub_video_filename: str = Field(..., description="The filename of the sub-video.")
    url: str = Field(..., description="The full HTTP URL to fetch this sub-video from the backend.")

class WebSocketMediaAvailablePayload(BaseModel):
    """Payload for the 'media_available' WebSocket message."""
    sub_video_batch_index: int = Field(..., description="0-indexed identifier for this batch of sub-videos.")
    media_urls: List[MediaURLEntry] = Field(default_factory=list)


# Generic WebSocket message structure
class WebSocketMessage(BaseModel):
    """
    Generic structure for messages pushed via WebSocket.
    The payload field can be one of several specific payload types.
    """
    type: str = Field(..., description="Type of WebSocket message (e.g., 'tracking_update', 'status_update', 'media_available').")
    # Using Dict[str, Any] for flexibility, but specific handlers will expect certain structures.
    # For stricter typing, you could use Union[WebSocketTrackingMessagePayload, TaskStatusResponse, WebSocketMediaAvailablePayload, ...]
    # but TaskStatusResponse isn't directly sent as payload, its dict version is.
    payload: Dict[str, Any] = Field(..., description="The actual message content, structure depends on 'type'.")