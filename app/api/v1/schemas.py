from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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

class TrackedPersonData(BaseModel): # Renamed for clarity and to match JSON structure
    """Data for a single tracked person in a frame for WebSocket output."""
    track_id: int # The temporary track ID from the intra-camera tracker
    global_id: Optional[str] = Field(None, description="Globally unique ID assigned by Re-ID (null if not identified).")
    bbox_xyxy: List[float] = Field(..., description="Bounding box in image coordinates [x1, y1, x2, y2]") # Changed name for JSON match
    confidence: Optional[float] = Field(None, description="Detection confidence score.")
    class_id: Optional[int] = Field(None, description="Class ID of the detection (e.g., 1 for person).") # Added class_id
    map_coords: Optional[List[float]] = Field(None, description="Projected [X, Y] coordinates on the map. Null if not available.") # Changed name

class CameraTracksData(BaseModel): # New model for per-camera data in WebSocket
    """Holds tracking data for a single camera in a frame for WebSocket output."""
    image_source: str = Field(..., description="Filename of the image source for this camera (e.g., '000000.jpg').")
    tracks: List[TrackedPersonData] = Field(default_factory=list)


class WebSocketTrackingMessagePayload(BaseModel): # Specific payload for tracking_update
    """Payload for the 'tracking_update' WebSocket message, matching JSON structure."""
    frame_index: int = Field(..., description="0-indexed frame number for the scene.")
    scene_id: str = Field(..., description="Identifier for the scene (e.g., 's10').")
    timestamp_processed_utc: str = Field(..., description="ISO UTC timestamp of when the frame was processed.")
    cameras: Dict[str, CameraTracksData] # Keyed by CameraID string (e.g., "c09")


# Generic WebSocket message structure
class WebSocketMessage(BaseModel):
    """Generic structure for messages pushed via WebSocket."""
    type: str = Field(..., description="Type of WebSocket message (e.g., 'tracking_update', 'status_update').")
    payload: Dict[str, Any] = Field(..., description="The actual message content, structure depends on 'type'.")

# Example for the structure that NotificationService will send for tracking updates:
# It will construct a dictionary for the payload based on WebSocketTrackingMessagePayload
# and then wrap it in WebSocketMessage.