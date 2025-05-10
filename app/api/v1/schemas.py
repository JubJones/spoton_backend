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

class MapCoordinates(BaseModel):
    """Represents coordinates on a 2D map (ignored for now, but kept for structure)."""
    x: Optional[float] = None
    y: Optional[float] = None

class TrackedPersonData(BaseModel):
    """Data for a single tracked person in a frame."""
    track_id: int # The temporary track ID assigned by the intra-camera tracker
    global_person_id: Optional[str] = Field(None, description="Globally unique ID assigned by Re-ID (null if not identified).")
    bbox_img: List[float] = Field(..., description="Bounding box in image coordinates [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Detection confidence score.")
    # map_coordinates: Optional[MapCoordinates] = None # Keep optional, but ignore for now

class WebSocketTrackingMessage(BaseModel):
    """Message pushed via WebSocket containing tracking data for a frame."""
    type: str = Field(default="tracking_update", description="Type of WebSocket message.")
    payload: Dict[str, Any] = Field(..., description="The actual message content.")

# --- Analytics Schemas (Keep as placeholders) ---
class TrajectoryPoint(BaseModel):
    timestamp: datetime
    camera_id: str
    map_x: float
    map_y: float
    bbox_img: Optional[List[float]] = None

class TrajectoryResponse(BaseModel):
    global_person_id: str
    trajectory: List[TrajectoryPoint]