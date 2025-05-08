from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any # Dict and Any are not used in this snippet directly, but good to keep if planning to expand.
from datetime import datetime
import uuid # For task IDs

class TimeRangeUTC(BaseModel):
    start: datetime
    end: datetime

class ProcessingTaskStartRequest(BaseModel):
    # cameras: List[str] # Removed, cameras are determined by environment_id
    # time_range_utc: TimeRangeUTC # Removed for simplification, can be added back
    environment_id: str = Field(..., description="Identifier for the environment (e.g., 'campus', 'factory') to process.")
    # Add other parameters like specific Re-ID rules or models if configurable

class ProcessingTaskCreateResponse(BaseModel):
    task_id: uuid.UUID # Using UUID object directly for Pydantic
    message: str = "Processing task initiated."
    status_url: str # Relative path for status
    websocket_url: str # Relative path for WebSocket

class TaskStatusResponse(BaseModel):
    task_id: uuid.UUID
    status: str # e.g., "INITIALIZING", "PREPARING_DATA", "PROCESSING", "COMPLETED", "FAILED"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    details: Optional[str] = None
    # Can add more specific progress indicators later if needed

# --- WebSocket Message Schemas (mirroring project doc) ---
class MapCoordinates(BaseModel):
    x: float
    y: float

class TrackedPersonData(BaseModel):
    global_person_id: str
    bbox_img: List[float] # [x1, y1, x2, y2] or [x,y,w,h] - ensure consistency
    map_coordinates: MapCoordinates

class WebSocketTrackingMessage(BaseModel):
    camera_id: str
    timestamp: datetime # Or str in ISO format
    image_url: HttpUrl # This will be local file paths now or data URLs if embedding
    tracking_data: List[TrackedPersonData]

# --- Analytics Schemas ---
class TrajectoryPoint(BaseModel):
    timestamp: datetime
    camera_id: str
    map_x: float
    map_y: float
    bbox_img: Optional[List[float]] = None

class TrajectoryResponse(BaseModel):
    global_person_id: str
    trajectory: List[TrajectoryPoint]