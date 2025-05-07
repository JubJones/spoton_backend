from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any # Dict and Any are not used in this snippet directly, but good to keep if planning to expand.
from datetime import datetime

class TimeRangeUTC(BaseModel):
    start: datetime
    end: datetime

class ProcessingTaskStartRequest(BaseModel):
    cameras: List[str]
    time_range_utc: TimeRangeUTC
    environment_id: str
    # Add other parameters like specific Re-ID rules or models if configurable

class ProcessingTaskCreateResponse(BaseModel):
    task_id: str
    status_url: HttpUrl # Or just str if full URL validation by Pydantic isn't strictly needed here.
    websocket_url: str # e.g., wss://server/ws/tracking/{task_id}

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
    image_url: HttpUrl
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
