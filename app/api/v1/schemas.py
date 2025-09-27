from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# --- Request Schemas ---

class ProcessingTaskStartRequest(BaseModel):
    """Request body to start a new processing task."""
    environment_id: str = Field(..., description="Identifier for the environment (e.g., 'campus', 'factory') to process.")
    # Optional feature flags (backward-compatible): if omitted, service-level defaults are used
    enable_tracking: Optional[bool] = Field(None, description="Enable intra-camera tracking for this task (overrides global setting if provided).")

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
    global_id: Optional[str] = Field(None, description="Globally unique ID (null if not assigned).")
    bbox_xyxy: List[float] = Field(..., description="Bounding box in image coordinates [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Detection confidence score.")
    class_id: Optional[int] = Field(None, description="Class ID of the detection (e.g., 1 for person).")
    map_coords: Optional[List[float]] = Field(None, description="Projected [X, Y] coordinates on the map. Null if not available.")

class CameraTracksData(BaseModel):
    """Holds tracking data for a single camera in a frame for WebSocket output."""
    image_source: str = Field(..., description="Filename or identifier of the image source for this camera (e.g., '000000.jpg').")
    frame_image_base64: Optional[str] = Field(None, description="Base64 encoded string of the frame image (e.g., JPEG). Null if frame not available for this camera in this batch.")
    tracks: List[TrackedPersonData] = Field(default_factory=list)

class WebSocketTrackingMessagePayload(BaseModel):
    """Payload for the 'tracking_update' WebSocket message, matching JSON structure."""
    global_frame_index: int = Field(..., description="Absolute frame index for the task, increments across all sub-video batches.")
    scene_id: str = Field(..., description="Identifier for the scene/environment (e.g., 'campus').")
    timestamp_processed_utc: str = Field(..., description="ISO UTC timestamp of when the frame was processed.")
    cameras: Dict[str, CameraTracksData] # Keyed by CameraID string (e.g., "c09")


# --- Schemas for Media Availability and Batch Completion ---
# MediaURLEntry and WebSocketMediaAvailablePayload are no longer needed for the primary frontend flow.
# They are removed. If needed for other purposes (e.g. debugging endpoint), they could be reinstated.

# class MediaURLEntry(BaseModel):
#     """Represents a single media URL entry for a camera within a sub-video batch."""
#     camera_id: str = Field(..., description="The camera ID this media URL pertains to.")
#     sub_video_filename: str = Field(..., description="The filename of the sub-video.")
#     url: str = Field(..., description="The backend-relative HTTP URL to fetch this sub-video.")
#     start_global_frame_index: int = Field(..., description="Global frame index corresponding to the start of this sub-video.")
#     num_frames_in_sub_video: int = Field(..., description="Expected number of frames in this sub-video at the processing FPS.")

# class WebSocketMediaAvailablePayload(BaseModel):
#     """Payload for the 'media_available' WebSocket message. DEPRECATED for primary frontend use."""
#     sub_video_batch_index: int = Field(..., description="0-indexed identifier for this batch of sub-videos.")
#     media_urls: List[MediaURLEntry] = Field(default_factory=list, description="List of media URLs for the current batch.")

class WebSocketBatchProcessingCompletePayload(BaseModel):
    """
    Payload for the 'batch_processing_complete' WebSocket message.
    This message's relevance to the frontend is reduced as video synchronization is now direct.
    It can still signal the backend's completion of a logical processing unit (a sub-video batch).
    """
    sub_video_batch_index: int = Field(..., description="0-indexed identifier of the sub-video batch that has finished processing.")


# Generic WebSocket message structure
class WebSocketMessage(BaseModel):
    """
    Generic structure for messages pushed via WebSocket.
    The payload field's structure depends on the 'type' field.
    """
    type: str = Field(..., description="Type of WebSocket message (e.g., 'tracking_update', 'status_update', 'batch_processing_complete').")
    payload: Dict[str, Any] = Field(..., description="The actual message content, structure depends on 'type'.")
# --- Playback Control Schemas ---

class PlaybackState(str, Enum):
    """Enumerates backend-managed playback states."""

    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


class PlaybackStatusResponse(BaseModel):
    """Represents the playback control status returned to clients."""

    task_id: str
    state: PlaybackState
    last_transition_at: datetime
    last_frame_index: Optional[int] = None
    last_error: Optional[str] = None
