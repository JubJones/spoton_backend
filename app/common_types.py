"""
Module for shared type aliases and data structures used across the application.
"""
from typing import Dict, Any, Tuple, Optional, List, NewType
import numpy as np
from pydantic import BaseModel


CameraID = NewType("CameraID", str)
TrackID = NewType("TrackID", int) # Intra-camera track ID
GlobalID = NewType("GlobalID", str) # System-wide unique person ID (UUID or similar string)
FeatureVector = NewType("FeatureVector", np.ndarray)
BoundingBoxXYXY = NewType("BoundingBoxXYXY", List[float]) # [x1, y1, x2, y2]

# Key for uniquely identifying a track within a specific camera's context
TrackKey = Tuple[CameraID, TrackID]

# Frame data as a numpy array (BGR) and its path
FrameData = Tuple[np.ndarray, str] # (frame_image_np, frame_path_str)
FrameBatch = Dict[CameraID, Optional[FrameData]]

class RawDetection(BaseModel):
    """Represents a raw detection before tracking."""
    bbox_xyxy: BoundingBoxXYXY
    confidence: float
    class_id: int

class TrackedObjectData(BaseModel):
    """Data for a single tracked object, enriched with global ID."""
    camera_id: CameraID
    track_id: TrackID # The temporary track ID from the per-camera tracker
    global_person_id: Optional[GlobalID]
    bbox_xyxy: BoundingBoxXYXY
    confidence: Optional[float] = None
    feature_vector: Optional[List[float]] = None # Optional: if features are explicitly passed

    class Config:
        arbitrary_types_allowed = True

# Example for a message payload, can be refined
class FrameProcessingUpdatePayload(BaseModel):
    """Payload for WebSocket tracking updates."""
    task_id: str
    camera_id: CameraID
    frame_path: str # Path of the frame being reported
    timestamp_utc: str # ISO format timestamp
    tracked_objects: List[TrackedObjectData]