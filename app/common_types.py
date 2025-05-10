"""
Module for shared type aliases and data structures used across the application.
"""
from typing import Dict, Any, Tuple, Optional, List, NewType, NamedTuple, Set, Callable
import numpy as np
from pydantic import BaseModel, Field


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
    map_coords: Optional[List[float]] = Field(None, description="Projected [X, Y] coordinates on the map. Null if not available.") # MODIFIED

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


# --- Handoff Logic Types (Inspired by POC) ---
ExitDirection = NewType("ExitDirection", str) # e.g., 'up', 'down', 'left', 'right'
QuadrantName = NewType("QuadrantName", str)   # e.g., 'upper_left', 'lower_right'

class ExitRuleModel(BaseModel):
    """
    Defines a rule for triggering a handoff based on exit direction.
    Pydantic model version of POC's ExitRule.
    """
    direction: ExitDirection = Field(..., description="Direction rule applies to (e.g., 'down', 'left').")
    target_cam_id: CameraID = Field(..., description="The camera ID this rule targets for handoff.")
    target_entry_area: str = Field(..., description="Descriptive name of the entry area in the target camera (e.g., 'upper_right').")
    notes: Optional[str] = Field(None, description="Optional notes about this rule.")

class CameraHandoffDetailConfig(BaseModel):
    """
    Detailed configuration for a camera, including handoff rules and homography.
    Used internally by settings, keyed by (env_id, cam_id).
    """
    exit_rules: List[ExitRuleModel] = Field(default_factory=list)
    homography_matrix_path: Optional[str] = Field(None, description="Path to the .npz file containing homography points for this camera and scene, relative to WEIGHTS_DIR/homography_points.")
    # Homography matrix itself will be loaded and cached by MultiCameraFrameProcessor

class HandoffTriggerInfo(NamedTuple):
    """
    Holds information about a triggered handoff event for a specific track.
    Directly from POC.
    """
    source_track_key: TrackKey
    rule: ExitRuleModel # Use the Pydantic model
    source_bbox: BoundingBoxXYXY # BBox that triggered the rule in the source camera


# --- Map for Quadrant Calculation ---
QUADRANT_REGIONS_TEMPLATE: Dict[QuadrantName, Callable[[int, int], Tuple[int, int, int, int]]] = {
    QuadrantName('upper_left'): lambda W, H: (0, 0, W // 2, H // 2),
    QuadrantName('upper_right'): lambda W, H: (W // 2, 0, W, H // 2),
    QuadrantName('lower_left'): lambda W, H: (0, H // 2, W // 2, H),
    QuadrantName('lower_right'): lambda W, H: (W // 2, H // 2, W, H),
}

DIRECTION_TO_QUADRANTS_MAP: Dict[ExitDirection, List[QuadrantName]] = {
    ExitDirection('up'): [QuadrantName('upper_left'), QuadrantName('upper_right')],
    ExitDirection('down'): [QuadrantName('lower_left'), QuadrantName('lower_right')],
    ExitDirection('left'): [QuadrantName('upper_left'), QuadrantName('lower_left')],
    ExitDirection('right'): [QuadrantName('upper_right'), QuadrantName('lower_right')],
}