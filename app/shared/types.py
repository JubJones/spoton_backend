# FILE: app/common_types.py
"""
Module for shared type aliases and data structures used across the application.
"""
from typing import Dict, Tuple, Optional, List, NewType, Callable, Any
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


class TrackedObjectData(BaseModel):
    """Data for a single tracked object, enriched with global ID."""
    camera_id: CameraID
    track_id: TrackID # The temporary track ID from the per-camera tracker
    global_person_id: Optional[GlobalID]
    bbox_xyxy: BoundingBoxXYXY
    confidence: Optional[float] = None
    feature_vector: Optional[List[float]] = None # Optional: if features are explicitly passed
    map_coords: Optional[List[float]] = Field(None, description="Projected [X, Y] coordinates on the map. Null if not available.")
    search_roi: Optional[Dict[str, Any]] = Field(None, description="ROI payload for spatial search.")
    transformation_quality: Optional[float] = Field(None, description="Quality score for world-plane projection.")
    geometric_match: Optional[Dict[str, Any]] = Field(None, description="Geometric match metadata.")

    class Config:
        arbitrary_types_allowed = True

# Removed unused FrameProcessingUpdatePayload model


# --- Handoff Logic Types (Inspired by POC) ---
# MODIFIED: ExitDirection removed as it's replaced by QuadrantName for source_exit_quadrant
QuadrantName = NewType("QuadrantName", str)   # e.g., 'upper_left', 'lower_right'

class ExitRuleModel(BaseModel):
    """
    Defines a rule for triggering a handoff based on exit from a specific source quadrant.
    Pydantic model version of POC's ExitRule.
    """
    source_exit_quadrant: QuadrantName = Field(..., description="The source quadrant in the current camera that triggers this rule (e.g., 'upper_right').") # MODIFIED
    target_cam_id: CameraID = Field(..., description="The camera ID this rule targets for handoff.")
    target_entry_area: str = Field(..., description="Descriptive name of the entry area in the target camera (e.g., 'lower_left').")
    notes: Optional[str] = Field(None, description="Optional notes about this rule.")

class CameraHandoffDetailConfig(BaseModel):
    """
    Detailed configuration for a camera, including handoff rules and homography.
    Used internally by settings, keyed by (env_id, cam_id).
    """
    exit_rules: List[ExitRuleModel] = Field(default_factory=list)
    homography_matrix_path: Optional[str] = Field(None, description="Path to the .npz file containing homography points for this camera and scene, relative to HOMOGRAPHY_DATA_DIR.")
    # Homography matrix itself will be loaded and cached by HomographyService



# --- Map for Quadrant Calculation ---
QUADRANT_REGIONS_TEMPLATE: Dict[QuadrantName, Callable[[int, int], Tuple[int, int, int, int]]] = {
    QuadrantName('upper_left'): lambda W, H: (0, 0, W // 2, H // 2),
    QuadrantName('upper_right'): lambda W, H: (W // 2, 0, W, H // 2),
    QuadrantName('lower_left'): lambda W, H: (0, H // 2, W // 2, H),
    QuadrantName('lower_right'): lambda W, H: (W // 2, H // 2, W, H),
}
