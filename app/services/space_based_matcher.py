
import logging
import math
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from app.core.config import settings

logger = logging.getLogger(__name__)

from app.services.global_person_registry import GlobalPersonRegistry

class SpaceBasedMatcher:
    """
    Service for matching tracks across cameras based on spatial proximity.
    
    It analyzes detections from multiple cameras in the same time frame and
    links them if their projected world coordinates are within a threshold distance.
    Uses Hungarian Algorithm (linear assignment) for robust 1-to-1 matching.
    """
    
    def __init__(self, registry: Optional[GlobalPersonRegistry] = None):
        self.threshold_meters = settings.SPATIAL_MATCH_THRESHOLD
        self.min_overlap_frames = settings.SPATIAL_MATCH_MIN_OVERLAP_FRAMES
        self.enabled = settings.SPATIAL_MATCH_ENABLED
        self.edge_margin = settings.SPATIAL_EDGE_MARGIN
        self.velocity_gate = settings.SPATIAL_VELOCITY_GATE
        
        # State to track potential matches over time
        # Key: (camera_id_a, track_id_a, camera_id_b, track_id_b) -> consecutive_frames
        self.potential_matches: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
        
        # Dependency: Global Person Registry
        self.registry = registry or GlobalPersonRegistry() # Fallback useful? Maybe better to require injection.
        # Note: If fallback creates new instance, it won't share state. 
        # But detection_video_service creates both, so it will pass the shared one.
        
        # Removed internal maps in favor of self.registry


        # Track history for velocity calculation (optional for later phases, kept effectively stateless per frame group here)
        
        # logger.info(
        #     f"SpaceBasedMatcher initialized (threshold={self.threshold_meters}m, "
        #     f"min_overlap={self.min_overlap_frames}, enabled={self.enabled})"
        # )

    def match_across_cameras(self, camera_detections: Dict[str, Dict[str, Any]]) -> None:
        """
        Main entry point to run matching logic for a single frame across all cameras.
        
        Args:
            camera_detections: Dictionary keying camera_id to its detection result dict.
                              The detection result dict is expected to contain a "tracks" list.
        
        Side Effects:
            Updates the "global_id" field in the track objects within `camera_detections`.
        """
        if not self.enabled:
            return

        # 1. Collect valid tracks with world coordinates
        valid_tracks = self._collect_tracks_with_coordinates(camera_detections)
        
        # DEBUG TRACE
        total_tracks = sum(len(t) for t in valid_tracks.values())
        # logger.info(f"SpaceMatcher: Found {total_tracks} valid tracks across {len(valid_tracks)} cameras")

        if len(valid_tracks) < 2:
            return  # Need at least two cameras with tracks to match anything

        # 2. Find matches between pairs of cameras using robust assignment
        new_matches = self._find_spatial_matches(valid_tracks)
        
        # 3. Update global ID assignments
        self._update_global_ids(new_matches)

        # 3.5 Ensure ALL tracks have a global_id (even unmatched ones)
        # Iterate over all valid tracks and assign a new global_id if they don't have one
        for camera_id, tracks in valid_tracks.items():
            for track in tracks:
                track_id = track.get("track_id")
                if track_id is not None:
                    key = (camera_id, str(track_id))
                    existing_gid = self.registry.get_global_id(camera_id, int(track_id))
                    if not existing_gid:
                         new_global_id = self.registry.allocate_new_id()
                         self.registry.assign_identity(camera_id, int(track_id), new_global_id)
                         pass # logger.debug(f"Created new global ID {new_global_id} for UNMATCHED track {camera_id}:{track_id}")
        
        # 4. Inject global IDs back into the detection results
        self._inject_global_ids(camera_detections)
        
        # 5. Cleanup stale entries
        self._cleanup_state(valid_tracks)

    def _collect_tracks_with_coordinates(self, camera_detections: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Flatten input into a structure easier to process: {camera_id: [track_dicts...]}"""
        valid_tracks = {}
        
        for camera_id, result in camera_detections.items():
            tracks = result.get("tracks", [])
            camera_valid_tracks = []
            
            frame_width = result.get("spatial_metadata", {}).get("frame_dimensions", {}).get("width")
            frame_height = result.get("spatial_metadata", {}).get("frame_dimensions", {}).get("height")
            
            for track in tracks:
                # Basic coordinate check
                map_coords = track.get("map_coords")
                spatial_data = track.get("spatial_data", {})
                
                is_projected = spatial_data.get("projection_successful") or (map_coords and map_coords.get("map_x") is not None)
                
                if is_projected:
                     # Edge Rejection Filter
                    if frame_width and frame_height and self._is_edge_detection(track, frame_width, frame_height):
                        continue
                        
                    camera_valid_tracks.append(track)
            
            if camera_valid_tracks:
                valid_tracks[camera_id] = camera_valid_tracks
                
        return valid_tracks

    def _is_edge_detection(self, track: Dict[str, Any], width: int, height: int) -> bool:
        """Check if bounding box center is too close to frame edge."""
        try:
            bbox = track.get("bbox", {})
            cx = bbox.get("center_x")
            cy = bbox.get("center_y")
            
            if cx is None or cy is None:
                return False
                
            norm_x = cx / width
            norm_y = cy / height
            
            margin = self.edge_margin
            if (norm_x < margin) or (norm_x > 1.0 - margin) or \
               (norm_y < margin) or (norm_y > 1.0 - margin):
                return True
                
            return False
        except Exception:
            return False

    def _find_spatial_matches(self, valid_tracks: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, str, str, str]]:
        """
        Compare all pairs of cameras and find optimal 1-to-1 matches using Hungarian Algorithm.
        """
        camera_ids = list(valid_tracks.keys())
        current_frame_pairs = set()
        confirmed_matches = []

        # Iterate all unique pairs of cameras
        for i in range(len(camera_ids)):
            for j in range(i + 1, len(camera_ids)):
                cam_a = camera_ids[i]
                cam_b = camera_ids[j]
                
                tracks_a = valid_tracks[cam_a]
                tracks_b = valid_tracks[cam_b]
                
                # Build Cost Matrix
                # Rows: tracks_a, Cols: tracks_b
                cost_matrix = np.full((len(tracks_a), len(tracks_b)), fill_value=float('inf'))
                
                for idx_a, track_a in enumerate(tracks_a):
                    for idx_b, track_b in enumerate(tracks_b):
                        dist = self._calculate_distance(track_a, track_b)
                        
                        if dist is not None:
                             # AUDIT LOG: Print distance if relatively close, to debug threshold issues
                            if dist < 5000.0:
                                pass # logger.debug(f"SpaceMatch Candidate: {cam_a}:{track_a.get('track_id')} vs {cam_b}:{track_b.get('track_id')} dist={dist:.2f}m")
                                pass

                            if dist <= self.threshold_meters:
                                # Optional: Velocity gate check (future Phase)
                                # if self.velocity_gate and opposite_direction(track_a, track_b): continue
                                
                                cost_matrix[idx_a, idx_b] = dist
                            
                # Solve Assignment Problem (Hungarian Algorithm)
                # rows (a_indices) assigned to cols (b_indices) with minimal cost
                # Scipy linear_sum_assignment handles 'inf' by avoiding assignments? 
                # No, we must mask it. However, let's let it minimize, then filter by threshold.
                # Actually, linear_sum_assignment needs finite weights.
                # We'll replace INF with a very large number for the solver.
                
                solve_matrix = np.where(cost_matrix == float('inf'), 1e9, cost_matrix)
                
                row_ind, col_ind = linear_sum_assignment(solve_matrix)
                
                for r, c in zip(row_ind, col_ind):
                    actual_cost = cost_matrix[r, c]
                    
                    if actual_cost <= self.threshold_meters:
                        track_a = tracks_a[r]
                        track_b = tracks_b[c]
                        
                        # Ensure consistent key ordering
                        if cam_a < cam_b:
                            key = (cam_a, track_a["track_id"], cam_b, track_b["track_id"])
                        else:
                            key = (cam_b, track_b["track_id"], cam_a, track_a["track_id"])
                            
                        current_frame_pairs.add(key)
                        self.potential_matches[key] += 1
                        
                        # if self.potential_matches[key] >= self.min_overlap_frames:
                        confirmed_matches.append(key)

        # Decay/Clean potential matches not seen in this frame
        self._update_match_counters(current_frame_pairs)
        
        return confirmed_matches

    def _update_match_counters(self, current_frame_pairs: Set[Tuple[str, str, str, str]]):
        keys_to_remove = []
        for key in self.potential_matches:
            if key not in current_frame_pairs:
                # Robustness: decay counter rather than immediate reset?
                # For now, immediate reset to require STRICT consecutiveness as per plan
                # Or maybe decay by 1? Plan says "Increment if match". Usually implies consecutive.
                # Let's reset to be safe against glitches.
                self.potential_matches[key] = max(0, self.potential_matches[key] - 1)
                # If we want strict consecutiveness, we should set to 0. 
                # But "voting" usually allows some gaps. Let's stick to decay.
                
                if self.potential_matches[key] == 0:
                    keys_to_remove.append(key)
        
        for k in keys_to_remove:
            del self.potential_matches[k]

    def _calculate_distance(self, track_a: Dict[str, Any], track_b: Dict[str, Any]) -> Optional[float]:
        try:
            ax = track_a["map_coords"]["map_x"]
            ay = track_a["map_coords"]["map_y"]
            bx = track_b["map_coords"]["map_x"]
            by = track_b["map_coords"]["map_y"]
            
            return math.sqrt((ax - bx)**2 + (ay - by)**2)
        except (KeyError, TypeError):
            return None

    def _update_global_ids(self, matches: List[Tuple[str, str, str, str]]):
        for cam_a, track_a_id, cam_b, track_b_id in matches:
            key_a = (cam_a, str(track_a_id)) # Ensure string track ID
            key_b = (cam_b, str(track_b_id))
            
            # Retrieve current global IDs from registry
            existing_global_a = self.registry.get_global_id(*key_a)
            existing_global_b = self.registry.get_global_id(*key_b)
            
            if existing_global_a and existing_global_b:
                if existing_global_a != existing_global_b:
                    # MERGE
                    if existing_global_a < existing_global_b:
                         self.registry.merge_identities(target_global_id=existing_global_a, source_global_id=existing_global_b)
                    else:
                         self.registry.merge_identities(target_global_id=existing_global_b, source_global_id=existing_global_a)
            
            elif existing_global_a:
                self.registry.assign_identity(cam_b, track_b_id, existing_global_a)
                
            elif existing_global_b:
                self.registry.assign_identity(cam_a, track_a_id, existing_global_b)
                
            else:
                new_global_id = self.registry.allocate_new_id()
                self.registry.assign_identity(cam_a, track_a_id, new_global_id)
                self.registry.assign_identity(cam_b, track_b_id, new_global_id)
                pass # logger.debug(f"Created new global ID {new_global_id} for match {cam_a}:{track_a_id} <-> {cam_b}:{track_b_id}")

    # Removed internal _assign_global_id and _merge_identities helpers as they are now in registry



    def _inject_global_ids(self, camera_detections: Dict[str, Dict[str, Any]]):
        """Write the global IDs into the track dictionaries."""
        for camera_id, result in camera_detections.items():
            tracks = result.get("tracks", [])
            for track in tracks:
                track_id = track.get("track_id")
                if track_id is not None:
                    global_id = self.registry.get_global_id(camera_id, int(track_id))
                    if global_id:
                        track["global_id"] = global_id
                        
    def _cleanup_state(self, current_valid_tracks: Dict[str, List[Dict[str, Any]]]):
        """Remove global ID mappings for tracks that have disappeared (optional)."""
        # Future: Implement cleanup based on time-to-live
        pass

    def is_global_id_shared(self, global_id: str) -> bool:
        """Check if a global ID is currently assigned to tracks in multiple cameras."""
        return self.registry.is_global_id_shared(global_id)
