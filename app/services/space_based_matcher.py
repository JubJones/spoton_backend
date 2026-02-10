
import logging
import math
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from app.core.config import settings

logger = logging.getLogger(__name__)

# File-based debug logger for spatial matching (write to local file)
SPATIAL_DEBUG_LOG = "/app/logs/spatial_debug.log"
try:
    import os
    os.makedirs(os.path.dirname(SPATIAL_DEBUG_LOG), exist_ok=True)
except:
    SPATIAL_DEBUG_LOG = "./spatial_debug.log"

def spatial_debug(msg: str):
    """Write debug message to spatial debug log file."""
    try:
        with open(SPATIAL_DEBUG_LOG, "a") as f:
            from datetime import datetime
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass

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
        self.no_match_distance = settings.SPATIAL_NO_MATCH_DISTANCE
        
        # Buffer TTL in frames (default 30 = ~1 second at 30fps)
        self.buffer_ttl_frames = int(getattr(settings, 'SPATIAL_BUFFER_FRAMES', 30))
        
        # State to track potential matches over time
        # Key: (camera_id_a, track_id_a, camera_id_b, track_id_b) -> consecutive_frames
        self.potential_matches: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
        
        # Recent tracks buffer: {camera_id: [{track_id, global_id, map_coords, frame_num}]}
        self._recent_tracks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._current_frame_num = 0
        
        # Dependency: Global Person Registry
        self.registry = registry or GlobalPersonRegistry()

        # Track history for velocity calculation (optional for later phases)
        
        # logger.info(
        #     f"SpaceBasedMatcher initialized (threshold={self.threshold_meters}m, "
        #     f"min_overlap={self.min_overlap_frames}, enabled={self.enabled})"
        # )

    def set_environment(self, environment_id: str):
        """Configure spatial thresholds based on the environment."""
        if environment_id == "factory":
            self.threshold_meters = settings.SPATIAL_MATCH_THRESHOLD_FACTORY
            self.no_match_distance = settings.SPATIAL_NO_MATCH_DISTANCE_FACTORY
            self.edge_margin = settings.SPATIAL_EDGE_MARGIN_FACTORY
            logger.info(f"Using FACTORY spatial thresholds: match={self.threshold_meters}, max={self.no_match_distance}, edge_margin={self.edge_margin}")
        else:
            self.threshold_meters = settings.SPATIAL_MATCH_THRESHOLD
            self.no_match_distance = settings.SPATIAL_NO_MATCH_DISTANCE
            self.edge_margin = settings.SPATIAL_EDGE_MARGIN
            logger.info(f"Using DEFAULT/CAMPUS spatial thresholds: match={self.threshold_meters}, max={self.no_match_distance}, edge_margin={self.edge_margin}")

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

        self._current_frame_num += 1

        # 1. Collect valid tracks with world coordinates
        valid_tracks = self._collect_tracks_with_coordinates(camera_detections)
        
        # DEBUG TRACE (c09/c16 only)
        if "c09" in valid_tracks and "c16" in valid_tracks:
            for cam_id in ["c09", "c16"]:
                for t in valid_tracks.get(cam_id, []):
                    mc = t.get("map_coords", {})
                    gid = t.get("global_id") or self.registry.get_global_id(cam_id, int(t.get('track_id', 0)))
                    spatial_debug(f"{cam_id}:{gid} -> ({mc.get('map_x'):.1f}, {mc.get('map_y'):.1f})")

        # 2. Find matches between pairs of cameras using robust assignment
        new_matches = []
        if len(valid_tracks) >= 2:
            new_matches = self._find_spatial_matches(valid_tracks)
            c09_c16_matches = [m for m in new_matches if {m[0], m[2]} == {"c09", "c16"}]
            if c09_c16_matches:
                spatial_debug(f"c09<->c16 matches: {len(c09_c16_matches)}")
        
        # 3. Update global ID assignments
        # Build map of active global IDs to check for conflicts (prevent merging distinct people in same view)
        active_global_ids = self._get_active_global_ids(camera_detections)
        self._update_global_ids(new_matches, active_global_ids)

        # 3.5 BUFFER MATCHING: For unmatched tracks, check against recent buffer from OTHER cameras
        for camera_id, tracks in valid_tracks.items():
            for track in tracks:
                track_id = track.get("track_id")
                if track_id is None:
                    continue
                
                # Skip if already has a global ID
                existing_gid = self.registry.get_global_id(camera_id, int(track_id))
                if existing_gid:
                    continue
                
                # Try to match against buffered tracks from OTHER cameras
                track_coords = track.get("map_coords", {})
                if not track_coords.get("map_x"):
                    continue
                
                buffer_match_gid = self._find_buffer_match(camera_id, track_coords)
                if buffer_match_gid:
                    self.registry.assign_identity(camera_id, int(track_id), buffer_match_gid)
                    if camera_id in ["c09", "c16"]:
                        spatial_debug(f"BUFFER MATCH: {camera_id}:T{track_id} -> {buffer_match_gid}")

        # 3.6 Ensure ALL tracks have a global_id (even unmatched ones)
        for camera_id, tracks in valid_tracks.items():
            for track in tracks:
                track_id = track.get("track_id")
                if track_id is not None:
                    existing_gid = self.registry.get_global_id(camera_id, int(track_id))
                    if not existing_gid:
                         new_global_id = self.registry.allocate_new_id()
                         self.registry.assign_identity(camera_id, int(track_id), new_global_id)
        
        # 4. Inject global IDs back into the detection results
        self._inject_global_ids(camera_detections)

        # 4.5 Safety Check: Resolve intra-camera conflicts (Same Global ID on multiple tracks in same camera)
        self._resolve_intra_camera_conflicts(camera_detections)

        # 5. Update buffer with current tracks (AFTER global IDs are assigned)
        self._update_recent_tracks_buffer(valid_tracks)
        
        # 6. Cleanup stale entries
        self._cleanup_state(valid_tracks)

    def _resolve_intra_camera_conflicts(self, camera_detections: Dict[str, Dict[str, Any]]):
        """
        Ensure that within a single camera frame, no two tracks share the same global ID.
        If duplicates are found, keep the ID for the highest confidence track and reassign others.
        """
        for camera_id, result in camera_detections.items():
            tracks = result.get("tracks", [])
            if not tracks:
                continue

            # Group by global_id
            id_map = defaultdict(list)
            for track in tracks:
                gid = track.get("global_id")
                if gid:
                    id_map[gid].append(track)
            
            # Check for duplicates
            for gid, sharing_tracks in id_map.items():
                if len(sharing_tracks) > 1:
                    # Conflict detected!
                    # logger.warning(f"Conflict: Global ID {gid} assigned to {len(sharing_tracks)} tracks in camera {camera_id}. Resolving...")
                    
                    # Sort by confidence (descending) -> Keep best
                    sharing_tracks.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                    
                    winner = sharing_tracks[0]
                    losers = sharing_tracks[1:]
                    
                    for track in losers:
                        lid = track.get("track_id")
                        if lid is not None:
                            new_gid = self.registry.allocate_new_id()
                            # Use force_assign_identity to SPLIT this specific track to a new ID
                            # without merging all other tracks that might share the old ID.
                            self.registry.force_assign_identity(camera_id, int(lid), new_gid)
                            track["global_id"] = new_gid
                            # logger.info(f"Resolved conflict: Reassigned {camera_id}:{lid} from {gid} to {new_gid}")

    def _get_active_global_ids(self, camera_detections: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
        """
        Build a map of which global IDs are currently active in which cameras.
        Used to prevent merging identities that are simultaneously present in the same camera.
        """
        active_map = defaultdict(set)
        for camera_id, result in camera_detections.items():
            for track in result.get("tracks", []):
                track_id = track.get("track_id")
                if track_id is not None:
                    gid = self.registry.get_global_id(camera_id, int(track_id))
                    if gid:
                        active_map[gid].add(camera_id)
        return active_map

    def _collect_tracks_with_coordinates(self, camera_detections: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Flatten input into a structure easier to process: {camera_id: [track_dicts...]}"""
        valid_tracks = {}
        
        for camera_id, result in camera_detections.items():
            tracks = result.get("tracks", [])
            camera_valid_tracks = []
            
            frame_width = result.get("spatial_metadata", {}).get("frame_dimensions", {}).get("width")
            frame_height = result.get("spatial_metadata", {}).get("frame_dimensions", {}).get("height")
            
            for track in tracks:
                track_id = track.get("track_id")
                # Basic coordinate check
                map_coords = track.get("map_coords")
                spatial_data = track.get("spatial_data", {})
                
                is_projected = spatial_data.get("projection_successful") or (map_coords and map_coords.get("map_x") is not None)
                
                # DEBUG: Log rejection for c09/c16 tracks
                if camera_id in ["c09", "c16"]:
                    if not is_projected:
                        spatial_debug(f"REJECTED {camera_id}:T{track_id} - No projection (map_coords={map_coords})")
                        continue
                    
                    if frame_width and frame_height and self._is_edge_detection(track, frame_width, frame_height):
                        spatial_debug(f"REJECTED {camera_id}:T{track_id} - Edge filter")
                        continue
                    
                    camera_valid_tracks.append(track)
                else:
                    # Non-debug path for other cameras
                    if is_projected:
                        if frame_width and frame_height and self._is_edge_detection(track, frame_width, frame_height):
                            continue
                        camera_valid_tracks.append(track)
            
            if camera_valid_tracks:
                valid_tracks[camera_id] = camera_valid_tracks
                
        return valid_tracks

    def _is_edge_detection(self, track: Dict[str, Any], width: int, height: int) -> bool:
        """Check if detection is too close to frame edge."""
        margin = self.edge_margin
        
        # If margin is 0.0 or less, disable edge filtering completely.
        # This allows tracking objects even if their center point is slightly outside the frame.
        if margin <= 0.0:
            return False

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

    def _find_buffer_match(self, current_camera_id: str, current_coords: Dict[str, float]) -> Optional[str]:
        """
        Search buffered tracks from OTHER cameras to find a spatial match.
        Returns the global_id of the matched buffered track, or None.
        """
        current_x = current_coords.get("map_x")
        current_y = current_coords.get("map_y")
        if current_x is None or current_y is None:
            return None
        
        best_match_gid = None
        best_distance = float('inf')
        
        for buffer_camera_id, buffered_tracks in self._recent_tracks.items():
            # Only match against OTHER cameras
            if buffer_camera_id == current_camera_id:
                continue
            
            for bt in buffered_tracks:
                # Check TTL
                age = self._current_frame_num - bt.get("frame_num", 0)
                if age > self.buffer_ttl_frames:
                    continue
                
                bt_coords = bt.get("map_coords", {})
                bt_x = bt_coords.get("map_x")
                bt_y = bt_coords.get("map_y")
                if bt_x is None or bt_y is None:
                    continue
                
                # Calculate distance
                dist = math.sqrt((current_x - bt_x) ** 2 + (current_y - bt_y) ** 2)
                
                # Must be within threshold
                if dist <= self.threshold_meters and dist < best_distance:
                    bt_gid = bt.get("global_id")
                    if bt_gid:
                        best_distance = dist
                        best_match_gid = bt_gid
        
        return best_match_gid

    def _update_recent_tracks_buffer(self, valid_tracks: Dict[str, List[Dict[str, Any]]]):
        """
        Store current valid tracks (with their assigned global IDs) into the buffer.
        Also prune expired entries.
        """
        # Prune expired entries first
        cutoff_frame = self._current_frame_num - self.buffer_ttl_frames
        for cam_id in list(self._recent_tracks.keys()):
            self._recent_tracks[cam_id] = [
                bt for bt in self._recent_tracks[cam_id]
                if bt.get("frame_num", 0) > cutoff_frame
            ]
        
        # Add current tracks to buffer
        for camera_id, tracks in valid_tracks.items():
            for track in tracks:
                track_id = track.get("track_id")
                if track_id is None:
                    continue
                
                # Get the global ID that was just assigned
                gid = self.registry.get_global_id(camera_id, int(track_id))
                if not gid:
                    continue
                
                map_coords = track.get("map_coords", {})
                if not map_coords.get("map_x"):
                    continue
                
                # Buffer this track
                self._recent_tracks[camera_id].append({
                    "track_id": track_id,
                    "global_id": gid,
                    "map_coords": map_coords,
                    "frame_num": self._current_frame_num
                })

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
                            if dist < 5000.0 and {cam_a, cam_b} == {"c09", "c16"}:
                                gid_a = track_a.get("global_id") or self.registry.get_global_id(cam_a, int(track_a.get('track_id', 0)))
                                gid_b = track_b.get("global_id") or self.registry.get_global_id(cam_b, int(track_b.get('track_id', 0)))
                                spatial_debug(f"Distance: {cam_a}:{gid_a} vs {cam_b}:{gid_b} = {dist:.2f}px (threshold={self.threshold_meters})")

                            # Hard cap: if distance exceeds no_match_distance, completely exclude
                            # This prevents false matches between people in non-overlapping camera views
                            if dist > self.no_match_distance:
                                continue  # Leave as infinity - never match these

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

    def _update_global_ids(self, matches: List[Tuple[str, str, str, str]], active_global_ids: Dict[str, Set[str]]):
        for cam_a, track_a_id, cam_b, track_b_id in matches:
            key_a = (cam_a, str(track_a_id)) # Ensure string track ID
            key_b = (cam_b, str(track_b_id))
            
            # Retrieve current global IDs from registry
            existing_global_a = self.registry.get_global_id(*key_a)
            existing_global_b = self.registry.get_global_id(*key_b)
            
            if existing_global_a and existing_global_b:
                if existing_global_a != existing_global_b:
                    # MERGE with conflict check
                    target_id = existing_global_a if existing_global_a < existing_global_b else existing_global_b
                    source_id = existing_global_b if target_id == existing_global_a else existing_global_a
                    
                    # Check if merging would cause a camera conflict
                    # A conflict exists if BOTH identities are active in the SAME camera set
                    cameras_target = active_global_ids.get(target_id, set())
                    cameras_source = active_global_ids.get(source_id, set())
                    
                    overlap = cameras_target.intersection(cameras_source)
                    if overlap:
                        # logger.warning(
                        #     f"[SpaceMatcher] ⚠️  MERGE ABORTED: {target_id} and {source_id} coexist in cameras {overlap}. "
                        #     f"Skipping merge to prevent duplicate ID assignment."
                        # )
                        continue

                    self.registry.merge_identities(target_global_id=target_id, source_global_id=source_id)
                    
                    # Update active map to reflect merge (source becomes target)
                    active_global_ids[target_id].update(cameras_source)
                    if source_id in active_global_ids:
                        del active_global_ids[source_id]
            
            elif existing_global_a:
                self.registry.assign_identity(cam_b, track_b_id, existing_global_a)
                # Update active map
                if existing_global_a in active_global_ids:
                    active_global_ids[existing_global_a].add(cam_b)
                else:
                    active_global_ids[existing_global_a] = {cam_b}
                
            elif existing_global_b:
                self.registry.assign_identity(cam_a, track_a_id, existing_global_b)
                # Update active map
                if existing_global_b in active_global_ids:
                     active_global_ids[existing_global_b].add(cam_a)
                else:
                     active_global_ids[existing_global_b] = {cam_a}
                
            else:
                new_global_id = self.registry.allocate_new_id()
                self.registry.assign_identity(cam_a, track_a_id, new_global_id)
                self.registry.assign_identity(cam_b, track_b_id, new_global_id)
                
                # Update active map
                active_global_ids[new_global_id] = {cam_a, cam_b}
                
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
