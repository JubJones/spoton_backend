import logging
import uuid
from typing import Dict, Set, Optional, Tuple, List
from collections import defaultdict

logger = logging.getLogger(__name__)

class GlobalPersonRegistry:
    """
    Centralized registry for managing global person identities across multiple cameras.
    Handles ID assignment, retrieval, and merging of identities.
    """
    _id_counter = 0  # Class-level counter for simple numbered IDs
    
    def __init__(self):
        # Map (camera_id, local_track_id) -> global_id
        self.global_id_map: Dict[Tuple[str, str], str] = {}
        
        # Reverse map: global_id -> Set[(camera_id, local_track_id)]
        self.global_id_assignments: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        
        logger.info("GlobalPersonRegistry initialized")

    def get_global_id(self, camera_id: str, track_id: int) -> Optional[str]:
        """Get the global ID assigned to a local track, if any."""
        key = (camera_id, str(track_id))
        return self.global_id_map.get(key)
        
    def allocate_new_id(self) -> str:
        """Generate a new unique global ID (simple numbered format: P1, P2, etc.)."""
        GlobalPersonRegistry._id_counter += 1
        return f"P{GlobalPersonRegistry._id_counter}"

    def force_assign_identity(self, camera_id: str, track_id: int, global_id: str):
        """
        Force assign a specific global_id to a local track, bypassing merge logic.
        Used for conflict resolution (splitting) where we want to change ONE track's ID
        without affecting others that might share the old ID.
        """
        key = (camera_id, str(track_id))
        self._set_mapping(key, global_id)

    def assign_identity(self, camera_id: str, track_id: int, global_id: str):
        """
        Assign a specific global_id to a local track.
        If the track already has a DIFFERENT global_id, this triggers a MERGE.
        """
        key = (camera_id, str(track_id))
        existing_id = self.global_id_map.get(key)
        
        if existing_id == global_id:
            return  # No change
            
        if existing_id:
            # Track already has an ID, but we want to assign a NEW one. 
            # This implies the two IDs are actually the same person.
            # We should merge the "old" ID into the "new" (or specified) ID.
            logger.info(f"Registry: Re-assigning {key} from {existing_id} to {global_id}. Merging identities.")
            self.merge_identities(target_global_id=global_id, source_global_id=existing_id)
        else:
            # Fresh assignment
            self._set_mapping(key, global_id)

    def merge_identities(self, target_global_id: str, source_global_id: str):
        """
        Merge source_global_id into target_global_id.
        All tracks currently assigned to source_global_id will be reassigned to target_global_id.
        """
        if target_global_id == source_global_id:
            return

        logger.info(f"Registry: Merging global identity {source_global_id} into {target_global_id}")
        
        # Get all tracks belonging to the source ID
        tracks_to_move = list(self.global_id_assignments[source_global_id]) # Copy list
        
        logger.info(f"[REGISTRY] 🔀 MERGE: Global ID {source_global_id} ({len(tracks_to_move)} tracks) merging into {target_global_id}")
        
        for key in tracks_to_move:
            self._set_mapping(key, target_global_id)
        
        # Clear the source ID record
        if source_global_id in self.global_id_assignments:
            del self.global_id_assignments[source_global_id]

    def _set_mapping(self, key: Tuple[str, str], global_id: str):
        """Internal helper to set map and reverse map."""
        # Remove from old assignment if exists (should be handled by caller logic, but safety check)
        old_id = self.global_id_map.get(key)
        if old_id and old_id != global_id:
             self.global_id_assignments[old_id].discard(key)
             if not self.global_id_assignments[old_id]:
                 del self.global_id_assignments[old_id]

        self.global_id_map[key] = global_id
        self.global_id_assignments[global_id].add(key)

    def is_global_id_shared(self, global_id: str) -> bool:
        """Check if a global ID is currently assigned to tracks in multiple cameras."""
        if not global_id:
            return False
        
        assignments = self.global_id_assignments.get(global_id, set())
        unique_cameras = {cam_id for cam_id, _ in assignments}
        return len(unique_cameras) > 1
