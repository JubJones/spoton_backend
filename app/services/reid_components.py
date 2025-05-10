# FILE: app/services/reid_components.py
"""
Components for managing Re-Identification (ReID) state and association logic
across multiple cameras. Adapted from reid_poc concepts.
"""
import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
from scipy.spatial.distance import cdist # For cosine similarity calculation

from app.common_types import CameraID, TrackID, GlobalID, FeatureVector, TrackKey
from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Constants for ReID (can be moved to config if needed) ---
REID_SIMILARITY_THRESHOLD = settings.REID_SIMILARITY_THRESHOLD # e.g., 0.65 from reid_poc
GALLERY_EMA_ALPHA = 0.9 # For updating gallery features
LOST_TRACK_BUFFER_FRAMES = 200 # Frames before a lost track is potentially purged
PRUNE_INTERVAL_FRAMES = 500 # How often to check for main gallery pruning
PRUNE_THRESHOLD_FRAMES = LOST_TRACK_BUFFER_FRAMES * 2 # Prune if unseen for this many frames

class ReIDStateManager:
    """Manages the state required for cross-camera Re-Identification."""

    def __init__(self, task_id: uuid.UUID):
        self.task_id = task_id
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {} # GID -> (Feature, frame_last_active)
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {} # (CamID, TrackID) -> GlobalID
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        self.global_id_last_seen_frame: Dict[GlobalID, int] = {} # Processed frame counter
        self.track_last_reid_frame: Dict[TrackKey, int] = {} # Last frame ReID was attempted
        self.next_global_id_counter: int = 1 # Simple counter for new GlobalIDs

        logger.info(f"[Task {task_id}] ReIDStateManager initialized.")

    def get_new_global_id(self) -> GlobalID:
        """Generates a new unique GlobalID for the task."""
        # Using UUIDs for global IDs to ensure uniqueness even if multiple tasks run
        new_id = GlobalID(str(uuid.uuid4()))
        # self.next_global_id_counter += 1 # Not strictly needed if using UUIDs
        return new_id

    def _normalize_embedding(self, embedding: Optional[FeatureVector]) -> Optional[FeatureVector]:
        """Normalizes a feature vector (L2 normalization). Handles None or zero vectors."""
        if embedding is None:
            return None
        norm = np.linalg.norm(embedding)
        if norm < 1e-6: # Avoid division by zero or near-zero
            return embedding # Or return None / np.zeros_like(embedding)
        return FeatureVector(embedding / norm)

    def add_to_gallery(self, global_id: GlobalID, feature_vector: FeatureVector, frame_processed_count: int, camera_id: CameraID):
        """Adds or updates a GID in the main gallery with EMA."""
        norm_feature = self._normalize_embedding(feature_vector)
        if norm_feature is None:
            logger.warning(f"[Task {self.task_id}] Attempted to add None or zero-norm feature for GID {global_id}.")
            return

        if global_id in self.reid_gallery:
            current_gallery_emb = self.reid_gallery[global_id]
            updated_embedding = (GALLERY_EMA_ALPHA * current_gallery_emb + (1.0 - GALLERY_EMA_ALPHA) * norm_feature)
            self.reid_gallery[global_id] = self._normalize_embedding(updated_embedding)
        else:
            self.reid_gallery[global_id] = norm_feature
        
        self.global_id_last_seen_cam[global_id] = camera_id
        self.global_id_last_seen_frame[global_id] = frame_processed_count
        # If it was in lost gallery, remove it
        if global_id in self.lost_track_gallery:
            self.lost_track_gallery.pop(global_id)
            logger.debug(f"[Task {self.task_id}] GID {global_id} moved from lost to main gallery.")


def calculate_similarity_matrix(
    query_embeddings: np.ndarray, gallery_embeddings: np.ndarray
) -> Optional[np.ndarray]:
    """Calculates cosine similarity matrix. Expects normalized embeddings."""
    if query_embeddings.ndim == 1: query_embeddings = query_embeddings.reshape(1, -1)
    if gallery_embeddings.ndim == 1: gallery_embeddings = gallery_embeddings.reshape(1, -1)

    if query_embeddings.size == 0 or gallery_embeddings.size == 0:
        return None
    try:
        # Cosine distance = 1 - cosine similarity.
        # cdist computes distance, so 1 - distance gives similarity.
        # Embeddings should be L2 normalized before this.
        distance_matrix = cdist(query_embeddings, gallery_embeddings, metric='cosine')
        similarity_matrix = 1.0 - distance_matrix
        return np.clip(similarity_matrix, 0.0, 1.0) # Clip to valid similarity range
    except Exception as e:
        logger.error(f"Batched similarity calculation failed: {e}", exc_info=True)
        return None

def associate_tracks_to_gallery(
    reid_manager: ReIDStateManager,
    tracks_with_features: Dict[TrackKey, FeatureVector], # TrackKey -> Normalized FeatureVector
    frame_processed_count: int # Current global processed frame counter for this task
) -> Dict[TrackKey, GlobalID]:
    """
    Associates active tracks with features to existing global IDs in the gallery
    or assigns new global IDs. Updates the ReIDStateManager.

    Args:
        reid_manager: The ReIDStateManager instance for this task.
        tracks_with_features: Dict where key is TrackKey and value is its feature vector.
                              Features are assumed to be ALREADY NORMALIZED.
        frame_processed_count: The current processed frame number for timestamping.

    Returns:
        A dictionary mapping TrackKey to its assigned/confirmed GlobalID.
    """
    if not tracks_with_features:
        return {}

    current_assignments: Dict[TrackKey, GlobalID] = {}
    
    query_track_keys: List[TrackKey] = list(tracks_with_features.keys())
    query_embeddings_list: List[FeatureVector] = [tracks_with_features[tk] for tk in query_track_keys]
    
    if not query_embeddings_list:
        logger.debug(f"[Task {reid_manager.task_id}] No valid query embeddings for ReID association.")
        return {}
        
    query_embeddings_np = np.array(query_embeddings_list, dtype=np.float32)

    # Prepare gallery (main + lost)
    gallery_gids: List[GlobalID] = []
    gallery_embeddings_list: List[FeatureVector] = []
    
    # Add lost gallery items first (higher priority for matching)
    for gid, (feat, _) in reid_manager.lost_track_gallery.items():
        norm_feat = reid_manager._normalize_embedding(feat)
        if norm_feat is not None:
            gallery_gids.append(gid)
            gallery_embeddings_list.append(norm_feat)

    # Add main gallery items
    main_gallery_gids_added_count = 0
    for gid, feat in reid_manager.reid_gallery.items():
        if gid not in gallery_gids: # Avoid duplicates if GID was in lost
            norm_feat = reid_manager._normalize_embedding(feat)
            if norm_feat is not None:
                gallery_gids.append(gid)
                gallery_embeddings_list.append(norm_feat)
                main_gallery_gids_added_count +=1
    
    num_lost_gallery_items = len(gallery_gids) - main_gallery_gids_added_count

    if not gallery_embeddings_list: # Gallery is empty
        logger.debug(f"[Task {reid_manager.task_id}] Gallery is empty. Assigning new GIDs to all query tracks.")
        for i, tk in enumerate(query_track_keys):
            new_gid = reid_manager.get_new_global_id()
            current_assignments[tk] = new_gid
            reid_manager.track_to_global_id[tk] = new_gid
            reid_manager.add_to_gallery(new_gid, query_embeddings_np[i], frame_processed_count, tk[0])
        return current_assignments

    gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32)
    similarity_matrix = calculate_similarity_matrix(query_embeddings_np, gallery_embeddings_np)

    if similarity_matrix is None:
        logger.warning(f"[Task {reid_manager.task_id}] Similarity matrix calculation failed. Assigning new GIDs.")
        for i, tk in enumerate(query_track_keys):
            new_gid = reid_manager.get_new_global_id()
            current_assignments[tk] = new_gid
            reid_manager.track_to_global_id[tk] = new_gid
            reid_manager.add_to_gallery(new_gid, query_embeddings_np[i], frame_processed_count, tk[0])
        return current_assignments

    # Iterate through query tracks and find best match
    # This is a simple greedy assignment. More complex methods like Hungarian algo could be used.
    assigned_gallery_indices: Set[int] = set()

    # Sort query tracks to process high-confidence ones first (if confidence available, else by order)
    # For now, processing in given order.
    
    temp_assignments: List[Tuple[float, int, int]] = [] # (score, query_idx, gallery_idx)

    for query_idx in range(similarity_matrix.shape[0]):
        for gallery_idx in range(similarity_matrix.shape[1]):
            score = similarity_matrix[query_idx, gallery_idx]
            if score >= REID_SIMILARITY_THRESHOLD:
                temp_assignments.append((score, query_idx, gallery_idx))
    
    # Sort by score descending to make greedy choices
    temp_assignments.sort(key=lambda x: x[0], reverse=True)

    final_assignments_this_batch: Dict[TrackKey, GlobalID] = {}

    for score, query_idx, gallery_idx in temp_assignments:
        track_key = query_track_keys[query_idx]
        gid_candidate = gallery_gids[gallery_idx]

        # If this query or this gallery GID already assigned in this batch, skip
        if track_key in final_assignments_this_batch or gallery_idx in assigned_gallery_indices:
            continue
        
        # Check for intra-camera conflict: if gid_candidate is already assigned to another track_id in the SAME camera
        conflict = False
        for tk_assigned, gid_assigned in final_assignments_this_batch.items():
            if tk_assigned[0] == track_key[0] and gid_assigned == gid_candidate: # Same camera, same GID candidate
                conflict = True
                logger.debug(f"[Task {reid_manager.task_id}] Intra-camera ReID conflict for GID {gid_candidate} on cam {track_key[0]}. Track {track_key} vs {tk_assigned}. Prioritizing higher score (handled by sort).")
                break
        if conflict:
            continue

        final_assignments_this_batch[track_key] = gid_candidate
        assigned_gallery_indices.add(gallery_idx)
        
        reid_manager.track_to_global_id[track_key] = gid_candidate
        reid_manager.add_to_gallery(gid_candidate, query_embeddings_np[query_idx], frame_processed_count, track_key[0])
        
        # If matched from lost gallery, it's now active in main gallery
        if gallery_idx < num_lost_gallery_items:
            logger.info(f"[Task {reid_manager.task_id}] Re-identified {track_key} as GID {gid_candidate} from LOST gallery (Sim: {score:.3f}).")
        else:
            logger.info(f"[Task {reid_manager.task_id}] Associated {track_key} with GID {gid_candidate} from MAIN gallery (Sim: {score:.3f}).")

    # Assign new GIDs to tracks that couldn't be matched
    for i, tk in enumerate(query_track_keys):
        if tk not in final_assignments_this_batch:
            new_gid = reid_manager.get_new_global_id()
            final_assignments_this_batch[tk] = new_gid
            reid_manager.track_to_global_id[tk] = new_gid
            reid_manager.add_to_gallery(new_gid, query_embeddings_np[i], frame_processed_count, tk[0])
            logger.info(f"[Task {reid_manager.task_id}] Assigned NEW Global ID {new_gid} to unmatched track {tk}.")
            
    return final_assignments_this_batch


def update_reid_state_after_frame(reid_manager: ReIDStateManager, active_track_keys_this_frame: Set[TrackKey], frame_processed_count: int):
    """Updates gallery states (lost, main pruning) after a frame batch is processed."""

    # 1. Handle disappeared tracks
    all_previously_known_track_keys = set(reid_manager.track_to_global_id.keys())
    disappeared_track_keys = all_previously_known_track_keys - active_track_keys_this_frame

    for tk_disappeared in disappeared_track_keys:
        gid = reid_manager.track_to_global_id.pop(tk_disappeared, None)
        reid_manager.track_last_reid_frame.pop(tk_disappeared, None) # Clean up

        if gid:
            # Move to lost gallery if feature exists in main
            feature = reid_manager.reid_gallery.get(gid)
            if feature is not None:
                # Only add if not already there from a very recent disappearance
                if gid not in reid_manager.lost_track_gallery:
                    reid_manager.lost_track_gallery[gid] = (feature, reid_manager.global_id_last_seen_frame.get(gid, frame_processed_count -1))
                    logger.debug(f"[Task {reid_manager.task_id}] Track {tk_disappeared} (GID {gid}) disappeared. Moved to lost gallery.")
            # else: No feature in main, GID might already have been pruned or is new and disappeared.

    # 2. Purge old tracks from lost gallery
    expired_lost_gids = [
        gid for gid, (_, frame_added) in reid_manager.lost_track_gallery.items()
        if (frame_processed_count - frame_added) > LOST_TRACK_BUFFER_FRAMES
    ]
    for gid in expired_lost_gids:
        reid_manager.lost_track_gallery.pop(gid, None)
        logger.debug(f"[Task {reid_manager.task_id}] Purged GID {gid} from lost gallery (buffer {LOST_TRACK_BUFFER_FRAMES} frames).")

    # 3. Periodically prune main gallery
    if frame_processed_count > 0 and (frame_processed_count % PRUNE_INTERVAL_FRAMES == 0):
        logger.info(f"[Task {reid_manager.task_id}] Performing main gallery pruning (Frame {frame_processed_count}). Current size: {len(reid_manager.reid_gallery)}.")
        gids_to_prune_main: List[GlobalID] = []
        prune_cutoff_frame = frame_processed_count - PRUNE_THRESHOLD_FRAMES
        
        # Candidate GIDs for pruning are those not seen recently and not in lost gallery
        for gid, last_seen_f in list(reid_manager.global_id_last_seen_frame.items()): # Iterate copy
            if last_seen_f < prune_cutoff_frame and gid not in reid_manager.lost_track_gallery:
                # Ensure it's not currently active either (should be covered by lost_track check, but good for safety)
                is_active_now = any(active_gid == gid for active_gid in reid_manager.track_to_global_id.values())
                if not is_active_now:
                    gids_to_prune_main.append(gid)
        
        if gids_to_prune_main:
            logger.info(f"[Task {reid_manager.task_id}] Pruning {len(gids_to_prune_main)} GIDs from main gallery. Examples: {gids_to_prune_main[:5]}")
            for gid in gids_to_prune_main:
                reid_manager.reid_gallery.pop(gid, None)
                reid_manager.global_id_last_seen_cam.pop(gid, None)
                reid_manager.global_id_last_seen_frame.pop(gid, None)
        logger.info(f"[Task {reid_manager.task_id}] Main gallery size after pruning: {len(reid_manager.reid_gallery)}.")