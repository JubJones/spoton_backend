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
from app.core.config import settings # Import main app settings

logger = logging.getLogger(__name__)

class ReIDStateManager:
    """
    Manages the state required for cross-camera Re-Identification,
    incorporating logic for explicit Re-ID triggering, handoff influence,
    and advanced conflict resolution.
    """

    def __init__(self, task_id: uuid.UUID):
        """Initializes the ReIDStateManager for a given processing task."""
        self.task_id = task_id
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        self.global_id_last_seen_frame: Dict[GlobalID, int] = {}
        self.track_last_reid_frame: Dict[TrackKey, int] = {} # Frame ReID was last attempted

        logger.info(f"[Task {task_id}] ReIDStateManager initialized with new logic.")
        logger.info(f"  ReID Refresh Interval: {settings.REID_REFRESH_INTERVAL_FRAMES} frames")
        logger.info(f"  ReID Lost Buffer: {settings.REID_LOST_TRACK_BUFFER_FRAMES} frames")
        logger.info(f"  ReID Similarity Threshold: {settings.REID_SIMILARITY_THRESHOLD}")
        logger.info(f"  Possible Camera Overlaps for Handoff Influence: {settings.normalized_possible_camera_overlaps}")


    def get_new_global_id(self) -> GlobalID:
        """Generates a new unique GlobalID (string UUID)."""
        return GlobalID(str(uuid.uuid4()))

    def _normalize_embedding(self, embedding: Optional[FeatureVector]) -> Optional[FeatureVector]:
        """Normalizes a feature vector (L2 normalization). Handles None or zero vectors."""
        if embedding is None:
            return None
        # Ensure embedding is a NumPy array if it's a list (from TrackedObjectData)
        if isinstance(embedding, list):
            embedding_np = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, np.ndarray):
            embedding_np = embedding
        else:
            logger.warning(f"[Task {self.task_id}] Invalid type for embedding: {type(embedding)}. Cannot normalize.")
            return None

        norm = np.linalg.norm(embedding_np)
        if norm < 1e-6:
            return embedding_np # Or return None / np.zeros_like(embedding_np)
        return FeatureVector(embedding_np / norm)


    def _calculate_similarity_matrix(
        self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray
    ) -> Optional[np.ndarray]:
        """Calculates cosine similarity. Expects normalized embeddings."""
        if query_embeddings.ndim == 1: query_embeddings = query_embeddings.reshape(1, -1)
        if gallery_embeddings.ndim == 1: gallery_embeddings = gallery_embeddings.reshape(1, -1)

        if query_embeddings.size == 0 or gallery_embeddings.size == 0:
            return None
        try:
            distance_matrix = cdist(query_embeddings, gallery_embeddings, metric='cosine')
            similarity_matrix = 1.0 - distance_matrix
            return np.clip(similarity_matrix, 0.0, 1.0)
        except Exception as e:
            logger.error(f"[Task {self.task_id}] Batched similarity calculation failed: {e}", exc_info=True)
            return None

    def _should_attempt_reid_for_track(
        self,
        track_key: TrackKey,
        frame_processed_count: int,
        is_triggering_handoff: bool = False # Placeholder for future full handoff logic
    ) -> bool:
        """
        Decides if Re-ID should be attempted for a track based on its state.
        """
        is_known = track_key in self.track_to_global_id
        last_attempt_frame = self.track_last_reid_frame.get(track_key, -settings.REID_REFRESH_INTERVAL_FRAMES -1)
        is_due_for_refresh = (frame_processed_count - last_attempt_frame) >= settings.REID_REFRESH_INTERVAL_FRAMES

        if not is_known:
            logger.debug(f"[Task {self.task_id}][{track_key}] Should Re-ID: Track is new/unknown.")
            return True
        if is_due_for_refresh:
            logger.debug(f"[Task {self.task_id}][{track_key}] Should Re-ID: Track due for refresh (last attempt: {last_attempt_frame}, current: {frame_processed_count}).")
            return True
        if is_triggering_handoff: # Currently not used actively from MultiCameraFrameProcessor
            logger.debug(f"[Task {self.task_id}][{track_key}] Should Re-ID: Track is triggering handoff.")
            return True
        
        logger.debug(f"[Task {self.task_id}][{track_key}] Skipping Re-ID: Known, not due for refresh, not triggering handoff.")
        return False

    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        """
        Gets the target camera and any cameras configured to possibly overlap with it.
        Used for conceptual handoff influence on Re-ID.
        """
        relevant_cams = {target_cam_id}
        for c1_str, c2_str in settings.normalized_possible_camera_overlaps:
            # Ensure comparison with CameraID type if necessary, though str should work
            c1, c2 = CameraID(c1_str), CameraID(c2_str)
            if c1 == target_cam_id: relevant_cams.add(c2)
            elif c2 == target_cam_id: relevant_cams.add(c1)
        return relevant_cams

    def _apply_handoff_filter(
        self,
        track_key: TrackKey,
        matched_gid: GlobalID,
        # active_triggers_map: Optional[Dict[TrackKey, Any]] = None # Full trigger info if available
    ) -> bool:
        """
        Conceptual filter based on camera overlaps.
        If a track is being considered for a GID, checks if that GID was last seen
        in a camera "topologically consistent" (i.e., overlapping) with the current track's camera.
        This is a simplified version of POC's handoff filter.
        """
        current_cam_id, _ = track_key
        # For this simplified filter, the "target" camera of a handoff is implicitly the current camera.
        # We check if the GID's last known location is compatible with appearing on current_cam_id.
        relevant_cams_for_current = self._get_relevant_handoff_cams(current_cam_id)
        
        last_seen_cam_for_gid = self.global_id_last_seen_cam.get(matched_gid)

        if last_seen_cam_for_gid is not None and last_seen_cam_for_gid != current_cam_id:
            # If GID was last seen on a different camera, that camera must be relevant (overlapping)
            # or it's a new appearance on a non-overlapping camera (which is allowed).
            # The filter is to *prevent* a match if last_seen_cam_for_gid is NOT relevant
            # AND they are different cameras.
            # This is subtle: we are checking if the GID *could have* handed off to current_cam_id.
            if last_seen_cam_for_gid not in relevant_cams_for_current:
                logger.debug(
                    f"[Task {self.task_id}] Handoff Filter: IGNORING match of {track_key} to GID {matched_gid} "
                    f"(last seen on {last_seen_cam_for_gid}, which is not in relevant set {relevant_cams_for_current} for {current_cam_id})."
                )
                return False # Filter out: GID's last position is inconsistent with appearing here via overlap.
        
        return True # Allow: Same camera, or last_seen_cam is relevant/overlapping, or GID is new to this area.

    def _update_gallery_with_ema(
        self,
        gid: GlobalID,
        new_embedding: FeatureVector, # Assumed normalized
        source_track_key: TrackKey,
        matched_gallery_type: str # 'lost' or 'main'
    ):
        """Updates the gallery embedding for a GID using EMA."""
        alpha = settings.REID_GALLERY_EMA_ALPHA

        if matched_gallery_type == 'lost':
            if gid in self.lost_track_gallery:
                lost_embedding, _ = self.lost_track_gallery.pop(gid)
                # Ensure lost_embedding is normalized before EMA
                norm_lost_emb = self._normalize_embedding(lost_embedding)
                if norm_lost_emb is not None:
                    updated_embedding = (alpha * norm_lost_emb + (1.0 - alpha) * new_embedding)
                    self.reid_gallery[gid] = self._normalize_embedding(updated_embedding)
                else: # Should not happen if lost gallery stores valid features
                    self.reid_gallery[gid] = new_embedding
                logger.debug(f"[Task {self.task_id}] GID {gid} from lost gallery updated with EMA from {source_track_key}, moved to main.")
            elif gid in self.reid_gallery: # Already re-appeared
                current_gallery_emb = self.reid_gallery[gid]
                updated_embedding = (alpha * current_gallery_emb + (1.0 - alpha) * new_embedding)
                self.reid_gallery[gid] = self._normalize_embedding(updated_embedding)
            else: # Should not typically happen if matched_gallery_type is correct
                self.reid_gallery[gid] = new_embedding
        elif matched_gallery_type == 'main':
            if gid in self.reid_gallery:
                current_gallery_emb = self.reid_gallery[gid]
                updated_embedding = (alpha * current_gallery_emb + (1.0 - alpha) * new_embedding)
                self.reid_gallery[gid] = self._normalize_embedding(updated_embedding)
            else: # GID disappeared from gallery (e.g., pruning race condition)
                self.reid_gallery[gid] = new_embedding
                logger.warning(f"[Task {self.task_id}] GID {gid} matched 'main' but not in gallery. Added feature from {source_track_key}.")
        else: # New GID or unknown type
             self.reid_gallery[gid] = new_embedding # Add directly

    async def associate_features_and_update_state(
        self,
        all_tracks_with_features: Dict[TrackKey, FeatureVector], # TrackKey -> Potentially unnormalized FeatureVector
        active_track_keys_this_frame: Set[TrackKey],
        frame_processed_count: int
    ):
        """
        Orchestrates the full Re-ID association process for a batch of features,
        including decision making, matching, conflict resolution, and state updates.
        This method directly updates the ReIDStateManager's internal state.
        """
        logger.debug(f"[Task {self.task_id}] Starting Re-ID association for frame {frame_processed_count}. Total features received: {len(all_tracks_with_features)}.")

        # 1. Decide which tracks to attempt Re-ID for and get their normalized features
        query_features: Dict[TrackKey, FeatureVector] = {}
        for tk, feat in all_tracks_with_features.items():
            # Handoff trigger placeholder: in a full system, this would come from handoff detection logic
            is_triggering_handoff_conceptual = False # For now, set to False
            if self._should_attempt_reid_for_track(tk, frame_processed_count, is_triggering_handoff_conceptual):
                norm_feat = self._normalize_embedding(feat)
                if norm_feat is not None:
                    query_features[tk] = norm_feat
                    self.track_last_reid_frame[tk] = frame_processed_count
        
        if not query_features:
            logger.debug(f"[Task {self.task_id}] No tracks selected for Re-ID attempt this frame.")
            # Still update gallery lifecycle for tracks that might have disappeared
            await self.update_galleries_lifecycle(active_track_keys_this_frame, frame_processed_count)
            return

        logger.info(f"[Task {self.task_id}] Attempting Re-ID for {len(query_features)} tracks.")
        query_track_keys_list = list(query_features.keys())
        query_embeddings_np = np.array([query_features[tk] for tk in query_track_keys_list], dtype=np.float32)

        # 2. Prepare Gallery (Main + Lost)
        gallery_gids: List[GlobalID] = []
        gallery_embeddings_list: List[FeatureVector] = []
        gallery_types: List[str] = [] # 'lost' or 'main'

        for gid, (feat, _) in self.lost_track_gallery.items():
            norm_feat = self._normalize_embedding(feat) # Lost gallery features also need normalization
            if norm_feat is not None:
                gallery_gids.append(gid); gallery_embeddings_list.append(norm_feat); gallery_types.append('lost')
        
        num_lost_gallery_items = len(gallery_gids)
        for gid, feat in self.reid_gallery.items(): # Main gallery features assumed normalized on add/update
            if gid not in gallery_gids: # Avoid duplicates if GID was in lost
                gallery_gids.append(gid); gallery_embeddings_list.append(feat); gallery_types.append('main')
        
        gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32) if gallery_embeddings_list else np.empty((0,0), dtype=np.float32)

        # 3. Initial Assignments (Tentative)
        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float, str]] = {} # tk -> (gid, score, type)
        
        if gallery_embeddings_np.size > 0:
            similarity_matrix = self._calculate_similarity_matrix(query_embeddings_np, gallery_embeddings_np)
            if similarity_matrix is not None:
                best_match_indices = np.argmax(similarity_matrix, axis=1)
                max_similarity_scores = similarity_matrix[np.arange(len(query_track_keys_list)), best_match_indices]

                for i, tk in enumerate(query_track_keys_list):
                    best_gallery_idx = best_match_indices[i]
                    max_sim = float(max_similarity_scores[i])
                    
                    if max_sim >= settings.REID_SIMILARITY_THRESHOLD:
                        matched_gid = gallery_gids[best_gallery_idx]
                        matched_type = gallery_types[best_gallery_idx]
                        if self._apply_handoff_filter(tk, matched_gid):
                            tentative_assignments[tk] = (matched_gid, max_sim, matched_type)
                        else: # Handoff filter rejected
                            tentative_assignments[tk] = (None, max_sim, "filtered_handoff")
                    else: # Below threshold
                        tentative_assignments[tk] = (None, max_sim, "below_threshold")
        
        # 4. Assign New GIDs to Unmatched query tracks
        current_assignments = tentative_assignments.copy() # tk -> (gid, score, type)
        for i, tk in enumerate(query_track_keys_list):
            if tk not in current_assignments or current_assignments[tk][0] is None:
                new_gid = self.get_new_global_id()
                current_assignments[tk] = (new_gid, -1.0, "new") # Score -1 for new
                self.reid_gallery[new_gid] = query_embeddings_np[i] # Add normalized feature
                self.global_id_last_seen_frame[new_gid] = frame_processed_count
                self.global_id_last_seen_cam[new_gid] = tk[0]
                logger.info(f"[Task {self.task_id}] Assigned NEW GID {new_gid} to {tk}.")

        # 5. Resolve Intra-Camera Conflicts
        assignments_by_cam_gid: Dict[CameraID, Dict[GlobalID, List[Tuple[TrackKey, float, str]]]] = defaultdict(lambda: defaultdict(list))
        for tk, (gid, score, type_str) in current_assignments.items():
            if gid is not None:
                assignments_by_cam_gid[tk[0]][gid].append((tk, score, type_str))

        reverted_keys_for_second_pass: List[TrackKey] = []
        for cam_id, gid_map in assignments_by_cam_gid.items():
            for gid, track_score_list in gid_map.items():
                if len(track_score_list) > 1: # Conflict!
                    track_score_list.sort(key=lambda x: x[1], reverse=True) # Sort by score desc
                    winner_tk, winner_score, winner_type = track_score_list[0]
                    logger.warning(
                        f"[Task {self.task_id}][{cam_id}] Conflict for GID {gid}. Tracks: "
                        f"{[(ts[0][1], f'{ts[1]:.2f}') for ts in track_score_list]}. Keeping T:{winner_tk[1]}."
                    )
                    # Update current_assignments: winner is kept, others are reverted
                    for i in range(1, len(track_score_list)):
                        reverted_tk, _, _ = track_score_list[i]
                        current_assignments[reverted_tk] = (None, -1.0, "reverted_conflict")
                        reverted_keys_for_second_pass.append(reverted_tk)
        
        # 6. Finalize Non-Reverted Assignments & Update State/Gallery (EMA)
        for tk, (gid, score, type_str) in current_assignments.items():
            if gid is not None and tk not in reverted_keys_for_second_pass : # Is assigned and not reverted
                self.track_to_global_id[tk] = gid
                self.global_id_last_seen_cam[gid] = tk[0]
                self.global_id_last_seen_frame[gid] = frame_processed_count
                
                # Update gallery with EMA if it was a match (not 'new' or 'reverted_conflict')
                # For 'new' GIDs, feature was already added.
                if type_str not in ["new", "reverted_conflict", "filtered_handoff", "below_threshold"]:
                    query_idx = query_track_keys_list.index(tk)
                    self._update_gallery_with_ema(gid, query_embeddings_np[query_idx], tk, type_str)

        # 7. Handle Reverted Tracks (Second Pass Matching)
        if reverted_keys_for_second_pass:
            logger.info(f"[Task {self.task_id}] Starting second pass Re-ID for {len(reverted_keys_for_second_pass)} reverted tracks.")
            # Re-prepare gallery (might have changed slightly due to new GID additions)
            # This is a simplified gallery prep for the second pass.
            current_main_gallery_gids = list(self.reid_gallery.keys())
            current_main_gallery_embeds_np = np.array([self.reid_gallery[g] for g in current_main_gallery_gids], dtype=np.float32) if current_main_gallery_gids else np.empty((0,0))

            for tk_reverted in reverted_keys_for_second_pass:
                reverted_query_idx = query_track_keys_list.index(tk_reverted)
                reverted_embedding = query_embeddings_np[reverted_query_idx]
                
                assigned_in_second_pass = False
                if current_main_gallery_embeds_np.size > 0:
                    sim_matrix_2pass = self._calculate_similarity_matrix(reverted_embedding.reshape(1, -1), current_main_gallery_embeds_np)
                    if sim_matrix_2pass is not None and sim_matrix_2pass.size > 0:
                        best_match_idx_2pass = np.argmax(sim_matrix_2pass[0])
                        max_sim_2pass = float(sim_matrix_2pass[0, best_match_idx_2pass])
                        
                        if max_sim_2pass >= settings.REID_SIMILARITY_THRESHOLD:
                            gid_2pass = current_main_gallery_gids[best_match_idx_2pass]
                            
                            # Avoid re-assigning to the GID that caused original conflict if it was the winner on this camera
                            can_assign_gid_2pass = True
                            original_conflict_gid_winner_on_cam = assignments_by_cam_gid[tk_reverted[0]].get(gid_2pass)
                            if original_conflict_gid_winner_on_cam:
                                if any(entry[0] != tk_reverted for entry in original_conflict_gid_winner_on_cam): # if gid_2pass was winner for *another* track on this cam
                                    can_assign_gid_2pass = False 
                                    logger.debug(f"[Task {self.task_id}][{tk_reverted}] 2nd pass: GID {gid_2pass} was winner for another track in this cam. Cannot assign.")

                            if can_assign_gid_2pass and self._apply_handoff_filter(tk_reverted, gid_2pass):
                                self.track_to_global_id[tk_reverted] = gid_2pass
                                self.global_id_last_seen_cam[gid_2pass] = tk_reverted[0]
                                self.global_id_last_seen_frame[gid_2pass] = frame_processed_count
                                self._update_gallery_with_ema(gid_2pass, reverted_embedding, tk_reverted, "main") # Assume matched main gallery
                                assigned_in_second_pass = True
                                logger.info(f"[Task {self.task_id}][{tk_reverted}] Second pass SUCCESS: Matched GID {gid_2pass} (Sim: {max_sim_2pass:.2f}).")
                
                if not assigned_in_second_pass:
                    new_gid_for_reverted = self.get_new_global_id()
                    self.track_to_global_id[tk_reverted] = new_gid_for_reverted
                    self.reid_gallery[new_gid_for_reverted] = reverted_embedding
                    self.global_id_last_seen_cam[new_gid_for_reverted] = tk_reverted[0]
                    self.global_id_last_seen_frame[new_gid_for_reverted] = frame_processed_count
                    logger.info(f"[Task {self.task_id}][{tk_reverted}] Second pass: Assigned NEW GID {new_gid_for_reverted}.")

        # Lifecycle update handled separately by `update_galleries_lifecycle`
        await self.update_galleries_lifecycle(active_track_keys_this_frame, frame_processed_count)


    async def update_galleries_lifecycle(self, active_track_keys_this_frame: Set[TrackKey], frame_processed_count: int):
        """Manages lost tracks and gallery pruning."""
        # 1. Handle Disappeared Tracks
        all_previously_known_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previously_known_track_keys - active_track_keys_this_frame

        for tk_disappeared in disappeared_track_keys:
            gid = self.track_to_global_id.pop(tk_disappeared, None)
            self.track_last_reid_frame.pop(tk_disappeared, None)

            if gid:
                feature = self.reid_gallery.get(gid) # Main gallery has normalized features
                if feature is not None:
                    if gid not in self.lost_track_gallery: # Add to lost only if not already there recently
                        last_active_frame = self.global_id_last_seen_frame.get(gid, frame_processed_count -1)
                        self.lost_track_gallery[gid] = (feature, last_active_frame)
                        logger.debug(f"[Task {self.task_id}] Track {tk_disappeared} (GID {gid}) disappeared. Moved to lost gallery from frame {last_active_frame}.")
        
        # 2. Purge Old Tracks from Lost Gallery
        expired_lost_gids = [
            gid for gid, (_, frame_added) in self.lost_track_gallery.items()
            if (frame_processed_count - frame_added) > settings.REID_LOST_TRACK_BUFFER_FRAMES
        ]
        for gid in expired_lost_gids:
            self.lost_track_gallery.pop(gid, None)
            logger.debug(f"[Task {self.task_id}] Purged GID {gid} from lost gallery (buffer {settings.REID_LOST_TRACK_BUFFER_FRAMES} frames).")

        # 3. Periodically Prune Main Gallery
        if frame_processed_count > 0 and (frame_processed_count % settings.REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES == 0):
            logger.info(f"[Task {self.task_id}] Performing main gallery pruning (Frame {frame_processed_count}). Current size: {len(self.reid_gallery)}.")
            gids_to_prune_main: List[GlobalID] = []
            prune_cutoff_frame = frame_processed_count - settings.REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES
            
            for gid, last_seen_f in list(self.global_id_last_seen_frame.items()):
                if last_seen_f < prune_cutoff_frame and gid not in self.lost_track_gallery:
                    is_active_now = any(active_gid == gid for active_gid in self.track_to_global_id.values()) # Check if somehow active
                    if not is_active_now:
                        gids_to_prune_main.append(gid)
            
            if gids_to_prune_main:
                logger.info(f"[Task {self.task_id}] Pruning {len(gids_to_prune_main)} GIDs from main gallery. Examples: {gids_to_prune_main[:3]}")
                for gid in gids_to_prune_main:
                    self.reid_gallery.pop(gid, None)
                    self.global_id_last_seen_cam.pop(gid, None)
                    self.global_id_last_seen_frame.pop(gid, None)
            logger.info(f"[Task {self.task_id}] Main gallery size after pruning: {len(self.reid_gallery)}.")