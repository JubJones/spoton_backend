"""
Components for managing Re-Identification (ReID) state and association logic
across multiple cameras. Adapted from reid_poc concepts.
"""
import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
from scipy.spatial.distance import cdist

from app.common_types import (
    CameraID, TrackID, GlobalID, FeatureVector, TrackKey,
    HandoffTriggerInfo # NEW IMPORT
)
from app.core.config import settings

logger = logging.getLogger(__name__)

class ReIDStateManager:
    """
    Manages the state required for cross-camera Re-Identification,
    incorporating logic for explicit Re-ID triggering, handoff influence,
    and advanced conflict resolution.
    """

    def __init__(self, task_id: uuid.UUID):
        self.task_id = task_id
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        self.global_id_last_seen_frame: Dict[GlobalID, int] = {}
        self.track_last_reid_frame: Dict[TrackKey, int] = {}

        logger.info(f"[Task {task_id}] ReIDStateManager initialized with handoff-aware logic.")
        logger.info(f"  ReID Refresh Interval: {settings.REID_REFRESH_INTERVAL_FRAMES} frames")
        logger.info(f"  ReID Lost Buffer: {settings.REID_LOST_TRACK_BUFFER_FRAMES} frames")
        logger.info(f"  ReID Similarity Threshold: {settings.REID_SIMILARITY_THRESHOLD}")
        logger.info(f"  Min BBox Overlap for Handoff: {settings.MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT}")
        logger.info(f"  Possible Camera Overlaps for Handoff Influence: {settings.normalized_possible_camera_overlaps}")


    def get_new_global_id(self) -> GlobalID:
        return GlobalID(str(uuid.uuid4()))

    def _normalize_embedding(self, embedding: Optional[FeatureVector]) -> Optional[FeatureVector]:
        if embedding is None: return None
        if isinstance(embedding, list): embedding_np = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, np.ndarray): embedding_np = embedding
        else: return None
        norm = np.linalg.norm(embedding_np)
        return FeatureVector(embedding_np / norm) if norm > 1e-6 else embedding_np

    def _calculate_similarity_matrix(
        self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray
    ) -> Optional[np.ndarray]:
        if query_embeddings.ndim == 1: query_embeddings = query_embeddings.reshape(1, -1)
        if gallery_embeddings.ndim == 1: gallery_embeddings = gallery_embeddings.reshape(1, -1)
        if query_embeddings.size == 0 or gallery_embeddings.size == 0: return None
        try:
            # Cosine distance: 0 for identical, 1 for orthogonal, 2 for opposite
            # Similarity = 1 - distance
            similarity_matrix = 1.0 - cdist(query_embeddings, gallery_embeddings, metric='cosine')
            return np.clip(similarity_matrix, 0.0, 1.0) # Clip to valid similarity range
        except Exception as e:
            logger.error(f"[Task {self.task_id}] Batched similarity calculation failed: {e}", exc_info=True)
            return None

    # --- MODIFIED: Adapted from POC's decide_reid_targets ---
    def _should_attempt_reid_for_track(
        self,
        track_key: TrackKey,
        frame_processed_count: int,
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo] # NEW param
    ) -> bool:
        """
        Decides if Re-ID should be attempted for a track based on its state and handoff triggers.
        """
        is_known = track_key in self.track_to_global_id
        last_attempt_frame = self.track_last_reid_frame.get(track_key, -settings.REID_REFRESH_INTERVAL_FRAMES - 1)
        is_due_for_refresh = (frame_processed_count - last_attempt_frame) >= settings.REID_REFRESH_INTERVAL_FRAMES
        
        # Check if this track_key is part of an active handoff trigger
        is_triggering_handoff = track_key in active_triggers_map

        if not is_known:
            logger.debug(f"[Task {self.task_id}][{track_key}] Should Re-ID: Track is new/unknown.")
            return True
        if is_due_for_refresh:
            logger.debug(f"[Task {self.task_id}][{track_key}] Should Re-ID: Track due for refresh.")
            return True
        if is_triggering_handoff:
            logger.info(f"[Task {self.task_id}][{track_key}] Should Re-ID: Track IS ACTIVELY TRIGGERING a handoff rule.")
            return True
        
        # logger.debug(f"[Task {self.task_id}][{track_key}] Skipping Re-ID: Known, not due for refresh, not triggering handoff.")
        return False

    # --- MODIFIED: Adapted from POC's _get_relevant_handoff_cams and _apply_handoff_filter ---
    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        """Gets the target camera and any cameras configured to possibly overlap with it."""
        relevant_cams = {target_cam_id}
        # Use normalized_possible_camera_overlaps which returns Set[Tuple[CameraID, CameraID]]
        for c1, c2 in settings.normalized_possible_camera_overlaps:
            if c1 == target_cam_id: relevant_cams.add(c2)
            elif c2 == target_cam_id: relevant_cams.add(c1)
        return relevant_cams

    def _apply_handoff_filter_for_match(
        self,
        query_track_key: TrackKey,
        matched_gid: GlobalID,
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> bool:
        """
        Checks if a potential ReID match should be filtered based on handoff context.
        This version is from POC: filters if an *active trigger* for query_track_key
        targets a camera, but the matched_gid's last known location is inconsistent with that target.
        """
        trigger_info = active_triggers_map.get(query_track_key)
        
        if not trigger_info:
            # If no active trigger FOR THIS TRACK, the default SpotOn logic applies:
            # Check if the GID's last known cam is generally plausible for an appearance on current cam.
            current_cam_id, _ = query_track_key
            relevant_cams_for_current = self._get_relevant_handoff_cams(current_cam_id)
            last_seen_cam_for_gid = self.global_id_last_seen_cam.get(matched_gid)

            if last_seen_cam_for_gid is not None and last_seen_cam_for_gid != current_cam_id:
                if last_seen_cam_for_gid not in relevant_cams_for_current:
                    logger.debug(
                        f"[Task {self.task_id}] Handoff Filter (No Active Trigger): IGNORING match of {query_track_key} to GID {matched_gid} "
                        f"(last seen on {last_seen_cam_for_gid}, not in relevant set {relevant_cams_for_current} for {current_cam_id})."
                    )
                    return False # Filter out: GID's last position is inconsistent with appearing here via overlap.
            return True # Allow if no active trigger and general overlap is plausible or GID is new to area.

        # If there IS an active trigger for this track_key:
        # The trigger rule specifies a target_cam_id for the handoff.
        # We check if the matched_gid was last seen in a camera relevant to THIS TRIGGER'S TARGET.
        trigger_target_cam_id = trigger_info.rule.target_cam_id
        relevant_cams_for_trigger_target = self._get_relevant_handoff_cams(trigger_target_cam_id)
        
        last_seen_cam_for_matched_gid = self.global_id_last_seen_cam.get(matched_gid)

        if last_seen_cam_for_matched_gid is not None:
            # If the GID was last seen, and it wasn't on a camera relevant to the trigger's target,
            # then this GID is not where the handoff expects it to be.
            if last_seen_cam_for_matched_gid not in relevant_cams_for_trigger_target:
                logger.info(
                    f"[Task {self.task_id}] Handoff Filter (Active Trigger): REJECTING match of {query_track_key} "
                    f"to GID {matched_gid} (last seen on {last_seen_cam_for_matched_gid}). "
                    f"Trigger expected GID near {trigger_target_cam_id} (relevant: {relevant_cams_for_trigger_target})."
                )
                return False # Filter out: This GID is not where the handoff expects it.
        
        # Allow if GID was new, or last seen on a camera relevant to the trigger's target.
        logger.debug(
            f"[Task {self.task_id}] Handoff Filter (Active Trigger): ALLOWING match of {query_track_key} to GID {matched_gid} "
            f"(last seen {last_seen_cam_for_matched_gid}, trigger target {trigger_target_cam_id}, relevant {relevant_cams_for_trigger_target})."
        )
        return True

    def _update_gallery_with_ema(
        self,
        gid: GlobalID,
        new_embedding: FeatureVector, # Assumed normalized
        source_track_key: TrackKey,
        matched_gallery_type: str # 'lost' or 'main'
    ):
        alpha = settings.REID_GALLERY_EMA_ALPHA
        current_gallery_embedding: Optional[FeatureVector] = None

        if matched_gallery_type == 'lost':
            if gid in self.lost_track_gallery:
                lost_embedding, _ = self.lost_track_gallery.pop(gid) # Remove from lost
                current_gallery_embedding = self._normalize_embedding(lost_embedding)
                logger.debug(f"[Task {self.task_id}] GID {gid} from lost gallery (source {source_track_key}) being merged into main.")
            elif gid in self.reid_gallery: # Already re-appeared and in main
                current_gallery_embedding = self.reid_gallery.get(gid)
        elif matched_gallery_type == 'main':
            current_gallery_embedding = self.reid_gallery.get(gid)
        
        if current_gallery_embedding is not None:
            updated_embedding = (alpha * current_gallery_embedding + (1.0 - alpha) * new_embedding)
            self.reid_gallery[gid] = self._normalize_embedding(updated_embedding)
        else: # GID not found in expected gallery (e.g. new, or race condition)
            self.reid_gallery[gid] = new_embedding # Add directly
            logger.warning(f"[Task {self.task_id}] GID {gid} (matched as {matched_gallery_type}) not in expected gallery. Added feature from {source_track_key} directly.")


    async def associate_features_and_update_state(
        self,
        all_tracks_with_features: Dict[TrackKey, FeatureVector],
        active_track_keys_this_frame: Set[TrackKey],
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo], # NEW param
        frame_processed_count: int
    ):
        logger.debug(f"[Task {self.task_id}] Starting Re-ID association for frame {frame_processed_count}. Features: {len(all_tracks_with_features)}, Active Triggers: {len(active_triggers_map)}")

        query_features: Dict[TrackKey, FeatureVector] = {}
        for tk, feat in all_tracks_with_features.items():
            if self._should_attempt_reid_for_track(tk, frame_processed_count, active_triggers_map):
                norm_feat = self._normalize_embedding(feat)
                if norm_feat is not None:
                    query_features[tk] = norm_feat
                    self.track_last_reid_frame[tk] = frame_processed_count
        
        if not query_features:
            await self.update_galleries_lifecycle(active_track_keys_this_frame, frame_processed_count)
            return

        logger.info(f"[Task {self.task_id}] Attempting Re-ID for {len(query_features)} tracks (with handoff context).")
        query_track_keys_list = list(query_features.keys())
        query_embeddings_np = np.array([query_features[tk] for tk in query_track_keys_list], dtype=np.float32)

        gallery_gids: List[GlobalID] = []
        gallery_embeddings_list: List[FeatureVector] = []
        gallery_types: List[str] = []

        for gid, (feat, _) in self.lost_track_gallery.items():
            norm_feat = self._normalize_embedding(feat)
            if norm_feat is not None:
                gallery_gids.append(gid); gallery_embeddings_list.append(norm_feat); gallery_types.append('lost')
        
        for gid, feat in self.reid_gallery.items():
            if gid not in gallery_gids: # Avoid duplicates
                gallery_gids.append(gid); gallery_embeddings_list.append(feat); gallery_types.append('main')
        
        gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32) if gallery_embeddings_list else np.empty((0,0), dtype=np.float32)

        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float, str]] = {} # tk -> (gid, score, type)
        
        if gallery_embeddings_np.size > 0 and query_embeddings_np.size > 0:
            similarity_matrix = self._calculate_similarity_matrix(query_embeddings_np, gallery_embeddings_np)
            if similarity_matrix is not None:
                best_match_indices = np.argmax(similarity_matrix, axis=1)
                max_similarity_scores = similarity_matrix[np.arange(len(query_track_keys_list)), best_match_indices]

                for i, tk_query in enumerate(query_track_keys_list):
                    best_gallery_idx = best_match_indices[i]
                    max_sim = float(max_similarity_scores[i])
                    
                    if max_sim >= settings.REID_SIMILARITY_THRESHOLD:
                        matched_gid = gallery_gids[best_gallery_idx]
                        matched_type = gallery_types[best_gallery_idx]
                        # Apply the handoff-aware filter
                        if self._apply_handoff_filter_for_match(tk_query, matched_gid, active_triggers_map):
                            tentative_assignments[tk_query] = (matched_gid, max_sim, matched_type)
                            logger.debug(f"[Task {self.task_id}] Tentative Match: {tk_query} -> GID {matched_gid} ({matched_type}) (Sim: {max_sim:.3f}). Handoff Filter Passed.")
                        else: # Handoff filter rejected the match
                            tentative_assignments[tk_query] = (None, max_sim, "filtered_handoff")
                            logger.info(f"[Task {self.task_id}] Tentative Match REJECTED by Handoff Filter: {tk_query} to GID {matched_gid} (Sim: {max_sim:.3f}).")
                    else: # Below similarity threshold
                        tentative_assignments[tk_query] = (None, max_sim, "below_threshold")
        
        current_assignments = tentative_assignments.copy()
        for i, tk in enumerate(query_track_keys_list):
            if tk not in current_assignments or current_assignments[tk][0] is None:
                new_gid = self.get_new_global_id()
                current_assignments[tk] = (new_gid, -1.0, "new")
                self.reid_gallery[new_gid] = query_embeddings_np[i] # Add normalized feature
                self.global_id_last_seen_frame[new_gid] = frame_processed_count
                self.global_id_last_seen_cam[new_gid] = tk[0]
                logger.info(f"[Task {self.task_id}] Assigned NEW GID {new_gid} to {tk}.")

        assignments_by_cam_gid: Dict[CameraID, Dict[GlobalID, List[Tuple[TrackKey, float, str]]]] = defaultdict(lambda: defaultdict(list))
        for tk, (gid, score, type_str) in current_assignments.items():
            if gid is not None:
                assignments_by_cam_gid[tk[0]][gid].append((tk, score, type_str))

        reverted_keys_for_second_pass: List[TrackKey] = []
        for cam_id, gid_map in assignments_by_cam_gid.items():
            for gid, track_score_list in gid_map.items():
                if len(track_score_list) > 1: # Conflict!
                    track_score_list.sort(key=lambda x: x[1], reverse=True)
                    winner_tk, winner_score, winner_type = track_score_list[0]
                    logger.warning(
                        f"[Task {self.task_id}][{cam_id}] Conflict for GID {gid}. Tracks: "
                        f"{[(ts[0][1], f'{ts[1]:.2f}') for ts in track_score_list]}. Keeping T:{winner_tk[1]}."
                    )
                    for i in range(1, len(track_score_list)):
                        reverted_tk, _, _ = track_score_list[i]
                        current_assignments[reverted_tk] = (None, -1.0, "reverted_conflict")
                        reverted_keys_for_second_pass.append(reverted_tk)
        
        for tk, (gid, score, type_str) in current_assignments.items():
            if gid is not None and tk not in reverted_keys_for_second_pass :
                self.track_to_global_id[tk] = gid
                self.global_id_last_seen_cam[gid] = tk[0]
                self.global_id_last_seen_frame[gid] = frame_processed_count
                if type_str not in ["new", "reverted_conflict", "filtered_handoff", "below_threshold"]:
                    query_idx = query_track_keys_list.index(tk)
                    self._update_gallery_with_ema(gid, query_embeddings_np[query_idx], tk, type_str)

        if reverted_keys_for_second_pass:
            logger.info(f"[Task {self.task_id}] Starting second pass Re-ID for {len(reverted_keys_for_second_pass)} reverted tracks.")
            # Re-prepare gallery (main only for simplicity in 2nd pass, could be more complex)
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
                            
                            can_assign_gid_2pass = True
                            # Check if this gid_2pass was the winner for ANOTHER track on this camera (which caused the original conflict)
                            # and if that other track is still assigned this gid_2pass.
                            original_conflict_info = assignments_by_cam_gid[tk_reverted[0]].get(gid_2pass)
                            if original_conflict_info: # gid_2pass was involved in a conflict on this camera
                                winner_of_conflict_tk = original_conflict_info[0][0] # The track that won gid_2pass
                                if winner_of_conflict_tk != tk_reverted and self.track_to_global_id.get(winner_of_conflict_tk) == gid_2pass:
                                    can_assign_gid_2pass = False 
                                    logger.debug(f"[Task {self.task_id}][{tk_reverted}] 2nd pass: GID {gid_2pass} was/is winner for another track {winner_of_conflict_tk} in this cam. Cannot assign.")
                            
                            # Apply handoff filter again
                            if can_assign_gid_2pass and self._apply_handoff_filter_for_match(tk_reverted, gid_2pass, active_triggers_map):
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

        await self.update_galleries_lifecycle(active_track_keys_this_frame, frame_processed_count)


    async def update_galleries_lifecycle(self, active_track_keys_this_frame: Set[TrackKey], frame_processed_count: int):
        all_previously_known_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previously_known_track_keys - active_track_keys_this_frame

        for tk_disappeared in disappeared_track_keys:
            gid = self.track_to_global_id.pop(tk_disappeared, None)
            self.track_last_reid_frame.pop(tk_disappeared, None)
            if gid:
                feature = self.reid_gallery.get(gid)
                if feature is not None:
                    if gid not in self.lost_track_gallery:
                        last_active_frame = self.global_id_last_seen_frame.get(gid, frame_processed_count -1)
                        self.lost_track_gallery[gid] = (feature, last_active_frame)
                        logger.debug(f"[Task {self.task_id}] Track {tk_disappeared} (GID {gid}) disappeared. Moved to lost gallery (from frame {last_active_frame}).")
        
        expired_lost_gids = [
            gid for gid, (_, frame_added) in self.lost_track_gallery.items()
            if (frame_processed_count - frame_added) > settings.REID_LOST_TRACK_BUFFER_FRAMES
        ]
        for gid in expired_lost_gids:
            self.lost_track_gallery.pop(gid, None)
            logger.debug(f"[Task {self.task_id}] Purged GID {gid} from lost gallery.")

        if frame_processed_count > 0 and (frame_processed_count % settings.REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES == 0):
            logger.info(f"[Task {self.task_id}] Main gallery pruning (Frame {frame_processed_count}). Size: {len(self.reid_gallery)}.")
            gids_to_prune_main: List[GlobalID] = []
            prune_cutoff_frame = frame_processed_count - settings.REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES
            
            for gid_candidate, last_seen_f in list(self.global_id_last_seen_frame.items()): # Iterate copy
                if last_seen_f < prune_cutoff_frame and gid_candidate not in self.lost_track_gallery:
                    is_active_now = any(active_gid == gid_candidate for active_gid in self.track_to_global_id.values())
                    if not is_active_now:
                        gids_to_prune_main.append(gid_candidate)
            
            if gids_to_prune_main:
                logger.info(f"[Task {self.task_id}] Pruning {len(gids_to_prune_main)} GIDs from main gallery. Examples: {gids_to_prune_main[:3]}")
                for gid_prune in gids_to_prune_main:
                    self.reid_gallery.pop(gid_prune, None)
                    self.global_id_last_seen_cam.pop(gid_prune, None)
                    self.global_id_last_seen_frame.pop(gid_prune, None)
            logger.info(f"[Task {self.task_id}] Main gallery size after pruning: {len(self.reid_gallery)}.")