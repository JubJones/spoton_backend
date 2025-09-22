"""
Components for managing Re-Identification (ReID) state and association logic
across multiple cameras.
"""
import logging
import uuid
import math
import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
from scipy.spatial.distance import cdist

from app.shared.types import (
    CameraID, TrackID, GlobalID, FeatureVector, TrackKey, HandoffTriggerInfo
)
from app.core.config import settings

# --- FAISS Conditional Import ---
try:
    import faiss
    FAISS_AVAILABLE = True
    logger_faiss = logging.getLogger(__name__ + ".faiss_support") # Separate logger for FAISS messages
    logger_faiss.info("FAISS library successfully imported.")
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "FAISS library not found. FAISS-based Re-ID methods will be unavailable. "
        "Install with `pip install faiss-cpu` or `faiss-gpu` to enable them."
    )
# --- End FAISS Conditional Import ---

logger = logging.getLogger(__name__)

class ReIDStateManager:
    """
    Manages the state required for cross-camera Re-Identification.
    Supports multiple feature comparison methods, including FAISS.
    """

    def __init__(self, task_id: uuid.UUID):
        self.task_id = task_id
        self.similarity_method = settings.REID_SIMILARITY_METHOD.lower()

        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        self.global_id_last_seen_frame: Dict[GlobalID, int] = {}
        self.track_last_reid_frame: Dict[TrackKey, int] = {}

        self.faiss_index: Optional[Any] = None
        self.faiss_gallery_gids: List[GlobalID] = []
        self.faiss_index_dirty: bool = True

        self.effective_threshold: float
        self.is_distance_metric: bool = False

        # Determine effective threshold based on method
        if self.similarity_method in ["l2_derived", "faiss_l2"]:
            self.is_distance_metric = True
            if self.similarity_method == "faiss_l2" and settings.REID_L2_DISTANCE_THRESHOLD is not None:
                 self.effective_threshold = settings.REID_L2_DISTANCE_THRESHOLD
                 logger.info(f"[Task {task_id}] Using EXPLICIT L2 distance threshold for FAISS: {self.effective_threshold}")
            else: # l2_derived or faiss_l2 without explicit L2 threshold
                 self.effective_threshold = settings.derived_l2_distance_threshold
                 logger.info(f"[Task {task_id}] Using DERIVED L2 distance threshold: {self.effective_threshold:.4f} (from cosine_sim_thresh {settings.REID_SIMILARITY_THRESHOLD})")
        elif self.similarity_method in ["cosine", "inner_product", "faiss_ip"]:
             self.is_distance_metric = False # Higher score is better
             self.effective_threshold = settings.REID_SIMILARITY_THRESHOLD
        else: # Default or unknown
            logger.warning(
                f"[Task {task_id}] Unknown REID_SIMILARITY_METHOD '{self.similarity_method}'. "
                f"Defaulting to 'cosine' logic with threshold {settings.REID_SIMILARITY_THRESHOLD}."
            )
            self.similarity_method = "cosine" # Fallback
            self.effective_threshold = settings.REID_SIMILARITY_THRESHOLD
            self.is_distance_metric = False
        
        if "faiss" in self.similarity_method and not FAISS_AVAILABLE:
            logger.error(
                f"[Task {task_id}] FAISS method '{self.similarity_method}' requested, but FAISS library is not available. "
                "Re-ID will likely fail or use fallback. Please install FAISS."
            )
            # Optionally, force a fallback if FAISS is critical for the chosen method
            if self.similarity_method == "faiss_l2": self.similarity_method = "l2_derived"
            elif self.similarity_method == "faiss_ip": self.similarity_method = "inner_product" # or cosine
            logger.warning(f"[Task {task_id}] Falling back to non-FAISS method: '{self.similarity_method}'")


        logger.info(f"[Task {task_id}] ReIDStateManager initialized. Method: '{self.similarity_method}', Effective Threshold: {self.effective_threshold:.4f}")

    def _build_faiss_index(self):
        if not FAISS_AVAILABLE or faiss is None:
            self.faiss_index = None
            logger.warning(f"[Task {self.task_id}] Attempted to build FAISS index, but FAISS is not available.")
            return

        self.faiss_gallery_gids = []
        gallery_embeddings_list: List[FeatureVector] = [
            feat for feat in self.reid_gallery.values() if feat.size > 0
        ]
        self.faiss_gallery_gids = [
            gid for gid, feat in self.reid_gallery.items() if feat.size > 0
        ]

        if not gallery_embeddings_list:
            self.faiss_index = None
            self.faiss_index_dirty = False
            logger.debug(f"[Task {self.task_id}] Main gallery empty. FAISS index reset.")
            return

        gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32)
        if gallery_embeddings_np.ndim == 1 and gallery_embeddings_np.size > 0: # Single vector case
            gallery_embeddings_np = gallery_embeddings_np.reshape(1, -1)
        elif gallery_embeddings_np.ndim != 2 and gallery_embeddings_np.size > 0:
            logger.error(f"[Task {self.task_id}] FAISS: Gallery embeddings have unexpected shape {gallery_embeddings_np.shape}. Cannot build index.")
            self.faiss_index = None
            return


        dimension = gallery_embeddings_np.shape[1]
        if dimension == 0 and gallery_embeddings_np.shape[0] > 0 : # If features are empty arrays themselves
            logger.warning(f"[Task {self.task_id}] FAISS: Gallery embeddings have 0 dimension. Cannot build index.")
            self.faiss_index = None
            return

        index_type_str = "Unknown FAISS Index"
        if self.similarity_method == "faiss_ip":
            self.faiss_index = faiss.IndexFlatIP(dimension)
            index_type_str = "IndexFlatIP"
        elif self.similarity_method == "faiss_l2":
            self.faiss_index = faiss.IndexFlatL2(dimension)
            index_type_str = "IndexFlatL2"
        else:
            self.faiss_index = None
            logger.error(f"[Task {self.task_id}] FAISS: Invalid similarity_method '{self.similarity_method}' for FAISS index building.")
            return
        
        if gallery_embeddings_np.shape[0] > 0: # Only add if there are embeddings
            try:
                self.faiss_index.add(gallery_embeddings_np)
            except Exception as faiss_add_err:
                logger.error(f"[Task {self.task_id}] FAISS: Error adding embeddings to index: {faiss_add_err}", exc_info=True)
                self.faiss_index = None # Invalidate index on error
                return

        self.faiss_index_dirty = False
        logger.info(f"[Task {self.task_id}] FAISS index ({index_type_str}) built/rebuilt with {len(self.faiss_gallery_gids)} GIDs from main gallery.")


    def get_new_global_id(self) -> GlobalID:
        return GlobalID(str(uuid.uuid4()))

    def _normalize_embedding(self, embedding: Optional[FeatureVector]) -> Optional[FeatureVector]:
        if embedding is None: return None
        if not isinstance(embedding, np.ndarray):
            try: embedding_np = np.array(embedding, dtype=np.float32)
            except Exception: logger.warning(f"[Task {self.task_id}] Invalid type for embedding: {type(embedding)}. Cannot normalize."); return None
        else: embedding_np = embedding.astype(np.float32)

        if embedding_np.size == 0: return FeatureVector(np.array([], dtype=np.float32))
        norm = np.linalg.norm(embedding_np)
        return FeatureVector(embedding_np / norm) if norm > 1e-6 else FeatureVector(embedding_np)

    def _calculate_scores_from_cdist(
        self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray
    ) -> Optional[np.ndarray]:
        if query_embeddings.ndim == 1: query_embeddings = query_embeddings.reshape(1, -1)
        if gallery_embeddings.ndim == 1: gallery_embeddings = gallery_embeddings.reshape(1, -1)
        if query_embeddings.size == 0 or gallery_embeddings.size == 0: return None
        try:
            if self.similarity_method == "cosine":
                scores = 1.0 - cdist(query_embeddings, gallery_embeddings, metric='cosine')
            elif self.similarity_method == "l2_derived": # L2 derived from cosine threshold
                scores = cdist(query_embeddings, gallery_embeddings, metric='euclidean')
            elif self.similarity_method == "inner_product":
                scores = query_embeddings @ gallery_embeddings.T
            else: return None
            return scores
        except Exception as e:
            logger.error(f"[Task {self.task_id}] cdist-based score calculation failed: {e}", exc_info=True)
            return None

    def _should_attempt_reid_for_track(
        self, track_key: TrackKey, frame_processed_count: int, active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> bool:
        is_known = track_key in self.track_to_global_id
        last_attempt_frame = self.track_last_reid_frame.get(track_key, -settings.REID_REFRESH_INTERVAL_FRAMES - 1)
        is_due_for_refresh = (frame_processed_count - last_attempt_frame) >= settings.REID_REFRESH_INTERVAL_FRAMES
        is_triggering_handoff = track_key in active_triggers_map
        if not is_known: return True
        if is_due_for_refresh: return True
        if is_triggering_handoff: logger.info(f"[Task {self.task_id}][{track_key}] Re-ID attempt due to active handoff trigger."); return True
        return False

    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        relevant_cams = {target_cam_id}
        for c1, c2 in settings.normalized_possible_camera_overlaps:
            if c1 == target_cam_id: relevant_cams.add(c2)
            elif c2 == target_cam_id: relevant_cams.add(c1)
        return relevant_cams
        
    def _apply_handoff_filter_for_match(
        self, query_track_key: TrackKey, matched_gid: GlobalID, active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> bool:
        trigger_info = active_triggers_map.get(query_track_key)
        current_cam_id, _ = query_track_key
        if not trigger_info:
            relevant_cams_for_current = self._get_relevant_handoff_cams(current_cam_id)
            last_seen_cam_for_gid = self.global_id_last_seen_cam.get(matched_gid)
            if last_seen_cam_for_gid is not None and last_seen_cam_for_gid != current_cam_id:
                if last_seen_cam_for_gid not in relevant_cams_for_current: return False
            return True 
        trigger_target_cam_id = trigger_info.rule.target_cam_id
        relevant_cams_for_trigger_target = self._get_relevant_handoff_cams(trigger_target_cam_id)
        last_seen_cam_for_matched_gid = self.global_id_last_seen_cam.get(matched_gid)
        if last_seen_cam_for_matched_gid is not None:
            if last_seen_cam_for_matched_gid not in relevant_cams_for_trigger_target: return False
        return True

    def _update_gallery_with_ema(
        self, gid: GlobalID, new_embedding: FeatureVector, tk_for_log: TrackKey, matched_gallery_type: str
    ): # Added tk_for_log parameter
        alpha = settings.REID_GALLERY_EMA_ALPHA
        current_gallery_embedding: Optional[FeatureVector] = None
        new_embedding_norm = self._normalize_embedding(new_embedding)
        if new_embedding_norm is None or new_embedding_norm.size == 0:
            logger.debug(f"[Task {self.task_id}][{tk_for_log}] EMA update skipped for GID {gid}: new embedding is empty/invalid.")
            return
        
        self.faiss_index_dirty = True 
        if matched_gallery_type == 'lost':
            if gid in self.lost_track_gallery:
                lost_embedding, _ = self.lost_track_gallery.pop(gid) # Remove from lost as it's becoming active again
                current_gallery_embedding = self._normalize_embedding(lost_embedding)
                logger.debug(f"[Task {self.task_id}][{tk_for_log}] GID {gid} matched from lost gallery. Feature will be updated in main gallery.")
            elif gid in self.reid_gallery: # Should ideally not happen if lost was correctly managed, but as fallback
                current_gallery_embedding = self.reid_gallery.get(gid)
                logger.debug(f"[Task {self.task_id}][{tk_for_log}] GID {gid} (matched as 'lost') found in main gallery. Updating main.")
            else: # GID marked as 'lost' match but not found in either gallery - this is unusual
                logger.warning(f"[Task {self.task_id}][{tk_for_log}] GID {gid} matched as 'lost' but not found in lost_track_gallery or reid_gallery. Initializing with new feature.")

        elif matched_gallery_type in ['main', 'main_2pass']: # 'main_2pass' was a typo or old logic, just 'main' is typical here
            current_gallery_embedding = self.reid_gallery.get(gid)
            if current_gallery_embedding is None:
                logger.warning(f"[Task {self.task_id}][{tk_for_log}] GID {gid} matched as 'main' but not found in reid_gallery. Initializing with new feature.")
        
        if current_gallery_embedding is not None and current_gallery_embedding.size > 0:
            updated_embedding = (alpha * current_gallery_embedding + (1.0 - alpha) * new_embedding_norm)
            self.reid_gallery[gid] = self._normalize_embedding(updated_embedding)
            # logger.debug(f"[Task {self.task_id}][{tk_for_log}] GID {gid} feature updated via EMA (from {matched_gallery_type}).")
        else: # If no existing embedding or it was empty, use the new one directly
            self.reid_gallery[gid] = new_embedding_norm
            # logger.debug(f"[Task {self.task_id}][{tk_for_log}] GID {gid} feature initialized with new embedding (from {matched_gallery_type}).")


    async def associate_features_and_update_state(
        self,
        all_tracks_with_features: Dict[TrackKey, FeatureVector],
        active_track_keys_this_frame: Set[TrackKey],
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo],
        current_frame_idx: int
    ):
        query_features: Dict[TrackKey, FeatureVector] = {}
        for tk, feat in all_tracks_with_features.items():
            if self._should_attempt_reid_for_track(tk, current_frame_idx, active_triggers_map):
                norm_feat = self._normalize_embedding(feat)
                if norm_feat is not None and norm_feat.size > 0:
                    query_features[tk] = norm_feat
                    self.track_last_reid_frame[tk] = current_frame_idx
        
        if not query_features:
            await self.update_galleries_lifecycle(active_track_keys_this_frame, current_frame_idx)
            return

        query_track_keys_list = list(query_features.keys())
        query_embeddings_np = np.array([query_features[tk] for tk in query_track_keys_list], dtype=np.float32)
        if query_embeddings_np.ndim == 1 and query_embeddings_np.size > 0 : 
            query_embeddings_np = query_embeddings_np.reshape(1,-1)


        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float, str]] = {}
        
        if query_embeddings_np.size == 0: 
            await self.update_galleries_lifecycle(active_track_keys_this_frame, current_frame_idx)
            return

        if "faiss" in self.similarity_method:
            if not FAISS_AVAILABLE: 
                logger.error(f"[Task {self.task_id}] FAISS method '{self.similarity_method}' selected but FAISS not available during association. Skipping Re-ID for this frame.")
                for tk_query in query_track_keys_list:
                    new_gid = self.get_new_global_id()
                    tentative_assignments[tk_query] = (new_gid, -1.0 if not self.is_distance_metric else float('inf'), "new_no_faiss")
            else:
                if self.faiss_index_dirty or self.faiss_index is None:
                    self._build_faiss_index()

                if self.faiss_index and self.faiss_index.ntotal > 0:
                    k_neighbors = 1
                    raw_scores_faiss, indices_faiss = await asyncio.to_thread(
                        self.faiss_index.search, query_embeddings_np, k_neighbors
                    )
                    for i, tk_query in enumerate(query_track_keys_list):
                        if indices_faiss[i, 0] < 0: continue 
                        
                        matched_faiss_gallery_idx = indices_faiss[i, 0]
                        score_or_dist = float(raw_scores_faiss[i, 0])
                        matched_gid = self.faiss_gallery_gids[matched_faiss_gallery_idx]
                        matched_type = 'main' 

                        match_passes_threshold = False
                        if self.similarity_method == "faiss_ip":
                            match_passes_threshold = score_or_dist >= self.effective_threshold
                        elif self.similarity_method == "faiss_l2":
                            # FAISS L2 returns squared L2, so compare against squared threshold
                            match_passes_threshold = score_or_dist <= (self.effective_threshold ** 2) 
                        
                        if match_passes_threshold:
                            if self._apply_handoff_filter_for_match(tk_query, matched_gid, active_triggers_map):
                                tentative_assignments[tk_query] = (matched_gid, score_or_dist, matched_type)
                            else: tentative_assignments[tk_query] = (None, score_or_dist, "filtered_handoff")
                        else: tentative_assignments[tk_query] = (None, score_or_dist, "below_threshold")
                else: 
                    logger.debug(f"[Task {self.task_id}] FAISS index empty. All queries will get new GIDs.")
                    for tk_query in query_track_keys_list: 
                        tentative_assignments[tk_query] = (None, -1.0 if not self.is_distance_metric else float('inf'), "new_empty_gallery")


        else: # cdist based methods
            gallery_gids_cdist: List[GlobalID] = []
            gallery_embeddings_list_cdist: List[FeatureVector] = []
            gallery_types_cdist: List[str] = []

            for gid, (feat, _) in self.lost_track_gallery.items():
                norm_feat = self._normalize_embedding(feat)
                if norm_feat is not None and norm_feat.size > 0:
                    gallery_gids_cdist.append(gid); gallery_embeddings_list_cdist.append(norm_feat); gallery_types_cdist.append('lost')
            
            main_gallery_gids_in_lost = set(gid for gid, _, type_str in zip(gallery_gids_cdist, gallery_embeddings_list_cdist, gallery_types_cdist) if type_str == 'lost')
            for gid, feat in self.reid_gallery.items():
                if gid not in main_gallery_gids_in_lost and feat.size > 0:
                    gallery_gids_cdist.append(gid); gallery_embeddings_list_cdist.append(feat); gallery_types_cdist.append('main')
            
            gallery_embeddings_np_cdist = np.array(gallery_embeddings_list_cdist, dtype=np.float32) if gallery_embeddings_list_cdist else np.empty((0,0), dtype=np.float32)

            if gallery_embeddings_np_cdist.size > 0 :
                scores_matrix = self._calculate_scores_from_cdist(query_embeddings_np, gallery_embeddings_np_cdist)
                if scores_matrix is not None:
                    if self.is_distance_metric:
                        best_match_indices = np.argmin(scores_matrix, axis=1)
                        best_scores = scores_matrix[np.arange(len(query_track_keys_list)), best_match_indices]
                        threshold_condition = best_scores <= self.effective_threshold
                    else:
                        best_match_indices = np.argmax(scores_matrix, axis=1)
                        best_scores = scores_matrix[np.arange(len(query_track_keys_list)), best_match_indices]
                        threshold_condition = best_scores >= self.effective_threshold

                    for i, tk_query in enumerate(query_track_keys_list):
                        if threshold_condition[i]:
                            matched_gallery_idx = best_match_indices[i]
                            matched_gid = gallery_gids_cdist[matched_gallery_idx]
                            matched_type = gallery_types_cdist[matched_gallery_idx]
                            score_val = float(best_scores[i])
                            if self._apply_handoff_filter_for_match(tk_query, matched_gid, active_triggers_map):
                                tentative_assignments[tk_query] = (matched_gid, score_val, matched_type)
                            else: tentative_assignments[tk_query] = (None, score_val, "filtered_handoff")
                        else: tentative_assignments[tk_query] = (None, float(best_scores[i]), "below_threshold")
            else: 
                 logger.debug(f"[Task {self.task_id}] Combined gallery for cdist empty. All queries will get new GIDs.")
                 for tk_query in query_track_keys_list:
                    tentative_assignments[tk_query] = (None, -1.0 if not self.is_distance_metric else float('inf'), "new_empty_gallery")


        current_assignments = tentative_assignments.copy()
        for i, tk in enumerate(query_track_keys_list):
            if tk not in current_assignments or current_assignments[tk][0] is None:
                new_gid = self.get_new_global_id()
                new_score = -1.0 if not self.is_distance_metric else float('inf')
                current_assignments[tk] = (new_gid, new_score, "new")
                original_embedding_for_new = query_features[tk]
                if original_embedding_for_new.size > 0:
                    self.reid_gallery[new_gid] = original_embedding_for_new
                    self.faiss_index_dirty = True
                self.global_id_last_seen_frame[new_gid] = current_frame_idx
                self.global_id_last_seen_cam[new_gid] = tk[0]

        assignments_by_cam_gid: Dict[Tuple[CameraID, GlobalID], List[Tuple[TrackKey, float, str]]] = defaultdict(list)
        for tk, (gid, score, type_str) in current_assignments.items():
            if gid is not None: assignments_by_cam_gid[(tk[0], gid)].append((tk, score, type_str))

        reverted_keys_for_second_pass: List[TrackKey] = []
        for (cam_id_conflict, gid_conflict), track_score_list_with_type in assignments_by_cam_gid.items():
            if len(track_score_list_with_type) > 1:
                sort_reverse = not self.is_distance_metric
                track_score_list_with_type.sort(key=lambda x: x[1], reverse=sort_reverse)
                for i_conflict in range(1, len(track_score_list_with_type)):
                    reverted_tk, _, _ = track_score_list_with_type[i_conflict]
                    reverted_score = float('inf') if self.is_distance_metric else -1.0
                    current_assignments[reverted_tk] = (None, reverted_score, "reverted_conflict")
                    reverted_keys_for_second_pass.append(reverted_tk)
        
        for tk, (gid, _, type_str) in current_assignments.items():
            if gid is not None and tk not in reverted_keys_for_second_pass :
                self.track_to_global_id[tk] = gid
                self.global_id_last_seen_cam[gid] = tk[0]
                self.global_id_last_seen_frame[gid] = current_frame_idx
                if type_str not in ["new", "reverted_conflict", "filtered_handoff", "below_threshold", "new_empty_gallery", "new_no_faiss"]:
                    original_embedding_for_update = query_features[tk]
                    # Call _update_gallery_with_ema with the correct signature
                    self._update_gallery_with_ema(gid, original_embedding_for_update, tk, type_str)
        
        if reverted_keys_for_second_pass: 
            for tk_reverted in reverted_keys_for_second_pass:
                new_gid_for_reverted = self.get_new_global_id()
                self.track_to_global_id[tk_reverted] = new_gid_for_reverted
                reverted_embedding = query_features[tk_reverted]
                if reverted_embedding.size > 0:
                    self.reid_gallery[new_gid_for_reverted] = reverted_embedding
                    self.faiss_index_dirty = True
                self.global_id_last_seen_cam[new_gid_for_reverted] = tk_reverted[0]
                self.global_id_last_seen_frame[new_gid_for_reverted] = current_frame_idx

        await self.update_galleries_lifecycle(active_track_keys_this_frame, current_frame_idx)


    async def update_galleries_lifecycle(self, active_track_keys_this_frame: Set[TrackKey], current_frame_idx: int):
        all_previously_known_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previously_known_track_keys - active_track_keys_this_frame

        gallery_changed = False
        for tk_disappeared in disappeared_track_keys:
            gid = self.track_to_global_id.pop(tk_disappeared, None)
            self.track_last_reid_frame.pop(tk_disappeared, None)
            if gid:
                feature = self.reid_gallery.get(gid)
                if feature is not None and feature.size > 0:
                    if gid not in self.lost_track_gallery:
                        last_active_frame_for_gid = self.global_id_last_seen_frame.get(gid, current_frame_idx -1)
                        self.lost_track_gallery[gid] = (feature, last_active_frame_for_gid)
        
        expired_lost_gids = [
            gid for gid, (_, frame_added_to_lost) in self.lost_track_gallery.items()
            if (current_frame_idx - frame_added_to_lost) > settings.REID_LOST_TRACK_BUFFER_FRAMES
        ]
        for gid in expired_lost_gids:
            self.lost_track_gallery.pop(gid, None)

        if current_frame_idx > 0 and (current_frame_idx % settings.REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES == 0):
            gids_to_prune_main: List[GlobalID] = []
            prune_cutoff_frame = current_frame_idx - settings.REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES
            active_gids_in_current_frame = set(self.track_to_global_id.values())
            for gid_candidate, last_seen_f in list(self.global_id_last_seen_frame.items()):
                if last_seen_f < prune_cutoff_frame and \
                   gid_candidate not in self.lost_track_gallery and \
                   gid_candidate not in active_gids_in_current_frame:
                    gids_to_prune_main.append(gid_candidate)
            
            if gids_to_prune_main:
                for gid_prune in gids_to_prune_main:
                    if self.reid_gallery.pop(gid_prune, None) is not None:
                        gallery_changed = True
                    self.global_id_last_seen_cam.pop(gid_prune, None)
                    self.global_id_last_seen_frame.pop(gid_prune, None)
                if gallery_changed:
                    self.faiss_index_dirty = True