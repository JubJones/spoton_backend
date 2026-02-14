
import time
import numpy as np
import logging
import faiss
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class HandoffEntry:
    global_id: str
    camera_id: str
    timestamp: datetime
    location: Optional[Tuple[float, float]] = None

class HandoffManager:
    def __init__(self):
        """
        Manages cross-camera handoffs using FAISS for efficient similarity search.
        Uses Inner Product (IP) index with L2 normalization to simulate Cosine Similarity.
        """
        self.similarity_threshold = getattr(settings, 'REID_SIMILARITY_THRESHOLD', 0.70)
        self.feature_dim = getattr(settings, 'REID_FEATURE_DIM', 512)
        self.time_window_sec = 60 # Keep entries for 60 seconds
        
        # FAISS Index for Cosine Similarity
        # IndexFlatIP = Inner Product. When vectors are L2 normalized, IP == Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.feature_dim)
        
        # Mappings
        # faiss_id (int) -> HandoffEntry
        self.metadata: Dict[int, HandoffEntry] = {} 
        # global_id (str) -> faiss_id (int) [Look up most recent vector for a person]
        self.gid_to_faiss_id: Dict[str, int] = {}
        
        logger.info(f"🤝 HANDOFF MANAGER INIT: FAISS Index (Dim={self.feature_dim}, Thresh={self.similarity_threshold})")

    def register_exit(self, global_id: str, embedding: np.ndarray, camera_id: str, location: Optional[Tuple[float, float]] = None):
        """
        Register a track that has exited or is in a handoff zone.
        Adds the embedding to FAISS and updates metadata.
        """
        if embedding is None or embedding.size == 0:
            return

        # Ensure correct dimension
        if embedding.shape[0] != self.feature_dim:
            logger.warning(f"Feature dim mismatch: got {embedding.shape[0]}, expected {self.feature_dim}")
            return
            
        # 1. L2 Normalize (Critical for Cosine Similarity with IndexFlatIP)
        # Reshape to (1, dim) for FAISS
        vector = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vector)
        
        # 2. Add to FAISS
        new_id = self.index.ntotal
        self.index.add(vector)
        
        # 3. Store Metadata
        entry = HandoffEntry(
            global_id=global_id,
            camera_id=camera_id,
            timestamp=datetime.now(),
            location=location
        )
        self.metadata[new_id] = entry
        self.gid_to_faiss_id[global_id] = new_id
        
        # logger.debug(f"Registered interaction for {global_id} (FAISS ID: {new_id})")
        
        # Periodic cleanup of metadata (though FAISS index grows, it's fast enough 
        # that we might not need to remove from index immediately for this use case. 
        # Re-building index for cleanup is expensive. For now, just clean metadata map to saving memory?)
        # Actually, if we don't remove from FAISS, we might match old vectors. 
        # But 'validity' check in find_match handles timestamps.
        
        if new_id % 100 == 0:
            self._cleanup_metadata()

    def find_match(self, embedding: np.ndarray, camera_id: str) -> Tuple[Optional[str], float]:
        """
        Find the best matching global_id from ANOTHER camera.
        """
        if self.index.ntotal == 0:
            return None, 0.0
            
        # 1. L2 Normalize
        vector = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vector)
        
        # 2. Search FAISS
        # k=5 to verify candidates (check time window, different camera)
        k = 5
        search_k = min(k, self.index.ntotal)
        scores, ids = self.index.search(vector, search_k)
        
        # 3. Filter Results
        best_id = None
        best_score = 0.0
        
        # Iterate through candidates
        for i in range(search_k):
            score = scores[0][i]
            faiss_id = ids[0][i]
            
            if faiss_id == -1: break
            if score < self.similarity_threshold: break # Sorted matches, so we can stop
            
            entry = self.metadata.get(faiss_id)
            if not entry: continue
            
            # Rule: Don't match with yourself from the SAME camera (unless enough time passed? No, usually distinct cams)
            if entry.camera_id == camera_id:
                continue
                
            # Rule: Time window check
            if datetime.now() - entry.timestamp > timedelta(seconds=self.time_window_sec):
                continue
                
            # Valid match found
            best_id = entry.global_id
            best_score = float(score)
            break # Return the first valid high-score match
            
        return best_id, best_score

    def _cleanup_metadata(self):
        """
        Lazy cleanup of metadata dictionary to prevent unbounded growth.
        Does NOT remove from FAISS index (indices are append-only), but marks them effectively
        unreachable by removing metadata.
        """
        now = datetime.now()
        threshold = now - timedelta(seconds=self.time_window_sec * 2) # Keep a bit longer safely
        
        expired_ids = [fid for fid, entry in self.metadata.items() if entry.timestamp < threshold]
        
        for fid in expired_ids:
            del self.metadata[fid]
            
        # Note: self.index still holds vectors. In a long-running production system, 
        # one would use IndexIVF or rebuild IndexFlatIP periodically. 
        # For this scale, IndexFlatIP growing to ~100k vectors is negligible memory/time.
