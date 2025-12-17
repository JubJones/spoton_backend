import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class HandoffEntry:
    global_id: str
    embedding: np.ndarray
    camera_id: str
    timestamp: datetime
    location: Optional[Tuple[float, float]] = None

class HandoffManager:
    def __init__(self, 
                 match_threshold: float = 0.70,
                 time_window_sec: int = 60,
                 max_queue_size: int = 100):
        self.match_threshold = match_threshold
        self.time_window_sec = time_window_sec
        self.max_queue_size = max_queue_size
        self.queue: List[HandoffEntry] = []

    def register_exit(self, global_id: str, embedding: np.ndarray, camera_id: str, location: Optional[Tuple[float, float]] = None):
        """
        Register a track that has exited or is about to exit.
        """
        # Remove existing entry for same global_id to update it
        self.queue = [e for e in self.queue if e.global_id != global_id]
        
        entry = HandoffEntry(
            global_id=global_id,
            embedding=embedding,
            camera_id=camera_id,
            timestamp=datetime.now(),
            location=location
        )
        self.queue.append(entry)
        
        # Enforce max size (FIFO-ish, or based on time? for now simple size limit)
        if len(self.queue) > self.max_queue_size:
            self.queue.pop(0)

    def find_match(self, embedding: np.ndarray, camera_id: str) -> Tuple[Optional[str], float]:
        """
        Find a matching global_id for the given embedding.
        Returns (global_id, score). If no match found, returns (None, 0.0).
        """
        self._cleanup_expired()
        
        if not self.queue:
            return None, 0.0
            
        best_match_id = None
        best_score = -1.0
        
        query_norm = embedding / np.linalg.norm(embedding)
        
        for entry in self.queue:
            # Skip if same camera (unless we want to handle re-entry to same cam after occlusion, 
            # but usually handoff is cross-camera. For now allow any camera matching)
            
            # Cosine similarity
            entry_norm = entry.embedding / np.linalg.norm(entry.embedding)
            score = np.dot(query_norm, entry_norm)
            
            if score > best_score:
                best_score = score
                best_match_id = entry.global_id
                
        if best_score >= self.match_threshold:
            return best_match_id, float(best_score)
            
        return None, 0.0
        
    def _cleanup_expired(self):
        """Remove entries older than time_window_sec"""
        now = datetime.now()
        threshold = now - timedelta(seconds=self.time_window_sec)
        self.queue = [e for e in self.queue if e.timestamp > threshold]
