"""
Track entity for individual camera tracking.

Represents a person track within a single camera view.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from app.shared.types import CameraID
from app.domains.detection.entities.detection import Detection

class TrackStatus(Enum):
    """Track status enumeration."""
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"
    MERGED = "merged"

@dataclass
class Track:
    """Individual camera track entity."""
    
    local_id: int
    camera_id: CameraID
    detections: List[Detection] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: TrackStatus = TrackStatus.ACTIVE
    
    # ReID attributes
    global_id: Optional[str] = None
    feature_vectors: List[List[float]] = field(default_factory=list)
    reid_confidence: float = 0.0
    
    # Quality metrics
    track_confidence: float = 0.0
    stability_score: float = 0.0
    detection_count: int = 0
    lost_frame_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate track parameters."""
        if self.local_id < 0:
            raise ValueError("Local ID must be non-negative")
        
        if not 0 <= self.track_confidence <= 1:
            raise ValueError("Track confidence must be between 0 and 1")
        
        if not 0 <= self.reid_confidence <= 1:
            raise ValueError("ReID confidence must be between 0 and 1")
        
        if not 0 <= self.stability_score <= 1:
            raise ValueError("Stability score must be between 0 and 1")
    
    def add_detection(self, detection: Detection) -> 'Track':
        """Add a detection to this track."""
        new_detections = self.detections.copy()
        new_detections.append(detection)
        
        # Update timing
        start_time = self.start_time or detection.timestamp
        end_time = detection.timestamp
        
        return Track(
            local_id=self.local_id,
            camera_id=self.camera_id,
            detections=new_detections,
            start_time=start_time,
            end_time=end_time,
            status=self.status,
            global_id=self.global_id,
            feature_vectors=self.feature_vectors.copy(),
            reid_confidence=self.reid_confidence,
            track_confidence=self.track_confidence,
            stability_score=self.stability_score,
            detection_count=self.detection_count + 1,
            lost_frame_count=self.lost_frame_count,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def add_feature_vector(self, feature_vector: List[float]) -> 'Track':
        """Add a feature vector to this track."""
        new_feature_vectors = self.feature_vectors.copy()
        new_feature_vectors.append(feature_vector)
        
        # Keep only the last N feature vectors to manage memory
        max_features = 10
        if len(new_feature_vectors) > max_features:
            new_feature_vectors = new_feature_vectors[-max_features:]
        
        return Track(
            local_id=self.local_id,
            camera_id=self.camera_id,
            detections=self.detections.copy(),
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
            global_id=self.global_id,
            feature_vectors=new_feature_vectors,
            reid_confidence=self.reid_confidence,
            track_confidence=self.track_confidence,
            stability_score=self.stability_score,
            detection_count=self.detection_count,
            lost_frame_count=self.lost_frame_count,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def assign_global_id(self, global_id: str, confidence: float) -> 'Track':
        """Assign a global identity to this track."""
        return Track(
            local_id=self.local_id,
            camera_id=self.camera_id,
            detections=self.detections.copy(),
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
            global_id=global_id,
            feature_vectors=self.feature_vectors.copy(),
            reid_confidence=confidence,
            track_confidence=self.track_confidence,
            stability_score=self.stability_score,
            detection_count=self.detection_count,
            lost_frame_count=self.lost_frame_count,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def update_status(self, status: TrackStatus) -> 'Track':
        """Update track status."""
        end_time = self.end_time
        if status in [TrackStatus.TERMINATED, TrackStatus.MERGED]:
            end_time = datetime.now()
        
        return Track(
            local_id=self.local_id,
            camera_id=self.camera_id,
            detections=self.detections.copy(),
            start_time=self.start_time,
            end_time=end_time,
            status=status,
            global_id=self.global_id,
            feature_vectors=self.feature_vectors.copy(),
            reid_confidence=self.reid_confidence,
            track_confidence=self.track_confidence,
            stability_score=self.stability_score,
            detection_count=self.detection_count,
            lost_frame_count=self.lost_frame_count,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def increment_lost_frames(self, count: int = 1) -> 'Track':
        """Increment lost frame count."""
        return Track(
            local_id=self.local_id,
            camera_id=self.camera_id,
            detections=self.detections.copy(),
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
            global_id=self.global_id,
            feature_vectors=self.feature_vectors.copy(),
            reid_confidence=self.reid_confidence,
            track_confidence=self.track_confidence,
            stability_score=self.stability_score,
            detection_count=self.detection_count,
            lost_frame_count=self.lost_frame_count + count,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    def update_confidence(self, track_confidence: float) -> 'Track':
        """Update track confidence."""
        return Track(
            local_id=self.local_id,
            camera_id=self.camera_id,
            detections=self.detections.copy(),
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
            global_id=self.global_id,
            feature_vectors=self.feature_vectors.copy(),
            reid_confidence=self.reid_confidence,
            track_confidence=track_confidence,
            stability_score=self.stability_score,
            detection_count=self.detection_count,
            lost_frame_count=self.lost_frame_count,
            created_at=self.created_at,
            last_updated=datetime.now()
        )
    
    @property
    def is_active(self) -> bool:
        """Check if track is active."""
        return self.status == TrackStatus.ACTIVE
    
    @property
    def is_lost(self) -> bool:
        """Check if track is lost."""
        return self.status == TrackStatus.LOST
    
    @property
    def is_terminated(self) -> bool:
        """Check if track is terminated."""
        return self.status == TrackStatus.TERMINATED
    
    @property
    def has_global_id(self) -> bool:
        """Check if track has been assigned a global ID."""
        return self.global_id is not None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get track duration in seconds."""
        if not self.start_time or not self.end_time:
            return None
        
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def latest_detection(self) -> Optional[Detection]:
        """Get the most recent detection."""
        return self.detections[-1] if self.detections else None
    
    @property
    def first_detection(self) -> Optional[Detection]:
        """Get the first detection."""
        return self.detections[0] if self.detections else None
    
    @property
    def average_confidence(self) -> float:
        """Get average detection confidence."""
        if not self.detections:
            return 0.0
        
        return sum(d.confidence for d in self.detections) / len(self.detections)
    
    @property
    def latest_feature_vector(self) -> Optional[List[float]]:
        """Get the most recent feature vector."""
        return self.feature_vectors[-1] if self.feature_vectors else None
    
    @property
    def feature_vector_count(self) -> int:
        """Get number of feature vectors."""
        return len(self.feature_vectors)
    
    def get_detections_in_frame_range(
        self, 
        start_frame: int, 
        end_frame: int
    ) -> List[Detection]:
        """Get detections within frame range."""
        return [
            d for d in self.detections 
            if start_frame <= d.frame_index <= end_frame
        ]
    
    def get_detection_by_frame(self, frame_index: int) -> Optional[Detection]:
        """Get detection at specific frame."""
        for detection in self.detections:
            if detection.frame_index == frame_index:
                return detection
        return None
    
    def calculate_stability_score(self) -> float:
        """Calculate track stability score based on detection pattern."""
        if not self.detections:
            return 0.0
        
        # Calculate based on detection consistency
        if len(self.detections) < 2:
            return 0.5
        
        # Check for gaps in detections
        frame_indices = [d.frame_index for d in self.detections]
        frame_indices.sort()
        
        total_frames = frame_indices[-1] - frame_indices[0] + 1
        detection_ratio = len(self.detections) / total_frames
        
        # Penalize for lost frames
        lost_penalty = min(self.lost_frame_count / 100, 0.5)
        
        stability = detection_ratio - lost_penalty
        return max(0.0, min(1.0, stability))
    
    def get_average_feature_vector(self) -> Optional[List[float]]:
        """Get average feature vector across all features."""
        if not self.feature_vectors:
            return None
        
        if len(self.feature_vectors) == 1:
            return self.feature_vectors[0]
        
        # Calculate element-wise average
        vector_length = len(self.feature_vectors[0])
        avg_vector = [0.0] * vector_length
        
        for vector in self.feature_vectors:
            for i, value in enumerate(vector):
                avg_vector[i] += value
        
        # Normalize by number of vectors
        avg_vector = [val / len(self.feature_vectors) for val in avg_vector]
        return avg_vector
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "local_id": self.local_id,
            "camera_id": self.camera_id,
            "detection_count": len(self.detections),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "global_id": self.global_id,
            "reid_confidence": self.reid_confidence,
            "track_confidence": self.track_confidence,
            "stability_score": self.stability_score,
            "lost_frame_count": self.lost_frame_count,
            "feature_vector_count": len(self.feature_vectors),
            "duration_seconds": self.duration_seconds,
            "average_confidence": self.average_confidence,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }