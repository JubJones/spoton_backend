"""
ReID service for cross-camera person re-identification.

Provides business logic for:
- Feature extraction from person detections
- Cross-camera similarity matching
- Identity assignment and management
- Track association and fusion
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import uuid

from app.domains.reid.entities.person_identity import PersonIdentity, IdentityStatus
from app.domains.reid.entities.track import Track, TrackStatus
from app.domains.reid.entities.feature_vector import FeatureVector, FeatureVectorBatch
from app.domains.detection.entities.detection import Detection, DetectionBatch
from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class ReIDService:
    """Service for person re-identification operations."""
    
    def __init__(self, feature_extractor=None, similarity_threshold: float = 0.65):
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold
        self.active_identities: Dict[str, PersonIdentity] = {}
        self.active_tracks: Dict[Tuple[CameraID, int], Track] = {}
        self.reid_stats = {
            "total_identities": 0,
            "successful_matches": 0,
            "new_identities": 0,
            "merged_identities": 0,
            "lost_identities": 0
        }
        logger.info("ReIDService initialized")
    
    async def process_detection_batch(
        self,
        detection_batch: DetectionBatch
    ) -> Dict[str, Any]:
        """
        Process a batch of detections for re-identification.
        
        Args:
            detection_batch: Batch of detections to process
            
        Returns:
            Dictionary with processing results and updated identities
        """
        try:
            # Extract features from all person detections
            person_detections = detection_batch.person_detections
            
            if not person_detections:
                return {
                    "identities": {},
                    "new_tracks": [],
                    "updated_tracks": [],
                    "processing_time": detection_batch.processing_time
                }
            
            # Extract features for all detections
            features = await self._extract_features_batch(person_detections)
            
            # Process each detection
            processing_results = {
                "identities": {},
                "new_tracks": [],
                "updated_tracks": [],
                "matched_identities": [],
                "new_identities": []
            }
            
            for detection, feature_vector in zip(person_detections, features):
                result = await self._process_single_detection(detection, feature_vector)
                
                # Accumulate results
                if result["identity"]:
                    processing_results["identities"][result["identity"].global_id] = result["identity"]
                
                if result["track"]:
                    if result["is_new_track"]:
                        processing_results["new_tracks"].append(result["track"])
                    else:
                        processing_results["updated_tracks"].append(result["track"])
                
                if result["is_new_identity"]:
                    processing_results["new_identities"].append(result["identity"])
                elif result["identity"]:
                    processing_results["matched_identities"].append(result["identity"])
            
            # Update statistics
            self._update_stats(processing_results)
            
            logger.info(
                f"ReID batch processing complete: "
                f"{len(processing_results['new_identities'])} new identities, "
                f"{len(processing_results['matched_identities'])} matched identities"
            )
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Error processing detection batch: {e}")
            raise
    
    async def _process_single_detection(
        self,
        detection: Detection,
        feature_vector: FeatureVector
    ) -> Dict[str, Any]:
        """Process a single detection for re-identification."""
        try:
            # Get or create track for this detection
            track_key = (detection.camera_id, detection.track_id or 0)
            
            if track_key in self.active_tracks:
                track = self.active_tracks[track_key]
                track = track.add_detection(detection)
                track = track.add_feature_vector(feature_vector.vector)
                is_new_track = False
            else:
                track = Track(
                    local_id=detection.track_id or 0,
                    camera_id=detection.camera_id,
                    detections=[detection],
                    start_time=detection.timestamp,
                    end_time=detection.timestamp,
                    feature_vectors=[feature_vector.vector]
                )
                is_new_track = True
            
            # Try to match with existing identity
            identity, match_confidence = await self._match_to_identity(feature_vector, detection.camera_id)
            
            if identity:
                # Update existing identity
                identity = identity.add_camera_track(detection.camera_id, track.local_id)
                identity = identity.update_last_seen(detection.timestamp)
                identity = identity.increment_detections()
                
                # Update track with global ID
                track = track.assign_global_id(identity.global_id, match_confidence)
                
                self.active_identities[identity.global_id] = identity
                is_new_identity = False
                
            else:
                # Create new identity
                global_id = str(uuid.uuid4())
                identity = PersonIdentity(
                    global_id=global_id,
                    track_ids_by_camera={detection.camera_id: track.local_id},
                    cameras_seen={detection.camera_id},
                    first_seen=detection.timestamp,
                    last_seen=detection.timestamp,
                    total_detections=1,
                    total_cameras=1,
                    identity_confidence=1.0
                )
                
                # Update track with new global ID
                track = track.assign_global_id(global_id, 1.0)
                
                self.active_identities[global_id] = identity
                is_new_identity = True
            
            # Store updated track
            self.active_tracks[track_key] = track
            
            return {
                "identity": identity,
                "track": track,
                "is_new_track": is_new_track,
                "is_new_identity": is_new_identity,
                "match_confidence": match_confidence if identity else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing detection {detection.id}: {e}")
            raise
    
    async def _extract_features_batch(
        self,
        detections: List[Detection]
    ) -> List[FeatureVector]:
        """Extract features from a batch of detections."""
        if not self.feature_extractor:
            # Return dummy features for testing
            return [
                FeatureVector(
                    vector=[0.0] * 512,  # Dummy 512-dimensional vector
                    extraction_timestamp=datetime.now(timezone.utc),
                    model_version="dummy",
                    detection_id=detection.id,
                    camera_id=detection.camera_id,
                    frame_index=detection.frame_index
                )
                for detection in detections
            ]
        
        try:
            # Extract features using the feature extractor
            features = await self.feature_extractor.extract_batch(detections)
            
            # Convert to FeatureVector objects
            feature_vectors = []
            for i, (detection, feature_array) in enumerate(zip(detections, features)):
                feature_vector = FeatureVector(
                    vector=feature_array.tolist(),
                    extraction_timestamp=datetime.now(timezone.utc),
                    model_version=self.feature_extractor.model_version,
                    detection_id=detection.id,
                    camera_id=detection.camera_id,
                    frame_index=detection.frame_index
                )
                feature_vectors.append(feature_vector)
            
            return feature_vectors
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    async def _match_to_identity(
        self,
        feature_vector: FeatureVector,
        camera_id: CameraID
    ) -> Tuple[Optional[PersonIdentity], float]:
        """Match feature vector to existing identity."""
        if not self.active_identities:
            return None, 0.0
        
        best_match = None
        best_confidence = 0.0
        
        try:
            # Compare with all active identities
            for identity in self.active_identities.values():
                # Skip if same camera (can't match within same camera)
                if camera_id in identity.cameras_seen:
                    continue
                
                # Get feature vectors from tracks of this identity
                identity_features = await self._get_identity_features(identity)
                
                if not identity_features:
                    continue
                
                # Calculate similarity with all features
                similarities = [
                    feature_vector.cosine_similarity(id_feature)
                    for id_feature in identity_features
                ]
                
                # Use maximum similarity
                max_similarity = max(similarities)
                
                if max_similarity > best_confidence and max_similarity > self.similarity_threshold:
                    best_match = identity
                    best_confidence = max_similarity
            
            return best_match, best_confidence
            
        except Exception as e:
            logger.error(f"Error matching to identity: {e}")
            return None, 0.0
    
    async def _get_identity_features(
        self,
        identity: PersonIdentity
    ) -> List[FeatureVector]:
        """Get feature vectors for an identity from its tracks."""
        features = []
        
        try:
            for camera_id, track_id in identity.track_ids_by_camera.items():
                track_key = (camera_id, track_id)
                
                if track_key in self.active_tracks:
                    track = self.active_tracks[track_key]
                    
                    # Get recent feature vectors from track
                    for feature_vector in track.feature_vectors[-3:]:  # Last 3 features
                        features.append(FeatureVector(
                            vector=feature_vector,
                            extraction_timestamp=track.last_updated,
                            model_version="current",
                            camera_id=camera_id
                        ))
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting identity features: {e}")
            return []
    
    def _update_stats(self, processing_results: Dict[str, Any]):
        """Update ReID statistics."""
        self.reid_stats["new_identities"] += len(processing_results["new_identities"])
        self.reid_stats["successful_matches"] += len(processing_results["matched_identities"])
        self.reid_stats["total_identities"] = len(self.active_identities)
    
    def get_reid_stats(self) -> Dict[str, Any]:
        """Get current ReID statistics."""
        return {
            **self.reid_stats,
            "active_identities": len(self.active_identities),
            "active_tracks": len(self.active_tracks),
            "similarity_threshold": self.similarity_threshold
        }
    
    def get_identity_by_id(self, global_id: str) -> Optional[PersonIdentity]:
        """Get identity by global ID."""
        return self.active_identities.get(global_id)
    
    def get_all_identities(self) -> List[PersonIdentity]:
        """Get all active identities."""
        return list(self.active_identities.values())
    
    def get_track_by_key(self, camera_id: CameraID, track_id: int) -> Optional[Track]:
        """Get track by camera and track ID."""
        return self.active_tracks.get((camera_id, track_id))
    
    def get_tracks_for_identity(self, global_id: str) -> List[Track]:
        """Get all tracks for an identity."""
        identity = self.active_identities.get(global_id)
        if not identity:
            return []
        
        tracks = []
        for camera_id, track_id in identity.track_ids_by_camera.items():
            track = self.active_tracks.get((camera_id, track_id))
            if track:
                tracks.append(track)
        
        return tracks
    
    async def merge_identities(
        self,
        primary_id: str,
        secondary_id: str
    ) -> Optional[PersonIdentity]:
        """Merge two identities."""
        try:
            primary_identity = self.active_identities.get(primary_id)
            secondary_identity = self.active_identities.get(secondary_id)
            
            if not primary_identity or not secondary_identity:
                logger.warning(f"Cannot merge identities: {primary_id} or {secondary_id} not found")
                return None
            
            # Merge identities
            merged_identity = primary_identity.merge_with(secondary_identity)
            
            # Update tracks to point to merged identity
            for camera_id, track_id in secondary_identity.track_ids_by_camera.items():
                track_key = (camera_id, track_id)
                if track_key in self.active_tracks:
                    track = self.active_tracks[track_key]
                    track = track.assign_global_id(merged_identity.global_id, 1.0)
                    self.active_tracks[track_key] = track
            
            # Update identities
            self.active_identities[primary_id] = merged_identity
            
            # Mark secondary identity as merged
            secondary_identity = secondary_identity.update_status(IdentityStatus.MERGED)
            del self.active_identities[secondary_id]
            
            self.reid_stats["merged_identities"] += 1
            
            logger.info(f"Merged identity {secondary_id} into {primary_id}")
            return merged_identity
            
        except Exception as e:
            logger.error(f"Error merging identities {primary_id} and {secondary_id}: {e}")
            return None
    
    async def cleanup_lost_identities(self, max_age_seconds: int = 3600):
        """Clean up identities that haven't been seen for a long time."""
        current_time = datetime.now(timezone.utc)
        lost_identities = []
        
        for global_id, identity in list(self.active_identities.items()):
            if identity.last_seen:
                age_seconds = (current_time - identity.last_seen).total_seconds()
                if age_seconds > max_age_seconds:
                    lost_identities.append(global_id)
        
        # Mark identities as lost
        for global_id in lost_identities:
            identity = self.active_identities[global_id]
            identity = identity.update_status(IdentityStatus.LOST)
            del self.active_identities[global_id]
            
            # Update associated tracks
            for camera_id, track_id in identity.track_ids_by_camera.items():
                track_key = (camera_id, track_id)
                if track_key in self.active_tracks:
                    track = self.active_tracks[track_key]
                    track = track.update_status(TrackStatus.LOST)
                    self.active_tracks[track_key] = track
        
        self.reid_stats["lost_identities"] += len(lost_identities)
        
        if lost_identities:
            logger.info(f"Cleaned up {len(lost_identities)} lost identities")
    
    def reset_stats(self):
        """Reset ReID statistics."""
        self.reid_stats = {
            "total_identities": 0,
            "successful_matches": 0,
            "new_identities": 0,
            "merged_identities": 0,
            "lost_identities": 0
        }
        logger.info("ReID statistics reset")