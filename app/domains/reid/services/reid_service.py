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
import numpy as np

from app.domains.reid.entities.person_identity import PersonIdentity, IdentityStatus
from app.domains.reid.entities.track import Track, TrackStatus
from app.domains.reid.entities.feature_vector import FeatureVector, FeatureVectorBatch
from app.domains.detection.entities.detection import Detection, DetectionBatch
from app.domains.reid.models.base_reid_model import AbstractReIDModel
from app.domains.reid.models import ReIDModelFactory
from app.infrastructure.gpu import get_gpu_manager
from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class ReIDService:
    """
    Enhanced service for person re-identification operations.
    
    Features:
    - CLIP-based feature extraction with GPU acceleration
    - Advanced similarity matching with FAISS support
    - Identity fusion and track management
    - Cross-camera person association
    - Performance monitoring and optimization
    """
    
    def __init__(
        self, 
        model: Optional[AbstractReIDModel] = None,
        model_type: str = "clip",
        similarity_threshold: float = 0.65,
        enable_gpu: bool = True,
        batch_size: int = 16
    ):
        """
        Initialize ReID service.
        
        Args:
            model: Pre-configured ReID model instance
            model_type: Type of model to create if model is None
            similarity_threshold: Minimum similarity threshold for matching
            enable_gpu: Whether to use GPU acceleration
            batch_size: Batch size for feature extraction
        """
        self.similarity_threshold = similarity_threshold
        self.enable_gpu = enable_gpu
        self.batch_size = batch_size
        
        # GPU manager for resource allocation (must be initialized before model creation)
        self.gpu_manager = get_gpu_manager()
        
        # Initialize model
        if model is None:
            self.model = self._create_model(model_type)
        else:
            self.model = model
        
        # State management
        self.active_identities: Dict[str, PersonIdentity] = {}
        self.active_tracks: Dict[Tuple[CameraID, int], Track] = {}
        
        # Performance tracking
        self.reid_stats = {
            "total_identities": 0,
            "successful_matches": 0,
            "new_identities": 0,
            "merged_identities": 0,
            "lost_identities": 0,
            "feature_extractions": 0,
            "similarity_comparisons": 0,
            "gpu_enabled": enable_gpu,
            "batch_size": batch_size
        }
        
        logger.info(f"ReIDService initialized with model: {model_type}, GPU: {enable_gpu}")
    
    def _create_model(self, model_type: str) -> AbstractReIDModel:
        """Create and configure a ReID model instance."""
        try:
            # Get optimal device
            device = self.gpu_manager.get_optimal_device() if self.enable_gpu else "cpu"
            
            # Create model with appropriate configuration
            if model_type.lower() in ["clip", "clip_reid"]:
                model = ReIDModelFactory.create_model(
                    "clip",
                    device=device,
                    batch_size=self.batch_size,
                    use_faiss=True
                )
            else:
                logger.warning(f"Unknown model type: {model_type}, defaulting to CLIP")
                model = ReIDModelFactory.create_model(
                    "clip",
                    device=device,
                    batch_size=self.batch_size,
                    use_faiss=True
                )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating ReID model: {e}")
            raise
    
    async def initialize_model(self):
        """Initialize and warm up the ReID model."""
        try:
            await self.model.load_model()
            await self.model.warm_up()
            logger.info("ReID model initialized and warmed up successfully")
        except Exception as e:
            logger.error(f"Error initializing ReID model: {e}")
            raise
    
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
        """Extract features from a batch of detections using CLIP model."""
        if not detections:
            return []
        
        try:
            # Extract person images from detections
            person_images = []
            for detection in detections:
                # Get person crop from detection
                person_crop = await self._extract_person_crop(detection)
                person_images.append(person_crop)
            
            # Extract features using the CLIP model
            clip_features = await self.model.extract_features_batch(person_images)
            
            # Convert to FeatureVector objects
            feature_vectors = []
            for i, (detection, clip_feature) in enumerate(zip(detections, clip_features)):
                # Handle case where clip_feature might be numpy array or have features attribute
                if hasattr(clip_feature, 'features'):
                    feature_data = clip_feature.features.tolist()
                elif hasattr(clip_feature, 'tolist'):
                    feature_data = clip_feature.tolist()
                else:
                    feature_data = list(clip_feature)
                
                feature_vector = FeatureVector(
                    vector=feature_data,
                    extraction_timestamp=datetime.now(timezone.utc),
                    model_version=self.model.get_model_info().get("model_name", "clip"),
                    detection_id=detection.id,
                    camera_id=detection.camera_id,
                    frame_index=detection.frame_index
                )
                feature_vectors.append(feature_vector)
            
            # Update statistics
            self.reid_stats["feature_extractions"] += len(feature_vectors)
            
            return feature_vectors
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return dummy features as fallback
            return [
                FeatureVector(
                    vector=[0.0] * 512,
                    extraction_timestamp=datetime.now(timezone.utc),
                    model_version="dummy",
                    detection_id=detection.id,
                    camera_id=detection.camera_id,
                    frame_index=detection.frame_index
                )
                for detection in detections
            ]
    
    async def _extract_person_crop(self, detection: Detection) -> np.ndarray:
        """
        Extract person crop from detection.
        
        Args:
            detection: Detection object containing bounding box
            
        Returns:
            Cropped person image as numpy array
        """
        try:
            # For now, return a dummy image
            # In real implementation, this would crop the person from the frame
            # using the detection bounding box
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            return dummy_image
            
        except Exception as e:
            logger.error(f"Error extracting person crop: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    async def _match_to_identity(
        self,
        feature_vector: FeatureVector,
        camera_id: CameraID
    ) -> Tuple[Optional[PersonIdentity], float]:
        """Match feature vector to existing identity using CLIP model."""
        if not self.active_identities:
            return None, 0.0
        
        try:
            # Build query from feature vector using existing model
            query_features = np.array(feature_vector.vector, dtype=np.float32)
            
            # Get all candidate identities (excluding same camera)
            candidates = []
            candidate_features = []
            
            for identity in self.active_identities.values():
                # Skip if same camera (can't match within same camera)
                if camera_id in identity.cameras_seen:
                    continue
                
                # Get most recent feature vector for this identity
                identity_feature = await self._get_best_identity_feature(identity)
                
                if identity_feature is not None:
                    candidates.append(identity)
                    candidate_features.append(identity_feature)
            
            if not candidates:
                return None, 0.0
            
            # Use CLIP model for similarity matching
            best_match = None
            best_confidence = 0.0
            
            for identity, identity_feature in zip(candidates, candidate_features):
                # Calculate similarity using CLIP model directly with numpy arrays
                similarity = self.model._calculate_cosine_similarity(query_features, identity_feature)
                
                if similarity > best_confidence and similarity > self.similarity_threshold:
                    best_match = identity
                    best_confidence = similarity
            
            # Update statistics
            self.reid_stats["similarity_comparisons"] += len(candidates)
            
            if best_match:
                logger.debug(f"Matched identity {best_match.global_id} with confidence {best_confidence:.3f}")
            
            return best_match, best_confidence
            
        except Exception as e:
            logger.error(f"Error matching to identity: {e}")
            return None, 0.0
    
    async def _get_best_identity_feature(self, identity: PersonIdentity) -> Optional[Any]:
        """Get the best representative feature vector for an identity."""
        try:
            # Get all feature vectors from tracks of this identity
            identity_features = await self._get_identity_features(identity)
            
            if not identity_features:
                return None
            
            # For now, use the most recent feature vector
            # In future, could use averaged features or select best quality
            most_recent_feature = identity_features[-1]
            
            # Convert to CLIP feature format - return as numpy array
            clip_feature = np.array(most_recent_feature.vector, dtype=np.float32)
            
            return clip_feature
            
        except Exception as e:
            logger.error(f"Error getting best identity feature: {e}")
            return None
    
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
    
    async def cleanup(self):
        """Clean up ReID service resources."""
        try:
            if self.model:
                await self.model.cleanup()
                
            logger.info("ReID service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during ReID service cleanup: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ReID model information."""
        if self.model:
            return self.model.get_model_info()
        return {}
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get ReID model performance metrics."""
        if self.model:
            return self.model.get_performance_metrics()
        return {}
    
    def set_similarity_threshold(self, threshold: float):
        """Set similarity threshold for matching."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Similarity threshold set to {self.similarity_threshold}")
    
    def get_similarity_threshold(self) -> float:
        """Get current similarity threshold."""
        return self.similarity_threshold
    
    async def add_identity_to_database(self, identity: PersonIdentity):
        """Add identity to the model database for future matching."""
        try:
            # Get the best feature vector for this identity
            identity_feature = await self._get_best_identity_feature(identity)
            
            if identity_feature is not None:
                # Add to model database
                self.model.add_to_database(identity_feature, identity.global_id)
                logger.debug(f"Added identity {identity.global_id} to database")
                
        except Exception as e:
            logger.error(f"Error adding identity to database: {e}")
    
    async def remove_identity_from_database(self, global_id: str):
        """Remove identity from the model database."""
        try:
            success = self.model.remove_from_database(global_id)
            if success:
                logger.debug(f"Removed identity {global_id} from database")
            else:
                logger.warning(f"Failed to remove identity {global_id} from database")
                
        except Exception as e:
            logger.error(f"Error removing identity from database: {e}")
    
    def get_database_size(self) -> int:
        """Get the size of the feature database."""
        return self.model.get_database_size()
    
    def clear_database(self):
        """Clear the feature database."""
        self.model.clear_database()
        logger.info("ReID feature database cleared")
    
    async def update_identity_features(self, global_id: str):
        """Update features for an identity in the database."""
        try:
            identity = self.active_identities.get(global_id)
            if not identity:
                logger.warning(f"Identity {global_id} not found")
                return
            
            # Get updated feature vector
            identity_feature = await self._get_best_identity_feature(identity)
            
            if identity_feature is not None:
                # Update in database
                success = self.model.update_feature_in_database(global_id, identity_feature)
                if success:
                    logger.debug(f"Updated features for identity {global_id}")
                else:
                    logger.warning(f"Failed to update features for identity {global_id}")
                    
        except Exception as e:
            logger.error(f"Error updating identity features: {e}")
    
    async def find_similar_identities(
        self, 
        feature_vector: FeatureVector, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar identities using the model database.
        
        Args:
            feature_vector: Query feature vector
            top_k: Number of top results to return
            
        Returns:
            List of (identity_id, similarity_score) tuples
        """
        try:
            # Convert feature vector to numpy array for model search
            query_vector = np.array(feature_vector.vector, dtype=np.float32)
            
            # Find similar persons in database
            similar_persons = self.model.find_similar_persons(
                query_vector, 
                top_k=top_k, 
                similarity_threshold=self.similarity_threshold
            )
            
            return similar_persons
            
        except Exception as e:
            logger.error(f"Error finding similar identities: {e}")
            return []