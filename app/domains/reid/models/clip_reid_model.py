"""
CLIP-based ReID model implementation with GPU acceleration.
Advanced person re-identification using CLIP features and similarity matching.
"""
import logging
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

from app.domains.reid.entities.feature_vector import FeatureVector
from app.domains.reid.models.base_reid_model import AbstractReIDModel, ReIDModelFactory
from app.infrastructure.gpu import get_gpu_manager

logger = logging.getLogger(__name__)


class CLIPReIDModel(AbstractReIDModel):
    """
    CLIP-based person re-identification model.
    
    Features:
    - CLIP feature extraction for robust person representation
    - GPU acceleration for fast inference
    - Batch processing for efficient multi-person handling
    - Advanced similarity matching with FAISS support
    - Feature normalization and dimensionality reduction
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        feature_dim: int = 512,
        batch_size: int = 32,
        normalize_features: bool = True,
        use_faiss: bool = True,
        faiss_index_type: str = "IndexFlatIP"  # Inner Product for cosine similarity
    ):
        """
        Initialize CLIP ReID model.
        
        Args:
            model_name: CLIP model name from HuggingFace
            device: Device for inference (auto-detected if None)
            feature_dim: Feature vector dimension
            batch_size: Batch size for inference
            normalize_features: Whether to normalize feature vectors
            use_faiss: Whether to use FAISS for similarity search
            faiss_index_type: FAISS index type
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install PyTorch.")
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.normalize_features = normalize_features
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index_type = faiss_index_type
        
        # Model and device setup
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device = device or get_gpu_manager().get_optimal_device()
        self.torch_device = torch.device(self.device)
        
        # Feature database for similarity matching
        self.feature_database: List[FeatureVector] = []
        self.feature_ids: List[str] = []
        self.faiss_index: Optional[Any] = None
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.total_extractions = 0
        self.total_comparisons = 0
        
        logger.info(f"CLIPReIDModel initialized with device: {self.device}")
    
    async def load_model(self) -> None:
        """Load the CLIP model and processor."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.torch_device)
            self.model.eval()
            
            # Initialize FAISS index if enabled
            if self.use_faiss:
                self._initialize_faiss_index()
            
            logger.info(f"CLIP ReID model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP ReID model: {e}")
            raise
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for similarity search."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, falling back to CPU search")
            return
        
        try:
            if self.faiss_index_type == "IndexFlatIP":
                self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
            elif self.faiss_index_type == "IndexFlatL2":
                self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
            else:
                logger.warning(f"Unknown FAISS index type: {self.faiss_index_type}")
                self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
            
            # Move to GPU if available
            if self.device.startswith('cuda') and hasattr(faiss, 'StandardGpuResources'):
                try:
                    gpu_res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, self.faiss_index)
                    logger.info("FAISS index moved to GPU")
                except Exception as e:
                    logger.warning(f"Failed to move FAISS to GPU: {e}")
            
            logger.info(f"FAISS index initialized: {self.faiss_index_type}")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            self.faiss_index = None
    
    async def extract_features(self, image: np.ndarray) -> FeatureVector:
        """
        Extract features from a single person image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Feature vector
        """
        if self.model is None:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            # Preprocess image
            preprocessed = await self.preprocess_image(image)
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(preprocessed)
            
            # Convert to numpy and normalize
            features = image_features.cpu().numpy().flatten()
            
            if self.normalize_features:
                features = self._normalize_features(features)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_extractions += 1
            
            # Keep only recent inference times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return FeatureVector(
                vector=features.tolist(),
                extraction_timestamp=datetime.now(),
                model_version=f"clip_{self.model_name.split('/')[-1]}"
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return FeatureVector(
                vector=np.zeros(self.feature_dim, dtype=np.float32).tolist(),
                extraction_timestamp=datetime.now(),
                model_version=f"clip_{self.model_name.split('/')[-1]}"
            )
    
    async def extract_features_batch(self, images: List[np.ndarray]) -> List[FeatureVector]:
        """
        Extract features from multiple person images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of feature vectors
        """
        if self.model is None:
            await self.load_model()
        
        if not images:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_features = await self._process_batch(batch_images)
            results.extend(batch_features)
        
        return results
    
    async def _process_batch(self, batch_images: List[np.ndarray]) -> List[FeatureVector]:
        """Process a batch of images."""
        start_time = time.time()
        
        try:
            # Preprocess all images in batch
            batch_tensors = []
            for image in batch_images:
                preprocessed = await self.preprocess_image(image)
                batch_tensors.append(preprocessed)
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.get_image_features(batch_tensor)
            
            # Convert to numpy and normalize
            features_list = []
            for i in range(batch_features.shape[0]):
                features = batch_features[i].cpu().numpy().flatten()
                
                if self.normalize_features:
                    features = self._normalize_features(features)
                
                features_list.append(FeatureVector(
                    vector=features.tolist(),
                    extraction_timestamp=datetime.now(),
                    model_version=f"clip_{self.model_name.split('/')[-1]}"
                ))
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_extractions += len(batch_images)
            
            return features_list
            
        except Exception as e:
            logger.error(f"Error in batch feature extraction: {e}")
            return [FeatureVector(
                vector=np.zeros(self.feature_dim, dtype=np.float32).tolist(),
                extraction_timestamp=datetime.now(),
                model_version=f"clip_{self.model_name.split('/')[-1]}"
            ) for _ in batch_images]
    
    async def preprocess_image(self, image: np.ndarray) -> Union[Any, 'torch.Tensor']:
        """
        Preprocess image for CLIP.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Process with CLIP processor
            inputs = self.processor(
                images=image_rgb,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            pixel_values = inputs['pixel_values'].to(self.torch_device)
            
            return pixel_values
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return dummy tensor
            return torch.zeros(1, 3, 224, 224, device=self.torch_device)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector."""
        norm = np.linalg.norm(features)
        if norm > 1e-6:
            return features / norm
        return features
    
    def add_to_database(self, feature_vector: FeatureVector, person_id: str):
        """
        Add feature vector to the database.
        
        Args:
            feature_vector: Feature vector to add
            person_id: Person identifier
        """
        try:
            self.feature_database.append(feature_vector)
            self.feature_ids.append(person_id)
            
            # Add to FAISS index if available
            if self.use_faiss and self.faiss_index is not None:
                features = feature_vector.features.reshape(1, -1).astype(np.float32)
                self.faiss_index.add(features)
            
            logger.debug(f"Added feature vector for person {person_id} to database")
            
        except Exception as e:
            logger.error(f"Error adding feature to database: {e}")
    
    def find_similar_persons(
        self, 
        query_features: FeatureVector,
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find similar persons in the database.
        
        Args:
            query_features: Query feature vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (person_id, similarity_score) tuples
        """
        if not self.feature_database:
            return []
        
        try:
            if self.use_faiss and self.faiss_index is not None:
                return self._faiss_search(query_features, top_k, similarity_threshold)
            else:
                return self._cpu_search(query_features, top_k, similarity_threshold)
                
        except Exception as e:
            logger.error(f"Error finding similar persons: {e}")
            return []
    
    def _faiss_search(
        self, 
        query_features: FeatureVector,
        top_k: int,
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """FAISS-based similarity search."""
        try:
            query = query_features.features.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(query, min(top_k, len(self.feature_database)))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.feature_ids):
                    # Convert score based on index type
                    if self.faiss_index_type == "IndexFlatIP":
                        # Inner product score (higher is better)
                        similarity = float(score)
                    else:
                        # L2 distance (lower is better, convert to similarity)
                        similarity = 1.0 / (1.0 + float(score))
                    
                    if similarity >= similarity_threshold:
                        results.append((self.feature_ids[idx], similarity))
            
            self.total_comparisons += len(results)
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return []
    
    def _cpu_search(
        self, 
        query_features: FeatureVector,
        top_k: int,
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """CPU-based similarity search."""
        try:
            query = query_features.features
            similarities = []
            
            for i, db_features in enumerate(self.feature_database):
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(query, db_features.features)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filter and return top-k results
            results = []
            for idx, similarity in similarities[:top_k]:
                if similarity >= similarity_threshold:
                    results.append((self.feature_ids[idx], similarity))
            
            self.total_comparisons += len(similarities)
            return results
            
        except Exception as e:
            logger.error(f"Error in CPU search: {e}")
            return []
    
    def _calculate_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        try:
            # Normalize features
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def clear_database(self):
        """Clear the feature database."""
        self.feature_database.clear()
        self.feature_ids.clear()
        
        # Reset FAISS index
        if self.use_faiss and self.faiss_index is not None:
            self._initialize_faiss_index()
        
        logger.info("Feature database cleared")
    
    def get_database_size(self) -> int:
        """Get the size of the feature database."""
        return len(self.feature_database)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'fps': 0.0,
                'total_extractions': 0,
                'total_comparisons': 0,
                'database_size': 0
            }
        
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        return {
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'total_extractions': self.total_extractions,
            'total_comparisons': self.total_comparisons,
            'database_size': len(self.feature_database),
            'recent_inference_times': self.inference_times[-10:],
            'use_faiss': self.use_faiss,
            'faiss_available': FAISS_AVAILABLE
        }
    
    async def warm_up(self) -> None:
        """Warm up the model with dummy inference."""
        if self.model is None:
            await self.load_model()
        
        logger.info("Warming up CLIP ReID model...")
        
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Run a few warm-up inferences
        for _ in range(3):
            await self.extract_features(dummy_image)
        
        logger.info("CLIP ReID model warmed up successfully")
    
    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            self.model = None
            self.processor = None
            
        # Clear database
        self.clear_database()
        
        # Clear CUDA cache if using GPU
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("CLIP ReID model cleaned up")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': 'CLIP ReID Model',
            'model_name': self.model_name,
            'device': self.device,
            'feature_dim': self.feature_dim,
            'batch_size': self.batch_size,
            'normalize_features': self.normalize_features,
            'use_faiss': self.use_faiss,
            'faiss_index_type': self.faiss_index_type,
            'loaded': self.is_loaded(),
            'database_size': self.get_database_size()
        }
    
    def calculate_similarity(
        self, 
        features1: FeatureVector, 
        features2: FeatureVector
    ) -> float:
        """
        Calculate similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        return self._calculate_cosine_similarity(features1.features, features2.features)
    
    def update_feature_in_database(
        self, 
        person_id: str, 
        new_features: FeatureVector
    ) -> bool:
        """
        Update feature vector for a person in the database.
        
        Args:
            person_id: Person identifier
            new_features: New feature vector
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Find person in database
            for i, existing_id in enumerate(self.feature_ids):
                if existing_id == person_id:
                    # Update feature vector
                    self.feature_database[i] = new_features
                    
                    # Rebuild FAISS index if using FAISS
                    if self.use_faiss and self.faiss_index is not None:
                        self._rebuild_faiss_index()
                    
                    logger.debug(f"Updated feature vector for person {person_id}")
                    return True
            
            logger.warning(f"Person {person_id} not found in database")
            return False
            
        except Exception as e:
            logger.error(f"Error updating feature in database: {e}")
            return False
    
    def remove_from_database(self, person_id: str) -> bool:
        """
        Remove person from the database.
        
        Args:
            person_id: Person identifier
            
        Returns:
            True if removal successful, False otherwise
        """
        try:
            # Find person in database
            for i, existing_id in enumerate(self.feature_ids):
                if existing_id == person_id:
                    # Remove from lists
                    del self.feature_database[i]
                    del self.feature_ids[i]
                    
                    # Rebuild FAISS index if using FAISS
                    if self.use_faiss and self.faiss_index is not None:
                        self._rebuild_faiss_index()
                    
                    logger.debug(f"Removed person {person_id} from database")
                    return True
            
            logger.warning(f"Person {person_id} not found in database")
            return False
            
        except Exception as e:
            logger.error(f"Error removing person from database: {e}")
            return False
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index after database changes."""
        if not self.use_faiss or not FAISS_AVAILABLE:
            return
        
        try:
            # Initialize new index
            self._initialize_faiss_index()
            
            # Add all features to index
            if self.feature_database:
                features_array = np.array([f.features for f in self.feature_database], dtype=np.float32)
                self.faiss_index.add(features_array)
            
            logger.debug("FAISS index rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
            self.faiss_index = None


# Register the model
ReIDModelFactory.register_model('clip', CLIPReIDModel)
ReIDModelFactory.register_model('clip_reid', CLIPReIDModel)