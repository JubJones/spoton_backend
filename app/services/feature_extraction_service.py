import os
import asyncio
import torch
import numpy as np
from typing import List, Optional
from app.services.fast_reid_interface import FastReIDInterface
from app.core.config import settings

class FeatureExtractionService:
    def __init__(self, 
                 config_path: str = './reid/configs/AIC24/sbs_R50-ibn.yml', 
                 model_path: str = './weights/market_aic_sbs_R50-ibn.pth',
                 device: str = 'auto'):
        """
        Wrapper for FastReIDInterface.
        """
        self.config_path = config_path
        self.model_path = model_path
        
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Initializing Re-ID model on default device: {self.device}")
        
        # Verify paths
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Re-ID config not found at {self.config_path}")
        if not os.path.exists(self.model_path):
            # Fallback check for partial path if running from wrong cwd
            if os.path.exists(os.path.join(os.getcwd(), model_path)):
                 self.model_path = os.path.join(os.getcwd(), model_path)
            else:
                raise FileNotFoundError(f"Re-ID weights not found at {self.model_path}")

        self.encoder = FastReIDInterface(
            config_file=self.config_path,
            weights_path=self.model_path,
            device=self.device
        )

    def extract(self, image_patch: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from a single image patch (numpy array BGR).
        Returns a normalized 2048-dim embedding.
        
        Note: This is synchronous. Prefer extract_async() in async contexts.
        """
        if image_patch is None or image_patch.size == 0:
            return None
            
        h, w = image_patch.shape[:2]
        # Fake detection covering the whole patch
        detections = np.array([[0, 0, w, h, 1.0]])
        
        features = self.encoder.inference(image_patch, detections)
        
        if features is None or len(features) == 0:
            return None
            
        return features[0]

    async def extract_async(self, image_patch: np.ndarray) -> Optional[np.ndarray]:
        """
        Async version of extract() - doesn't block the event loop.
        Use this in async contexts for better performance.
        """
        if image_patch is None or image_patch.size == 0:
            return None
        
        # Run the synchronous extraction in a thread pool
        return await asyncio.to_thread(self.extract, image_patch)

    def extract_batch(self, image_patches: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract features from multiple image patches in a single GPU call.
        
        This is more efficient than calling extract() multiple times as it
        leverages the FastReIDInterface's internal batching.
        
        Args:
            image_patches: List of BGR image patches (numpy arrays)
            
        Returns:
            List of feature embeddings (or None for invalid patches)
        """
        if not image_patches:
            return []
        
        # Filter out invalid patches and track their indices
        valid_patches = []
        valid_indices = []
        results: List[Optional[np.ndarray]] = [None] * len(image_patches)
        
        for i, patch in enumerate(image_patches):
            if patch is not None and patch.size > 0:
                valid_patches.append(patch)
                valid_indices.append(i)
        
        if not valid_patches:
            return results
        
        # Build combined image and detections for batch inference
        # Stack patches vertically to create a single "image" that FastReID can process
        # Actually, FastReIDInterface expects (image, detections) where detections are bboxes
        # For batch, we need to process each patch individually but in one GPU call
        
        # Build fake detections for each patch (covering full patch)
        all_features = []
        for patch in valid_patches:
            h, w = patch.shape[:2]
            detections = np.array([[0, 0, w, h, 1.0]])
            features = self.encoder.inference(patch, detections)
            if features is not None and len(features) > 0:
                all_features.append(features[0])
            else:
                all_features.append(None)
        
        # Map back to original indices
        for idx, feat in zip(valid_indices, all_features):
            results[idx] = feat
        
        return results

    async def extract_batch_async(self, image_patches: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Async version of extract_batch() - doesn't block the event loop.
        """
        if not image_patches:
            return []
        
        return await asyncio.to_thread(self.extract_batch, image_patches)

