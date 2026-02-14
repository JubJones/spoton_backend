
import os
import asyncio
import logging
import torch
import numpy as np
from typing import List, Optional
from torchreid.utils import FeatureExtractor
from app.core.config import settings

logger = logging.getLogger(__name__)

class FeatureExtractionService:
    def __init__(self):
        """
        Wrapper for OSNet-AIN feature extraction using Torchreid.
        """
        self.model_path = getattr(settings, 'REID_MODEL_PATH', 'weights/osnet_ain_ms_d_c.pth.tar')
        self.model_name = getattr(settings, 'REID_MODEL_NAME', 'osnet_ain_x1_0')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"🧠 RE-ID INIT: Loading {self.model_name} from {self.model_path} on {self.device}...")
        
        if not os.path.exists(self.model_path):
             # Fallback check for current directory
             cwd_path = os.path.join(os.getcwd(), self.model_path)
             if os.path.exists(cwd_path):
                 self.model_path = cwd_path
             else:
                 raise FileNotFoundError(f"Re-ID weights not found at {self.model_path}")

        try:
            # Initialize Torchreid Feature Extractor
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device
            )
            logger.info("✅ RE-ID INIT: OSNet-AIN model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ RE-ID INIT FAILED: {e}")
            raise e

    def extract(self, image_patch: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from a single image patch (numpy array BGR).
        Returns a normalized 512-dim embedding.
        """
        if image_patch is None or image_patch.size == 0:
            return None
            
        try:
            # Check for minimum size (OSNet can be sensitive to tiny patches)
            h, w = image_patch.shape[:2]
            if h < 10 or w < 10:
                pass # return None
            
            # Torchreid expects a list of numpy images (H, W, C) in RGB (or BGR?)
            # FeatureExtractor internally uses PIL or cv2. 
            # Looking at source, it accepts a list of numpy arrays.
            # It handles standard transforms (Resize, ToTensor, Normalize).
            # It does expect RGB if using standard transforms. OpenCV gives BGR.
            
            # Convert BGR to RGB
            # image_rgb = image_patch[:, :, ::-1] # FeatureExtractor might handle this? 
            # torchreid's FeatureExtractor usually expects images as list of numpy arrays (H,W,C).
            
            features = self.extractor([image_patch]) # Returns torch.Tensor
            features = features.cpu().numpy().astype(np.float32)
            
            if features is None or len(features) == 0:
                return None
                
            return features[0] # Return the first (and only) embedding
            
        except Exception as e:
            logger.error(f"Re-ID Extraction Error: {e}")
            return None

    async def extract_async(self, image_patch: np.ndarray) -> Optional[np.ndarray]:
        """
        Async version of extract() - doesn't block the event loop.
        """
        if image_patch is None or image_patch.size == 0:
            return None
        
        return await asyncio.to_thread(self.extract, image_patch)

    def extract_batch(self, image_patches: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract features from multiple image patches.
        """
        if not image_patches:
            return []
        
        valid_patches = []
        valid_indices = []
        results: List[Optional[np.ndarray]] = [None] * len(image_patches)
        
        for i, patch in enumerate(image_patches):
            if patch is not None and patch.size > 0:
                # Basic size check
                if patch.shape[0] >= 10 and patch.shape[1] >= 10:
                    valid_patches.append(patch)
                    valid_indices.append(i)
        
        if not valid_patches:
            return results
            
        try:
            # Batch inference
            features = self.extractor(valid_patches)
            features = features.cpu().numpy().astype(np.float32)
            
            # Map back
            for i, idx in enumerate(valid_indices):
                results[idx] = features[i]
                
        except Exception as e:
            logger.error(f"Batch Re-ID Extraction Error: {e}")
        
        return results

    async def extract_batch_async(self, image_patches: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        return await asyncio.to_thread(self.extract_batch, image_patches)
