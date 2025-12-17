import os
import torch
import numpy as np
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

    def extract(self, image_patch: np.ndarray) -> np.ndarray:
        """
        Extract features from a single image patch (numpy array BGR).
        Returns a normalized 2048-dim embedding.
        """
        if image_patch is None or image_patch.size == 0:
            return None
            
        # FastReIDInterface.inference expects a list or array of detections [x1, y1, x2, y2, score]
        # But here we already have the cropped patch.
        # FastReIDInterface.inference logic includes cropping from the original image based on detections.
        # We need to adapt this. 
        # The inference method in the reference code takes (image, detections). 
        # If we pass the patch itself as the "image" and a detection covering the whole patch, it should work.
        
        h, w = image_patch.shape[:2]
        # Fake detection covering the whole patch
        detections = np.array([[0, 0, w, h, 1.0]])
        
        features = self.encoder.inference(image_patch, detections)
        
        if features is None or len(features) == 0:
            return None
            
        return features[0]
