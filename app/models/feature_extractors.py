"""
Module for standalone Re-ID feature extraction models.
"""
import logging
from typing import Optional, Any
import numpy as np
import torch
import asyncio
from pathlib import Path

try:
    from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    BOXMOT_REID_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Failed to import BoxMOT ReidAutoBackend. Standalone ReID unavailable. Error: {e}")
    BOXMOT_REID_AVAILABLE = False
    ReidAutoBackend = type(None) # Dummy type

from app.models.base_models import AbstractFeatureExtractor
from app.core.config import settings
from app.utils.device_utils import get_boxmot_device_string

logger = logging.getLogger(__name__)

class BoxMOTFeatureExtractor(AbstractFeatureExtractor):
    """
    Standalone Re-ID feature extractor using BoxMOT's ReidAutoBackend.
    """
    def __init__(self, device: torch.device):
        """
        Initializes the BoxMOTFeatureExtractor.

        Args:
            device: The torch device to load the model onto.
        """
        if not BOXMOT_REID_AVAILABLE or ReidAutoBackend is None:
            raise ImportError("BoxMOT ReidAutoBackend is required but not available.")

        self.device = device
        self.reid_model_handler: Optional[Any] = None # Stores the ReidAutoBackend instance
        self.reid_model_path: Path = settings.resolved_reid_weights_path
        self.model_type: str = settings.REID_MODEL_TYPE # e.g. 'clip', 'osnet'
        self.use_half: bool = settings.REID_MODEL_HALF_PRECISION
        self._model_loaded_flag = False
        logger.info(
            f"BoxMOTFeatureExtractor configured. Model: {self.model_type}, "
            f"Weights: {self.reid_model_path}, Device: {self.device}, Half: {self.use_half}"
        )

    async def load_model(self):
        """Loads the Re-ID model using ReidAutoBackend."""
        if self._model_loaded_flag:
            logger.info("Standalone ReID model already loaded.")
            return

        logger.info(f"Loading standalone ReID model ({self.model_type}) on device: {self.device}...")
        boxmot_device_str = get_boxmot_device_string(self.device)
        effective_half = self.use_half and self.device.type == 'cuda'

        try:
            # ReidAutoBackend handles model loading based on type and weights
            self.reid_model_handler = await asyncio.to_thread(
                ReidAutoBackend,
                weights=self.reid_model_path,
                device=boxmot_device_str,
                half=effective_half,
                model_type=self.model_type # Pass model_type if ReidAutoBackend supports it explicitly
                                           # Otherwise, it might infer from weights filename.
            )
            # Warmup (optional, check if ReidAutoBackend or its underlying model supports it)
            if hasattr(self.reid_model_handler.model, "warmup"): # Access underlying model
                 logger.info("Warming up standalone ReID model...")
                 await asyncio.to_thread(self.reid_model_handler.model.warmup)
                 logger.info("Standalone ReID model warmup complete.")
            
            self._model_loaded_flag = True
            logger.info(f"Standalone ReID model '{self.model_type}' loaded successfully from '{self.reid_model_path}'.")

        except Exception as e:
            logger.exception(f"Error loading standalone ReID model: {e}")
            self.reid_model_handler = None
            self._model_loaded_flag = False
            raise RuntimeError(f"Failed to load standalone ReID model: {e}") from e

    async def get_features(self, bboxes_xyxy: np.ndarray, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts features for multiple bounding boxes from a single image.
        """
        if not self._model_loaded_flag or self.reid_model_handler is None:
            logger.error("Standalone ReID model not loaded. Call load_model() first.")
            return None
        
        if bboxes_xyxy is None or bboxes_xyxy.size == 0:
            return np.empty((0, 0)) # Return empty array if no bboxes

        try:
            # ReidAutoBackend's get_features method expects bboxes and the image
            features_np = await asyncio.to_thread(
                self.reid_model_handler.get_features, # Call method of the handler instance
                bboxes_xyxy,
                image_bgr
            )
            return features_np
        except Exception as e:
            logger.error(f"Error during standalone ReID feature extraction: {e}", exc_info=True)
            return None