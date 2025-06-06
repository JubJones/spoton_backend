"""
Module for object detection models.

This module contains implementations of object detectors, adhering to the
`AbstractDetector` interface defined in `base_models.py`. It demonstrates the
Strategy Pattern, allowing different detection algorithms to be used interchangeably
by the application's pipeline.
"""
from typing import List, Any, Dict, Optional, Tuple
import numpy as np
import asyncio
import logging
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import cv2

from .base_models import AbstractDetector, Detection, BoundingBox
from app.core.config import settings
from app.utils.video_processing import ensure_rgb

logger = logging.getLogger(__name__)

class FasterRCNNDetector(AbstractDetector):
    """
    Implementation of a Faster R-CNN detector using torchvision.

    This detector loads a pre-trained Faster R-CNN model with a ResNet50 FPN
    backbone. It is configured to detect persons based on the class ID specified
    in the application settings. It handles model loading, preprocessing,
    inference, and postprocessing to return a list of `Detection` objects.
    """
    def __init__(self):
        """
        Initializes the FasterRCNNDetector.

        Sets up model placeholders, configuration parameters from settings,
        and COCO class names.
        """
        self.model: Optional[torchvision.models.detection.FasterRCNN] = None
        self.transforms: Optional[torchvision.transforms.Compose] = None
        self.device: Optional[torch.device] = None # Set during load_model
        self._model_loaded_flag: bool = False # Added flag

        # Configuration from settings
        self.person_class_id: int = settings.PERSON_CLASS_ID
        self.detection_confidence_threshold: float = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.use_amp: bool = settings.DETECTION_USE_AMP

        # COCO class names (V2 weights are trained on COCO)
        self.coco_names = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta['categories']
        # Add background class at index 0 if necessary (torchvision usually doesn't include it in meta)
        if self.coco_names[0] != '__background__':
             self.coco_names.insert(0, '__background__')

        logger.info(f"FasterRCNNDetector configured for Person Class ID: {self.person_class_id}, Threshold: {self.detection_confidence_threshold}")


    async def load_model(self):
        """
        Loads the Faster R-CNN model and its preprocessor.

        The model is loaded onto the appropriate device (CUDA if available, else CPU).
        If the model is already loaded, this method does nothing.

        Raises:
            RuntimeError: If there is an error loading the model.
        """
        if self._model_loaded_flag and self.model is not None: # Check flag
            logger.info("Faster R-CNN model already loaded.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Faster R-CNN model on device: {self.device}...")

        try:
            # Load default weights V2
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, progress=True)
            self.transforms = weights.transforms()

            self.model.to(self.device)
            self.model.eval()
            self._model_loaded_flag = True # Set flag
            logger.info("Faster R-CNN model loaded and configured successfully.")

        except Exception as e:
            logger.exception(f"Error loading Faster R-CNN model: {e}")
            self.model = None
            self.transforms = None
            self._model_loaded_flag = False
            raise # Re-raise the exception to signal failure

    async def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """
        Warms up the model by performing a dummy inference.
        Should be called after load_model().
        """
        if not self._model_loaded_flag or not self.model or not self.device:
            logger.warning("Detector model not loaded. Cannot perform warmup.")
            return

        logger.info(f"Warming up FasterRCNN detector on device {self.device}...")
        try:
            dummy_np_image = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            # Use existing detect method for warmup logic, ensuring it's thread-safe
            # The detect method itself handles preprocessing and inference.
            _ = await self.detect(dummy_np_image)
            logger.info("FasterRCNN detector warmup successful.")
        except Exception as e:
            logger.error(f"FasterRCNN detector warmup failed: {e}", exc_info=True)


    async def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Performs person detection on an image.

        Args:
            image: A NumPy array representing the image (expected in BGR format from OpenCV).

        Returns:
            A list of Detection objects for persons found.

        Raises:
            RuntimeError: If the detector model has not been loaded.
        """
        if not self._model_loaded_flag or not self.model or not self.transforms or not self.device: # Check flag
            raise RuntimeError("Detector model not loaded. Call load_model() first.")

        # 1. Preprocessing
        try:
            img_rgb = await asyncio.to_thread(ensure_rgb, image)
            pil_image = Image.fromarray(img_rgb)
            # Apply transforms (typically just ToTensor)
            input_tensor = self.transforms(pil_image).to(self.device)
        except Exception as e:
            logger.error(f"Error during image preprocessing for detection: {e}")
            return []

        # 2. Inference (Run in thread pool)
        try:
            with torch.no_grad():
                use_amp_runtime = self.use_amp and self.device.type == 'cuda'
                with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                    # Model expects a list of tensors
                    predictions_list = await asyncio.to_thread(self.model, [input_tensor])
                    prediction_output = predictions_list[0] # Get result for the single image
        except Exception as e:
            logger.error(f"Error during detector inference: {e}")
            return []

        # 3. Postprocessing
        detections_result: List[Detection] = []
        try:
            # Detach tensors from computation graph before converting to NumPy
            pred_boxes = prediction_output['boxes'].detach().cpu().numpy()
            pred_labels = prediction_output['labels'].detach().cpu().numpy()
            pred_scores = prediction_output['scores'].detach().cpu().numpy()
            original_h, original_w = image.shape[:2]

            for box_coords, label_id, score_val in zip(pred_boxes, pred_labels, pred_scores):
                if int(label_id) == self.person_class_id and score_val >= self.detection_confidence_threshold:
                    x1, y1, x2, y2 = box_coords
                    # Clip coordinates to image bounds
                    x1_clipped = max(0.0, float(x1))
                    y1_clipped = max(0.0, float(y1))
                    x2_clipped = min(float(original_w), float(x2)) # Use W, H as upper bounds
                    y2_clipped = min(float(original_h), float(y2))

                    # Ensure box has positive area after clipping
                    if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                        class_name = "unknown"
                        if 0 <= int(label_id) < len(self.coco_names):
                            class_name = self.coco_names[int(label_id)]

                        detections_result.append(
                            Detection(
                                bbox=BoundingBox(x1=x1_clipped, y1=y1_clipped, x2=x2_clipped, y2=y2_clipped),
                                confidence=float(score_val),
                                class_id=int(label_id),
                                class_name=class_name
                            )
                        )
        except Exception as e:
            logger.error(f"Error during detection postprocessing: {e}")
            return [] # Return empty list on postprocessing error

        return detections_result