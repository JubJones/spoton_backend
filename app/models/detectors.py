from typing import List, Any, Dict, Optional
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
    Loads a pre-trained Faster R-CNN model with a ResNet50 FPN backbone.
    Focuses on detecting persons based on configured class ID.
    """
    def __init__(self):
        self.model: Optional[torchvision.models.detection.FasterRCNN] = None
        self.transforms: Optional[torchvision.transforms.Compose] = None
        self.device: Optional[torch.device] = None # Set during load_model

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
        """Loads the Faster R-CNN model."""
        if self.model is not None:
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
            logger.info("Faster R-CNN model loaded and configured successfully.")

        except Exception as e:
            logger.exception(f"Error loading Faster R-CNN model: {e}")
            self.model = None
            self.transforms = None
            raise # Re-raise the exception to signal failure

    async def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Performs person detection on an image.

        Args:
            image: A NumPy array representing the image (expected in BGR format from OpenCV).

        Returns:
            A list of Detection objects for persons found.
        """
        if not self.model or not self.transforms or not self.device:
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
            pred_boxes = prediction_output['boxes'].cpu().numpy()
            pred_labels = prediction_output['labels'].cpu().numpy()
            pred_scores = prediction_output['scores'].cpu().numpy()
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