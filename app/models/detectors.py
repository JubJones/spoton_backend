from typing import List, Any, Dict, Optional
import numpy as np
import asyncio
import torch
import torchvision
# Using V2 weights for potentially better accuracy and more recent pre-training
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image # For converting numpy array to PIL Image for transforms
import cv2 # For BGR to RGB conversion

from .base_models import AbstractDetector, Detection, BoundingBox
from app.core.config import settings
from app.utils.image_processing import ensure_rgb # Assuming a utility for BGR->RGB if not done here

class FasterRCNNDetector(AbstractDetector):
    """
    Implementation of a Faster R-CNN detector using torchvision.
    It loads a pre-trained Faster R-CNN model with a ResNet50 FPN backbone.
    """
    def __init__(self):
        self.model: Optional[torchvision.models.detection.FasterRCNN] = None
        self.transforms: Optional[torchvision.transforms.Compose] = None
        self.device: Optional[torch.device] = None
        
        # Configuration from settings
        self.person_class_id: int = settings.PERSON_CLASS_ID
        self.detection_confidence_threshold: float = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.use_amp: bool = settings.DETECTION_USE_AMP

        # COCO class names are useful for human-readable labels.
        # FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT is trained on COCO.
        # Class ID 1 is 'person'.
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        settings.DETECTOR_MODEL_PATH # To silence unused var if not used, but it's available from config

    async def load_model(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Loads the Faster R-CNN model.
        If `settings.DETECTOR_MODEL_PATH` (via `model_path` arg or directly from settings)
        is provided, it attempts to load custom weights. Otherwise, uses default
        pre-trained weights from torchvision.
        """
        if self.model is not None:
            print("Faster R-CNN model already loaded.")
            return

        print("Loading Faster R-CNN model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for FasterRCNNDetector: {self.device}")

        try:
            # Determine if custom weights path is provided
            custom_weights_path = model_path or settings.DETECTOR_MODEL_PATH

            if custom_weights_path:
                print(f"Attempting to load custom weights from: {custom_weights_path}")
                # Load model architecture without pre-trained weights
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, progress=True)
                # Load state dict from the path
                # Running torch.load in a thread as it can be disk I/O bound
                state_dict = await asyncio.to_thread(torch.load, custom_weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded custom weights from: {custom_weights_path}")
                # For custom weights, transforms might need to be defined differently or assumed to be simple ToTensor
                # For now, we'll use standard ToTensor, but this might need adjustment based on how custom model was trained
                self.transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()
                ])

            else:
                print("Loading default pre-trained COCO weights for FasterRCNN_ResNet50_FPN_V2.")
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, progress=True)
                # The transforms associated with these weights typically just convert to tensor.
                # Normalization and resizing are handled by GeneralizedRCNNTransform within the model.
                self.transforms = weights.transforms() # Usually Compose(ConvertImageDtype())

            # Move model to device (model.to is generally quick, not warranting asyncio.to_thread itself unless model is huge)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            
            print("Faster R-CNN model loaded and configured successfully.")

        except Exception as e:
            print(f"Error loading Faster R-CNN model: {e}")
            self.model = None
            self.transforms = None
            # self.device = None # Keep device if it was set, for potential retry? Or clear it.
            raise # Re-raise the exception to signal failure

    async def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Performs object detection on an image.

        Args:
            image: A NumPy array representing the image (expected in BGR format from OpenCV).

        Returns:
            A list of Detection objects.
        """
        if not self.model or not self.transforms or not self.device:
            raise RuntimeError("Detector model not loaded. Call load_model() first.")

        # 1. Preprocessing
        # Ensure image is RGB. If it's from OpenCV, it's BGR.
        # Using cv2.cvtColor is explicit. Ensure 'image' is a NumPy array.
        if image.shape[2] == 3: # Basic check for 3 channels
             # Using the utility function for clarity, or cv2.cvtColor directly
            img_rgb = await asyncio.to_thread(ensure_rgb, image) # Assuming ensure_rgb handles BGR->RGB or keeps RGB
            # Or: img_rgb = await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_BGR2RGB)
        else: # Grayscale or other formats not directly supported by standard RGB models
            print(f"Warning: Input image has {image.shape[2]} channels. Expected 3 (RGB/BGR). Detection might fail or be inaccurate.")
            img_rgb = image # Pass as is, model might handle or fail.

        pil_image = Image.fromarray(img_rgb)
        
        try:
            # The default transforms for detection models (e.g., FasterRCNN_ResNet50_FPN_V2_Weights.transforms())
            # is typically just `Compose([ConvertImageDtype()])`. This converts to float tensor and scales to [0,1].
            # It does NOT do normalization (mean/std) or resizing. These are handled by the
            # `GeneralizedRCNNTransform` *inside* the FasterRCNN model.
            # Importantly, the output bounding boxes from the model are in the original image's coordinate system.
            input_tensor = self.transforms(pil_image).to(self.device)
        except Exception as e:
            print(f"Error during image transformation: {e}")
            return []

        # 2. Inference (heavy computation, run in a thread pool)
        try:
            with torch.no_grad():
                use_amp_runtime = self.use_amp and self.device.type == 'cuda' and torch.cuda.is_available()
                with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                    # Model expects a batch, so wrap the tensor in a list
                    predictions_list = await asyncio.to_thread(self.model, [input_tensor])
                    # Prediction is a list of dicts, one per image in the batch. We have one image.
                    prediction_output = predictions_list[0] 
        except Exception as e:
            print(f"Error during model inference: {e}")
            return []

        # 3. Postprocessing
        detections_result: List[Detection] = []
        pred_boxes = prediction_output['boxes'].cpu().numpy()
        pred_labels = prediction_output['labels'].cpu().numpy()
        pred_scores = prediction_output['scores'].cpu().numpy()

        original_h, original_w = image.shape[:2] # Height, Width of the original input image

        for box_coords, label_id, score_val in zip(pred_boxes, pred_labels, pred_scores):
            if int(label_id) == self.person_class_id and score_val >= self.detection_confidence_threshold:
                # Boxes from torchvision FasterRCNN are already in original image coordinates (x1, y1, x2, y2)
                # No scaling back is needed if using standard torchvision model and its minimal transforms.
                x1, y1, x2, y2 = box_coords
                
                # Clip coordinates to be strictly within image bounds and ensure valid box
                x1_clipped = max(0.0, float(x1))
                y1_clipped = max(0.0, float(y1))
                x2_clipped = min(float(original_w - 1), float(x2)) # Max index is W-1 or H-1
                y2_clipped = min(float(original_h - 1), float(y2))

                # Skip if the box is invalid after clipping (e.g., zero width/height)
                if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
                    continue

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
        return detections_result