
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import glob

logger = logging.getLogger(__name__)

class GroundTruthReIDService:
    """
    Service to provide perfect ground-truth Re-ID assignments from dataset files.
    Bypasses neural feature extraction and matching.
    """

    def __init__(self, data_root: str):
        """
        Initialize the service.
        
        Args:
            data_root: Path to directory containing gt_{cam_id}.txt files
                       (e.g., /app/videos/gt/campus)
        """
        self.data_root = data_root
        # Structure: camera_id -> frame_id -> List of (person_id, bbox_dict)
        self.gt_data: Dict[str, Dict[int, List[Tuple[int, Dict]]]] = {}
        self._load_ground_truth()

    def _load_ground_truth(self):
        """Load all gt_*.txt files from data_root."""
        if not os.path.exists(self.data_root):
            logger.error(f"❌ GT-REID: Data root not found: {self.data_root}")
            return

        pattern = os.path.join(self.data_root, "gt_*.txt")
        gt_files = glob.glob(pattern)
        
        if not gt_files:
            logger.warning(f"⚠️ GT-REID: No gt_*.txt files found in {self.data_root}")
            return

        logger.info(f"🔍 GT-REID: Loading {len(gt_files)} GT files from {self.data_root}...")

        for file_path in gt_files:
            try:
                # Extract camera ID from filename: gt_c01.txt -> c01
                filename = os.path.basename(file_path)
                cam_id = filename.replace("gt_", "").replace(".txt", "")
                
                self.gt_data[cam_id] = {}
                count = 0
                
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 6:
                            continue
                        
                        # parsed format: frame, id, x, y, w, h, ...
                        # Frame is 0-indexed in our system, check if file is 1-indexed?
                        # Standard MOT/Market format is usually 1-indexed frame? 
                        # Looking at user's gt.txt sample: "3682,071,..."
                        # Let's assume it matches the frame number from the video stream directly.
                        
                        frame_idx = int(parts[0])
                        pid = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])
                        
                        bbox = {
                            'x1': x,
                            'y1': y,
                            'x2': x + w,
                            'y2': y + h
                        }
                        
                        if frame_idx not in self.gt_data[cam_id]:
                            self.gt_data[cam_id][frame_idx] = []
                        
                        self.gt_data[cam_id][frame_idx].append((pid, bbox))
                        count += 1
                logger.warning(f"  ✅ Loaded {cam_id}: {count} annotations")
                
            except Exception as e:
                logger.error(f"❌ GT-REID: Error loading {file_path}: {e}")

    def get_identity(self, camera_id: str, frame_number: int, detection_bbox: Dict[str, float]) -> Optional[str]:
        """
        Match a detection to ground truth and return the Global ID.
        
        Args:
            camera_id: Camera identifier (e.g. 'c01')
            frame_number: Current frame index
            detection_bbox: Dict with x1, y1, x2, y2 keys
            
        Returns:
            Global Person ID (str) if matched, else None
        """
        if camera_id not in self.gt_data:
            return None
            
        # Check if we have GT for this exact frame
        if frame_number not in self.gt_data[camera_id]:
            # GT might be sparse or offset? 
            # For now assume exact match.
            return None
            
        candidates = self.gt_data[camera_id][frame_number]
        
        best_iou = 0
        best_pid = None
        
        for pid, gt_box in candidates:
            # Calculate IoU
            det_box = [detection_bbox['x1'], detection_bbox['y1'], detection_bbox['x2'], detection_bbox['y2']]
            gt_box_list = [gt_box['x1'], gt_box['y1'], gt_box['x2'], gt_box['y2']]
            
            iou = self._calculate_iou(det_box, gt_box_list)
            if iou > best_iou:
                best_iou = iou
                best_pid = pid
        
        logger.warning(f"[GT DEBUG] get_identity: best_iou={best_iou} for Frame {frame_number} Cam {camera_id}")

        # Threshold for matching (e.g. 0.3 IoU)
        if best_iou > 0.3:
            return str(best_pid)
            
        return None

    def get_detections(self, camera_id: str, frame_number: int) -> List[Dict[str, Any]]:
        """
        Return all ground truth detections (bounding boxes and identities) for a given frame.
        These can be used identically to YOLO detections to map precise GT coordinates.
        """
        if camera_id not in self.gt_data:
            logger.warning(f"[GT DEBUG] get_detections: camera_id {camera_id} not in gt_data keys ({list(self.gt_data.keys())})")
            return []
            
        if frame_number not in self.gt_data[camera_id]:
            logger.warning(f"[GT DEBUG] get_detections: frame_number {frame_number} not in gt_data[{camera_id}]")
            return []
            
        detections = []
        for pid, bbox in self.gt_data[camera_id][frame_number]:
            detections.append({
                'track_id': str(pid),
                'bbox': bbox,
                'confidence': 1.0,
                'class_id': 0 # Class 0 represents Person in our pipeline
            })
            
        return detections

    def _calculate_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
