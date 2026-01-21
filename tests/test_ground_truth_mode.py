
import sys
from unittest.mock import MagicMock

# Mock faiss before it is imported by fastreid
sys.modules["faiss"] = MagicMock()

import asyncio
import numpy as np
from app.core.config import settings
from app.services.detection_video_service import DetectionVideoService

async def test_gt_mode():
    print("--- Testing Ground Truth Mode ---")
    
    # 1. Enable GT Mode in Settings Mock (Temporarily override global settings if possible, or just set env vars logic is skipped for simplicity, we modify the object)
    settings.ENABLE_GROUND_TRUTH_MODE = True
    settings.GROUND_TRUTH_DATASET_PATH = "/Users/krittinsetdhavanich/Documents/spoton/spoton_backend/mock_gt_data"
    
    print(f"Settings Enabled: {settings.ENABLE_GROUND_TRUTH_MODE}")
    print(f"Dataset Path: {settings.GROUND_TRUTH_DATASET_PATH}")
    
    # 2. Initialize Service
    service = DetectionVideoService()
    await service.initialize_detection_services(environment_id="campus")
    
    if service.ground_truth_service:
        print("✅ GroundTruthService initialized successfully.")
    else:
        print("❌ GroundTruthService FAILED to initialize.")
        return

    # 3. Process a frame (mock frame)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    camera_id = "c01"
    frame_number = 1
    
    print(f"Processing frame {frame_number} for {camera_id}...")
    result = await service.process_frame_with_tracking(frame, camera_id, frame_number)
    
    # 4. Verify Output
    tracks = result.get("tracks", [])
    print(f"Tracks found: {len(tracks)}")
    
    if len(tracks) > 0:
        t = tracks[0]
        print(f"Track 0: ID={t.get('track_id')}, BBox={t.get('bbox_xyxy')}")
        if t.get('track_id') == 1 and t.get('bbox_xyxy') == [100.0, 100.0, 150.0, 200.0]:
             print("✅ Track data matches mock GT.")
        else:
             print("❌ Track data mismatch.")
    else:
        print("❌ No tracks returned (Expected 1).")

    print("--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_gt_mode())
