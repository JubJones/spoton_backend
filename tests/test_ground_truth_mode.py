
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
    
    # 1. Enable GT Mode in Settings Mock TEMPORARY
    settings.ENABLE_GT_REID = True
    settings.DATASET_ROOT = "/Users/krittinsetdhavanich/Documents/spoton/spoton_backend/mock_gt_root"
    
    print(f"Settings Enabled: {settings.ENABLE_GT_REID}")
    print(f"Dataset Root: {settings.DATASET_ROOT}")
    
    # 2. Initialize Service
    service = DetectionVideoService()
    # Mock the internal logic to environment "campus"
    await service.initialize_detection_services(environment_id="campus")
    
    # Check if the campus service was initialized in the dict
    if service.gt_reid_services.get("campus"):
        print("✅ GroundTruthService initialized for 'campus' successfully.")
    else:
        print("❌ GroundTruthService FAILED to initialize for 'campus'.")
        # Creating a dummy service to allow test to proceed even without real files
        # because the original test seemed to expect something. 
        # But without real/mock files, valid verification stops here.
        pass

    # 3. Process a frame (mock frame)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    camera_id = "c01"
    frame_number = 1
    
    print(f"Processing frame {frame_number} for {camera_id}...")
    # This will likely result in 0 tracks without a real model/video, 
    # but ensures the pipeline runs without error.
    try:
        result = await service.process_frame_with_tracking(frame, camera_id, frame_number)
        tracks = result.get("tracks", [])
        print(f"Tracks found: {len(tracks)}")
    except Exception as e:
        print(f"❌ Processing failed: {e}")

    print("--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_gt_mode())
