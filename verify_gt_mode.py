
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set env vars BEFORE importing app modules to trigger GT mode
os.environ["USE_GROUND_TRUTH"] = "true"
os.environ["GROUND_TRUTH_DATA_DIR"] = "./mock_gt_data"

# Create mock data directory
Path("./mock_gt_data/c01/gt").mkdir(parents=True, exist_ok=True)
with open("./mock_gt_data/c01/gt/gt.txt", "w") as f:
    # Frame 1, ID 1, x, y, w, h, conf, class, vis
    f.write("1,1,100,100,50,100,0.9,1,1.0\n")
    f.write("1,2,200,200,50,100,0.85,1,1.0\n")

# Import app modules
from app.core.config import settings
from app.services.detection_video_service import DetectionVideoService

async def main():
    print(f"USE_GROUND_TRUTH: {settings.USE_GROUND_TRUTH}")
    print(f"GROUND_TRUTH_DATA_DIR: {settings.GROUND_TRUTH_DATA_DIR}")
    
    assert settings.USE_GROUND_TRUTH is True, "Setting not effective"

    service = DetectionVideoService()
    
    # Mock parent initialization to avoid S3 calls etc.
    service.initialize_services = MagicMock(return_value= asyncio.Future())
    service.initialize_services.return_value.set_result(True)
    
    # Initialize implementation
    print("Initializing detection services...")
    success = await service.initialize_detection_services(environment_id="campus")
    
    if not success:
        print("Failed to initialize detection service")
        sys.exit(1)
        
    if not service.ground_truth_service:
        print("GroundTruthService NOT initialized!")
        sys.exit(1)
        
    print("GroundTruthService initialized successfully.")
    
    # Test processing a frame
    import numpy as np
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    print("Processing frame 1...")
    result = await service.process_frame_with_tracking(dummy_frame, "c01", 1)
    
    tracks = result.get("tracks", [])
    print(f"Got {len(tracks)} tracks.")
    
    if len(tracks) != 2:
        print("Expected 2 tracks!")
        sys.exit(1)
        
    t1 = next(t for t in tracks if t['track_id'] == 1)
    print("Track 1 validation passed:", t1['bbox']['x1'] == 100.0)
    
    print("✅ Verification Successful!")
    
    # Cleanup
    import shutil
    shutil.rmtree("./mock_gt_data")

if __name__ == "__main__":
    asyncio.run(main())
