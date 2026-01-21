import sys
from unittest.mock import MagicMock
sys.modules["faiss"] = MagicMock()

import pytest
import numpy as np
from app.services.feature_extraction_service import FeatureExtractionService
from app.services.handoff_manager import HandoffManager
import time

# Mock FastReID for faster testing without GPU/Model loading
class MockEncoder:
    def inference(self, image, detections):
        # Return random embedding
        return np.random.rand(1, 2048)

@pytest.fixture
def mock_feature_service(monkeypatch):
    monkeypatch.setattr("app.services.feature_extraction_service.FastReIDInterface", 
                        lambda *args, **kwargs: MockEncoder())
    # Bypass file checks
    monkeypatch.setattr("os.path.exists", lambda path: True)
    return FeatureExtractionService(config_path="dummy", model_path="dummy", device="cpu")

def test_feature_extraction(mock_feature_service):
    patch = np.zeros((128, 64, 3), dtype=np.uint8)
    embedding = mock_feature_service.extract(patch)
    assert embedding is not None
    assert embedding.shape == (2048,)

def test_handoff_manager_match():
    manager = HandoffManager(match_threshold=0.5)
    
    # Register an entry
    emb1 = np.array([1.0, 0.0]) # Simplified embedding
    manager.register_exit(global_id="person_1", embedding=emb1, camera_id="cam_A")
    
    # Test perfect match
    gid, score = manager.find_match(emb1, "cam_B")
    assert gid == "person_1"
    assert score > 0.99
    
    # Test mismatch
    emb2 = np.array([0.0, 1.0])
    gid, score = manager.find_match(emb2, "cam_B")
    assert gid is None
    
def test_handoff_manager_expiry():
    manager = HandoffManager(time_window_sec=1)
    
    # Register
    emb = np.array([1.0])
    manager.register_exit("person_1", emb, "cam_A")
    
    # Wait
    time.sleep(1.1)
    
    # Must be gone
    gid, _ = manager.find_match(emb, "cam_B")
    assert gid is None
