import sys
import types
import numpy as np


class _FakeIndexFlatIP:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = []

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add(self, vectors):
        for vec in vectors:
            self._vectors.append(np.array(vec, dtype=np.float32))

    def search(self, vectors, k: int):
        if self.ntotal == 0:
            return np.zeros((1, k), dtype=np.float32), -np.ones((1, k), dtype=np.int64)

        query = np.array(vectors[0], dtype=np.float32)
        scores = [float(np.dot(query, vec)) for vec in self._vectors]
        order = np.argsort(scores)[::-1]

        top_scores = [scores[i] for i in order[:k]]
        top_ids = [order[i] for i in range(min(k, len(order)))]

        # pad to length k
        while len(top_scores) < k:
            top_scores.append(0.0)
            top_ids.append(-1)

        return np.array([top_scores], dtype=np.float32), np.array([top_ids], dtype=np.int64)


def _normalize_L2(vectors):
    for row in vectors:
        norm = np.linalg.norm(row)
        if norm > 0:
            row /= norm


faiss_stub = types.SimpleNamespace(IndexFlatIP=_FakeIndexFlatIP, normalize_L2=_normalize_L2)
sys.modules["faiss"] = faiss_stub

import pytest
import time
from app.services import handoff_manager as handoff_module
from app.services.feature_extraction_service import FeatureExtractionService
from app.services.handoff_manager import HandoffManager


@pytest.fixture(autouse=True)
def patch_faiss(monkeypatch):
    monkeypatch.setattr(handoff_module, "faiss", faiss_stub)


class _MockTensor:
    """Mimics the chained .cpu().numpy().astype() API from torch tensors."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def astype(self, dtype):
        self._array = self._array.astype(dtype)
        return self


class _MockExtractor:
    def __call__(self, patches):
        embeddings = np.stack([
            np.random.rand(512).astype(np.float32) for _ in patches
        ])
        return _MockTensor(embeddings)


@pytest.fixture
def mock_feature_service(monkeypatch):
    def _fake_init(self):
        self.extractor = _MockExtractor()

    monkeypatch.setattr(FeatureExtractionService, "__init__", _fake_init)
    return FeatureExtractionService()

def test_feature_extraction(mock_feature_service):
    patch = np.zeros((128, 64, 3), dtype=np.uint8)
    embedding = mock_feature_service.extract(patch)
    assert embedding is not None
    assert embedding.shape == (512,)

def test_handoff_manager_match():
    manager = HandoffManager()
    
    emb1 = np.zeros(manager.feature_dim, dtype=np.float32)
    emb1[0] = 1.0
    manager.register_exit(global_id="person_1", embedding=emb1, camera_id="cam_A")
    
    gid, score = manager.find_match(emb1, "cam_B")
    assert gid == "person_1"
    assert score > 0.99
    
    emb2 = np.zeros(manager.feature_dim, dtype=np.float32)
    emb2[1] = 1.0
    gid, score = manager.find_match(emb2, "cam_B")
    assert gid is None

def test_handoff_manager_expiry():
    manager = HandoffManager()
    manager.time_window_sec = 1
    
    emb = np.zeros(manager.feature_dim, dtype=np.float32)
    emb[0] = 1.0
    manager.register_exit("person_1", emb, "cam_A")
    
    # Wait
    time.sleep(1.1)
    
    # Must be gone
    gid, _ = manager.find_match(emb, "cam_B")
    assert gid is None
