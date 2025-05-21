"""
Unit tests for ReIDStateManager in app.services.reid_components.
"""
import pytest
import uuid
import numpy as np
from unittest.mock import MagicMock, PropertyMock, call

from app.services.reid_components import ReIDStateManager
from app.common_types import ( # MODIFIED: ExitDirection removed, QuadrantName added
    CameraID, TrackID, GlobalID, FeatureVector, TrackKey, QuadrantName,
    HandoffTriggerInfo, ExitRuleModel
)
# mock_settings is available from conftest.py

@pytest.fixture
def reid_manager_instance(mock_settings, mocker):
    """Provides an instance of ReIDStateManager with mocked settings."""
    mocker.patch("app.services.reid_components.settings", mock_settings)
    task_id = uuid.uuid4()
    return ReIDStateManager(task_id=task_id)

# --- Helper to create a FeatureVector ---
def FV(arr: list) -> FeatureVector:
    return FeatureVector(np.array(arr, dtype=np.float32))

def test_reid_manager_init(reid_manager_instance: ReIDStateManager, mock_settings):
    """Tests ReIDStateManager initialization."""
    assert reid_manager_instance.task_id is not None
    assert reid_manager_instance.reid_gallery == {}
    assert reid_manager_instance.lost_track_gallery == {}
    assert reid_manager_instance.track_to_global_id == {}

def test_get_new_global_id(reid_manager_instance: ReIDStateManager):
    """Tests generation of new GlobalIDs."""
    gid1 = reid_manager_instance.get_new_global_id()
    gid2 = reid_manager_instance.get_new_global_id()
    assert isinstance(gid1, str)
    assert isinstance(gid2, str)
    assert gid1 != gid2

def test_normalize_embedding(reid_manager_instance: ReIDStateManager):
    """Tests embedding normalization."""
    emb = FV([1.0, 2.0, 2.0]) 
    norm_emb = reid_manager_instance._normalize_embedding(emb)
    assert norm_emb is not None
    np.testing.assert_array_almost_equal(norm_emb, np.array([1/3, 2/3, 2/3], dtype=np.float32))
    assert np.isclose(np.linalg.norm(norm_emb), 1.0)

    zero_emb = FV([0.0, 0.0])
    norm_zero_emb = reid_manager_instance._normalize_embedding(zero_emb)
    assert norm_zero_emb is not None
    np.testing.assert_array_equal(norm_zero_emb, np.array([0.0, 0.0], dtype=np.float32))

    assert reid_manager_instance._normalize_embedding(None) is None
    assert np.array_equal(reid_manager_instance._normalize_embedding(FV([])), FV([]))


def test_calculate_similarity_matrix(reid_manager_instance: ReIDStateManager):
    """Tests similarity matrix calculation."""
    q_emb = np.array([[1,0], [0,1]], dtype=np.float32)
    g_emb = np.array([[1,0], [0,1], [0.707, 0.707]], dtype=np.float32) 
    
    sim_matrix = reid_manager_instance._calculate_similarity_matrix(q_emb, g_emb)
    assert sim_matrix is not None
    expected_sim = np.array([
        [1.0, 0.0, 0.707], 
        [0.0, 1.0, 0.707]  
    ])
    np.testing.assert_array_almost_equal(sim_matrix, expected_sim, decimal=3)

    assert reid_manager_instance._calculate_similarity_matrix(np.empty((0,2)), g_emb) is None
    assert reid_manager_instance._calculate_similarity_matrix(q_emb, np.empty((0,2))) is None


def test_should_attempt_reid_for_track(reid_manager_instance: ReIDStateManager, mock_settings):
    """Tests logic for deciding whether to attempt Re-ID."""
    tk1: TrackKey = (CameraID("c1"), TrackID(1))
    frame_count = 100
    mock_settings.REID_REFRESH_INTERVAL_FRAMES = 10

    # Test Case 1: New track
    assert reid_manager_instance._should_attempt_reid_for_track(tk1, frame_count, {}) is True

    # Test Case 2: Known track, due for refresh
    reid_manager_instance.track_to_global_id[tk1] = GlobalID("g1")
    reid_manager_instance.track_last_reid_frame[tk1] = frame_count - mock_settings.REID_REFRESH_INTERVAL_FRAMES
    assert reid_manager_instance._should_attempt_reid_for_track(tk1, frame_count, {}) is True

    # Test Case 3: Known track, not due for refresh
    reid_manager_instance.track_last_reid_frame[tk1] = frame_count - 1
    assert reid_manager_instance._should_attempt_reid_for_track(tk1, frame_count, {}) is False

    # Test Case 4: Known track, not due for refresh, BUT has an active handoff trigger
    # MODIFIED: Use source_exit_quadrant in ExitRuleModel
    rule = ExitRuleModel(source_exit_quadrant=QuadrantName("upper_left"), target_cam_id=CameraID("c2"), target_entry_area="test_entry")
    trigger_info = HandoffTriggerInfo(source_track_key=tk1, rule=rule, source_bbox=[0,0,1,1])
    assert reid_manager_instance._should_attempt_reid_for_track(tk1, frame_count, {tk1: trigger_info}) is True


@pytest.mark.asyncio
async def test_associate_features_basic_new_id(reid_manager_instance: ReIDStateManager, mock_settings):
    """Tests basic association: new track gets a new GlobalID."""
    mock_settings.REID_SIMILARITY_THRESHOLD = 0.7
    tk1: TrackKey = (CameraID("c1"), TrackID(10))
    feat1 = FV(np.random.rand(512).tolist())
    
    features_input = {tk1: feat1}
    active_tracks = {tk1}
    
    await reid_manager_instance.associate_features_and_update_state(features_input, active_tracks, {}, 0)
    
    assert tk1 in reid_manager_instance.track_to_global_id
    gid1 = reid_manager_instance.track_to_global_id[tk1]
    assert gid1 in reid_manager_instance.reid_gallery
    np.testing.assert_array_almost_equal(reid_manager_instance.reid_gallery[gid1], reid_manager_instance._normalize_embedding(feat1)) 


@pytest.mark.asyncio
async def test_associate_features_match_main_gallery(reid_manager_instance: ReIDStateManager, mock_settings):
    """Tests matching an existing track in the main gallery."""
    mock_settings.REID_SIMILARITY_THRESHOLD = 0.8
    mock_settings.REID_GALLERY_EMA_ALPHA = 0.5 

    gid_existing = reid_manager_instance.get_new_global_id()
    feat_gallery = FV(np.array([1.0] + [0.0]*511)) 
    reid_manager_instance.reid_gallery[gid_existing] = feat_gallery

    tk_new: TrackKey = (CameraID("c1"), TrackID(20))
    feat_new = FV(np.array([0.95] + [0.01]*2 + [0.0]*509)) 
    
    feat_new_norm = reid_manager_instance._normalize_embedding(feat_new)
    assert feat_new_norm is not None

    features_input = {tk_new: feat_new_norm}
    active_tracks = {tk_new}

    await reid_manager_instance.associate_features_and_update_state(features_input, active_tracks, {}, 1)

    assert reid_manager_instance.track_to_global_id.get(tk_new) == gid_existing
    
    expected_ema_feat = (mock_settings.REID_GALLERY_EMA_ALPHA * feat_gallery + \
                        (1.0 - mock_settings.REID_GALLERY_EMA_ALPHA) * feat_new_norm)
    expected_ema_feat_norm = reid_manager_instance._normalize_embedding(expected_ema_feat) 
    
    assert expected_ema_feat_norm is not None
    np.testing.assert_array_almost_equal(reid_manager_instance.reid_gallery[gid_existing], expected_ema_feat_norm, decimal=5)


@pytest.mark.asyncio
async def test_associate_features_match_lost_gallery(reid_manager_instance: ReIDStateManager, mock_settings):
    """Tests matching a track that was in the lost gallery."""
    mock_settings.REID_SIMILARITY_THRESHOLD = 0.8
    gid_lost = reid_manager_instance.get_new_global_id()
    feat_lost = FV(np.random.rand(512).tolist())
    norm_feat_lost = reid_manager_instance._normalize_embedding(feat_lost)
    assert norm_feat_lost is not None
    reid_manager_instance.lost_track_gallery[gid_lost] = (norm_feat_lost, 0) 

    tk_reappearing: TrackKey = (CameraID("c2"), TrackID(30))
    # Ensure the reappearing feature is similar enough to the lost one
    feat_reappearing_array = norm_feat_lost * 0.99 # Simulate slight change but still similar
    feat_reappearing_norm = FV(feat_reappearing_array.tolist())
    feat_reappearing_norm = reid_manager_instance._normalize_embedding(feat_reappearing_norm)
    assert feat_reappearing_norm is not None


    features_input = {tk_reappearing: feat_reappearing_norm}
    active_tracks = {tk_reappearing}

    await reid_manager_instance.associate_features_and_update_state(features_input, active_tracks, {}, 1)

    assert reid_manager_instance.track_to_global_id.get(tk_reappearing) == gid_lost
    assert gid_lost not in reid_manager_instance.lost_track_gallery 
    assert gid_lost in reid_manager_instance.reid_gallery


@pytest.mark.asyncio
async def test_associate_features_conflict_resolution(reid_manager_instance: ReIDStateManager, mock_settings):
    """Tests conflict resolution: two new tracks in same cam match same gallery GID."""
    mock_settings.REID_SIMILARITY_THRESHOLD = 0.8
    gid_target = reid_manager_instance.get_new_global_id()
    feat_target_gallery = FV(np.array([1.0] + [0.0]*511))
    reid_manager_instance.reid_gallery[gid_target] = feat_target_gallery

    cam_id = CameraID("c1")
    tk1: TrackKey = (cam_id, TrackID(1)) # Higher similarity to gid_target
    tk2: TrackKey = (cam_id, TrackID(2)) # Lower similarity to gid_target

    # feat1 will be more similar to feat_target_gallery
    feat1_array = np.array([0.95] + [0.01]*511) 
    feat1_norm = FV(feat1_array.tolist())
    feat1_norm = reid_manager_instance._normalize_embedding(feat1_norm)
    assert feat1_norm is not None

    # feat2 will be less similar to feat_target_gallery but still above threshold
    feat2_array = np.array([0.90] + [0.02]*511) 
    feat2_norm = FV(feat2_array.tolist())
    feat2_norm = reid_manager_instance._normalize_embedding(feat2_norm)
    assert feat2_norm is not None


    features_input = {tk1: feat1_norm, tk2: feat2_norm}
    active_tracks = {tk1, tk2}

    await reid_manager_instance.associate_features_and_update_state(features_input, active_tracks, {}, 0)

    # tk1 (higher sim) should get gid_target
    assert reid_manager_instance.track_to_global_id.get(tk1) == gid_target
    
    # tk2 (lower sim) should get a new GID because tk1 already took gid_target for this camera
    gid_tk2 = reid_manager_instance.track_to_global_id.get(tk2)
    assert gid_tk2 is not None
    assert gid_tk2 != gid_target # Must be a new GID
    assert gid_tk2 in reid_manager_instance.reid_gallery 
    np.testing.assert_array_almost_equal(reid_manager_instance.reid_gallery[gid_tk2], feat2_norm)


@pytest.mark.asyncio
async def test_update_galleries_lifecycle(reid_manager_instance: ReIDStateManager, mock_settings):
    """Tests the lifecycle updates: moving to lost, purging lost, pruning main."""
    mock_settings.REID_LOST_TRACK_BUFFER_FRAMES = 5
    mock_settings.REID_MAIN_GALLERY_PRUNE_INTERVAL_FRAMES = 10
    mock_settings.REID_MAIN_GALLERY_PRUNE_THRESHOLD_FRAMES = 7 

    tk1: TrackKey = (CameraID("c1"), TrackID(1))
    gid1 = GlobalID("g1_active")
    feat1 = FV(np.random.rand(512).tolist())
    reid_manager_instance.track_to_global_id[tk1] = gid1
    reid_manager_instance.reid_gallery[gid1] = feat1
    reid_manager_instance.global_id_last_seen_frame[gid1] = 0
    reid_manager_instance.global_id_last_seen_cam[gid1] = CameraID("c1")


    tk2: TrackKey = (CameraID("c1"), TrackID(2))
    gid2_will_disappear = GlobalID("g2_disappear")
    feat2 = FV(np.random.rand(512).tolist())
    reid_manager_instance.track_to_global_id[tk2] = gid2_will_disappear
    reid_manager_instance.reid_gallery[gid2_will_disappear] = feat2
    reid_manager_instance.global_id_last_seen_frame[gid2_will_disappear] = 0
    reid_manager_instance.global_id_last_seen_cam[gid2_will_disappear] = CameraID("c1")


    gid3_old_main = GlobalID("g3_old_main") 
    feat3 = FV(np.random.rand(512).tolist())
    reid_manager_instance.reid_gallery[gid3_old_main] = feat3
    reid_manager_instance.global_id_last_seen_frame[gid3_old_main] = 0 
    reid_manager_instance.global_id_last_seen_cam[gid3_old_main] = CameraID("c3")


    # Frame 1: tk2 disappears
    active_tracks_frame1 = {tk1} 
    await reid_manager_instance.update_galleries_lifecycle(active_tracks_frame1, 1)
    assert tk2 not in reid_manager_instance.track_to_global_id
    assert gid2_will_disappear in reid_manager_instance.lost_track_gallery
    # Check the frame number when it was added to lost gallery
    assert reid_manager_instance.lost_track_gallery[gid2_will_disappear][1] == 0 

    # Frame 6: tk2 (gid2_will_disappear) should be purged from lost gallery (0 + 5 < 6)
    await reid_manager_instance.update_galleries_lifecycle(active_tracks_frame1, 6)
    assert gid2_will_disappear not in reid_manager_instance.lost_track_gallery

    # Frame 10: Main gallery pruning. gid1 should remain, gid3_old_main should be pruned.
    # Update last_seen_frame for gid1 to be recent enough to survive pruning.
    reid_manager_instance.global_id_last_seen_frame[gid1] = 9 # Seen at frame 9, prune threshold is 7 frames old from frame 10 (i.e. < 10-7 = 3)

    await reid_manager_instance.update_galleries_lifecycle(active_tracks_frame1, 10)
    assert gid3_old_main not in reid_manager_instance.reid_gallery # Pruned
    assert gid1 in reid_manager_instance.reid_gallery # Should still be there