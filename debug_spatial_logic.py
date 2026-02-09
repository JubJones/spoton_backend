import logging
import sys
import unittest
from unittest.mock import MagicMock
from app.services.space_based_matcher import SpaceBasedMatcher
from app.services.global_person_registry import GlobalPersonRegistry
from app.core.config import settings

# Configure logging to stdout
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("app.services.space_based_matcher")
logger.setLevel(logging.DEBUG)

class TestSpatialMatching(unittest.TestCase):
    def setUp(self):
        # Force factory settings
        settings.SPATIAL_MATCH_THRESHOLD_FACTORY = 2000.0
        settings.SPATIAL_NO_MATCH_DISTANCE_FACTORY = 4000.0
        settings.SPATIAL_EDGE_MARGIN_FACTORY = 0.0
        settings.SPATIAL_MATCH_ENABLED = True
        
        self.registry = GlobalPersonRegistry()
        self.matcher = SpaceBasedMatcher(registry=self.registry)
        self.matcher.set_environment("factory")
        
        # Mock homography service or just assume pre-projected points
        # The matcher expects "map_coords" in the track dict.

    def test_basic_match(self):
        print("\n--- Test 1: Basic Matching (New Tracks) ---")
        # c09 detection (Edge)
        t1 = {
            "track_id": 101,
            "map_coords": {"map_x": 1389.14, "map_y": 297.74}, # From debug_projection_edge.py
            "bbox": {"center_x": 1850, "center_y": 450}, # Edge
            "confidence": 0.9
        }
        # c16 detection (Right side)
        t2 = {
            "track_id": 202,
            "map_coords": {"map_x": 1584.98, "map_y": 512.68}, # From debug_projection_edge.py
            "bbox": {"center_x": 1600, "center_y": 350},
            "confidence": 0.85
        }
        
        detections = {
            "c09": {"tracks": [t1], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}},
            "c16": {"tracks": [t2], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}}
        }
        
        self.matcher.match_across_cameras(detections)
        
        gid1 = t1.get("global_id")
        gid2 = t2.get("global_id")
        print(f"c09_ID: {gid1}, c16_ID: {gid2}")
        
        self.assertIsNotNone(gid1)
        self.assertIsNotNone(gid2)
        self.assertEqual(gid1, gid2)
        
    def test_merge_existing_ids(self):
        print("\n--- Test 2: Merging Existing IDs (P41 vs P43) ---")
        # Manually assign different IDs first
        self.registry.assign_identity("c09", 101, "P41")
        self.registry.assign_identity("c16", 202, "P43")
        
        t1 = {
            "track_id": 101,
            "global_id": "P41",
            "map_coords": {"map_x": 1389.14, "map_y": 297.74},
            "bbox": {"center_x": 1850, "center_y": 450},
            "confidence": 0.9
        }
        t2 = {
            "track_id": 202,
            "global_id": "P43",
            "map_coords": {"map_x": 1584.98, "map_y": 512.68},
            "bbox": {"center_x": 1600, "center_y": 350},
            "confidence": 0.85
        }
        
        detections = {
            "c09": {"tracks": [t1], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}},
            "c16": {"tracks": [t2], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}}
        }
        
        self.matcher.match_across_cameras(detections)
        
        gid1 = self.registry.get_global_id("c09", 101)
        gid2 = self.registry.get_global_id("c16", 202)
        print(f"c09_ID: {gid1}, c16_ID: {gid2}")
        
        self.assertEqual(gid1, gid2)

    def test_temporal_desync(self):
        print("\n--- Test 4: Temporal Desync (Different Frames) ---")
        # c09 detection at frame 100
        t1 = {
            "track_id": 104,
            "bbox_xyxy": [1800, 400, 1900, 500],
            "map_coords": {"map_x": 1389.14, "map_y": 297.74},
            "bbox": {"center_x": 1850, "center_y": 450},
            "confidence": 0.9
        }
        # c16 detection at frame 105 (5 frames later)
        t2 = {
            "track_id": 204,
            "bbox_xyxy": [1550, 300, 1650, 400],
            "map_coords": {"map_x": 1584.98, "map_y": 512.68},
            "bbox": {"center_x": 1600, "center_y": 350},
            "confidence": 0.85
        }
        
        # Simulating how detection_video_service groups frames
        # It calls match_across_cameras with a map of {camera_id: detection_result}
        
        # Frame 100: Only c09 has a detection (c16 frame is from time T, but maybe empty or different)
        # Let's say c16 had NO detection at this exact moment in the batch
        detections_t1 = {
            "c09": {"tracks": [t1], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}},
            "c16": {"tracks": [], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}}
        }
        self.matcher.match_across_cameras(detections_t1)
        
        # Frame 105: Now c16 has detection, but c09 might have lost it or it moved
        detections_t2 = {
            "c09": {"tracks": [], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}},
            "c16": {"tracks": [t2], "spatial_metadata": {"frame_dimensions": {"width": 1920, "height": 1080}}}
        }
        self.matcher.match_across_cameras(detections_t2)
        
        # EXPECTATION (FIXED):
        # Now that early return is gone, single-camera tracks get assigned new Global IDs.
        # They will be different (because no spatial overlap in time), but they will exist.
        
        gid1 = t1.get("global_id")
        gid2 = t2.get("global_id")
        print(f"Desync_ID_1: {gid1}, Desync_ID_2: {gid2}")
        
        self.assertIsNotNone(gid1)
        self.assertIsNotNone(gid2)
        # self.assertNotEqual(gid1, gid2) # They are different because they never met in time/space buffer

if __name__ == '__main__':
    unittest.main()
