"""
Handoff Detection Service for identifying camera transition zones.

This service detects when a person is in a handoff zone (boundary area between cameras)
and determines candidate cameras for cross-camera tracking transitions.

Features:
- Define camera handoff zones based on normalized image coordinates
- Detect when detections are in handoff areas
- Calculate bbox overlap ratios with handoff zones
- Identify candidate cameras for potential handoffs based on overlap rules
"""

import logging
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CameraZone:
    """
    Represents a handoff zone within a camera's field of view.
    
    Coordinates are normalized (0.0 to 1.0) relative to image dimensions.
    """
    camera_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    def __post_init__(self):
        """Validate zone coordinates."""
        if not (0.0 <= self.x_min <= self.x_max <= 1.0):
            raise ValueError(f"Invalid x coordinates: {self.x_min} - {self.x_max}")
        if not (0.0 <= self.y_min <= self.y_max <= 1.0):
            raise ValueError(f"Invalid y coordinates: {self.y_min} - {self.y_max}")
    
    def contains_point(self, norm_x: float, norm_y: float) -> bool:
        """Check if a normalized point is within this zone."""
        return (self.x_min <= norm_x <= self.x_max and 
                self.y_min <= norm_y <= self.y_max)
    
    def calculate_overlap_ratio(self, bbox_x: float, bbox_y: float, 
                              bbox_width: float, bbox_height: float) -> float:
        """
        Calculate overlap ratio between a bounding box and this zone.
        
        Args:
            bbox_x, bbox_y: Center coordinates of bbox (normalized)
            bbox_width, bbox_height: Dimensions of bbox (normalized)
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        # Calculate bbox boundaries
        bbox_x_min = bbox_x - bbox_width / 2
        bbox_x_max = bbox_x + bbox_width / 2
        bbox_y_min = bbox_y - bbox_height / 2
        bbox_y_max = bbox_y + bbox_height / 2
        
        # Calculate intersection
        intersect_x_min = max(bbox_x_min, self.x_min)
        intersect_x_max = min(bbox_x_max, self.x_max)
        intersect_y_min = max(bbox_y_min, self.y_min)
        intersect_y_max = min(bbox_y_max, self.y_max)
        
        # Check if there's any intersection
        if intersect_x_min >= intersect_x_max or intersect_y_min >= intersect_y_max:
            return 0.0
        
        # Calculate areas
        intersect_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
        bbox_area = bbox_width * bbox_height
        
        # Return overlap ratio
        return intersect_area / bbox_area if bbox_area > 0 else 0.0


class HandoffDetectionService:
    """
    Service for detecting camera handoff situations and identifying candidate cameras.
    
    Uses predefined camera zones and overlap rules to determine when a person
    is transitioning between camera views and which cameras should be considered
    for cross-camera tracking.
    """
    
    def __init__(self):
        """Initialize HandoffDetectionService with configuration."""
        self.camera_zones = self._define_camera_zones()
        self.handoff_rules = getattr(settings, 'POSSIBLE_CAMERA_OVERLAPS', [])
        self.overlap_threshold = getattr(settings, 'MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT', 0.5)
        
        logger.info(f"HandoffDetectionService initialized with {len(self.camera_zones)} camera zones")
        logger.debug(f"Handoff rules: {self.handoff_rules}")
        logger.debug(f"Overlap threshold: {self.overlap_threshold}")
    
    def _define_camera_zones(self) -> Dict[str, List[CameraZone]]:
        """
        Define handoff zones for each camera based on typical campus layout.
        
        These zones represent areas where people are likely to transition
        between camera views. Coordinates are normalized (0.0 to 1.0).
        
        Returns:
            Dictionary mapping camera_id to list of handoff zones
        """
        try:
            configured_zones = getattr(settings, "CAMERA_HANDOFF_ZONES", {})
            zones: Dict[str, List[CameraZone]] = {}
            total_zones = 0

            for camera_id, zone_configs in configured_zones.items():
                camera_zone_defs: List[CameraZone] = []
                for zone_cfg in zone_configs:
                    try:
                        camera_zone = CameraZone(
                            camera_id=camera_id,
                            x_min=float(zone_cfg.get("x_min", 0.0)),
                            x_max=float(zone_cfg.get("x_max", 1.0)),
                            y_min=float(zone_cfg.get("y_min", 0.0)),
                            y_max=float(zone_cfg.get("y_max", 1.0))
                        )
                        camera_zone_defs.append(camera_zone)
                        total_zones += 1
                        logger.debug(
                            f"Defined zone for {camera_id}: "
                            f"x={camera_zone.x_min:.2f}-{camera_zone.x_max:.2f}, "
                            f"y={camera_zone.y_min:.2f}-{camera_zone.y_max:.2f}"
                        )
                    except Exception as zone_error:
                        logger.warning(
                            f"Skipping invalid handoff zone config for camera {camera_id}: {zone_error}"
                        )

                if camera_zone_defs:
                    zones[camera_id] = camera_zone_defs
                else:
                    # Generate default matching zones (4 sides) if none configured
                    # This ensures "Boundary Triggers" work out-of-the-box
                    default_margin = 0.1 # 10%
                    defaults = [
                        CameraZone(camera_id, 0.0, default_margin, 0.0, 1.0), # Left
                        CameraZone(camera_id, 1.0 - default_margin, 1.0, 0.0, 1.0), # Right
                        CameraZone(camera_id, 0.0, 1.0, 0.0, default_margin), # Top
                        CameraZone(camera_id, 0.0, 1.0, 1.0 - default_margin, 1.0) # Bottom
                    ]
                    zones[camera_id] = defaults
                    total_zones += 4
                    logger.info(f"Generated 4 default handoff zones for camera {camera_id}")

            logger.info(f"Defined {total_zones} handoff zones across {len(zones)} cameras")
            return zones

        except Exception as e:
            logger.error(f"Error defining camera zones: {e}")
            return {}
    
    def check_handoff_trigger(self, camera_id: str, bbox: Dict, 
                             frame_width: int, frame_height: int) -> Tuple[bool, List[str]]:
        """
        Check if detection is in a handoff zone and identify candidate cameras.
        
        Args:
            camera_id: Current camera identifier
            bbox: Bounding box dictionary with center_x, center_y, width, height
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Tuple of (is_handoff_trigger, candidate_cameras)
        """
        try:
            # Normalize bounding box coordinates
            norm_center_x = bbox["center_x"] / frame_width
            norm_center_y = bbox["center_y"] / frame_height
            norm_width = bbox["width"] / frame_width
            norm_height = bbox["height"] / frame_height
            
            # Validate normalized coordinates
            if not (0.0 <= norm_center_x <= 1.0 and 0.0 <= norm_center_y <= 1.0):
                logger.warning(f"Invalid normalized coordinates for {camera_id}: "
                             f"center=({norm_center_x:.3f}, {norm_center_y:.3f})")
                return False, []
            
            candidate_cameras = []
            is_handoff = False
            max_overlap_ratio = 0.0
            
            # Check if camera has defined zones
            if camera_id not in self.camera_zones:
                # Lazy-load default zones (4 sides) if not configured
                logger.info(f"Initializing default handoff zones (4 edges) for camera {camera_id}")
                default_margin = 0.1 # 10%
                defaults = [
                    CameraZone(camera_id, 0.0, default_margin, 0.0, 1.0), # Left
                    CameraZone(camera_id, 1.0 - default_margin, 1.0, 0.0, 1.0), # Right
                    CameraZone(camera_id, 0.0, 1.0, 0.0, default_margin), # Top
                    CameraZone(camera_id, 0.0, 1.0, 1.0 - default_margin, 1.0) # Bottom
                ]
                self.camera_zones[camera_id] = defaults
            
            # Check each zone for this camera
            for zone in self.camera_zones[camera_id]:
                # Check if bbox center is in zone
                if zone.contains_point(norm_center_x, norm_center_y):
                    # Calculate overlap ratio between bbox and zone
                    overlap_ratio = zone.calculate_overlap_ratio(
                        norm_center_x, norm_center_y, norm_width, norm_height
                    )
                    
                    max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
                    
                    # Check if overlap exceeds threshold
                    if overlap_ratio >= self.overlap_threshold:
                        is_handoff = True
                        
                        # Find candidate cameras based on handoff rules
                        for rule in self.handoff_rules:
                            if camera_id in rule:
                                other_camera = rule[1] if rule[0] == camera_id else rule[0]
                                if other_camera not in candidate_cameras:
                                    candidate_cameras.append(other_camera)
                        
                        logger.debug(f"Handoff triggered for {camera_id}: "
                                   f"overlap={overlap_ratio:.3f}, candidates={candidate_cameras}")
                        break
            
            # Log details for debugging
            if max_overlap_ratio > 0:
                logger.debug(f"Handoff check for {camera_id}: "
                           f"max_overlap={max_overlap_ratio:.3f}, "
                           f"threshold={self.overlap_threshold:.3f}, "
                           f"triggered={is_handoff}")
            
            return is_handoff, candidate_cameras
            
        except KeyError as e:
            logger.error(f"Missing bbox field for handoff detection: {e}")
            return False, []
        except Exception as e:
            logger.error(f"Error in handoff detection for {camera_id}: {e}")
            return False, []
    
    def get_zone_info(self, camera_id: str) -> Dict:
        """
        Get handoff zone information for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary with zone information for debugging/monitoring
        """
        info = {
            "camera_id": camera_id,
            "zones_defined": camera_id in self.camera_zones,
            "zone_count": len(self.camera_zones.get(camera_id, [])),
            "zones": [],
            "overlap_threshold": self.overlap_threshold
        }
        
        if camera_id in self.camera_zones:
            for i, zone in enumerate(self.camera_zones[camera_id]):
                zone_info = {
                    "zone_id": i,
                    "x_range": [zone.x_min, zone.x_max],
                    "y_range": [zone.y_min, zone.y_max],
                    "area": (zone.x_max - zone.x_min) * (zone.y_max - zone.y_min)
                }
                info["zones"].append(zone_info)
        
        return info
    
    def get_handoff_rules(self) -> List[Tuple[str, str]]:
        """
        Get configured handoff rules.
        
        Returns:
            List of camera pairs that can hand off to each other
        """
        return self.handoff_rules.copy()
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate handoff detection configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "zones_defined": len(self.camera_zones) > 0,
            "rules_defined": len(self.handoff_rules) > 0,
            "threshold_valid": 0.0 < self.overlap_threshold <= 1.0,
            "zone_coordinates_valid": True,
            "rule_cameras_have_zones": True
        }
        
        # Validate zone coordinates
        try:
            for camera_id, zones in self.camera_zones.items():
                for zone in zones:
                    # Validation happens in CameraZone.__post_init__
                    pass
        except Exception as e:
            logger.error(f"Invalid zone coordinates: {e}")
            validation["zone_coordinates_valid"] = False
        
        # Check if cameras in handoff rules have defined zones
        rule_cameras = set()
        for rule in self.handoff_rules:
            rule_cameras.update(rule)
        
        for camera_id in rule_cameras:
            if camera_id not in self.camera_zones:
                logger.warning(f"Camera {camera_id} in handoff rules but no zones defined")
                validation["rule_cameras_have_zones"] = False
        
        return validation
