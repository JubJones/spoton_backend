"""
Trail Management Service for person movement history.

This service manages 3-frame movement trails for detected persons across multiple cameras.
It integrates with the detection endpoint to provide real-time trail data for the 2D mapping feature.

Features:
- 3-frame trail storage per person per camera
- Automatic cleanup of old trails
- Memory-efficient circular buffer implementation
- Thread-safe operations for concurrent camera processing
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class TrailPoint:
    """Single point in a person's movement trail."""
    map_x: float
    map_y: float
    timestamp: datetime
    frame_offset: int  # 0 = current, -1 = previous, -2 = two frames ago


class TrailManagementService:
    """
    Manages 3-frame movement trails for detected persons.
    
    This service stores the last 3 positions for each detected person per camera,
    providing trail data for the 2D mapping visualization. It's designed to integrate
    seamlessly with the DetectionVideoService.
    """
    
    def __init__(self, trail_length: int = 3):
        """
        Initialize the trail management service.
        
        Args:
            trail_length: Maximum number of trail points to store per person (default: 3)
        """
        self.trail_length = trail_length
        # Structure: {camera_id: {detection_id: [TrailPoint...]}}
        self.trails: Dict[str, Dict[str, List[TrailPoint]]] = {}
        self._lock = Lock()  # Thread safety for concurrent access
        
        logger.info(f"TrailManagementService initialized with trail_length={trail_length}")
    
    async def update_trail(
        self, 
        camera_id: str, 
        detection_id: str, 
        map_x: float, 
        map_y: float
    ) -> List[TrailPoint]:
        """
        Update trail for a detection and return current trail.
        
        Args:
            camera_id: Camera identifier (e.g., 'c09', 'c12', 'c13', 'c16')
            detection_id: Unique detection identifier
            map_x: Map coordinate X (in meters)
            map_y: Map coordinate Y (in meters)
            
        Returns:
            List of TrailPoint objects representing the current trail
        """
        with self._lock:
            # Initialize camera if not exists
            if camera_id not in self.trails:
                self.trails[camera_id] = {}
            
            # Initialize detection trail if not exists
            if detection_id not in self.trails[camera_id]:
                self.trails[camera_id][detection_id] = []
            
            trail = self.trails[camera_id][detection_id]
            
            # Shift existing points (increment frame_offset to represent older frames)
            for point in trail:
                point.frame_offset -= 1
            
            # Add new current point at the beginning
            new_point = TrailPoint(
                map_x=map_x,
                map_y=map_y,
                timestamp=datetime.now(timezone.utc),
                frame_offset=0  # Current frame
            )
            trail.insert(0, new_point)
            
            # Keep only the last N points (circular buffer behavior)
            if len(trail) > self.trail_length:
                trail = trail[:self.trail_length]
                self.trails[camera_id][detection_id] = trail
            
            logger.debug(f"Updated trail for {camera_id}:{detection_id} - {len(trail)} points")
            return trail.copy()  # Return copy to avoid external modification
    
    async def get_trail(self, camera_id: str, detection_id: str) -> List[TrailPoint]:
        """
        Get current trail for a detection.
        
        Args:
            camera_id: Camera identifier
            detection_id: Detection identifier
            
        Returns:
            List of TrailPoint objects (empty list if no trail exists)
        """
        with self._lock:
            trail = self.trails.get(camera_id, {}).get(detection_id, [])
            return trail.copy()  # Return copy to avoid external modification
    
    async def get_all_trails_for_camera(self, camera_id: str) -> Dict[str, List[TrailPoint]]:
        """
        Get all trails for a specific camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary mapping detection_id to trail points
        """
        with self._lock:
            camera_trails = self.trails.get(camera_id, {})
            return {det_id: trail.copy() for det_id, trail in camera_trails.items()}
    
    async def cleanup_old_trails(self, max_age_seconds: int = 30):
        """
        Remove trails older than max_age_seconds.
        
        This prevents memory leaks by removing trails for persons who are no longer
        being detected. Called periodically by background cleanup task.
        
        Args:
            max_age_seconds: Maximum age in seconds before trail is removed
        """
        current_time = datetime.now(timezone.utc)
        removed_count = 0
        
        with self._lock:
            for camera_id in list(self.trails.keys()):
                for detection_id in list(self.trails[camera_id].keys()):
                    trail = self.trails[camera_id][detection_id]
                    
                    # Check if trail exists and is old
                    if trail and (current_time - trail[0].timestamp).total_seconds() > max_age_seconds:
                        del self.trails[camera_id][detection_id]
                        removed_count += 1
                        logger.debug(f"Removed old trail for {camera_id}:{detection_id}")
                
                # Clean up empty camera entries
                if not self.trails[camera_id]:
                    del self.trails[camera_id]
        
        if removed_count > 0:
            logger.info(f"Cleanup completed: removed {removed_count} old trails")
    
    async def get_statistics(self) -> Dict[str, any]:
        """
        Get trail management statistics for monitoring and debugging.
        
        Returns:
            Dictionary containing statistics about current trails
        """
        with self._lock:
            stats = {
                "total_cameras": len(self.trails),
                "total_active_trails": sum(len(camera_trails) for camera_trails in self.trails.values()),
                "trails_per_camera": {
                    camera_id: len(camera_trails) 
                    for camera_id, camera_trails in self.trails.items()
                },
                "trail_length_setting": self.trail_length,
                "memory_usage_estimate": self._estimate_memory_usage()
            }
        
        return stats
    
    def _estimate_memory_usage(self) -> str:
        """
        Estimate memory usage of stored trails.
        
        Returns:
            Human-readable memory usage estimate
        """
        total_points = sum(
            len(trail) 
            for camera_trails in self.trails.values() 
            for trail in camera_trails.values()
        )
        
        # Rough estimate: each TrailPoint is about 100 bytes
        estimated_bytes = total_points * 100
        
        if estimated_bytes < 1024:
            return f"{estimated_bytes} bytes"
        elif estimated_bytes < 1024 * 1024:
            return f"{estimated_bytes // 1024} KB"
        else:
            return f"{estimated_bytes // (1024 * 1024)} MB"
    
    async def clear_all_trails(self):
        """
        Clear all trails. Used for testing and emergency cleanup.
        """
        with self._lock:
            self.trails.clear()
            logger.info("All trails cleared")
    
    async def remove_detection_trail(self, camera_id: str, detection_id: str) -> bool:
        """
        Remove a specific detection's trail.
        
        Args:
            camera_id: Camera identifier
            detection_id: Detection identifier
            
        Returns:
            True if trail was removed, False if it didn't exist
        """
        with self._lock:
            if camera_id in self.trails and detection_id in self.trails[camera_id]:
                del self.trails[camera_id][detection_id]
                logger.debug(f"Removed trail for {camera_id}:{detection_id}")
                
                # Clean up empty camera entry
                if not self.trails[camera_id]:
                    del self.trails[camera_id]
                
                return True
            
            return False