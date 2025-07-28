"""
Multi-camera processing coordination.

Manages:
- Camera synchronization
- Frame batching across cameras
- Camera-specific processing parameters
- Camera health monitoring
"""

import asyncio
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages multi-camera processing coordination."""
    
    def __init__(self):
        self.active_cameras: Dict[CameraID, Dict[str, Any]] = {}
        self.camera_health: Dict[CameraID, bool] = {}
        self.frame_sync_buffer: Dict[CameraID, List[Any]] = {}
        logger.info("CameraManager initialized")
    
    def register_camera(self, camera_id: CameraID, config: Dict[str, Any]):
        """Register a camera for processing."""
        self.active_cameras[camera_id] = {
            "config": config,
            "status": "ACTIVE",
            "last_frame_time": None,
            "frame_count": 0,
            "error_count": 0,
            "registered_at": datetime.now(timezone.utc)
        }
        
        self.camera_health[camera_id] = True
        self.frame_sync_buffer[camera_id] = []
        
        logger.info(f"Camera {camera_id} registered for processing")
    
    def unregister_camera(self, camera_id: CameraID):
        """Unregister a camera from processing."""
        if camera_id in self.active_cameras:
            del self.active_cameras[camera_id]
            del self.camera_health[camera_id]
            del self.frame_sync_buffer[camera_id]
            logger.info(f"Camera {camera_id} unregistered")
    
    async def synchronize_frames(
        self, 
        frame_data: Dict[CameraID, Any], 
        sync_threshold_ms: int = 100
    ) -> Optional[Dict[CameraID, Any]]:
        """
        Synchronize frames from multiple cameras.
        
        Returns synchronized frame batch or None if synchronization fails.
        """
        if not frame_data:
            return None
        
        # Add frames to sync buffer
        for camera_id, frame in frame_data.items():
            if camera_id in self.frame_sync_buffer:
                self.frame_sync_buffer[camera_id].append(frame)
        
        # Check if we have frames from all active cameras
        active_camera_ids = set(self.active_cameras.keys())
        available_camera_ids = set(camera_id for camera_id, buffer in self.frame_sync_buffer.items() if buffer)
        
        if not active_camera_ids.issubset(available_camera_ids):
            logger.debug("Waiting for frames from all cameras")
            return None
        
        # Find synchronized frame set
        synchronized_frames = {}
        reference_timestamp = None
        
        for camera_id in active_camera_ids:
            frames = self.frame_sync_buffer[camera_id]
            if not frames:
                continue
            
            # Use first camera as reference
            if reference_timestamp is None:
                reference_timestamp = frames[0].get("timestamp")
                synchronized_frames[camera_id] = frames.pop(0)
            else:
                # Find closest frame to reference timestamp
                best_frame = None
                best_diff = float('inf')
                best_index = -1
                
                for i, frame in enumerate(frames):
                    timestamp_diff = abs(frame.get("timestamp", 0) - reference_timestamp)
                    if timestamp_diff < best_diff and timestamp_diff <= sync_threshold_ms:
                        best_frame = frame
                        best_diff = timestamp_diff
                        best_index = i
                
                if best_frame:
                    synchronized_frames[camera_id] = frames.pop(best_index)
                else:
                    logger.warning(f"No synchronized frame found for camera {camera_id}")
                    return None
        
        # Update camera statistics
        for camera_id in synchronized_frames:
            if camera_id in self.active_cameras:
                self.active_cameras[camera_id]["frame_count"] += 1
                self.active_cameras[camera_id]["last_frame_time"] = datetime.now(timezone.utc)
        
        logger.debug(f"Synchronized frames from {len(synchronized_frames)} cameras")
        return synchronized_frames if len(synchronized_frames) > 0 else None
    
    def update_camera_health(self, camera_id: CameraID, is_healthy: bool):
        """Update camera health status."""
        if camera_id in self.camera_health:
            self.camera_health[camera_id] = is_healthy
            
            if camera_id in self.active_cameras:
                self.active_cameras[camera_id]["status"] = "ACTIVE" if is_healthy else "ERROR"
                if not is_healthy:
                    self.active_cameras[camera_id]["error_count"] += 1
            
            logger.info(f"Camera {camera_id} health updated: {'healthy' if is_healthy else 'unhealthy'}")
    
    def get_camera_status(self, camera_id: CameraID) -> Optional[Dict[str, Any]]:
        """Get status information for a camera."""
        return self.active_cameras.get(camera_id)
    
    def get_all_camera_status(self) -> Dict[CameraID, Dict[str, Any]]:
        """Get status information for all cameras."""
        return self.active_cameras.copy()
    
    def get_healthy_cameras(self) -> List[CameraID]:
        """Get list of healthy camera IDs."""
        return [camera_id for camera_id, is_healthy in self.camera_health.items() if is_healthy]
    
    def get_camera_count(self) -> int:
        """Get total number of registered cameras."""
        return len(self.active_cameras)
    
    def get_healthy_camera_count(self) -> int:
        """Get number of healthy cameras."""
        return len(self.get_healthy_cameras())
    
    def clear_sync_buffers(self):
        """Clear all frame synchronization buffers."""
        for camera_id in self.frame_sync_buffer:
            self.frame_sync_buffer[camera_id].clear()
        logger.debug("Frame sync buffers cleared")
    
    async def process_camera_batch(
        self, 
        camera_frames: Dict[CameraID, Any]
    ) -> Dict[str, Any]:
        """
        Process a batch of frames from multiple cameras.
        
        Coordinates frame processing across all cameras.
        """
        try:
            # Synchronize frames
            synchronized_frames = await self.synchronize_frames(camera_frames)
            
            if not synchronized_frames:
                logger.warning("Frame synchronization failed")
                return {"error": "Frame synchronization failed"}
            
            # Process synchronized frames
            processing_results = {}
            
            for camera_id, frame in synchronized_frames.items():
                # Process individual camera frame
                result = await self._process_single_camera_frame(camera_id, frame)
                processing_results[camera_id] = result
            
            return {
                "camera_results": processing_results,
                "frame_count": len(synchronized_frames),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing camera batch: {e}")
            return {"error": str(e)}
    
    async def _process_single_camera_frame(
        self, 
        camera_id: CameraID, 
        frame: Any
    ) -> Dict[str, Any]:
        """Process a single camera frame."""
        try:
            # Placeholder for camera-specific processing
            await asyncio.sleep(0.001)  # Simulate processing time
            
            return {
                "camera_id": camera_id,
                "frame_processed": True,
                "processing_time": 0.001,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            self.update_camera_health(camera_id, False)
            return {
                "camera_id": camera_id,
                "frame_processed": False,
                "error": str(e)
            }

# Global camera manager instance
camera_manager = CameraManager()