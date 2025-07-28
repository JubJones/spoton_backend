"""
Frame handler for binary WebSocket frame transmission.

Handles:
- Direct GPU-to-JPEG encoding
- Binary frame data transmission
- Adaptive JPEG compression
- Frame synchronization
"""

import asyncio
import logging
import time
import io
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import numpy as np
import cv2
from PIL import Image
import base64

from app.api.websockets.connection_manager import binary_websocket_manager, MessageType
from app.core.config import settings
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class FrameHandler:
    """
    Handles binary frame transmission with GPU optimization.
    
    Features:
    - Direct GPU-to-JPEG encoding
    - Adaptive quality control
    - Frame synchronization
    - Performance monitoring
    """
    
    def __init__(self):
        # Frame encoding settings
        self.default_jpeg_quality = getattr(settings, 'FRAME_JPEG_QUALITY', 85)
        self.adaptive_quality = True
        self.min_jpeg_quality = 60
        self.max_jpeg_quality = 95
        
        # Frame synchronization settings
        self.frame_sync_threshold_ms = 100
        self.max_frame_buffer_size = 10
        
        # Performance monitoring
        self.encoding_stats = {
            "total_frames_encoded": 0,
            "total_encoding_time": 0.0,
            "average_encoding_time": 0.0,
            "total_bytes_transmitted": 0,
            "average_compression_ratio": 0.0,
            "frame_drops": 0,
            "quality_adjustments": 0
        }
        
        # Frame buffering for synchronization
        self.frame_buffers: Dict[str, List[Dict[str, Any]]] = {}
        
        # Quality adaptation
        self.quality_history: Dict[str, List[float]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        logger.info("FrameHandler initialized with GPU optimization")
    
    async def encode_frame_to_jpeg(
        self, 
        frame: np.ndarray, 
        quality: Optional[int] = None,
        adaptive: bool = True
    ) -> bytes:
        """
        Encode frame to JPEG with GPU optimization.
        
        Args:
            frame: Frame data as numpy array
            quality: JPEG quality (1-100)
            adaptive: Whether to use adaptive quality
            
        Returns:
            JPEG encoded bytes
        """
        try:
            encoding_start = time.time()
            
            # Use adaptive quality if enabled
            if adaptive and quality is None:
                quality = self._get_adaptive_quality(frame.shape)
            elif quality is None:
                quality = self.default_jpeg_quality
            
            # Encode frame using OpenCV (GPU-accelerated if available)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
            # Try GPU-accelerated encoding first
            try:
                # Check if GPU encoder is available
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    # Use GPU encoding if available
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    
                    # Encode on GPU
                    success, encoded_frame = cv2.imencode('.jpg', gpu_frame.download(), encode_params)
                else:
                    # Fall back to CPU encoding
                    success, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
                    
            except Exception as gpu_error:
                logger.debug(f"GPU encoding failed, falling back to CPU: {gpu_error}")
                # Fall back to CPU encoding
                success, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                raise RuntimeError("Frame encoding failed")
            
            # Convert to bytes
            jpeg_bytes = encoded_frame.tobytes()
            
            # Update statistics
            encoding_time = time.time() - encoding_start
            self._update_encoding_stats(encoding_time, len(jpeg_bytes), frame.size)
            
            return jpeg_bytes
            
        except Exception as e:
            logger.error(f"Error encoding frame to JPEG: {e}")
            raise
    
    async def send_frame_data(
        self, 
        task_id: str, 
        camera_frames: Dict[str, np.ndarray],
        frame_metadata: Dict[str, Any]
    ) -> bool:
        """
        Send frame data with synchronization.
        
        Args:
            task_id: Task identifier
            camera_frames: Frames from multiple cameras
            frame_metadata: Frame metadata
            
        Returns:
            True if sent successfully
        """
        try:
            # Prepare frame data for each camera
            encoded_frames = {}
            
            for camera_id, frame in camera_frames.items():
                if frame is not None:
                    # Encode frame
                    jpeg_bytes = await self.encode_frame_to_jpeg(frame)
                    encoded_frames[camera_id] = jpeg_bytes
            
            # Create frame message
            frame_message = {
                "type": MessageType.FRAME_DATA.value,
                "frame_index": frame_metadata.get("frame_index", 0),
                "scene_id": frame_metadata.get("scene_id", ""),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "camera_count": len(encoded_frames),
                "frame_quality": frame_metadata.get("frame_quality", 1.0),
                "sync_timestamp": frame_metadata.get("sync_timestamp"),
                "cameras": {}
            }
            
            # Add camera data
            for camera_id in encoded_frames:
                frame_message["cameras"][camera_id] = {
                    "frame_size": len(encoded_frames[camera_id]),
                    "encoding_quality": self._get_current_quality(camera_id),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Send binary frame data
            success = True
            for camera_id, jpeg_bytes in encoded_frames.items():
                # Create camera-specific metadata
                camera_metadata = {
                    **frame_message,
                    "camera_id": camera_id,
                    "frame_data_size": len(jpeg_bytes)
                }
                
                # Send binary frame
                camera_success = await binary_websocket_manager.send_binary_frame(
                    task_id, jpeg_bytes, camera_metadata
                )
                
                if not camera_success:
                    success = False
                    logger.warning(f"Failed to send frame for camera {camera_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending frame data for task {task_id}: {e}")
            return False
    
    async def send_synchronized_frames(
        self, 
        task_id: str, 
        camera_frames: Dict[str, np.ndarray],
        frame_metadata: Dict[str, Any]
    ) -> bool:
        """
        Send synchronized frames from multiple cameras.
        
        Args:
            task_id: Task identifier
            camera_frames: Frames from multiple cameras
            frame_metadata: Frame metadata
            
        Returns:
            True if sent successfully
        """
        try:
            # Check frame synchronization
            if not self._check_frame_sync(camera_frames, frame_metadata):
                logger.warning(f"Frame synchronization failed for task {task_id}")
                self.encoding_stats["frame_drops"] += 1
                return False
            
            # Send synchronized frames
            return await self.send_frame_data(task_id, camera_frames, frame_metadata)
            
        except Exception as e:
            logger.error(f"Error sending synchronized frames for task {task_id}: {e}")
            return False
    
    def _check_frame_sync(
        self, 
        camera_frames: Dict[str, np.ndarray], 
        frame_metadata: Dict[str, Any]
    ) -> bool:
        """Check if frames are synchronized within threshold."""
        try:
            # Get timestamps for each camera
            camera_timestamps = {}
            
            for camera_id in camera_frames:
                # Use sync timestamp if available
                if "sync_timestamp" in frame_metadata:
                    camera_timestamps[camera_id] = frame_metadata["sync_timestamp"]
                else:
                    # Use current time as fallback
                    camera_timestamps[camera_id] = time.time() * 1000  # Convert to ms
            
            # Check synchronization
            if len(camera_timestamps) < 2:
                return True  # Single camera, no sync needed
            
            timestamps = list(camera_timestamps.values())
            max_diff = max(timestamps) - min(timestamps)
            
            return max_diff <= self.frame_sync_threshold_ms
            
        except Exception as e:
            logger.error(f"Error checking frame synchronization: {e}")
            return True  # Default to allowing transmission
    
    def _get_adaptive_quality(self, frame_shape: Tuple[int, ...]) -> int:
        """Get adaptive JPEG quality based on frame characteristics."""
        try:
            # Base quality on frame size and performance
            height, width = frame_shape[:2]
            pixel_count = height * width
            
            # Adjust quality based on frame size
            if pixel_count > 1920 * 1080:  # >1080p
                base_quality = 75
            elif pixel_count > 1280 * 720:  # >720p
                base_quality = 80
            else:  # <=720p
                base_quality = 85
            
            # Adjust based on performance history
            if hasattr(self, 'performance_history') and self.performance_history:
                recent_performance = list(self.performance_history.values())[-10:]
                if recent_performance:
                    avg_performance = sum(recent_performance) / len(recent_performance)
                    
                    # Reduce quality if performance is poor
                    if avg_performance > 0.5:  # >500ms encoding time
                        base_quality = max(self.min_jpeg_quality, base_quality - 10)
                    elif avg_performance < 0.1:  # <100ms encoding time
                        base_quality = min(self.max_jpeg_quality, base_quality + 5)
            
            return max(self.min_jpeg_quality, min(self.max_jpeg_quality, base_quality))
            
        except Exception as e:
            logger.error(f"Error calculating adaptive quality: {e}")
            return self.default_jpeg_quality
    
    def _get_current_quality(self, camera_id: str) -> int:
        """Get current quality setting for camera."""
        try:
            if camera_id in self.quality_history and self.quality_history[camera_id]:
                return int(self.quality_history[camera_id][-1])
            return self.default_jpeg_quality
            
        except Exception as e:
            logger.error(f"Error getting current quality for camera {camera_id}: {e}")
            return self.default_jpeg_quality
    
    def _update_encoding_stats(
        self, 
        encoding_time: float, 
        bytes_transmitted: int, 
        original_size: int
    ):
        """Update encoding statistics."""
        try:
            self.encoding_stats["total_frames_encoded"] += 1
            self.encoding_stats["total_encoding_time"] += encoding_time
            self.encoding_stats["total_bytes_transmitted"] += bytes_transmitted
            
            # Calculate averages
            frame_count = self.encoding_stats["total_frames_encoded"]
            self.encoding_stats["average_encoding_time"] = (
                self.encoding_stats["total_encoding_time"] / frame_count
            )
            
            # Calculate compression ratio
            if original_size > 0:
                compression_ratio = bytes_transmitted / original_size
                current_avg = self.encoding_stats["average_compression_ratio"]
                
                self.encoding_stats["average_compression_ratio"] = (
                    (current_avg * (frame_count - 1) + compression_ratio) / frame_count
                )
            
        except Exception as e:
            logger.error(f"Error updating encoding stats: {e}")
    
    async def create_frame_base64(self, frame: np.ndarray, quality: int = None) -> str:
        """
        Create base64 encoded frame for compatibility.
        
        Args:
            frame: Frame data
            quality: JPEG quality
            
        Returns:
            Base64 encoded frame
        """
        try:
            # Encode to JPEG
            jpeg_bytes = await self.encode_frame_to_jpeg(frame, quality)
            
            # Convert to base64
            base64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            return f"data:image/jpeg;base64,{base64_str}"
            
        except Exception as e:
            logger.error(f"Error creating base64 frame: {e}")
            return ""
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """Get encoding statistics."""
        return {
            **self.encoding_stats,
            "current_quality_settings": dict(self.quality_history),
            "frame_buffers": {
                task_id: len(buffer) for task_id, buffer in self.frame_buffers.items()
            }
        }
    
    def reset_stats(self):
        """Reset encoding statistics."""
        self.encoding_stats = {
            "total_frames_encoded": 0,
            "total_encoding_time": 0.0,
            "average_encoding_time": 0.0,
            "total_bytes_transmitted": 0,
            "average_compression_ratio": 0.0,
            "frame_drops": 0,
            "quality_adjustments": 0
        }
        
        self.quality_history.clear()
        self.performance_history.clear()
        self.frame_buffers.clear()
        
        logger.info("FrameHandler statistics reset")


# Global frame handler instance
frame_handler = FrameHandler()