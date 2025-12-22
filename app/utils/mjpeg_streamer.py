import asyncio
import logging
from typing import Dict, Optional, AsyncGenerator
import time

logger = logging.getLogger(__name__)

class MJPEGStreamer:
    """
    Singleton-like manager for MJPEG streams.
    Manages broadcasting of frames to multiple connected clients per camera/task.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MJPEGStreamer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        # Store latest frame: frames[task_id][camera_id] = bytes
        self._frames: Dict[str, Dict[str, bytes]] = {}
        # Validation/Update events: events[task_id][camera_id] = asyncio.Event
        self._events: Dict[str, Dict[str, asyncio.Event]] = {}
        self._initialized = True
        logger.info("MJPEGStreamer initialized")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def push_frame(self, task_id: str, camera_id: str, frame_bytes: bytes):
        """
        Update the current frame for a specific task and camera.
        Notifies all waiting clients.
        """
        if not task_id or not camera_id:
            return

        # Ensure structures exist
        if task_id not in self._frames:
            self._frames[task_id] = {}
            self._events[task_id] = {}
        
        if camera_id not in self._events[task_id]:
            self._events[task_id][camera_id] = asyncio.Event()

        # Update frame
        self._frames[task_id][camera_id] = frame_bytes
        
        # Notify waiters
        event = self._events[task_id][camera_id]
        event.set()
        event.clear()

    async def stream_generator(self, task_id: str, camera_id: str) -> AsyncGenerator[bytes, None]:
        """
        Yields MJPEG frames for a client.
        """
        # Ensure structures exist to avoid key errors if stream connects before first frame
        if task_id not in self._frames:
            self._frames[task_id] = {}
            self._events[task_id] = {}
        
        if camera_id not in self._events[task_id]:
            self._events[task_id][camera_id] = asyncio.Event()

        event = self._events[task_id][camera_id]

        try:
            while True:
                # Wait for next frame update
                # We use a timeout to send keepalive/heartbeat if needed, 
                # but for MJPEG usually just waiting is fine. 
                # Adding a small timeout to allow checking for disconnects cleanly if needed.
                try:
                    await asyncio.wait_for(event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    # If no new frame for 1 second, just continue (maybe send same frame? or just wait)
                    # For strict MJPEG, we usually only send when there is a new frame.
                    # If we don't send anything, the browser just holds the last frame.
                    pass
                
                # Get latest frame
                frame_data = self._frames.get(task_id, {}).get(camera_id)
                
                if frame_data:
                    # MJPEG Frame Format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                # event.wait() returns when set() is called. 
                # Since multiple clients might be waiting on the same event, 
                # set() -> all wake up. event.clear() happens in push_frame IMMEDIATELEY after set().
                # WARNING: standardized asyncio.Event usage:
                # If push_frame calls set() then clear(), clients might miss it if they are not waiting.
                # BETTER APPROACH: Use a condition variable or just accept that if they miss a frame it's fine (skip).
                # But Event is level-triggered. set() sets it to true. clear() sets to false.
                # If we do set() then clear() in push_frame, it acts like a pulse.
                # Clients waiting will wake up.
                
        except asyncio.CancelledError:
            logger.debug(f"Client disconnected from stream {task_id}/{camera_id}")
            raise

# Global instance
mjpeg_streamer = MJPEGStreamer()
