import asyncio
import logging
import uuid
import time
from typing import Dict, List, Optional, AsyncGenerator
from collections import defaultdict

logger = logging.getLogger(__name__)


class MJPEGStreamer:
    """
    Singleton manager for MJPEG streams with multi-subscriber support.
    
    Uses a pub/sub pattern where each connected client gets their own queue.
    This allows multiple simultaneous connections per camera without interference.
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
        
        # Subscriber queues: {task_id: {camera_id: {subscriber_id: asyncio.Queue}}}
        self._subscribers: Dict[str, Dict[str, Dict[str, asyncio.Queue]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        
        # Latest frame cache for new subscribers to get immediate frame
        self._latest_frames: Dict[str, Dict[str, bytes]] = defaultdict(dict)
        
        # Lock for thread-safe subscriber management
        self._lock = asyncio.Lock()
        
        # FPS tracking: {task_id: {camera_id: {"frame_count": int, "start_time": float}}}
        self._fps_counters: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"frame_count": 0, "start_time": time.time(), "last_log_time": time.time()})
        )
        
        self._initialized = True
        logger.info("MJPEGStreamer initialized with multi-subscriber support")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def subscribe(self, task_id: str, camera_id: str) -> tuple[str, asyncio.Queue]:
        """
        Create a new subscriber queue for this camera.
        
        Returns:
            Tuple of (subscriber_id, queue) for this subscription
        """
        subscriber_id = str(uuid.uuid4())[:8]
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)  # Small buffer for backpressure
        
        async with self._lock:
            self._subscribers[task_id][camera_id][subscriber_id] = queue
            subscriber_count = len(self._subscribers[task_id][camera_id])
        
        logger.debug(f"MJPEG subscriber {subscriber_id} connected to {task_id}/{camera_id} (total: {subscriber_count})")
        
        # Send latest frame immediately if available (new subscriber gets instant video)
        latest = self._latest_frames.get(task_id, {}).get(camera_id)
        if latest:
            try:
                queue.put_nowait(latest)
            except asyncio.QueueFull:
                pass
        
        return subscriber_id, queue

    async def unsubscribe(self, task_id: str, camera_id: str, subscriber_id: str):
        """
        Remove subscriber when connection closes.
        """
        async with self._lock:
            if subscriber_id in self._subscribers.get(task_id, {}).get(camera_id, {}):
                del self._subscribers[task_id][camera_id][subscriber_id]
                subscriber_count = len(self._subscribers[task_id][camera_id])
                
                # Cleanup empty structures
                if not self._subscribers[task_id][camera_id]:
                    del self._subscribers[task_id][camera_id]
                if not self._subscribers[task_id]:
                    del self._subscribers[task_id]
                
                logger.debug(f"MJPEG subscriber {subscriber_id} disconnected from {task_id}/{camera_id} (remaining: {subscriber_count})")

    async def push_frame(self, task_id: str, camera_id: str, frame_bytes: bytes):
        """
        Broadcast frame to ALL subscribers for this camera.
        
        Non-blocking: drops frames for slow consumers instead of blocking.
        """
        if not task_id or not camera_id or not frame_bytes:
            return
        
        # FPS tracking
        fps_data = self._fps_counters[task_id][camera_id]
        fps_data["frame_count"] += 1
        current_time = time.time()
        
        # Log FPS every 5 seconds
        if current_time - fps_data["last_log_time"] >= 5.0:
            elapsed = current_time - fps_data["start_time"]
            if elapsed > 0:
                fps = fps_data["frame_count"] / elapsed
                logger.info(f"[FPS_DEBUG] Task={task_id[:8]} Camera={camera_id} FPS={fps:.1f} (frames={fps_data['frame_count']} elapsed={elapsed:.1f}s)")
            fps_data["last_log_time"] = current_time
        
        # Cache latest frame for new subscribers
        self._latest_frames[task_id][camera_id] = frame_bytes
        
        # Get current subscribers (snapshot to avoid lock during iteration)
        subscribers = list(self._subscribers.get(task_id, {}).get(camera_id, {}).values())
        
        if not subscribers:
            return
        
        # Broadcast to all subscribers
        for queue in subscribers:
            try:
                # Non-blocking put - drop frame if queue full (backpressure)
                queue.put_nowait(frame_bytes)
            except asyncio.QueueFull:
                # Clear queue and put new frame (prefer latest frame over old)
                try:
                    while not queue.empty():
                        queue.get_nowait()
                    queue.put_nowait(frame_bytes)
                except:
                    pass  # Ignore race conditions


    async def stream_generator(self, task_id: str, camera_id: str) -> AsyncGenerator[bytes, None]:
        """
        Yields MJPEG frames for a connected client.
        
        Each client gets their own queue-based subscription.
        """
        subscriber_id, queue = await self.subscribe(task_id, camera_id)
        
        try:
            while True:
                try:
                    # Wait for next frame with timeout
                    frame_data = await asyncio.wait_for(queue.get(), timeout=5.0)
                    
                    # MJPEG Frame Format
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n'
                    )
                except asyncio.TimeoutError:
                    # No frame for 5 seconds - send empty to keep connection alive
                    # Browser will just hold the last frame
                    continue
                    
        except asyncio.CancelledError:
            logger.debug(f"MJPEG client {subscriber_id} cancelled for {task_id}/{camera_id}")
            raise
        except GeneratorExit:
            pass
        finally:
            await self.unsubscribe(task_id, camera_id, subscriber_id)

    def get_subscriber_count(self, task_id: str, camera_id: str) -> int:
        """Get current number of subscribers for a camera."""
        return len(self._subscribers.get(task_id, {}).get(camera_id, {}))

    def get_stats(self) -> dict:
        """Get streaming statistics."""
        total_subscribers = sum(
            len(cameras)
            for task in self._subscribers.values()
            for cameras in task.values()
        )
        return {
            "total_active_subscribers": total_subscribers,
            "tasks_streaming": len(self._subscribers),
        }


# Global instance
mjpeg_streamer = MJPEGStreamer()
