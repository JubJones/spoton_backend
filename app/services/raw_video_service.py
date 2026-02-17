"""
Raw video streaming service - streams video frames without AI processing.

Provides:
- Raw video frame streaming from multiple cameras
    - No AI processing (no detection/tracking/id association)
- Direct frame-to-WebSocket streaming
- Background task management for raw video streaming
"""

import asyncio
from pathlib import Path
import uuid
from uuid import UUID
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import logging
from datetime import datetime, timezone
import time
import numpy as np
import cv2

from app.core.config import settings
from app.services.video_data_manager_service import VideoDataManagerService
from app.utils.asset_downloader import AssetDownloader
from app.api.websockets.connection_manager import binary_websocket_manager, MessageType
from app.api.websockets.frame_handler import frame_handler

if TYPE_CHECKING:
    from app.services.playback_status_store import PlaybackStatusStore
    from app.services.task_runtime_registry import TaskRuntimeRegistry

logger = logging.getLogger(__name__)


class RawVideoService:
    """
    Service for streaming raw video frames without AI processing.
    
    Features:
    - Multi-camera raw video streaming
    - Direct frame extraction and streaming
    - Background task management
    - WebSocket frame delivery
    """
    
    def __init__(self):
        self.tasks: Dict[UUID, Dict[str, Any]] = {}
        self.active_tasks: set = set()
        self.environment_tasks: Dict[str, UUID] = {}
        
        # Video processing services
        self.video_data_manager: Optional[VideoDataManagerService] = None
        self.asset_downloader: Optional[AssetDownloader] = None
        
        # WebSocket connection guard state per task
        self._client_connection_watch: Dict[UUID, Dict[str, Any]] = {}
        
        # Raw streaming statistics
        self.streaming_stats = {
            "total_frames_streamed": 0,
            "total_tasks_created": 0,
            "successful_streams": 0,
            "failed_streams": 0,
            "average_streaming_time": 0.0
        }
        
        # Performance tracking
        self.streaming_times: List[float] = []

        # Playback coordination (optional dependencies)
        self.playback_status_store: Optional["PlaybackStatusStore"] = None
        self.playback_runtime_registry: Optional["TaskRuntimeRegistry"] = None

        logger.info("RawVideoService initialized")

    def attach_playback_interfaces(
        self,
        *,
        status_store: Optional["PlaybackStatusStore"] = None,
        runtime_registry: Optional["TaskRuntimeRegistry"] = None,
    ) -> None:
        """Attach playback coordination collaborators provided via dependency injection."""

        if status_store is not None:
            self.playback_status_store = status_store
        if runtime_registry is not None:
            self.playback_runtime_registry = runtime_registry

    async def _register_playback_task(self, task_id: UUID) -> None:
        if self.playback_runtime_registry is not None:
            await self.playback_runtime_registry.register(str(task_id))
        if self.playback_status_store is not None:
            await self.playback_status_store.get_status(str(task_id))

    async def _cleanup_playback_task(self, task_id: UUID) -> None:
        if self.playback_runtime_registry is not None:
            await self.playback_runtime_registry.remove(str(task_id))

    async def _wait_for_playback(self, task_id: UUID) -> None:
        if self.playback_runtime_registry is not None:
            await self.playback_runtime_registry.wait_until_playing(str(task_id))

    async def _record_playback_progress(self, task_id: UUID, frame_index: int) -> None:
        if self.playback_runtime_registry is not None:
            self.playback_runtime_registry.update_frame_index(str(task_id), frame_index)
        if self.playback_status_store is not None:
            await self.playback_status_store.update_last_frame_index(str(task_id), frame_index)

    def _apply_seek_if_pending(self, task_id: UUID, video_data: Dict[str, Any]) -> Optional[int]:
        """Check for a pending seek and jump all VideoCaptures. Returns new frame_index or None."""
        if self.playback_runtime_registry is None:
            return None
        runtime = self.playback_runtime_registry.get_runtime(str(task_id))
        if runtime is None or runtime.seek_frame_index is None:
            return None
        target = runtime.seek_frame_index
        runtime.seek_frame_index = None  # consume the seek request
        import cv2
        for camera_id, data in video_data.items():
            cap = data.get("video_capture")
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        return target

    def _get_client_timeout_config(self, *, detection_mode: bool = False) -> Tuple[float, float]:
        """Resolve grace and idle timeouts for the current streaming mode."""
        if detection_mode:
            grace = (
                getattr(settings, "DETECTION_CLIENT_INITIAL_GRACE_SECONDS", None)
                or settings.STREAMING_CLIENT_INITIAL_GRACE_SECONDS
            )
            idle = (
                getattr(settings, "DETECTION_CLIENT_IDLE_TIMEOUT_SECONDS", None)
                or settings.STREAMING_CLIENT_IDLE_TIMEOUT_SECONDS
            )
        else:
            grace = settings.STREAMING_CLIENT_INITIAL_GRACE_SECONDS
            idle = settings.STREAMING_CLIENT_IDLE_TIMEOUT_SECONDS
        return float(grace), float(idle)

    def _ensure_client_watch(self, task_id: UUID) -> Dict[str, Any]:
        """Ensure a connection watch entry exists for a task."""
        watch = self._client_connection_watch.get(task_id)
        if watch is None:
            watch = {
                "last_seen": time.time(),
                "ever_connected": False,
                "timeout_logged": False,
            }
            self._client_connection_watch[task_id] = watch
        return watch

    def _clear_client_watch(self, task_id: UUID) -> None:
        """Clear connection guard state for a task."""
        self._client_connection_watch.pop(task_id, None)

    def _should_continue_stream(self, task_id: UUID, *, detection_mode: bool = False) -> bool:
        """Evaluate whether streaming should continue based on WebSocket connections."""
        watch = self._ensure_client_watch(task_id)
        now = time.time()
        connection_count = binary_websocket_manager.get_connection_count(str(task_id))

        if connection_count > 0:
            watch["ever_connected"] = True
            watch["last_seen"] = now
            watch["timeout_logged"] = False
            return True

        grace, idle = self._get_client_timeout_config(detection_mode=detection_mode)
        timeout = idle if watch.get("ever_connected") else grace
        elapsed = now - watch.get("last_seen", now)

        if elapsed <= timeout:
            return True

        if not watch.get("timeout_logged"):
            logger.info(
                "🛑 STREAM GUARD: No active WebSocket clients for task %s for %.1fs (ever_connected=%s); initiating shutdown",
                task_id,
                elapsed,
                watch.get("ever_connected"),
            )
            watch["timeout_logged"] = True
        return False

    def _mark_task_stopped_due_to_idle_clients(self, task_id: UUID, reason: str) -> None:
        """Mark a task as stopped because no WebSocket clients remain."""
        task_state = self.tasks.get(task_id)
        if task_state is not None:
            task_state.update({
                "status": "STOPPED",
                "current_step": reason,
                "details": reason,
                "updated_at": datetime.now(timezone.utc),
            })
        self.active_tasks.discard(task_id)
        self._clear_client_watch(task_id)

    def _task_marked_stopped(self, task_id: UUID) -> bool:
        """Check if a task has been marked as stopped."""
        task_state = self.tasks.get(task_id)
        return bool(task_state and task_state.get("status") == "STOPPED")

    async def initialize_services(self, environment_id: str = "default") -> bool:
        """Initialize video processing services for raw streaming."""
        try:
            # logger.info(f"🚀 RAW SERVICE INIT: Starting raw video service initialization for environment: {environment_id}")
            
            # Initialize asset downloader
            self.asset_downloader = AssetDownloader(
                s3_endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                s3_bucket_name=settings.S3_BUCKET_NAME
            )
            
            # Initialize video data manager
            self.video_data_manager = VideoDataManagerService(
                asset_downloader=self.asset_downloader
            )
            
            # logger.info("✅ RAW SERVICE INIT: Raw video services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAW SERVICE INIT: Failed to initialize raw video services: {e}")
            return False
    
    async def initialize_raw_task(self, environment_id: str) -> UUID:
        """Initialize a new raw video streaming task."""
        try:
            # Check if environment already has an active task
            if environment_id in self.environment_tasks:
                existing_task_id = self.environment_tasks[environment_id]
                if existing_task_id in self.active_tasks:
                    logger.warning(f"Environment {environment_id} already has active raw task: {existing_task_id}")
                    raise ValueError(f"Environment {environment_id} already has an active raw streaming task")
            
            # Generate new task ID
            task_id = uuid.uuid4()
            
            # Initialize task state
            task_state = {
                "task_id": str(task_id),
                "environment_id": environment_id,
                "status": "QUEUED",
                "progress": 0.0,
                "current_step": "Initializing raw video streaming",
                "details": f"Raw video streaming task created for environment {environment_id}",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "mode": "raw_streaming"
            }
            
            # Store task state
            self.tasks[task_id] = task_state
            self.active_tasks.add(task_id)
            self.environment_tasks[environment_id] = task_id

            await self._register_playback_task(task_id)
            
            # Update statistics
            self.streaming_stats["total_tasks_created"] += 1
            
            # logger.info(f"✅ RAW TASK INIT: Task {task_id} initialized for environment {environment_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"❌ RAW TASK INIT: Error initializing raw task for environment {environment_id}: {e}")
            raise
    
    async def run_raw_streaming_pipeline(self, task_id: UUID, environment_id: str):
        """
        Main pipeline for raw video streaming without AI processing.
        
        Process:
        1. Initialize services
        2. Download video data
        3. Extract and stream raw frames
        4. Send frames via WebSocket
        """
        streaming_start = time.time()
        
        try:
            # logger.info(f"🎬 RAW PIPELINE: Starting raw streaming pipeline for task {task_id}, environment {environment_id}")
            
            # Update task status
            await self._update_task_status(task_id, "INITIALIZING", 0.05, "Initializing raw video services")
            
            # Step 1: Initialize services
            # logger.info(f"📹 RAW PIPELINE: Step 1/4 - Initializing services for task {task_id}")
            services_initialized = await self.initialize_services(environment_id)
            if not services_initialized:
                raise RuntimeError("Failed to initialize raw video services")
            
            await self._update_task_status(task_id, "DOWNLOADING", 0.25, "Downloading video data")
            
            # Step 2: Download video data
            # logger.info(f"⬇️ RAW PIPELINE: Step 2/4 - Downloading video data for task {task_id}")
            video_data = await self._download_video_data(environment_id)
            if not video_data:
                raise RuntimeError("Failed to download video data")
            
            await self._update_task_status(task_id, "EXTRACTING", 0.50, "Extracting raw frames")
            
            # Step 3: Extract frames
            # logger.info(f"🖼️ RAW PIPELINE: Step 3/4 - Extracting frames for task {task_id}")
            frames_extracted = await self._extract_raw_frames(task_id, video_data)
            if not frames_extracted:
                raise RuntimeError("Failed to extract frames")
            
            await self._update_task_status(task_id, "STREAMING", 0.75, "Streaming raw video frames")
            
            # Step 4: Stream frames
            # logger.info(f"📡 RAW PIPELINE: Step 4/4 - Streaming frames for task {task_id}")
            streaming_success = await self._stream_raw_frames(task_id, video_data)
            if self._task_marked_stopped(task_id):
                logger.info(f"🛑 RAW PIPELINE: Task {task_id} stopped early (likely no active WebSocket clients)")
                return
            if not streaming_success:
                raise RuntimeError("Failed to stream frames")

            # Mark as completed
            await self._update_task_status(task_id, "COMPLETED", 1.0, "Raw video streaming completed successfully")
            
            # Update statistics
            streaming_time = time.time() - streaming_start
            self.streaming_times.append(streaming_time)
            self.streaming_stats["successful_streams"] += 1
            self._update_streaming_stats()
            
            # logger.info(f"✅ RAW PIPELINE: Raw streaming pipeline completed successfully for task {task_id}")
            
        except Exception as e:
            logger.error(f"❌ RAW PIPELINE: Error in raw streaming pipeline for task {task_id}: {e}")
            await self._update_task_status(task_id, "FAILED", 0.0, f"Raw streaming failed: {str(e)}")
            self.streaming_stats["failed_streams"] += 1
            
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            if environment_id in self.environment_tasks:
                del self.environment_tasks[environment_id]
            self._clear_client_watch(task_id)
            await self._cleanup_playback_task(task_id)
    
    async def _download_video_data(self, environment_id: str, sub_video_index: int = 0) -> Dict[str, Any]:
        """Download video data for the environment.
        
        Args:
            environment_id: Environment identifier (e.g., 'campus', 'factory')
            sub_video_index: 0-based index of the sub-video batch to load
        """
        try:
            # Get video configuration for environment
            video_configs = [vc for vc in settings.VIDEO_SETS if vc.env_id == environment_id]
            if not video_configs:
                raise ValueError(f"No video configuration found for environment {environment_id}")
            
            # Download all cameras for this environment using the batch method
            video_paths = await self.video_data_manager.download_sub_videos_for_environment_batch(
                task_id=uuid.uuid4(), environment_id=environment_id, sub_video_index=sub_video_index
            )
            
            # Process the downloaded video paths
            video_data = {}
            for video_config in video_configs:
                camera_id = video_config.cam_id
                video_path = video_paths.get(camera_id)
                
                if video_path:
                    video_data[camera_id] = {
                        "video_path": video_path,
                        "config": video_config
                    }
                else:
                    logger.warning(f"⚠️ RAW DOWNLOAD: Failed to download video for camera {camera_id} (sub_video_index={sub_video_index})")
            
            return video_data
            
        except Exception as e:
            logger.error(f"❌ RAW DOWNLOAD: Error downloading video data: {e}")
            raise
    
    async def _extract_raw_frames(self, task_id: UUID, video_data: Dict[str, Any]) -> bool:
        """Extract frames from video files."""
        try:
            # logger.info(f"🖼️ RAW EXTRACT: Extracting frames for task {task_id}")
            
            # Process each camera's video
            for camera_id, data in video_data.items():
                video_path = data["video_path"]
                
                # Open video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"❌ RAW EXTRACT: Cannot open video file: {video_path}")
                    continue
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # logger.info(f"📹 RAW EXTRACT: Camera {camera_id} - FPS: {fps}, Frames: {frame_count}")
                
                # Store frame extraction info
                data["fps"] = fps
                data["frame_count"] = frame_count
                data["video_capture"] = cap
                
            # logger.info(f"✅ RAW EXTRACT: Frame extraction setup completed for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAW EXTRACT: Error extracting frames: {e}")
            return False
    
    async def _stream_raw_frames(self, task_id: UUID, video_data: Dict[str, Any]) -> bool:
        """Stream raw frames via WebSocket."""
        try:
            # logger.info(f"📡 RAW STREAM: Starting frame streaming for task {task_id}")
            
            # Calculate target FPS and frame interval
            target_fps = getattr(settings, 'TARGET_FPS', 10)
            frame_interval = 1.0 / target_fps
            
            # Get frame count for progress tracking
            frame_counts = [data.get("frame_count", 0) for data in video_data.values() if data.get("frame_count", 0) > 0]
            total_frames = min(frame_counts) if frame_counts else 0
            if total_frames == 0:
                logger.warning("No frames available for streaming")
                return False

            # Broadcast total_frames so frontend can show progress bar
            if self.playback_status_store is not None:
                await self.playback_status_store.set_total_frames(str(task_id), total_frames)
            
            frame_index = 0
            last_frame_time = time.time()
            aborted_due_to_no_clients = False
            
            # Stream frames
            while frame_index < total_frames:
                # Check if task is still active (can be stopped by cleanup)
                if task_id not in self.active_tasks:
                    logger.info(f"📡 RAW STREAM: Task {task_id} was stopped, ending streaming at frame {frame_index}")
                    break

                await self._wait_for_playback(task_id)

                if task_id not in self.active_tasks:
                    logger.info(f"📡 RAW STREAM: Task {task_id} stopped during pause wait")
                    break

                # Handle pending seek — jump all cameras to target frame
                seek_target = self._apply_seek_if_pending(task_id, video_data)
                if seek_target is not None:
                    frame_index = seek_target
                    if frame_index >= total_frames:
                        break
                    continue  # re-enter loop with new frame_index

                if not self._should_continue_stream(task_id, detection_mode=False):
                    aborted_due_to_no_clients = True
                    break
                    
                current_time = time.time()
                
                # Read frames from all cameras
                camera_frames = {}
                all_frames_valid = True
                
                for camera_id, data in video_data.items():
                    cap = data.get("video_capture")
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            camera_frames[camera_id] = frame
                        else:
                            all_frames_valid = False
                            break
                    else:
                        all_frames_valid = False
                        break
                
                if not all_frames_valid or not camera_frames:
                    logger.info(f"📡 RAW STREAM: End of video reached at frame {frame_index}")
                    break
                
                # Send frames via WebSocket
                frame_metadata = {
                    "frame_index": frame_index,
                    "scene_id": f"raw_{task_id}",
                    "frame_quality": 1.0,
                    "sync_timestamp": current_time * 1000,
                    "mode": "raw_streaming"
                }
                
                # Send individual messages per camera to avoid oversized WebSocket messages
                success_count = 0
                timestamp_utc = datetime.now(timezone.utc).isoformat()
                
                for camera_id, frame in camera_frames.items():
                    # Convert frame to base64
                    frame_base64 = await frame_handler.create_frame_base64(frame, quality=85)
                    # Optional on-disk frame cache (sampled)
                    try:
                        if getattr(settings, 'STORE_EXTRACTED_FRAMES', False):
                            sample_rate = int(getattr(settings, 'FRAME_CACHE_SAMPLE_RATE', 0))
                            if sample_rate and frame_index % sample_rate == 0:
                                from base64 import b64decode
                                cache_dir = Path(getattr(settings, 'FRAME_CACHE_DIR', './extracted_frames')) / str(task_id) / str(camera_id)
                                cache_dir.mkdir(parents=True, exist_ok=True)
                                out_path = cache_dir / f"frame_{frame_index:06d}.jpg"
                                # Strip potential data URL prefix
                                b64_str = frame_base64.split(',', 1)[-1]
                                await asyncio.to_thread(out_path.write_bytes, b64decode(b64_str))
                    except Exception:
                        pass
                    
                    # Create individual camera message
                    camera_message = {
                        "type": MessageType.TRACKING_UPDATE.value,
                        "task_id": str(task_id),
                        "camera_id": camera_id,
                        "global_frame_index": frame_index,
                        "timestamp_processed_utc": timestamp_utc,
                        "mode": "raw_streaming",
                        "camera_data": {
                            "frame_image_base64": frame_base64,
                            "tracks": [],  # No tracks in raw mode
                            "frame_width": frame.shape[1],
                            "frame_height": frame.shape[0],
                            "timestamp": timestamp_utc
                        }
                    }
                    
                    # Send individual camera message
                    camera_success = await binary_websocket_manager.send_json_message(
                        str(task_id), camera_message, MessageType.TRACKING_UPDATE
                    )
                    
                    if camera_success:
                        success_count += 1
                
                success = success_count > 0

                if success:
                    self.streaming_stats["total_frames_streamed"] += len(camera_frames)

                    await self._record_playback_progress(task_id, frame_index)

                    # Log progress periodically
                    if frame_index % 30 == 0:  # Every 30 frames
                        progress = (frame_index / total_frames) * 0.25 + 0.75  # 0.75-1.0 range
                        await self._update_task_status(task_id, "STREAMING", progress, 
                                                     f"Streaming frame {frame_index}/{total_frames}")
                        # logger.info(f"📡 RAW STREAM: Streamed frame {frame_index}/{total_frames} for task {task_id}")
                
                # Frame rate control
                elapsed = time.time() - last_frame_time
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)
                
                last_frame_time = time.time()
                frame_index += 1
            
            # Cleanup video captures
            for data in video_data.values():
                cap = data.get("video_capture")
                if cap:
                    cap.release()
            
            if aborted_due_to_no_clients:
                reason = "Streaming stopped - no active WebSocket clients"
                self._mark_task_stopped_due_to_idle_clients(task_id, reason)
                logger.info(f"🛑 RAW STREAM: {reason} (task {task_id})")
                return True

            # logger.info(f"✅ RAW STREAM: Frame streaming completed for task {task_id}")
            return True

        except Exception as e:
            logger.error(f"❌ RAW STREAM: Error streaming frames: {e}")
            return False
    
    async def _update_task_status(self, task_id: UUID, status: str, progress: float, details: str):
        """Update task status."""
        try:
            if task_id in self.tasks:
                self.tasks[task_id].update({
                    "status": status,
                    "progress": progress,
                    "current_step": details,
                    "details": details,
                    "updated_at": datetime.now(timezone.utc)
                })
                
                # Send status update via WebSocket
                status_message = {
                    "type": "status_update",
                    "task_id": str(task_id),
                    "status": status,
                    "progress": progress,
                    "current_step": details,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": "raw_streaming"
                }
                
                await binary_websocket_manager.send_json_message(
                    str(task_id), status_message, MessageType.STATUS_UPDATE, target_channel="tracking"
                )
                
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
    
    def _update_streaming_stats(self):
        """Update streaming statistics."""
        try:
            if self.streaming_times:
                self.streaming_stats["average_streaming_time"] = sum(self.streaming_times) / len(self.streaming_times)
                
                # Keep only recent times
                if len(self.streaming_times) > 100:
                    self.streaming_times = self.streaming_times[-50:]
                    
        except Exception as e:
            logger.error(f"Error updating streaming stats: {e}")
    
    async def get_raw_task_status(self, task_id: UUID) -> Optional[Dict[str, Any]]:
        """Get raw task status."""
        return self.tasks.get(task_id)
    
    async def get_all_raw_task_statuses(self) -> List[Dict[str, Any]]:
        """Get all raw task statuses."""
        return list(self.tasks.values())

    async def stop_raw_task(self, task_id: uuid.UUID) -> bool:
        """
        Stop and cleanup a raw processing task.
        
        Args:
            task_id: UUID of the task to stop
            
        Returns:
            bool: True if task was stopped successfully, False if not found
        """
        try:
            task_id_str = str(task_id)
            logger.info(f"🛑 RAW TASK STOP: Attempting to stop task {task_id_str}")
            
            # Check if task exists (use UUID object as key, not string)
            if task_id not in self.tasks:
                logger.warning(f"🛑 RAW TASK STOP: Task {task_id_str} not found in tasks")
                return False
            
            # Get task info
            task_info = self.tasks[task_id]
            environment_id = task_info.get("environment_id")
            
            # Update task status to STOPPED
            task_info["status"] = "STOPPED"
            task_info["details"] = "Task stopped by user request"
            task_info["updated_at"] = str(datetime.utcnow().isoformat())
            
            # Remove from active tasks (use UUID object as key)
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
                logger.info(f"🛑 RAW TASK STOP: Removed {task_id_str} from active tasks")
            self._clear_client_watch(task_id)
            
            # Remove environment association
            if environment_id and environment_id in self.environment_tasks:
                if self.environment_tasks[environment_id] == task_id:
                    del self.environment_tasks[environment_id]
                    logger.info(f"🛑 RAW TASK STOP: Cleared environment {environment_id} association")
            
            # Clean up any WebSocket connections for this task
            from app.api.websockets.connection_manager import binary_websocket_manager
            connection_count = binary_websocket_manager.get_connection_count(task_id_str)
            if connection_count > 0:
                logger.info(f"🛑 RAW TASK STOP: Found {connection_count} WebSocket connections for task {task_id_str}")
                # Disconnect all WebSocket connections for this task
                if task_id_str in binary_websocket_manager.active_connections:
                    connections_to_close = binary_websocket_manager.active_connections[task_id_str].copy()
                    for websocket in connections_to_close:
                        await binary_websocket_manager.disconnect(websocket, task_id_str)
                    logger.info(f"🛑 RAW TASK STOP: Disconnected {len(connections_to_close)} WebSocket connections")
            
            logger.info(f"✅ RAW TASK STOP: Successfully stopped task {task_id_str}")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAW TASK STOP: Error stopping task {task_id}: {e}")
            return False
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            **self.streaming_stats,
            "active_tasks_count": len(self.active_tasks),
            "total_tasks_count": len(self.tasks)
        }


# Global raw video service instance
raw_video_service = RawVideoService()
