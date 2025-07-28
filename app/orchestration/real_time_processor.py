"""
Enhanced real-time data flow management.

Handles:
- Real-time frame processing pipeline
- Performance monitoring and optimization
- Data flow between domains
- Resource management
- Frame-by-frame processing coordination
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timezone
from collections import deque
import numpy as np

from app.core.config import settings
from app.shared.types import CameraID
from app.orchestration.pipeline_orchestrator import orchestrator
from app.orchestration.camera_manager import camera_manager

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """Enhanced real-time data processing pipeline."""
    
    def __init__(self):
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.result_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.frame_buffer: Dict[CameraID, deque] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, deque] = {
            "processing_times": deque(maxlen=100),
            "throughput": deque(maxlen=100),
            "queue_sizes": deque(maxlen=100),
            "error_rates": deque(maxlen=100),
            "frame_sync_times": deque(maxlen=100),
            "detection_times": deque(maxlen=100),
            "reid_times": deque(maxlen=100),
            "mapping_times": deque(maxlen=100)
        }
        
        # Processing workers and tasks
        self.processing_workers: List[asyncio.Task] = []
        self.sync_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Processing state
        self.is_running: bool = False
        self.target_fps: int = getattr(settings, 'TARGET_FPS', 30)
        self.frame_interval: float = 1.0 / self.target_fps
        self.sync_threshold_ms: int = 100  # Frame sync threshold
        
        # Frame processing statistics
        self.frame_stats = {
            "total_frames_received": 0,
            "total_frames_processed": 0,
            "total_frames_dropped": 0,
            "sync_failures": 0,
            "processing_failures": 0,
            "average_latency": 0.0
        }
        
        # Result callbacks
        self.result_callbacks: List[Callable] = []
        
        logger.info("Enhanced RealTimeProcessor initialized")
    
    async def start(self, num_workers: int = 4, environment_id: str = "default"):
        """Start the real-time processing pipeline."""
        if self.is_running:
            logger.warning("RealTimeProcessor already running")
            return
        
        # Initialize pipeline services
        if not orchestrator.is_initialized():
            logger.info("Initializing pipeline services...")
            if not await orchestrator.initialize_services(environment_id):
                logger.error("Failed to initialize pipeline services")
                return
        
        self.is_running = True
        
        # Start frame synchronization task
        self.sync_task = asyncio.create_task(self._frame_sync_worker())
        
        # Start processing workers
        self.processing_workers = [
            asyncio.create_task(self._processing_worker(f"worker_{i}"))
            for i in range(num_workers)
        ]
        
        # Start performance monitoring
        self.monitoring_task = asyncio.create_task(self._performance_monitor())
        
        logger.info(f"Enhanced RealTimeProcessor started with {num_workers} workers")
    
    async def stop(self):
        """Stop the real-time processing pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all tasks
        tasks_to_cancel = []
        
        if self.sync_task:
            tasks_to_cancel.append(self.sync_task)
        
        if self.monitoring_task:
            tasks_to_cancel.append(self.monitoring_task)
        
        tasks_to_cancel.extend(self.processing_workers)
        
        # Cancel all tasks
        for task in tasks_to_cancel:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Clear task references
        self.processing_workers.clear()
        self.sync_task = None
        self.monitoring_task = None
        
        logger.info("Enhanced RealTimeProcessor stopped")
    
    async def submit_frame(
        self, 
        camera_id: CameraID, 
        frame_data: Dict[str, Any]
    ) -> bool:
        """Submit a single frame for processing."""
        try:
            # Add timestamp if not present
            if "timestamp" not in frame_data:
                frame_data["timestamp"] = time.time() * 1000  # Convert to milliseconds
            
            # Initialize frame buffer for camera if needed
            if camera_id not in self.frame_buffer:
                self.frame_buffer[camera_id] = deque(maxlen=10)
            
            # Add frame to buffer
            self.frame_buffer[camera_id].append({
                "camera_id": camera_id,
                "frame_data": frame_data,
                "received_at": time.time()
            })
            
            self.frame_stats["total_frames_received"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error submitting frame for camera {camera_id}: {e}")
            return False
    
    async def submit_frame_batch(
        self, 
        frame_batch: Dict[str, Any], 
        priority: int = 0
    ) -> bool:
        """Submit a frame batch for processing."""
        try:
            processing_item = {
                "data": frame_batch,
                "priority": priority,
                "submitted_at": time.time(),
                "id": id(frame_batch)
            }
            
            self.processing_queue.put_nowait(processing_item)
            return True
            
        except asyncio.QueueFull:
            logger.warning("Processing queue full, dropping frame batch")
            return False
        except Exception as e:
            logger.error(f"Error submitting frame batch: {e}")
            return False
    
    async def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get a processing result."""
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error getting result: {e}")
            return None
    
    async def _frame_sync_worker(self):
        """Worker for synchronizing frames across cameras."""
        logger.info("Frame synchronization worker started")
        
        while self.is_running:
            try:
                # Wait for frame interval
                await asyncio.sleep(self.frame_interval)
                
                if not self.frame_buffer:
                    continue
                
                sync_start_time = time.time()
                
                # Collect latest frames from each camera
                camera_frames = {}
                reference_timestamp = None
                
                for camera_id, frame_buffer in self.frame_buffer.items():
                    if not frame_buffer:
                        continue
                    
                    # Find the most recent frame
                    latest_frame = frame_buffer[-1]
                    frame_timestamp = latest_frame["frame_data"]["timestamp"]
                    
                    if reference_timestamp is None:
                        reference_timestamp = frame_timestamp
                        camera_frames[camera_id] = latest_frame
                    else:
                        # Check if frame is within sync threshold
                        time_diff = abs(frame_timestamp - reference_timestamp)
                        if time_diff <= self.sync_threshold_ms:
                            camera_frames[camera_id] = latest_frame
                        else:
                            # Try to find a closer frame
                            best_frame = None
                            best_diff = float('inf')
                            
                            for frame in reversed(frame_buffer):
                                diff = abs(frame["frame_data"]["timestamp"] - reference_timestamp)
                                if diff < best_diff and diff <= self.sync_threshold_ms:
                                    best_frame = frame
                                    best_diff = diff
                            
                            if best_frame:
                                camera_frames[camera_id] = best_frame
                
                # If we have synchronized frames, submit for processing
                if len(camera_frames) > 1:  # Need at least 2 cameras
                    synchronized_batch = {
                        "camera_frames": {
                            camera_id: frame["frame_data"] 
                            for camera_id, frame in camera_frames.items()
                        },
                        "sync_timestamp": reference_timestamp,
                        "sync_quality": len(camera_frames) / len(self.frame_buffer)
                    }
                    
                    # Submit synchronized batch for processing
                    await self.submit_frame_batch(synchronized_batch, priority=1)
                    
                    # Track sync performance
                    sync_time = time.time() - sync_start_time
                    self.performance_metrics["frame_sync_times"].append(sync_time)
                    
                else:
                    self.frame_stats["sync_failures"] += 1
                    
            except Exception as e:
                logger.error(f"Error in frame synchronization worker: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("Frame synchronization worker stopped")
    
    async def _processing_worker(self, worker_name: str):
        """Worker coroutine for processing frame batches."""
        logger.info(f"Processing worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get next item from queue
                processing_item = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=1.0
                )
                
                start_time = time.time()
                
                # Process the frame batch through the pipeline
                result = await self._process_frame_batch(processing_item["data"])
                
                processing_time = time.time() - start_time
                
                # Add processing metadata
                result.update({
                    "processing_time": processing_time,
                    "worker": worker_name,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "queue_wait_time": start_time - processing_item["submitted_at"]
                })
                
                # Submit result
                try:
                    self.result_queue.put_nowait(result)
                    
                    # Call result callbacks
                    for callback in self.result_callbacks:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.error(f"Error in result callback: {e}")
                            
                except asyncio.QueueFull:
                    logger.warning("Result queue full, dropping result")
                    self.frame_stats["total_frames_dropped"] += 1
                
                # Update performance metrics
                self.performance_metrics["processing_times"].append(processing_time)
                
                # Track stage-specific times
                if result.get("stage_times"):
                    stage_times = result["stage_times"]
                    if "detection" in stage_times:
                        self.performance_metrics["detection_times"].append(stage_times["detection"])
                    if "reid" in stage_times:
                        self.performance_metrics["reid_times"].append(stage_times["reid"])
                    if "mapping" in stage_times:
                        self.performance_metrics["mapping_times"].append(stage_times["mapping"])
                
                # Update frame stats
                if result.get("success", False):
                    self.frame_stats["total_frames_processed"] += 1
                else:
                    self.frame_stats["processing_failures"] += 1
                
                # Mark queue task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker {worker_name}: {e}")
                self.frame_stats["processing_failures"] += 1
                await asyncio.sleep(0.1)
        
        logger.info(f"Processing worker {worker_name} stopped")
    
    async def _process_frame_batch(self, frame_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a frame batch through the pipeline."""
        try:
            # Create a task for this processing batch
            task_id = await orchestrator.initialize_task("real_time_processing")
            
            # Process through the pipeline orchestrator
            pipeline_result = await orchestrator.process_frame_batch(task_id, frame_batch)
            
            # Complete the task
            await orchestrator.complete_task(task_id, success=True)
            
            return {
                "success": True,
                "pipeline_result": pipeline_result,
                "task_id": str(task_id),
                "stage": "complete"
            }
            
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "error"
            }
    
    async def _performance_monitor(self):
        """Monitor and log performance metrics."""
        while self.is_running:
            try:
                # Calculate current metrics
                processing_queue_size = self.processing_queue.qsize()
                result_queue_size = self.result_queue.qsize()
                
                # Calculate throughput (frames per second)
                if self.performance_metrics["processing_times"]:
                    recent_times = list(self.performance_metrics["processing_times"])
                    avg_processing_time = sum(recent_times) / len(recent_times)
                    throughput = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                else:
                    throughput = 0
                
                # Store metrics
                self.performance_metrics["queue_sizes"].append(processing_queue_size)
                self.performance_metrics["throughput"].append(throughput)
                
                # Log metrics every 10 seconds
                if int(time.time()) % 10 == 0:
                    logger.info(
                        f"Performance metrics - "
                        f"Processing queue: {processing_queue_size}, "
                        f"Result queue: {result_queue_size}, "
                        f"Throughput: {throughput:.2f} fps, "
                        f"Avg processing time: {avg_processing_time:.4f}s"
                    )
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(1.0)
    
    def add_result_callback(self, callback: Callable):
        """Add a callback to be called when processing results are available."""
        self.result_callbacks.append(callback)
    
    def remove_result_callback(self, callback: Callable):
        """Remove a result callback."""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)
    
    def get_frame_stats(self) -> Dict[str, Any]:
        """Get frame processing statistics."""
        return {
            **self.frame_stats,
            "processing_rate": (
                self.frame_stats["total_frames_processed"] / 
                max(1, self.frame_stats["total_frames_received"])
            ),
            "drop_rate": (
                self.frame_stats["total_frames_dropped"] / 
                max(1, self.frame_stats["total_frames_received"])
            ),
            "sync_success_rate": (
                1.0 - (self.frame_stats["sync_failures"] / 
                       max(1, self.frame_stats["total_frames_received"]))
            ),
            "processing_success_rate": (
                1.0 - (self.frame_stats["processing_failures"] / 
                       max(1, self.frame_stats["total_frames_processed"]))
            )
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_metrics["processing_times"]:
            return {
                "avg_processing_time": 0,
                "throughput": 0,
                "queue_size": 0,
                "error_rate": 0,
                "stage_times": {}
            }
        
        processing_times = list(self.performance_metrics["processing_times"])
        throughput_values = list(self.performance_metrics["throughput"])
        queue_sizes = list(self.performance_metrics["queue_sizes"])
        
        # Calculate stage-specific metrics
        stage_metrics = {}
        for stage in ["frame_sync", "detection", "reid", "mapping"]:
            stage_times = list(self.performance_metrics.get(f"{stage}_times", []))
            if stage_times:
                stage_metrics[stage] = {
                    "avg_time": sum(stage_times) / len(stage_times),
                    "max_time": max(stage_times),
                    "min_time": min(stage_times),
                    "total_runs": len(stage_times)
                }
            else:
                stage_metrics[stage] = {
                    "avg_time": 0,
                    "max_time": 0,
                    "min_time": 0,
                    "total_runs": 0
                }
        
        return {
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "max_processing_time": max(processing_times),
            "min_processing_time": min(processing_times),
            "throughput": sum(throughput_values) / len(throughput_values) if throughput_values else 0,
            "queue_size": self.processing_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "total_processed": len(processing_times),
            "target_fps": self.target_fps,
            "actual_fps": len(processing_times) / max(1, len(processing_times)) if processing_times else 0,
            "is_running": self.is_running,
            "stage_metrics": stage_metrics,
            "frame_buffer_sizes": {
                camera_id: len(buffer) for camera_id, buffer in self.frame_buffer.items()
            }
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "processing_queue_size": self.processing_queue.qsize(),
            "processing_queue_maxsize": self.processing_queue.maxsize,
            "result_queue_size": self.result_queue.qsize(),
            "result_queue_maxsize": self.result_queue.maxsize,
            "processing_workers": len(self.processing_workers),
            "is_running": self.is_running
        }

# Global real-time processor instance
real_time_processor = RealTimeProcessor()