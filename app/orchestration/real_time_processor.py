"""
Real-time data flow management.

Handles:
- Real-time frame processing pipeline
- Performance monitoring and optimization
- Data flow between domains
- Resource management
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timezone
from collections import deque

from app.core.config import settings
from app.shared.types import CameraID

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """Manages real-time data processing pipeline."""
    
    def __init__(self):
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.result_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.performance_metrics: Dict[str, deque] = {
            "processing_times": deque(maxlen=100),
            "throughput": deque(maxlen=100),
            "queue_sizes": deque(maxlen=100),
            "error_rates": deque(maxlen=100)
        }
        self.processing_workers: List[asyncio.Task] = []
        self.is_running: bool = False
        self.target_fps: int = settings.TARGET_FPS
        self.frame_interval: float = 1.0 / self.target_fps
        logger.info("RealTimeProcessor initialized")
    
    async def start(self, num_workers: int = 4):
        """Start the real-time processing pipeline."""
        if self.is_running:
            logger.warning("RealTimeProcessor already running")
            return
        
        self.is_running = True
        
        # Start processing workers
        self.processing_workers = [
            asyncio.create_task(self._processing_worker(f"worker_{i}"))
            for i in range(num_workers)
        ]
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitor())
        
        logger.info(f"RealTimeProcessor started with {num_workers} workers")
    
    async def stop(self):
        """Stop the real-time processing pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.processing_workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.processing_workers, return_exceptions=True)
        
        self.processing_workers.clear()
        logger.info("RealTimeProcessor stopped")
    
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
                
                # Process the frame batch
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
                except asyncio.QueueFull:
                    logger.warning("Result queue full, dropping result")
                
                # Update performance metrics
                self.performance_metrics["processing_times"].append(processing_time)
                
                # Mark queue task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker {worker_name}: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"Processing worker {worker_name} stopped")
    
    async def _process_frame_batch(self, frame_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a frame batch through the pipeline."""
        try:
            # Simulate processing pipeline
            # In actual implementation, this would call the domain services
            await asyncio.sleep(0.01)  # Simulate processing time
            
            return {
                "success": True,
                "frame_count": len(frame_batch.get("cameras", {})),
                "stage": "complete",
                "data": frame_batch
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
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_metrics["processing_times"]:
            return {
                "avg_processing_time": 0,
                "throughput": 0,
                "queue_size": 0,
                "error_rate": 0
            }
        
        processing_times = list(self.performance_metrics["processing_times"])
        throughput_values = list(self.performance_metrics["throughput"])
        queue_sizes = list(self.performance_metrics["queue_sizes"])
        
        return {
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "max_processing_time": max(processing_times),
            "min_processing_time": min(processing_times),
            "throughput": sum(throughput_values) / len(throughput_values) if throughput_values else 0,
            "queue_size": self.processing_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "total_processed": len(processing_times),
            "target_fps": self.target_fps,
            "is_running": self.is_running
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