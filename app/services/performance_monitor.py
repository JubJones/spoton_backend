"""
Performance monitor for frame processing optimization.

Handles:
- Frame processing performance monitoring
- Adaptive quality control
- GPU memory usage monitoring
- Performance metrics collection
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import psutil
import threading

from app.infrastructure.gpu import get_gpu_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    processing_time: float
    memory_usage: float
    gpu_utilization: float
    frame_count: int
    dropped_frames: int
    quality_level: int
    throughput: float


@dataclass
class PerformanceThresholds:
    """Performance thresholds for adaptive control."""
    max_processing_time: float = 0.5  # seconds
    max_memory_usage: float = 0.8  # 80%
    max_gpu_utilization: float = 0.9  # 90%
    min_throughput: float = 10.0  # fps
    quality_adjustment_threshold: float = 0.1  # 10% performance degradation


class PerformanceMonitor:
    """
    Performance monitor for frame processing optimization.
    
    Features:
    - Real-time performance monitoring
    - Adaptive quality control
    - GPU memory usage tracking
    - Performance metrics collection
    """
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self.thresholds = PerformanceThresholds()
        
        # Performance history
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=100)
        self.processing_times: Deque[float] = deque(maxlen=50)
        self.memory_usage_history: Deque[float] = deque(maxlen=50)
        self.gpu_utilization_history: Deque[float] = deque(maxlen=50)
        
        # Performance state
        self.current_quality_level = getattr(settings, 'FRAME_JPEG_QUALITY', 85)
        self.frame_skip_enabled = False
        self.frame_skip_ratio = 0.0
        self.adaptive_quality_enabled = True
        
        # Statistics
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.quality_adjustments = 0
        self.performance_degradation_count = 0
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.performance_lock = threading.Lock()
        
        logger.info("PerformanceMonitor initialized")
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
                self.monitoring_task = None
            
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                # Collect performance metrics
                await self._collect_metrics()
                
                # Check for performance issues
                await self._check_performance_thresholds()
                
                # Apply adaptive optimizations
                await self._apply_adaptive_optimizations()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _collect_metrics(self):
        """Collect current performance metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Collect system metrics
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent / 100.0
            
            # Collect GPU metrics
            gpu_utilization = 0.0
            if self.gpu_manager and self.gpu_manager.is_available():
                gpu_utilization = self.gpu_manager.get_utilization() / 100.0
            
            # Calculate current throughput
            throughput = self._calculate_throughput()
            
            # Calculate average processing time
            avg_processing_time = 0.0
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            # Create metrics
            metrics = PerformanceMetrics(
                timestamp=current_time,
                processing_time=avg_processing_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                frame_count=self.total_frames_processed,
                dropped_frames=self.total_frames_dropped,
                quality_level=self.current_quality_level,
                throughput=throughput
            )
            
            # Store metrics
            with self.performance_lock:
                self.metrics_history.append(metrics)
                self.memory_usage_history.append(memory_usage)
                self.gpu_utilization_history.append(gpu_utilization)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _check_performance_thresholds(self):
        """Check if performance thresholds are exceeded."""
        try:
            if not self.metrics_history:
                return
            
            current_metrics = self.metrics_history[-1]
            
            # Check processing time threshold
            if current_metrics.processing_time > self.thresholds.max_processing_time:
                logger.warning(f"Processing time threshold exceeded: {current_metrics.processing_time:.3f}s")
                self.performance_degradation_count += 1
            
            # Check memory usage threshold
            if current_metrics.memory_usage > self.thresholds.max_memory_usage:
                logger.warning(f"Memory usage threshold exceeded: {current_metrics.memory_usage:.1%}")
                self.performance_degradation_count += 1
            
            # Check GPU utilization threshold
            if current_metrics.gpu_utilization > self.thresholds.max_gpu_utilization:
                logger.warning(f"GPU utilization threshold exceeded: {current_metrics.gpu_utilization:.1%}")
                self.performance_degradation_count += 1
            
            # Check throughput threshold
            if current_metrics.throughput < self.thresholds.min_throughput:
                logger.warning(f"Throughput below threshold: {current_metrics.throughput:.1f} fps")
                self.performance_degradation_count += 1
                
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
    
    async def _apply_adaptive_optimizations(self):
        """Apply adaptive optimizations based on performance."""
        try:
            if not self.adaptive_quality_enabled or not self.metrics_history:
                return
            
            current_metrics = self.metrics_history[-1]
            
            # Calculate performance degradation
            degradation_score = self._calculate_performance_degradation()
            
            # Apply quality adjustments
            if degradation_score > self.thresholds.quality_adjustment_threshold:
                await self._reduce_quality()
            elif degradation_score < -self.thresholds.quality_adjustment_threshold:
                await self._increase_quality()
            
            # Apply frame skipping if needed
            if current_metrics.processing_time > self.thresholds.max_processing_time * 1.5:
                await self._enable_frame_skipping()
            elif current_metrics.processing_time < self.thresholds.max_processing_time * 0.5:
                await self._disable_frame_skipping()
                
        except Exception as e:
            logger.error(f"Error applying adaptive optimizations: {e}")
    
    def _calculate_performance_degradation(self) -> float:
        """Calculate performance degradation score."""
        try:
            if len(self.metrics_history) < 2:
                return 0.0
            
            # Compare recent performance with baseline
            recent_metrics = list(self.metrics_history)[-10:]
            baseline_metrics = list(self.metrics_history)[-20:-10] if len(self.metrics_history) >= 20 else recent_metrics
            
            # Calculate average performance
            recent_avg_time = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
            baseline_avg_time = sum(m.processing_time for m in baseline_metrics) / len(baseline_metrics)
            
            # Calculate degradation
            if baseline_avg_time > 0:
                return (recent_avg_time - baseline_avg_time) / baseline_avg_time
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating performance degradation: {e}")
            return 0.0
    
    async def _reduce_quality(self):
        """Reduce quality to improve performance."""
        try:
            if self.current_quality_level > 60:
                self.current_quality_level = max(60, self.current_quality_level - 10)
                self.quality_adjustments += 1
                
                logger.info(f"Reduced quality to {self.current_quality_level} for performance optimization")
                
        except Exception as e:
            logger.error(f"Error reducing quality: {e}")
    
    async def _increase_quality(self):
        """Increase quality when performance allows."""
        try:
            if self.current_quality_level < 95:
                self.current_quality_level = min(95, self.current_quality_level + 5)
                self.quality_adjustments += 1
                
                logger.info(f"Increased quality to {self.current_quality_level} due to good performance")
                
        except Exception as e:
            logger.error(f"Error increasing quality: {e}")
    
    async def _enable_frame_skipping(self):
        """Enable frame skipping for performance."""
        try:
            if not self.frame_skip_enabled:
                self.frame_skip_enabled = True
                self.frame_skip_ratio = 0.2  # Skip 20% of frames
                
                logger.info("Enabled frame skipping for performance optimization")
            elif self.frame_skip_ratio < 0.5:
                self.frame_skip_ratio = min(0.5, self.frame_skip_ratio + 0.1)
                
                logger.info(f"Increased frame skip ratio to {self.frame_skip_ratio:.1%}")
                
        except Exception as e:
            logger.error(f"Error enabling frame skipping: {e}")
    
    async def _disable_frame_skipping(self):
        """Disable frame skipping when performance improves."""
        try:
            if self.frame_skip_enabled:
                if self.frame_skip_ratio > 0.1:
                    self.frame_skip_ratio = max(0.0, self.frame_skip_ratio - 0.1)
                    logger.info(f"Reduced frame skip ratio to {self.frame_skip_ratio:.1%}")
                else:
                    self.frame_skip_enabled = False
                    self.frame_skip_ratio = 0.0
                    logger.info("Disabled frame skipping due to improved performance")
                    
        except Exception as e:
            logger.error(f"Error disabling frame skipping: {e}")
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput in fps."""
        try:
            if len(self.processing_times) < 2:
                return 0.0
            
            # Calculate average processing time
            avg_time = sum(self.processing_times) / len(self.processing_times)
            
            # Calculate throughput
            if avg_time > 0:
                return 1.0 / avg_time
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating throughput: {e}")
            return 0.0
    
    def record_frame_processing(self, processing_time: float, frame_dropped: bool = False):
        """Record frame processing metrics."""
        try:
            with self.performance_lock:
                self.processing_times.append(processing_time)
                self.total_frames_processed += 1
                
                if frame_dropped:
                    self.total_frames_dropped += 1
                    
        except Exception as e:
            logger.error(f"Error recording frame processing: {e}")
    
    def should_skip_frame(self) -> bool:
        """Check if current frame should be skipped."""
        try:
            if not self.frame_skip_enabled:
                return False
            
            # Simple frame skipping logic
            import random
            return random.random() < self.frame_skip_ratio
            
        except Exception as e:
            logger.error(f"Error checking frame skip: {e}")
            return False
    
    def get_current_quality(self) -> int:
        """Get current quality level."""
        return self.current_quality_level
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            current_metrics = None
            if self.metrics_history:
                current_metrics = self.metrics_history[-1]
            
            return {
                "current_quality": self.current_quality_level,
                "frame_skip_enabled": self.frame_skip_enabled,
                "frame_skip_ratio": self.frame_skip_ratio,
                "total_frames_processed": self.total_frames_processed,
                "total_frames_dropped": self.total_frames_dropped,
                "quality_adjustments": self.quality_adjustments,
                "performance_degradation_count": self.performance_degradation_count,
                "current_metrics": {
                    "processing_time": current_metrics.processing_time if current_metrics else 0.0,
                    "memory_usage": current_metrics.memory_usage if current_metrics else 0.0,
                    "gpu_utilization": current_metrics.gpu_utilization if current_metrics else 0.0,
                    "throughput": current_metrics.throughput if current_metrics else 0.0
                } if current_metrics else {},
                "drop_rate": (
                    self.total_frames_dropped / max(1, self.total_frames_processed)
                ),
                "monitoring_active": self.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history."""
        try:
            history = []
            
            with self.performance_lock:
                for metrics in self.metrics_history:
                    history.append({
                        "timestamp": metrics.timestamp.isoformat(),
                        "processing_time": metrics.processing_time,
                        "memory_usage": metrics.memory_usage,
                        "gpu_utilization": metrics.gpu_utilization,
                        "frame_count": metrics.frame_count,
                        "dropped_frames": metrics.dropped_frames,
                        "quality_level": metrics.quality_level,
                        "throughput": metrics.throughput
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []
    
    def set_quality_level(self, quality: int):
        """Set quality level manually."""
        try:
            if 60 <= quality <= 95:
                self.current_quality_level = quality
                logger.info(f"Quality level set to {quality}")
            else:
                logger.warning(f"Invalid quality level: {quality}. Must be between 60 and 95")
                
        except Exception as e:
            logger.error(f"Error setting quality level: {e}")
    
    def enable_adaptive_quality(self, enabled: bool):
        """Enable or disable adaptive quality control."""
        try:
            self.adaptive_quality_enabled = enabled
            logger.info(f"Adaptive quality control {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Error setting adaptive quality: {e}")
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        try:
            with self.performance_lock:
                self.metrics_history.clear()
                self.processing_times.clear()
                self.memory_usage_history.clear()
                self.gpu_utilization_history.clear()
                
                self.total_frames_processed = 0
                self.total_frames_dropped = 0
                self.quality_adjustments = 0
                self.performance_degradation_count = 0
                
                # Reset to defaults
                self.current_quality_level = getattr(settings, 'FRAME_JPEG_QUALITY', 85)
                self.frame_skip_enabled = False
                self.frame_skip_ratio = 0.0
            
            logger.info("Performance metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting metrics: {e}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()