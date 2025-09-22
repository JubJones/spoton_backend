"""
Production Metrics Collection System for Phase 5: Production Readiness

Comprehensive metrics collection and monitoring system for production deployment
as specified in DETECTION.md Phase 5. Provides performance monitoring, quality metrics,
and operational insights for production optimization.

Features:
- Real-time performance metrics collection
- Pipeline component monitoring (Detection, Re-ID, Homography, Handoff)
- Resource utilization tracking (CPU, GPU, Memory)
- Quality metrics and SLA monitoring
- Alert generation and threshold management
- Metrics export for external monitoring systems
"""

import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque, defaultdict
import statistics
import json
import asyncio
from enum import Enum

# Try to import GPU monitoring (optional)
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected by the system."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RESOURCE = "resource"
    PIPELINE = "pipeline"
    ERROR = "error"
    BUSINESS = "business"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline performance metrics."""
    # Detection metrics
    detection_times: List[float] = field(default_factory=list)
    detection_counts: List[int] = field(default_factory=list)
    detection_success_rate: float = 0.0
    
    # Tracking and Re-ID metrics
    tracking_times: List[float] = field(default_factory=list)
    reid_times: List[float] = field(default_factory=list)
    reid_success_rate: float = 0.0
    global_id_assignments: int = 0
    
    # Spatial intelligence metrics
    homography_times: List[float] = field(default_factory=list)
    homography_success_rate: float = 0.0
    handoff_detections: int = 0
    
    # Overall pipeline metrics
    total_frame_times: List[float] = field(default_factory=list)
    frames_processed: int = 0
    pipeline_throughput: float = 0.0  # frames per second
    
    def add_frame_metrics(self, detection_time: float, tracking_time: float,
                         reid_time: float, homography_time: float,
                         detection_count: int, reid_success: bool,
                         homography_success: bool):
        """Add metrics for a processed frame."""
        self.detection_times.append(detection_time)
        self.tracking_times.append(tracking_time)
        self.reid_times.append(reid_time)
        self.homography_times.append(homography_time)
        self.detection_counts.append(detection_count)
        
        total_time = detection_time + tracking_time + reid_time + homography_time
        self.total_frame_times.append(total_time)
        self.frames_processed += 1
        
        # Update success rates
        if reid_success:
            self.global_id_assignments += 1
        
        # Calculate pipeline throughput
        if total_time > 0:
            current_fps = 1.0 / total_time
            alpha = 0.1  # EMA factor
            self.pipeline_throughput = (1 - alpha) * self.pipeline_throughput + alpha * current_fps
        
        # Keep metrics lists manageable
        max_samples = 1000
        if len(self.total_frame_times) > max_samples:
            self.detection_times = self.detection_times[-max_samples//2:]
            self.tracking_times = self.tracking_times[-max_samples//2:]
            self.reid_times = self.reid_times[-max_samples//2:]
            self.homography_times = self.homography_times[-max_samples//2:]
            self.detection_counts = self.detection_counts[-max_samples//2:]
            self.total_frame_times = self.total_frame_times[-max_samples//2:]


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality and SLA metrics."""
    # Performance SLA metrics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Quality metrics
    detection_accuracy: float = 0.0
    tracking_accuracy: float = 0.0
    reid_accuracy: float = 0.0
    
    # Availability metrics
    uptime_percentage: float = 100.0
    error_rate: float = 0.0
    
    # Throughput metrics
    requests_per_minute: float = 0.0
    frames_per_second: float = 0.0


class ProductionMetricsCollector:
    """
    Comprehensive metrics collection system for production monitoring.
    
    Collects, processes, and exports metrics for external monitoring systems
    while providing real-time alerting and performance optimization insights.
    """
    
    def __init__(self, collection_interval: float = 10.0, retention_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: How often to collect system metrics (seconds)
            retention_hours: How long to retain metrics in memory
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.max_data_points = int((retention_hours * 3600) / collection_interval)
        
        # Metrics storage
        self.metrics_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_data_points))
        self.pipeline_metrics = PipelineMetrics()
        self.resource_metrics = ResourceMetrics()
        self.quality_metrics = QualityMetrics()
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'gpu_usage': {'warning': 85.0, 'critical': 95.0},
            'response_time': {'warning': 1000.0, 'critical': 2000.0},  # milliseconds
            'error_rate': {'warning': 5.0, 'critical': 10.0},  # percentage
            'throughput': {'warning': 10.0, 'critical': 5.0}  # minimum fps
        }
        
        # Alert tracking
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.start_time = time.time()
        
        # Performance tracking
        self.collection_stats = {
            'collections_performed': 0,
            'collection_errors': 0,
            'last_collection_time': 0.0,
            'avg_collection_duration': 0.0
        }
        
        logger.info(f"ProductionMetricsCollector initialized - interval: {collection_interval}s, retention: {retention_hours}h")
    
    async def start_collection(self):
        """Start continuous metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started continuous metrics collection")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                collection_start = time.time()
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check alert conditions
                await self._check_alert_conditions()
                
                # Update collection statistics
                collection_duration = time.time() - collection_start
                self._update_collection_stats(collection_duration)
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                self.collection_stats['collection_errors'] += 1
                await asyncio.sleep(min(self.collection_interval, 30.0))  # Fallback interval
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            current_time = time.time()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.resource_metrics.cpu_usage = cpu_percent
            self._add_metric_point('cpu_usage', cpu_percent, current_time)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.resource_metrics.memory_usage = memory.percent
            self.resource_metrics.memory_available = memory.available / (1024**3)  # GB
            self._add_metric_point('memory_usage', memory.percent, current_time)
            self._add_metric_point('memory_available_gb', self.resource_metrics.memory_available, current_time)
            
            # GPU metrics (if available)
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        self.resource_metrics.gpu_usage = gpu.load * 100
                        self.resource_metrics.gpu_memory_usage = gpu.memoryUtil * 100
                        self._add_metric_point('gpu_usage', self.resource_metrics.gpu_usage, current_time)
                        self._add_metric_point('gpu_memory_usage', self.resource_metrics.gpu_memory_usage, current_time)
                except Exception as gpu_error:
                    logger.debug(f"GPU metrics collection failed: {gpu_error}")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.resource_metrics.disk_usage = disk.percent
            self._add_metric_point('disk_usage', disk.percent, current_time)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.resource_metrics.network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            self.collection_stats['collections_performed'] += 1
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            self.collection_stats['collection_errors'] += 1
    
    def record_pipeline_metrics(self, detection_time: float, tracking_time: float = 0.0,
                              reid_time: float = 0.0, homography_time: float = 0.0,
                              detection_count: int = 0, reid_success: bool = False,
                              homography_success: bool = False):
        """
        Record pipeline processing metrics.
        
        Args:
            detection_time: Time spent on detection (seconds)
            tracking_time: Time spent on tracking (seconds)
            reid_time: Time spent on re-identification (seconds)
            homography_time: Time spent on homography processing (seconds)
            detection_count: Number of detections found
            reid_success: Whether re-ID was successful
            homography_success: Whether homography processing was successful
        """
        try:
            current_time = time.time()
            
            # Add to pipeline metrics
            self.pipeline_metrics.add_frame_metrics(
                detection_time, tracking_time, reid_time, homography_time,
                detection_count, reid_success, homography_success
            )
            
            # Record individual component metrics
            self._add_metric_point('detection_time_ms', detection_time * 1000, current_time)
            self._add_metric_point('tracking_time_ms', tracking_time * 1000, current_time)
            self._add_metric_point('reid_time_ms', reid_time * 1000, current_time)
            self._add_metric_point('homography_time_ms', homography_time * 1000, current_time)
            self._add_metric_point('detection_count', detection_count, current_time)
            self._add_metric_point('pipeline_fps', self.pipeline_metrics.pipeline_throughput, current_time)
            
            # Update quality metrics
            self._update_quality_metrics()
            
        except Exception as e:
            logger.error(f"Error recording pipeline metrics: {e}")
    
    def record_business_metrics(self, active_tasks: int, total_cameras: int, 
                              websocket_connections: int, alerts_active: int):
        """
        Record business and operational metrics.
        
        Args:
            active_tasks: Number of active processing tasks
            total_cameras: Total number of cameras being processed
            websocket_connections: Number of active WebSocket connections
            alerts_active: Number of active alerts
        """
        try:
            current_time = time.time()
            
            self._add_metric_point('active_tasks', active_tasks, current_time)
            self._add_metric_point('total_cameras', total_cameras, current_time)
            self._add_metric_point('websocket_connections', websocket_connections, current_time)
            self._add_metric_point('alerts_active', alerts_active, current_time)
            
        except Exception as e:
            logger.error(f"Error recording business metrics: {e}")
    
    def _add_metric_point(self, metric_name: str, value: float, timestamp: float,
                         labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Add a metric data point to storage."""
        try:
            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                labels=labels or {},
                metadata=metadata or {}
            )
            self.metrics_data[metric_name].append(point)
            
        except Exception as e:
            logger.error(f"Error adding metric point for {metric_name}: {e}")
    
    def _update_quality_metrics(self):
        """Update quality and SLA metrics based on recent performance."""
        try:
            if not self.pipeline_metrics.total_frame_times:
                return
            
            # Calculate response time percentiles
            recent_times = list(self.pipeline_metrics.total_frame_times)[-100:]  # Last 100 frames
            if recent_times:
                self.quality_metrics.avg_response_time = statistics.mean(recent_times) * 1000  # ms
                sorted_times = sorted(recent_times)
                n = len(sorted_times)
                self.quality_metrics.p95_response_time = sorted_times[int(n * 0.95)] * 1000  # ms
                self.quality_metrics.p99_response_time = sorted_times[int(n * 0.99)] * 1000  # ms
            
            # Calculate throughput
            self.quality_metrics.frames_per_second = self.pipeline_metrics.pipeline_throughput
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            total_seconds = uptime_seconds
            self.quality_metrics.uptime_percentage = min(100.0, (uptime_seconds / total_seconds) * 100)
            
        except Exception as e:
            logger.error(f"Error updating quality metrics: {e}")
    
    async def _check_alert_conditions(self):
        """Check for alert conditions and generate alerts."""
        try:
            current_time = time.time()
            alerts_to_generate = []
            alerts_to_clear = []
            
            # Check CPU usage
            if self.resource_metrics.cpu_usage >= self.alert_thresholds['cpu_usage']['critical']:
                alerts_to_generate.append(('cpu_usage', AlertLevel.CRITICAL, 
                                         f"CPU usage critical: {self.resource_metrics.cpu_usage:.1f}%"))
            elif self.resource_metrics.cpu_usage >= self.alert_thresholds['cpu_usage']['warning']:
                alerts_to_generate.append(('cpu_usage', AlertLevel.WARNING,
                                         f"CPU usage high: {self.resource_metrics.cpu_usage:.1f}%"))
            else:
                alerts_to_clear.append('cpu_usage')
            
            # Check memory usage
            if self.resource_metrics.memory_usage >= self.alert_thresholds['memory_usage']['critical']:
                alerts_to_generate.append(('memory_usage', AlertLevel.CRITICAL,
                                         f"Memory usage critical: {self.resource_metrics.memory_usage:.1f}%"))
            elif self.resource_metrics.memory_usage >= self.alert_thresholds['memory_usage']['warning']:
                alerts_to_generate.append(('memory_usage', AlertLevel.WARNING,
                                         f"Memory usage high: {self.resource_metrics.memory_usage:.1f}%"))
            else:
                alerts_to_clear.append('memory_usage')
            
            # Check GPU usage (if available)
            if GPU_AVAILABLE and self.resource_metrics.gpu_usage > 0:
                if self.resource_metrics.gpu_usage >= self.alert_thresholds['gpu_usage']['critical']:
                    alerts_to_generate.append(('gpu_usage', AlertLevel.CRITICAL,
                                             f"GPU usage critical: {self.resource_metrics.gpu_usage:.1f}%"))
                elif self.resource_metrics.gpu_usage >= self.alert_thresholds['gpu_usage']['warning']:
                    alerts_to_generate.append(('gpu_usage', AlertLevel.WARNING,
                                             f"GPU usage high: {self.resource_metrics.gpu_usage:.1f}%"))
                else:
                    alerts_to_clear.append('gpu_usage')
            
            # Check response time
            if self.quality_metrics.avg_response_time >= self.alert_thresholds['response_time']['critical']:
                alerts_to_generate.append(('response_time', AlertLevel.CRITICAL,
                                         f"Response time critical: {self.quality_metrics.avg_response_time:.1f}ms"))
            elif self.quality_metrics.avg_response_time >= self.alert_thresholds['response_time']['warning']:
                alerts_to_generate.append(('response_time', AlertLevel.WARNING,
                                         f"Response time high: {self.quality_metrics.avg_response_time:.1f}ms"))
            else:
                alerts_to_clear.append('response_time')
            
            # Check throughput
            if self.quality_metrics.frames_per_second <= self.alert_thresholds['throughput']['critical']:
                alerts_to_generate.append(('throughput', AlertLevel.CRITICAL,
                                         f"Throughput critical: {self.quality_metrics.frames_per_second:.1f} fps"))
            elif self.quality_metrics.frames_per_second <= self.alert_thresholds['throughput']['warning']:
                alerts_to_generate.append(('throughput', AlertLevel.WARNING,
                                         f"Throughput low: {self.quality_metrics.frames_per_second:.1f} fps"))
            else:
                alerts_to_clear.append('throughput')
            
            # Generate alerts
            for alert_key, level, message in alerts_to_generate:
                await self._generate_alert(alert_key, level, message, current_time)
            
            # Clear resolved alerts
            for alert_key in alerts_to_clear:
                await self._clear_alert(alert_key, current_time)
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    async def _generate_alert(self, alert_key: str, level: AlertLevel, message: str, timestamp: float):
        """Generate an alert."""
        try:
            if alert_key not in self.active_alerts:
                alert = {
                    'key': alert_key,
                    'level': level.value,
                    'message': message,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'count': 1
                }
                
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert.copy())
                
                logger.warning(f"ALERT [{level.value.upper()}]: {message}")
            else:
                # Update existing alert
                self.active_alerts[alert_key]['last_seen'] = timestamp
                self.active_alerts[alert_key]['count'] += 1
                
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    async def _clear_alert(self, alert_key: str, timestamp: float):
        """Clear a resolved alert."""
        try:
            if alert_key in self.active_alerts:
                alert = self.active_alerts.pop(alert_key)
                alert['resolved_at'] = timestamp
                alert['duration'] = timestamp - alert['first_seen']
                
                logger.info(f"ALERT CLEARED: {alert['message']} (duration: {alert['duration']:.1f}s)")
                
        except Exception as e:
            logger.error(f"Error clearing alert: {e}")
    
    def _update_collection_stats(self, collection_duration: float):
        """Update collection performance statistics."""
        try:
            self.collection_stats['last_collection_time'] = time.time()
            
            # Update average collection duration
            alpha = 0.1  # EMA factor
            current_avg = self.collection_stats['avg_collection_duration']
            self.collection_stats['avg_collection_duration'] = (
                (1 - alpha) * current_avg + alpha * collection_duration
            )
            
        except Exception as e:
            logger.error(f"Error updating collection stats: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for monitoring dashboard."""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            return {
                'system_metrics': {
                    'uptime_seconds': uptime,
                    'cpu_usage_percent': self.resource_metrics.cpu_usage,
                    'memory_usage_percent': self.resource_metrics.memory_usage,
                    'memory_available_gb': self.resource_metrics.memory_available,
                    'gpu_usage_percent': self.resource_metrics.gpu_usage,
                    'gpu_memory_usage_percent': self.resource_metrics.gpu_memory_usage,
                    'disk_usage_percent': self.resource_metrics.disk_usage
                },
                'pipeline_metrics': {
                    'frames_processed': self.pipeline_metrics.frames_processed,
                    'current_fps': self.pipeline_metrics.pipeline_throughput,
                    'avg_detection_time_ms': statistics.mean(self.pipeline_metrics.detection_times[-100:]) * 1000 if self.pipeline_metrics.detection_times else 0,
                    'avg_total_time_ms': statistics.mean(self.pipeline_metrics.total_frame_times[-100:]) * 1000 if self.pipeline_metrics.total_frame_times else 0,
                    'total_detections': sum(self.pipeline_metrics.detection_counts),
                    'global_id_assignments': self.pipeline_metrics.global_id_assignments
                },
                'quality_metrics': {
                    'avg_response_time_ms': self.quality_metrics.avg_response_time,
                    'p95_response_time_ms': self.quality_metrics.p95_response_time,
                    'p99_response_time_ms': self.quality_metrics.p99_response_time,
                    'uptime_percentage': self.quality_metrics.uptime_percentage,
                    'error_rate_percent': self.quality_metrics.error_rate
                },
                'alerts': {
                    'active_alerts_count': len(self.active_alerts),
                    'active_alerts': list(self.active_alerts.values()),
                    'total_alerts_generated': len(self.alert_history)
                },
                'collection_stats': dict(self.collection_stats),
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def get_time_series_data(self, metric_name: str, start_time: float = None, 
                            end_time: float = None) -> List[Tuple[float, float]]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric_name: Name of the metric
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of (timestamp, value) tuples
        """
        try:
            if metric_name not in self.metrics_data:
                return []
            
            points = list(self.metrics_data[metric_name])
            
            # Filter by time range if specified
            if start_time is not None:
                points = [p for p in points if p.timestamp >= start_time]
            if end_time is not None:
                points = [p for p in points if p.timestamp <= end_time]
            
            return [(p.timestamp, p.value) for p in points]
            
        except Exception as e:
            logger.error(f"Error getting time series data for {metric_name}: {e}")
            return []
    
    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        try:
            lines = []
            current_time = time.time()
            
            # System metrics
            lines.append(f"# HELP spoton_cpu_usage_percent CPU usage percentage")
            lines.append(f"# TYPE spoton_cpu_usage_percent gauge")
            lines.append(f"spoton_cpu_usage_percent {self.resource_metrics.cpu_usage}")
            
            lines.append(f"# HELP spoton_memory_usage_percent Memory usage percentage")
            lines.append(f"# TYPE spoton_memory_usage_percent gauge")
            lines.append(f"spoton_memory_usage_percent {self.resource_metrics.memory_usage}")
            
            lines.append(f"# HELP spoton_pipeline_fps Pipeline processing rate in frames per second")
            lines.append(f"# TYPE spoton_pipeline_fps gauge")
            lines.append(f"spoton_pipeline_fps {self.pipeline_metrics.pipeline_throughput}")
            
            lines.append(f"# HELP spoton_frames_processed_total Total frames processed")
            lines.append(f"# TYPE spoton_frames_processed_total counter")
            lines.append(f"spoton_frames_processed_total {self.pipeline_metrics.frames_processed}")
            
            lines.append(f"# HELP spoton_active_alerts Number of active alerts")
            lines.append(f"# TYPE spoton_active_alerts gauge")
            lines.append(f"spoton_active_alerts {len(self.active_alerts)}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
            return f"# Error exporting metrics: {e}"


# Global metrics collector instance
production_metrics_collector = ProductionMetricsCollector()