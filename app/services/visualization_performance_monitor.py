"""
Visualization Performance Monitoring Service

Comprehensive performance monitoring for visualization components including:
- Real-time performance metrics collection
- Resource utilization tracking
- Bottleneck identification and analysis
- Performance trend analysis
- Automated alerting and optimization suggestions
- Historical performance data storage
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    metric_name: str
    value: float
    timestamp: datetime
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'unit': self.unit,
            'tags': self.tags
        }


@dataclass
class ComponentPerformanceProfile:
    """Performance profile for a specific component."""
    component_name: str
    metrics: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: datetime = field(default_factory=datetime.utcnow)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a metric to the profile."""
        self.metrics.append(metric)
        self.last_updated = datetime.utcnow()
    
    def get_recent_metrics(self, minutes: int = 10) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    def get_average_value(self, metric_name: str, minutes: int = 10) -> Optional[float]:
        """Get average value for a metric over the last N minutes."""
        recent_metrics = [
            m for m in self.get_recent_metrics(minutes)
            if m.metric_name == metric_name
        ]
        if not recent_metrics:
            return None
        
        return sum(m.value for m in recent_metrics) / len(recent_metrics)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts based on thresholds."""
        alerts = []
        
        for metric_name, threshold in self.alert_thresholds.items():
            avg_value = self.get_average_value(metric_name, minutes=5)
            if avg_value is not None and avg_value > threshold:
                alerts.append({
                    'component': self.component_name,
                    'metric': metric_name,
                    'value': avg_value,
                    'threshold': threshold,
                    'severity': 'warning' if avg_value < threshold * 1.5 else 'critical',
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return alerts


class VisualizationPerformanceMonitor:
    """Comprehensive performance monitoring service for visualization components."""
    
    def __init__(self):
        # Component performance profiles
        self.component_profiles: Dict[str, ComponentPerformanceProfile] = {}
        
        # System resource monitoring
        self.system_metrics: deque = deque(maxlen=1000)
        self.gpu_metrics: deque = deque(maxlen=1000)
        
        # Performance alerts
        self.alert_handlers: Set[Callable] = set()
        self.alert_history: deque = deque(maxlen=500)
        
        # Monitoring configuration
        self.monitoring_interval = 1.0  # seconds
        self.enable_gpu_monitoring = True
        self.enable_detailed_profiling = True
        
        # Background monitoring tasks
        self._monitoring_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {
            'frame_processing': {
                'processing_time_ms': 50.0,
                'memory_usage_mb': 100.0,
                'gpu_utilization': 70.0
            },
            'websocket_messaging': {
                'message_latency_ms': 10.0,
                'throughput_msg_per_sec': 100.0,
                'connection_count': 50
            },
            'analytics_computation': {
                'computation_time_ms': 100.0,
                'memory_usage_mb': 200.0,
                'cpu_utilization': 50.0
            }
        }
        
        # Initialize component profiles with default thresholds
        self._initialize_default_profiles()
        
        logger.info("VisualizationPerformanceMonitor initialized")
    
    async def start_monitoring(self):
        """Start all monitoring tasks."""
        if self._running:
            logger.warning("Performance monitoring already running")
            return
        
        self._running = True
        
        # Start background monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_system_resources()),
            asyncio.create_task(self._monitor_component_performance()),
            asyncio.create_task(self._check_performance_alerts()),
            asyncio.create_task(self._analyze_performance_trends())
        ]
        
        if self.enable_gpu_monitoring:
            tasks.append(asyncio.create_task(self._monitor_gpu_resources()))
        
        self._monitoring_tasks.update(tasks)
        logger.info(f"Started {len(tasks)} performance monitoring tasks")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks."""
        self._running = False
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        logger.info("Performance monitoring stopped")
    
    # --- Metric Collection ---
    
    def record_metric(
        self,
        component_name: str,
        metric_name: str,
        value: float,
        unit: str = "count",
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a performance metric for a component."""
        try:
            # Get or create component profile
            if component_name not in self.component_profiles:
                self.component_profiles[component_name] = ComponentPerformanceProfile(
                    component_name=component_name
                )
            
            # Create metric
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=value,
                timestamp=datetime.utcnow(),
                unit=unit,
                tags=tags or {}
            )
            
            # Add to profile
            self.component_profiles[component_name].add_metric(metric)
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    def record_timing(self, component_name: str, operation_name: str):
        """Context manager for timing operations."""
        return TimingContext(self, component_name, operation_name)
    
    def record_frame_processing_metrics(
        self,
        processing_time_ms: float,
        frame_count: int,
        memory_usage_mb: float,
        gpu_utilization: Optional[float] = None
    ):
        """Record frame processing performance metrics."""
        component = "frame_processing"
        
        self.record_metric(component, "processing_time_ms", processing_time_ms, "ms")
        self.record_metric(component, "frame_count", frame_count, "count")
        self.record_metric(component, "memory_usage_mb", memory_usage_mb, "MB")
        
        if gpu_utilization is not None:
            self.record_metric(component, "gpu_utilization", gpu_utilization, "percent")
        
        # Calculate FPS
        if processing_time_ms > 0:
            fps = 1000.0 / processing_time_ms
            self.record_metric(component, "fps", fps, "fps")
    
    def record_websocket_metrics(
        self,
        message_count: int,
        latency_ms: float,
        connection_count: int,
        bytes_transmitted: int
    ):
        """Record WebSocket performance metrics."""
        component = "websocket_messaging"
        
        self.record_metric(component, "message_count", message_count, "count")
        self.record_metric(component, "message_latency_ms", latency_ms, "ms")
        self.record_metric(component, "connection_count", connection_count, "count")
        self.record_metric(component, "bytes_transmitted", bytes_transmitted, "bytes")
        
        # Calculate throughput
        if latency_ms > 0:
            throughput = message_count / (latency_ms / 1000.0)
            self.record_metric(component, "throughput_msg_per_sec", throughput, "msg/sec")
    
    def record_analytics_metrics(
        self,
        computation_time_ms: float,
        data_points_processed: int,
        memory_usage_mb: float,
        cpu_utilization: float
    ):
        """Record analytics computation performance metrics."""
        component = "analytics_computation"
        
        self.record_metric(component, "computation_time_ms", computation_time_ms, "ms")
        self.record_metric(component, "data_points_processed", data_points_processed, "count")
        self.record_metric(component, "memory_usage_mb", memory_usage_mb, "MB")
        self.record_metric(component, "cpu_utilization", cpu_utilization, "percent")
        
        # Calculate processing rate
        if computation_time_ms > 0:
            processing_rate = data_points_processed / (computation_time_ms / 1000.0)
            self.record_metric(component, "processing_rate_per_sec", processing_rate, "points/sec")
    
    # --- Background Monitoring Tasks ---
    
    async def _monitor_system_resources(self):
        """Monitor system-level resource utilization."""
        while self._running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_total_gb = memory.total / (1024**3)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                
                # Network metrics
                network = psutil.net_io_counters()
                
                # Record system metrics
                system_metric = {
                    'timestamp': datetime.utcnow(),
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_percent': memory_percent,
                    'memory_used_gb': memory_used_gb,
                    'memory_total_gb': memory_total_gb,
                    'disk_percent': disk_percent,
                    'network_bytes_sent': network.bytes_sent,
                    'network_bytes_recv': network.bytes_recv
                }
                
                self.system_metrics.append(system_metric)
                
                # Record as individual metrics
                self.record_metric("system", "cpu_percent", cpu_percent, "percent")
                self.record_metric("system", "memory_percent", memory_percent, "percent")
                self.record_metric("system", "memory_used_gb", memory_used_gb, "GB")
                self.record_metric("system", "disk_percent", disk_percent, "percent")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)
    
    async def _monitor_gpu_resources(self):
        """Monitor GPU resource utilization."""
        try:
            import GPUtil
        except ImportError:
            logger.warning("GPUtil not available, skipping GPU monitoring")
            return
        
        while self._running:
            try:
                gpus = GPUtil.getGPUs()
                
                for i, gpu in enumerate(gpus):
                    gpu_metric = {
                        'timestamp': datetime.utcnow(),
                        'gpu_id': i,
                        'gpu_name': gpu.name,
                        'gpu_load': gpu.load * 100,
                        'gpu_memory_used': gpu.memoryUsed,
                        'gpu_memory_total': gpu.memoryTotal,
                        'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'gpu_temperature': gpu.temperature
                    }
                    
                    self.gpu_metrics.append(gpu_metric)
                    
                    # Record as individual metrics
                    tags = {'gpu_id': str(i), 'gpu_name': gpu.name}
                    self.record_metric("gpu", "gpu_load", gpu.load * 100, "percent", tags)
                    self.record_metric("gpu", "gpu_memory_used", gpu.memoryUsed, "MB", tags)
                    self.record_metric("gpu", "gpu_memory_percent", 
                                     (gpu.memoryUsed / gpu.memoryTotal) * 100, "percent", tags)
                    self.record_metric("gpu", "gpu_temperature", gpu.temperature, "celsius", tags)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring GPU resources: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)
    
    async def _monitor_component_performance(self):
        """Monitor component-specific performance metrics."""
        while self._running:
            try:
                # Check each component profile for performance issues
                for component_name, profile in self.component_profiles.items():
                    recent_metrics = profile.get_recent_metrics(minutes=5)
                    
                    if recent_metrics:
                        # Calculate performance statistics
                        avg_values = defaultdict(list)
                        for metric in recent_metrics:
                            avg_values[metric.metric_name].append(metric.value)
                        
                        # Record aggregated metrics
                        for metric_name, values in avg_values.items():
                            if values:
                                avg_value = sum(values) / len(values)
                                max_value = max(values)
                                min_value = min(values)
                                
                                # Record aggregated metrics
                                tags = {'component': component_name}
                                self.record_metric("performance_summary", f"{metric_name}_avg", 
                                                 avg_value, "average", tags)
                                self.record_metric("performance_summary", f"{metric_name}_max", 
                                                 max_value, "maximum", tags)
                                self.record_metric("performance_summary", f"{metric_name}_min", 
                                                 min_value, "minimum", tags)
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring component performance: {e}")
                await asyncio.sleep(10.0)
    
    async def _check_performance_alerts(self):
        """Check for performance alerts and notify handlers."""
        while self._running:
            try:
                all_alerts = []
                
                # Check each component for alerts
                for component_name, profile in self.component_profiles.items():
                    component_alerts = profile.check_alerts()
                    all_alerts.extend(component_alerts)
                
                # Check system-level alerts
                system_alerts = self._check_system_alerts()
                all_alerts.extend(system_alerts)
                
                # Process alerts
                for alert in all_alerts:
                    await self._handle_alert(alert)
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error checking performance alerts: {e}")
                await asyncio.sleep(30.0)
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and generate insights."""
        while self._running:
            try:
                # Analyze trends every 5 minutes
                await asyncio.sleep(300.0)
                
                # Analyze component performance trends
                trends = self._calculate_performance_trends()
                
                # Log significant trends
                for component, trend_data in trends.items():
                    for metric, trend in trend_data.items():
                        if abs(trend.get('change_percent', 0)) > 20:  # 20% change threshold
                            logger.info(
                                f"Performance trend detected - {component}.{metric}: "
                                f"{trend['change_percent']:.1f}% change over last hour"
                            )
                
            except Exception as e:
                logger.error(f"Error analyzing performance trends: {e}")
                await asyncio.sleep(300.0)
    
    # --- Alert Handling ---
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.add(handler)
    
    def remove_alert_handler(self, handler: Callable):
        """Remove an alert handler function."""
        self.alert_handlers.discard(handler)
    
    async def _handle_alert(self, alert: Dict[str, Any]):
        """Handle a performance alert."""
        try:
            # Add to alert history
            self.alert_history.append(alert)
            
            # Log alert
            logger.warning(
                f"Performance alert: {alert['component']}.{alert['metric']} = "
                f"{alert['value']:.2f} (threshold: {alert['threshold']:.2f})"
            )
            
            # Notify handlers
            for handler in self.alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        # Run synchronous handler in thread pool
                        await asyncio.get_event_loop().run_in_executor(
                            None, handler, alert
                        )
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    def _check_system_alerts(self) -> List[Dict[str, Any]]:
        """Check for system-level performance alerts."""
        alerts = []
        
        try:
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                
                # CPU alert
                if latest_system['cpu_percent'] > 80:
                    alerts.append({
                        'component': 'system',
                        'metric': 'cpu_percent',
                        'value': latest_system['cpu_percent'],
                        'threshold': 80,
                        'severity': 'warning',
                        'timestamp': latest_system['timestamp'].isoformat()
                    })
                
                # Memory alert
                if latest_system['memory_percent'] > 85:
                    alerts.append({
                        'component': 'system',
                        'metric': 'memory_percent',
                        'value': latest_system['memory_percent'],
                        'threshold': 85,
                        'severity': 'warning',
                        'timestamp': latest_system['timestamp'].isoformat()
                    })
                
                # Disk alert
                if latest_system['disk_percent'] > 90:
                    alerts.append({
                        'component': 'system',
                        'metric': 'disk_percent',
                        'value': latest_system['disk_percent'],
                        'threshold': 90,
                        'severity': 'critical',
                        'timestamp': latest_system['timestamp'].isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")
        
        return alerts
    
    # --- Performance Analysis ---
    
    def _calculate_performance_trends(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate performance trends for all components."""
        trends = {}
        
        try:
            for component_name, profile in self.component_profiles.items():
                component_trends = {}
                
                # Get metrics from last hour and last 10 minutes
                hour_metrics = profile.get_recent_metrics(minutes=60)
                recent_metrics = profile.get_recent_metrics(minutes=10)
                
                if len(hour_metrics) > 10 and len(recent_metrics) > 5:
                    # Group metrics by name
                    hour_by_metric = defaultdict(list)
                    recent_by_metric = defaultdict(list)
                    
                    for metric in hour_metrics:
                        hour_by_metric[metric.metric_name].append(metric.value)
                    
                    for metric in recent_metrics:
                        recent_by_metric[metric.metric_name].append(metric.value)
                    
                    # Calculate trends
                    for metric_name in hour_by_metric.keys():
                        if metric_name in recent_by_metric:
                            hour_avg = sum(hour_by_metric[metric_name]) / len(hour_by_metric[metric_name])
                            recent_avg = sum(recent_by_metric[metric_name]) / len(recent_by_metric[metric_name])
                            
                            if hour_avg > 0:
                                change_percent = ((recent_avg - hour_avg) / hour_avg) * 100
                                
                                component_trends[metric_name] = {
                                    'hour_average': hour_avg,
                                    'recent_average': recent_avg,
                                    'change_percent': change_percent,
                                    'trend': 'increasing' if change_percent > 5 else 'decreasing' if change_percent < -5 else 'stable'
                                }
                
                if component_trends:
                    trends[component_name] = component_trends
        
        except Exception as e:
            logger.error(f"Error calculating performance trends: {e}")
        
        return trends
    
    # --- Performance Reports ---
    
    def get_performance_report(
        self,
        component_name: Optional[str] = None,
        minutes: int = 60
    ) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            report = {
                'report_timestamp': datetime.utcnow().isoformat(),
                'time_window_minutes': minutes,
                'components': {}
            }
            
            # Filter components
            if component_name:
                components = {component_name: self.component_profiles.get(component_name)}
                if not components[component_name]:
                    return {'error': f'Component {component_name} not found'}
            else:
                components = self.component_profiles
            
            # Generate report for each component
            for comp_name, profile in components.items():
                if not profile:
                    continue
                
                recent_metrics = profile.get_recent_metrics(minutes)
                
                if recent_metrics:
                    # Group metrics by name
                    metrics_by_name = defaultdict(list)
                    for metric in recent_metrics:
                        metrics_by_name[metric.metric_name].append(metric.value)
                    
                    # Calculate statistics
                    component_stats = {}
                    for metric_name, values in metrics_by_name.items():
                        component_stats[metric_name] = {
                            'count': len(values),
                            'average': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values),
                            'latest': values[-1] if values else None
                        }
                    
                    report['components'][comp_name] = {
                        'metrics': component_stats,
                        'last_updated': profile.last_updated.isoformat(),
                        'alert_count': len(profile.check_alerts())
                    }
            
            # Add system metrics
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                report['system_resources'] = {
                    'cpu_percent': latest_system['cpu_percent'],
                    'memory_percent': latest_system['memory_percent'],
                    'memory_used_gb': latest_system['memory_used_gb'],
                    'disk_percent': latest_system['disk_percent'],
                    'timestamp': latest_system['timestamp'].isoformat()
                }
            
            # Add GPU metrics if available
            if self.gpu_metrics:
                latest_gpu = self.gpu_metrics[-1]
                report['gpu_resources'] = {
                    'gpu_load': latest_gpu['gpu_load'],
                    'gpu_memory_percent': latest_gpu['gpu_memory_percent'],
                    'gpu_temperature': latest_gpu['gpu_temperature'],
                    'timestamp': latest_gpu['timestamp'].isoformat()
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def get_performance_baselines(self) -> Dict[str, Dict[str, float]]:
        """Get performance baselines for comparison."""
        return self.performance_baselines.copy()
    
    def update_performance_baseline(
        self,
        component_name: str,
        metric_name: str,
        baseline_value: float
    ):
        """Update performance baseline for a component metric."""
        if component_name not in self.performance_baselines:
            self.performance_baselines[component_name] = {}
        
        self.performance_baselines[component_name][metric_name] = baseline_value
        logger.info(f"Updated baseline for {component_name}.{metric_name}: {baseline_value}")
    
    # --- Service Management ---
    
    def _initialize_default_profiles(self):
        """Initialize default component profiles with alert thresholds."""
        default_profiles = {
            'frame_processing': {
                'processing_time_ms': 100.0,
                'memory_usage_mb': 200.0,
                'gpu_utilization': 85.0
            },
            'websocket_messaging': {
                'message_latency_ms': 50.0,
                'connection_count': 100
            },
            'analytics_computation': {
                'computation_time_ms': 200.0,
                'memory_usage_mb': 300.0,
                'cpu_utilization': 80.0
            }
        }
        
        for component_name, thresholds in default_profiles.items():
            profile = ComponentPerformanceProfile(
                component_name=component_name,
                alert_thresholds=thresholds
            )
            self.component_profiles[component_name] = profile
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get monitoring service status."""
        return {
            'service_name': 'VisualizationPerformanceMonitor',
            'monitoring_active': self._running,
            'monitoring_interval_seconds': self.monitoring_interval,
            'components_monitored': len(self.component_profiles),
            'active_tasks': len(self._monitoring_tasks),
            'system_metrics_count': len(self.system_metrics),
            'gpu_metrics_count': len(self.gpu_metrics),
            'alert_handlers_count': len(self.alert_handlers),
            'recent_alerts_count': len(self.alert_history),
            'gpu_monitoring_enabled': self.enable_gpu_monitoring,
            'detailed_profiling_enabled': self.enable_detailed_profiling
        }
    
    async def reset_metrics(self):
        """Reset all collected metrics."""
        try:
            # Clear component metrics
            for profile in self.component_profiles.values():
                profile.metrics.clear()
                profile.last_updated = datetime.utcnow()
            
            # Clear system metrics
            self.system_metrics.clear()
            self.gpu_metrics.clear()
            
            # Clear alert history
            self.alert_history.clear()
            
            logger.info("All performance metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting metrics: {e}")


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: VisualizationPerformanceMonitor, component_name: str, operation_name: str):
        self.monitor = monitor
        self.component_name = component_name
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.monitor.record_metric(
                self.component_name,
                f"{self.operation_name}_time_ms",
                duration_ms,
                "ms"
            )


# Global monitor instance
_performance_monitor: Optional[VisualizationPerformanceMonitor] = None


def get_performance_monitor() -> Optional[VisualizationPerformanceMonitor]:
    """Get the global performance monitor instance."""
    return _performance_monitor


def initialize_performance_monitor() -> VisualizationPerformanceMonitor:
    """Initialize the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = VisualizationPerformanceMonitor()
    return _performance_monitor


def record_performance_metric(
    component_name: str,
    metric_name: str,
    value: float,
    unit: str = "count"
):
    """Utility function to record a performance metric."""
    monitor = get_performance_monitor()
    if monitor:
        monitor.record_metric(component_name, metric_name, value, unit)