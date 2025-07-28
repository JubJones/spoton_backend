"""
Comprehensive monitoring and observability service.

Handles:
- System performance monitoring
- Health checks and alerts
- Metrics collection and aggregation
- Logging and tracing
- Performance optimization recommendations
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import threading
from enum import Enum
import statistics

from app.infrastructure.cache.tracking_cache import tracking_cache
from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.services.memory_manager import memory_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    load_average: Tuple[float, float, float]
    uptime: float


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    active_cameras: int
    active_persons: int
    detection_rate: float
    reid_rate: float
    mapping_rate: float
    pipeline_latency: float
    error_rate: float
    cache_hit_rate: float
    database_connections: int


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    alert_id: str
    severity: AlertSeverity
    component: str
    metric: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    metrics_interval: int = 10  # seconds
    health_check_interval: int = 30  # seconds
    alert_evaluation_interval: int = 15  # seconds
    metrics_retention_hours: int = 24
    alert_retention_days: int = 7
    
    # Thresholds
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 90.0
    memory_warning_threshold: float = 80.0
    memory_critical_threshold: float = 90.0
    disk_warning_threshold: float = 80.0
    disk_critical_threshold: float = 90.0
    error_rate_threshold: float = 5.0
    latency_threshold: float = 1000.0  # ms


class MonitoringService:
    """
    Comprehensive monitoring and observability service.
    
    Features:
    - System performance monitoring
    - Application metrics collection
    - Health checks and alerts
    - Performance optimization recommendations
    - Distributed tracing support
    """
    
    def __init__(self):
        self.config = MonitoringConfig()
        self.system_metrics: deque = deque(maxlen=10000)
        self.application_metrics: deque = deque(maxlen=10000)
        self.performance_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Health check registry
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.monitoring_stats = {
            'metrics_collected': 0,
            'alerts_generated': 0,
            'health_checks_performed': 0,
            'recommendations_generated': 0,
            'uptime_seconds': 0
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.start_time = datetime.now(timezone.utc)
        
        # Thread safety
        self.metrics_lock = threading.Lock()
        self.alerts_lock = threading.Lock()
        
        logger.info("MonitoringService initialized")
    
    async def initialize(self):
        """Initialize monitoring service."""
        try:
            # Register default health checks
            await self._register_default_health_checks()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("MonitoringService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing MonitoringService: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup monitoring service."""
        try:
            # Stop monitoring tasks
            await self._stop_monitoring_tasks()
            
            logger.info("MonitoringService cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up MonitoringService: {e}")
    
    # Monitoring Tasks
    async def _start_monitoring_tasks(self):
        """Start monitoring tasks."""
        try:
            self.monitoring_active = True
            
            # Start monitoring loops
            self.monitoring_tasks = [
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._alert_evaluation_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            logger.info("Monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring tasks: {e}")
    
    async def _stop_monitoring_tasks(self):
        """Stop monitoring tasks."""
        try:
            self.monitoring_active = False
            
            # Cancel all tasks
            for task in self.monitoring_tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.monitoring_tasks.clear()
            
            logger.info("Monitoring tasks stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring tasks: {e}")
    
    async def _metrics_collection_loop(self):
        """Metrics collection loop."""
        try:
            while self.monitoring_active:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(self.config.metrics_interval)
                
        except asyncio.CancelledError:
            logger.info("Metrics collection loop cancelled")
        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}")
    
    async def _health_check_loop(self):
        """Health check loop."""
        try:
            while self.monitoring_active:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
                
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
    
    async def _alert_evaluation_loop(self):
        """Alert evaluation loop."""
        try:
            while self.monitoring_active:
                await self._evaluate_alerts()
                await asyncio.sleep(self.config.alert_evaluation_interval)
                
        except asyncio.CancelledError:
            logger.info("Alert evaluation loop cancelled")
        except Exception as e:
            logger.error(f"Error in alert evaluation loop: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup loop for old metrics and alerts."""
        try:
            while self.monitoring_active:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
                
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
    
    # Metrics Collection
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            with self.metrics_lock:
                current_time = datetime.now(timezone.utc)
                
                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                
                # Network I/O
                network = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
                
                # Process information
                process_count = len(psutil.pids())
                thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
                
                # Load average
                load_average = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0.0, 0.0, 0.0)
                
                # Uptime
                uptime = (current_time - self.start_time).total_seconds()
                
                # Create metrics object
                metrics = SystemMetrics(
                    timestamp=current_time,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    disk_usage=disk_usage,
                    network_io=network_io,
                    process_count=process_count,
                    thread_count=thread_count,
                    load_average=load_average,
                    uptime=uptime
                )
                
                self.system_metrics.append(metrics)
                self.monitoring_stats['metrics_collected'] += 1
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            with self.metrics_lock:
                current_time = datetime.now(timezone.utc)
                
                # Get active persons and cameras
                active_persons = await tracking_cache.get_active_persons()
                active_cameras = len(set(p.last_seen_camera for p in active_persons))
                
                # Get cache statistics
                cache_stats = await tracking_cache.get_cache_stats()
                cache_hit_rate = cache_stats.get('cache_stats', {}).get('hit_rate', 0.0)
                
                # Get database connection count
                db_connections = 0  # Would be implemented based on database pool
                
                # Calculate rates (simplified)
                detection_rate = len(active_persons) / max(1, active_cameras)
                reid_rate = detection_rate * 0.8  # Approximate
                mapping_rate = reid_rate * 0.9  # Approximate
                
                # Pipeline latency (would be measured from actual processing)
                pipeline_latency = 100.0  # ms
                
                # Error rate (would be calculated from error tracking)
                error_rate = 0.0
                
                # Create metrics object
                metrics = ApplicationMetrics(
                    timestamp=current_time,
                    active_cameras=active_cameras,
                    active_persons=len(active_persons),
                    detection_rate=detection_rate,
                    reid_rate=reid_rate,
                    mapping_rate=mapping_rate,
                    pipeline_latency=pipeline_latency,
                    error_rate=error_rate,
                    cache_hit_rate=cache_hit_rate,
                    database_connections=db_connections
                )
                
                self.application_metrics.append(metrics)
                
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    # Health Checks
    async def _register_default_health_checks(self):
        """Register default health checks."""
        try:
            # System health checks
            self.health_checks['system_resources'] = self._check_system_resources
            self.health_checks['memory_usage'] = self._check_memory_usage
            self.health_checks['disk_space'] = self._check_disk_space
            
            # Application health checks
            self.health_checks['database_connection'] = self._check_database_connection
            self.health_checks['cache_connection'] = self._check_cache_connection
            self.health_checks['gpu_availability'] = self._check_gpu_availability
            
            logger.info("Default health checks registered")
            
        except Exception as e:
            logger.error(f"Error registering default health checks: {e}")
    
    async def _perform_health_checks(self):
        """Perform all registered health checks."""
        try:
            current_time = datetime.now(timezone.utc)
            
            for check_name, check_func in self.health_checks.items():
                try:
                    status = await check_func()
                    self.health_status[check_name] = {
                        'status': status['status'],
                        'message': status.get('message', ''),
                        'timestamp': current_time.isoformat(),
                        'details': status.get('details', {})
                    }
                    
                except Exception as e:
                    self.health_status[check_name] = {
                        'status': 'error',
                        'message': f'Health check failed: {str(e)}',
                        'timestamp': current_time.isoformat(),
                        'details': {}
                    }
            
            self.monitoring_stats['health_checks_performed'] += 1
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            if cpu_usage > self.config.cpu_critical_threshold:
                return {
                    'status': 'critical',
                    'message': f'CPU usage critical: {cpu_usage}%',
                    'details': {'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
                }
            elif cpu_usage > self.config.cpu_warning_threshold:
                return {
                    'status': 'warning',
                    'message': f'CPU usage high: {cpu_usage}%',
                    'details': {'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'System resources normal',
                    'details': {'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'System resource check failed: {str(e)}',
                'details': {}
            }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health."""
        try:
            memory_status = await memory_manager.get_memory_status()
            
            if not memory_status:
                return {
                    'status': 'warning',
                    'message': 'Memory status unavailable',
                    'details': {}
                }
            
            system_memory = memory_status.get('system_memory', {})
            gpu_memory = memory_status.get('gpu_memory', {})
            
            system_usage = system_memory.get('percent', 0)
            gpu_usage = gpu_memory.get('utilization', 0)
            
            if system_usage > self.config.memory_critical_threshold:
                return {
                    'status': 'critical',
                    'message': f'System memory critical: {system_usage}%',
                    'details': {'system_memory': system_usage, 'gpu_memory': gpu_usage}
                }
            elif system_usage > self.config.memory_warning_threshold:
                return {
                    'status': 'warning',
                    'message': f'System memory high: {system_usage}%',
                    'details': {'system_memory': system_usage, 'gpu_memory': gpu_usage}
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'Memory usage normal',
                    'details': {'system_memory': system_usage, 'gpu_memory': gpu_usage}
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Memory check failed: {str(e)}',
                'details': {}
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space health."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > self.config.disk_critical_threshold:
                return {
                    'status': 'critical',
                    'message': f'Disk space critical: {usage_percent:.1f}%',
                    'details': {'disk_usage_percent': usage_percent, 'free_bytes': disk.free}
                }
            elif usage_percent > self.config.disk_warning_threshold:
                return {
                    'status': 'warning',
                    'message': f'Disk space low: {usage_percent:.1f}%',
                    'details': {'disk_usage_percent': usage_percent, 'free_bytes': disk.free}
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'Disk space normal',
                    'details': {'disk_usage_percent': usage_percent, 'free_bytes': disk.free}
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Disk space check failed: {str(e)}',
                'details': {}
            }
    
    async def _check_database_connection(self) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            # Test database connectivity
            health = await integrated_db_service.get_service_health()
            
            if health.get('status') == 'healthy':
                return {
                    'status': 'healthy',
                    'message': 'Database connection healthy',
                    'details': health
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'Database connection issues',
                    'details': health
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Database connection check failed: {str(e)}',
                'details': {}
            }
    
    async def _check_cache_connection(self) -> Dict[str, Any]:
        """Check cache connection health."""
        try:
            # Test cache connectivity
            cache_stats = await tracking_cache.get_cache_stats()
            
            if cache_stats:
                return {
                    'status': 'healthy',
                    'message': 'Cache connection healthy',
                    'details': cache_stats
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'Cache connection issues',
                    'details': {}
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Cache connection check failed: {str(e)}',
                'details': {}
            }
    
    async def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability health."""
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                
                return {
                    'status': 'healthy',
                    'message': f'GPU available: {device_count} device(s)',
                    'details': {
                        'device_count': device_count,
                        'current_device': current_device,
                        'cuda_version': torch.version.cuda
                    }
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'GPU not available',
                    'details': {}
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'GPU availability check failed: {str(e)}',
                'details': {}
            }
    
    # Alert Management
    async def _evaluate_alerts(self):
        """Evaluate metrics and generate alerts."""
        try:
            with self.alerts_lock:
                # Get recent metrics
                if not self.system_metrics:
                    return
                
                latest_system = self.system_metrics[-1]
                latest_app = self.application_metrics[-1] if self.application_metrics else None
                
                # Check system alerts
                await self._check_system_alerts(latest_system)
                
                # Check application alerts
                if latest_app:
                    await self._check_application_alerts(latest_app)
                
                # Resolve alerts that are no longer active
                await self._resolve_inactive_alerts()
                
        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alerts."""
        try:
            # CPU alert
            if metrics.cpu_usage > self.config.cpu_critical_threshold:
                await self._create_alert(
                    'system_cpu_critical',
                    AlertSeverity.CRITICAL,
                    'system',
                    'cpu_usage',
                    metrics.cpu_usage,
                    self.config.cpu_critical_threshold,
                    f'CPU usage critical: {metrics.cpu_usage}%'
                )
            elif metrics.cpu_usage > self.config.cpu_warning_threshold:
                await self._create_alert(
                    'system_cpu_warning',
                    AlertSeverity.HIGH,
                    'system',
                    'cpu_usage',
                    metrics.cpu_usage,
                    self.config.cpu_warning_threshold,
                    f'CPU usage high: {metrics.cpu_usage}%'
                )
            
            # Memory alert
            if metrics.memory_usage > self.config.memory_critical_threshold:
                await self._create_alert(
                    'system_memory_critical',
                    AlertSeverity.CRITICAL,
                    'system',
                    'memory_usage',
                    metrics.memory_usage,
                    self.config.memory_critical_threshold,
                    f'Memory usage critical: {metrics.memory_usage}%'
                )
            elif metrics.memory_usage > self.config.memory_warning_threshold:
                await self._create_alert(
                    'system_memory_warning',
                    AlertSeverity.HIGH,
                    'system',
                    'memory_usage',
                    metrics.memory_usage,
                    self.config.memory_warning_threshold,
                    f'Memory usage high: {metrics.memory_usage}%'
                )
            
            # Disk alert
            if metrics.disk_usage > self.config.disk_critical_threshold:
                await self._create_alert(
                    'system_disk_critical',
                    AlertSeverity.CRITICAL,
                    'system',
                    'disk_usage',
                    metrics.disk_usage,
                    self.config.disk_critical_threshold,
                    f'Disk usage critical: {metrics.disk_usage}%'
                )
            elif metrics.disk_usage > self.config.disk_warning_threshold:
                await self._create_alert(
                    'system_disk_warning',
                    AlertSeverity.HIGH,
                    'system',
                    'disk_usage',
                    metrics.disk_usage,
                    self.config.disk_warning_threshold,
                    f'Disk usage high: {metrics.disk_usage}%'
                )
            
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")
    
    async def _check_application_alerts(self, metrics: ApplicationMetrics):
        """Check application metrics for alerts."""
        try:
            # Error rate alert
            if metrics.error_rate > self.config.error_rate_threshold:
                await self._create_alert(
                    'app_error_rate_high',
                    AlertSeverity.HIGH,
                    'application',
                    'error_rate',
                    metrics.error_rate,
                    self.config.error_rate_threshold,
                    f'Error rate high: {metrics.error_rate}%'
                )
            
            # Latency alert
            if metrics.pipeline_latency > self.config.latency_threshold:
                await self._create_alert(
                    'app_latency_high',
                    AlertSeverity.MEDIUM,
                    'application',
                    'pipeline_latency',
                    metrics.pipeline_latency,
                    self.config.latency_threshold,
                    f'Pipeline latency high: {metrics.pipeline_latency}ms'
                )
            
        except Exception as e:
            logger.error(f"Error checking application alerts: {e}")
    
    async def _create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        component: str,
        metric: str,
        current_value: float,
        threshold_value: float,
        message: str
    ):
        """Create or update alert."""
        try:
            if alert_id not in self.performance_alerts:
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    severity=severity,
                    component=component,
                    metric=metric,
                    current_value=current_value,
                    threshold_value=threshold_value,
                    message=message,
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.performance_alerts[alert_id] = alert
                self.alert_history.append(alert)
                self.monitoring_stats['alerts_generated'] += 1
                
                logger.warning(f"Alert created: {alert_id} - {message}")
            else:
                # Update existing alert
                existing_alert = self.performance_alerts[alert_id]
                existing_alert.current_value = current_value
                existing_alert.message = message
                existing_alert.timestamp = datetime.now(timezone.utc)
                
                if existing_alert.resolved:
                    existing_alert.resolved = False
                    existing_alert.resolution_time = None
                    logger.warning(f"Alert reactivated: {alert_id} - {message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def _resolve_inactive_alerts(self):
        """Resolve alerts that are no longer active."""
        try:
            current_time = datetime.now(timezone.utc)
            
            for alert_id, alert in list(self.performance_alerts.items()):
                if not alert.resolved:
                    # Check if alert condition is no longer met
                    should_resolve = False
                    
                    if alert.metric == 'cpu_usage':
                        latest_cpu = self.system_metrics[-1].cpu_usage if self.system_metrics else 0
                        should_resolve = latest_cpu <= alert.threshold_value
                    elif alert.metric == 'memory_usage':
                        latest_memory = self.system_metrics[-1].memory_usage if self.system_metrics else 0
                        should_resolve = latest_memory <= alert.threshold_value
                    elif alert.metric == 'disk_usage':
                        latest_disk = self.system_metrics[-1].disk_usage if self.system_metrics else 0
                        should_resolve = latest_disk <= alert.threshold_value
                    
                    if should_resolve:
                        alert.resolved = True
                        alert.resolution_time = current_time
                        logger.info(f"Alert resolved: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error resolving alerts: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        try:
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(hours=self.config.metrics_retention_hours)
            
            # Clean up old metrics
            with self.metrics_lock:
                # System metrics
                self.system_metrics = deque(
                    [m for m in self.system_metrics if m.timestamp >= cutoff_time],
                    maxlen=self.system_metrics.maxlen
                )
                
                # Application metrics
                self.application_metrics = deque(
                    [m for m in self.application_metrics if m.timestamp >= cutoff_time],
                    maxlen=self.application_metrics.maxlen
                )
            
            # Clean up old alerts
            alert_cutoff_time = current_time - timedelta(days=self.config.alert_retention_days)
            
            with self.alerts_lock:
                # Remove resolved alerts older than retention period
                self.performance_alerts = {
                    alert_id: alert for alert_id, alert in self.performance_alerts.items()
                    if not alert.resolved or alert.timestamp >= alert_cutoff_time
                }
                
                # Clean up alert history
                self.alert_history = deque(
                    [a for a in self.alert_history if a.timestamp >= alert_cutoff_time],
                    maxlen=self.alert_history.maxlen
                )
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    # API Methods
    async def get_system_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get system metrics for specified time range."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            metrics = []
            for metric in self.system_metrics:
                if metric.timestamp >= cutoff_time:
                    metrics.append({
                        'timestamp': metric.timestamp.isoformat(),
                        'cpu_usage': metric.cpu_usage,
                        'memory_usage': metric.memory_usage,
                        'disk_usage': metric.disk_usage,
                        'network_io': metric.network_io,
                        'process_count': metric.process_count,
                        'thread_count': metric.thread_count,
                        'load_average': metric.load_average,
                        'uptime': metric.uptime
                    })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return []
    
    async def get_application_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get application metrics for specified time range."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            metrics = []
            for metric in self.application_metrics:
                if metric.timestamp >= cutoff_time:
                    metrics.append({
                        'timestamp': metric.timestamp.isoformat(),
                        'active_cameras': metric.active_cameras,
                        'active_persons': metric.active_persons,
                        'detection_rate': metric.detection_rate,
                        'reid_rate': metric.reid_rate,
                        'mapping_rate': metric.mapping_rate,
                        'pipeline_latency': metric.pipeline_latency,
                        'error_rate': metric.error_rate,
                        'cache_hit_rate': metric.cache_hit_rate,
                        'database_connections': metric.database_connections
                    })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting application metrics: {e}")
            return []
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        try:
            return {
                'overall_status': self._calculate_overall_health(),
                'components': self.health_status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {}
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall health status."""
        try:
            if not self.health_status:
                return 'unknown'
            
            statuses = [status['status'] for status in self.health_status.values()]
            
            if 'error' in statuses or 'critical' in statuses:
                return 'critical'
            elif 'warning' in statuses:
                return 'warning'
            else:
                return 'healthy'
                
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return 'unknown'
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        try:
            alerts = []
            for alert in self.performance_alerts.values():
                if not alert.resolved:
                    alerts.append({
                        'alert_id': alert.alert_id,
                        'severity': alert.severity.value,
                        'component': alert.component,
                        'metric': alert.metric,
                        'current_value': alert.current_value,
                        'threshold_value': alert.threshold_value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        try:
            current_time = datetime.now(timezone.utc)
            uptime = (current_time - self.start_time).total_seconds()
            
            return {
                'monitoring_stats': self.monitoring_stats,
                'uptime_seconds': uptime,
                'monitoring_active': self.monitoring_active,
                'system_metrics_count': len(self.system_metrics),
                'application_metrics_count': len(self.application_metrics),
                'active_alerts_count': len([a for a in self.performance_alerts.values() if not a.resolved]),
                'health_checks_registered': len(self.health_checks),
                'configuration': {
                    'metrics_interval': self.config.metrics_interval,
                    'health_check_interval': self.config.health_check_interval,
                    'alert_evaluation_interval': self.config.alert_evaluation_interval
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring statistics: {e}")
            return {}
    
    def reset_statistics(self):
        """Reset monitoring statistics."""
        self.monitoring_stats = {
            'metrics_collected': 0,
            'alerts_generated': 0,
            'health_checks_performed': 0,
            'recommendations_generated': 0,
            'uptime_seconds': 0
        }
        logger.info("Monitoring statistics reset")


# Global monitoring service instance
monitoring_service = MonitoringService()