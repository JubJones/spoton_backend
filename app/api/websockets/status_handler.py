"""
Status handler for WebSocket system status messages.

Handles:
- System status messages
- Performance metrics
- Health indicators
- Resource utilization
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import psutil
import time

from app.api.websockets.connection_manager import binary_websocket_manager, MessageType
from app.infrastructure.gpu import get_gpu_manager
from app.orchestration.pipeline_orchestrator import orchestrator
from app.core.config import settings

logger = logging.getLogger(__name__)


class StatusHandler:
    """
    Handles system status messages for WebSocket transmission.
    
    Features:
    - Real-time system health monitoring
    - Performance metrics collection
    - Resource utilization tracking
    - Alert generation
    """
    
    def __init__(self):
        # System monitoring
        self.gpu_manager = get_gpu_manager()
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Status monitoring intervals
        self.status_update_interval = 5.0  # seconds
        self.performance_update_interval = 1.0  # seconds
        
        # Performance history
        self.performance_history = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_utilization": [],
            "processing_fps": [],
            "connection_count": []
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "gpu_utilization": 90.0,
            "processing_fps": 10.0,  # minimum fps
            "connection_failures": 5
        }
        
        # Status cache
        self.cached_status = {}
        self.last_status_update = 0
        
        logger.info("StatusHandler initialized")
    
    async def start_monitoring(self):
        """Start system status monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("System status monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting status monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop system status monitoring."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
                self.monitoring_task = None
            
            logger.info("System status monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping status monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                # Collect system status
                system_status = await self._collect_system_status()
                
                # Broadcast to all connections
                await self._broadcast_system_status(system_status)
                
                # Check for alerts
                await self._check_alerts(system_status)
                
                # Update performance history
                self._update_performance_history(system_status)
                
                # Wait for next update
                await asyncio.sleep(self.status_update_interval)
                
        except asyncio.CancelledError:
            logger.info("Status monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in status monitoring loop: {e}")
    
    async def _collect_system_status(self) -> Dict[str, Any]:
        """Collect comprehensive system status."""
        try:
            current_time = time.time()
            
            # Use cached status if recent
            if (current_time - self.last_status_update) < 1.0 and self.cached_status:
                return self.cached_status
            
            # Collect CPU and memory info
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            # Collect GPU information
            gpu_info = {}
            if self.gpu_manager:
                try:
                    gpu_info = {
                        "gpu_available": self.gpu_manager.is_available(),
                        "gpu_count": self.gpu_manager.get_device_count(),
                        "gpu_utilization": self.gpu_manager.get_utilization(),
                        "gpu_memory_used": self.gpu_manager.get_memory_usage(),
                        "gpu_temperature": self.gpu_manager.get_temperature()
                    }
                except Exception as e:
                    logger.warning(f"Error collecting GPU info: {e}")
                    gpu_info = {
                        "gpu_available": False,
                        "gpu_count": 0,
                        "gpu_utilization": 0.0,
                        "gpu_memory_used": 0.0,
                        "gpu_temperature": 0.0
                    }
            
            # Collect pipeline information
            pipeline_info = {}
            if orchestrator:
                try:
                    pipeline_stats = orchestrator.get_pipeline_stats()
                    pipeline_info = {
                        "pipeline_active": orchestrator.is_initialized(),
                        "active_tasks": len(orchestrator.get_active_tasks()),
                        "total_frames_processed": pipeline_stats.get("total_frames_processed", 0),
                        "processing_fps": self._calculate_processing_fps(pipeline_stats),
                        "average_processing_time": pipeline_stats.get("average_processing_time", 0.0)
                    }
                except Exception as e:
                    logger.warning(f"Error collecting pipeline info: {e}")
                    pipeline_info = {
                        "pipeline_active": False,
                        "active_tasks": 0,
                        "total_frames_processed": 0,
                        "processing_fps": 0.0,
                        "average_processing_time": 0.0
                    }
            
            # Collect connection information
            connection_info = {
                "active_connections": binary_websocket_manager.get_connection_count(),
                "connection_quality": self._calculate_connection_quality(),
                "message_queue_size": self._get_message_queue_size(),
                "websocket_performance": binary_websocket_manager.get_performance_stats()
            }
            
            # Compile system status
            system_status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_info.percent,
                    "memory_available": memory_info.available,
                    "memory_total": memory_info.total,
                    "uptime": time.time() - psutil.boot_time()
                },
                "gpu": gpu_info,
                "pipeline": pipeline_info,
                "connections": connection_info,
                "alerts": self._get_current_alerts(),
                "health_status": self._calculate_health_status()
            }
            
            # Cache the status
            self.cached_status = system_status
            self.last_status_update = current_time
            
            return system_status
            
        except Exception as e:
            logger.error(f"Error collecting system status: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "health_status": "error"
            }
    
    async def _broadcast_system_status(self, system_status: Dict[str, Any]):
        """Broadcast system status to all connections."""
        try:
            # Create status message
            status_message = {
                "type": MessageType.SYSTEM_STATUS.value,
                "data": system_status
            }
            
            # Broadcast to all connections
            await binary_websocket_manager.broadcast_system_status(status_message)
            
        except Exception as e:
            logger.error(f"Error broadcasting system status: {e}")
    
    async def _check_alerts(self, system_status: Dict[str, Any]):
        """Check for system alerts."""
        try:
            alerts = []
            
            # Check CPU usage
            cpu_usage = system_status.get("system", {}).get("cpu_usage", 0)
            if cpu_usage > self.alert_thresholds["cpu_usage"]:
                alerts.append({
                    "type": "cpu_high",
                    "message": f"High CPU usage: {cpu_usage:.1f}%",
                    "severity": "warning",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Check memory usage
            memory_usage = system_status.get("system", {}).get("memory_usage", 0)
            if memory_usage > self.alert_thresholds["memory_usage"]:
                alerts.append({
                    "type": "memory_high",
                    "message": f"High memory usage: {memory_usage:.1f}%",
                    "severity": "warning",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Check GPU utilization
            gpu_utilization = system_status.get("gpu", {}).get("gpu_utilization", 0)
            if gpu_utilization > self.alert_thresholds["gpu_utilization"]:
                alerts.append({
                    "type": "gpu_high",
                    "message": f"High GPU utilization: {gpu_utilization:.1f}%",
                    "severity": "warning",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Check processing FPS
            processing_fps = system_status.get("pipeline", {}).get("processing_fps", 0)
            if processing_fps < self.alert_thresholds["processing_fps"]:
                alerts.append({
                    "type": "fps_low",
                    "message": f"Low processing FPS: {processing_fps:.1f}",
                    "severity": "warning",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Send alerts if any
            if alerts:
                await self._send_alerts(alerts)
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alert notifications."""
        try:
            alert_message = {
                "type": "system_alerts",
                "alerts": alerts,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await binary_websocket_manager.broadcast_system_status(alert_message)
            
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    def _calculate_processing_fps(self, pipeline_stats: Dict[str, Any]) -> float:
        """Calculate current processing FPS."""
        try:
            # Use average processing time to estimate FPS
            avg_processing_time = pipeline_stats.get("average_processing_time", 0.0)
            if avg_processing_time > 0:
                return 1.0 / avg_processing_time
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating processing FPS: {e}")
            return 0.0
    
    def _calculate_connection_quality(self) -> str:
        """Calculate connection quality status."""
        try:
            performance_stats = binary_websocket_manager.get_performance_stats()
            connection_failures = performance_stats.get("connection_failures", 0)
            
            if connection_failures > self.alert_thresholds["connection_failures"]:
                return "poor"
            elif connection_failures > 2:
                return "fair"
            else:
                return "good"
                
        except Exception as e:
            logger.error(f"Error calculating connection quality: {e}")
            return "unknown"
    
    def _get_message_queue_size(self) -> int:
        """Get current message queue size."""
        try:
            performance_stats = binary_websocket_manager.get_performance_stats()
            return performance_stats.get("pending_batches", 0)
            
        except Exception as e:
            logger.error(f"Error getting message queue size: {e}")
            return 0
    
    def _get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get current system alerts."""
        # This would be implemented to return active alerts
        return []
    
    def _calculate_health_status(self) -> str:
        """Calculate overall system health status."""
        try:
            # Check critical components
            if self.gpu_manager and not self.gpu_manager.is_available():
                return "degraded"
            
            if not orchestrator.is_initialized():
                return "down"
            
            # Check resource usage
            if self.cached_status:
                cpu_usage = self.cached_status.get("system", {}).get("cpu_usage", 0)
                memory_usage = self.cached_status.get("system", {}).get("memory_usage", 0)
                
                if cpu_usage > 90 or memory_usage > 90:
                    return "critical"
                elif cpu_usage > 70 or memory_usage > 70:
                    return "degraded"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Error calculating health status: {e}")
            return "unknown"
    
    def _update_performance_history(self, system_status: Dict[str, Any]):
        """Update performance history for trending."""
        try:
            # Extract metrics
            cpu_usage = system_status.get("system", {}).get("cpu_usage", 0)
            memory_usage = system_status.get("system", {}).get("memory_usage", 0)
            gpu_utilization = system_status.get("gpu", {}).get("gpu_utilization", 0)
            processing_fps = system_status.get("pipeline", {}).get("processing_fps", 0)
            connection_count = system_status.get("connections", {}).get("active_connections", 0)
            
            # Add to history
            self.performance_history["cpu_usage"].append(cpu_usage)
            self.performance_history["memory_usage"].append(memory_usage)
            self.performance_history["gpu_utilization"].append(gpu_utilization)
            self.performance_history["processing_fps"].append(processing_fps)
            self.performance_history["connection_count"].append(connection_count)
            
            # Limit history size
            max_history = 100
            for metric in self.performance_history:
                if len(self.performance_history[metric]) > max_history:
                    self.performance_history[metric] = self.performance_history[metric][-max_history:]
            
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return await self._collect_system_status()
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """Get performance history."""
        return self.performance_history.copy()
    
    def get_alert_thresholds(self) -> Dict[str, float]:
        """Get alert thresholds."""
        return self.alert_thresholds.copy()
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a metric."""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            logger.info(f"Updated alert threshold for {metric}: {threshold}")
    
    def reset_performance_history(self):
        """Reset performance history."""
        self.performance_history = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_utilization": [],
            "processing_fps": [],
            "connection_count": []
        }
        logger.info("Performance history reset")


# Global status handler instance
status_handler = StatusHandler()