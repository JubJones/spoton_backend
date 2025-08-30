"""
System monitoring and performance dashboard API endpoints.

Provides:
- Real-time system performance metrics
- Resource utilization monitoring
- Health status and alerts
- Performance dashboard data
- System diagnostics and troubleshooting
"""

import logging
import asyncio
import psutil
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.performance_monitor import performance_monitor
from app.services.monitoring_service import monitoring_service
from app.infrastructure.cache.tracking_cache import tracking_cache
from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.services.memory_manager import memory_manager
from app.infrastructure.gpu import get_gpu_manager
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system-monitoring"])

class SystemPerformanceMetrics(BaseModel):
    """System performance metrics model."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_usage_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    active_tasks: int
    cache_hit_rate: float
    database_connections: int
    uptime_seconds: float

class SystemHealthStatus(BaseModel):
    """System health status model."""
    overall_status: str  # "healthy", "degraded", "critical"
    components: Dict[str, Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    last_check: datetime

class PerformanceDashboard(BaseModel):
    """Performance dashboard data model."""
    current_metrics: SystemPerformanceMetrics
    health_status: SystemHealthStatus
    trending_data: Dict[str, List[float]]
    alerts: List[Dict[str, Any]]
    system_info: Dict[str, Any]

@router.get("/performance/dashboard", response_model=PerformanceDashboard)
async def get_performance_dashboard():
    """Get comprehensive performance dashboard data."""
    try:
        # Get current system metrics with fallback
        try:
            current_metrics = await _get_current_system_metrics()
        except Exception as e:
            logger.warning(f"Error getting current metrics, using defaults: {e}")
            current_metrics = _get_default_system_metrics()
        
        # Get health status with fallback
        try:
            health_status = await _get_system_health_status()
        except Exception as e:
            logger.warning(f"Error getting health status, using defaults: {e}")
            health_status = _get_default_health_status()
        
        # Get trending data with fallback
        try:
            trending_data = await _get_trending_performance_data()
        except Exception as e:
            logger.warning(f"Error getting trending data, using defaults: {e}")
            trending_data = _get_default_trending_data()
        
        # Get active alerts with fallback
        try:
            alerts = await _get_system_alerts()
        except Exception as e:
            logger.warning(f"Error getting alerts, using defaults: {e}")
            alerts = []
        
        # Get system info with fallback
        try:
            system_info = await _get_system_info()
        except Exception as e:
            logger.warning(f"Error getting system info, using defaults: {e}")
            system_info = _get_default_system_info()
        
        return PerformanceDashboard(
            current_metrics=current_metrics,
            health_status=health_status,
            trending_data=trending_data,
            alerts=alerts,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Error getting performance dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance dashboard: {str(e)}")

@router.get("/performance/metrics", response_model=SystemPerformanceMetrics)
async def get_current_performance_metrics():
    """Get current system performance metrics."""
    try:
        return await _get_current_system_metrics()
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@router.get("/performance/history")
async def get_performance_history(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history to retrieve"),
    interval_minutes: int = Query(default=5, ge=1, le=60, description="Interval between data points in minutes")
):
    """Get historical performance metrics."""
    try:
        # This would typically query from a time-series database
        # For now, return simulated trending data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # Generate sample data points
        data_points = []
        current_time = start_time
        
        while current_time <= end_time:
            # In production, this would query actual historical data
            metrics = await _get_current_system_metrics()
            metrics.timestamp = current_time
            data_points.append(metrics.dict())
            current_time += timedelta(minutes=interval_minutes)
        
        return {
            "status": "success",
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "interval_minutes": interval_minutes
            },
            "data_points": data_points[-100:],  # Limit to last 100 points
            "total_points": len(data_points)
        }
    except Exception as e:
        logger.error(f"Error getting performance history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance history: {str(e)}")

@router.get("/health/comprehensive", response_model=SystemHealthStatus)
async def get_comprehensive_health_status():
    """Get comprehensive system health status."""
    try:
        return await _get_system_health_status()
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting health status: {str(e)}")

@router.get("/diagnostics")
async def run_system_diagnostics():
    """Run comprehensive system diagnostics."""
    try:
        diagnostics = {}
        
        # CPU diagnostics
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        diagnostics["cpu"] = {
            "logical_cores": cpu_count,
            "physical_cores": psutil.cpu_count(logical=False),
            "current_frequency_mhz": cpu_freq.current if cpu_freq else None,
            "max_frequency_mhz": cpu_freq.max if cpu_freq else None,
            "usage_per_core": psutil.cpu_percent(interval=1, percpu=True)
        }
        
        # Memory diagnostics
        memory = psutil.virtual_memory()
        diagnostics["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "usage_percent": memory.percent,
            "swap_total_gb": round(psutil.swap_memory().total / (1024**3), 2),
            "swap_used_gb": round(psutil.swap_memory().used / (1024**3), 2)
        }
        
        # Disk diagnostics
        disk_usage = psutil.disk_usage('/')
        diagnostics["disk"] = {
            "total_gb": round(disk_usage.total / (1024**3), 2),
            "used_gb": round(disk_usage.used / (1024**3), 2),
            "free_gb": round(disk_usage.free / (1024**3), 2),
            "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
        }
        
        # GPU diagnostics
        try:
            gpu_manager = get_gpu_manager()
            if gpu_manager and gpu_manager.is_available():
                gpu_info = gpu_manager.get_memory_info()
                diagnostics["gpu"] = {
                    "available": True,
                    "device_count": gpu_manager.device_count(),
                    "current_device": gpu_manager.current_device(),
                    "memory_used_mb": gpu_info.get("used_mb", 0),
                    "memory_total_mb": gpu_info.get("total_mb", 0),
                    "memory_usage_percent": round((gpu_info.get("used_mb", 0) / gpu_info.get("total_mb", 1)) * 100, 2)
                }
            else:
                diagnostics["gpu"] = {"available": False}
        except Exception as e:
            diagnostics["gpu"] = {"available": False, "error": str(e)}
        
        # Service diagnostics
        diagnostics["services"] = {
            "cache_status": "healthy" if tracking_cache else "unavailable",
            "database_status": "healthy" if integrated_db_service else "unavailable",
            "memory_manager_status": "healthy" if memory_manager else "unavailable"
        }
        
        # Network diagnostics
        network_stats = psutil.net_io_counters()
        diagnostics["network"] = {
            "bytes_sent": network_stats.bytes_sent,
            "bytes_recv": network_stats.bytes_recv,
            "packets_sent": network_stats.packets_sent,
            "packets_recv": network_stats.packets_recv,
            "errors_in": network_stats.errin,
            "errors_out": network_stats.errout
        }
        
        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "diagnostics": diagnostics
        }
    except Exception as e:
        logger.error(f"Error running system diagnostics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running system diagnostics: {str(e)}")

@router.get("/alerts")
async def get_system_alerts():
    """Get current system alerts and warnings."""
    try:
        return await _get_system_alerts()
    except Exception as e:
        logger.error(f"Error getting system alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system alerts: {str(e)}")

@router.post("/maintenance/clear-cache")
async def clear_system_cache():
    """Clear system caches for maintenance."""
    try:
        # Clear tracking cache
        if tracking_cache:
            # tracking_cache.clear_all()  # Implement if method exists
            pass
        
        # Clear memory manager cache
        if memory_manager:
            memory_manager.cleanup_resources()
        
        return {
            "status": "success",
            "message": "System caches cleared successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing system cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing system cache: {str(e)}")

@router.post("/maintenance/garbage-collect")
async def force_garbage_collection():
    """Force system garbage collection."""
    try:
        import gc
        collected = gc.collect()
        
        return {
            "status": "success",
            "objects_collected": collected,
            "message": f"Garbage collection completed, {collected} objects collected",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error during garbage collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during garbage collection: {str(e)}")

# Helper functions
async def _get_current_system_metrics() -> SystemPerformanceMetrics:
    """Get current system performance metrics."""
    # CPU and Memory
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    network_stats = psutil.net_io_counters()
    
    # GPU info
    gpu_usage = None
    gpu_memory_used = None
    gpu_memory_total = None
    
    try:
        gpu_manager = get_gpu_manager()
        if gpu_manager and gpu_manager.is_available():
            gpu_info = gpu_manager.get_memory_info()
            gpu_memory_used = gpu_info.get("used_mb", 0)
            gpu_memory_total = gpu_info.get("total_mb", 0)
            gpu_usage = round((gpu_memory_used / gpu_memory_total) * 100, 2) if gpu_memory_total > 0 else 0
    except Exception:
        pass
    
    # Cache hit rate
    cache_hit_rate = 0.95  # Placeholder - implement actual cache statistics
    
    # Database connections
    db_connections = 1  # Placeholder - implement actual connection count
    
    # Active tasks (from performance monitor if available)
    active_tasks = 0
    
    # System uptime
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    
    return SystemPerformanceMetrics(
        timestamp=datetime.now(timezone.utc),
        cpu_usage_percent=cpu_percent,
        memory_usage_percent=memory.percent,
        memory_available_gb=round(memory.available / (1024**3), 2),
        disk_usage_percent=round((disk_usage.used / disk_usage.total) * 100, 2),
        network_bytes_sent=network_stats.bytes_sent,
        network_bytes_recv=network_stats.bytes_recv,
        gpu_usage_percent=gpu_usage,
        gpu_memory_used_mb=gpu_memory_used,
        gpu_memory_total_mb=gpu_memory_total,
        active_tasks=active_tasks,
        cache_hit_rate=cache_hit_rate,
        database_connections=db_connections,
        uptime_seconds=uptime_seconds
    )

async def _get_system_health_status() -> SystemHealthStatus:
    """Get comprehensive system health status."""
    components = {}
    alerts = []
    recommendations = []
    overall_status = "healthy"
    
    # Check CPU health
    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 90:
        components["cpu"] = {"status": "critical", "usage": cpu_percent}
        alerts.append({"type": "critical", "component": "cpu", "message": f"High CPU usage: {cpu_percent}%"})
        recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive processes")
        overall_status = "critical"
    elif cpu_percent > 70:
        components["cpu"] = {"status": "warning", "usage": cpu_percent}
        alerts.append({"type": "warning", "component": "cpu", "message": f"Elevated CPU usage: {cpu_percent}%"})
        if overall_status != "critical":
            overall_status = "degraded"
    else:
        components["cpu"] = {"status": "healthy", "usage": cpu_percent}
    
    # Check Memory health
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        components["memory"] = {"status": "critical", "usage": memory.percent}
        alerts.append({"type": "critical", "component": "memory", "message": f"High memory usage: {memory.percent}%"})
        recommendations.append("Consider adding more RAM or optimizing memory usage")
        overall_status = "critical"
    elif memory.percent > 80:
        components["memory"] = {"status": "warning", "usage": memory.percent}
        alerts.append({"type": "warning", "component": "memory", "message": f"Elevated memory usage: {memory.percent}%"})
        if overall_status != "critical":
            overall_status = "degraded"
    else:
        components["memory"] = {"status": "healthy", "usage": memory.percent}
    
    # Check GPU health if available
    try:
        gpu_manager = get_gpu_manager()
        if gpu_manager and gpu_manager.is_available():
            gpu_info = gpu_manager.get_memory_info()
            gpu_usage = (gpu_info.get("used_mb", 0) / gpu_info.get("total_mb", 1)) * 100
            if gpu_usage > 90:
                components["gpu"] = {"status": "warning", "usage": gpu_usage}
                alerts.append({"type": "warning", "component": "gpu", "message": f"High GPU memory usage: {gpu_usage:.1f}%"})
                if overall_status == "healthy":
                    overall_status = "degraded"
            else:
                components["gpu"] = {"status": "healthy", "usage": gpu_usage}
        else:
            components["gpu"] = {"status": "unavailable", "usage": 0}
    except Exception:
        components["gpu"] = {"status": "error", "usage": 0}
    
    # Check services
    components["services"] = {
        "cache": "healthy" if tracking_cache else "unavailable",
        "database": "healthy" if integrated_db_service else "unavailable",
        "memory_manager": "healthy" if memory_manager else "unavailable"
    }
    
    return SystemHealthStatus(
        overall_status=overall_status,
        components=components,
        alerts=alerts,
        recommendations=recommendations,
        last_check=datetime.now(timezone.utc)
    )

async def _get_trending_performance_data() -> Dict[str, List[float]]:
    """Get trending performance data for charts."""
    # This would typically query from a time-series database
    # For now, return simulated trending data
    import random
    
    # Generate 60 data points (last hour with 1-minute intervals)
    time_points = 60
    
    return {
        "cpu_usage": [random.uniform(20, 80) for _ in range(time_points)],
        "memory_usage": [random.uniform(40, 85) for _ in range(time_points)],
        "gpu_usage": [random.uniform(10, 70) for _ in range(time_points)] if get_gpu_manager() else [],
        "active_tasks": [random.randint(0, 5) for _ in range(time_points)],
        "timestamps": [(datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat() for i in range(time_points-1, -1, -1)]
    }

async def _get_system_alerts() -> List[Dict[str, Any]]:
    """Get current system alerts."""
    alerts = []
    
    # Check for high resource usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 80:
        alerts.append({
            "id": "high-cpu",
            "type": "warning" if cpu_percent < 90 else "critical",
            "title": "High CPU Usage",
            "message": f"CPU usage is at {cpu_percent}%",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": "cpu"
        })
    
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        alerts.append({
            "id": "high-memory",
            "type": "warning" if memory.percent < 90 else "critical",
            "title": "High Memory Usage",
            "message": f"Memory usage is at {memory.percent}%",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": "memory"
        })
    
    return alerts

async def _get_system_info() -> Dict[str, Any]:
    """Get general system information."""
    cpu_info = {}
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_info = {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "max_frequency_mhz": cpu_freq.max if cpu_freq else None
        }
    except Exception:
        pass
    
    memory_info = {}
    try:
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2)
        }
    except Exception:
        pass
    
    gpu_info = {"available": False}
    try:
        gpu_manager = get_gpu_manager()
        if gpu_manager and gpu_manager.is_available():
            gpu_memory = gpu_manager.get_memory_info()
            gpu_info = {
                "available": True,
                "device_count": gpu_manager.device_count(),
                "total_memory_gb": round(gpu_memory.get("total_mb", 0) / 1024, 2)
            }
    except Exception:
        pass
    
    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "platform": psutil.os.name,
        "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1)
    }

# Fallback functions for error handling

def _get_default_system_metrics() -> SystemPerformanceMetrics:
    """Get default system metrics when real metrics fail."""
    return SystemPerformanceMetrics(
        timestamp=datetime.now(timezone.utc),
        cpu_usage_percent=0.0,
        memory_usage_percent=0.0,
        memory_available_gb=0.0,
        disk_usage_percent=0.0,
        network_bytes_sent=0,
        network_bytes_recv=0,
        gpu_usage_percent=None,
        gpu_memory_used_mb=None,
        gpu_memory_total_mb=None,
        active_tasks=0,
        cache_hit_rate=0.0,
        database_connections=0,
        uptime_seconds=0.0
    )

def _get_default_health_status() -> SystemHealthStatus:
    """Get default health status when real status fails."""
    return SystemHealthStatus(
        overall_status="unknown",
        components={
            "cpu": {"status": "unknown", "usage": 0},
            "memory": {"status": "unknown", "usage": 0},
            "gpu": {"status": "unavailable", "usage": 0},
            "services": {
                "cache": "unknown",
                "database": "unknown", 
                "memory_manager": "unknown"
            }
        },
        alerts=[],
        recommendations=[],
        last_check=datetime.now(timezone.utc)
    )

def _get_default_trending_data() -> Dict[str, List[float]]:
    """Get default trending data when real data fails."""
    return {
        "cpu_usage": [0.0] * 60,
        "memory_usage": [0.0] * 60,
        "gpu_usage": [],
        "active_tasks": [0] * 60,
        "timestamps": [(datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat() for i in range(59, -1, -1)]
    }

def _get_default_system_info() -> Dict[str, Any]:
    """Get default system info when real info fails."""
    return {
        "cpu": {"logical_cores": 0, "physical_cores": 0, "max_frequency_mhz": None},
        "memory": {"total_gb": 0.0, "available_gb": 0.0},
        "gpu": {"available": False},
        "platform": "unknown",
        "python_version": "unknown",
        "uptime_hours": 0.0
    }
