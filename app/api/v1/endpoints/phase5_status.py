"""
Phase 5: Production Readiness Status API

Provides comprehensive system status endpoints that integrate with all
Phase 5 production readiness components for monitoring and observability.
"""

import time
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

# Phase 5 Production Components
from app.utils.memory_manager import memory_manager
from app.utils.error_handler import production_error_handler
from app.utils.metrics_collector import production_metrics_collector
from app.services.batch_optimization_service import batch_optimization_service

# Health checks
from app.core.database import get_database_health, get_database_performance_metrics
from app.core.redis_client import get_redis_health, get_redis_performance_metrics

# Core services
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system/phase5", response_model=Dict[str, Any])
async def get_phase5_status():
    """
    Get comprehensive Phase 5 production readiness status.
    
    Returns detailed status of all Phase 5 components including:
    - Memory management and resource optimization
    - Error handling and recovery system
    - Metrics collection and monitoring
    - Batch optimization performance
    - Infrastructure health (database, Redis)
    """
    try:
        status_start = time.time()
        
        # Collect status from all Phase 5 components
        memory_status = memory_manager.get_memory_status()
        error_stats = production_error_handler.get_error_statistics()
        metrics_summary = production_metrics_collector.get_metrics_summary()
        batch_metrics = batch_optimization_service.get_performance_metrics()
        
        # Get infrastructure health
        db_health = await get_database_health()
        redis_health = await get_redis_health()
        
        # Calculate overall Phase 5 health
        component_health = {
            "memory_management": _assess_memory_health(memory_status),
            "error_handling": _assess_error_health(error_stats),
            "metrics_collection": _assess_metrics_health(metrics_summary),
            "batch_optimization": _assess_batch_health(batch_metrics),
            "database": db_health.get("status", "unknown"),
            "redis": redis_health.get("status", "unknown")
        }
        
        # Determine overall status
        healthy_components = sum(1 for status in component_health.values() if status == "healthy")
        degraded_components = sum(1 for status in component_health.values() if status == "degraded")
        unhealthy_components = sum(1 for status in component_health.values() if status == "unhealthy")
        
        if unhealthy_components > 0:
            overall_status = "unhealthy"
        elif degraded_components > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        response_time = (time.time() - status_start) * 1000
        
        phase5_status = {
            "phase": "Phase 5: Production Readiness",
            "version": "5.0-production",
            "overall_status": overall_status,
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "component_health": component_health,
            "component_summary": {
                "total_components": len(component_health),
                "healthy": healthy_components,
                "degraded": degraded_components,
                "unhealthy": unhealthy_components
            },
            "detailed_status": {
                "memory_management": memory_status,
                "error_handling": error_stats,
                "metrics_collection": metrics_summary,
                "batch_optimization": batch_metrics,
                "infrastructure": {
                    "database": db_health,
                    "redis": redis_health
                }
            },
            "production_readiness": {
                "memory_optimization": memory_status.get("resource_pressure", {}).get("monitoring_active", False),
                "error_recovery": len(error_stats.get("recent_errors", [])) == 0,
                "metrics_active": metrics_summary.get("collection_stats", {}).get("collections_performed", 0) > 0,
                "batch_optimization": batch_metrics.get("system_utilization", {}).get("current_efficiency", 0) > 0.5,
                "infrastructure_healthy": db_health.get("connected", False) and redis_health.get("connected", False)
            }
        }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=phase5_status
        )
        
    except Exception as e:
        logger.error(f"Phase 5 status check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "phase": "Phase 5: Production Readiness",
                "overall_status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/system/memory", response_model=Dict[str, Any])
async def get_memory_status():
    """Get detailed memory management status."""
    try:
        return memory_manager.get_memory_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory status check failed: {str(e)}"
        )


@router.get("/system/errors", response_model=Dict[str, Any])
async def get_error_statistics():
    """Get detailed error handling statistics."""
    try:
        return production_error_handler.get_error_statistics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error statistics check failed: {str(e)}"
        )


@router.get("/system/metrics", response_model=Dict[str, Any])
async def get_metrics_summary():
    """Get detailed metrics collection summary."""
    try:
        return production_metrics_collector.get_metrics_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics summary check failed: {str(e)}"
        )


@router.get("/system/performance", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get detailed batch optimization performance metrics."""
    try:
        return batch_optimization_service.get_performance_metrics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics check failed: {str(e)}"
        )


@router.get("/system/infrastructure", response_model=Dict[str, Any])
async def get_infrastructure_status():
    """Get infrastructure health status (database, Redis, etc.)."""
    try:
        infrastructure_start = time.time()
        
        # Get infrastructure health
        db_health = await get_database_health()
        redis_health = await get_redis_health()
        
        # Get performance metrics
        db_performance = await get_database_performance_metrics()
        redis_performance = await get_redis_performance_metrics()
        
        infrastructure_status = {
            "database": {
                "health": db_health,
                "performance": db_performance
            },
            "redis": {
                "health": redis_health,
                "performance": redis_performance
            },
            "overall_status": "healthy" if (
                db_health.get("connected", False) and 
                redis_health.get("connected", False)
            ) else "unhealthy",
            "response_time_ms": (time.time() - infrastructure_start) * 1000,
            "timestamp": time.time()
        }
        
        return infrastructure_status
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Infrastructure status check failed: {str(e)}"
        )


@router.post("/system/memory/cleanup", response_model=Dict[str, Any])
async def trigger_memory_cleanup():
    """Trigger manual memory cleanup."""
    try:
        cleanup_result = await memory_manager.perform_cleanup()
        return {
            "cleanup_triggered": True,
            "result": cleanup_result,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory cleanup failed: {str(e)}"
        )


@router.get("/system/prometheus", response_model=str)
async def get_prometheus_metrics():
    """Get metrics in Prometheus format for external monitoring."""
    try:
        return production_metrics_collector.export_metrics_for_prometheus()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prometheus metrics export failed: {str(e)}"
        )


# Helper functions for health assessment
def _assess_memory_health(memory_status: Dict[str, Any]) -> str:
    """Assess memory management component health."""
    try:
        current_memory = memory_status.get("current_stats", {}).get("ram_percentage", 0)
        monitoring_active = memory_status.get("resource_pressure", {}).get("monitoring_active", False)
        
        if not monitoring_active:
            return "unhealthy"
        elif current_memory < 70:
            return "healthy"
        elif current_memory < 85:
            return "degraded"
        else:
            return "unhealthy"
    except Exception:
        return "unknown"


def _assess_error_health(error_stats: Dict[str, Any]) -> str:
    """Assess error handling component health."""
    try:
        critical_errors = error_stats.get("error_statistics", {}).get("critical_errors", 0)
        recovery_rate = error_stats.get("recovery_effectiveness", {}).get("recovery_rate", 0)
        breakers_open = sum(1 for cb in error_stats.get("circuit_breaker_status", {}).values() 
                          if cb.get("is_open", False))
        
        if critical_errors == 0 and breakers_open == 0 and recovery_rate > 80:
            return "healthy"
        elif critical_errors < 5 and breakers_open < 2:
            return "degraded"
        else:
            return "unhealthy"
    except Exception:
        return "unknown"


def _assess_metrics_health(metrics_summary: Dict[str, Any]) -> str:
    """Assess metrics collection component health."""
    try:
        collections = metrics_summary.get("collection_stats", {}).get("collections_performed", 0)
        errors = metrics_summary.get("collection_stats", {}).get("collection_errors", 0)
        active_alerts = metrics_summary.get("alerts", {}).get("active_alerts_count", 0)
        
        if collections > 0 and errors == 0 and active_alerts < 5:
            return "healthy"
        elif collections > 0 and active_alerts < 10:
            return "degraded"
        else:
            return "unhealthy"
    except Exception:
        return "unknown"


def _assess_batch_health(batch_metrics: Dict[str, Any]) -> str:
    """Assess batch optimization component health."""
    try:
        efficiency = batch_metrics.get("system_utilization", {}).get("current_efficiency", 0)
        throughput = batch_metrics.get("batch_processing", {}).get("throughput_fps", 0)
        
        if efficiency > 0.7 and throughput > 0:
            return "healthy"
        elif efficiency > 0.4:
            return "degraded"
        else:
            return "unhealthy"
    except Exception:
        return "unknown"