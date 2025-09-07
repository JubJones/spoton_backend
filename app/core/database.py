"""
Database health check functions for Phase 5 production health monitoring.

Provides database connectivity and performance health checks for the
enhanced health check system.
"""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from app.infrastructure.integration.database_integration_service import database_integration_service

logger = logging.getLogger(__name__)


async def get_database_health() -> Dict[str, Any]:
    """
    Get database health status for production monitoring.
    
    Returns:
        Dict containing database health metrics and status
    """
    try:
        health_start = time.time()
        
        # Get service health from database integration service
        service_health = await database_integration_service.get_service_health()
        
        # Extract key metrics
        integration_stats = service_health.get('integration_layer', {})
        integrated_service = service_health.get('integrated_service', {})
        
        # Perform a simple connectivity test
        try:
            # Test basic database operations
            test_start = time.time()
            
            # Create a test analytics summary (lightweight operation)
            test_result = await database_integration_service.get_analytics_summary(
                environment_id="health_check",
                start_time=None,
                end_time=None
            )
            
            query_time = (time.time() - test_start) * 1000
            connectivity_test_passed = isinstance(test_result, dict)
            
        except Exception as e:
            logger.warning(f"Database connectivity test failed: {e}")
            connectivity_test_passed = False
            query_time = 0
        
        # Calculate overall health
        success_rate = integration_stats.get('success_rate', 0)
        connected = connectivity_test_passed and success_rate > 50
        
        # Determine health status
        if connected and success_rate > 90:
            status = "healthy"
        elif connected and success_rate > 70:
            status = "degraded"
        else:
            status = "unhealthy"
        
        total_time = (time.time() - health_start) * 1000
        
        return {
            "connected": connected,
            "status": status,
            "query_time_ms": query_time,
            "total_check_time_ms": total_time,
            "integration_statistics": {
                "total_operations": integration_stats.get('total_operations', 0),
                "successful_operations": integration_stats.get('successful_operations', 0),
                "failed_operations": integration_stats.get('failed_operations', 0),
                "success_rate": success_rate,
                "cache_hit_rate": integration_stats.get('cache_hit_rate', 0)
            },
            "pool_size": "unknown",  # Would need connection pool access
            "active_connections": "unknown",  # Would need connection pool access
            "database_service": integrated_service
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "connected": False,
            "status": "unhealthy",
            "error": str(e),
            "query_time_ms": 0,
            "total_check_time_ms": 0
        }


async def get_database_performance_metrics() -> Dict[str, Any]:
    """
    Get detailed database performance metrics.
    
    Returns:
        Dict containing performance metrics
    """
    try:
        metrics_start = time.time()
        
        # Get comprehensive service health
        service_health = await database_integration_service.get_service_health()
        
        # Get integration statistics
        integration_stats = service_health.get('integration_layer', {})
        
        performance_metrics = {
            "response_time_ms": (time.time() - metrics_start) * 1000,
            "operations": {
                "total": integration_stats.get('total_operations', 0),
                "successful": integration_stats.get('successful_operations', 0),
                "failed": integration_stats.get('failed_operations', 0),
                "success_rate_percent": integration_stats.get('success_rate', 0)
            },
            "cache": {
                "hits": integration_stats.get('cache_hits', 0),
                "misses": integration_stats.get('cache_misses', 0),
                "hit_rate_percent": integration_stats.get('cache_hit_rate', 0)
            },
            "service_layer": service_health.get('integrated_service', {})
        }
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Database performance metrics failed: {e}")
        return {
            "response_time_ms": 0,
            "error": str(e)
        }


async def test_database_operations() -> Dict[str, Any]:
    """
    Test basic database operations for health verification.
    
    Returns:
        Dict containing test results
    """
    try:
        test_results = {
            "read_test": False,
            "analytics_test": False,
            "service_health_test": False,
            "errors": []
        }
        
        # Test analytics functionality
        try:
            analytics = await database_integration_service.get_analytics_summary(
                environment_id="health_test"
            )
            test_results["analytics_test"] = isinstance(analytics, dict)
        except Exception as e:
            test_results["errors"].append(f"Analytics test failed: {str(e)}")
        
        # Test service health
        try:
            health = await database_integration_service.get_service_health()
            test_results["service_health_test"] = isinstance(health, dict)
        except Exception as e:
            test_results["errors"].append(f"Service health test failed: {str(e)}")
        
        # Set read test to true if either analytics or service health worked
        test_results["read_test"] = (
            test_results["analytics_test"] or 
            test_results["service_health_test"]
        )
        
        return test_results
        
    except Exception as e:
        logger.error(f"Database operations test failed: {e}")
        return {
            "read_test": False,
            "analytics_test": False,
            "service_health_test": False,
            "errors": [str(e)]
        }