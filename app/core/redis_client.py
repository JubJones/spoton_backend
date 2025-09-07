"""
Redis health check functions for Phase 5 production health monitoring.

Provides Redis connectivity and performance health checks for the
enhanced health check system.
"""

import time
import logging
from typing import Dict, Any, Optional

from app.infrastructure.cache.tracking_cache import tracking_cache

logger = logging.getLogger(__name__)


async def get_redis_health() -> Dict[str, Any]:
    """
    Get Redis health status for production monitoring.
    
    Returns:
        Dict containing Redis health metrics and status
    """
    try:
        health_start = time.time()
        
        # Test basic Redis connectivity
        connectivity_test = await test_redis_connectivity()
        connected = connectivity_test.get("connected", False)
        ping_time = connectivity_test.get("ping_time_ms", 0)
        
        # Get Redis performance metrics if connected
        if connected:
            try:
                performance_metrics = await get_redis_performance_metrics()
                memory_usage = performance_metrics.get("memory_usage_mb", 0)
                connected_clients = performance_metrics.get("connected_clients", 0)
                operations_per_sec = performance_metrics.get("operations_per_sec", 0)
                
                # Determine health status based on performance
                if ping_time < 10 and memory_usage < 1000:  # < 10ms ping, < 1GB memory
                    status = "healthy"
                elif ping_time < 50 and memory_usage < 2000:  # < 50ms ping, < 2GB memory
                    status = "degraded"
                else:
                    status = "unhealthy"
                    
            except Exception as e:
                logger.warning(f"Redis performance check failed: {e}")
                status = "degraded"
                memory_usage = "unknown"
                connected_clients = "unknown"
                operations_per_sec = "unknown"
        else:
            status = "unhealthy"
            memory_usage = "unknown"
            connected_clients = "unknown"
            operations_per_sec = "unknown"
        
        total_time = (time.time() - health_start) * 1000
        
        return {
            "connected": connected,
            "status": status,
            "ping_time_ms": ping_time,
            "total_check_time_ms": total_time,
            "memory_usage": memory_usage,
            "connected_clients": connected_clients,
            "operations_per_sec": operations_per_sec,
            "error": connectivity_test.get("error") if not connected else None
        }
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "connected": False,
            "status": "unhealthy",
            "error": str(e),
            "ping_time_ms": 0,
            "total_check_time_ms": 0
        }


async def test_redis_connectivity() -> Dict[str, Any]:
    """
    Test basic Redis connectivity.
    
    Returns:
        Dict containing connectivity test results
    """
    try:
        ping_start = time.time()
        
        # Use tracking cache to test Redis connectivity
        # Try to get cache statistics which involves Redis operations
        try:
            cache_stats = await tracking_cache.get_cache_statistics()
            ping_time = (time.time() - ping_start) * 1000
            
            # If we got stats back, Redis is working
            connected = isinstance(cache_stats, dict)
            
            return {
                "connected": connected,
                "ping_time_ms": ping_time,
                "cache_stats": cache_stats
            }
            
        except Exception as cache_error:
            # Try a more basic operation
            try:
                # Test basic cache operation
                test_key = "health_check_test"
                test_value = {"timestamp": time.time()}
                
                # This should use Redis internally
                await tracking_cache.cache_frame_data("health_test", test_value)
                
                ping_time = (time.time() - ping_start) * 1000
                
                return {
                    "connected": True,
                    "ping_time_ms": ping_time,
                    "basic_test": "passed"
                }
                
            except Exception as basic_error:
                logger.warning(f"Basic Redis test failed: {basic_error}")
                return {
                    "connected": False,
                    "ping_time_ms": 0,
                    "error": str(basic_error)
                }
        
    except Exception as e:
        logger.error(f"Redis connectivity test failed: {e}")
        return {
            "connected": False,
            "ping_time_ms": 0,
            "error": str(e)
        }


async def get_redis_performance_metrics() -> Dict[str, Any]:
    """
    Get Redis performance metrics.
    
    Returns:
        Dict containing performance metrics
    """
    try:
        metrics_start = time.time()
        
        # Get cache statistics
        cache_stats = await tracking_cache.get_cache_statistics()
        
        # Extract performance metrics from cache stats
        performance_metrics = {
            "response_time_ms": (time.time() - metrics_start) * 1000,
            "memory_usage_mb": cache_stats.get("memory_usage_mb", 0),
            "connected_clients": cache_stats.get("connected_clients", 0),
            "operations_per_sec": cache_stats.get("operations_per_sec", 0),
            "cache_hit_rate": cache_stats.get("hit_rate_percent", 0),
            "total_keys": cache_stats.get("total_keys", 0),
            "expired_keys": cache_stats.get("expired_keys", 0),
            "keyspace_stats": cache_stats.get("keyspace", {})
        }
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Redis performance metrics failed: {e}")
        return {
            "response_time_ms": 0,
            "error": str(e)
        }


async def test_redis_operations() -> Dict[str, Any]:
    """
    Test various Redis operations for health verification.
    
    Returns:
        Dict containing test results
    """
    try:
        test_results = {
            "cache_test": False,
            "statistics_test": False,
            "frame_data_test": False,
            "errors": []
        }
        
        # Test cache statistics
        try:
            stats = await tracking_cache.get_cache_statistics()
            test_results["statistics_test"] = isinstance(stats, dict)
        except Exception as e:
            test_results["errors"].append(f"Statistics test failed: {str(e)}")
        
        # Test frame data caching
        try:
            test_data = {"test": True, "timestamp": time.time()}
            await tracking_cache.cache_frame_data("health_test", test_data)
            test_results["frame_data_test"] = True
        except Exception as e:
            test_results["errors"].append(f"Frame data test failed: {str(e)}")
        
        # Set overall cache test
        test_results["cache_test"] = (
            test_results["statistics_test"] or 
            test_results["frame_data_test"]
        )
        
        return test_results
        
    except Exception as e:
        logger.error(f"Redis operations test failed: {e}")
        return {
            "cache_test": False,
            "statistics_test": False,
            "frame_data_test": False,
            "errors": [str(e)]
        }


async def get_redis_info() -> Dict[str, Any]:
    """
    Get detailed Redis server information (if available).
    
    Returns:
        Dict containing Redis server info
    """
    try:
        # Try to get cache statistics which may include server info
        cache_stats = await tracking_cache.get_cache_statistics()
        
        # Extract server information
        server_info = {
            "version": cache_stats.get("redis_version", "unknown"),
            "uptime_seconds": cache_stats.get("uptime_seconds", 0),
            "used_memory_mb": cache_stats.get("memory_usage_mb", 0),
            "max_memory_mb": cache_stats.get("max_memory_mb", 0),
            "connected_clients": cache_stats.get("connected_clients", 0),
            "total_connections_received": cache_stats.get("total_connections", 0),
            "keyspace_hits": cache_stats.get("keyspace_hits", 0),
            "keyspace_misses": cache_stats.get("keyspace_misses", 0)
        }
        
        return server_info
        
    except Exception as e:
        logger.error(f"Redis info retrieval failed: {e}")
        return {
            "error": str(e),
            "available": False
        }