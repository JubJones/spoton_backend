"""
Tests for system monitoring API endpoints.

Tests:
- Performance monitoring dashboard
- System health checks
- Resource utilization metrics
- Diagnostic information
- Alert systems
- Maintenance operations
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import status

from app.main import app
from app.services.performance_monitor import performance_monitor
from app.services.monitoring_service import monitoring_service

# Test client
client = TestClient(app)


class TestSystemMonitoringEndpoints:
    """Test system monitoring API endpoints."""
    
    def test_performance_dashboard_success(self):
        """Test successful performance dashboard retrieval."""
        response = client.get("/api/v1/system/performance/dashboard")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "current_metrics" in data
        assert "health_status" in data
        assert "trending_data" in data
        assert "alerts" in data
        assert "system_info" in data
        
        # Check current metrics structure
        metrics = data["current_metrics"]
        assert "timestamp" in metrics
        assert "cpu_usage_percent" in metrics
        assert "memory_usage_percent" in metrics
        assert "memory_available_gb" in metrics
        assert "disk_usage_percent" in metrics
        assert "uptime_seconds" in metrics
        
        # Check health status structure
        health = data["health_status"]
        assert "overall_status" in health
        assert "components" in health
        assert "alerts" in health
        assert "last_check" in health
        
        # Check trending data structure
        trending = data["trending_data"]
        assert "cpu_usage" in trending
        assert "memory_usage" in trending
        assert "timestamps" in trending
    
    def test_current_performance_metrics_success(self):
        """Test successful current performance metrics retrieval."""
        response = client.get("/api/v1/system/performance/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check required fields
        assert "timestamp" in data
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "memory_available_gb" in data
        assert "disk_usage_percent" in data
        assert "network_bytes_sent" in data
        assert "network_bytes_recv" in data
        assert "active_tasks" in data
        assert "cache_hit_rate" in data
        assert "uptime_seconds" in data
        
        # Check data types
        assert isinstance(data["cpu_usage_percent"], (int, float))
        assert isinstance(data["memory_usage_percent"], (int, float))
        assert isinstance(data["memory_available_gb"], (int, float))
        assert isinstance(data["uptime_seconds"], (int, float))
        
        # Check reasonable values
        assert 0 <= data["cpu_usage_percent"] <= 100
        assert 0 <= data["memory_usage_percent"] <= 100
        assert data["memory_available_gb"] >= 0
    
    def test_performance_history_success(self):
        """Test successful performance history retrieval."""
        response = client.get("/api/v1/system/performance/history?hours=1&interval_minutes=5")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert "time_range" in data
        assert "data_points" in data
        assert "total_points" in data
        
        assert data["status"] == "success"
        
        # Check time range
        time_range = data["time_range"]
        assert "start" in time_range
        assert "end" in time_range
        assert "interval_minutes" in time_range
        assert time_range["interval_minutes"] == 5
        
        # Check data points
        data_points = data["data_points"]
        assert isinstance(data_points, list)
        if data_points:  # Check first point structure if exists
            point = data_points[0]
            assert "timestamp" in point
            assert "cpu_usage_percent" in point
            assert "memory_usage_percent" in point
    
    def test_performance_history_validation(self):
        """Test performance history parameter validation."""
        # Test invalid hours
        response = client.get("/api/v1/system/performance/history?hours=200")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test invalid interval
        response = client.get("/api/v1/system/performance/history?interval_minutes=100")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_comprehensive_health_status_success(self):
        """Test successful comprehensive health status retrieval."""
        response = client.get("/api/v1/system/health/comprehensive")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "overall_status" in data
        assert "components" in data
        assert "alerts" in data
        assert "recommendations" in data
        assert "last_check" in data
        
        # Check overall status is valid
        assert data["overall_status"] in ["healthy", "degraded", "critical"]
        
        # Check components structure
        components = data["components"]
        assert isinstance(components, dict)
        
        # Check alerts structure
        alerts = data["alerts"]
        assert isinstance(alerts, list)
        
        # Check recommendations structure
        recommendations = data["recommendations"]
        assert isinstance(recommendations, list)
    
    def test_system_diagnostics_success(self):
        """Test successful system diagnostics."""
        response = client.get("/api/v1/system/diagnostics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert "timestamp" in data
        assert "diagnostics" in data
        
        assert data["status"] == "success"
        
        # Check diagnostics structure
        diagnostics = data["diagnostics"]
        assert "cpu" in diagnostics
        assert "memory" in diagnostics
        assert "disk" in diagnostics
        assert "gpu" in diagnostics
        assert "services" in diagnostics
        assert "network" in diagnostics
        
        # Check CPU diagnostics
        cpu = diagnostics["cpu"]
        assert "logical_cores" in cpu
        assert "physical_cores" in cpu
        assert isinstance(cpu["logical_cores"], int)
        assert isinstance(cpu["physical_cores"], int)
        
        # Check memory diagnostics
        memory = diagnostics["memory"]
        assert "total_gb" in memory
        assert "available_gb" in memory
        assert "usage_percent" in memory
        assert isinstance(memory["total_gb"], (int, float))
        assert isinstance(memory["available_gb"], (int, float))
    
    def test_system_alerts_success(self):
        """Test successful system alerts retrieval."""
        response = client.get("/api/v1/system/alerts")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response is a list
        assert isinstance(data, list)
        
        # Check alert structure if any alerts exist
        if data:
            alert = data[0]
            assert "id" in alert
            assert "type" in alert
            assert "title" in alert
            assert "message" in alert
            assert "timestamp" in alert
            assert "component" in alert
            
            # Check alert type is valid
            assert alert["type"] in ["info", "warning", "critical"]
    
    def test_clear_system_cache_success(self):
        """Test successful system cache clearing."""
        response = client.post("/api/v1/system/maintenance/clear-cache")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert "message" in data
        assert "timestamp" in data
        
        assert data["status"] == "success"
        assert "cleared" in data["message"].lower()
    
    def test_force_garbage_collection_success(self):
        """Test successful garbage collection."""
        response = client.post("/api/v1/system/maintenance/garbage-collect")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert "objects_collected" in data
        assert "message" in data
        assert "timestamp" in data
        
        assert data["status"] == "success"
        assert isinstance(data["objects_collected"], int)
        assert data["objects_collected"] >= 0


class TestSystemMonitoringIntegration:
    """Integration tests for system monitoring."""
    
    @pytest.mark.asyncio
    async def test_monitoring_service_integration(self):
        """Test integration with monitoring service."""
        # This would test actual integration with monitoring service
        # For now, just check that the endpoint responds
        response = client.get("/api/v1/system/performance/metrics")
        assert response.status_code == status.HTTP_200_OK
    
    def test_resource_monitoring_accuracy(self):
        """Test that resource monitoring provides accurate data."""
        response = client.get("/api/v1/system/performance/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        
        # Check that values are within reasonable ranges
        assert 0 <= data["cpu_usage_percent"] <= 100
        assert 0 <= data["memory_usage_percent"] <= 100
        assert data["memory_available_gb"] > 0
        assert data["uptime_seconds"] > 0
    
    def test_health_status_consistency(self):
        """Test that health status is consistent across endpoints."""
        # Get health from comprehensive endpoint
        response1 = client.get("/api/v1/system/health/comprehensive")
        assert response1.status_code == status.HTTP_200_OK
        health1 = response1.json()
        
        # Get health from dashboard
        response2 = client.get("/api/v1/system/performance/dashboard")
        assert response2.status_code == status.HTTP_200_OK
        dashboard = response2.json()
        health2 = dashboard["health_status"]
        
        # Check consistency
        assert health1["overall_status"] == health2["overall_status"]


class TestSystemMonitoringErrors:
    """Test error handling for system monitoring endpoints."""
    
    @patch('app.api.v1.endpoints.system_monitoring._get_current_system_metrics')
    def test_performance_metrics_error_handling(self, mock_metrics):
        """Test error handling when metrics collection fails."""
        mock_metrics.side_effect = Exception("Metrics collection failed")
        
        response = client.get("/api/v1/system/performance/metrics")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        data = response.json()
        assert "detail" in data
        assert "error getting performance metrics" in data["detail"].lower()
    
    @patch('app.api.v1.endpoints.system_monitoring._get_system_health_status')
    def test_health_status_error_handling(self, mock_health):
        """Test error handling when health check fails."""
        mock_health.side_effect = Exception("Health check failed")
        
        response = client.get("/api/v1/system/health/comprehensive")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        data = response.json()
        assert "detail" in data
        assert "error getting health status" in data["detail"].lower()


class TestSystemMonitoringSecurity:
    """Test security aspects of system monitoring endpoints."""
    
    def test_monitoring_endpoints_accessible(self):
        """Test that monitoring endpoints are accessible (no auth required for ops)."""
        endpoints = [
            "/api/v1/system/performance/metrics",
            "/api/v1/system/health/comprehensive", 
            "/api/v1/system/diagnostics",
            "/api/v1/system/alerts"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should be accessible for operational monitoring
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]
            assert response.status_code != status.HTTP_401_UNAUTHORIZED
            assert response.status_code != status.HTTP_403_FORBIDDEN
    
    def test_maintenance_endpoints_exist(self):
        """Test that maintenance endpoints exist."""
        maintenance_endpoints = [
            "/api/v1/system/maintenance/clear-cache",
            "/api/v1/system/maintenance/garbage-collect"
        ]
        
        for endpoint in maintenance_endpoints:
            response = client.post(endpoint)
            # Should exist (may require auth in production)
            assert response.status_code != status.HTTP_404_NOT_FOUND


@pytest.mark.performance
class TestSystemMonitoringPerformance:
    """Performance tests for system monitoring endpoints."""
    
    def test_metrics_endpoint_performance(self):
        """Test that metrics endpoint responds quickly."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/system/performance/metrics")
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        assert (end_time - start_time) < 2.0  # Should respond within 2 seconds
    
    def test_dashboard_endpoint_performance(self):
        """Test that dashboard endpoint responds within reasonable time."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/system/performance/dashboard")
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        assert (end_time - start_time) < 5.0  # Should respond within 5 seconds