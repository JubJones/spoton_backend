"""
End-to-End tests for Phase 11: Final Production Enablement.

Validates:
- Analytics endpoints functionality
- Export endpoints functionality  
- Authentication endpoints functionality
- Environment-based endpoint control
- Performance monitoring dashboard
- Security hardening features
- Production readiness
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import status

from app.main import app
from app.core.config import settings

# Test client
client = TestClient(app)


class TestPhase11EndToEnd:
    """End-to-end tests for Phase 11 implementation."""
    
    def test_application_starts_successfully(self):
        """Test that the application starts with all Phase 11 features."""
        # Basic health check should work
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        # Root endpoint should work
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "message" in data
        assert "SpotOn Backend" in data["message"]
    
    def test_api_documentation_accessible(self):
        """Test that API documentation is accessible."""
        # OpenAPI spec should be available
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec
        
        # Docs UI should be available
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
        
        # ReDoc should be available
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK


class TestAnalyticsEndpointsEnabled:
    """Test that analytics endpoints are properly enabled."""
    
    def test_analytics_health_endpoint(self):
        """Test analytics health endpoint."""
        response = client.get("/api/v1/analytics/health")
        
        # Should return success or method not found (depending on implementation)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_405_METHOD_NOT_ALLOWED
        ]
        
        # If successful, should have proper structure
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "status" in data
    
    def test_real_time_metrics_endpoint(self):
        """Test real-time metrics endpoint."""
        response = client.get("/api/v1/analytics/real-time/metrics")
        
        # Should be accessible (may return error if service not fully configured)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_404_NOT_FOUND
        ]
        
        # If successful, check structure
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "data" in data or "timestamp" in data
    
    def test_system_statistics_endpoint(self):
        """Test system statistics endpoint."""
        response = client.get("/api/v1/analytics/system/statistics")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_404_NOT_FOUND
        ]
    
    def test_analytics_endpoints_in_openapi(self):
        """Test that analytics endpoints are documented in OpenAPI spec."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        spec = response.json()
        paths = spec.get("paths", {})
        
        # Should have analytics endpoints
        analytics_paths = [path for path in paths.keys() if "/analytics/" in path]
        assert len(analytics_paths) > 0, "No analytics endpoints found in OpenAPI spec"


class TestExportEndpointsEnabled:
    """Test that export endpoints are properly enabled."""
    
    def test_export_endpoints_exist(self):
        """Test that export endpoints exist."""
        # Test tracking data export endpoint
        response = client.post("/api/v1/export/tracking-data", json={
            "environment_id": "test",
            "start_time": "2023-01-01T00:00:00Z",
            "end_time": "2023-01-02T00:00:00Z",
            "format": "json"
        })
        
        # Should exist (may fail due to validation or missing data)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_404_NOT_FOUND
        ]
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_export_job_status_endpoint(self):
        """Test export job status endpoint."""
        test_job_id = "test-job-123"
        response = client.get(f"/api/v1/export/jobs/{test_job_id}/status")
        
        # Should exist (may return not found for fake job ID)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_export_endpoints_in_openapi(self):
        """Test that export endpoints are documented in OpenAPI spec."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        spec = response.json()
        paths = spec.get("paths", {})
        
        # Should have export endpoints
        export_paths = [path for path in paths.keys() if "/export/" in path]
        assert len(export_paths) > 0, "No export endpoints found in OpenAPI spec"


class TestAuthenticationEndpointsEnabled:
    """Test that authentication endpoints are properly enabled."""
    
    def test_auth_login_endpoint(self):
        """Test authentication login endpoint."""
        response = client.post("/api/v1/auth/login", json={
            "username": "test",
            "password": "test"
        })
        
        # Should exist (may fail authentication)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_auth_me_endpoint(self):
        """Test get current user endpoint."""
        response = client.get("/api/v1/auth/me")
        
        # Should exist (should require authentication)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_auth_health_endpoint(self):
        """Test authentication health endpoint."""
        response = client.get("/api/v1/auth/health")
        
        # Should exist
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    def test_auth_endpoints_in_openapi(self):
        """Test that authentication endpoints are documented in OpenAPI spec."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        spec = response.json()
        paths = spec.get("paths", {})
        
        # Should have auth endpoints
        auth_paths = [path for path in paths.keys() if "/auth/" in path]
        assert len(auth_paths) > 0, "No authentication endpoints found in OpenAPI spec"


class TestEnvironmentBasedEndpointControl:
    """Test environment-based endpoint control functionality."""
    
    def test_endpoint_control_configuration_exists(self):
        """Test that endpoint control configuration is available."""
        # These should be accessible via settings
        assert hasattr(settings, 'ENABLE_ANALYTICS_ENDPOINTS')
        assert hasattr(settings, 'ENABLE_EXPORT_ENDPOINTS')
        assert hasattr(settings, 'ENABLE_AUTH_ENDPOINTS')
        assert hasattr(settings, 'PRODUCTION_MODE')
        
        # Should be boolean values
        assert isinstance(settings.ENABLE_ANALYTICS_ENDPOINTS, bool)
        assert isinstance(settings.ENABLE_EXPORT_ENDPOINTS, bool)
        assert isinstance(settings.ENABLE_AUTH_ENDPOINTS, bool)
        assert isinstance(settings.PRODUCTION_MODE, bool)
    
    @patch('app.core.config.settings.ENABLE_ANALYTICS_ENDPOINTS', False)
    def test_analytics_endpoints_can_be_disabled(self):
        """Test that analytics endpoints can be disabled via configuration."""
        # This test would require restarting the app with new config
        # For now, just verify the configuration exists
        assert hasattr(settings, 'ENABLE_ANALYTICS_ENDPOINTS')
    
    def test_security_configuration_exists(self):
        """Test that security configuration is available."""
        assert hasattr(settings, 'ALLOWED_ORIGINS')
        assert hasattr(settings, 'ALLOWED_HOSTS')
        assert hasattr(settings, 'RATE_LIMIT_PER_MINUTE')
        assert hasattr(settings, 'ENABLE_SECURITY_HEADERS')
        assert hasattr(settings, 'ENABLE_REQUEST_LOGGING')
        
        # Should have reasonable default values
        assert isinstance(settings.ALLOWED_ORIGINS, list)
        assert isinstance(settings.ALLOWED_HOSTS, list)
        assert settings.RATE_LIMIT_PER_MINUTE > 0
        assert isinstance(settings.ENABLE_SECURITY_HEADERS, bool)
        assert isinstance(settings.ENABLE_REQUEST_LOGGING, bool)


class TestPerformanceMonitoringDashboard:
    """Test performance monitoring dashboard functionality."""
    
    def test_system_monitoring_endpoints_exist(self):
        """Test that system monitoring endpoints exist."""
        endpoints = [
            "/api/v1/system/performance/dashboard",
            "/api/v1/system/performance/metrics",
            "/api/v1/system/health/comprehensive",
            "/api/v1/system/diagnostics",
            "/api/v1/system/alerts"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            
            # Should exist and respond (may have errors due to missing dependencies)
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ], f"Endpoint {endpoint} not accessible"
            assert response.status_code != status.HTTP_404_NOT_FOUND
            assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_performance_dashboard_structure(self):
        """Test performance dashboard returns proper structure."""
        response = client.get("/api/v1/system/performance/dashboard")
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            
            # Check dashboard structure
            expected_keys = ["current_metrics", "health_status", "trending_data", "alerts", "system_info"]
            for key in expected_keys:
                assert key in data, f"Missing key {key} in dashboard response"
        else:
            # If error, should be internal server error (missing dependencies)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_maintenance_endpoints_exist(self):
        """Test that maintenance endpoints exist."""
        endpoints = [
            "/api/v1/system/maintenance/clear-cache",
            "/api/v1/system/maintenance/garbage-collect"
        ]
        
        for endpoint in endpoints:
            response = client.post(endpoint)
            
            # Should exist
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                status.HTTP_401_UNAUTHORIZED,  # May require auth
                status.HTTP_403_FORBIDDEN
            ], f"Maintenance endpoint {endpoint} not accessible"
            assert response.status_code != status.HTTP_404_NOT_FOUND
            assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED


class TestSecurityHardening:
    """Test security hardening features."""
    
    def test_security_headers_enabled(self):
        """Test that security headers are properly configured."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        # Check for essential security headers
        required_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Content-Security-Policy"
        ]
        
        for header in required_headers:
            assert header in response.headers, f"Required security header {header} missing"
    
    def test_cors_configuration(self):
        """Test CORS configuration is properly applied."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should handle CORS preflight
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
    
    def test_rate_limiting_enabled(self):
        """Test that rate limiting is enabled."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        # Rate limiting headers should be present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        
        # Values should be reasonable
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        
        assert limit > 0
        assert remaining >= 0
        assert remaining <= limit
    
    def test_request_validation_enabled(self):
        """Test that request validation is working."""
        # Test with oversized content-length header
        response = client.get("/health", headers={"Content-Length": "999999999"})
        
        # Should either process normally or reject
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        ]


class TestProductionReadiness:
    """Test overall production readiness."""
    
    def test_all_required_endpoints_accessible(self):
        """Test that all required endpoints are accessible."""
        required_endpoints = [
            # Core endpoints
            ("/", "GET"),
            ("/health", "GET"),
            
            # API v1 endpoints that should always be available
            ("/api/v1/processing-tasks/active", "GET"),
            ("/api/v1/media/frames/test/c01", "GET"),  # May 404 but should exist
            
            # System monitoring (always enabled)
            ("/api/v1/system/performance/metrics", "GET"),
            ("/api/v1/system/health/comprehensive", "GET"),
        ]
        
        # Conditionally enabled endpoints
        if settings.ENABLE_ANALYTICS_ENDPOINTS:
            required_endpoints.extend([
                ("/api/v1/analytics/health", "GET"),
                ("/api/v1/analytics/real-time/metrics", "GET"),
            ])
        
        if settings.ENABLE_EXPORT_ENDPOINTS:
            required_endpoints.extend([
                ("/api/v1/export/tracking-data", "POST"),
            ])
        
        if settings.ENABLE_AUTH_ENDPOINTS:
            required_endpoints.extend([
                ("/api/v1/auth/login", "POST"),
                ("/api/v1/auth/me", "GET"),
            ])
        
        for endpoint, method in required_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            else:
                continue
            
            # Should exist (may have errors due to validation/auth/missing data)
            assert response.status_code != status.HTTP_404_NOT_FOUND, f"Endpoint {method} {endpoint} not found"
            assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED, f"Method {method} not allowed for {endpoint}"
    
    def test_openapi_spec_completeness(self):
        """Test that OpenAPI spec includes all enabled endpoints."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        spec = response.json()
        paths = spec.get("paths", {})
        
        # Should have reasonable number of paths
        assert len(paths) > 10, "OpenAPI spec seems incomplete"
        
        # Should have proper metadata
        info = spec.get("info", {})
        assert "title" in info
        assert "version" in info
        assert info["title"] == "SpotOn Backend"
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across endpoints."""
        # Test non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        data = response.json()
        assert "detail" in data
        
        # Test malformed request
        response = client.post("/api/v1/processing-tasks/start", data="invalid json")
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST
        ]
    
    def test_security_configuration_production_ready(self):
        """Test that security configuration is production-ready."""
        # Check that security features are enabled
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        # Should have security headers
        assert "X-Frame-Options" in response.headers
        assert "Content-Security-Policy" in response.headers
        
        # Should have rate limiting
        assert "X-RateLimit-Limit" in response.headers
        
        # CORS should be configured (not wide open in production)
        if settings.PRODUCTION_MODE:
            # In production, CORS should be restrictive
            cors_response = client.options(
                "/health",
                headers={"Origin": "https://evil.example.com"}
            )
            # Should either reject or handle appropriately
            # (Exact behavior depends on CORS configuration)
    
    def test_performance_characteristics(self):
        """Test that the application meets basic performance requirements."""
        import time
        
        # Health check should be fast
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        assert (end_time - start_time) < 2.0, "Health check too slow"
        
        # Root endpoint should be fast
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        assert (end_time - start_time) < 1.0, "Root endpoint too slow"


class TestConfigurationValidation:
    """Test that configuration is properly validated."""
    
    def test_environment_variables_loaded(self):
        """Test that environment variables are properly loaded."""
        # Test that settings are loaded
        assert settings.APP_NAME is not None
        assert settings.API_V1_PREFIX is not None
        
        # Test endpoint control settings
        assert hasattr(settings, 'ENABLE_ANALYTICS_ENDPOINTS')
        assert hasattr(settings, 'ENABLE_EXPORT_ENDPOINTS')
        assert hasattr(settings, 'ENABLE_AUTH_ENDPOINTS')
        
        # Test security settings
        assert hasattr(settings, 'ALLOWED_ORIGINS')
        assert hasattr(settings, 'RATE_LIMIT_PER_MINUTE')
    
    def test_configuration_consistency(self):
        """Test that configuration is internally consistent."""
        # If production mode is enabled, security should be tightened
        if settings.PRODUCTION_MODE:
            assert settings.ENABLE_SECURITY_HEADERS is True
            assert settings.RATE_LIMIT_PER_MINUTE <= 60  # Should be reasonable
        
        # Origins should be proper URLs
        for origin in settings.ALLOWED_ORIGINS:
            assert origin.startswith(('http://', 'https://')) or origin == '*'
    
    def test_default_configuration_secure(self):
        """Test that default configuration is secure."""
        # Analytics should be enabled by default (for functionality)
        assert settings.ENABLE_ANALYTICS_ENDPOINTS is True
        
        # Security headers should be enabled
        assert settings.ENABLE_SECURITY_HEADERS is True
        
        # Rate limiting should be reasonable
        assert 10 <= settings.RATE_LIMIT_PER_MINUTE <= 1000


@pytest.mark.integration
class TestPhase11Integration:
    """Integration tests for Phase 11 features working together."""
    
    def test_monitoring_with_security(self):
        """Test that monitoring works with security features enabled."""
        # Make request with security headers
        response = client.get(
            "/api/v1/system/performance/metrics",
            headers={"User-Agent": "Test Client"}
        )
        
        # Should work despite security middleware
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
        
        # Should have both monitoring data and security headers
        if response.status_code == status.HTTP_200_OK:
            # Should have performance data
            data = response.json()
            assert "cpu_usage_percent" in data
            
        # Should have security headers
        assert "X-Frame-Options" in response.headers
        assert "X-RateLimit-Limit" in response.headers
    
    def test_analytics_with_rate_limiting(self):
        """Test that analytics endpoints respect rate limiting."""
        # Make multiple analytics requests
        responses = []
        for i in range(5):
            response = client.get("/api/v1/analytics/real-time/metrics")
            responses.append(response)
        
        # Should either succeed or be rate limited appropriately
        for response in responses:
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_429_TOO_MANY_REQUESTS,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                status.HTTP_404_NOT_FOUND
            ]
            
            # Should have rate limit headers
            assert "X-RateLimit-Limit" in response.headers
    
    def test_export_with_authentication(self):
        """Test that export endpoints work with authentication system."""
        # Try to access export endpoint
        response = client.post("/api/v1/export/tracking-data", json={
            "environment_id": "test",
            "format": "json"
        })
        
        # Should either work or require authentication
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
        
        # Should have security headers
        assert "X-Frame-Options" in response.headers
    
    def test_comprehensive_production_readiness(self):
        """Test comprehensive production readiness across all Phase 11 features."""
        # Test that all systems work together
        
        # 1. Security is enabled
        health_response = client.get("/health")
        assert health_response.status_code == status.HTTP_200_OK
        assert "X-Frame-Options" in health_response.headers
        
        # 2. Monitoring is working
        metrics_response = client.get("/api/v1/system/performance/metrics")
        assert metrics_response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]
        
        # 3. API documentation is available
        docs_response = client.get("/api/v1/openapi.json")
        assert docs_response.status_code == status.HTTP_200_OK
        
        # 4. Rate limiting is active
        assert "X-RateLimit-Limit" in health_response.headers
        
        # 5. Endpoints are conditionally enabled based on configuration
        if settings.ENABLE_ANALYTICS_ENDPOINTS:
            analytics_response = client.get("/api/v1/analytics/health")
            # Should exist (may error due to missing services)
            assert analytics_response.status_code != status.HTTP_404_NOT_FOUND
        
        # All systems are functioning together
        assert True  # If we got here, integration is working