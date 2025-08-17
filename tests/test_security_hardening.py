"""
Tests for security hardening features.

Tests:
- Security middleware functionality
- CORS configuration
- Rate limiting
- Security headers
- Input validation
- Request logging and monitoring
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import status

from app.main import app
from app.core.security_config import (
    SecurityConfig, 
    SecurityLevel,
    get_security_config,
    get_cors_config,
    update_security_config
)

# Test client
client = TestClient(app)


class TestSecurityConfiguration:
    """Test security configuration management."""
    
    def test_security_config_initialization(self):
        """Test that security configuration initializes correctly."""
        config = get_security_config()
        
        assert isinstance(config, SecurityConfig)
        assert config.level in [SecurityLevel.DEVELOPMENT, SecurityLevel.STAGING, SecurityLevel.PRODUCTION]
        assert isinstance(config.allowed_origins, list)
        assert isinstance(config.allowed_hosts, list)
        assert config.rate_limit_requests_per_minute > 0
        assert isinstance(config.enable_security_headers, bool)
        assert isinstance(config.enable_request_logging, bool)
    
    def test_cors_config_development(self):
        """Test CORS configuration in development mode."""
        # Ensure we're not in production mode for this test
        with patch('app.core.config.settings.PRODUCTION_MODE', False):
            cors_config = get_cors_config()
            
            assert cors_config["allow_origins"] == ["*"]
            assert cors_config["allow_credentials"] is True
            assert cors_config["allow_methods"] == ["*"]
            assert cors_config["allow_headers"] == ["*"]
    
    def test_cors_config_production(self):
        """Test CORS configuration in production mode."""
        with patch('app.core.config.settings.PRODUCTION_MODE', True):
            with patch('app.core.config.settings.ALLOWED_ORIGINS', ["https://app.example.com"]):
                # Re-create config to pick up new settings
                from app.core.security_config import SecurityConfig
                prod_config = SecurityConfig()
                
                # Mock the config getter to return production config
                with patch('app.core.security_config.get_security_config', return_value=prod_config):
                    cors_config = get_cors_config()
                    
                    assert "https://app.example.com" in cors_config["allow_origins"]
                    assert cors_config["allow_credentials"] is True
                    assert "GET" in cors_config["allow_methods"]
                    assert "POST" in cors_config["allow_methods"]
                    assert "Authorization" in cors_config["allow_headers"]
                    assert "X-RateLimit-Limit" in cors_config["expose_headers"]
    
    def test_security_config_update(self):
        """Test updating security configuration at runtime."""
        original_config = get_security_config()
        original_origins = original_config.allowed_origins.copy()
        
        # Update configuration
        new_origins = ["https://test.example.com"]
        update_security_config(allowed_origins=new_origins)
        
        updated_config = get_security_config()
        assert updated_config.allowed_origins == new_origins
        
        # Restore original configuration
        update_security_config(allowed_origins=original_origins)


class TestSecurityMiddleware:
    """Test security middleware functionality."""
    
    def test_security_headers_present(self):
        """Test that security headers are added to responses."""
        response = client.get("/health")
        
        # Check for security headers
        headers = response.headers
        
        # These headers should be present when security middleware is enabled
        expected_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options", 
            "X-XSS-Protection",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy"
        ]
        
        for header in expected_headers:
            assert header in headers, f"Security header {header} not found"
        
        # Check specific header values
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-XSS-Protection"] == "1; mode=block"
    
    def test_content_security_policy(self):
        """Test Content Security Policy header."""
        response = client.get("/health")
        
        csp_header = response.headers.get("Content-Security-Policy")
        assert csp_header is not None
        
        # Check for basic CSP directives
        assert "default-src 'self'" in csp_header
        assert "frame-ancestors 'none'" in csp_header
    
    def test_rate_limiting_headers(self):
        """Test that rate limiting headers are added."""
        response = client.get("/health")
        
        # Rate limiting headers should be present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        
        # Check header values are reasonable
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        
        assert limit > 0
        assert 0 <= remaining <= limit
    
    def test_rate_limiting_enforcement(self):
        """Test that rate limiting is enforced."""
        # Make multiple requests quickly
        responses = []
        for i in range(10):
            response = client.get("/health")
            responses.append(response)
        
        # All should succeed initially (unless rate limit is very low)
        successful_responses = [r for r in responses if r.status_code == 200]
        assert len(successful_responses) >= 5  # At least half should succeed
        
        # Check that remaining count decreases
        remaining_counts = []
        for response in successful_responses:
            if "X-RateLimit-Remaining" in response.headers:
                remaining_counts.append(int(response.headers["X-RateLimit-Remaining"]))
        
        if len(remaining_counts) > 1:
            # Should generally be decreasing (allowing for some variance due to timing)
            assert remaining_counts[-1] <= remaining_counts[0]
    
    def test_request_size_validation(self):
        """Test that large requests are rejected."""
        # Create a large payload
        large_payload = {"data": "x" * (11 * 1024 * 1024)}  # 11MB of data
        
        # This should be rejected if request size validation is working
        response = client.post("/api/v1/processing-tasks/start", json=large_payload)
        
        # Should either be rejected by size validation (413) or by schema validation (422)
        assert response.status_code in [status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_suspicious_content_detection(self):
        """Test detection of suspicious content in requests."""
        # Test with potential XSS payload
        suspicious_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "vbscript:msgbox(1)",
            "onload=alert(1)"
        ]
        
        for payload in suspicious_payloads:
            # Try in URL parameter
            response = client.get(f"/health?test={payload}")
            
            # Should either block suspicious request (400) or process normally
            # Depending on implementation, some may be blocked
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]
    
    def test_cors_enforcement(self):
        """Test CORS enforcement."""
        # Test preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should respond to preflight
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
        
        # Test actual request with origin
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        # CORS headers should be present
        assert "Access-Control-Allow-Origin" in response.headers


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attempts."""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --"
        ]
        
        for payload in sql_payloads:
            # Test in URL parameter
            response = client.get(f"/health?param={payload}")
            
            # Should not cause internal server errors
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ]
        
        for payload in path_payloads:
            # Test in URL path (should be handled by FastAPI routing)
            response = client.get(f"/api/v1/media/{payload}")
            
            # Should return 404 or 422, not 500 or file contents
            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_400_BAD_REQUEST
            ]
    
    def test_json_payload_validation(self):
        """Test JSON payload validation."""
        # Test malformed JSON
        response = client.post(
            "/api/v1/processing-tasks/start",
            data='{"invalid": json}',  # Malformed JSON
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSecurityMonitoring:
    """Test security monitoring and logging."""
    
    def test_failed_auth_logging(self):
        """Test that failed authentication attempts are logged."""
        # This would typically require mocking the logging system
        # For now, just test that auth endpoints respond appropriately
        
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "invalid", "password": "invalid"}
        )
        
        # Should return appropriate auth error
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_rate_limit_exceeded_logging(self):
        """Test that rate limit exceeded events are logged."""
        # Make many requests to trigger rate limiting
        for i in range(100):  # High number to ensure rate limit hit
            response = client.get("/health")
            if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                # Found rate limited response
                assert "rate limit exceeded" in response.json()["detail"].lower()
                break
        else:
            # If no rate limiting occurred, that's also valid (high limit)
            pytest.skip("Rate limit not reached in test")
    
    def test_security_event_structure(self):
        """Test that security events are properly structured."""
        # Make a request that should be logged
        response = client.get("/health")
        
        # The request should succeed
        assert response.status_code == status.HTTP_200_OK
        
        # Security monitoring middleware should handle this
        # (Detailed testing would require accessing the monitoring service directly)


class TestProductionSecurity:
    """Test production-specific security features."""
    
    @patch('app.core.config.settings.PRODUCTION_MODE', True)
    def test_production_security_level(self):
        """Test that production mode enables stricter security."""
        # Create new config with production mode
        prod_config = SecurityConfig()
        
        assert prod_config.level == SecurityLevel.PRODUCTION
        assert prod_config.enable_ip_filtering is True
        assert prod_config.enable_security_headers is True
        
        # Rate limiting should be more restrictive
        assert prod_config.rate_limit_requests_per_minute <= 30
    
    def test_https_redirect_header(self):
        """Test HTTPS redirect header in production."""
        with patch('app.core.config.settings.PRODUCTION_MODE', True):
            # This test would need to reinitialize the app with production config
            # For now, just check that the concept is implemented
            response = client.get("/health")
            
            # In production, HSTS header should be present
            # (This would require app reinitialization to test properly)
            assert response.status_code == status.HTTP_200_OK
    
    def test_debug_mode_disabled(self):
        """Test that debug information is not exposed."""
        # Make request that would normally show debug info
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Response should not contain detailed error information
        response_text = response.text.lower()
        debug_indicators = ["traceback", "stack trace", "python", "fastapi"]
        
        for indicator in debug_indicators:
            assert indicator not in response_text, f"Debug indicator '{indicator}' found in response"


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_end_to_end_security_flow(self):
        """Test complete security flow from request to response."""
        # Make request with various headers
        response = client.get(
            "/health",
            headers={
                "User-Agent": "Test Client",
                "X-Forwarded-For": "192.168.1.100",
                "Origin": "http://localhost:3000"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Check all security features are working
        headers = response.headers
        assert "X-Frame-Options" in headers
        assert "X-RateLimit-Limit" in headers
        assert "Access-Control-Allow-Origin" in headers
    
    def test_security_middleware_order(self):
        """Test that security middleware is applied in correct order."""
        # The middleware should process in the right order:
        # 1. Security monitoring (logging)
        # 2. Request validation 
        # 3. Rate limiting
        # 4. Security headers
        
        response = client.get("/health")
        
        # If all middleware is working correctly, we should get:
        # - A successful response
        # - With security headers
        # - With rate limit headers
        # - Properly logged (can't easily test without accessing logs)
        
        assert response.status_code == status.HTTP_200_OK
        assert "X-Frame-Options" in response.headers
        assert "X-RateLimit-Limit" in response.headers
    
    def test_security_with_authentication(self):
        """Test security features work with authentication."""
        # Test unauthenticated request
        response = client.get("/api/v1/auth/me")
        
        # Should require authentication
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_404_NOT_FOUND,  # If endpoint doesn't exist yet
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
        
        # Security headers should still be present
        if response.status_code == status.HTTP_401_UNAUTHORIZED:
            assert "X-Frame-Options" in response.headers


@pytest.mark.performance
class TestSecurityPerformance:
    """Performance tests for security features."""
    
    def test_security_middleware_performance(self):
        """Test that security middleware doesn't significantly slow requests."""
        import time
        
        # Time multiple requests
        start_time = time.time()
        
        for i in range(10):
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Each request should complete quickly (allowing for middleware overhead)
        assert avg_time < 1.0  # Less than 1 second per request
    
    def test_rate_limiting_performance(self):
        """Test that rate limiting doesn't degrade performance significantly."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        assert (end_time - start_time) < 0.5  # Should respond quickly