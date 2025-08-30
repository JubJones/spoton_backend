"""
Security configuration and hardening for production deployment.

Provides:
- CORS security configuration
- Security headers middleware
- Rate limiting protection
- Input validation and sanitization
- Security monitoring and logging
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
from collections import defaultdict, deque
import ipaddress
import re
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import secrets
import hashlib

from app.core.config import settings

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security configuration levels."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass  
class SecurityConfig:
    """Security configuration settings."""
    level: SecurityLevel = SecurityLevel.DEVELOPMENT
    allowed_origins: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10
    enable_security_headers: bool = True
    enable_request_logging: bool = True
    enable_ip_filtering: bool = False
    blocked_ips: Set[str] = field(default_factory=set)
    allowed_ips: Optional[Set[str]] = None
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    session_timeout_minutes: int = 60
    
    def __post_init__(self):
        """Configure security settings based on production mode and settings."""
        # Get values from settings
        self.allowed_origins = settings.ALLOWED_ORIGINS
        self.allowed_hosts = settings.ALLOWED_HOSTS  
        self.rate_limit_requests_per_minute = settings.RATE_LIMIT_PER_MINUTE
        self.enable_security_headers = settings.ENABLE_SECURITY_HEADERS
        self.enable_request_logging = settings.ENABLE_REQUEST_LOGGING
        self.max_request_size = settings.MAX_REQUEST_SIZE_MB * 1024 * 1024
        
        if settings.PRODUCTION_MODE:
            self.level = SecurityLevel.PRODUCTION
            # In production, be more restrictive
            if not self.allowed_origins or self.allowed_origins == ["http://localhost:3000", "http://localhost:5173"]:
                # These should be overridden with actual production URLs
                logger.warning("Production mode enabled but no production origins configured. CORS will be restrictive.")
                self.allowed_origins = []
            self.rate_limit_requests_per_minute = min(self.rate_limit_requests_per_minute, 30)
            self.enable_ip_filtering = True
            self.enable_security_headers = True
            self.max_request_size = min(self.max_request_size, 5 * 1024 * 1024)  # Max 5MB for production


# Global security configuration - will be initialized after settings are loaded
security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get or create security configuration."""
    global security_config
    if security_config is None:
        security_config = SecurityConfig()
    return security_config


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    def __init__(self, app: Callable, config: SecurityConfig):
        super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            
            if response is None:
                logger.error("SecurityHeadersMiddleware: No response returned from next middleware")
                from fastapi.responses import JSONResponse
                response = JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error"}
                )
        except Exception as e:
            logger.error(f"SecurityHeadersMiddleware: Exception in middleware chain: {e}")
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
        
        if self.config.enable_security_headers:
            # Prevent clickjacking
            response.headers["X-Frame-Options"] = "DENY"
            
            # Prevent MIME type sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"
            
            # Enable XSS protection
            response.headers["X-XSS-Protection"] = "1; mode=block"
            
            # Enforce HTTPS in production
            if self.config.level == SecurityLevel.PRODUCTION:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            # Content Security Policy
            csp_policy = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
            response.headers["Content-Security-Policy"] = csp_policy
            
            # Referrer Policy
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Permissions Policy
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse."""
    
    def __init__(self, app: Callable, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_until: Dict[str, datetime] = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        current_time = datetime.now(timezone.utc)
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_until:
            if current_time < self.blocked_until[client_ip]:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Try again later."}
                )
            else:
                del self.blocked_until[client_ip]
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        requests = self.request_counts[client_ip]
        
        # Remove old requests
        while requests and datetime.fromisoformat(requests[0]) < cutoff_time:
            requests.popleft()
        
        # Check rate limit
        if len(requests) >= self.config.rate_limit_requests_per_minute:
            # Block IP for 5 minutes
            self.blocked_until[client_ip] = current_time + timedelta(minutes=5)
            
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": "300"  # 5 minutes
                }
            )
        
        # Add current request
        requests.append(current_time.isoformat())
        
        try:
            response = await call_next(request)
            
            if response is None:
                logger.error("RateLimitingMiddleware: No response returned from next middleware")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Internal server error"}
                )
        except Exception as e:
            logger.error(f"RateLimitingMiddleware: Exception in middleware chain: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.config.rate_limit_requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.config.rate_limit_requests_per_minute - len(requests))
        response.headers["X-RateLimit-Reset"] = str(int((cutoff_time + timedelta(minutes=1)).timestamp()))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (when behind a proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        client_host = request.client.host if request.client else "unknown"
        return client_host


class IPFilteringMiddleware(BaseHTTPMiddleware):
    """IP filtering middleware for additional security."""
    
    def __init__(self, app: Callable, config: SecurityConfig):
        super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next):
        if not self.config.enable_ip_filtering:
            try:
                response = await call_next(request)
                if response is None:
                    logger.error("IPFilteringMiddleware: No response returned from next middleware")
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"detail": "Internal server error"}
                    )
                return response
            except Exception as e:
                logger.error(f"IPFilteringMiddleware: Exception in middleware chain: {e}")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Internal server error"}
                )
        
        client_ip = self._get_client_ip(request)
        
        # Check blocked IPs
        if client_ip in self.config.blocked_ips:
            logger.warning(f"Blocked request from IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"}
            )
        
        # Check allowed IPs (if whitelist is configured)
        if self.config.allowed_ips and client_ip not in self.config.allowed_ips:
            logger.warning(f"Request from non-whitelisted IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"}
            )
        
        try:
            response = await call_next(request)
            if response is None:
                logger.error("IPFilteringMiddleware: No response returned from next middleware")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Internal server error"}
                )
            return response
        except Exception as e:
            logger.error(f"IPFilteringMiddleware: Exception in middleware chain: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        client_host = request.client.host if request.client else "unknown"
        return client_host


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Request validation and sanitization middleware."""
    
    def __init__(self, app: Callable, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.suspicious_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE),
        ]
    
    async def dispatch(self, request: Request, call_next):
        logger.debug(f"RequestValidationMiddleware: Processing request {request.method} {request.url.path}")
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.config.max_request_size:
            logger.debug(f"RequestValidationMiddleware: Request too large {content_length}")
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request too large"}
            )
        
        # Check for suspicious patterns in URL and headers
        url_str = str(request.url)
        if self._contains_suspicious_content(url_str):
            logger.warning(f"Suspicious URL detected: {url_str}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid request"}
            )
        
        # Check headers for suspicious content
        for header_name, header_value in request.headers.items():
            if self._contains_suspicious_content(header_value):
                logger.warning(f"Suspicious header content detected: {header_name}: {header_value}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid request"}
                )
        
        logger.debug(f"RequestValidationMiddleware: Calling next middleware for {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            
            if response is None:
                logger.error(f"RequestValidationMiddleware: No response returned from next middleware for {request.method} {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Internal server error"}
                )
            
            logger.debug(f"RequestValidationMiddleware: Got response {response.status_code} for {request.method} {request.url.path}")
            return response
        except Exception as e:
            logger.error(f"RequestValidationMiddleware: Exception in {request.method} {request.url.path}: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
    
    def _contains_suspicious_content(self, content: str) -> bool:
        """Check if content contains suspicious patterns."""
        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                return True
        return False


class SecurityMonitoringMiddleware(BaseHTTPMiddleware):
    """Security monitoring and logging middleware."""
    
    def __init__(self, app: Callable, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.security_events: deque = deque(maxlen=1000)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        # Log entry to debug middleware chain
        logger.debug(f"SecurityMonitoringMiddleware: Processing request {request.method} {request.url.path}")
        
        # Initialize security event data regardless of logging setting
        security_event = None
        if self.config.enable_request_logging:
            security_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ip": client_ip,
                "method": request.method,
                "path": request.url.path,
                "user_agent": request.headers.get("user-agent", ""),
                "referer": request.headers.get("referer", ""),
            }
        
        try:
            logger.debug(f"SecurityMonitoringMiddleware: Calling next middleware for {request.method} {request.url.path}")
            response = await call_next(request)
            logger.debug(f"SecurityMonitoringMiddleware: Got response {response.status_code} for {request.method} {request.url.path}")
            
            # Log completed request
            if self.config.enable_request_logging and security_event is not None:
                security_event.update({
                    "status_code": response.status_code,
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                })
                
                # Log suspicious activity
                if response.status_code in [401, 403, 429]:
                    logger.warning(f"Security event: {security_event}")
                
                self.security_events.append(security_event)
            
            logger.debug(f"SecurityMonitoringMiddleware: Returning response for {request.method} {request.url.path}")
            return response
            
        except Exception as e:
            logger.error(f"SecurityMonitoringMiddleware: Exception in {request.method} {request.url.path}: {str(e)}")
            # Log errors
            if self.config.enable_request_logging and security_event is not None:
                security_event.update({
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                })
                logger.error(f"Request error: {security_event}")
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        client_host = request.client.host if request.client else "unknown"
        return client_host


def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration based on security level."""
    config = get_security_config()
    if config.level == SecurityLevel.PRODUCTION:
        return {
            "allow_origins": config.allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            "allow_headers": [
                "Authorization",
                "Content-Type",
                "X-Requested-With",
                "Accept",
                "Origin",
                "User-Agent",
                "DNT",
                "Cache-Control",
                "X-Mx-ReqToken",
                "Keep-Alive",
                "X-Requested-With",
            ],
            "expose_headers": [
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining", 
                "X-RateLimit-Reset"
            ]
        }
    else:
        # Development/Staging - more permissive
        return {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }


def configure_security_middleware(app):
    """Configure all security middleware for the application."""
    config = get_security_config()
    
    # Add security monitoring (should be first to log everything)
    app.add_middleware(SecurityMonitoringMiddleware, config=config)
    
    # Add request validation
    app.add_middleware(RequestValidationMiddleware, config=config)
    
    # Add IP filtering (if enabled)
    if config.enable_ip_filtering:
        app.add_middleware(IPFilteringMiddleware, config=config)
    
    # Add rate limiting
    app.add_middleware(RateLimitingMiddleware, config=config)
    
    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware, config=config)
    
    # Add trusted host middleware for production
    if config.level == SecurityLevel.PRODUCTION and config.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.allowed_hosts)
    
    logger.info(f"Security middleware configured for {config.level.value} environment")


def update_security_config(
    allowed_origins: Optional[List[str]] = None,
    allowed_hosts: Optional[List[str]] = None,
    rate_limit: Optional[int] = None,
    blocked_ips: Optional[Set[str]] = None,
    allowed_ips: Optional[Set[str]] = None
):
    """Update security configuration at runtime."""
    global security_config
    
    if allowed_origins is not None:
        security_config.allowed_origins = allowed_origins
    
    if allowed_hosts is not None:
        security_config.allowed_hosts = allowed_hosts
    
    if rate_limit is not None:
        security_config.rate_limit_requests_per_minute = rate_limit
    
    if blocked_ips is not None:
        security_config.blocked_ips = blocked_ips
    
    if allowed_ips is not None:
        security_config.allowed_ips = allowed_ips
    
    logger.info("Security configuration updated")


# Export the configured security instance
__all__ = [
    "SecurityConfig",
    "SecurityLevel", 
    "security_config",
    "configure_security_middleware",
    "get_cors_config",
    "update_security_config"
]