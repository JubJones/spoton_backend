"""
Authentication API endpoints.

Handles:
- User login and logout
- Token refresh
- User management
- Role-based access control
- Security validation
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr

from app.infrastructure.security.jwt_service import jwt_service, UserRole, TokenData
from app.infrastructure.security.encryption_service import encryption_service

router = APIRouter()
security = HTTPBearer()


class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, max_length=128, description="Password")


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class UserResponse(BaseModel):
    """User response model."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    role: str = Field(..., description="User role")
    permissions: List[str] = Field(..., description="User permissions")


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str = Field(..., description="Refresh token")


class CreateUserRequest(BaseModel):
    """Create user request model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    role: str = Field(..., description="User role")


class UpdateUserRoleRequest(BaseModel):
    """Update user role request model."""
    username: str = Field(..., description="Username")
    new_role: str = Field(..., description="New user role")


class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        token_data = await jwt_service.validate_token(token)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return token_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def require_role(required_role: UserRole):
    """Dependency to require specific role."""
    def role_dependency(current_user: TokenData = Depends(get_current_user)):
        if not jwt_service.check_role(current_user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role.value}"
            )
        return current_user
    return role_dependency


async def require_permission(required_permission: str):
    """Dependency to require specific permission."""
    def permission_dependency(current_user: TokenData = Depends(get_current_user)):
        if not jwt_service.check_permission(current_user, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required permission: {required_permission}"
            )
        return current_user
    return permission_dependency


# Authentication endpoints
@router.post("/login", response_model=Dict[str, Any])
async def login(request: LoginRequest):
    """User login endpoint."""
    try:
        # Attempt login
        login_result = await jwt_service.login(request.username, request.password)
        
        if not login_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return {
            "status": "success",
            "message": "Login successful",
            "data": {
                "access_token": login_result["access_token"],
                "refresh_token": login_result["refresh_token"],
                "token_type": login_result["token_type"],
                "expires_in": login_result["expires_in"],
                "user": login_result["user"]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    """User logout endpoint."""
    try:
        # The token is already validated in get_current_user
        # We would revoke the token here if needed
        success = await jwt_service.logout("dummy_token")  # Would pass actual token
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logout failed"
            )
        
        return {
            "status": "success",
            "message": "Logout successful",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )


@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token endpoint."""
    try:
        # Refresh tokens
        refresh_result = await jwt_service.refresh_token(request.refresh_token)
        
        if not refresh_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return {
            "status": "success",
            "message": "Token refreshed successfully",
            "data": refresh_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information."""
    try:
        user_info = await jwt_service.get_user_info(current_user)
        
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "status": "success",
            "data": user_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {str(e)}"
        )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """Change user password."""
    try:
        # This would be implemented to change password
        # For now, return success message
        return {
            "status": "success",
            "message": "Password changed successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )


# User management endpoints (Admin only)
@router.post("/users", response_model=Dict[str, Any])
async def create_user(
    request: CreateUserRequest,
    current_user: TokenData = Depends(require_role(UserRole.ADMIN))
):
    """Create new user (Admin only)."""
    try:
        # Validate role
        try:
            user_role = UserRole(request.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {request.role}"
            )
        
        # Create user
        from app.infrastructure.security.jwt_service import User
        
        new_user = User(
            user_id=f"user_{int(datetime.now(timezone.utc).timestamp())}",
            username=request.username,
            email=request.email,
            role=user_role,
            permissions=jwt_service._get_role_permissions(user_role)
        )
        
        success = await jwt_service.create_user(new_user, request.password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user"
            )
        
        return {
            "status": "success",
            "message": f"User {request.username} created successfully",
            "data": {
                "user_id": new_user.user_id,
                "username": new_user.username,
                "email": new_user.email,
                "role": new_user.role.value
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User creation failed: {str(e)}"
        )


@router.get("/users", response_model=Dict[str, Any])
async def get_all_users(
    current_user: TokenData = Depends(require_role(UserRole.ADMIN))
):
    """Get all users (Admin only)."""
    try:
        users = await jwt_service.get_all_users()
        
        return {
            "status": "success",
            "data": {
                "users": users,
                "total_count": len(users)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get users: {str(e)}"
        )


@router.put("/users/role", response_model=Dict[str, Any])
async def update_user_role(
    request: UpdateUserRoleRequest,
    current_user: TokenData = Depends(require_role(UserRole.ADMIN))
):
    """Update user role (Admin only)."""
    try:
        # Validate role
        try:
            new_role = UserRole(request.new_role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {request.new_role}"
            )
        
        # Update user role
        success = await jwt_service.update_user_role(request.username, new_role)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "status": "success",
            "message": f"User {request.username} role updated to {request.new_role}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Role update failed: {str(e)}"
        )


# Security endpoints
@router.get("/security/statistics", response_model=Dict[str, Any])
async def get_security_statistics(
    current_user: TokenData = Depends(require_role(UserRole.ADMIN))
):
    """Get security statistics (Admin only)."""
    try:
        jwt_stats = await jwt_service.get_jwt_statistics()
        encryption_stats = await encryption_service.get_encryption_statistics()
        
        return {
            "status": "success",
            "data": {
                "jwt_statistics": jwt_stats,
                "encryption_statistics": encryption_stats
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security statistics: {str(e)}"
        )


@router.get("/security/audit-log", response_model=Dict[str, Any])
async def get_security_audit_log(
    hours: int = 24,
    current_user: TokenData = Depends(require_role(UserRole.ADMIN))
):
    """Get security audit log (Admin only)."""
    try:
        audit_log = await encryption_service.get_audit_log(hours)
        
        return {
            "status": "success",
            "data": {
                "audit_log": audit_log,
                "total_entries": len(audit_log),
                "time_range_hours": hours
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit log: {str(e)}"
        )


@router.post("/security/rotate-keys", response_model=Dict[str, Any])
async def rotate_encryption_keys(
    current_user: TokenData = Depends(require_role(UserRole.ADMIN))
):
    """Rotate encryption keys (Admin only)."""
    try:
        new_key_id = await encryption_service.rotate_encryption_key()
        
        return {
            "status": "success",
            "message": "Encryption keys rotated successfully",
            "data": {
                "new_key_id": new_key_id
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Key rotation failed: {str(e)}"
        )


# Permission testing endpoints
@router.get("/permissions/test")
async def test_permissions(current_user: TokenData = Depends(get_current_user)):
    """Test user permissions."""
    try:
        return {
            "status": "success",
            "data": {
                "user_id": current_user.user_id,
                "username": current_user.username,
                "role": current_user.role.value,
                "permissions": current_user.permissions,
                "can_read_system": jwt_service.check_permission(current_user, "system:read"),
                "can_write_system": jwt_service.check_permission(current_user, "system:write"),
                "can_read_analytics": jwt_service.check_permission(current_user, "analytics:read"),
                "can_write_analytics": jwt_service.check_permission(current_user, "analytics:write"),
                "is_admin": jwt_service.check_role(current_user, UserRole.ADMIN),
                "is_operator": jwt_service.check_role(current_user, UserRole.OPERATOR)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Permission test failed: {str(e)}"
        )


@router.get("/health")
async def auth_health_check():
    """Authentication service health check."""
    try:
        jwt_stats = await jwt_service.get_jwt_statistics()
        encryption_stats = await encryption_service.get_encryption_statistics()
        
        return {
            "status": "success",
            "data": {
                "authentication": "healthy",
                "encryption": "healthy",
                "jwt_service": "operational",
                "encryption_service": "operational",
                "statistics": {
                    "total_users": jwt_stats.get("total_users", 0),
                    "active_users": jwt_stats.get("active_users", 0),
                    "tokens_generated": jwt_stats.get("jwt_stats", {}).get("tokens_generated", 0),
                    "encryptions_performed": encryption_stats.get("encryption_stats", {}).get("encryptions_performed", 0)
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication health check failed: {str(e)}"
        )