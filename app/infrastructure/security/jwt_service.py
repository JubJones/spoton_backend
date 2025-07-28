"""
JWT authentication service for secure API access.

Handles:
- JWT token generation and validation
- Token refresh mechanisms
- Role-based access control
- Secure token storage
- Token blacklisting
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import jwt
import secrets
import hashlib
from enum import Enum
import json
import aioredis

from app.core.config import settings
from app.infrastructure.cache.tracking_cache import tracking_cache

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


@dataclass
class JWTConfig:
    """JWT configuration."""
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_hours: int = 24
    issuer: str = "spoton-backend"
    audience: str = "spoton-client"


@dataclass
class TokenData:
    """Token data structure."""
    user_id: str
    username: str
    role: UserRole
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    jti: str  # JWT ID for blacklisting


@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


class JWTService:
    """
    JWT authentication service for secure API access.
    
    Features:
    - JWT token generation and validation
    - Role-based access control
    - Token refresh mechanisms
    - Secure token blacklisting
    - User management
    """
    
    def __init__(self):
        self.config = JWTConfig()
        self.redis_client: Optional[aioredis.Redis] = None
        
        # User storage (in production, this would be a database)
        self.users: Dict[str, User] = {}
        self.user_credentials: Dict[str, Dict[str, str]] = {}  # username -> {password_hash, salt}
        
        # Token blacklist
        self.token_blacklist: set = set()
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
        
        # Performance tracking
        self.jwt_stats = {
            'tokens_generated': 0,
            'tokens_validated': 0,
            'tokens_refreshed': 0,
            'failed_validations': 0,
            'users_created': 0,
            'login_attempts': 0,
            'failed_logins': 0
        }
        
        logger.info("JWTService initialized")
    
    async def initialize(self):
        """Initialize JWT service."""
        try:
            # Initialize Redis connection for token blacklisting
            await self._initialize_redis()
            
            # Create default admin user
            await self._create_default_users()
            
            # Load configuration from environment
            await self._load_configuration()
            
            logger.info("JWTService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing JWTService: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            self.redis_client = await aioredis.from_url(redis_url)
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection initialized for JWT service")
            
        except Exception as e:
            logger.warning(f"Redis connection failed for JWT service: {e}")
            # JWT service can work without Redis, but with limited blacklisting
    
    async def _create_default_users(self):
        """Create default users."""
        try:
            # Create default admin user
            admin_user = User(
                user_id="admin_001",
                username="admin",
                email="admin@spoton.local",
                role=UserRole.ADMIN,
                permissions=self._get_role_permissions(UserRole.ADMIN)
            )
            
            # Create default operator user
            operator_user = User(
                user_id="operator_001",
                username="operator",
                email="operator@spoton.local",
                role=UserRole.OPERATOR,
                permissions=self._get_role_permissions(UserRole.OPERATOR)
            )
            
            # Create default viewer user
            viewer_user = User(
                user_id="viewer_001",
                username="viewer",
                email="viewer@spoton.local",
                role=UserRole.VIEWER,
                permissions=self._get_role_permissions(UserRole.VIEWER)
            )
            
            # Set default passwords (in production, these would be set securely)
            default_password = "SpotOn2024!"
            
            await self.create_user(admin_user, default_password)
            await self.create_user(operator_user, default_password)
            await self.create_user(viewer_user, default_password)
            
            logger.info("Default users created successfully")
            
        except Exception as e:
            logger.error(f"Error creating default users: {e}")
    
    async def _load_configuration(self):
        """Load JWT configuration from environment."""
        try:
            if hasattr(settings, 'JWT_SECRET_KEY'):
                self.config.secret_key = settings.JWT_SECRET_KEY
            
            if hasattr(settings, 'JWT_ALGORITHM'):
                self.config.algorithm = settings.JWT_ALGORITHM
            
            if hasattr(settings, 'JWT_ACCESS_TOKEN_EXPIRE_MINUTES'):
                self.config.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
            
            if hasattr(settings, 'JWT_REFRESH_TOKEN_EXPIRE_HOURS'):
                self.config.refresh_token_expire_hours = settings.JWT_REFRESH_TOKEN_EXPIRE_HOURS
            
            logger.info("JWT configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading JWT configuration: {e}")
    
    # User Management
    async def create_user(self, user: User, password: str) -> bool:
        """Create a new user."""
        try:
            if user.username in self.users:
                logger.warning(f"User {user.username} already exists")
                return False
            
            # Hash password
            salt = secrets.token_hex(32)
            password_hash = self._hash_password(password, salt)
            
            # Store user
            self.users[user.username] = user
            self.user_credentials[user.username] = {
                'password_hash': password_hash,
                'salt': salt
            }
            
            self.jwt_stats['users_created'] += 1
            
            logger.info(f"User {user.username} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials."""
        try:
            self.jwt_stats['login_attempts'] += 1
            
            # Check if user exists
            if username not in self.users:
                logger.warning(f"Authentication failed: user {username} not found")
                self.jwt_stats['failed_logins'] += 1
                return None
            
            user = self.users[username]
            
            # Check if user is locked
            if user.locked_until and datetime.now(timezone.utc) < user.locked_until:
                logger.warning(f"User {username} is locked until {user.locked_until}")
                self.jwt_stats['failed_logins'] += 1
                return None
            
            # Check if user is active
            if not user.is_active:
                logger.warning(f"User {username} is inactive")
                self.jwt_stats['failed_logins'] += 1
                return None
            
            # Verify password
            credentials = self.user_credentials[username]
            password_hash = self._hash_password(password, credentials['salt'])
            
            if password_hash != credentials['password_hash']:
                # Increment failed attempts
                user.failed_login_attempts += 1
                
                # Lock user if too many failed attempts
                if user.failed_login_attempts >= self.max_failed_attempts:
                    user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=self.lockout_duration_minutes)
                    logger.warning(f"User {username} locked due to too many failed attempts")
                
                logger.warning(f"Authentication failed: invalid password for user {username}")
                self.jwt_stats['failed_logins'] += 1
                return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now(timezone.utc)
            
            logger.info(f"User {username} authenticated successfully")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            self.jwt_stats['failed_logins'] += 1
            return None
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt."""
        return hashlib.pbkdf2_hex(password.encode('utf-8'), salt.encode('utf-8'), 100000)
    
    def _get_role_permissions(self, role: UserRole) -> List[str]:
        """Get permissions for user role."""
        permissions = {
            UserRole.ADMIN: [
                'system:read', 'system:write', 'system:delete',
                'users:read', 'users:write', 'users:delete',
                'analytics:read', 'analytics:write',
                'tracking:read', 'tracking:write',
                'settings:read', 'settings:write'
            ],
            UserRole.OPERATOR: [
                'system:read',
                'analytics:read', 'analytics:write',
                'tracking:read', 'tracking:write',
                'settings:read'
            ],
            UserRole.VIEWER: [
                'analytics:read',
                'tracking:read'
            ]
        }
        
        return permissions.get(role, [])
    
    # Token Management
    async def generate_token(self, user: User, token_type: str = "access") -> str:
        """Generate JWT token."""
        try:
            now = datetime.now(timezone.utc)
            
            if token_type == "access":
                expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)
            else:  # refresh token
                expires_delta = timedelta(hours=self.config.refresh_token_expire_hours)
            
            expires_at = now + expires_delta
            jti = secrets.token_urlsafe(32)
            
            # Create token payload
            payload = {
                'sub': user.user_id,
                'username': user.username,
                'role': user.role.value,
                'permissions': user.permissions,
                'iat': now.timestamp(),
                'exp': expires_at.timestamp(),
                'iss': self.config.issuer,
                'aud': self.config.audience,
                'jti': jti,
                'type': token_type
            }
            
            # Generate token
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
            
            self.jwt_stats['tokens_generated'] += 1
            
            logger.debug(f"Generated {token_type} token for user {user.username}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating token: {e}")
            raise
    
    async def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate JWT token."""
        try:
            self.jwt_stats['tokens_validated'] += 1
            
            # Decode token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience
            )
            
            # Check if token is blacklisted
            jti = payload.get('jti')
            if jti and await self._is_token_blacklisted(jti):
                logger.warning(f"Token {jti} is blacklisted")
                self.jwt_stats['failed_validations'] += 1
                return None
            
            # Extract token data
            token_data = TokenData(
                user_id=payload['sub'],
                username=payload['username'],
                role=UserRole(payload['role']),
                permissions=payload['permissions'],
                issued_at=datetime.fromtimestamp(payload['iat'], timezone.utc),
                expires_at=datetime.fromtimestamp(payload['exp'], timezone.utc),
                jti=jti
            )
            
            logger.debug(f"Token validated for user {token_data.username}")
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            self.jwt_stats['failed_validations'] += 1
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            self.jwt_stats['failed_validations'] += 1
            return None
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            self.jwt_stats['failed_validations'] += 1
            return None
    
    async def refresh_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access token using refresh token."""
        try:
            # Validate refresh token
            token_data = await self.validate_token(refresh_token)
            
            if not token_data:
                logger.warning("Invalid refresh token")
                return None
            
            # Check if it's actually a refresh token
            payload = jwt.decode(
                refresh_token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}
            )
            
            if payload.get('type') != 'refresh':
                logger.warning("Token is not a refresh token")
                return None
            
            # Get user
            user = self.users.get(token_data.username)
            if not user or not user.is_active:
                logger.warning(f"User {token_data.username} not found or inactive")
                return None
            
            # Generate new tokens
            new_access_token = await self.generate_token(user, "access")
            new_refresh_token = await self.generate_token(user, "refresh")
            
            # Blacklist old refresh token
            await self._blacklist_token(token_data.jti)
            
            self.jwt_stats['tokens_refreshed'] += 1
            
            logger.info(f"Tokens refreshed for user {user.username}")
            
            return {
                'access_token': new_access_token,
                'refresh_token': new_refresh_token,
                'token_type': 'bearer'
            }
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke token by adding to blacklist."""
        try:
            # Validate token first
            token_data = await self.validate_token(token)
            
            if not token_data:
                logger.warning("Cannot revoke invalid token")
                return False
            
            # Add to blacklist
            await self._blacklist_token(token_data.jti)
            
            logger.info(f"Token revoked for user {token_data.username}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    async def _blacklist_token(self, jti: str):
        """Add token to blacklist."""
        try:
            # Add to local blacklist
            self.token_blacklist.add(jti)
            
            # Add to Redis blacklist if available
            if self.redis_client:
                await self.redis_client.sadd("jwt_blacklist", jti)
                # Set expiration for cleanup
                await self.redis_client.expire("jwt_blacklist", 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error blacklisting token: {e}")
    
    async def _is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        try:
            # Check local blacklist
            if jti in self.token_blacklist:
                return True
            
            # Check Redis blacklist if available
            if self.redis_client:
                return await self.redis_client.sismember("jwt_blacklist", jti)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking token blacklist: {e}")
            return False
    
    # Authorization
    def check_permission(self, token_data: TokenData, required_permission: str) -> bool:
        """Check if user has required permission."""
        try:
            # Admin has all permissions
            if token_data.role == UserRole.ADMIN:
                return True
            
            # Check specific permission
            return required_permission in token_data.permissions
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    def check_role(self, token_data: TokenData, required_role: UserRole) -> bool:
        """Check if user has required role or higher."""
        try:
            role_hierarchy = {
                UserRole.VIEWER: 1,
                UserRole.OPERATOR: 2,
                UserRole.ADMIN: 3
            }
            
            user_level = role_hierarchy.get(token_data.role, 0)
            required_level = role_hierarchy.get(required_role, 0)
            
            return user_level >= required_level
            
        except Exception as e:
            logger.error(f"Error checking role: {e}")
            return False
    
    # API Methods
    async def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login user and return tokens."""
        try:
            # Authenticate user
            user = await self.authenticate_user(username, password)
            
            if not user:
                return None
            
            # Generate tokens
            access_token = await self.generate_token(user, "access")
            refresh_token = await self.generate_token(user, "refresh")
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'bearer',
                'expires_in': self.config.access_token_expire_minutes * 60,
                'user': {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role.value,
                    'permissions': user.permissions
                }
            }
            
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return None
    
    async def logout(self, token: str) -> bool:
        """Logout user by revoking token."""
        try:
            return await self.revoke_token(token)
            
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            return False
    
    async def get_user_info(self, token_data: TokenData) -> Optional[Dict[str, Any]]:
        """Get user information from token."""
        try:
            user = self.users.get(token_data.username)
            
            if not user:
                return None
            
            return {
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value,
                'permissions': user.permissions,
                'is_active': user.is_active,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None
            }
            
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
    
    async def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users (admin only)."""
        try:
            users = []
            for user in self.users.values():
                users.append({
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role.value,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'failed_login_attempts': user.failed_login_attempts,
                    'locked_until': user.locked_until.isoformat() if user.locked_until else None
                })
            
            return users
            
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []
    
    async def update_user_role(self, username: str, new_role: UserRole) -> bool:
        """Update user role (admin only)."""
        try:
            if username not in self.users:
                return False
            
            user = self.users[username]
            user.role = new_role
            user.permissions = self._get_role_permissions(new_role)
            
            logger.info(f"User {username} role updated to {new_role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user role: {e}")
            return False
    
    async def get_jwt_statistics(self) -> Dict[str, Any]:
        """Get JWT service statistics."""
        try:
            return {
                'jwt_stats': self.jwt_stats,
                'total_users': len(self.users),
                'active_users': sum(1 for u in self.users.values() if u.is_active),
                'locked_users': sum(1 for u in self.users.values() if u.locked_until and u.locked_until > datetime.now(timezone.utc)),
                'blacklisted_tokens': len(self.token_blacklist),
                'configuration': {
                    'algorithm': self.config.algorithm,
                    'access_token_expire_minutes': self.config.access_token_expire_minutes,
                    'refresh_token_expire_hours': self.config.refresh_token_expire_hours,
                    'max_failed_attempts': self.max_failed_attempts,
                    'lockout_duration_minutes': self.lockout_duration_minutes
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting JWT statistics: {e}")
            return {}
    
    def reset_statistics(self):
        """Reset JWT statistics."""
        self.jwt_stats = {
            'tokens_generated': 0,
            'tokens_validated': 0,
            'tokens_refreshed': 0,
            'failed_validations': 0,
            'users_created': 0,
            'login_attempts': 0,
            'failed_logins': 0
        }
        logger.info("JWT statistics reset")


# Global JWT service instance
jwt_service = JWTService()