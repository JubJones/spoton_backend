"""Authentication models and data structures."""

from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["User", "UserRole", "TokenData"]


class UserRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    OPERATOR = "operator" 
    VIEWER = "viewer"


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