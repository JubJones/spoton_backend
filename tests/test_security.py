"""
Security tests for JWT authentication and encryption services.

Tests:
- JWT token generation and validation
- Role-based access control
- Encryption/decryption functionality
- Security audit logging
- GDPR compliance
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import jwt as python_jwt
import secrets
import hashlib

from app.infrastructure.security.jwt_service import jwt_service
from app.infrastructure.auth.models import User, UserRole, TokenData
from app.infrastructure.security.encryption_service import encryption_service, EncryptionAlgorithm, HashAlgorithm


@pytest.mark.security
class TestJWTService:
    """Test JWT authentication service."""
    
    @pytest.mark.asyncio
    async def test_jwt_service_initialization(self):
        """Test JWT service initialization."""
        await jwt_service.initialize()
        
        # Check that default users are created
        assert len(jwt_service.users) >= 3
        assert 'admin' in jwt_service.users
        assert 'operator' in jwt_service.users
        assert 'viewer' in jwt_service.users
        
        # Check that JWT stats are initialized
        stats = await jwt_service.get_jwt_statistics()
        assert 'jwt_stats' in stats
        assert stats['total_users'] >= 3
    
    @pytest.mark.asyncio
    async def test_user_creation(self):
        """Test user creation functionality."""
        await jwt_service.initialize()
        
        # Create test user
        test_user = User(
            user_id="test_user_123",
            username="testuser123",
            email="test123@example.com",
            role=UserRole.OPERATOR,
            permissions=["analytics:read", "tracking:read"]
        )
        
        # Create user
        result = await jwt_service.create_user(test_user, "test_password_123")
        assert result is True
        
        # Verify user exists
        assert test_user.username in jwt_service.users
        assert test_user.username in jwt_service.user_credentials
        
        # Try to create duplicate user
        duplicate_result = await jwt_service.create_user(test_user, "different_password")
        assert duplicate_result is False
    
    @pytest.mark.asyncio
    async def test_user_authentication(self):
        """Test user authentication."""
        await jwt_service.initialize()
        
        # Test valid authentication
        user = await jwt_service.authenticate_user("admin", "SpotOn2024!")
        assert user is not None
        assert user.username == "admin"
        assert user.role == UserRole.ADMIN
        
        # Test invalid username
        invalid_user = await jwt_service.authenticate_user("nonexistent", "password")
        assert invalid_user is None
        
        # Test invalid password
        invalid_password = await jwt_service.authenticate_user("admin", "wrong_password")
        assert invalid_password is None
    
    @pytest.mark.asyncio
    async def test_token_generation_and_validation(self):
        """Test JWT token generation and validation."""
        await jwt_service.initialize()
        
        # Get test user
        user = jwt_service.users["admin"]
        
        # Generate access token
        access_token = await jwt_service.generate_token(user, "access")
        assert access_token is not None
        assert isinstance(access_token, str)
        
        # Generate refresh token
        refresh_token = await jwt_service.generate_token(user, "refresh")
        assert refresh_token is not None
        assert isinstance(refresh_token, str)
        
        # Validate access token
        token_data = await jwt_service.validate_token(access_token)
        assert token_data is not None
        assert token_data.user_id == user.user_id
        assert token_data.username == user.username
        assert token_data.role == user.role
        
        # Test token expiration (mock expired token)
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = python_jwt.ExpiredSignatureError("Token has expired")
            expired_token_data = await jwt_service.validate_token(access_token)
            assert expired_token_data is None
    
    @pytest.mark.asyncio
    async def test_token_refresh(self):
        """Test token refresh functionality."""
        await jwt_service.initialize()
        
        # Get test user
        user = jwt_service.users["admin"]
        
        # Generate refresh token
        refresh_token = await jwt_service.generate_token(user, "refresh")
        
        # Refresh tokens
        new_tokens = await jwt_service.refresh_token(refresh_token)
        assert new_tokens is not None
        assert 'access_token' in new_tokens
        assert 'refresh_token' in new_tokens
        assert 'token_type' in new_tokens
        
        # Validate new access token
        token_data = await jwt_service.validate_token(new_tokens['access_token'])
        assert token_data is not None
        assert token_data.username == user.username
    
    @pytest.mark.asyncio
    async def test_role_based_access_control(self):
        """Test role-based access control."""
        await jwt_service.initialize()
        
        # Get users with different roles
        admin_user = jwt_service.users["admin"]
        operator_user = jwt_service.users["operator"]
        viewer_user = jwt_service.users["viewer"]
        
        # Generate tokens
        admin_token = await jwt_service.generate_token(admin_user)
        operator_token = await jwt_service.generate_token(operator_user)
        viewer_token = await jwt_service.generate_token(viewer_user)
        
        # Validate tokens
        admin_data = await jwt_service.validate_token(admin_token)
        operator_data = await jwt_service.validate_token(operator_token)
        viewer_data = await jwt_service.validate_token(viewer_token)
        
        # Test role checking
        assert jwt_service.check_role(admin_data, UserRole.ADMIN)
        assert jwt_service.check_role(admin_data, UserRole.OPERATOR)
        assert jwt_service.check_role(admin_data, UserRole.VIEWER)
        
        assert not jwt_service.check_role(operator_data, UserRole.ADMIN)
        assert jwt_service.check_role(operator_data, UserRole.OPERATOR)
        assert jwt_service.check_role(operator_data, UserRole.VIEWER)
        
        assert not jwt_service.check_role(viewer_data, UserRole.ADMIN)
        assert not jwt_service.check_role(viewer_data, UserRole.OPERATOR)
        assert jwt_service.check_role(viewer_data, UserRole.VIEWER)
        
        # Test permission checking
        assert jwt_service.check_permission(admin_data, "system:write")
        assert jwt_service.check_permission(operator_data, "analytics:read")
        assert jwt_service.check_permission(viewer_data, "tracking:read")
        
        assert not jwt_service.check_permission(operator_data, "system:write")
        assert not jwt_service.check_permission(viewer_data, "analytics:write")
    
    @pytest.mark.asyncio
    async def test_token_blacklisting(self):
        """Test token blacklisting functionality."""
        await jwt_service.initialize()
        
        # Get test user
        user = jwt_service.users["admin"]
        
        # Generate token
        token = await jwt_service.generate_token(user)
        
        # Validate token initially
        token_data = await jwt_service.validate_token(token)
        assert token_data is not None
        
        # Revoke token
        revoke_result = await jwt_service.revoke_token(token)
        assert revoke_result is True
        
        # Try to validate revoked token
        revoked_token_data = await jwt_service.validate_token(token)
        assert revoked_token_data is None
    
    @pytest.mark.asyncio
    async def test_failed_login_attempts(self):
        """Test failed login attempt tracking."""
        await jwt_service.initialize()
        
        # Get test user
        user = jwt_service.users["admin"]
        initial_failed_attempts = user.failed_login_attempts
        
        # Perform failed login attempts
        for i in range(3):
            failed_user = await jwt_service.authenticate_user("admin", "wrong_password")
            assert failed_user is None
        
        # Check failed attempts counter
        assert user.failed_login_attempts == initial_failed_attempts + 3
        
        # Successful login should reset counter
        successful_user = await jwt_service.authenticate_user("admin", "SpotOn2024!")
        assert successful_user is not None
        assert user.failed_login_attempts == 0
    
    @pytest.mark.asyncio
    async def test_user_lockout(self):
        """Test user lockout after max failed attempts."""
        await jwt_service.initialize()
        
        # Create test user
        test_user = User(
            user_id="lockout_test_user",
            username="lockout_user",
            email="lockout@example.com",
            role=UserRole.VIEWER,
            permissions=["tracking:read"]
        )
        
        await jwt_service.create_user(test_user, "correct_password")
        
        # Perform max failed attempts
        for i in range(jwt_service.max_failed_attempts):
            failed_user = await jwt_service.authenticate_user("lockout_user", "wrong_password")
            assert failed_user is None
        
        # User should be locked
        assert test_user.locked_until is not None
        assert test_user.locked_until > datetime.now(timezone.utc)
        
        # Even correct password should fail when locked
        locked_user = await jwt_service.authenticate_user("lockout_user", "correct_password")
        assert locked_user is None


@pytest.mark.security
class TestEncryptionService:
    """Test encryption service."""
    
    @pytest.mark.asyncio
    async def test_encryption_service_initialization(self):
        """Test encryption service initialization."""
        await encryption_service.initialize()
        
        # Check that master key is generated
        assert encryption_service.active_key_id is not None
        assert encryption_service.active_key_id in encryption_service.encryption_keys
        
        # Check that RSA keys are generated
        assert encryption_service.rsa_private_key is not None
        assert encryption_service.rsa_public_key is not None
        
        # Check statistics
        stats = await encryption_service.get_encryption_statistics()
        assert 'encryption_stats' in stats
        assert stats['total_keys'] >= 1
    
    @pytest.mark.asyncio
    async def test_symmetric_encryption_decryption(self):
        """Test symmetric encryption and decryption."""
        await encryption_service.initialize()
        
        # Test data
        test_data = "This is sensitive test data"
        
        # Encrypt data
        encrypted_data = await encryption_service.encrypt_data(test_data)
        assert encrypted_data is not None
        assert encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert encrypted_data.key_id == encryption_service.active_key_id
        
        # Decrypt data
        decrypted_data = await encryption_service.decrypt_data(encrypted_data)
        assert decrypted_data is not None
        assert decrypted_data.decode('utf-8') == test_data
    
    @pytest.mark.asyncio
    async def test_asymmetric_encryption_decryption(self):
        """Test asymmetric encryption and decryption."""
        await encryption_service.initialize()
        
        # Test data (small data for RSA)
        test_data = b"Small test data for RSA"
        
        # Encrypt with RSA
        encrypted_data = await encryption_service.encrypt_with_rsa(test_data)
        assert encrypted_data is not None
        assert len(encrypted_data) > 0
        
        # Decrypt with RSA
        decrypted_data = await encryption_service.decrypt_with_rsa(encrypted_data)
        assert decrypted_data == test_data
    
    @pytest.mark.asyncio
    async def test_hashing_functionality(self):
        """Test hashing functionality."""
        await encryption_service.initialize()
        
        # Test data
        test_data = "password123"
        
        # Test SHA256 hashing
        hash_value, salt = await encryption_service.hash_data(test_data, HashAlgorithm.SHA256)
        assert hash_value is not None
        assert len(hash_value) == 64  # SHA256 hex length
        assert len(salt) == 0  # No salt for SHA256
        
        # Test PBKDF2 hashing
        pbkdf2_hash, pbkdf2_salt = await encryption_service.hash_data(test_data, HashAlgorithm.PBKDF2_SHA256)
        assert pbkdf2_hash is not None
        assert len(pbkdf2_hash) == 64  # SHA256 hex length
        assert len(pbkdf2_salt) == 32  # Salt length
        
        # Test hash verification
        is_valid = await encryption_service.verify_hash(test_data, pbkdf2_hash, pbkdf2_salt, HashAlgorithm.PBKDF2_SHA256)
        assert is_valid is True
        
        # Test invalid hash verification
        is_invalid = await encryption_service.verify_hash("wrong_password", pbkdf2_hash, pbkdf2_salt, HashAlgorithm.PBKDF2_SHA256)
        assert is_invalid is False
    
    @pytest.mark.asyncio
    async def test_key_rotation(self):
        """Test encryption key rotation."""
        await encryption_service.initialize()
        
        # Get initial key
        initial_key_id = encryption_service.active_key_id
        initial_key = encryption_service.encryption_keys[initial_key_id]
        
        # Rotate key
        new_key_id = await encryption_service.rotate_encryption_key()
        assert new_key_id != initial_key_id
        assert encryption_service.active_key_id == new_key_id
        
        # Check that old key is marked as inactive
        assert initial_key.is_active is False
        
        # Check that new key is active
        new_key = encryption_service.encryption_keys[new_key_id]
        assert new_key.is_active is True
    
    @pytest.mark.asyncio
    async def test_data_masking(self):
        """Test sensitive data masking."""
        await encryption_service.initialize()
        
        # Test partial masking
        test_data = "sensitive_data_123"
        partial_masked = await encryption_service.mask_sensitive_data(test_data, "partial")
        assert partial_masked == "se***********23"
        
        # Test full masking
        full_masked = await encryption_service.mask_sensitive_data(test_data, "full")
        assert full_masked == "*" * len(test_data)
        
        # Test email masking
        email = "test@example.com"
        email_masked = await encryption_service.mask_sensitive_data(email, "email")
        assert email_masked == "t**t@example.com"
        
        # Test phone masking
        phone = "1234567890"
        phone_masked = await encryption_service.mask_sensitive_data(phone, "phone")
        assert phone_masked == "123****890"
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        await encryption_service.initialize()
        
        # Test personal data
        personal_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "1234567890",
            "address": "123 Main St",
            "age": 30  # Non-sensitive field
        }
        
        # Encrypt personal data
        encrypted_personal_data = await encryption_service.encrypt_personal_data(personal_data)
        assert encrypted_personal_data is not None
        
        # Check that sensitive fields are encrypted
        assert encrypted_personal_data['name']['encrypted'] is True
        assert encrypted_personal_data['email']['encrypted'] is True
        assert encrypted_personal_data['phone']['encrypted'] is True
        assert encrypted_personal_data['address']['encrypted'] is True
        
        # Check that non-sensitive field is not encrypted
        assert encrypted_personal_data['age'] == 30
        
        # Decrypt personal data
        decrypted_personal_data = await encryption_service.decrypt_personal_data(encrypted_personal_data)
        assert decrypted_personal_data == personal_data
        
        # Test GDPR access request
        access_data = await encryption_service.process_gdpr_request("access", encrypted_personal_data)
        assert access_data is not None
        assert "name" in access_data
        assert "email" in access_data
        
        # Test GDPR portability request
        portability_data = await encryption_service.process_gdpr_request("portability", encrypted_personal_data)
        assert portability_data == personal_data
        
        # Test GDPR erasure request
        erasure_data = await encryption_service.process_gdpr_request("erasure", encrypted_personal_data)
        assert erasure_data['erased'] is True
        assert 'timestamp' in erasure_data
    
    @pytest.mark.asyncio
    async def test_audit_logging(self):
        """Test security audit logging."""
        await encryption_service.initialize()
        
        # Perform some operations to generate audit logs
        test_data = "audit test data"
        encrypted_data = await encryption_service.encrypt_data(test_data)
        await encryption_service.decrypt_data(encrypted_data)
        await encryption_service.hash_data(test_data)
        
        # Get audit log
        audit_log = await encryption_service.get_audit_log(1)
        assert len(audit_log) > 0
        
        # Check audit log entries
        for entry in audit_log:
            assert 'timestamp' in entry
            assert 'event_type' in entry
            assert 'details' in entry
            assert 'service' in entry
            assert entry['service'] == 'encryption_service'


@pytest.mark.security
@pytest.mark.integration
class TestSecurityIntegration:
    """Test security integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_authentication_flow(self):
        """Test complete authentication flow."""
        await jwt_service.initialize()
        
        # Login
        login_result = await jwt_service.login("admin", "SpotOn2024!")
        assert login_result is not None
        assert 'access_token' in login_result
        assert 'refresh_token' in login_result
        
        # Validate access token
        token_data = await jwt_service.validate_token(login_result['access_token'])
        assert token_data is not None
        
        # Refresh token
        refresh_result = await jwt_service.refresh_token(login_result['refresh_token'])
        assert refresh_result is not None
        assert 'access_token' in refresh_result
        
        # Logout
        logout_result = await jwt_service.logout(login_result['access_token'])
        assert logout_result is True
    
    @pytest.mark.asyncio
    async def test_encrypted_user_data_flow(self):
        """Test encrypted user data handling."""
        await encryption_service.initialize()
        
        # User data with sensitive information
        user_data = {
            "user_id": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "phone": "1234567890",
            "preferences": {"theme": "dark"}
        }
        
        # Encrypt sensitive data
        encrypted_data = await encryption_service.encrypt_personal_data(user_data)
        
        # Simulate storing encrypted data (in real scenario, this would go to database)
        stored_data = encrypted_data
        
        # Retrieve and decrypt data
        decrypted_data = await encryption_service.decrypt_personal_data(stored_data)
        
        # Verify data integrity
        assert decrypted_data["user_id"] == user_data["user_id"]
        assert decrypted_data["email"] == user_data["email"]
        assert decrypted_data["phone"] == user_data["phone"]
    
    @pytest.mark.asyncio
    async def test_security_statistics_collection(self):
        """Test security statistics collection."""
        await jwt_service.initialize()
        await encryption_service.initialize()
        
        # Perform various operations
        await jwt_service.login("admin", "SpotOn2024!")
        await encryption_service.encrypt_data("test data")
        await encryption_service.hash_data("test hash")
        
        # Get JWT statistics
        jwt_stats = await jwt_service.get_jwt_statistics()
        assert jwt_stats['jwt_stats']['login_attempts'] > 0
        assert jwt_stats['jwt_stats']['tokens_generated'] > 0
        
        # Get encryption statistics
        encryption_stats = await encryption_service.get_encryption_statistics()
        assert encryption_stats['encryption_stats']['encryptions_performed'] > 0
        assert encryption_stats['encryption_stats']['hash_operations'] > 0
    
    @pytest.mark.asyncio
    async def test_security_error_handling(self):
        """Test security error handling."""
        await jwt_service.initialize()
        await encryption_service.initialize()
        
        # Test invalid token validation
        invalid_token = "invalid.token.here"
        token_data = await jwt_service.validate_token(invalid_token)
        assert token_data is None
        
        # Test encryption with invalid key
        encryption_service.active_key_id = "nonexistent_key"
        try:
            await encryption_service.encrypt_data("test data")
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "key" in str(e).lower()
        
        # Reset to valid key
        await encryption_service.rotate_encryption_key()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_security_performance(self):
        """Test security operations performance."""
        await jwt_service.initialize()
        await encryption_service.initialize()
        
        # Test token generation performance
        user = jwt_service.users["admin"]
        
        start_time = datetime.now(timezone.utc)
        for _ in range(100):
            await jwt_service.generate_token(user)
        token_generation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Token generation should be fast
        assert token_generation_time < 5.0  # 100 tokens in less than 5 seconds
        
        # Test encryption performance
        test_data = "A" * 1000  # 1KB of data
        
        start_time = datetime.now(timezone.utc)
        for _ in range(100):
            encrypted_data = await encryption_service.encrypt_data(test_data)
            await encryption_service.decrypt_data(encrypted_data)
        encryption_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Encryption should be reasonably fast
        assert encryption_time < 10.0  # 100 encrypt/decrypt cycles in less than 10 seconds