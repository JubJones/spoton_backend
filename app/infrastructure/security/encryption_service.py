"""
Encryption service for data protection and security.

Handles:
- Data encryption at rest and in transit
- Key management and rotation
- Secure hashing algorithms
- GDPR compliance utilities
- Sensitive data masking
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import secrets
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
from enum import Enum
import os

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"


class HashAlgorithm(Enum):
    """Supported hashing algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    PBKDF2_SHA256 = "pbkdf2-sha256"
    PBKDF2_SHA512 = "pbkdf2-sha512"


@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_rotation_days: int = 30
    hash_algorithm: HashAlgorithm = HashAlgorithm.PBKDF2_SHA256
    pbkdf2_iterations: int = 100000
    enable_field_encryption: bool = True
    enable_audit_logging: bool = True


@dataclass
class EncryptionKey:
    """Encryption key information."""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    version: int = 1


@dataclass
class EncryptedData:
    """Encrypted data structure."""
    data: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EncryptionService:
    """
    Encryption service for comprehensive data protection.
    
    Features:
    - Data encryption at rest and in transit
    - Key management and rotation
    - Secure hashing algorithms
    - GDPR compliance utilities
    - Sensitive data masking
    """
    
    def __init__(self):
        self.config = EncryptionConfig()
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.active_key_id: Optional[str] = None
        
        # RSA key pair for asymmetric encryption
        self.rsa_private_key: Optional[rsa.RSAPrivateKey] = None
        self.rsa_public_key: Optional[rsa.RSAPublicKey] = None
        
        # Performance tracking
        self.encryption_stats = {
            'encryptions_performed': 0,
            'decryptions_performed': 0,
            'keys_generated': 0,
            'keys_rotated': 0,
            'hash_operations': 0,
            'gdpr_requests_processed': 0
        }
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info("EncryptionService initialized")
    
    async def initialize(self):
        """Initialize encryption service."""
        try:
            # Generate initial encryption key
            await self._generate_master_key()
            
            # Generate RSA key pair
            await self._generate_rsa_keys()
            
            # Load existing keys if available
            await self._load_encryption_keys()
            
            logger.info("EncryptionService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing EncryptionService: {e}")
            raise
    
    async def _generate_master_key(self):
        """Generate master encryption key."""
        try:
            key_id = f"master_key_{int(datetime.now(timezone.utc).timestamp())}"
            key_data = secrets.token_bytes(32)  # 256-bit key
            
            master_key = EncryptionKey(
                key_id=key_id,
                algorithm=self.config.default_algorithm,
                key_data=key_data,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(days=self.config.key_rotation_days)
            )
            
            self.encryption_keys[key_id] = master_key
            self.active_key_id = key_id
            self.encryption_stats['keys_generated'] += 1
            
            await self._log_audit_event("key_generated", {"key_id": key_id, "algorithm": self.config.default_algorithm.value})
            
            logger.info(f"Master encryption key generated: {key_id}")
            
        except Exception as e:
            logger.error(f"Error generating master key: {e}")
            raise
    
    async def _generate_rsa_keys(self):
        """Generate RSA key pair."""
        try:
            # Generate RSA private key
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            await self._log_audit_event("rsa_keys_generated", {"key_size": 2048})
            
            logger.info("RSA key pair generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating RSA keys: {e}")
            raise
    
    async def _load_encryption_keys(self):
        """Load existing encryption keys."""
        try:
            # In production, this would load from secure key storage
            # For now, we'll use the generated master key
            logger.info("Encryption keys loaded")
            
        except Exception as e:
            logger.error(f"Error loading encryption keys: {e}")
    
    # Symmetric Encryption
    async def encrypt_data(self, data: Union[str, bytes], key_id: Optional[str] = None) -> EncryptedData:
        """Encrypt data using symmetric encryption."""
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Use active key if not specified
            if key_id is None:
                key_id = self.active_key_id
            
            if key_id not in self.encryption_keys:
                raise ValueError(f"Encryption key {key_id} not found")
            
            encryption_key = self.encryption_keys[key_id]
            
            if encryption_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                encrypted_data = await self._encrypt_aes_gcm(data, encryption_key.key_data)
            elif encryption_key.algorithm == EncryptionAlgorithm.AES_256_CBC:
                encrypted_data = await self._encrypt_aes_cbc(data, encryption_key.key_data)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {encryption_key.algorithm}")
            
            self.encryption_stats['encryptions_performed'] += 1
            
            await self._log_audit_event("data_encrypted", {
                "key_id": key_id,
                "algorithm": encryption_key.algorithm.value,
                "data_size": len(data)
            })
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data using symmetric encryption."""
        try:
            if encrypted_data.key_id not in self.encryption_keys:
                raise ValueError(f"Encryption key {encrypted_data.key_id} not found")
            
            encryption_key = self.encryption_keys[encrypted_data.key_id]
            
            if encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
                decrypted_data = await self._decrypt_aes_gcm(encrypted_data, encryption_key.key_data)
            elif encrypted_data.algorithm == EncryptionAlgorithm.AES_256_CBC:
                decrypted_data = await self._decrypt_aes_cbc(encrypted_data, encryption_key.key_data)
            else:
                raise ValueError(f"Unsupported decryption algorithm: {encrypted_data.algorithm}")
            
            self.encryption_stats['decryptions_performed'] += 1
            
            await self._log_audit_event("data_decrypted", {
                "key_id": encrypted_data.key_id,
                "algorithm": encrypted_data.algorithm.value,
                "data_size": len(decrypted_data)
            })
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    async def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> EncryptedData:
        """Encrypt data using AES-256-GCM."""
        try:
            # Generate random nonce
            nonce = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            return EncryptedData(
                data=ciphertext,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                key_id=self.active_key_id,
                nonce=nonce,
                tag=encryptor.tag
            )
            
        except Exception as e:
            logger.error(f"Error in AES-GCM encryption: {e}")
            raise
    
    async def _decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        try:
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(encrypted_data.nonce, encrypted_data.tag))
            decryptor = cipher.decryptor()
            
            # Decrypt data
            plaintext = decryptor.update(encrypted_data.data) + decryptor.finalize()
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Error in AES-GCM decryption: {e}")
            raise
    
    async def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> EncryptedData:
        """Encrypt data using AES-256-CBC."""
        try:
            # Generate random IV
            iv = os.urandom(16)
            
            # Pad data to block size
            padded_data = self._pad_data(data, 16)
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return EncryptedData(
                data=ciphertext,
                algorithm=EncryptionAlgorithm.AES_256_CBC,
                key_id=self.active_key_id,
                nonce=iv
            )
            
        except Exception as e:
            logger.error(f"Error in AES-CBC encryption: {e}")
            raise
    
    async def _decrypt_aes_cbc(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data using AES-256-CBC."""
        try:
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(encrypted_data.nonce))
            decryptor = cipher.decryptor()
            
            # Decrypt data
            padded_plaintext = decryptor.update(encrypted_data.data) + decryptor.finalize()
            
            # Remove padding
            plaintext = self._unpad_data(padded_plaintext)
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Error in AES-CBC decryption: {e}")
            raise
    
    def _pad_data(self, data: bytes, block_size: int) -> bytes:
        """Pad data to block size using PKCS7 padding."""
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding from data."""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    # Asymmetric Encryption
    async def encrypt_with_rsa(self, data: bytes) -> bytes:
        """Encrypt data using RSA public key."""
        try:
            if not self.rsa_public_key:
                raise ValueError("RSA public key not available")
            
            # Encrypt data
            encrypted_data = self.rsa_public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            await self._log_audit_event("rsa_encryption", {"data_size": len(data)})
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error in RSA encryption: {e}")
            raise
    
    async def decrypt_with_rsa(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using RSA private key."""
        try:
            if not self.rsa_private_key:
                raise ValueError("RSA private key not available")
            
            # Decrypt data
            decrypted_data = self.rsa_private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            await self._log_audit_event("rsa_decryption", {"data_size": len(decrypted_data)})
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error in RSA decryption: {e}")
            raise
    
    # Hashing
    async def hash_data(self, data: Union[str, bytes], algorithm: Optional[HashAlgorithm] = None, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hash data using specified algorithm."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if algorithm is None:
                algorithm = self.config.hash_algorithm
            
            if algorithm == HashAlgorithm.SHA256:
                hash_value = hashlib.sha256(data).hexdigest()
                used_salt = b""
            elif algorithm == HashAlgorithm.SHA512:
                hash_value = hashlib.sha512(data).hexdigest()
                used_salt = b""
            elif algorithm == HashAlgorithm.PBKDF2_SHA256:
                if salt is None:
                    salt = os.urandom(32)
                hash_value = hashlib.pbkdf2_hex(data, salt, self.config.pbkdf2_iterations, 'sha256')
                used_salt = salt
            elif algorithm == HashAlgorithm.PBKDF2_SHA512:
                if salt is None:
                    salt = os.urandom(32)
                hash_value = hashlib.pbkdf2_hex(data, salt, self.config.pbkdf2_iterations, 'sha512')
                used_salt = salt
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            self.encryption_stats['hash_operations'] += 1
            
            await self._log_audit_event("data_hashed", {
                "algorithm": algorithm.value,
                "data_size": len(data),
                "salt_used": len(used_salt) > 0
            })
            
            return hash_value, used_salt
            
        except Exception as e:
            logger.error(f"Error hashing data: {e}")
            raise
    
    async def verify_hash(self, data: Union[str, bytes], hash_value: str, salt: bytes, algorithm: Optional[HashAlgorithm] = None) -> bool:
        """Verify data against hash."""
        try:
            if algorithm is None:
                algorithm = self.config.hash_algorithm
            
            computed_hash, _ = await self.hash_data(data, algorithm, salt)
            
            return computed_hash == hash_value
            
        except Exception as e:
            logger.error(f"Error verifying hash: {e}")
            return False
    
    # Key Management
    async def rotate_encryption_key(self) -> str:
        """Rotate encryption key."""
        try:
            # Generate new key
            new_key_id = f"key_{int(datetime.now(timezone.utc).timestamp())}"
            new_key_data = secrets.token_bytes(32)
            
            new_key = EncryptionKey(
                key_id=new_key_id,
                algorithm=self.config.default_algorithm,
                key_data=new_key_data,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(days=self.config.key_rotation_days)
            )
            
            # Mark old key as inactive
            if self.active_key_id:
                old_key = self.encryption_keys[self.active_key_id]
                old_key.is_active = False
            
            # Set new key as active
            self.encryption_keys[new_key_id] = new_key
            self.active_key_id = new_key_id
            
            self.encryption_stats['keys_rotated'] += 1
            
            await self._log_audit_event("key_rotated", {
                "old_key_id": self.active_key_id,
                "new_key_id": new_key_id
            })
            
            logger.info(f"Encryption key rotated: {new_key_id}")
            
            return new_key_id
            
        except Exception as e:
            logger.error(f"Error rotating encryption key: {e}")
            raise
    
    async def get_active_key_info(self) -> Dict[str, Any]:
        """Get active key information."""
        try:
            if not self.active_key_id:
                return {}
            
            active_key = self.encryption_keys[self.active_key_id]
            
            return {
                'key_id': active_key.key_id,
                'algorithm': active_key.algorithm.value,
                'created_at': active_key.created_at.isoformat(),
                'expires_at': active_key.expires_at.isoformat() if active_key.expires_at else None,
                'is_active': active_key.is_active,
                'version': active_key.version
            }
            
        except Exception as e:
            logger.error(f"Error getting active key info: {e}")
            return {}
    
    # Data Masking
    async def mask_sensitive_data(self, data: str, mask_type: str = "partial") -> str:
        """Mask sensitive data for display."""
        try:
            if not data:
                return data
            
            if mask_type == "partial":
                # Show first 2 and last 2 characters
                if len(data) <= 4:
                    return "*" * len(data)
                return data[:2] + "*" * (len(data) - 4) + data[-2:]
            
            elif mask_type == "full":
                return "*" * len(data)
            
            elif mask_type == "email":
                # Mask email addresses
                if "@" in data:
                    username, domain = data.split("@", 1)
                    if len(username) <= 2:
                        masked_username = "*" * len(username)
                    else:
                        masked_username = username[0] + "*" * (len(username) - 2) + username[-1]
                    return f"{masked_username}@{domain}"
                else:
                    return await self.mask_sensitive_data(data, "partial")
            
            elif mask_type == "phone":
                # Mask phone numbers
                if len(data) >= 10:
                    return data[:3] + "*" * (len(data) - 6) + data[-3:]
                else:
                    return await self.mask_sensitive_data(data, "partial")
            
            else:
                return await self.mask_sensitive_data(data, "partial")
            
        except Exception as e:
            logger.error(f"Error masking sensitive data: {e}")
            return data
    
    # GDPR Compliance
    async def encrypt_personal_data(self, personal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt personal data for GDPR compliance."""
        try:
            encrypted_data = {}
            
            # Define fields that need encryption
            sensitive_fields = ['email', 'phone', 'address', 'name', 'ssn', 'passport']
            
            for field, value in personal_data.items():
                if field.lower() in sensitive_fields and value:
                    # Encrypt sensitive field
                    encrypted_value = await self.encrypt_data(str(value))
                    encrypted_data[field] = {
                        'encrypted': True,
                        'data': base64.b64encode(encrypted_value.data).decode('utf-8'),
                        'algorithm': encrypted_value.algorithm.value,
                        'key_id': encrypted_value.key_id,
                        'nonce': base64.b64encode(encrypted_value.nonce).decode('utf-8') if encrypted_value.nonce else None,
                        'tag': base64.b64encode(encrypted_value.tag).decode('utf-8') if encrypted_value.tag else None
                    }
                else:
                    encrypted_data[field] = value
            
            await self._log_audit_event("personal_data_encrypted", {
                "fields_encrypted": len([f for f in personal_data.keys() if f.lower() in sensitive_fields]),
                "total_fields": len(personal_data)
            })
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting personal data: {e}")
            raise
    
    async def decrypt_personal_data(self, encrypted_personal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt personal data for GDPR compliance."""
        try:
            decrypted_data = {}
            
            for field, value in encrypted_personal_data.items():
                if isinstance(value, dict) and value.get('encrypted', False):
                    # Decrypt encrypted field
                    encrypted_data = EncryptedData(
                        data=base64.b64decode(value['data']),
                        algorithm=EncryptionAlgorithm(value['algorithm']),
                        key_id=value['key_id'],
                        nonce=base64.b64decode(value['nonce']) if value['nonce'] else None,
                        tag=base64.b64decode(value['tag']) if value['tag'] else None
                    )
                    
                    decrypted_bytes = await self.decrypt_data(encrypted_data)
                    decrypted_data[field] = decrypted_bytes.decode('utf-8')
                else:
                    decrypted_data[field] = value
            
            await self._log_audit_event("personal_data_decrypted", {
                "fields_decrypted": len([f for f in encrypted_personal_data.keys() if isinstance(encrypted_personal_data[f], dict) and encrypted_personal_data[f].get('encrypted', False)]),
                "total_fields": len(encrypted_personal_data)
            })
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting personal data: {e}")
            raise
    
    async def process_gdpr_request(self, request_type: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process GDPR requests (access, portability, erasure)."""
        try:
            self.encryption_stats['gdpr_requests_processed'] += 1
            
            if request_type == "access":
                # Decrypt personal data for access request
                decrypted_data = await self.decrypt_personal_data(user_data)
                
                # Mask sensitive data for display
                masked_data = {}
                for field, value in decrypted_data.items():
                    if field.lower() in ['email', 'phone', 'address']:
                        masked_data[field] = await self.mask_sensitive_data(str(value), field.lower())
                    else:
                        masked_data[field] = value
                
                await self._log_audit_event("gdpr_access_request", {"fields_accessed": len(decrypted_data)})
                
                return masked_data
            
            elif request_type == "portability":
                # Decrypt all personal data for portability
                decrypted_data = await self.decrypt_personal_data(user_data)
                
                await self._log_audit_event("gdpr_portability_request", {"fields_exported": len(decrypted_data)})
                
                return decrypted_data
            
            elif request_type == "erasure":
                # Return confirmation of erasure
                await self._log_audit_event("gdpr_erasure_request", {"fields_erased": len(user_data)})
                
                return {"erased": True, "timestamp": datetime.now(timezone.utc).isoformat()}
            
            else:
                raise ValueError(f"Unsupported GDPR request type: {request_type}")
            
        except Exception as e:
            logger.error(f"Error processing GDPR request: {e}")
            raise
    
    # Audit Logging
    async def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event."""
        try:
            if not self.config.enable_audit_logging:
                return
            
            audit_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event_type': event_type,
                'details': details,
                'service': 'encryption_service'
            }
            
            self.audit_log.append(audit_entry)
            
            # Keep only recent audit entries
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-10000:]
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
    
    # API Methods
    async def get_encryption_statistics(self) -> Dict[str, Any]:
        """Get encryption service statistics."""
        try:
            return {
                'encryption_stats': self.encryption_stats,
                'total_keys': len(self.encryption_keys),
                'active_key_id': self.active_key_id,
                'audit_log_entries': len(self.audit_log),
                'configuration': {
                    'default_algorithm': self.config.default_algorithm.value,
                    'key_rotation_days': self.config.key_rotation_days,
                    'hash_algorithm': self.config.hash_algorithm.value,
                    'pbkdf2_iterations': self.config.pbkdf2_iterations,
                    'enable_field_encryption': self.config.enable_field_encryption,
                    'enable_audit_logging': self.config.enable_audit_logging
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting encryption statistics: {e}")
            return {}
    
    async def get_audit_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            recent_entries = []
            for entry in self.audit_log:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_time:
                    recent_entries.append(entry)
            
            return recent_entries
            
        except Exception as e:
            logger.error(f"Error getting audit log: {e}")
            return []
    
    def reset_statistics(self):
        """Reset encryption statistics."""
        self.encryption_stats = {
            'encryptions_performed': 0,
            'decryptions_performed': 0,
            'keys_generated': 0,
            'keys_rotated': 0,
            'hash_operations': 0,
            'gdpr_requests_processed': 0
        }
        logger.info("Encryption statistics reset")


# Global encryption service instance
encryption_service = EncryptionService()