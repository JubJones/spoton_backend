"""
Clean S3 Service implementation.

Modernized S3 service integration following clean architecture principles
with proper error handling, configuration management, and interface abstraction.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
import logging
import asyncio
from datetime import datetime, timezone
from enum import Enum

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from botocore.config import Config
# aioboto3 would be preferred for true async, but using asyncio wrapper for now
import concurrent.futures

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3OperationType(Enum):
    """Types of S3 operations."""
    DOWNLOAD = "download"
    UPLOAD = "upload"
    DELETE = "delete"
    LIST = "list"
    CHECK_EXISTS = "check_exists"


@dataclass(frozen=True)
class S3Configuration:
    """S3 service configuration."""
    endpoint_url: Optional[str]
    access_key_id: Optional[str]
    secret_access_key: Optional[str]
    bucket_name: str
    region_name: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 300
    max_concurrent_requests: int = 10


@dataclass
class S3OperationResult:
    """Result of S3 operation."""
    success: bool
    operation_type: S3OperationType
    s3_key: str
    local_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None


class S3ServiceError(Exception):
    """S3 service specific exceptions."""
    pass


class IS3Service(ABC):
    """Abstract interface for S3 service operations."""
    
    @abstractmethod
    async def download_file(
        self, 
        s3_key: str, 
        local_path: str,
        overwrite: bool = True
    ) -> S3OperationResult:
        """Download file from S3."""
        pass
    
    @abstractmethod
    async def upload_file(
        self, 
        local_path: str, 
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> S3OperationResult:
        """Upload file to S3."""
        pass
    
    @abstractmethod
    async def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3."""
        pass
    
    @abstractmethod
    async def list_objects(
        self, 
        prefix: str = "", 
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """List objects in S3 bucket with prefix."""
        pass
    
    @abstractmethod
    async def delete_file(self, s3_key: str) -> S3OperationResult:
        """Delete file from S3."""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from S3."""
        pass


class ModernS3Service(IS3Service):
    """
    Modern S3 service implementation with clean architecture patterns.
    
    Features:
    - Async/await support with aioboto3
    - Proper error handling and retry logic
    - Configuration validation
    - Operation metrics and logging
    - Resource management and cleanup
    """
    
    def __init__(self, config: S3Configuration):
        """
        Initialize modern S3 service.
        
        Args:
            config: S3 service configuration
        """
        self.config = config
        self._validate_configuration()
        self._client = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        self._operation_count = 0
        self._last_operation_time: Optional[datetime] = None
        
        logger.info(f"ModernS3Service initialized for bucket: {config.bucket_name}")
    
    def _validate_configuration(self) -> None:
        """Validate S3 configuration."""
        if not self.config.bucket_name:
            raise S3ServiceError("S3 bucket name is required")
        
        if not self.config.access_key_id or not self.config.secret_access_key:
            logger.warning("S3 credentials not provided - will use default AWS credential chain")
        
        if self.config.max_retries < 0:
            raise S3ServiceError("Max retries must be non-negative")
        
        if self.config.timeout_seconds <= 0:
            raise S3ServiceError("Timeout must be positive")
        
        logger.debug("S3 configuration validated successfully")
    
    def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            # Configure client with proper settings
            boto_config = Config(
                retries={'max_attempts': self.config.max_retries},
                max_pool_connections=self.config.max_concurrent_requests,
                connect_timeout=self.config.timeout_seconds,
                read_timeout=self.config.timeout_seconds
            )
            
            self._client = boto3.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region_name,
                config=boto_config
            )
            
            logger.debug("S3 client created")
        
        return self._client
    
    async def _execute_with_error_handling(
        self, 
        operation_type: S3OperationType,
        s3_key: str,
        operation_func,
        local_path: Optional[str] = None
    ) -> S3OperationResult:
        """Execute S3 operation with proper error handling."""
        start_time = datetime.now(timezone.utc)
        
        try:
            result = await operation_func()
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            self._operation_count += 1
            self._last_operation_time = datetime.now(timezone.utc)
            
            logger.debug(f"S3 {operation_type.value} completed for {s3_key} in {duration:.2f}s")
            
            return S3OperationResult(
                success=True,
                operation_type=operation_type,
                s3_key=s3_key,
                local_path=local_path,
                duration_seconds=duration,
                metadata=result if isinstance(result, dict) else None
            )
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = f"S3 ClientError [{error_code}]: {str(e)}"
            logger.error(f"S3 {operation_type.value} failed for {s3_key}: {error_message}")
            
            return S3OperationResult(
                success=False,
                operation_type=operation_type,
                s3_key=s3_key,
                local_path=local_path,
                error_message=error_message
            )
            
        except (NoCredentialsError, PartialCredentialsError) as e:
            error_message = f"S3 Credentials Error: {str(e)}"
            logger.error(f"S3 {operation_type.value} failed for {s3_key}: {error_message}")
            
            return S3OperationResult(
                success=False,
                operation_type=operation_type,
                s3_key=s3_key,
                local_path=local_path,
                error_message=error_message
            )
            
        except Exception as e:
            error_message = f"Unexpected S3 Error: {str(e)}"
            logger.error(f"S3 {operation_type.value} failed for {s3_key}: {error_message}", exc_info=True)
            
            return S3OperationResult(
                success=False,
                operation_type=operation_type,
                s3_key=s3_key,
                local_path=local_path,
                error_message=error_message
            )
    
    async def download_file(
        self, 
        s3_key: str, 
        local_path: str,
        overwrite: bool = True
    ) -> S3OperationResult:
        """
        Download file from S3 with clean architecture patterns.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path destination  
            overwrite: Whether to overwrite existing local file
            
        Returns:
            Operation result with success status and metadata
        """
        if not s3_key or not local_path:
            return S3OperationResult(
                success=False,
                operation_type=S3OperationType.DOWNLOAD,
                s3_key=s3_key or "",
                local_path=local_path,
                error_message="S3 key and local path are required"
            )
        
        local_file_path = Path(local_path)
        
        # Check if local file exists and handle overwrite
        if local_file_path.exists() and not overwrite:
            logger.info(f"Local file exists and overwrite=False: {local_path}")
            return S3OperationResult(
                success=True,
                operation_type=S3OperationType.DOWNLOAD,
                s3_key=s3_key,
                local_path=local_path,
                metadata={"skipped": True, "reason": "file_exists_no_overwrite"}
            )
        
        # Create parent directories
        try:
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return S3OperationResult(
                success=False,
                operation_type=S3OperationType.DOWNLOAD,
                s3_key=s3_key,
                local_path=local_path,
                error_message=f"Failed to create parent directory: {e}"
            )
        
        async def download_operation():
            def sync_download():
                client = self._get_client()
                client.download_file(self.config.bucket_name, s3_key, local_path)
                return {"downloaded": True}
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, sync_download)
        
        return await self._execute_with_error_handling(
            S3OperationType.DOWNLOAD,
            s3_key,
            download_operation,
            local_path
        )
    
    async def upload_file(
        self, 
        local_path: str, 
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> S3OperationResult:
        """
        Upload file to S3 with metadata support.
        
        Args:
            local_path: Local file path to upload
            s3_key: S3 object key destination
            metadata: Optional metadata to attach to S3 object
            
        Returns:
            Operation result with success status and metadata
        """
        if not local_path or not s3_key:
            return S3OperationResult(
                success=False,
                operation_type=S3OperationType.UPLOAD,
                s3_key=s3_key or "",
                local_path=local_path,
                error_message="Local path and S3 key are required"
            )
        
        local_file_path = Path(local_path)
        if not local_file_path.exists():
            return S3OperationResult(
                success=False,
                operation_type=S3OperationType.UPLOAD,
                s3_key=s3_key,
                local_path=local_path,
                error_message=f"Local file does not exist: {local_path}"
            )
        
        async def upload_operation():
            def sync_upload():
                client = self._get_client()
                extra_args = {}
                if metadata:
                    extra_args['Metadata'] = metadata
                
                client.upload_file(local_path, self.config.bucket_name, s3_key, ExtraArgs=extra_args)
                return {"uploaded": True, "file_size": local_file_path.stat().st_size}
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, sync_upload)
        
        return await self._execute_with_error_handling(
            S3OperationType.UPLOAD,
            s3_key,
            upload_operation,
            local_path
        )
    
    async def file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3.
        
        Args:
            s3_key: S3 object key to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            def sync_head_object():
                client = self._get_client()
                client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
                return True
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, sync_head_object)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return False
            # Re-raise for other errors
            logger.error(f"Error checking if S3 file exists {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking S3 file existence {s3_key}: {e}")
            return False
    
    async def list_objects(
        self, 
        prefix: str = "", 
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with prefix.
        
        Args:
            prefix: Object key prefix filter
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object metadata dictionaries
        """
        try:
            def sync_list_objects():
                client = self._get_client()
                return client.list_objects_v2(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                    MaxKeys=max_keys
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(self._executor, sync_list_objects)
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                        'Key': obj['Key'],
                        'Size': obj['Size'],
                        'LastModified': obj['LastModified'],
                        'ETag': obj.get('ETag', ''),
                        'StorageClass': obj.get('StorageClass', 'STANDARD')
                    })
                
                logger.debug(f"Listed {len(objects)} objects with prefix '{prefix}'")
                return objects
                
        except Exception as e:
            logger.error(f"Error listing S3 objects with prefix '{prefix}': {e}")
            return []
    
    async def delete_file(self, s3_key: str) -> S3OperationResult:
        """
        Delete file from S3.
        
        Args:
            s3_key: S3 object key to delete
            
        Returns:
            Operation result with success status
        """
        if not s3_key:
            return S3OperationResult(
                success=False,
                operation_type=S3OperationType.DELETE,
                s3_key="",
                error_message="S3 key is required for deletion"
            )
        
        async def delete_operation():
            def sync_delete():
                client = self._get_client()
                client.delete_object(Bucket=self.config.bucket_name, Key=s3_key)
                return {"deleted": True}
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, sync_delete)
        
        return await self._execute_with_error_handling(
            S3OperationType.DELETE,
            s3_key,
            delete_operation
        )
    
    async def get_file_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get file metadata from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            def sync_get_metadata():
                client = self._get_client()
                return client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(self._executor, sync_get_metadata)
            
            return {
                    'ContentLength': response.get('ContentLength', 0),
                    'ContentType': response.get('ContentType', ''),
                    'LastModified': response.get('LastModified'),
                    'ETag': response.get('ETag', ''),
                    'Metadata': response.get('Metadata', {}),
                    'StorageClass': response.get('StorageClass', 'STANDARD')
                }
                
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return None
            logger.error(f"Error getting S3 file metadata {s3_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting S3 file metadata {s3_key}: {e}")
            return None
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get service usage statistics."""
        return {
            'operation_count': self._operation_count,
            'last_operation_time': self._last_operation_time.isoformat() if self._last_operation_time else None,
            'bucket_name': self.config.bucket_name,
            'endpoint_url': self.config.endpoint_url,
            'max_retries': self.config.max_retries,
            'timeout_seconds': self.config.timeout_seconds
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._client = None
        logger.debug("S3 service resources cleaned up")


def create_s3_service(
    endpoint_url: Optional[str] = None,
    access_key_id: Optional[str] = None, 
    secret_access_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
    region_name: Optional[str] = None
) -> ModernS3Service:
    """
    Factory function to create S3 service with configuration.
    
    Args:
        endpoint_url: S3 endpoint URL (defaults to settings)
        access_key_id: AWS access key (defaults to settings)
        secret_access_key: AWS secret key (defaults to settings)
        bucket_name: S3 bucket name (defaults to settings)
        region_name: AWS region name
        
    Returns:
        Configured S3 service instance
    """
    config = S3Configuration(
        endpoint_url=endpoint_url or settings.S3_ENDPOINT_URL,
        access_key_id=access_key_id or settings.AWS_ACCESS_KEY_ID,
        secret_access_key=secret_access_key or settings.AWS_SECRET_ACCESS_KEY,
        bucket_name=bucket_name or settings.S3_BUCKET_NAME,
        region_name=region_name,
        max_retries=3,
        timeout_seconds=300,
        max_concurrent_requests=10
    )
    
    return ModernS3Service(config)