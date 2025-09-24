"""
Module for downloading assets from S3-compatible storage.
"""
import os
import asyncio
from pathlib import Path
import logging
from typing import Optional
from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig
from app.core.config import settings

import boto3 # For S3 communication
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

logger = logging.getLogger(__name__)

class AssetDownloader:
    """
    Utility class for downloading assets from S3-compatible storage (e.g., DagsHub, AWS S3).
    """
    def __init__(self,
                 s3_endpoint_url: Optional[str],
                 aws_access_key_id: Optional[str],
                 aws_secret_access_key: Optional[str],
                 s3_bucket_name: str):
        """
        Initializes the AssetDownloader with S3 client configuration.

        Args:
            s3_endpoint_url: The S3 endpoint URL.
            aws_access_key_id: AWS Access Key ID.
            aws_secret_access_key: AWS Secret Access Key.
            s3_bucket_name: The S3 bucket name to interact with.
        """
        self.s3_endpoint_url = s3_endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_bucket_name = s3_bucket_name
        self.s3_client = None

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            logger.warning(
                "AWS Access Key ID or Secret Access Key is not configured. "
                "S3 downloads might fail if credentials are required by the endpoint."
            )
        
        if not self.s3_bucket_name:
            logger.error("S3_BUCKET_NAME is not configured. Cannot initialize S3 client.")
            return # Do not initialize client if bucket name is missing

        try:
            boto_cfg = BotoConfig(
                retries={"max_attempts": settings.S3_MAX_ATTEMPTS, "mode": "standard"},
                connect_timeout=settings.S3_CONNECT_TIMEOUT,
                read_timeout=settings.S3_READ_TIMEOUT,
                max_pool_connections=20,
            )
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.s3_endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                config=boto_cfg
            )
            # Multipart transfer config
            self.transfer_config = TransferConfig(
                multipart_threshold=settings.S3_MULTIPART_THRESHOLD_MB * 1024 * 1024,
                max_concurrency=settings.S3_MAX_TRANSFER_CONCURRENCY,
                multipart_chunksize=8 * 1024 * 1024,
                use_threads=True,
            )
            # Optional: Add a connection test here if desired, e.g., list_objects_v2 with MaxKeys=0
            logger.info(
                f"Boto3 S3 client initialized. Endpoint: '{self.s3_endpoint_url}', "
                f"Bucket: '{self.s3_bucket_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Boto3 S3 client: {e}", exc_info=True)
            self.s3_client = None # Ensure client is None if initialization fails

    async def download_file_from_dagshub( # Method name kept for compatibility
        self,
        remote_s3_key: str,
        local_destination_path: str
    ) -> bool:
        """
        Downloads a file from the configured S3 bucket.
        The method name "download_file_from_dagshub" is kept for backward
        compatibility with services that previously used a DagsHub-specific library.

        Args:
            remote_s3_key: The key (path) of the file within the S3 bucket.
            local_destination_path: Full local path to save the downloaded file.

        Returns:
            True if download was successful, False otherwise.
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot download.")
            return False

        if not remote_s3_key:
            logger.error("Remote S3 key cannot be empty.")
            return False
        
        if not local_destination_path:
            logger.error("Local destination path cannot be empty.")
            return False

        try:
            Path(local_destination_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create parent directory for {local_destination_path}: {e}")
            return False
        
        logger.info(
            f"Attempting to download S3 object: Bucket='{self.s3_bucket_name}', "
            f"Key='{remote_s3_key}', To='{local_destination_path}'"
        )

        try:
            # ETag-based cache check
            etag = None
            try:
                head_obj = await asyncio.to_thread(
                    self.s3_client.head_object, Bucket=self.s3_bucket_name, Key=remote_s3_key
                )
                etag = head_obj.get('ETag', '').strip('"')
            except Exception as e:
                logger.debug(f"HEAD failed for {remote_s3_key}: {e}")

            if os.path.exists(local_destination_path) and etag:
                etag_path = f"{local_destination_path}.etag"
                try:
                    if os.path.exists(etag_path):
                        with open(etag_path, 'r') as ef:
                            cached_etag = ef.read().strip()
                        if cached_etag == etag:
                            logger.info(f"Cache hit for {remote_s3_key} (etag match). Skipping download.")
                            return True
                except Exception:
                    pass

            # boto3 client methods are blocking, so run in a separate thread
            await asyncio.to_thread(
                self.s3_client.download_file,
                Bucket=self.s3_bucket_name,
                Key=remote_s3_key,
                Filename=local_destination_path,
                Config=getattr(self, 'transfer_config', None)
            )
            logger.info(
                f"Successfully downloaded '{remote_s3_key}' from bucket "
                f"'{self.s3_bucket_name}' to '{local_destination_path}'"
            )
            # Write ETag cache
            if etag:
                try:
                    with open(f"{local_destination_path}.etag", 'w') as ef:
                        ef.write(etag)
                except Exception:
                    pass
            return True
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(
                f"S3 credentials error for Bucket='{self.s3_bucket_name}', Key='{remote_s3_key}': {e}. "
                "Ensure credentials are set and valid for the S3 endpoint."
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == '404' or error_code == 'NoSuchKey':
                logger.warning(
                    f"File not found: Bucket='{self.s3_bucket_name}', Key='{remote_s3_key}'."
                )
            elif error_code == '403' or 'Forbidden' in str(e) or 'AccessDenied' in str(e):
                logger.error(
                    f"Access Denied for Bucket='{self.s3_bucket_name}', Key='{remote_s3_key}'. "
                    "Check S3 credentials and permissions."
                )
            else:
                logger.error(
                    f"S3 ClientError for Bucket='{self.s3_bucket_name}', Key='{remote_s3_key}': {e}"
                )
        except Exception as e:
            logger.error(
                f"Unexpected error downloading from Bucket='{self.s3_bucket_name}', Key='{remote_s3_key}': {e}",
                exc_info=True
            )
        
        return False
