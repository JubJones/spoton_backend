import os
import asyncio
from pathlib import Path
import logging
from typing import Optional # Added for clarity with s3_endpoint_url
from dagshub import get_repo_bucket_client
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Note: settings import is removed as it's now expected to be passed during instantiation
# from app.core.config import settings

logger = logging.getLogger(__name__)

class AssetDownloader: # Renamed class for clarity as a utility
    """
    Utility class for downloading assets, specifically from DagsHub.
    """
    def __init__(self, dagshub_repo_owner: str, dagshub_repo_name: str, s3_endpoint_url: Optional[str] = None):
        self.dagshub_repo_owner = dagshub_repo_owner
        self.dagshub_repo_name = dagshub_repo_name
        self.s3_endpoint_url = s3_endpoint_url
        self.dagshub_full_repo_name = f"{dagshub_repo_owner}/{dagshub_repo_name}"

    async def download_file_from_dagshub(
        self,
        remote_s3_key: str,
        local_destination_path: str,
        dagshub_bucket_name_override: Optional[str] = None
    ) -> bool:
        """
        Downloads a file from DagsHub using its S3-like interface.

        Args:
            remote_s3_key: The key (path) of the file within the DagsHub repository's DVC storage.
            local_destination_path: Full local path to save the downloaded file.
            dagshub_bucket_name_override: Optional. If provided, uses this as the bucket name.
                                         Defaults to self.dagshub_repo_name.

        Returns:
            True if download was successful, False otherwise.
        """
        Path(local_destination_path).parent.mkdir(parents=True, exist_ok=True)
        
        bucket_to_use = dagshub_bucket_name_override if dagshub_bucket_name_override else self.dagshub_repo_name

        logger.info(f"Attempting to download from DagsHub: repo='{self.dagshub_full_repo_name}', key='{remote_s3_key}', bucket='{bucket_to_use}' to '{local_destination_path}'")

        try:
            boto_client = await asyncio.to_thread(
                get_repo_bucket_client,
                self.dagshub_full_repo_name,
                flavor="boto"
            )
            
            await asyncio.to_thread(
                boto_client.download_file,
                Bucket=bucket_to_use,
                Key=remote_s3_key,
                Filename=local_destination_path
            )
            logger.info(f"Successfully downloaded '{remote_s3_key}' from DagsHub repo '{self.dagshub_full_repo_name}' to '{local_destination_path}'")
            return True
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"DagsHub/S3 credentials error for repo {self.dagshub_full_repo_name}, key {remote_s3_key}: {e}. Ensure 'dagshub login' has been run.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == '404' or error_code == 'NoSuchKey':
                logger.warning(f"File not found on DagsHub: repo='{self.dagshub_full_repo_name}', bucket='{bucket_to_use}', key='{remote_s3_key}'.")
            else:
                logger.error(f"DagsHub/S3 ClientError for repo {self.dagshub_full_repo_name}, key {remote_s3_key}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading from DagsHub repo {self.dagshub_full_repo_name}, key {remote_s3_key}: {e}")
        
        return False