#!/usr/bin/env python3
"""
Script to download S3 video files using the same method as the SpotOn backend.
Downloads s3://spoton_ml/video_s14/c09/sub_video_01.mp4 to local directory.
"""
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3VideoDownloader:
    """Downloads video files from S3 using the same method as the backend."""
    
    def __init__(self, 
                 s3_endpoint_url: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 s3_bucket_name: str = "spoton_ml"):
        """
        Initialize the S3 video downloader.
        
        Args:
            s3_endpoint_url: S3 endpoint URL (defaults to DagsHub)
            aws_access_key_id: AWS Access Key ID
            aws_secret_access_key: AWS Secret Access Key  
            s3_bucket_name: S3 bucket name
        """
        self.s3_endpoint_url = s3_endpoint_url or "https://s3.dagshub.com"
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_bucket_name = s3_bucket_name
        self.s3_client = None
        
        self._initialize_s3_client()
    
    def _initialize_s3_client(self):
        """Initialize the boto3 S3 client."""
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            logger.warning(
                "AWS credentials not provided. Downloads might fail if credentials are required."
            )
        
        if not self.s3_bucket_name:
            logger.error("S3 bucket name is required.")
            return
        
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.s3_endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            logger.info(
                f"S3 client initialized. Endpoint: '{self.s3_endpoint_url}', "
                f"Bucket: '{self.s3_bucket_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3 to local path.
        
        Args:
            s3_key: S3 object key (e.g., "video_s14/c09/sub_video_01.mp4")
            local_path: Local file path to save the downloaded file
            
        Returns:
            True if download successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot download.")
            return False
        
        if not s3_key or not local_path:
            logger.error("S3 key and local path are required.")
            return False
        
        # Create parent directories if they don't exist
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create parent directory for {local_path}: {e}")
            return False
        
        logger.info(
            f"Downloading S3 object: Bucket='{self.s3_bucket_name}', "
            f"Key='{s3_key}', To='{local_path}'"
        )
        
        try:
            # Use asyncio.to_thread to run blocking boto3 call asynchronously
            await asyncio.to_thread(
                self.s3_client.download_file,
                Bucket=self.s3_bucket_name,
                Key=s3_key,
                Filename=local_path
            )
            
            # Verify file was downloaded
            if Path(local_path).exists():
                file_size = Path(local_path).stat().st_size
                logger.info(
                    f"Successfully downloaded '{s3_key}' from bucket "
                    f"'{self.s3_bucket_name}' to '{local_path}' "
                    f"(Size: {file_size:,} bytes)"
                )
                return True
            else:
                logger.error(f"Download completed but file not found at {local_path}")
                return False
                
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(
                f"S3 credentials error for Bucket='{self.s3_bucket_name}', Key='{s3_key}': {e}. "
                "Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set."
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == '404' or error_code == 'NoSuchKey':
                logger.error(
                    f"File not found: Bucket='{self.s3_bucket_name}', Key='{s3_key}'."
                )
            elif error_code == '403' or 'Forbidden' in str(e) or 'AccessDenied' in str(e):
                logger.error(
                    f"Access denied for Bucket='{self.s3_bucket_name}', Key='{s3_key}'. "
                    "Check S3 credentials and permissions."
                )
            else:
                logger.error(
                    f"S3 ClientError for Bucket='{self.s3_bucket_name}', Key='{s3_key}': {e}"
                )
        except Exception as e:
            logger.error(
                f"Unexpected error downloading from Bucket='{self.s3_bucket_name}', "
                f"Key='{s3_key}': {e}",
                exc_info=True
            )
        
        return False


def load_env_vars() -> dict:
    """Load environment variables from .env file if it exists."""
    env_vars = {}
    env_file = Path('.env')
    
    if env_file.exists():
        logger.info(f"Loading environment variables from {env_file}")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove spaces and quotes from key and value
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        env_vars[key] = value

        except Exception as e:
            logger.warning(f"Error reading .env file: {e}")
    else:
        logger.info("No .env file found, using environment variables only")
    
    return env_vars


async def main():
    """Main function to download the specified video file."""
    
    # Load environment variables from .env file if it exists
    env_vars = load_env_vars()
    
    # Get configuration from environment variables or .env file
    s3_endpoint_url = env_vars.get('S3_ENDPOINT_URL') or os.getenv('S3_ENDPOINT_URL', 'https://s3.dagshub.com')
    aws_access_key_id = env_vars.get('AWS_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = env_vars.get('AWS_SECRET_ACCESS_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')
    s3_bucket_name = env_vars.get('S3_BUCKET_NAME') or os.getenv('S3_BUCKET_NAME', 'spoton_ml')
    
    # Target file to download
    s3_key = "video_s14/c09/sub_video_01.mp4"
    local_path = "./downloaded_videos/video_s14/c09/sub_video_01.mp4"
    
    logger.info("=" * 60)
    logger.info("SpotOn S3 Video Downloader")
    logger.info("=" * 60)
    logger.info(f"S3 Endpoint: {s3_endpoint_url}")
    logger.info(f"S3 Bucket: {s3_bucket_name}")
    logger.info(f"S3 Key: {s3_key}")
    logger.info(f"Local Path: {local_path}")
    logger.info(f"Credentials provided: {bool(aws_access_key_id and aws_secret_access_key)}")
    logger.info("=" * 60)
    
    # Initialize downloader
    downloader = S3VideoDownloader(
        s3_endpoint_url=s3_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        s3_bucket_name=s3_bucket_name
    )
    
    # Download the file
    success = await downloader.download_file(s3_key, local_path)
    
    if success:
        logger.info("✅ Download completed successfully!")
        
        # Show file info
        file_path = Path(local_path)
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"📁 File saved to: {file_path.absolute()}")
            logger.info(f"📏 File size: {file_size_mb:.2f} MB")
    else:
        logger.error("❌ Download failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)