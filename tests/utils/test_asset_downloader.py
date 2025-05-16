"""
Unit tests for the AssetDownloader class in app.utils.asset_downloader.
"""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.utils.asset_downloader import AssetDownloader
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Use the mock_settings fixture from conftest.py
@pytest.fixture
def s3_config(mock_settings):
    """Provides S3 configuration from mock_settings."""
    return {
        "s3_endpoint_url": mock_settings.S3_ENDPOINT_URL,
        "aws_access_key_id": mock_settings.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": mock_settings.AWS_SECRET_ACCESS_KEY,
        "s3_bucket_name": mock_settings.S3_BUCKET_NAME,
    }

@pytest.fixture
def mock_boto_s3_client(mocker):
    """Mocks the boto3 S3 client."""
    mock_client_instance = MagicMock()
    mocker.patch("boto3.client", return_value=mock_client_instance)
    return mock_client_instance


def test_asset_downloader_init_success(s3_config, mock_boto_s3_client, mocker):
    """Tests successful initialization of AssetDownloader."""
    mock_logger_info = mocker.patch("app.utils.asset_downloader.logger.info")
    downloader = AssetDownloader(**s3_config)
    assert downloader.s3_client == mock_boto_s3_client
    mock_logger_info.assert_called_with(
        f"Boto3 S3 client initialized. Endpoint: '{s3_config['s3_endpoint_url']}', "
        f"Bucket: '{s3_config['s3_bucket_name']}'"
    )

def test_asset_downloader_init_missing_credentials(s3_config, mock_boto_s3_client, mocker):
    """Tests initialization with missing credentials (should still init client but log warning)."""
    config_no_creds = s3_config.copy()
    config_no_creds["aws_access_key_id"] = None
    mock_logger_warning = mocker.patch("app.utils.asset_downloader.logger.warning")
    downloader = AssetDownloader(**config_no_creds)
    assert downloader.s3_client is not None # Client should still be attempted
    mock_logger_warning.assert_called_with(
        "AWS Access Key ID or Secret Access Key is not configured. "
        "S3 downloads might fail if credentials are required by the endpoint."
    )

def test_asset_downloader_init_missing_bucket_name(s3_config, mocker):
    """Tests initialization with missing bucket name (should not init client and log error)."""
    config_no_bucket = s3_config.copy()
    config_no_bucket["s3_bucket_name"] = "" # type: ignore
    mock_logger_error = mocker.patch("app.utils.asset_downloader.logger.error")
    # Prevent boto3.client from actually being called since it would error earlier
    mocker.patch("boto3.client")

    downloader = AssetDownloader(**config_no_bucket)
    assert downloader.s3_client is None
    mock_logger_error.assert_called_with("S3_BUCKET_NAME is not configured. Cannot initialize S3 client.")


@pytest.mark.asyncio
async def test_download_file_successful(tmp_path, s3_config, mock_boto_s3_client, mocker):
    """Tests successful file download."""
    downloader = AssetDownloader(**s3_config)
    downloader.s3_client = mock_boto_s3_client # Ensure it's using the mock

    remote_key = "test/video.mp4"
    local_path = tmp_path / "video.mp4"

    mock_async_to_thread = mocker.patch("asyncio.to_thread", return_value=None) # Simulate successful download

    success = await downloader.download_file_from_dagshub(remote_key, str(local_path))

    assert success is True
    mock_async_to_thread.assert_called_once_with(
        mock_boto_s3_client.download_file,
        Bucket=s3_config["s3_bucket_name"],
        Key=remote_key,
        Filename=str(local_path),
    )
    assert local_path.parent.exists() # Ensure parent directory was created

@pytest.mark.asyncio
async def test_download_file_s3_client_not_initialized(s3_config, mocker):
    """Tests download attempt when S3 client failed to initialize."""
    config_no_bucket = s3_config.copy()
    config_no_bucket["s3_bucket_name"] = "" # type: ignore
    mocker.patch("boto3.client") # Ensure client init fails
    mock_logger_error = mocker.patch("app.utils.asset_downloader.logger.error")

    downloader = AssetDownloader(**config_no_bucket) # This will set s3_client to None
    assert downloader.s3_client is None

    success = await downloader.download_file_from_dagshub("test/file.txt", "local_file.txt")
    assert success is False
    mock_logger_error.assert_any_call("S3 client not initialized. Cannot download.")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_type, expected_log_keyword",
    [
        (ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "download_file"), "File not found"),
        (ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "download_file"), "File not found"),
        (ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "download_file"), "Access Denied"),
        (NoCredentialsError(), "S3 credentials error"),
        (PartialCredentialsError(provider="aws", cred_var="secret_key"), "S3 credentials error"),
        (Exception("Some generic error"), "Unexpected error downloading"),
    ],
)
async def test_download_file_various_errors(
    tmp_path, s3_config, mock_boto_s3_client, mocker, exception_type, expected_log_keyword
):
    """Tests download failures due to various S3 errors."""
    downloader = AssetDownloader(**s3_config)
    downloader.s3_client = mock_boto_s3_client

    mock_boto_s3_client.download_file.side_effect = exception_type
    mock_async_to_thread = mocker.patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))


    mock_logger_warning = mocker.patch("app.utils.asset_downloader.logger.warning")
    mock_logger_error = mocker.patch("app.utils.asset_downloader.logger.error")

    success = await downloader.download_file_from_dagshub("test/file.txt", str(tmp_path / "file.txt"))
    assert success is False

    # Check if the expected keyword is in any of the error/warning log messages
    log_found = False
    for call_args in mock_logger_error.call_args_list + mock_logger_warning.call_args_list:
        if expected_log_keyword in call_args[0][0]:
            log_found = True
            break
    assert log_found, f"Expected log keyword '{expected_log_keyword}' not found for exception {exception_type}"

@pytest.mark.asyncio
async def test_download_file_cannot_create_parent_dir(s3_config, mock_boto_s3_client, mocker):
    """Tests download failure if parent directory for local file cannot be created."""
    downloader = AssetDownloader(**s3_config)
    downloader.s3_client = mock_boto_s3_client
    mock_logger_error = mocker.patch("app.utils.asset_downloader.logger.error")

    mocker.patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied"))

    success = await downloader.download_file_from_dagshub("test/file.txt", "/forbidden_dir/file.txt")
    assert success is False
    mock_logger_error.assert_any_call("Could not create parent directory for /forbidden_dir/file.txt: Permission denied")