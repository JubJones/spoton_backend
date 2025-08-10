"""
Test suite for data ingestion functionality.

Tests the DagHub S3 service and data ingestion API endpoints.
"""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import tempfile
import shutil

from app.services.dagshub_s3_service import (
    DagHubS3Service, 
    VideoFileMetadata, 
    DownloadStatus, 
    DownloadProgress
)
from app.api.v1.endpoints.data_ingestion import router
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Test fixtures

@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    mock_client = Mock()
    mock_client.head_bucket.return_value = None
    mock_client.get_paginator.return_value.paginate.return_value = [
        {
            'Contents': [
                {'Key': 'video_s14/c09/sub_video_01.mp4'},
                {'Key': 'video_s14/c09/sub_video_02.mp4'},
                {'Key': 'video_s14/c12/sub_video_01.mp4'},
                {'Key': 'video_s37/c01/sub_video_01.mp4'},
            ]
        }
    ]
    mock_client.head_object.return_value = {
        'ContentLength': 1024000,
        'LastModified': datetime.utcnow(),
        'ETag': '"abcd1234"'
    }
    return mock_client


@pytest.fixture
def temp_download_dir():
    """Temporary directory for downloads."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_storage_settings():
    """Mock storage settings."""
    with patch('app.services.dagshub_s3_service.get_storage_settings') as mock_settings:
        settings = Mock()
        settings.s3.provider.value = "dagshub"
        settings.s3.bucket_name = "test_bucket"
        settings.s3.endpoint_url = "https://test.dagshub.com"
        settings.s3.access_key_id = "test_key"
        settings.s3.secret_access_key = "test_secret"
        settings.s3.region = "us-east-1"
        settings.s3.max_retries = 3
        settings.s3.max_concurrency = 5
        settings.s3.read_timeout = 300
        settings.s3.connect_timeout = 60
        settings.local.video_download_dir = "/tmp/test_downloads"
        mock_settings.return_value = settings
        yield settings


# Unit Tests for DagHubS3Service

class TestDagHubS3Service:
    """Test cases for DagHub S3 service."""

    @patch('app.services.dagshub_s3_service.boto3.client')
    def test_service_initialization(self, mock_boto_client, mock_storage_settings, mock_s3_client):
        """Test service initialization."""
        mock_boto_client.return_value = mock_s3_client
        
        service = DagHubS3Service()
        
        assert service.s3_client is not None
        mock_boto_client.assert_called_once()
        mock_s3_client.head_bucket.assert_called_once()

    @patch('app.services.dagshub_s3_service.boto3.client')
    def test_get_dataset_structure(self, mock_boto_client, mock_storage_settings, mock_s3_client):
        """Test dataset structure discovery."""
        mock_boto_client.return_value = mock_s3_client
        
        service = DagHubS3Service()
        structure = service.get_dataset_structure()
        
        expected_structure = {
            'video_s14': {
                'c09': ['sub_video_01.mp4', 'sub_video_02.mp4'],
                'c12': ['sub_video_01.mp4']
            },
            'video_s37': {
                'c01': ['sub_video_01.mp4']
            }
        }
        
        assert structure == expected_structure

    @patch('app.services.dagshub_s3_service.boto3.client')
    def test_create_video_file_inventory(self, mock_boto_client, mock_storage_settings, mock_s3_client):
        """Test video file inventory creation."""
        mock_boto_client.return_value = mock_s3_client
        
        service = DagHubS3Service()
        inventory = service.create_video_file_inventory(
            video_sets=['video_s14'],
            cameras=['c09']
        )
        
        assert len(inventory) == 2  # Two sub_videos for c09
        assert all(isinstance(item, VideoFileMetadata) for item in inventory)
        assert inventory[0].video_set == 'video_s14'
        assert inventory[0].camera_id == 'c09'

    @patch('app.services.dagshub_s3_service.boto3.client')
    @patch('app.services.dagshub_s3_service.asyncio.to_thread')
    async def test_get_file_metadata(self, mock_to_thread, mock_boto_client, mock_storage_settings, mock_s3_client):
        """Test S3 file metadata retrieval."""
        mock_boto_client.return_value = mock_s3_client
        mock_to_thread.return_value = {
            'ContentLength': 1024000,
            'LastModified': datetime.utcnow(),
            'ETag': '"abcd1234"'
        }
        
        service = DagHubS3Service()
        metadata = await service.get_file_metadata('video_s14/c09/sub_video_01.mp4')
        
        assert metadata is not None
        assert metadata['size'] == 1024000
        assert 'etag' in metadata

    @patch('app.services.dagshub_s3_service.boto3.client')
    @patch('app.services.dagshub_s3_service.asyncio.to_thread')
    async def test_download_file_with_validation_success(self, mock_to_thread, mock_boto_client, 
                                                        mock_storage_settings, mock_s3_client, temp_download_dir):
        """Test successful file download with validation."""
        mock_boto_client.return_value = mock_s3_client
        
        # Mock asyncio.to_thread calls
        async def mock_to_thread_side_effect(func, *args, **kwargs):
            if func == mock_s3_client.head_object:
                return {
                    'ContentLength': 1024000,
                    'LastModified': datetime.utcnow(),
                    'ETag': '"abcd1234"'
                }
            elif func == mock_s3_client.download_file:
                # Create a dummy file
                local_path = args[2] if len(args) > 2 else kwargs.get('Filename')
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                Path(local_path).write_bytes(b'0' * 1024000)  # 1MB dummy file
                return None
            return None
        
        mock_to_thread.side_effect = mock_to_thread_side_effect
        
        service = DagHubS3Service()
        
        file_metadata = VideoFileMetadata(
            video_set='video_s14',
            camera_id='c09',
            sub_video_id='sub_video_01',
            s3_key='video_s14/c09/sub_video_01.mp4',
            local_path=temp_download_dir / 'video_s14' / 'c09' / 'sub_video_01.mp4'
        )
        
        result = await service.download_file_with_validation(file_metadata)
        
        assert result is True
        assert file_metadata.status == DownloadStatus.COMPLETED
        assert file_metadata.local_path.exists()

    @patch('app.services.dagshub_s3_service.boto3.client')
    async def test_batch_download_videos(self, mock_boto_client, mock_storage_settings, mock_s3_client, temp_download_dir):
        """Test batch download functionality."""
        mock_boto_client.return_value = mock_s3_client
        
        service = DagHubS3Service()
        
        # Mock the download method to avoid actual S3 calls
        async def mock_download_file(file_metadata, progress_callback=None):
            file_metadata.status = DownloadStatus.COMPLETED
            file_metadata.file_size = 1024000
            if progress_callback:
                progress_callback(file_metadata)
            return True
        
        service.download_file_with_validation = mock_download_file
        
        progress = await service.batch_download_videos(
            video_sets=['video_s14'],
            cameras=['c09'],
            max_concurrent=2,
            progress_id='test-progress'
        )
        
        assert isinstance(progress, DownloadProgress)
        assert progress.total_files > 0
        assert progress.completed_files == progress.total_files
        assert progress.is_complete

    @patch('app.services.dagshub_s3_service.boto3.client')
    async def test_verify_dataset_integrity(self, mock_boto_client, mock_storage_settings, mock_s3_client, temp_download_dir):
        """Test dataset integrity verification."""
        mock_boto_client.return_value = mock_s3_client
        
        service = DagHubS3Service()
        
        # Create some test files
        test_file_1 = temp_download_dir / 'video_s14' / 'c09' / 'sub_video_01.mp4'
        test_file_1.parent.mkdir(parents=True, exist_ok=True)
        test_file_1.write_bytes(b'test content')
        
        # Mock the inventory creation to use our temp directory
        def mock_create_inventory(video_sets=None, cameras=None):
            return [
                VideoFileMetadata(
                    video_set='video_s14',
                    camera_id='c09',
                    sub_video_id='sub_video_01',
                    s3_key='video_s14/c09/sub_video_01.mp4',
                    local_path=test_file_1
                ),
                VideoFileMetadata(
                    video_set='video_s14',
                    camera_id='c09',
                    sub_video_id='sub_video_02',
                    s3_key='video_s14/c09/sub_video_02.mp4',
                    local_path=temp_download_dir / 'video_s14' / 'c09' / 'sub_video_02.mp4'
                )
            ]
        
        service.create_video_file_inventory = mock_create_inventory
        
        results = await service.verify_dataset_integrity(
            video_sets=['video_s14'],
            cameras=['c09']
        )
        
        assert results['total_files'] == 2
        assert results['existing_files'] == 1
        assert results['missing_files'] == 1
        assert results['completeness_percentage'] == 50.0


# Integration Tests for API Endpoints

class TestDataIngestionAPI:
    """Test cases for data ingestion API endpoints."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1/data-ingestion")
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    @patch('app.api.v1.endpoints.data_ingestion.get_dagshub_s3_service')
    def test_get_dataset_structure_endpoint(self, mock_service_dep, client):
        """Test dataset structure endpoint."""
        mock_service = Mock()
        mock_service.get_dataset_structure.return_value = {
            'video_s14': {'c09': ['sub_video_01.mp4']},
            'video_s37': {'c01': ['sub_video_01.mp4']}
        }
        mock_service_dep.return_value = mock_service
        
        response = client.get("/api/v1/data-ingestion/dataset/structure")
        
        assert response.status_code == 200
        data = response.json()
        assert 'structure' in data
        assert 'total_video_sets' in data
        assert data['total_video_sets'] == 2

    @patch('app.api.v1.endpoints.data_ingestion.get_dagshub_s3_service')
    def test_start_batch_download_endpoint(self, mock_service_dep, client):
        """Test batch download start endpoint."""
        mock_service = Mock()
        mock_service.get_dataset_structure.return_value = {
            'video_s14': {'c09': ['sub_video_01.mp4']}
        }
        mock_service_dep.return_value = mock_service
        
        request_data = {
            "video_sets": ["video_s14"],
            "cameras": ["c09"],
            "max_concurrent": 5
        }
        
        response = client.post(
            "/api/v1/data-ingestion/dataset/download", 
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'progress_id' in data
        assert data['download_started'] is True

    @patch('app.api.v1.endpoints.data_ingestion.get_dagshub_s3_service')
    def test_get_download_progress_endpoint(self, mock_service_dep, client):
        """Test download progress endpoint."""
        mock_service = Mock()
        mock_progress = DownloadProgress(
            total_files=10,
            completed_files=5,
            failed_files=1,
            start_time=datetime.utcnow()
        )
        mock_service.get_download_progress.return_value = mock_progress
        mock_service_dep.return_value = mock_service
        
        response = client.get("/api/v1/data-ingestion/dataset/download/test-id/progress")
        
        assert response.status_code == 200
        data = response.json()
        assert 'progress' in data
        assert data['progress']['total_files'] == 10
        assert data['progress']['completed_files'] == 5

    @patch('app.api.v1.endpoints.data_ingestion.get_dagshub_s3_service')
    def test_verify_dataset_integrity_endpoint(self, mock_service_dep, client):
        """Test dataset verification endpoint."""
        mock_service = Mock()
        mock_service.verify_dataset_integrity = AsyncMock(return_value={
            'total_files': 100,
            'existing_files': 95,
            'missing_files': 5,
            'completeness_percentage': 95.0,
            'integrity_score': 95.0
        })
        mock_service_dep.return_value = mock_service
        
        request_data = {
            "video_sets": ["video_s14"],
            "cameras": ["c09"]
        }
        
        response = client.post(
            "/api/v1/data-ingestion/dataset/verify",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'verification_results' in data
        assert data['verification_results']['completeness_percentage'] == 95.0

    @patch('app.api.v1.endpoints.data_ingestion.get_dagshub_s3_service')
    def test_health_check_endpoint(self, mock_service_dep, client):
        """Test health check endpoint."""
        mock_service = Mock()
        mock_service.get_dataset_structure.return_value = {
            'video_s14': {'c09': ['sub_video_01.mp4']},
            'video_s37': {'c01': ['sub_video_01.mp4']}
        }
        mock_service_dep.return_value = mock_service
        
        response = client.get("/api/v1/data-ingestion/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'data_ingestion'
        assert 'available_video_sets' in data


# Error Handling Tests

class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('app.services.dagshub_s3_service.boto3.client')
    def test_s3_connection_failure(self, mock_boto_client, mock_storage_settings):
        """Test handling of S3 connection failures."""
        mock_client = Mock()
        mock_client.head_bucket.side_effect = Exception("Connection failed")
        mock_boto_client.return_value = mock_client
        
        with pytest.raises(Exception):
            DagHubS3Service()

    @patch('app.services.dagshub_s3_service.boto3.client')
    async def test_download_failure_handling(self, mock_boto_client, mock_storage_settings, mock_s3_client, temp_download_dir):
        """Test handling of download failures."""
        mock_boto_client.return_value = mock_s3_client
        
        # Mock download to fail
        async def mock_failing_to_thread(func, *args, **kwargs):
            if func == mock_s3_client.download_file:
                raise Exception("Download failed")
            return {
                'ContentLength': 1024000,
                'LastModified': datetime.utcnow(),
                'ETag': '"abcd1234"'
            }
        
        with patch('app.services.dagshub_s3_service.asyncio.to_thread', side_effect=mock_failing_to_thread):
            service = DagHubS3Service()
            
            file_metadata = VideoFileMetadata(
                video_set='video_s14',
                camera_id='c09',
                sub_video_id='sub_video_01',
                s3_key='video_s14/c09/sub_video_01.mp4',
                local_path=temp_download_dir / 'test.mp4'
            )
            
            result = await service.download_file_with_validation(file_metadata)
            
            assert result is False
            assert file_metadata.status == DownloadStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])