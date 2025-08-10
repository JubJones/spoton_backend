"""
Tests for export API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from app.main import app


class TestExportEndpoints:
    """Test cases for export API endpoints."""
    
    def setup_method(self):
        """Set up test client and common test data."""
        self.client = TestClient(app)
        self.base_url = "/api/v1/export"
        
        # Common test data
        self.test_environment_id = "campus" 
        self.test_start_time = datetime.utcnow() - timedelta(hours=1)
        self.test_end_time = datetime.utcnow()
        
    def test_export_tracking_data_csv(self):
        """Test exporting tracking data in CSV format."""
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "format": "csv",
            "camera_ids": ["c01", "c02"],
            "confidence_threshold": 0.8,
            "include_metadata": True
        }
        
        response = self.client.post(f"{self.base_url}/tracking-data", json=request_data)
        
        # Should return job creation response
        assert response.status_code == 200
        response_data = response.json()
        
        assert "job_id" in response_data
        assert response_data["status"] == "pending"
        assert "created_at" in response_data
        assert "expires_at" in response_data
        assert response_data.get("estimated_duration_minutes") == 5
    
    def test_export_analytics_report_json(self):
        """Test exporting analytics report in JSON format."""
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "format": "json",
            "include_zone_analytics": True,
            "include_camera_analytics": True,
            "include_behavioral_patterns": True,
            "include_heatmap_data": False,
            "report_language": "en"
        }
        
        response = self.client.post(f"{self.base_url}/analytics-report", json=request_data)
        
        # Should return job creation response
        assert response.status_code == 200
        response_data = response.json()
        
        assert "job_id" in response_data
        assert response_data["status"] == "pending"
        assert "created_at" in response_data
        assert "expires_at" in response_data
        assert response_data.get("estimated_duration_minutes") == 10
    
    def test_export_video_with_overlays(self):
        """Test exporting video with overlays."""
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "camera_ids": ["c01", "c02"],
            "include_overlays": True,
            "quality": "medium",
            "fps": 10,
            "format": "mp4"
        }
        
        response = self.client.post(f"{self.base_url}/video-with-overlays", json=request_data)
        
        # Should return job creation response
        assert response.status_code == 200
        response_data = response.json()
        
        assert "job_id" in response_data
        assert response_data["status"] == "pending"
        assert "created_at" in response_data
        assert "expires_at" in response_data
        # Video exports take longer
        assert response_data.get("estimated_duration_minutes") >= 1
    
    def test_get_export_job_status(self):
        """Test getting export job status."""
        # First create a job
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "format": "csv"
        }
        
        create_response = self.client.post(f"{self.base_url}/tracking-data", json=request_data)
        assert create_response.status_code == 200
        job_id = create_response.json()["job_id"]
        
        # Get job status
        status_response = self.client.get(f"{self.base_url}/jobs/{job_id}/status")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        assert status_data["job_id"] == job_id
        assert "status" in status_data
        assert "progress" in status_data
        assert status_data["progress"] >= 0.0
        assert status_data["progress"] <= 100.0
    
    def test_list_export_jobs(self):
        """Test listing export jobs."""
        response = self.client.get(f"{self.base_url}/jobs")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "jobs" in response_data
        assert "total_count" in response_data
        assert isinstance(response_data["jobs"], list)
        assert isinstance(response_data["total_count"], int)
    
    def test_list_export_jobs_with_pagination(self):
        """Test listing export jobs with pagination."""
        response = self.client.get(f"{self.base_url}/jobs?limit=10&offset=0")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "jobs" in response_data
        assert "total_count" in response_data
        assert len(response_data["jobs"]) <= 10
    
    def test_list_export_jobs_with_status_filter(self):
        """Test listing export jobs with status filter."""
        response = self.client.get(f"{self.base_url}/jobs?status_filter=completed")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "jobs" in response_data
        assert "total_count" in response_data
        # All returned jobs should have completed status
        for job in response_data["jobs"]:
            assert job["status"] == "completed"
    
    def test_cancel_export_job(self):
        """Test canceling an export job."""
        # First create a job
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "format": "csv"
        }
        
        create_response = self.client.post(f"{self.base_url}/tracking-data", json=request_data)
        assert create_response.status_code == 200
        job_id = create_response.json()["job_id"]
        
        # Cancel the job
        cancel_response = self.client.delete(f"{self.base_url}/jobs/{job_id}")
        
        assert cancel_response.status_code == 200
        cancel_data = cancel_response.json()
        
        assert "message" in cancel_data
        assert "cancelled" in cancel_data["message"].lower()
    
    def test_download_export_file_not_found(self):
        """Test downloading export file that doesn't exist."""
        fake_job_id = "non-existent-job-id"
        
        response = self.client.get(f"{self.base_url}/jobs/{fake_job_id}/download")
        
        assert response.status_code == 404
    
    def test_get_job_status_not_found(self):
        """Test getting status of non-existent job."""
        fake_job_id = "non-existent-job-id"
        
        response = self.client.get(f"{self.base_url}/jobs/{fake_job_id}/status")
        
        assert response.status_code == 404
    
    def test_cancel_job_not_found(self):
        """Test canceling non-existent job."""
        fake_job_id = "non-existent-job-id"
        
        response = self.client.delete(f"{self.base_url}/jobs/{fake_job_id}")
        
        assert response.status_code == 404
    
    def test_export_tracking_data_validation_errors(self):
        """Test validation errors in tracking data export."""
        # Missing required fields
        request_data = {
            "format": "csv"
            # Missing environment_id, start_time, end_time
        }
        
        response = self.client.post(f"{self.base_url}/tracking-data", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_export_analytics_report_validation_errors(self):
        """Test validation errors in analytics report export."""
        # Invalid date format
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": "invalid-date-format",
            "end_time": self.test_end_time.isoformat(),
            "format": "json"
        }
        
        response = self.client.post(f"{self.base_url}/analytics-report", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_export_video_validation_errors(self):
        """Test validation errors in video export."""
        # Invalid FPS value
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "camera_ids": ["c01"],
            "fps": 100  # Too high, should be <=30
        }
        
        response = self.client.post(f"{self.base_url}/video-with-overlays", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_export_formats_supported(self):
        """Test all supported export formats."""
        formats = ["csv", "json", "excel"]
        
        for format_type in formats:
            request_data = {
                "environment_id": self.test_environment_id,
                "start_time": self.test_start_time.isoformat(),
                "end_time": self.test_end_time.isoformat(),
                "format": format_type
            }
            
            response = self.client.post(f"{self.base_url}/tracking-data", json=request_data)
            
            assert response.status_code == 200, f"Format {format_type} should be supported"
            response_data = response.json()
            assert "job_id" in response_data
    
    def test_authentication_required(self):
        """Test that authentication is handled properly."""
        # Test with Authorization header
        request_data = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "format": "csv"
        }
        
        headers = {"Authorization": "Bearer test-token"}
        response = self.client.post(
            f"{self.base_url}/tracking-data", 
            json=request_data,
            headers=headers
        )
        
        # Should still work (our mock authentication accepts any token)
        assert response.status_code == 200
    
    def test_large_date_range_estimation(self):
        """Test that larger date ranges have higher time estimates."""
        # Small date range (1 hour)
        small_request = {
            "environment_id": self.test_environment_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "camera_ids": ["c01"],
        }
        
        # Large date range (24 hours)
        large_start = datetime.utcnow() - timedelta(hours=24)
        large_request = {
            "environment_id": self.test_environment_id,
            "start_time": large_start.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "camera_ids": ["c01", "c02", "c03", "c04"],
        }
        
        small_response = self.client.post(f"{self.base_url}/video-with-overlays", json=small_request)
        large_response = self.client.post(f"{self.base_url}/video-with-overlays", json=large_request)
        
        assert small_response.status_code == 200
        assert large_response.status_code == 200
        
        small_estimate = small_response.json().get("estimated_duration_minutes", 0)
        large_estimate = large_response.json().get("estimated_duration_minutes", 0)
        
        # Large range should have higher estimate
        assert large_estimate > small_estimate


@pytest.mark.asyncio
class TestExportServiceIntegration:
    """Integration tests for export service functionality."""
    
    async def test_export_data_serialization(self):
        """Test data serialization functionality."""
        from app.domains.export.services import DataSerializationService
        from app.domains.export.entities import TrackingDataExport
        from datetime import datetime
        import tempfile
        from pathlib import Path
        
        service = DataSerializationService()
        
        # Create test data
        test_data = [
            TrackingDataExport(
                person_id="person_1",
                timestamp=datetime.utcnow(),
                camera_id="c01",
                bbox_x=100.0,
                bbox_y=200.0,
                bbox_width=50.0,
                bbox_height=100.0,
                confidence=0.85,
                map_x=10.5,
                map_y=20.5,
                tracking_duration=15.5,
                detection_metadata={"test": "data"}
            ),
            TrackingDataExport(
                person_id="person_2",
                timestamp=datetime.utcnow(),
                camera_id="c02",
                bbox_x=150.0,
                bbox_y=250.0,
                bbox_width=60.0,
                bbox_height=110.0,
                confidence=0.90,
                map_x=15.5,
                map_y=25.5,
                tracking_duration=20.5,
                detection_metadata={"test": "data2"}
            )
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test CSV serialization
            csv_file = await service._serialize_to_csv(test_data, "test_export", temp_path)
            assert csv_file.exists()
            assert csv_file.suffix == ".csv"
            
            # Test JSON serialization
            json_file = await service._serialize_to_json(test_data, "test_export", temp_path)
            assert json_file.exists()
            assert json_file.suffix == ".json"
            
            # Verify JSON content
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                assert len(json_data) == 2
                assert json_data[0]["person_id"] == "person_1"
                assert json_data[1]["person_id"] == "person_2"
    
    async def test_report_generation(self):
        """Test report generation functionality."""
        from app.domains.export.services import ReportGeneratorService
        from app.domains.export.entities import AnalyticsReportData, ExportFormat
        import tempfile
        from pathlib import Path
        
        service = ReportGeneratorService()
        
        # Create test analytics data
        test_data = AnalyticsReportData(
            environment_id="campus",
            report_period="2024-01-01 to 2024-01-02",
            total_persons=100,
            unique_persons=75,
            avg_dwell_time=120.5,
            peak_occupancy=25,
            zone_analytics={
                "zone_1": {"occupancy": 15, "dwell_time": 45.2},
                "zone_2": {"occupancy": 10, "dwell_time": 60.1}
            },
            camera_analytics={
                "c01": {"detections": 150, "confidence": 0.85},
                "c02": {"detections": 120, "confidence": 0.82}
            },
            behavioral_patterns={"loitering": 5, "rushing": 3},
            movement_patterns=[
                {"pattern": "linear", "count": 45},
                {"pattern": "circular", "count": 20}
            ]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test JSON report generation
            json_report = await service._generate_json_report(test_data, "test_report", temp_path)
            assert json_report.exists()
            assert json_report.suffix == ".json"
            
            # Verify JSON content structure
            with open(json_report, 'r') as f:
                report_data = json.load(f)
                assert report_data["environment_id"] == "campus"
                assert report_data["total_persons"] == 100
                assert "zone_analytics" in report_data
                assert "camera_analytics" in report_data


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])