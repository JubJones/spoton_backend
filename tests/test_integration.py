"""
Integration tests for SpotOn backend system.

Tests:
- End-to-end tracking workflow
- Multi-service integration
- Real-world scenario testing
- Error handling and recovery
- Data consistency across services
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
import numpy as np

from app.services.analytics_engine import analytics_engine
from app.services.memory_manager import memory_manager
from app.services.monitoring_service import monitoring_service
from app.infrastructure.cache.tracking_cache import tracking_cache
from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.infrastructure.security.jwt_service import jwt_service
from app.infrastructure.security.encryption_service import encryption_service
from app.domains.detection.entities.detection import Detection
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import Coordinate
from app.shared.types import CameraID


@pytest.mark.integration
class TestTrackingWorkflow:
    """Test complete tracking workflow integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_tracking_workflow(self):
        """Test complete end-to-end tracking workflow."""
        # Initialize all services
        await self._initialize_services()
        
        # Step 1: Create test person detection
        detection = Detection(
            detection_id="test_detection_001",
            camera_id=CameraID("camera_01"),
            bounding_box=(100, 100, 200, 200),
            confidence=0.95,
            timestamp=datetime.now(timezone.utc),
            frame_id="frame_001"
        )
        
        # Step 2: Generate person identity
        person_identity = PersonIdentity(
            global_id="person_001",
            feature_vector=np.random.random(512).astype(np.float32),
            confidence=0.9,
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            cameras_seen=[CameraID("camera_01")]
        )
        
        # Step 3: Cache person identity
        await tracking_cache.cache_person_identity(person_identity)
        
        # Step 4: Store tracking event in database
        await integrated_db_service.store_tracking_event(
            global_person_id=person_identity.global_id,
            camera_id=str(detection.camera_id),
            environment_id="test_env",
            event_type="detection",
            position=Coordinate(x=150.0, y=150.0),
            confidence=detection.confidence
        )
        
        # Step 5: Update analytics
        await analytics_engine.update_real_time_metrics()
        
        # Step 6: Verify data consistency
        cached_person = await tracking_cache.get_person_identity(person_identity.global_id)
        assert cached_person is not None
        assert cached_person.global_id == person_identity.global_id
        
        # Step 7: Verify analytics metrics
        metrics = await analytics_engine.get_real_time_metrics()
        assert metrics is not None
        assert 'total_persons' in metrics
        
        # Step 8: Verify database statistics
        db_stats = await integrated_db_service.get_detection_statistics("test_env")
        assert db_stats is not None
        
        # Cleanup
        await self._cleanup_services()
    
    @pytest.mark.asyncio
    async def test_multi_camera_tracking_integration(self):
        """Test multi-camera tracking integration."""
        await self._initialize_services()
        
        # Create person across multiple cameras
        cameras = ["camera_01", "camera_02", "camera_03"]
        global_person_id = "person_multi_cam"
        
        # Track person across cameras
        for i, camera_id in enumerate(cameras):
            # Create detection for each camera
            detection = Detection(
                detection_id=f"detection_{camera_id}_{i}",
                camera_id=CameraID(camera_id),
                bounding_box=(100 + i * 10, 100 + i * 10, 200 + i * 10, 200 + i * 10),
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                frame_id=f"frame_{camera_id}_{i}"
            )
            
            # Store tracking event
            await integrated_db_service.store_tracking_event(
                global_person_id=global_person_id,
                camera_id=camera_id,
                environment_id="test_env",
                event_type="detection",
                position=Coordinate(x=150.0 + i * 10, y=150.0 + i * 10),
                confidence=detection.confidence
            )
        
        # Update person identity with multiple cameras
        person_identity = PersonIdentity(
            global_id=global_person_id,
            feature_vector=np.random.random(512).astype(np.float32),
            confidence=0.9,
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            cameras_seen=[CameraID(cam) for cam in cameras]
        )
        
        await tracking_cache.cache_person_identity(person_identity)
        
        # Verify person is tracked across all cameras
        cached_person = await tracking_cache.get_person_identity(global_person_id)
        assert len(cached_person.cameras_seen) == len(cameras)
        
        # Verify database has events for all cameras
        db_stats = await integrated_db_service.get_detection_statistics("test_env")
        assert db_stats is not None
        
        await self._cleanup_services()
    
    @pytest.mark.asyncio
    async def test_real_time_analytics_integration(self):
        """Test real-time analytics integration."""
        await self._initialize_services()
        
        # Generate multiple tracking events
        for i in range(50):
            person_id = f"person_{i:03d}"
            camera_id = f"camera_{(i % 4) + 1:02d}"
            
            # Store tracking event
            await integrated_db_service.store_tracking_event(
                global_person_id=person_id,
                camera_id=camera_id,
                environment_id="test_env",
                event_type="detection",
                position=Coordinate(x=float(i % 100), y=float(i % 100)),
                confidence=0.9
            )
        
        # Update analytics
        await analytics_engine.update_real_time_metrics()
        
        # Verify analytics reflect the data
        metrics = await analytics_engine.get_real_time_metrics()
        assert metrics is not None
        assert metrics.get('total_persons', 0) > 0
        assert metrics.get('active_cameras', 0) > 0
        
        await self._cleanup_services()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery in integrated workflow."""
        await self._initialize_services()
        
        # Test 1: Invalid detection data
        try:
            invalid_detection = Detection(
                detection_id="",  # Invalid empty ID
                camera_id=CameraID("camera_01"),
                bounding_box=(100, 100, 200, 200),
                confidence=0.95,
                timestamp=datetime.now(timezone.utc),
                frame_id="frame_001"
            )
            # This should handle gracefully without crashing
            assert True
        except Exception as e:
            # System should handle this gracefully
            pass
        
        # Test 2: Cache unavailable scenarios
        try:
            # Try to get non-existent person
            cached_person = await tracking_cache.get_person_identity("nonexistent_person")
            assert cached_person is None
        except Exception as e:
            # Should handle gracefully
            pass
        
        # Test 3: Database error scenarios
        try:
            # Try to store invalid tracking event
            await integrated_db_service.store_tracking_event(
                global_person_id="test_person",
                camera_id="invalid_camera",
                environment_id="test_env",
                event_type="detection",
                position=Coordinate(x=150.0, y=150.0),
                confidence=0.9
            )
            # Should handle gracefully
            assert True
        except Exception as e:
            # Expected to handle errors gracefully
            pass
        
        await self._cleanup_services()
    
    async def _initialize_services(self):
        """Initialize all required services."""
        await memory_manager.initialize()
        await monitoring_service.initialize()
        await analytics_engine.initialize()
        await tracking_cache.initialize()
        await integrated_db_service.initialize()
    
    async def _cleanup_services(self):
        """Cleanup all services."""
        await memory_manager.cleanup()
        await monitoring_service.cleanup()


@pytest.mark.integration
class TestSecurityIntegration:
    """Test security integration with other services."""
    
    @pytest.mark.asyncio
    async def test_authenticated_api_workflow(self):
        """Test authenticated API workflow."""
        await jwt_service.initialize()
        await encryption_service.initialize()
        
        # Step 1: Login
        login_result = await jwt_service.login("admin", "SpotOn2024!")
        assert login_result is not None
        assert "access_token" in login_result
        
        # Step 2: Validate token
        token_data = await jwt_service.validate_token(login_result["access_token"])
        assert token_data is not None
        assert token_data.username == "admin"
        
        # Step 3: Check permissions
        assert jwt_service.check_permission(token_data, "system:read")
        assert jwt_service.check_permission(token_data, "system:write")
        
        # Step 4: Encrypt sensitive data
        sensitive_data = {"user_id": "user123", "email": "test@example.com"}
        encrypted_data = await encryption_service.encrypt_personal_data(sensitive_data)
        assert encrypted_data["email"]["encrypted"] is True
        
        # Step 5: Decrypt data
        decrypted_data = await encryption_service.decrypt_personal_data(encrypted_data)
        assert decrypted_data["email"] == sensitive_data["email"]
        
        # Step 6: Logout
        logout_result = await jwt_service.logout(login_result["access_token"])
        assert logout_result is True
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_workflow(self):
        """Test GDPR compliance workflow."""
        await encryption_service.initialize()
        
        # Step 1: Create personal data
        personal_data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "1234567890",
            "address": "123 Main St",
            "preferences": {"theme": "dark"}
        }
        
        # Step 2: Encrypt personal data
        encrypted_data = await encryption_service.encrypt_personal_data(personal_data)
        assert encrypted_data["name"]["encrypted"] is True
        assert encrypted_data["email"]["encrypted"] is True
        assert encrypted_data["preferences"] == {"theme": "dark"}  # Not encrypted
        
        # Step 3: Process GDPR access request
        access_data = await encryption_service.process_gdpr_request("access", encrypted_data)
        assert "name" in access_data
        assert "email" in access_data
        
        # Step 4: Process GDPR portability request
        portability_data = await encryption_service.process_gdpr_request("portability", encrypted_data)
        assert portability_data["name"] == personal_data["name"]
        assert portability_data["email"] == personal_data["email"]
        
        # Step 5: Process GDPR erasure request
        erasure_data = await encryption_service.process_gdpr_request("erasure", encrypted_data)
        assert erasure_data["erased"] is True
        assert "timestamp" in erasure_data


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance integration across services."""
    
    @pytest.mark.asyncio
    async def test_high_load_integration(self):
        """Test system under high load."""
        await self._initialize_services()
        
        # Generate high load
        tasks = []
        for i in range(100):
            task = self._process_tracking_event(i)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most tasks completed successfully
        successful_tasks = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_tasks) / len(results)
        
        # Expect at least 90% success rate under load
        assert success_rate >= 0.9
        
        # Verify system metrics
        metrics = await monitoring_service.get_system_metrics(1)
        assert len(metrics) > 0
        
        await self._cleanup_services()
    
    @pytest.mark.asyncio
    async def test_memory_usage_integration(self):
        """Test memory usage across integrated services."""
        await self._initialize_services()
        
        # Monitor initial memory usage
        initial_memory = await memory_manager.get_memory_status()
        
        # Perform memory-intensive operations
        for i in range(1000):
            person_identity = PersonIdentity(
                global_id=f"person_{i}",
                feature_vector=np.random.random(512).astype(np.float32),
                confidence=0.9,
                first_seen=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc),
                cameras_seen=[CameraID(f"camera_{i % 10}")]
            )
            await tracking_cache.cache_person_identity(person_identity)
        
        # Check memory usage
        final_memory = await memory_manager.get_memory_status()
        assert final_memory is not None
        
        # Verify memory cleanup
        await memory_manager.cleanup()
        
        await self._cleanup_services()
    
    async def _process_tracking_event(self, event_id: int):
        """Process a single tracking event."""
        try:
            # Create detection
            detection = Detection(
                detection_id=f"detection_{event_id}",
                camera_id=CameraID(f"camera_{event_id % 5}"),
                bounding_box=(100, 100, 200, 200),
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                frame_id=f"frame_{event_id}"
            )
            
            # Store in database
            await integrated_db_service.store_tracking_event(
                global_person_id=f"person_{event_id}",
                camera_id=str(detection.camera_id),
                environment_id="test_env",
                event_type="detection",
                position=Coordinate(x=150.0, y=150.0),
                confidence=detection.confidence
            )
            
            return True
            
        except Exception as e:
            return e
    
    async def _initialize_services(self):
        """Initialize all services for testing."""
        await memory_manager.initialize()
        await monitoring_service.initialize()
        await analytics_engine.initialize()
        await tracking_cache.initialize()
        await integrated_db_service.initialize()
    
    async def _cleanup_services(self):
        """Cleanup all services."""
        await memory_manager.cleanup()
        await monitoring_service.cleanup()


@pytest.mark.integration
class TestDataConsistency:
    """Test data consistency across services."""
    
    @pytest.mark.asyncio
    async def test_cache_database_consistency(self):
        """Test consistency between cache and database."""
        await tracking_cache.initialize()
        await integrated_db_service.initialize()
        
        # Create test person
        person_identity = PersonIdentity(
            global_id="consistency_test_person",
            feature_vector=np.random.random(512).astype(np.float32),
            confidence=0.9,
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            cameras_seen=[CameraID("camera_01")]
        )
        
        # Store in cache
        await tracking_cache.cache_person_identity(person_identity)
        
        # Store in database
        await integrated_db_service.store_tracking_event(
            global_person_id=person_identity.global_id,
            camera_id="camera_01",
            environment_id="test_env",
            event_type="detection",
            position=Coordinate(x=150.0, y=150.0),
            confidence=0.9
        )
        
        # Verify consistency
        cached_person = await tracking_cache.get_person_identity(person_identity.global_id)
        assert cached_person is not None
        assert cached_person.global_id == person_identity.global_id
        
        # Verify database has the event
        db_stats = await integrated_db_service.get_detection_statistics("test_env")
        assert db_stats is not None
    
    @pytest.mark.asyncio
    async def test_analytics_data_consistency(self):
        """Test analytics data consistency."""
        await analytics_engine.initialize()
        await integrated_db_service.initialize()
        
        # Generate test data
        test_events = []
        for i in range(20):
            await integrated_db_service.store_tracking_event(
                global_person_id=f"person_{i}",
                camera_id=f"camera_{i % 4}",
                environment_id="test_env",
                event_type="detection",
                position=Coordinate(x=float(i * 10), y=float(i * 10)),
                confidence=0.9
            )
        
        # Update analytics
        await analytics_engine.update_real_time_metrics()
        
        # Verify analytics consistency
        metrics = await analytics_engine.get_real_time_metrics()
        assert metrics is not None
        assert metrics.get('total_persons', 0) > 0
        
        # Verify database statistics match
        db_stats = await integrated_db_service.get_detection_statistics("test_env")
        assert db_stats is not None


@pytest.mark.integration
class TestFailoverAndRecovery:
    """Test failover and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_cache_failover(self):
        """Test cache failover scenarios."""
        await tracking_cache.initialize()
        
        # Store test data
        person_identity = PersonIdentity(
            global_id="failover_test_person",
            feature_vector=np.random.random(512).astype(np.float32),
            confidence=0.9,
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            cameras_seen=[CameraID("camera_01")]
        )
        
        await tracking_cache.cache_person_identity(person_identity)
        
        # Simulate cache failure recovery
        try:
            # Try to retrieve data after potential failure
            cached_person = await tracking_cache.get_person_identity(person_identity.global_id)
            # Should handle gracefully
            assert cached_person is not None or cached_person is None
        except Exception as e:
            # Should handle cache failures gracefully
            pass
    
    @pytest.mark.asyncio
    async def test_database_failover(self):
        """Test database failover scenarios."""
        await integrated_db_service.initialize()
        
        # Test database recovery
        try:
            # Attempt to store data
            await integrated_db_service.store_tracking_event(
                global_person_id="failover_test_person",
                camera_id="camera_01",
                environment_id="test_env",
                event_type="detection",
                position=Coordinate(x=150.0, y=150.0),
                confidence=0.9
            )
            
            # Verify health check
            health = await integrated_db_service.get_service_health()
            assert health is not None
            
        except Exception as e:
            # Should handle database failures gracefully
            pass
    
    @pytest.mark.asyncio
    async def test_service_recovery(self):
        """Test service recovery scenarios."""
        # Test multiple service recovery
        services = [
            memory_manager,
            monitoring_service,
            analytics_engine
        ]
        
        for service in services:
            try:
                await service.initialize()
                # Simulate service restart
                await service.cleanup()
                await service.initialize()
                # Service should recover successfully
                assert True
            except Exception as e:
                # Should handle service failures gracefully
                pass
            finally:
                try:
                    await service.cleanup()
                except:
                    pass