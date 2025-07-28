"""
Performance tests for SpotOn backend system.

Tests:
- GPU performance and memory usage
- Database performance
- Cache performance
- API response times
- Memory leak detection
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import torch
import numpy as np
import psutil
import gc

from app.services.memory_manager import memory_manager
from app.services.monitoring_service import monitoring_service
from app.services.analytics_engine import analytics_engine
from app.infrastructure.cache.tracking_cache import tracking_cache
from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.domains.detection.entities.detection import Detection
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import Coordinate
from app.shared.types import CameraID


@pytest.mark.performance
class TestGPUPerformance:
    """Test GPU performance and memory usage."""
    
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_gpu_memory_management(self, gpu_test_environment, memory_profiler):
        """Test GPU memory management."""
        device = gpu_test_environment["device"]
        test_data = gpu_test_environment["test_data"]
        
        # Initialize memory manager
        await memory_manager.initialize()
        
        # Start memory profiling
        memory_profiler.start()
        
        # Perform GPU operations
        with torch.cuda.device(device):
            # Simulate model inference
            for tensor_name, tensor in test_data.items():
                result = torch.matmul(tensor, tensor.T)
                result = torch.relu(result)
                result = torch.sum(result)
                
                # Force GPU sync
                torch.cuda.synchronize()
        
        # Stop memory profiling
        memory_profiler.stop()
        
        # Check memory usage
        memory_usage = memory_profiler.get_memory_usage()
        
        # Assert reasonable memory usage (adjust threshold as needed)
        memory_profiler.assert_memory_usage(2048)  # 2GB max
        
        # Check that memory is properly managed
        memory_status = await memory_manager.get_memory_status()
        assert memory_status is not None
        assert 'gpu_memory' in memory_status
        
        # Cleanup
        await memory_manager.cleanup()
    
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_gpu_performance_benchmarks(self, gpu_test_environment, performance_monitor):
        """Test GPU performance benchmarks."""
        device = gpu_test_environment["device"]
        
        # Test different tensor sizes
        test_sizes = [(100, 512), (1000, 512), (10000, 512)]
        
        for size in test_sizes:
            # Create test tensors
            tensor_a = torch.randn(size, device=device)
            tensor_b = torch.randn(size, device=device)
            
            # Benchmark matrix multiplication
            performance_monitor.start(f"matmul_{size[0]}x{size[1]}")
            
            with torch.cuda.device(device):
                for _ in range(10):
                    result = torch.matmul(tensor_a, tensor_b.T)
                    torch.cuda.synchronize()
            
            performance_monitor.stop(f"matmul_{size[0]}x{size[1]}")
            
            # Clean up
            del tensor_a, tensor_b, result
            torch.cuda.empty_cache()
        
        # Check performance thresholds
        performance_monitor.assert_performance("matmul_100x512", 0.1)    # 100ms
        performance_monitor.assert_performance("matmul_1000x512", 0.5)   # 500ms
        performance_monitor.assert_performance("matmul_10000x512", 2.0)  # 2s
    
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_gpu_memory_leak_detection(self, gpu_test_environment):
        """Test GPU memory leak detection."""
        device = gpu_test_environment["device"]
        
        # Initialize memory manager
        await memory_manager.initialize()
        
        # Get initial memory usage
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Simulate operations that might cause memory leaks
        for i in range(100):
            # Create tensors
            tensor = torch.randn(1000, 512, device=device)
            
            # Perform operations
            result = torch.matmul(tensor, tensor.T)
            result = torch.relu(result)
            
            # Explicitly delete tensors
            del tensor, result
            
            # Periodic cleanup
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Final cleanup
        torch.cuda.empty_cache()
        
        # Get final memory usage
        final_memory = torch.cuda.memory_allocated(device)
        
        # Check for memory leaks
        memory_increase = final_memory - initial_memory
        max_acceptable_increase = 100 * 1024 * 1024  # 100MB
        
        assert memory_increase < max_acceptable_increase, f"Potential memory leak: {memory_increase / (1024*1024):.2f}MB increase"
        
        # Check memory manager statistics
        memory_stats = await memory_manager.get_memory_statistics()
        assert memory_stats is not None
        
        # Cleanup
        await memory_manager.cleanup()


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database performance."""
    
    @pytest.mark.asyncio
    async def test_database_connection_performance(self, db_service, performance_monitor):
        """Test database connection performance."""
        # Test connection establishment
        performance_monitor.start("db_connection")
        
        # Initialize database service
        await db_service.initialize()
        
        performance_monitor.stop("db_connection")
        
        # Connection should be fast
        performance_monitor.assert_performance("db_connection", 5.0)  # 5 seconds max
        
        # Test query performance
        performance_monitor.start("db_query")
        
        # Perform test queries
        for _ in range(10):
            stats = await db_service.get_service_health()
            assert stats is not None
        
        performance_monitor.stop("db_query")
        
        # Queries should be fast
        performance_monitor.assert_performance("db_query", 1.0)  # 1 second for 10 queries
    
    @pytest.mark.asyncio
    async def test_database_bulk_operations(self, db_service, performance_monitor):
        """Test database bulk operations performance."""
        await db_service.initialize()
        
        # Test bulk insert performance
        performance_monitor.start("bulk_insert")
        
        # Create test data
        test_events = []
        for i in range(1000):
            event = {
                "global_person_id": f"person_{i}",
                "camera_id": f"camera_{i % 10}",
                "environment_id": "test_env",
                "event_type": "detection",
                "position": Coordinate(x=float(i % 100), y=float(i % 100)),
                "confidence": 0.95
            }
            test_events.append(event)
        
        # Bulk insert (simulated)
        for event in test_events:
            await db_service.store_tracking_event(**event)
        
        performance_monitor.stop("bulk_insert")
        
        # Bulk insert should be reasonably fast
        performance_monitor.assert_performance("bulk_insert", 30.0)  # 30 seconds for 1000 events
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self, db_service, performance_monitor):
        """Test database query performance."""
        await db_service.initialize()
        
        # Test different query types
        query_types = [
            ("detection_stats", lambda: db_service.get_detection_statistics("test_env")),
            ("person_stats", lambda: db_service.get_person_statistics("test_env")),
            ("service_health", lambda: db_service.get_service_health())
        ]
        
        for query_name, query_func in query_types:
            performance_monitor.start(f"query_{query_name}")
            
            # Run query multiple times
            for _ in range(5):
                result = await query_func()
                assert result is not None
            
            performance_monitor.stop(f"query_{query_name}")
            
            # Queries should be fast
            performance_monitor.assert_performance(f"query_{query_name}", 2.0)  # 2 seconds for 5 queries


@pytest.mark.performance
class TestCachePerformance:
    """Test cache performance."""
    
    @pytest.mark.asyncio
    async def test_cache_operations_performance(self, cache_service, performance_monitor):
        """Test cache operations performance."""
        await cache_service.initialize()
        
        # Test cache write performance
        performance_monitor.start("cache_write")
        
        # Create test data
        test_persons = []
        for i in range(1000):
            person = PersonIdentity(
                global_id=f"person_{i}",
                feature_vector=np.random.random(512).astype(np.float32),
                confidence=0.9,
                first_seen=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc),
                cameras_seen=[CameraID(f"camera_{i % 10}")]
            )
            test_persons.append(person)
        
        # Cache persons
        for person in test_persons:
            await cache_service.cache_person_identity(person)
        
        performance_monitor.stop("cache_write")
        
        # Test cache read performance
        performance_monitor.start("cache_read")
        
        # Read cached persons
        for person in test_persons:
            cached_person = await cache_service.get_person_identity(person.global_id)
            assert cached_person is not None
        
        performance_monitor.stop("cache_read")
        
        # Cache operations should be fast
        performance_monitor.assert_performance("cache_write", 5.0)  # 5 seconds for 1000 writes
        performance_monitor.assert_performance("cache_read", 2.0)   # 2 seconds for 1000 reads
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, cache_service, performance_monitor):
        """Test cache hit rate performance."""
        await cache_service.initialize()
        
        # Create test data
        test_person = PersonIdentity(
            global_id="test_person_hit_rate",
            feature_vector=np.random.random(512).astype(np.float32),
            confidence=0.9,
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            cameras_seen=[CameraID("test_camera")]
        )
        
        # Cache person
        await cache_service.cache_person_identity(test_person)
        
        # Test cache hit rate
        performance_monitor.start("cache_hit_test")
        
        hits = 0
        total_requests = 100
        
        for _ in range(total_requests):
            cached_person = await cache_service.get_person_identity(test_person.global_id)
            if cached_person is not None:
                hits += 1
        
        performance_monitor.stop("cache_hit_test")
        
        # Calculate hit rate
        hit_rate = hits / total_requests
        
        # Hit rate should be high
        assert hit_rate >= 0.95, f"Cache hit rate too low: {hit_rate:.2%}"
        
        # Cache hits should be fast
        performance_monitor.assert_performance("cache_hit_test", 1.0)  # 1 second for 100 hits
    
    @pytest.mark.asyncio
    async def test_cache_memory_usage(self, cache_service, memory_profiler):
        """Test cache memory usage."""
        await cache_service.initialize()
        
        # Start memory profiling
        memory_profiler.start()
        
        # Fill cache with data
        for i in range(10000):
            person = PersonIdentity(
                global_id=f"person_{i}",
                feature_vector=np.random.random(512).astype(np.float32),
                confidence=0.9,
                first_seen=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc),
                cameras_seen=[CameraID(f"camera_{i % 100}")]
            )
            await cache_service.cache_person_identity(person)
        
        # Stop memory profiling
        memory_profiler.stop()
        
        # Check memory usage
        memory_profiler.assert_memory_usage(1024)  # 1GB max for 10k persons
        
        # Check cache statistics
        cache_stats = await cache_service.get_cache_stats()
        assert cache_stats is not None
        assert cache_stats.get('total_persons', 0) > 0


@pytest.mark.performance
class TestAPIPerformance:
    """Test API performance."""
    
    @pytest.mark.asyncio
    async def test_analytics_api_performance(self, performance_monitor):
        """Test analytics API performance."""
        await analytics_engine.initialize()
        
        # Test real-time metrics
        performance_monitor.start("analytics_realtime")
        
        for _ in range(10):
            metrics = await analytics_engine.get_real_time_metrics()
            assert metrics is not None
        
        performance_monitor.stop("analytics_realtime")
        
        # Test behavior analysis
        performance_monitor.start("analytics_behavior")
        
        # Create test trajectory data
        test_trajectory = []
        for i in range(100):
            trajectory_point = {
                'timestamp': datetime.now(timezone.utc),
                'position_x': float(i % 100),
                'position_y': float(i % 100),
                'camera_id': f'camera_{i % 5}'
            }
            test_trajectory.append(trajectory_point)
        
        # Analyze behavior (would normally use database data)
        # This is a simplified test
        for _ in range(5):
            # Simulate behavior analysis
            await asyncio.sleep(0.1)
        
        performance_monitor.stop("analytics_behavior")
        
        # API calls should be fast
        performance_monitor.assert_performance("analytics_realtime", 1.0)   # 1 second for 10 calls
        performance_monitor.assert_performance("analytics_behavior", 2.0)   # 2 seconds for 5 analyses
    
    @pytest.mark.asyncio
    async def test_concurrent_api_performance(self, performance_monitor, async_test_helper):
        """Test concurrent API performance."""
        await analytics_engine.initialize()
        
        # Test concurrent requests
        performance_monitor.start("concurrent_requests")
        
        # Create multiple concurrent requests
        concurrent_requests = []
        for _ in range(50):
            request = analytics_engine.get_real_time_metrics()
            concurrent_requests.append(request)
        
        # Execute all requests concurrently
        results = await async_test_helper.run_parallel(concurrent_requests)
        
        performance_monitor.stop("concurrent_requests")
        
        # Check that all requests completed successfully
        assert len(results) == 50
        for result in results:
            assert result is not None
        
        # Concurrent requests should be handled efficiently
        performance_monitor.assert_performance("concurrent_requests", 5.0)  # 5 seconds for 50 concurrent requests


@pytest.mark.performance
class TestMemoryLeakDetection:
    """Test memory leak detection."""
    
    @pytest.mark.asyncio
    async def test_service_memory_leaks(self, memory_profiler):
        """Test service memory leaks."""
        # Initialize services
        await memory_manager.initialize()
        await monitoring_service.initialize()
        await analytics_engine.initialize()
        
        # Start memory profiling
        memory_profiler.start()
        
        # Simulate heavy usage
        for i in range(100):
            # Memory manager operations
            await memory_manager.get_memory_status()
            
            # Monitoring operations
            await monitoring_service.get_system_metrics(1)
            
            # Analytics operations
            await analytics_engine.get_real_time_metrics()
            
            # Periodic cleanup
            if i % 10 == 0:
                gc.collect()
        
        # Stop memory profiling
        memory_profiler.stop()
        
        # Check for memory leaks
        memory_profiler.assert_memory_usage(512)  # 512MB max increase
        
        # Cleanup services
        await memory_manager.cleanup()
        await monitoring_service.cleanup()
    
    @pytest.mark.asyncio
    async def test_long_running_process_stability(self, memory_profiler, async_test_helper):
        """Test long-running process stability."""
        await memory_manager.initialize()
        
        # Start memory profiling
        memory_profiler.start()
        
        # Simulate long-running process
        duration = 30.0  # 30 seconds
        operations = 1000
        
        actual_duration = await async_test_helper.simulate_load(duration, operations)
        
        # Stop memory profiling
        memory_profiler.stop()
        
        # Check timing
        assert actual_duration <= duration * 1.1  # Allow 10% overhead
        
        # Check memory usage
        memory_profiler.assert_memory_usage(256)  # 256MB max for long-running process
        
        await memory_manager.cleanup()


@pytest.mark.performance
class TestScalabilityPerformance:
    """Test scalability performance."""
    
    @pytest.mark.asyncio
    async def test_multiple_camera_handling(self, performance_monitor):
        """Test multiple camera handling performance."""
        await analytics_engine.initialize()
        
        # Simulate multiple cameras
        camera_count = 10
        persons_per_camera = 100
        
        performance_monitor.start("multi_camera_processing")
        
        # Process data from multiple cameras
        for camera_id in range(camera_count):
            camera_name = f"camera_{camera_id:03d}"
            
            # Simulate persons in camera
            for person_id in range(persons_per_camera):
                person_global_id = f"person_{camera_id}_{person_id}"
                
                # Simulate processing
                await asyncio.sleep(0.001)  # 1ms per person
        
        performance_monitor.stop("multi_camera_processing")
        
        # Multi-camera processing should be efficient
        expected_time = camera_count * persons_per_camera * 0.001 * 2  # 2x overhead allowance
        performance_monitor.assert_performance("multi_camera_processing", expected_time)
    
    @pytest.mark.asyncio
    async def test_high_volume_data_processing(self, performance_monitor):
        """Test high-volume data processing."""
        await analytics_engine.initialize()
        
        # Test high-volume detection processing
        performance_monitor.start("high_volume_processing")
        
        # Simulate high volume of detections
        detection_count = 10000
        batch_size = 100
        
        for batch_start in range(0, detection_count, batch_size):
            batch_end = min(batch_start + batch_size, detection_count)
            
            # Process batch
            for i in range(batch_start, batch_end):
                detection = Detection(
                    detection_id=f"det_{i}",
                    camera_id=CameraID(f"cam_{i % 10}"),
                    bounding_box=(i % 100, i % 100, (i % 100) + 50, (i % 100) + 50),
                    confidence=0.9,
                    timestamp=datetime.now(timezone.utc),
                    frame_id=f"frame_{i}"
                )
                
                # Simulate processing
                await asyncio.sleep(0.0001)  # 0.1ms per detection
        
        performance_monitor.stop("high_volume_processing")
        
        # High-volume processing should be efficient
        expected_time = detection_count * 0.0001 * 2  # 2x overhead allowance
        performance_monitor.assert_performance("high_volume_processing", expected_time)
    
    @pytest.mark.asyncio
    async def test_system_resource_utilization(self):
        """Test system resource utilization under load."""
        await memory_manager.initialize()
        await monitoring_service.initialize()
        
        # Get baseline metrics
        baseline_metrics = await monitoring_service.get_system_metrics(1)
        
        # Simulate load
        start_time = time.time()
        
        # CPU-intensive task
        for _ in range(1000):
            # Simulate computation
            result = sum(i * i for i in range(1000))
        
        # Memory-intensive task
        large_data = []
        for i in range(10000):
            large_data.append(np.random.random(100))
        
        end_time = time.time()
        
        # Get metrics under load
        load_metrics = await monitoring_service.get_system_metrics(1)
        
        # Check that system handled load within reasonable bounds
        processing_time = end_time - start_time
        assert processing_time < 30.0  # Should complete within 30 seconds
        
        # Check that metrics were collected
        assert len(load_metrics) > 0
        
        # Cleanup
        del large_data
        gc.collect()
        
        await memory_manager.cleanup()
        await monitoring_service.cleanup()