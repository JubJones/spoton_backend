"""
Test fixtures for SpotOn backend tests.

Provides reusable test fixtures for:
- Database connections
- Cache services
- Authentication services
- Performance monitoring
- GPU testing environment
"""

import pytest
import asyncio
import torch
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock
import time
import psutil
import gc

from app.infrastructure.cache.tracking_cache import tracking_cache
from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.infrastructure.security.jwt_service import jwt_service
from app.infrastructure.security.encryption_service import encryption_service
from app.services.memory_manager import memory_manager
from app.services.monitoring_service import monitoring_service
from app.services.analytics_engine import analytics_engine


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_service():
    """Database service fixture."""
    await integrated_db_service.initialize()
    yield integrated_db_service
    # Cleanup handled by service


@pytest.fixture
async def cache_service():
    """Cache service fixture."""
    await tracking_cache.initialize()
    yield tracking_cache
    # Cleanup handled by service


@pytest.fixture
async def jwt_auth_service():
    """JWT authentication service fixture."""
    await jwt_service.initialize()
    yield jwt_service
    # Cleanup handled by service


@pytest.fixture
async def encryption_service_fixture():
    """Encryption service fixture."""
    await encryption_service.initialize()
    yield encryption_service
    # Cleanup handled by service


@pytest.fixture
async def memory_service():
    """Memory manager service fixture."""
    await memory_manager.initialize()
    yield memory_manager
    await memory_manager.cleanup()


@pytest.fixture
async def monitoring_service_fixture():
    """Monitoring service fixture."""
    await monitoring_service.initialize()
    yield monitoring_service
    await monitoring_service.cleanup()


@pytest.fixture
async def analytics_service():
    """Analytics engine service fixture."""
    await analytics_engine.initialize()
    yield analytics_engine
    # Cleanup handled by service


@pytest.fixture
def gpu_test_environment():
    """GPU test environment fixture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    
    # Create test data
    test_data = {
        "small_tensor": torch.randn(100, 512, device=device),
        "medium_tensor": torch.randn(1000, 512, device=device),
        "large_tensor": torch.randn(10000, 512, device=device)
    }
    
    yield {
        "device": device,
        "test_data": test_data
    }
    
    # Cleanup
    for tensor in test_data.values():
        del tensor
    torch.cuda.empty_cache()


@pytest.fixture
def memory_profiler():
    """Memory profiler fixture."""
    class MemoryProfiler:
        def __init__(self):
            self.start_memory = None
            self.end_memory = None
            self.peak_memory = None
            self.is_running = False
        
        def start(self):
            """Start memory profiling."""
            self.start_memory = psutil.Process().memory_info().rss
            self.is_running = True
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        
        def stop(self):
            """Stop memory profiling."""
            self.end_memory = psutil.Process().memory_info().rss
            self.is_running = False
            if torch.cuda.is_available():
                self.peak_memory = torch.cuda.max_memory_allocated()
        
        def get_memory_usage(self):
            """Get memory usage in MB."""
            if self.start_memory and self.end_memory:
                return (self.end_memory - self.start_memory) / (1024 * 1024)
            return 0
        
        def get_peak_gpu_memory(self):
            """Get peak GPU memory usage in MB."""
            if self.peak_memory:
                return self.peak_memory / (1024 * 1024)
            return 0
        
        def assert_memory_usage(self, max_mb: float):
            """Assert memory usage is within limits."""
            usage = self.get_memory_usage()
            assert usage < max_mb, f"Memory usage {usage:.2f}MB exceeds limit {max_mb}MB"
    
    return MemoryProfiler()


@pytest.fixture
def performance_monitor():
    """Performance monitor fixture."""
    class PerformanceMonitor:
        def __init__(self):
            self.timings = {}
            self.start_times = {}
        
        def start(self, operation_name: str):
            """Start timing an operation."""
            self.start_times[operation_name] = time.time()
        
        def stop(self, operation_name: str):
            """Stop timing an operation."""
            if operation_name in self.start_times:
                elapsed = time.time() - self.start_times[operation_name]
                self.timings[operation_name] = elapsed
                del self.start_times[operation_name]
        
        def get_timing(self, operation_name: str) -> float:
            """Get timing for an operation."""
            return self.timings.get(operation_name, 0.0)
        
        def assert_performance(self, operation_name: str, max_seconds: float):
            """Assert operation completed within time limit."""
            timing = self.get_timing(operation_name)
            assert timing <= max_seconds, f"Operation {operation_name} took {timing:.3f}s, exceeds limit {max_seconds}s"
        
        def get_all_timings(self) -> Dict[str, float]:
            """Get all recorded timings."""
            return self.timings.copy()
    
    return PerformanceMonitor()


@pytest.fixture
def async_test_helper():
    """Async test helper fixture."""
    class AsyncTestHelper:
        @staticmethod
        async def run_parallel(tasks: List):
            """Run tasks in parallel."""
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        @staticmethod
        async def run_sequential(tasks: List):
            """Run tasks sequentially."""
            results = []
            for task in tasks:
                result = await task
                results.append(result)
            return results
        
        @staticmethod
        async def simulate_load(duration: float, operations: int) -> float:
            """Simulate load for specified duration."""
            start_time = time.time()
            end_time = start_time + duration
            
            operation_count = 0
            while time.time() < end_time and operation_count < operations:
                # Simulate work
                await asyncio.sleep(0.001)
                operation_count += 1
            
            return time.time() - start_time
        
        @staticmethod
        async def wait_for_condition(condition_func, timeout: float = 10.0, interval: float = 0.1):
            """Wait for a condition to be true."""
            start_time = time.time()
            while time.time() - start_time < timeout:
                if await condition_func():
                    return True
                await asyncio.sleep(interval)
            return False
    
    return AsyncTestHelper()


@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    class TestDataGenerator:
        @staticmethod
        def generate_detection_data(count: int = 10):
            """Generate test detection data."""
            detections = []
            for i in range(count):
                detection = {
                    "detection_id": f"detection_{i:03d}",
                    "camera_id": f"camera_{(i % 5) + 1:02d}",
                    "bounding_box": (i * 10, i * 10, i * 10 + 100, i * 10 + 100),
                    "confidence": 0.9,
                    "timestamp": datetime.now(timezone.utc),
                    "frame_id": f"frame_{i:03d}"
                }
                detections.append(detection)
            return detections
        
        @staticmethod
        def generate_person_identities(count: int = 10):
            """Generate test person identities."""
            identities = []
            for i in range(count):
                identity = {
                    "global_id": f"person_{i:03d}",
                    "feature_vector": np.random.random(512).astype(np.float32),
                    "confidence": 0.9,
                    "first_seen": datetime.now(timezone.utc),
                    "last_seen": datetime.now(timezone.utc),
                    "cameras_seen": [f"camera_{(i % 5) + 1:02d}"]
                }
                identities.append(identity)
            return identities
        
        @staticmethod
        def generate_tracking_events(count: int = 100):
            """Generate test tracking events."""
            events = []
            for i in range(count):
                event = {
                    "global_person_id": f"person_{i % 20:03d}",
                    "camera_id": f"camera_{(i % 5) + 1:02d}",
                    "environment_id": "test_env",
                    "event_type": "detection",
                    "position": {"x": float(i % 100), "y": float(i % 100)},
                    "confidence": 0.9,
                    "timestamp": datetime.now(timezone.utc)
                }
                events.append(event)
            return events
    
    return TestDataGenerator()


@pytest.fixture
def mock_services():
    """Mock services fixture."""
    class MockServices:
        def __init__(self):
            self.mock_cache = MagicMock()
            self.mock_db = MagicMock()
            self.mock_analytics = MagicMock()
            self.mock_monitoring = MagicMock()
            
            # Setup async methods
            self.mock_cache.get_person_identity = AsyncMock()
            self.mock_cache.cache_person_identity = AsyncMock()
            self.mock_db.store_tracking_event = AsyncMock()
            self.mock_db.get_detection_statistics = AsyncMock()
            self.mock_analytics.update_real_time_metrics = AsyncMock()
            self.mock_analytics.get_real_time_metrics = AsyncMock()
            self.mock_monitoring.get_system_metrics = AsyncMock()
    
    return MockServices()


@pytest.fixture
def cleanup_handler():
    """Cleanup handler fixture."""
    cleanup_tasks = []
    
    def register_cleanup(cleanup_func):
        """Register a cleanup function."""
        cleanup_tasks.append(cleanup_func)
    
    yield register_cleanup
    
    # Execute cleanup tasks
    for cleanup_func in cleanup_tasks:
        try:
            if asyncio.iscoroutinefunction(cleanup_func):
                asyncio.run(cleanup_func())
            else:
                cleanup_func()
        except Exception as e:
            print(f"Warning: Cleanup task failed: {e}")


@pytest.fixture
def test_environment():
    """Test environment fixture."""
    class TestEnvironment:
        def __init__(self):
            self.environment_id = "test_env"
            self.cameras = ["camera_01", "camera_02", "camera_03", "camera_04"]
            self.test_persons = []
            self.test_events = []
        
        def add_test_person(self, person_data):
            """Add a test person."""
            self.test_persons.append(person_data)
        
        def add_test_event(self, event_data):
            """Add a test event."""
            self.test_events.append(event_data)
        
        def get_camera_count(self):
            """Get number of cameras."""
            return len(self.cameras)
        
        def get_person_count(self):
            """Get number of test persons."""
            return len(self.test_persons)
        
        def get_event_count(self):
            """Get number of test events."""
            return len(self.test_events)
    
    return TestEnvironment()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "test_environment": "test_env",
        "test_timeout": 30.0,
        "performance_thresholds": {
            "api_response_time": 1.0,
            "database_query_time": 0.5,
            "cache_operation_time": 0.1,
            "gpu_operation_time": 0.5
        },
        "memory_limits": {
            "max_memory_usage_mb": 1024,
            "max_gpu_memory_mb": 2048,
            "max_cache_size_mb": 512
        },
        "load_test_config": {
            "concurrent_users": 50,
            "operations_per_user": 100,
            "test_duration": 60.0
        }
    }


@pytest.fixture
def error_injector():
    """Error injector fixture for testing error handling."""
    class ErrorInjector:
        def __init__(self):
            self.error_rate = 0.0
            self.error_type = Exception
            self.error_message = "Injected test error"
        
        def set_error_rate(self, rate: float):
            """Set error injection rate (0.0 to 1.0)."""
            self.error_rate = max(0.0, min(1.0, rate))
        
        def set_error_type(self, error_type: type, message: str = "Injected test error"):
            """Set error type and message."""
            self.error_type = error_type
            self.error_message = message
        
        def maybe_inject_error(self):
            """Maybe inject an error based on error rate."""
            if np.random.random() < self.error_rate:
                raise self.error_type(self.error_message)
    
    return ErrorInjector()


@pytest.fixture(autouse=True)
def cleanup_resources():
    """Auto-cleanup fixture that runs after each test."""
    yield
    
    # Cleanup Python garbage
    gc.collect()
    
    # Cleanup GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()