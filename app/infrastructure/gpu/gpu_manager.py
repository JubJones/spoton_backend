"""
GPU Resource Manager for device allocation and monitoring.
Manages GPU allocation, memory monitoring, and device optimization.
"""
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import time

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Device types for model deployment."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class GPUInfo:
    """GPU device information."""
    device_id: int
    name: str
    total_memory: int  # in bytes
    available_memory: int  # in bytes
    utilization: float  # 0.0 to 1.0
    temperature: Optional[float] = None
    power_usage: Optional[float] = None


@dataclass
class DeviceAllocation:
    """Device allocation information."""
    device_id: str
    device_type: DeviceType
    allocated_memory: int
    max_memory: int
    allocation_time: float
    task_id: Optional[str] = None


class GPUManager:
    """
    Manages GPU resources for the SpotOn backend.
    Handles device allocation, memory monitoring, and optimization.
    """
    
    def __init__(self):
        self.device_allocations: Dict[str, DeviceAllocation] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 10.0  # seconds
        self.memory_threshold = 0.8  # 80% memory usage warning
        self._lock = threading.Lock()
        
        # Initialize device information
        self.available_devices: List[str] = []
        self.device_info: Dict[str, GPUInfo] = {}
        self.preferred_device: Optional[str] = None
        
        # Initialize monitoring
        self._initialize_devices()
        
        logger.info("GPUManager initialized")
    
    def _initialize_devices(self):
        """Initialize available devices and gather information."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. GPU management will be limited.")
            return
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} devices")
            
            for i in range(device_count):
                device_id = f"cuda:{i}"
                self.available_devices.append(device_id)
                
                # Get device properties
                props = torch.cuda.get_device_properties(i)
                gpu_info = GPUInfo(
                    device_id=i,
                    name=props.name,
                    total_memory=props.total_memory,
                    available_memory=props.total_memory,
                    utilization=0.0
                )
                self.device_info[device_id] = gpu_info
                
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            
            # Set preferred device (usually first GPU)
            if self.available_devices:
                self.preferred_device = self.available_devices[0]
        
        # Check MPS availability (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_id = "mps"
            self.available_devices.append(device_id)
            self.preferred_device = device_id
            logger.info("MPS (Apple Silicon) GPU available")
        
        # Fallback to CPU
        if not self.available_devices:
            self.available_devices.append("cpu")
            self.preferred_device = "cpu"
            logger.info("No GPU available, using CPU")
    
    def get_optimal_device(self, task_id: Optional[str] = None) -> str:
        """
        Get the optimal device for a task.
        
        Args:
            task_id: Optional task identifier for tracking
            
        Returns:
            Device string (e.g., "cuda:0", "cpu")
        """
        if not self.available_devices:
            return "cpu"
        
        # If only CPU available
        if self.available_devices == ["cpu"]:
            return "cpu"
        
        # For GPU devices, select based on memory availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            best_device = None
            max_available_memory = 0
            
            for device_id in self.available_devices:
                if device_id.startswith("cuda"):
                    gpu_id = int(device_id.split(":")[1])
                    try:
                        torch.cuda.set_device(gpu_id)
                        available_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                        allocated_memory = torch.cuda.memory_allocated(gpu_id)
                        free_memory = available_memory - allocated_memory
                        
                        if free_memory > max_available_memory:
                            max_available_memory = free_memory
                            best_device = device_id
                    except Exception as e:
                        logger.warning(f"Error checking memory for {device_id}: {e}")
            
            if best_device:
                logger.info(f"Selected device {best_device} for task {task_id}")
                return best_device
        
        # Return preferred device as fallback
        return self.preferred_device or "cpu"
    
    def allocate_device(self, task_id: str, requested_memory: Optional[int] = None) -> str:
        """
        Allocate a device for a specific task.
        
        Args:
            task_id: Task identifier
            requested_memory: Requested memory in bytes
            
        Returns:
            Allocated device string
        """
        with self._lock:
            device = self.get_optimal_device(task_id)
            
            # Create allocation record
            allocation = DeviceAllocation(
                device_id=device,
                device_type=DeviceType.CUDA if device.startswith("cuda") else 
                           DeviceType.MPS if device == "mps" else DeviceType.CPU,
                allocated_memory=requested_memory or 0,
                max_memory=self._get_device_memory(device),
                allocation_time=time.time(),
                task_id=task_id
            )
            
            self.device_allocations[task_id] = allocation
            logger.info(f"Allocated device {device} for task {task_id}")
            return device
    
    def deallocate_device(self, task_id: str):
        """
        Deallocate device for a task.
        
        Args:
            task_id: Task identifier
        """
        with self._lock:
            if task_id in self.device_allocations:
                allocation = self.device_allocations.pop(task_id)
                
                # Clear GPU cache if it was a GPU device
                if allocation.device_type == DeviceType.CUDA and TORCH_AVAILABLE:
                    try:
                        torch.cuda.empty_cache()
                        logger.info(f"Cleared CUDA cache for task {task_id}")
                    except Exception as e:
                        logger.warning(f"Error clearing CUDA cache: {e}")
                
                logger.info(f"Deallocated device {allocation.device_id} for task {task_id}")
    
    def _get_device_memory(self, device: str) -> int:
        """Get total memory for a device."""
        if device.startswith("cuda") and TORCH_AVAILABLE:
            gpu_id = int(device.split(":")[1])
            return torch.cuda.get_device_properties(gpu_id).total_memory
        return 0
    
    def get_device_stats(self) -> Dict[str, Any]:
        """Get current device statistics."""
        stats = {
            "available_devices": self.available_devices,
            "preferred_device": self.preferred_device,
            "active_allocations": len(self.device_allocations),
            "device_info": {}
        }
        
        # Update GPU information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for device_id in self.available_devices:
                if device_id.startswith("cuda"):
                    gpu_id = int(device_id.split(":")[1])
                    try:
                        props = torch.cuda.get_device_properties(gpu_id)
                        allocated = torch.cuda.memory_allocated(gpu_id)
                        reserved = torch.cuda.memory_reserved(gpu_id)
                        
                        stats["device_info"][device_id] = {
                            "name": props.name,
                            "total_memory": props.total_memory,
                            "allocated_memory": allocated,
                            "reserved_memory": reserved,
                            "free_memory": props.total_memory - reserved,
                            "utilization": allocated / props.total_memory
                        }
                    except Exception as e:
                        logger.warning(f"Error getting stats for {device_id}: {e}")
        
        return stats
    
    def start_monitoring(self):
        """Start GPU monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("GPU monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_device_stats()
                self._check_memory_warnings()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_device_stats(self):
        """Update device statistics."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        for device_id in self.available_devices:
            if device_id.startswith("cuda"):
                gpu_id = int(device_id.split(":")[1])
                try:
                    if device_id in self.device_info:
                        gpu_info = self.device_info[device_id]
                        allocated = torch.cuda.memory_allocated(gpu_id)
                        total = torch.cuda.get_device_properties(gpu_id).total_memory
                        
                        gpu_info.available_memory = total - allocated
                        gpu_info.utilization = allocated / total
                except Exception as e:
                    logger.warning(f"Error updating stats for {device_id}: {e}")
    
    def _check_memory_warnings(self):
        """Check for memory usage warnings."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        for device_id in self.available_devices:
            if device_id.startswith("cuda") and device_id in self.device_info:
                gpu_info = self.device_info[device_id]
                if gpu_info.utilization > self.memory_threshold:
                    logger.warning(
                        f"High memory usage on {device_id}: "
                        f"{gpu_info.utilization:.1%} ({gpu_info.name})"
                    )
    
    def optimize_memory(self, device: str):
        """
        Optimize memory usage for a device.
        
        Args:
            device: Device string to optimize
        """
        if device.startswith("cuda") and TORCH_AVAILABLE:
            try:
                gpu_id = int(device.split(":")[1])
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"Optimized memory for {device}")
            except Exception as e:
                logger.error(f"Error optimizing memory for {device}: {e}")
    
    def get_allocation_info(self, task_id: str) -> Optional[DeviceAllocation]:
        """Get allocation information for a task."""
        return self.device_allocations.get(task_id)
    
    def is_available(self) -> bool:
        """Check if GPU is available."""
        if not TORCH_AVAILABLE:
            return False
        return torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    
    def get_device_count(self) -> int:
        """Get number of available devices."""
        if not TORCH_AVAILABLE:
            return 0
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 1
        return 0
    
    def get_utilization(self) -> float:
        """Get average GPU utilization across all devices."""
        if not TORCH_AVAILABLE or not self.is_available():
            return 0.0
        
        total_utilization = 0.0
        device_count = 0
        
        if torch.cuda.is_available():
            for device_id in self.available_devices:
                if device_id.startswith("cuda"):
                    gpu_id = int(device_id.split(":")[1])
                    try:
                        allocated = torch.cuda.memory_allocated(gpu_id)
                        total = torch.cuda.get_device_properties(gpu_id).total_memory
                        utilization = allocated / total if total > 0 else 0.0
                        total_utilization += utilization
                        device_count += 1
                    except Exception as e:
                        logger.warning(f"Error getting utilization for {device_id}: {e}")
        
        return total_utilization / device_count if device_count > 0 else 0.0
    
    def get_memory_usage(self) -> float:
        """Get average memory usage across all devices in GB."""
        if not TORCH_AVAILABLE or not self.is_available():
            return 0.0
        
        total_memory_used = 0.0
        device_count = 0
        
        if torch.cuda.is_available():
            for device_id in self.available_devices:
                if device_id.startswith("cuda"):
                    gpu_id = int(device_id.split(":")[1])
                    try:
                        allocated = torch.cuda.memory_allocated(gpu_id)
                        total_memory_used += allocated / (1024**3)  # Convert to GB
                        device_count += 1
                    except Exception as e:
                        logger.warning(f"Error getting memory usage for {device_id}: {e}")
        
        return total_memory_used / device_count if device_count > 0 else 0.0
    
    def get_temperature(self) -> float:
        """Get average GPU temperature (not available via PyTorch, returns 0)."""
        # PyTorch doesn't provide temperature monitoring
        # This would require nvidia-ml-py or similar for NVIDIA GPUs
        return 0.0
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()
        
        # Clear all allocations
        with self._lock:
            for task_id in list(self.device_allocations.keys()):
                self.deallocate_device(task_id)
        
        # Clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache on cleanup")
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache on cleanup: {e}")
        
        logger.info("GPUManager cleanup completed")


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def initialize_gpu_manager() -> GPUManager:
    """Initialize the global GPU manager."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
        _gpu_manager.start_monitoring()
    return _gpu_manager


def cleanup_gpu_manager():
    """Cleanup the global GPU manager."""
    global _gpu_manager
    if _gpu_manager is not None:
        _gpu_manager.cleanup()
        _gpu_manager = None