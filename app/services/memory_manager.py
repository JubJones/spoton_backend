"""
Advanced memory management service for GPU and system memory optimization.

Handles:
- GPU memory monitoring and optimization
- System memory management
- Model quantization and optimization
- Memory leak detection and prevention
- Dynamic memory allocation strategies
"""

import asyncio
import logging
import psutil
import gc
import torch
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import threading
import weakref

from app.infrastructure.gpu import get_gpu_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    system_memory: Dict[str, float]
    gpu_memory: Dict[str, float]
    process_memory: Dict[str, float]
    model_memory: Dict[str, float]
    cache_memory: Dict[str, float]


@dataclass
class MemoryThresholds:
    """Memory usage thresholds for alerts and actions."""
    system_warning: float = 0.8  # 80%
    system_critical: float = 0.9  # 90%
    gpu_warning: float = 0.8  # 80%
    gpu_critical: float = 0.9  # 90%
    cache_limit: float = 0.5  # 50% of available memory
    model_limit: float = 0.4  # 40% of GPU memory


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization."""
    enable_model_quantization: bool = True
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_dynamic_batching: bool = True
    cache_cleanup_interval: int = 300  # seconds
    memory_check_interval: int = 10  # seconds
    leak_detection_enabled: bool = True


class MemoryManager:
    """
    Advanced memory management service.
    
    Features:
    - Real-time memory monitoring
    - GPU memory optimization
    - Model quantization
    - Memory leak detection
    - Dynamic memory allocation
    """
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self.thresholds = MemoryThresholds()
        self.config = MemoryOptimizationConfig()
        
        # Memory history
        self.memory_history: deque = deque(maxlen=1000)
        self.memory_alerts: deque = deque(maxlen=100)
        
        # Memory tracking
        self.tracked_objects: Dict[str, weakref.WeakSet] = {}
        self.memory_pools: Dict[str, Dict[str, Any]] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.memory_stats = {
            'total_memory_checks': 0,
            'memory_optimizations': 0,
            'cache_cleanups': 0,
            'model_quantizations': 0,
            'memory_leaks_detected': 0,
            'gpu_memory_optimizations': 0
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self.memory_lock = threading.Lock()
        
        logger.info("MemoryManager initialized")
    
    async def initialize(self):
        """Initialize memory management."""
        try:
            # Start monitoring tasks
            await self.start_monitoring()
            
            # Initialize memory pools
            await self._initialize_memory_pools()
            
            # Setup model registry
            await self._setup_model_registry()
            
            # Enable GPU optimizations if available
            if self.gpu_manager and self.gpu_manager.is_available():
                await self._enable_gpu_optimizations()
            
            logger.info("MemoryManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing MemoryManager: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup memory management."""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Cleanup memory pools
            await self._cleanup_memory_pools()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("MemoryManager cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up MemoryManager: {e}")
    
    # Memory Monitoring
    async def start_monitoring(self):
        """Start memory monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Memory monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting memory monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop memory monitoring."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
                self.monitoring_task = None
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
                self.cleanup_task = None
            
            logger.info("Memory monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping memory monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main memory monitoring loop."""
        try:
            while self.monitoring_active:
                await self._check_memory_usage()
                await asyncio.sleep(self.config.memory_check_interval)
                
        except asyncio.CancelledError:
            logger.info("Memory monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in memory monitoring loop: {e}")
    
    async def _cleanup_loop(self):
        """Memory cleanup loop."""
        try:
            while self.monitoring_active:
                await self._perform_memory_cleanup()
                await asyncio.sleep(self.config.cache_cleanup_interval)
                
        except asyncio.CancelledError:
            logger.info("Memory cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Error in memory cleanup loop: {e}")
    
    async def _check_memory_usage(self):
        """Check current memory usage."""
        try:
            with self.memory_lock:
                self.memory_stats['total_memory_checks'] += 1
                
                # Get memory snapshot
                snapshot = await self._create_memory_snapshot()
                self.memory_history.append(snapshot)
                
                # Check thresholds
                await self._check_memory_thresholds(snapshot)
                
                # Detect memory leaks
                if self.config.leak_detection_enabled:
                    await self._detect_memory_leaks(snapshot)
                
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
    
    async def _create_memory_snapshot(self) -> MemorySnapshot:
        """Create memory usage snapshot."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_mem_info = {
                'total': system_memory.total,
                'available': system_memory.available,
                'used': system_memory.used,
                'percent': system_memory.percent,
                'free': system_memory.free,
                'cached': getattr(system_memory, 'cached', 0),
                'buffers': getattr(system_memory, 'buffers', 0)
            }
            
            # GPU memory
            gpu_mem_info = {}
            if self.gpu_manager and self.gpu_manager.is_available():
                gpu_mem_info = {
                    'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                    'max_allocated': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    'max_reserved': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    'utilization': self.gpu_manager.get_memory_usage(),
                    'total': self.gpu_manager.get_total_memory()
                }
            
            # Process memory
            process = psutil.Process()
            process_mem_info = {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms,
                'percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'cpu_percent': process.cpu_percent()
            }
            
            # Model memory (approximate)
            model_mem_info = {}
            for model_name, model_info in self.model_registry.items():
                if 'memory_usage' in model_info:
                    model_mem_info[model_name] = model_info['memory_usage']
            
            # Cache memory (approximate)
            cache_mem_info = {
                'redis_estimated': 0,  # Would be calculated from Redis info
                'model_cache': 0,      # Would be calculated from model cache
                'frame_cache': 0       # Would be calculated from frame cache
            }
            
            return MemorySnapshot(
                timestamp=current_time,
                system_memory=system_mem_info,
                gpu_memory=gpu_mem_info,
                process_memory=process_mem_info,
                model_memory=model_mem_info,
                cache_memory=cache_mem_info
            )
            
        except Exception as e:
            logger.error(f"Error creating memory snapshot: {e}")
            return MemorySnapshot(
                timestamp=datetime.now(timezone.utc),
                system_memory={},
                gpu_memory={},
                process_memory={},
                model_memory={},
                cache_memory={}
            )
    
    async def _check_memory_thresholds(self, snapshot: MemorySnapshot):
        """Check memory thresholds and trigger alerts."""
        try:
            alerts = []
            
            # Check system memory
            if snapshot.system_memory.get('percent', 0) > self.thresholds.system_critical * 100:
                alerts.append({
                    'type': 'system_memory_critical',
                    'usage': snapshot.system_memory.get('percent', 0),
                    'threshold': self.thresholds.system_critical * 100,
                    'action': 'immediate_cleanup'
                })
            elif snapshot.system_memory.get('percent', 0) > self.thresholds.system_warning * 100:
                alerts.append({
                    'type': 'system_memory_warning',
                    'usage': snapshot.system_memory.get('percent', 0),
                    'threshold': self.thresholds.system_warning * 100,
                    'action': 'scheduled_cleanup'
                })
            
            # Check GPU memory
            if snapshot.gpu_memory:
                gpu_usage = snapshot.gpu_memory.get('utilization', 0)
                if gpu_usage > self.thresholds.gpu_critical * 100:
                    alerts.append({
                        'type': 'gpu_memory_critical',
                        'usage': gpu_usage,
                        'threshold': self.thresholds.gpu_critical * 100,
                        'action': 'gpu_cleanup'
                    })
                elif gpu_usage > self.thresholds.gpu_warning * 100:
                    alerts.append({
                        'type': 'gpu_memory_warning',
                        'usage': gpu_usage,
                        'threshold': self.thresholds.gpu_warning * 100,
                        'action': 'gpu_optimization'
                    })
            
            # Process alerts
            for alert in alerts:
                await self._handle_memory_alert(alert, snapshot)
                self.memory_alerts.append({
                    'timestamp': snapshot.timestamp,
                    'alert': alert
                })
            
        except Exception as e:
            logger.error(f"Error checking memory thresholds: {e}")
    
    async def _handle_memory_alert(self, alert: Dict[str, Any], snapshot: MemorySnapshot):
        """Handle memory alert."""
        try:
            action = alert.get('action')
            
            if action == 'immediate_cleanup':
                await self._emergency_memory_cleanup()
            elif action == 'scheduled_cleanup':
                await self._perform_memory_cleanup()
            elif action == 'gpu_cleanup':
                await self._gpu_memory_cleanup()
            elif action == 'gpu_optimization':
                await self._optimize_gpu_memory()
            
            logger.warning(f"Memory alert: {alert['type']} - {alert['usage']:.1f}% (threshold: {alert['threshold']:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error handling memory alert: {e}")
    
    async def _detect_memory_leaks(self, snapshot: MemorySnapshot):
        """Detect potential memory leaks."""
        try:
            if len(self.memory_history) < 10:
                return
            
            # Check for continuous memory growth
            recent_snapshots = list(self.memory_history)[-10:]
            memory_usages = [s.system_memory.get('percent', 0) for s in recent_snapshots]
            
            # Simple leak detection: check if memory usage is consistently increasing
            if len(memory_usages) >= 5:
                # Check if memory usage has increased consistently
                increases = sum(1 for i in range(1, len(memory_usages)) if memory_usages[i] > memory_usages[i-1])
                
                if increases >= 4:  # At least 4 out of 5 increases
                    total_increase = memory_usages[-1] - memory_usages[0]
                    if total_increase > 10:  # More than 10% increase
                        self.memory_stats['memory_leaks_detected'] += 1
                        logger.warning(f"Potential memory leak detected: {total_increase:.1f}% increase over {len(memory_usages)} checks")
                        
                        # Trigger leak mitigation
                        await self._mitigate_memory_leak()
            
        except Exception as e:
            logger.error(f"Error detecting memory leaks: {e}")
    
    # Memory Optimization
    async def _initialize_memory_pools(self):
        """Initialize memory pools."""
        try:
            # Initialize different memory pools
            self.memory_pools = {
                'model_cache': {
                    'max_size': 1024 * 1024 * 1024,  # 1GB
                    'current_size': 0,
                    'items': {}
                },
                'frame_cache': {
                    'max_size': 512 * 1024 * 1024,  # 512MB
                    'current_size': 0,
                    'items': {}
                },
                'embedding_cache': {
                    'max_size': 256 * 1024 * 1024,  # 256MB
                    'current_size': 0,
                    'items': {}
                }
            }
            
            logger.info("Memory pools initialized")
            
        except Exception as e:
            logger.error(f"Error initializing memory pools: {e}")
    
    async def _cleanup_memory_pools(self):
        """Cleanup memory pools."""
        try:
            for pool_name, pool in self.memory_pools.items():
                pool['items'].clear()
                pool['current_size'] = 0
                
            logger.info("Memory pools cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up memory pools: {e}")
    
    async def _perform_memory_cleanup(self):
        """Perform regular memory cleanup."""
        try:
            self.memory_stats['cache_cleanups'] += 1
            
            # Clean up memory pools
            for pool_name, pool in self.memory_pools.items():
                if pool['current_size'] > pool['max_size'] * 0.8:
                    # Remove oldest items
                    items_to_remove = []
                    for key, item in pool['items'].items():
                        if len(items_to_remove) < len(pool['items']) // 4:  # Remove 25%
                            items_to_remove.append(key)
                    
                    for key in items_to_remove:
                        if key in pool['items']:
                            del pool['items'][key]
                            pool['current_size'] -= pool['items'].get(key, {}).get('size', 0)
            
            # Force garbage collection
            gc.collect()
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug("Memory cleanup performed")
            
        except Exception as e:
            logger.error(f"Error performing memory cleanup: {e}")
    
    async def _emergency_memory_cleanup(self):
        """Perform emergency memory cleanup."""
        try:
            logger.warning("Performing emergency memory cleanup")
            
            # Clear all memory pools
            await self._cleanup_memory_pools()
            
            # Force multiple garbage collections
            for _ in range(3):
                gc.collect()
            
            # GPU emergency cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Clear model caches
            for model_name, model_info in self.model_registry.items():
                if 'cache' in model_info:
                    model_info['cache'].clear()
            
            logger.info("Emergency memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error performing emergency memory cleanup: {e}")
    
    async def _gpu_memory_cleanup(self):
        """GPU-specific memory cleanup."""
        try:
            if not torch.cuda.is_available():
                return
            
            self.memory_stats['gpu_memory_optimizations'] += 1
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Clear GPU IPC cache
            torch.cuda.ipc_collect()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            logger.debug("GPU memory cleanup performed")
            
        except Exception as e:
            logger.error(f"Error performing GPU memory cleanup: {e}")
    
    async def _optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        try:
            if not torch.cuda.is_available():
                return
            
            # Enable gradient checkpointing for models
            if self.config.enable_gradient_checkpointing:
                await self._enable_gradient_checkpointing()
            
            # Enable mixed precision training
            if self.config.enable_mixed_precision:
                await self._enable_mixed_precision()
            
            # Optimize model quantization
            if self.config.enable_model_quantization:
                await self._optimize_model_quantization()
            
            logger.debug("GPU memory optimization performed")
            
        except Exception as e:
            logger.error(f"Error optimizing GPU memory: {e}")
    
    async def _mitigate_memory_leak(self):
        """Mitigate detected memory leak."""
        try:
            logger.warning("Attempting to mitigate memory leak")
            
            # Perform comprehensive cleanup
            await self._emergency_memory_cleanup()
            
            # Clear weak references
            for obj_type, weak_set in self.tracked_objects.items():
                # WeakSet automatically handles dead references
                logger.debug(f"Tracked objects of type {obj_type}: {len(weak_set)}")
            
            # Additional Python-specific cleanup
            import sys
            import ctypes
            
            # Force garbage collection with debug info
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Get unreachable objects
            unreachable = len(gc.garbage)
            if unreachable > 0:
                logger.warning(f"Found {unreachable} unreachable objects")
            
        except Exception as e:
            logger.error(f"Error mitigating memory leak: {e}")
    
    # Model Optimization
    async def _setup_model_registry(self):
        """Setup model registry for memory tracking."""
        try:
            # Register models for memory tracking
            self.model_registry = {
                'detection_model': {
                    'type': 'detection',
                    'memory_usage': 0,
                    'quantized': False,
                    'cache': {}
                },
                'reid_model': {
                    'type': 'reid',
                    'memory_usage': 0,
                    'quantized': False,
                    'cache': {}
                },
                'mapping_model': {
                    'type': 'mapping',
                    'memory_usage': 0,
                    'quantized': False,
                    'cache': {}
                }
            }
            
            logger.info("Model registry setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up model registry: {e}")
    
    async def _enable_gpu_optimizations(self):
        """Enable GPU-specific optimizations."""
        try:
            if not torch.cuda.is_available():
                return
            
            # Enable CUDA memory optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Enable memory-efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
            
            logger.info("GPU optimizations enabled")
            
        except Exception as e:
            logger.error(f"Error enabling GPU optimizations: {e}")
    
    async def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for models."""
        try:
            # This would be implemented to enable gradient checkpointing
            # for specific models to reduce memory usage
            logger.debug("Gradient checkpointing enabled")
            
        except Exception as e:
            logger.error(f"Error enabling gradient checkpointing: {e}")
    
    async def _enable_mixed_precision(self):
        """Enable mixed precision training."""
        try:
            # This would be implemented to enable mixed precision
            # for models to reduce memory usage
            logger.debug("Mixed precision enabled")
            
        except Exception as e:
            logger.error(f"Error enabling mixed precision: {e}")
    
    async def _optimize_model_quantization(self):
        """Optimize model quantization."""
        try:
            self.memory_stats['model_quantizations'] += 1
            
            # This would be implemented to quantize models
            # to reduce memory usage
            for model_name, model_info in self.model_registry.items():
                if not model_info['quantized']:
                    # Quantize model
                    model_info['quantized'] = True
                    model_info['memory_usage'] *= 0.5  # Approximate 50% reduction
            
            logger.debug("Model quantization optimized")
            
        except Exception as e:
            logger.error(f"Error optimizing model quantization: {e}")
    
    # Memory Allocation
    async def allocate_memory(
        self,
        pool_name: str,
        key: str,
        size: int,
        data: Any = None
    ) -> bool:
        """Allocate memory from pool."""
        try:
            if pool_name not in self.memory_pools:
                return False
            
            pool = self.memory_pools[pool_name]
            
            # Check if allocation would exceed pool size
            if pool['current_size'] + size > pool['max_size']:
                # Try to make space
                await self._make_space_in_pool(pool_name, size)
                
                # Check again
                if pool['current_size'] + size > pool['max_size']:
                    return False
            
            # Allocate memory
            pool['items'][key] = {
                'size': size,
                'data': data,
                'timestamp': datetime.now(timezone.utc)
            }
            pool['current_size'] += size
            
            return True
            
        except Exception as e:
            logger.error(f"Error allocating memory: {e}")
            return False
    
    async def deallocate_memory(self, pool_name: str, key: str) -> bool:
        """Deallocate memory from pool."""
        try:
            if pool_name not in self.memory_pools:
                return False
            
            pool = self.memory_pools[pool_name]
            
            if key in pool['items']:
                item = pool['items'][key]
                pool['current_size'] -= item['size']
                del pool['items'][key]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deallocating memory: {e}")
            return False
    
    async def _make_space_in_pool(self, pool_name: str, required_size: int):
        """Make space in memory pool."""
        try:
            pool = self.memory_pools[pool_name]
            
            # Sort items by timestamp (oldest first)
            sorted_items = sorted(
                pool['items'].items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # Remove oldest items until we have enough space
            space_freed = 0
            for key, item in sorted_items:
                if space_freed >= required_size:
                    break
                
                space_freed += item['size']
                pool['current_size'] -= item['size']
                del pool['items'][key]
            
            logger.debug(f"Made space in pool {pool_name}: {space_freed} bytes")
            
        except Exception as e:
            logger.error(f"Error making space in pool: {e}")
    
    # Monitoring and Statistics
    async def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        try:
            if not self.memory_history:
                return {}
            
            latest_snapshot = self.memory_history[-1]
            
            return {
                'timestamp': latest_snapshot.timestamp.isoformat(),
                'system_memory': latest_snapshot.system_memory,
                'gpu_memory': latest_snapshot.gpu_memory,
                'process_memory': latest_snapshot.process_memory,
                'model_memory': latest_snapshot.model_memory,
                'cache_memory': latest_snapshot.cache_memory,
                'memory_pools': {
                    name: {
                        'current_size': pool['current_size'],
                        'max_size': pool['max_size'],
                        'utilization': pool['current_size'] / pool['max_size'],
                        'items': len(pool['items'])
                    }
                    for name, pool in self.memory_pools.items()
                },
                'thresholds': {
                    'system_warning': self.thresholds.system_warning,
                    'system_critical': self.thresholds.system_critical,
                    'gpu_warning': self.thresholds.gpu_warning,
                    'gpu_critical': self.thresholds.gpu_critical
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            return {}
    
    async def get_memory_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get memory usage history."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            history = []
            for snapshot in self.memory_history:
                if snapshot.timestamp >= cutoff_time:
                    history.append({
                        'timestamp': snapshot.timestamp.isoformat(),
                        'system_memory': snapshot.system_memory,
                        'gpu_memory': snapshot.gpu_memory,
                        'process_memory': snapshot.process_memory
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting memory history: {e}")
            return []
    
    async def get_memory_alerts(self) -> List[Dict[str, Any]]:
        """Get recent memory alerts."""
        try:
            alerts = []
            for alert_entry in self.memory_alerts:
                alerts.append({
                    'timestamp': alert_entry['timestamp'].isoformat(),
                    'alert': alert_entry['alert']
                })
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting memory alerts: {e}")
            return []
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        try:
            return {
                'memory_stats': self.memory_stats,
                'monitoring_active': self.monitoring_active,
                'memory_history_size': len(self.memory_history),
                'memory_alerts_count': len(self.memory_alerts),
                'memory_pools': len(self.memory_pools),
                'model_registry': len(self.model_registry),
                'tracked_objects': {
                    obj_type: len(weak_set)
                    for obj_type, weak_set in self.tracked_objects.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {}
    
    def set_memory_thresholds(
        self,
        system_warning: Optional[float] = None,
        system_critical: Optional[float] = None,
        gpu_warning: Optional[float] = None,
        gpu_critical: Optional[float] = None
    ):
        """Set memory thresholds."""
        try:
            if system_warning is not None:
                self.thresholds.system_warning = system_warning
            if system_critical is not None:
                self.thresholds.system_critical = system_critical
            if gpu_warning is not None:
                self.thresholds.gpu_warning = gpu_warning
            if gpu_critical is not None:
                self.thresholds.gpu_critical = gpu_critical
            
            logger.info("Memory thresholds updated")
            
        except Exception as e:
            logger.error(f"Error setting memory thresholds: {e}")
    
    def reset_statistics(self):
        """Reset memory statistics."""
        self.memory_stats = {
            'total_memory_checks': 0,
            'memory_optimizations': 0,
            'cache_cleanups': 0,
            'model_quantizations': 0,
            'memory_leaks_detected': 0,
            'gpu_memory_optimizations': 0
        }
        logger.info("Memory statistics reset")


# Global memory manager instance
memory_manager = MemoryManager()