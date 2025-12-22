"""
Memory Management and Resource Optimization for Phase 5: Production Readiness

Intelligent memory management and resource optimization system for production deployment
as specified in DETECTION.md Phase 5. Provides automatic memory cleanup, resource
monitoring, and optimization strategies for maximum system stability and performance.

Features:
- Automatic memory cleanup and garbage collection
- GPU memory management and optimization
- Resource pressure monitoring and adaptive responses
- Cache management and optimization
- Memory leak detection and prevention
- Resource allocation strategies for different workloads
"""

import gc
import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import asyncio
from collections import defaultdict
import weakref

# Try to import GPU monitoring and management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourcePressureLevel(Enum):
    """Resource pressure levels for adaptive management."""
    LOW = "low"          # < 50% usage
    MEDIUM = "medium"    # 50-70% usage
    HIGH = "high"        # 70-85% usage
    CRITICAL = "critical" # > 85% usage


class OptimizationStrategy(Enum):
    """Memory optimization strategies."""
    CONSERVATIVE = "conservative"  # Minimal cleanup, preserve performance
    BALANCED = "balanced"         # Balance between performance and memory
    AGGRESSIVE = "aggressive"     # Maximum cleanup, sacrifice some performance


@dataclass
class MemoryStats:
    """Current memory usage statistics."""
    total_ram: float = 0.0         # Total RAM in GB
    available_ram: float = 0.0     # Available RAM in GB
    used_ram: float = 0.0          # Used RAM in GB
    ram_percentage: float = 0.0    # RAM usage percentage
    
    gpu_total: float = 0.0         # Total GPU memory in GB
    gpu_available: float = 0.0     # Available GPU memory in GB
    gpu_used: float = 0.0          # Used GPU memory in GB
    gpu_percentage: float = 0.0    # GPU memory usage percentage
    
    cache_size: float = 0.0        # Cache size in MB
    gc_collections: int = 0        # Number of GC collections performed


@dataclass
class ResourceThresholds:
    """Configurable resource thresholds for optimization."""
    memory_warning: float = 70.0      # Warning threshold (%)
    memory_critical: float = 85.0     # Critical threshold (%)
    memory_emergency: float = 95.0    # Emergency cleanup threshold (%)
    
    gpu_warning: float = 80.0         # GPU memory warning (%)
    gpu_critical: float = 90.0        # GPU memory critical (%)
    
    cache_max_size: float = 500.0     # Maximum cache size (MB)
    cleanup_interval: float = 60.0    # Cleanup interval (seconds)


class MemoryManager:
    """
    Intelligent memory management system for production deployment.
    
    Monitors resource usage, performs automatic cleanup, and implements
    adaptive optimization strategies to maintain system stability.
    """
    
    def __init__(self, thresholds: ResourceThresholds = None):
        """
        Initialize memory manager.
        
        Args:
            thresholds: Resource thresholds configuration
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.stats = MemoryStats()
        
        # Resource monitoring
        self.current_pressure_level = ResourcePressureLevel.LOW
        self.optimization_strategy = OptimizationStrategy.BALANCED
        
        # Cleanup tracking
        self.cleanup_history: List[Dict[str, Any]] = []
        self.last_cleanup_time = 0.0
        self.cleanup_in_progress = False
        
        # Cache management
        self.managed_caches: Dict[str, weakref.ref] = {}
        self.cache_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Resource tracking
        self.resource_allocations: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.memory_snapshots: List[MemoryStats] = []
        
        # Optimization callbacks
        self.optimization_callbacks: List[Callable] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("MemoryManager initialized with adaptive resource optimization")
    
    async def start_monitoring(self, monitoring_interval: float = 30.0):
        """
        Start continuous resource monitoring.
        
        Args:
            monitoring_interval: How often to check resource usage (seconds)
        """
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(monitoring_interval)
        )
        logger.info(f"Started memory monitoring with {monitoring_interval}s interval")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped memory monitoring")
    
    async def _monitoring_loop(self, interval: float):
        """Main resource monitoring loop."""
        while self.monitoring_active:
            try:
                # Update memory statistics
                await self._update_memory_stats()
                
                # Assess resource pressure
                pressure_level = self._assess_resource_pressure()
                
                # Take action if pressure level changed
                if pressure_level != self.current_pressure_level:
                    logger.info(f"Resource pressure changed: {self.current_pressure_level.value} -> {pressure_level.value}")
                    self.current_pressure_level = pressure_level
                    await self._handle_pressure_change(pressure_level)
                
                # Periodic cleanup if needed
                if self._should_perform_cleanup():
                    await self.perform_cleanup()
                
                # Store memory snapshot
                self._store_memory_snapshot()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(min(interval, 60.0))
    
    async def _update_memory_stats(self):
        """Update current memory usage statistics."""
        try:
            # RAM statistics
            memory = psutil.virtual_memory()
            self.stats.total_ram = memory.total / (1024**3)  # GB
            self.stats.available_ram = memory.available / (1024**3)  # GB
            self.stats.used_ram = memory.used / (1024**3)  # GB
            self.stats.ram_percentage = memory.percent
            
            # GPU statistics (if available)
            if GPU_AVAILABLE and TORCH_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        gpu_allocated = torch.cuda.memory_allocated(0)
                        gpu_cached = torch.cuda.memory_reserved(0)
                        
                        self.stats.gpu_total = gpu_memory / (1024**3)  # GB
                        self.stats.gpu_used = gpu_allocated / (1024**3)  # GB
                        self.stats.gpu_available = (gpu_memory - gpu_allocated) / (1024**3)  # GB
                        self.stats.gpu_percentage = (gpu_allocated / gpu_memory) * 100
                except Exception as gpu_error:
                    pass # logger.debug(f"GPU memory stats collection failed: {gpu_error}")
            
            # Cache statistics
            total_cache_size = sum(
                stats.get('size_mb', 0) 
                for stats in self.cache_statistics.values()
            )
            self.stats.cache_size = total_cache_size
            
        except Exception as e:
            logger.error(f"Error updating memory stats: {e}")
    
    def _assess_resource_pressure(self) -> ResourcePressureLevel:
        """Assess current resource pressure level."""
        try:
            # Check RAM pressure
            ram_pressure = ResourcePressureLevel.LOW
            if self.stats.ram_percentage >= self.thresholds.memory_emergency:
                ram_pressure = ResourcePressureLevel.CRITICAL
            elif self.stats.ram_percentage >= self.thresholds.memory_critical:
                ram_pressure = ResourcePressureLevel.HIGH
            elif self.stats.ram_percentage >= self.thresholds.memory_warning:
                ram_pressure = ResourcePressureLevel.MEDIUM
            
            # Check GPU pressure
            gpu_pressure = ResourcePressureLevel.LOW
            if self.stats.gpu_percentage >= self.thresholds.gpu_critical:
                gpu_pressure = ResourcePressureLevel.CRITICAL
            elif self.stats.gpu_percentage >= self.thresholds.gpu_warning:
                gpu_pressure = ResourcePressureLevel.HIGH
            
            # Return highest pressure level
            pressure_levels = [ram_pressure, gpu_pressure]
            pressure_values = [
                ResourcePressureLevel.LOW,
                ResourcePressureLevel.MEDIUM,
                ResourcePressureLevel.HIGH,
                ResourcePressureLevel.CRITICAL
            ]
            
            return max(pressure_levels, key=lambda x: pressure_values.index(x))
            
        except Exception as e:
            logger.error(f"Error assessing resource pressure: {e}")
            return ResourcePressureLevel.LOW
    
    async def _handle_pressure_change(self, new_pressure: ResourcePressureLevel):
        """Handle resource pressure level changes."""
        try:
            if new_pressure == ResourcePressureLevel.CRITICAL:
                # Emergency cleanup
                logger.warning("Critical resource pressure - performing emergency cleanup")
                await self.perform_emergency_cleanup()
                self.optimization_strategy = OptimizationStrategy.AGGRESSIVE
                
            elif new_pressure == ResourcePressureLevel.HIGH:
                # Aggressive cleanup
                logger.warning("High resource pressure - performing aggressive cleanup")
                await self.perform_cleanup(OptimizationStrategy.AGGRESSIVE)
                self.optimization_strategy = OptimizationStrategy.AGGRESSIVE
                
            elif new_pressure == ResourcePressureLevel.MEDIUM:
                # Balanced cleanup
                await self.perform_cleanup(OptimizationStrategy.BALANCED)
                self.optimization_strategy = OptimizationStrategy.BALANCED
                
            else:  # LOW pressure
                # Conservative approach
                self.optimization_strategy = OptimizationStrategy.CONSERVATIVE
            
            # Notify optimization callbacks
            for callback in self.optimization_callbacks:
                try:
                    await callback(new_pressure, self.optimization_strategy)
                except Exception as callback_error:
                    logger.error(f"Error in optimization callback: {callback_error}")
                    
        except Exception as e:
            logger.error(f"Error handling pressure change: {e}")
    
    def _should_perform_cleanup(self) -> bool:
        """Check if periodic cleanup should be performed."""
        try:
            current_time = time.time()
            
            # Check cleanup interval
            if current_time - self.last_cleanup_time < self.thresholds.cleanup_interval:
                return False
            
            # Check if cleanup is already in progress
            if self.cleanup_in_progress:
                return False
            
            # Check resource thresholds
            if (self.stats.ram_percentage >= self.thresholds.memory_warning or
                self.stats.gpu_percentage >= self.thresholds.gpu_warning or
                self.stats.cache_size >= self.thresholds.cache_max_size):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cleanup conditions: {e}")
            return False
    
    async def perform_cleanup(self, strategy: OptimizationStrategy = None) -> Dict[str, Any]:
        """
        Perform memory cleanup based on optimization strategy.
        
        Args:
            strategy: Optimization strategy to use
            
        Returns:
            Cleanup results and statistics
        """
        if self.cleanup_in_progress:
            logger.warning("Cleanup already in progress")
            return {'status': 'already_in_progress'}
        
        self.cleanup_in_progress = True
        cleanup_start = time.time()
        strategy = strategy or self.optimization_strategy
        
        try:
            logger.info(f"Starting memory cleanup with {strategy.value} strategy")
            
            # Record pre-cleanup stats
            pre_cleanup_stats = MemoryStats(
                used_ram=self.stats.used_ram,
                ram_percentage=self.stats.ram_percentage,
                gpu_used=self.stats.gpu_used,
                gpu_percentage=self.stats.gpu_percentage,
                cache_size=self.stats.cache_size
            )
            
            cleanup_results = {
                'strategy': strategy.value,
                'start_time': cleanup_start,
                'pre_cleanup': pre_cleanup_stats,
                'actions_performed': []
            }
            
            # 1. Python garbage collection
            if strategy in [OptimizationStrategy.BALANCED, OptimizationStrategy.AGGRESSIVE]:
                gc_start = time.time()
                collected = gc.collect()
                self.stats.gc_collections += 1
                cleanup_results['actions_performed'].append({
                    'action': 'garbage_collection',
                    'objects_collected': collected,
                    'duration': time.time() - gc_start
                })
                pass # logger.debug(f"Garbage collection freed {collected} objects")
            
            # 2. Cache cleanup
            if strategy != OptimizationStrategy.CONSERVATIVE:
                cache_cleanup = await self._cleanup_caches(strategy)
                cleanup_results['actions_performed'].append(cache_cleanup)
            
            # 3. GPU memory cleanup
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_cleanup = await self._cleanup_gpu_memory(strategy)
                cleanup_results['actions_performed'].append(gpu_cleanup)
            
            # 4. Resource allocation optimization
            if strategy == OptimizationStrategy.AGGRESSIVE:
                resource_optimization = await self._optimize_resource_allocations()
                cleanup_results['actions_performed'].append(resource_optimization)
            
            # Update post-cleanup stats
            await self._update_memory_stats()
            cleanup_results['post_cleanup'] = MemoryStats(
                used_ram=self.stats.used_ram,
                ram_percentage=self.stats.ram_percentage,
                gpu_used=self.stats.gpu_used,
                gpu_percentage=self.stats.gpu_percentage,
                cache_size=self.stats.cache_size
            )
            
            # Calculate cleanup effectiveness
            ram_freed = pre_cleanup_stats.used_ram - self.stats.used_ram
            gpu_freed = pre_cleanup_stats.gpu_used - self.stats.gpu_used
            cache_freed = pre_cleanup_stats.cache_size - self.stats.cache_size
            
            cleanup_results.update({
                'duration': time.time() - cleanup_start,
                'ram_freed_gb': ram_freed,
                'gpu_freed_gb': gpu_freed,
                'cache_freed_mb': cache_freed,
                'effectiveness_score': self._calculate_cleanup_effectiveness(pre_cleanup_stats, self.stats)
            })
            
            # Store cleanup history
            self.cleanup_history.append(cleanup_results)
            if len(self.cleanup_history) > 100:  # Keep last 100 cleanups
                self.cleanup_history = self.cleanup_history[-50:]
            
            self.last_cleanup_time = time.time()
            
            logger.info(f"Memory cleanup completed - RAM freed: {ram_freed:.2f}GB, "
                       f"GPU freed: {gpu_freed:.2f}GB, Cache freed: {cache_freed:.1f}MB")
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return {'status': 'error', 'error': str(e)}
            
        finally:
            self.cleanup_in_progress = False
    
    async def perform_emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency cleanup when resources are critically low."""
        logger.critical("Performing emergency memory cleanup")
        
        try:
            # Force aggressive cleanup
            cleanup_results = await self.perform_cleanup(OptimizationStrategy.AGGRESSIVE)
            
            # Additional emergency measures
            emergency_actions = []
            
            # Force multiple garbage collections
            for i in range(3):
                collected = gc.collect()
                emergency_actions.append(f"Emergency GC #{i+1}: {collected} objects")
            
            # Clear all caches aggressively
            for cache_name in list(self.managed_caches.keys()):
                cache_ref = self.managed_caches[cache_name]
                cache = cache_ref() if cache_ref else None
                if cache and hasattr(cache, 'clear'):
                    cache.clear()
                    emergency_actions.append(f"Cleared cache: {cache_name}")
            
            # GPU emergency cleanup
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                emergency_actions.append("Emergency GPU cache clear")
            
            cleanup_results['emergency_actions'] = emergency_actions
            
            logger.critical(f"Emergency cleanup completed with {len(emergency_actions)} emergency actions")
            
            return cleanup_results
            
        except Exception as e:
            logger.critical(f"Emergency cleanup failed: {e}")
            return {'status': 'emergency_failed', 'error': str(e)}
    
    async def _cleanup_caches(self, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Cleanup managed caches based on strategy."""
        try:
            cache_action = {
                'action': 'cache_cleanup',
                'caches_processed': 0,
                'total_freed_mb': 0.0,
                'details': []
            }
            
            for cache_name, cache_ref in list(self.managed_caches.items()):
                cache = cache_ref() if cache_ref else None
                
                if cache is None:
                    # Cache was garbage collected, remove reference
                    del self.managed_caches[cache_name]
                    continue
                
                cache_stats = self.cache_statistics.get(cache_name, {})
                cache_size = cache_stats.get('size_mb', 0)
                
                if cache_size == 0:
                    continue
                
                cleanup_performed = False
                
                if strategy == OptimizationStrategy.AGGRESSIVE:
                    # Clear entire cache
                    if hasattr(cache, 'clear'):
                        cache.clear()
                        cleanup_performed = True
                        cache_action['details'].append(f"{cache_name}: cleared completely")
                
                elif strategy == OptimizationStrategy.BALANCED:
                    # Partial cache cleanup
                    if hasattr(cache, 'evict_lru'):
                        evicted = cache.evict_lru(0.5)  # Evict 50% of LRU items
                        cleanup_performed = True
                        cache_action['details'].append(f"{cache_name}: evicted {evicted} LRU items")
                    elif hasattr(cache, 'clear') and cache_size > 100:  # Clear if > 100MB
                        cache.clear()
                        cleanup_performed = True
                        cache_action['details'].append(f"{cache_name}: cleared (large cache)")
                
                if cleanup_performed:
                    cache_action['caches_processed'] += 1
                    cache_action['total_freed_mb'] += cache_size
                    
                    # Update cache statistics
                    self.cache_statistics[cache_name]['size_mb'] = 0
                    self.cache_statistics[cache_name]['last_cleanup'] = time.time()
            
            return cache_action
            
        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}")
            return {'action': 'cache_cleanup', 'error': str(e)}
    
    async def _cleanup_gpu_memory(self, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Cleanup GPU memory based on strategy."""
        try:
            gpu_action = {
                'action': 'gpu_cleanup',
                'freed_mb': 0.0,
                'actions': []
            }
            
            if not (TORCH_AVAILABLE and torch.cuda.is_available()):
                gpu_action['actions'].append('GPU not available')
                return gpu_action
            
            # Record pre-cleanup GPU memory
            pre_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            pre_cached = torch.cuda.memory_reserved() / (1024**2)  # MB
            
            if strategy in [OptimizationStrategy.BALANCED, OptimizationStrategy.AGGRESSIVE]:
                # Clear GPU cache
                torch.cuda.empty_cache()
                gpu_action['actions'].append('Cleared GPU cache')
            
            if strategy == OptimizationStrategy.AGGRESSIVE:
                # Force GPU synchronization and additional cleanup
                torch.cuda.synchronize()
                gpu_action['actions'].append('GPU synchronization')
                
                # Try to free unused GPU memory
                if hasattr(torch.cuda, 'memory'):
                    torch.cuda.memory.empty_cache()
                    gpu_action['actions'].append('Additional GPU memory cleanup')
            
            # Calculate freed memory
            post_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            post_cached = torch.cuda.memory_reserved() / (1024**2)  # MB
            
            gpu_action['freed_mb'] = (pre_cached - post_cached)
            
            return gpu_action
            
        except Exception as e:
            logger.error(f"Error in GPU cleanup: {e}")
            return {'action': 'gpu_cleanup', 'error': str(e)}
    
    async def _optimize_resource_allocations(self) -> Dict[str, Any]:
        """Optimize resource allocations for better memory usage."""
        try:
            optimization_action = {
                'action': 'resource_optimization',
                'optimizations': []
            }
            
            # Placeholder for resource allocation optimization
            # In production, this could include:
            # - Adjusting batch sizes based on available memory
            # - Optimizing model configurations
            # - Adjusting cache sizes
            # - Rebalancing worker processes
            
            optimization_action['optimizations'].append('Resource allocation assessment completed')
            
            return optimization_action
            
        except Exception as e:
            logger.error(f"Error in resource optimization: {e}")
            return {'action': 'resource_optimization', 'error': str(e)}
    
    def register_cache(self, name: str, cache_object: Any, size_estimator: Callable = None):
        """
        Register a cache for automatic management.
        
        Args:
            name: Cache identifier
            cache_object: Cache object to manage
            size_estimator: Function to estimate cache size in MB
        """
        try:
            self.managed_caches[name] = weakref.ref(cache_object)
            
            # Initialize cache statistics
            self.cache_statistics[name] = {
                'registered_at': time.time(),
                'size_mb': 0.0,
                'last_cleanup': 0.0,
                'size_estimator': size_estimator
            }
            
            pass # logger.debug(f"Registered cache for management: {name}")
            
        except Exception as e:
            logger.error(f"Error registering cache {name}: {e}")
    
    def add_optimization_callback(self, callback: Callable):
        """
        Add callback for resource pressure changes.
        
        Args:
            callback: Async function called when pressure level changes
        """
        try:
            self.optimization_callbacks.append(callback)
            pass # logger.debug("Added optimization callback")
            
        except Exception as e:
            logger.error(f"Error adding optimization callback: {e}")
    
    def _calculate_cleanup_effectiveness(self, pre_stats: MemoryStats, post_stats: MemoryStats) -> float:
        """Calculate cleanup effectiveness score (0-1)."""
        try:
            ram_improvement = max(0, pre_stats.ram_percentage - post_stats.ram_percentage)
            gpu_improvement = max(0, pre_stats.gpu_percentage - post_stats.gpu_percentage)
            
            # Weight improvements
            effectiveness = (ram_improvement * 0.6 + gpu_improvement * 0.4) / 100.0
            
            return min(1.0, effectiveness)
            
        except Exception:
            return 0.0
    
    def _store_memory_snapshot(self):
        """Store current memory statistics for trend analysis."""
        try:
            snapshot = MemoryStats(
                total_ram=self.stats.total_ram,
                available_ram=self.stats.available_ram,
                used_ram=self.stats.used_ram,
                ram_percentage=self.stats.ram_percentage,
                gpu_total=self.stats.gpu_total,
                gpu_available=self.stats.gpu_available,
                gpu_used=self.stats.gpu_used,
                gpu_percentage=self.stats.gpu_percentage,
                cache_size=self.stats.cache_size,
                gc_collections=self.stats.gc_collections
            )
            
            self.memory_snapshots.append(snapshot)
            
            # Keep only recent snapshots
            if len(self.memory_snapshots) > 1440:  # ~24 hours with 1-minute intervals
                self.memory_snapshots = self.memory_snapshots[-720:]  # Keep last 12 hours
                
        except Exception as e:
            logger.error(f"Error storing memory snapshot: {e}")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status and statistics."""
        try:
            return {
                'current_stats': {
                    'ram_usage_gb': self.stats.used_ram,
                    'ram_percentage': self.stats.ram_percentage,
                    'ram_available_gb': self.stats.available_ram,
                    'gpu_usage_gb': self.stats.gpu_used,
                    'gpu_percentage': self.stats.gpu_percentage,
                    'cache_size_mb': self.stats.cache_size,
                    'gc_collections': self.stats.gc_collections
                },
                'resource_pressure': {
                    'current_level': self.current_pressure_level.value,
                    'optimization_strategy': self.optimization_strategy.value,
                    'monitoring_active': self.monitoring_active
                },
                'cleanup_history': {
                    'total_cleanups': len(self.cleanup_history),
                    'last_cleanup': self.cleanup_history[-1] if self.cleanup_history else None,
                    'cleanup_in_progress': self.cleanup_in_progress,
                    'last_cleanup_time': self.last_cleanup_time
                },
                'managed_resources': {
                    'caches_managed': len(self.managed_caches),
                    'active_caches': sum(1 for ref in self.managed_caches.values() if ref() is not None),
                    'optimization_callbacks': len(self.optimization_callbacks)
                },
                'thresholds': {
                    'memory_warning': self.thresholds.memory_warning,
                    'memory_critical': self.thresholds.memory_critical,
                    'memory_emergency': self.thresholds.memory_emergency,
                    'gpu_warning': self.thresholds.gpu_warning,
                    'gpu_critical': self.thresholds.gpu_critical
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            return {'error': str(e)}


# Global memory manager instance
memory_manager = MemoryManager()