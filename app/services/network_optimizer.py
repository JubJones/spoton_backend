"""
Network optimizer for WebSocket connections.

Handles:
- Connection pooling and management
- Bandwidth monitoring and adaptation
- Message queuing for high load
- Compression optimization
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import statistics

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    timestamp: datetime
    bytes_sent: int
    bytes_received: int
    message_count: int
    connection_count: int
    latency: float
    bandwidth_usage: float
    compression_ratio: float


@dataclass
class ConnectionPool:
    """Connection pool management."""
    active_connections: List[Any]
    max_connections: int
    connection_timeout: float
    keepalive_interval: float
    
    def __post_init__(self):
        self.active_connections = []


class NetworkOptimizer:
    """
    Network optimizer for WebSocket connections.
    
    Features:
    - Connection pooling and management
    - Bandwidth monitoring and adaptation
    - Message queuing and batching
    - Compression optimization
    """
    
    def __init__(self):
        # Connection management
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.max_connections_per_pool = 100
        self.connection_timeout = 60.0  # seconds
        self.keepalive_interval = 30.0  # seconds
        
        # Bandwidth monitoring
        self.bandwidth_limit = 10 * 1024 * 1024  # 10MB/s default
        self.bandwidth_usage_history: Deque[float] = deque(maxlen=100)
        self.bytes_sent_history: Deque[int] = deque(maxlen=100)
        self.message_latency_history: Deque[float] = deque(maxlen=100)
        
        # Message queuing
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.queue_processors: Dict[str, asyncio.Task] = {}
        self.max_queue_size = 1000
        self.batch_size = 50
        self.batch_timeout = 0.1  # seconds
        
        # Compression settings
        self.compression_enabled = True
        self.compression_threshold = 1024  # bytes
        self.compression_level = 6
        self.compression_ratio_history: Deque[float] = deque(maxlen=100)
        
        # Network statistics
        self.network_stats = {
            "total_bytes_sent": 0,
            "total_bytes_received": 0,
            "total_messages_sent": 0,
            "total_connections": 0,
            "active_connections": 0,
            "connection_failures": 0,
            "queue_overflows": 0,
            "average_latency": 0.0,
            "average_bandwidth": 0.0,
            "compression_savings": 0.0
        }
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("NetworkOptimizer initialized")
    
    async def start_monitoring(self):
        """Start network monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Network monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting network monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop network monitoring."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
                self.monitoring_task = None
            
            # Stop all queue processors
            for task in self.queue_processors.values():
                task.cancel()
            
            self.queue_processors.clear()
            
            logger.info("Network monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping network monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main network monitoring loop."""
        try:
            while self.monitoring_active:
                # Collect network metrics
                await self._collect_network_metrics()
                
                # Check bandwidth usage
                await self._check_bandwidth_usage()
                
                # Optimize connections
                await self._optimize_connections()
                
                # Update statistics
                await self._update_statistics()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("Network monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in network monitoring loop: {e}")
    
    async def _collect_network_metrics(self):
        """Collect network performance metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Calculate current bandwidth usage
            bandwidth_usage = self._calculate_bandwidth_usage()
            
            # Calculate average latency
            avg_latency = 0.0
            if self.message_latency_history:
                avg_latency = statistics.mean(self.message_latency_history)
            
            # Calculate compression ratio
            compression_ratio = 1.0
            if self.compression_ratio_history:
                compression_ratio = statistics.mean(self.compression_ratio_history)
            
            # Store metrics
            self.bandwidth_usage_history.append(bandwidth_usage)
            
            # Update network stats
            self.network_stats["average_latency"] = avg_latency
            self.network_stats["average_bandwidth"] = bandwidth_usage
            self.network_stats["active_connections"] = self._count_active_connections()
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
    
    async def _check_bandwidth_usage(self):
        """Check and manage bandwidth usage."""
        try:
            current_bandwidth = self._calculate_bandwidth_usage()
            
            if current_bandwidth > self.bandwidth_limit * 0.8:
                logger.warning(f"High bandwidth usage: {current_bandwidth / 1024 / 1024:.1f} MB/s")
                
                # Apply bandwidth throttling
                await self._apply_bandwidth_throttling()
                
            elif current_bandwidth < self.bandwidth_limit * 0.5:
                # Remove throttling if bandwidth is low
                await self._remove_bandwidth_throttling()
                
        except Exception as e:
            logger.error(f"Error checking bandwidth usage: {e}")
    
    async def _optimize_connections(self):
        """Optimize connection pools."""
        try:
            for pool_id, pool in self.connection_pools.items():
                # Remove inactive connections
                active_connections = []
                
                for conn in pool.active_connections:
                    if await self._is_connection_active(conn):
                        active_connections.append(conn)
                    else:
                        await self._close_connection(conn)
                
                pool.active_connections = active_connections
                
                # Log connection pool status
                logger.debug(f"Connection pool {pool_id}: {len(active_connections)} active connections")
                
        except Exception as e:
            logger.error(f"Error optimizing connections: {e}")
    
    async def _apply_bandwidth_throttling(self):
        """Apply bandwidth throttling measures."""
        try:
            # Reduce batch sizes
            self.batch_size = max(10, self.batch_size - 5)
            
            # Increase compression level
            if self.compression_level < 9:
                self.compression_level = min(9, self.compression_level + 1)
            
            # Reduce compression threshold
            self.compression_threshold = max(512, self.compression_threshold - 256)
            
            logger.info(f"Applied bandwidth throttling: batch_size={self.batch_size}, compression_level={self.compression_level}")
            
        except Exception as e:
            logger.error(f"Error applying bandwidth throttling: {e}")
    
    async def _remove_bandwidth_throttling(self):
        """Remove bandwidth throttling measures."""
        try:
            # Restore default settings gradually
            self.batch_size = min(50, self.batch_size + 5)
            self.compression_level = max(6, self.compression_level - 1)
            self.compression_threshold = min(1024, self.compression_threshold + 256)
            
            logger.debug("Removed bandwidth throttling")
            
        except Exception as e:
            logger.error(f"Error removing bandwidth throttling: {e}")
    
    def _calculate_bandwidth_usage(self) -> float:
        """Calculate current bandwidth usage in bytes per second."""
        try:
            if len(self.bytes_sent_history) < 2:
                return 0.0
            
            # Calculate bytes per second over last 10 seconds
            recent_bytes = list(self.bytes_sent_history)[-10:]
            time_span = min(10.0, len(recent_bytes))
            
            if time_span > 0:
                return sum(recent_bytes) / time_span
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating bandwidth usage: {e}")
            return 0.0
    
    def _count_active_connections(self) -> int:
        """Count total active connections."""
        try:
            total = 0
            for pool in self.connection_pools.values():
                total += len(pool.active_connections)
            return total
            
        except Exception as e:
            logger.error(f"Error counting active connections: {e}")
            return 0
    
    async def _is_connection_active(self, connection: Any) -> bool:
        """Check if connection is active."""
        try:
            # This would be implemented based on the connection type
            # For now, assume all connections are active
            return True
            
        except Exception as e:
            logger.error(f"Error checking connection status: {e}")
            return False
    
    async def _close_connection(self, connection: Any):
        """Close inactive connection."""
        try:
            # This would be implemented based on the connection type
            logger.debug("Closed inactive connection")
            
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def _update_statistics(self):
        """Update network statistics."""
        try:
            # Update compression savings
            if self.compression_ratio_history:
                avg_ratio = statistics.mean(self.compression_ratio_history)
                self.network_stats["compression_savings"] = (1.0 - avg_ratio) * 100
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def create_connection_pool(self, pool_id: str, max_connections: int = None) -> ConnectionPool:
        """Create a new connection pool."""
        try:
            if pool_id in self.connection_pools:
                return self.connection_pools[pool_id]
            
            pool = ConnectionPool(
                active_connections=[],
                max_connections=max_connections or self.max_connections_per_pool,
                connection_timeout=self.connection_timeout,
                keepalive_interval=self.keepalive_interval
            )
            
            self.connection_pools[pool_id] = pool
            
            logger.info(f"Created connection pool: {pool_id}")
            
            return pool
            
        except Exception as e:
            logger.error(f"Error creating connection pool: {e}")
            raise
    
    async def add_connection_to_pool(self, pool_id: str, connection: Any) -> bool:
        """Add connection to pool."""
        try:
            if pool_id not in self.connection_pools:
                await self.create_connection_pool(pool_id)
            
            pool = self.connection_pools[pool_id]
            
            if len(pool.active_connections) < pool.max_connections:
                pool.active_connections.append(connection)
                self.network_stats["active_connections"] += 1
                
                logger.debug(f"Added connection to pool {pool_id}")
                return True
            else:
                logger.warning(f"Connection pool {pool_id} is full")
                return False
                
        except Exception as e:
            logger.error(f"Error adding connection to pool: {e}")
            return False
    
    async def remove_connection_from_pool(self, pool_id: str, connection: Any) -> bool:
        """Remove connection from pool."""
        try:
            if pool_id not in self.connection_pools:
                return False
            
            pool = self.connection_pools[pool_id]
            
            if connection in pool.active_connections:
                pool.active_connections.remove(connection)
                self.network_stats["active_connections"] -= 1
                
                logger.debug(f"Removed connection from pool {pool_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing connection from pool: {e}")
            return False
    
    async def create_message_queue(self, queue_id: str) -> asyncio.Queue:
        """Create a message queue for batching."""
        try:
            if queue_id in self.message_queues:
                return self.message_queues[queue_id]
            
            queue = asyncio.Queue(maxsize=self.max_queue_size)
            self.message_queues[queue_id] = queue
            
            # Start queue processor
            processor = asyncio.create_task(self._process_message_queue(queue_id))
            self.queue_processors[queue_id] = processor
            
            logger.info(f"Created message queue: {queue_id}")
            
            return queue
            
        except Exception as e:
            logger.error(f"Error creating message queue: {e}")
            raise
    
    async def _process_message_queue(self, queue_id: str):
        """Process messages from queue with batching."""
        try:
            queue = self.message_queues[queue_id]
            batch = []
            last_batch_time = time.time()
            
            while True:
                try:
                    # Get message with timeout
                    timeout = max(0.01, self.batch_timeout - (time.time() - last_batch_time))
                    message = await asyncio.wait_for(queue.get(), timeout=timeout)
                    
                    batch.append(message)
                    
                    # Send batch if full or timeout reached
                    if (len(batch) >= self.batch_size or 
                        time.time() - last_batch_time >= self.batch_timeout):
                        
                        await self._send_message_batch(queue_id, batch)
                        batch = []
                        last_batch_time = time.time()
                        
                except asyncio.TimeoutError:
                    # Send batch on timeout
                    if batch:
                        await self._send_message_batch(queue_id, batch)
                        batch = []
                        last_batch_time = time.time()
                        
                except Exception as e:
                    logger.error(f"Error processing message queue {queue_id}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in message queue processor {queue_id}: {e}")
    
    async def _send_message_batch(self, queue_id: str, batch: List[Any]):
        """Send a batch of messages."""
        try:
            if not batch:
                return
            
            # This would be implemented to send the batch
            # For now, just log the batch size
            logger.debug(f"Sending batch of {len(batch)} messages for queue {queue_id}")
            
            # Update statistics
            self.network_stats["total_messages_sent"] += len(batch)
            
        except Exception as e:
            logger.error(f"Error sending message batch: {e}")
    
    def record_message_sent(self, bytes_sent: int, latency: float, compression_ratio: float = 1.0):
        """Record message transmission metrics."""
        try:
            self.bytes_sent_history.append(bytes_sent)
            self.message_latency_history.append(latency)
            self.compression_ratio_history.append(compression_ratio)
            
            # Update network stats
            self.network_stats["total_bytes_sent"] += bytes_sent
            self.network_stats["total_messages_sent"] += 1
            
        except Exception as e:
            logger.error(f"Error recording message metrics: {e}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        try:
            return {
                **self.network_stats,
                "bandwidth_usage": self._calculate_bandwidth_usage(),
                "connection_pools": {
                    pool_id: {
                        "active_connections": len(pool.active_connections),
                        "max_connections": pool.max_connections
                    }
                    for pool_id, pool in self.connection_pools.items()
                },
                "message_queues": {
                    queue_id: queue.qsize()
                    for queue_id, queue in self.message_queues.items()
                },
                "compression_enabled": self.compression_enabled,
                "compression_level": self.compression_level,
                "batch_size": self.batch_size,
                "monitoring_active": self.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"Error getting network stats: {e}")
            return {"error": str(e)}
    
    def set_bandwidth_limit(self, limit_bytes_per_second: int):
        """Set bandwidth limit."""
        try:
            self.bandwidth_limit = limit_bytes_per_second
            logger.info(f"Bandwidth limit set to {limit_bytes_per_second / 1024 / 1024:.1f} MB/s")
            
        except Exception as e:
            logger.error(f"Error setting bandwidth limit: {e}")
    
    def enable_compression(self, enabled: bool):
        """Enable or disable compression."""
        try:
            self.compression_enabled = enabled
            logger.info(f"Compression {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Error setting compression: {e}")
    
    def reset_stats(self):
        """Reset network statistics."""
        try:
            self.network_stats = {
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "total_messages_sent": 0,
                "total_connections": 0,
                "active_connections": self._count_active_connections(),
                "connection_failures": 0,
                "queue_overflows": 0,
                "average_latency": 0.0,
                "average_bandwidth": 0.0,
                "compression_savings": 0.0
            }
            
            # Clear history
            self.bandwidth_usage_history.clear()
            self.bytes_sent_history.clear()
            self.message_latency_history.clear()
            self.compression_ratio_history.clear()
            
            logger.info("Network statistics reset")
            
        except Exception as e:
            logger.error(f"Error resetting network stats: {e}")


# Global network optimizer instance
network_optimizer = NetworkOptimizer()