"""
Enhanced WebSocket connection manager with binary frame support.

Handles:
- Binary frame message protocol
- Connection state management
- Message compression and batching
- Client reconnection handling
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import zlib
import gzip
from io import BytesIO

from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types."""
    FRAME_DATA = "frame_data"
    TRACKING_UPDATE = "tracking_update"
    SYSTEM_STATUS = "system_status"
    CONTROL_MESSAGE = "control_message"
    BINARY_FRAME = "binary_frame"


@dataclass
class ConnectionMetrics:
    """Connection performance metrics."""
    connected_at: datetime
    last_message_at: datetime
    messages_sent: int
    messages_failed: int
    bytes_sent: int
    compression_ratio: float
    average_latency: float


class BinaryWebSocketManager:
    """
    Enhanced WebSocket manager with binary frame support.
    
    Features:
    - Binary frame transmission
    - Message compression and batching
    - Connection health monitoring
    - Automatic reconnection handling
    """
    
    def __init__(self):
        # Active connections grouped by task_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
        # Connection metrics tracking
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        
        # Message compression settings
        self.enable_compression = True
        self.compression_threshold = 1024  # bytes
        self.compression_level = 6
        
        # Message batching settings
        self.enable_batching = True
        self.batch_size = 10
        self.batch_timeout = 0.1  # seconds
        self.message_batches: Dict[str, List[Dict[str, Any]]] = {}
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
        # Performance monitoring
        self.performance_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_messages_sent": 0,
            "total_bytes_sent": 0,
            "average_compression_ratio": 0.0,
            "connection_failures": 0
        }
        
        logger.info("BinaryWebSocketManager initialized with binary frame support")
    
    async def connect(self, websocket: WebSocket, task_id: str) -> bool:
        """
        Accept WebSocket connection and register it.
        
        Args:
            websocket: WebSocket connection
            task_id: Task identifier
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Accepting WebSocket connection for task_id: {task_id}")
            
            # Accept the WebSocket connection
            await websocket.accept()
            
            # Initialize connection tracking
            if task_id not in self.active_connections:
                self.active_connections[task_id] = []
            
            self.active_connections[task_id].append(websocket)
            
            # Initialize metrics
            connection_key = f"{task_id}_{id(websocket)}"
            self.connection_metrics[connection_key] = ConnectionMetrics(
                connected_at=datetime.now(timezone.utc),
                last_message_at=datetime.now(timezone.utc),
                messages_sent=0,
                messages_failed=0,
                bytes_sent=0,
                compression_ratio=1.0,
                average_latency=0.0
            )
            
            # Initialize batching for this connection
            if task_id not in self.message_batches:
                self.message_batches[task_id] = []
            
            # Update performance stats
            self.performance_stats["total_connections"] += 1
            self.performance_stats["active_connections"] += 1
            
            logger.info(f"WebSocket connected successfully for task_id: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection for task_id {task_id}: {e}")
            self.performance_stats["connection_failures"] += 1
            return False
    
    async def disconnect(self, websocket: WebSocket, task_id: str):
        """
        Disconnect and cleanup WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            task_id: Task identifier
        """
        try:
            logger.info(f"Disconnecting WebSocket for task_id: {task_id}")
            
            # Remove from active connections
            if task_id in self.active_connections:
                if websocket in self.active_connections[task_id]:
                    self.active_connections[task_id].remove(websocket)
                    
                    # Clean up empty task connections
                    if not self.active_connections[task_id]:
                        del self.active_connections[task_id]
                        
                        # Cancel batch timer if exists
                        if task_id in self.batch_timers:
                            self.batch_timers[task_id].cancel()
                            del self.batch_timers[task_id]
                        
                        # Clear message batches
                        if task_id in self.message_batches:
                            del self.message_batches[task_id]
            
            # Clean up metrics
            connection_key = f"{task_id}_{id(websocket)}"
            if connection_key in self.connection_metrics:
                del self.connection_metrics[connection_key]
            
            # Update performance stats
            self.performance_stats["active_connections"] -= 1
            
            logger.info(f"WebSocket disconnected for task_id: {task_id}")
            
        except Exception as e:
            logger.error(f"Error during disconnect for task_id {task_id}: {e}")
    
    async def send_binary_frame(
        self, 
        task_id: str, 
        frame_data: bytes, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Send binary frame data with JSON metadata.
        
        Args:
            task_id: Task identifier
            frame_data: Binary frame data (JPEG)
            metadata: JSON metadata
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if task_id not in self.active_connections:
                return False
            
            # Prepare binary message
            # Format: [metadata_length][metadata_json][frame_data]
            metadata_json = json.dumps(metadata).encode('utf-8')
            metadata_length = len(metadata_json).to_bytes(4, byteorder='big')
            
            # Combine metadata and frame data
            binary_message = metadata_length + metadata_json + frame_data
            
            # Apply compression if enabled and beneficial
            if self.enable_compression and len(binary_message) > self.compression_threshold:
                compressed_data = gzip.compress(binary_message, compresslevel=self.compression_level)
                compression_ratio = len(compressed_data) / len(binary_message)
                
                if compression_ratio < 0.9:  # Only use compression if it saves >10%
                    binary_message = compressed_data
                    metadata["compressed"] = True
                    
                    # Update compression stats
                    self._update_compression_stats(compression_ratio)
            
            # Send to all connections for this task
            success_count = 0
            connections_to_remove = []
            
            for websocket in self.active_connections[task_id]:
                try:
                    await websocket.send_bytes(binary_message)
                    success_count += 1
                    
                    # Update metrics
                    self._update_connection_metrics(task_id, websocket, len(binary_message))
                    
                except Exception as e:
                    logger.error(f"Failed to send binary frame to connection: {e}")
                    connections_to_remove.append(websocket)
            
            # Clean up failed connections
            for websocket in connections_to_remove:
                await self.disconnect(websocket, task_id)
            
            # Update performance stats
            self.performance_stats["total_messages_sent"] += success_count
            self.performance_stats["total_bytes_sent"] += len(binary_message) * success_count
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending binary frame for task_id {task_id}: {e}")
            return False
    
    async def send_json_message(
        self, 
        task_id: str, 
        message: Dict[str, Any],
        message_type: MessageType = MessageType.TRACKING_UPDATE
    ) -> bool:
        """
        Send JSON message with optional batching.
        
        Args:
            task_id: Task identifier
            message: Message data
            message_type: Type of message
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if task_id not in self.active_connections:
                return False
            
            # Add message type and timestamp
            message["type"] = message_type.value
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Handle batching
            if self.enable_batching and message_type == MessageType.TRACKING_UPDATE:
                return await self._handle_batched_message(task_id, message)
            else:
                return await self._send_immediate_message(task_id, message)
            
        except Exception as e:
            logger.error(f"Error sending JSON message for task_id {task_id}: {e}")
            return False
    
    async def _handle_batched_message(self, task_id: str, message: Dict[str, Any]) -> bool:
        """Handle batched message sending."""
        try:
            # Add to batch
            self.message_batches[task_id].append(message)
            
            # Check if batch is full
            if len(self.message_batches[task_id]) >= self.batch_size:
                return await self._send_batch(task_id)
            
            # Start batch timer if not already running
            if task_id not in self.batch_timers:
                self.batch_timers[task_id] = asyncio.create_task(
                    self._batch_timer(task_id)
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling batched message for task_id {task_id}: {e}")
            return False
    
    async def _batch_timer(self, task_id: str):
        """Timer for batched message sending."""
        try:
            await asyncio.sleep(self.batch_timeout)
            
            if task_id in self.message_batches and self.message_batches[task_id]:
                await self._send_batch(task_id)
            
            # Clean up timer
            if task_id in self.batch_timers:
                del self.batch_timers[task_id]
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in batch timer for task_id {task_id}: {e}")
    
    async def _send_batch(self, task_id: str) -> bool:
        """Send batched messages."""
        try:
            if task_id not in self.message_batches or not self.message_batches[task_id]:
                return True
            
            # Create batch message
            batch_message = {
                "type": "message_batch",
                "messages": self.message_batches[task_id],
                "batch_size": len(self.message_batches[task_id]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Send batch
            success = await self._send_immediate_message(task_id, batch_message)
            
            # Clear batch
            self.message_batches[task_id] = []
            
            # Cancel timer
            if task_id in self.batch_timers:
                self.batch_timers[task_id].cancel()
                del self.batch_timers[task_id]
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending batch for task_id {task_id}: {e}")
            return False
    
    async def _send_immediate_message(self, task_id: str, message: Dict[str, Any]) -> bool:
        """Send message immediately to all connections."""
        try:
            if task_id not in self.active_connections:
                return False
            
            # Convert to JSON
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            
            # Apply compression if beneficial
            if self.enable_compression and len(message_bytes) > self.compression_threshold:
                compressed_data = gzip.compress(message_bytes, compresslevel=self.compression_level)
                compression_ratio = len(compressed_data) / len(message_bytes)
                
                if compression_ratio < 0.9:
                    message_bytes = compressed_data
                    message["compressed"] = True
                    self._update_compression_stats(compression_ratio)
            
            # Send to all connections
            success_count = 0
            connections_to_remove = []
            
            for websocket in self.active_connections[task_id]:
                try:
                    if message.get("compressed"):
                        await websocket.send_bytes(message_bytes)
                    else:
                        await websocket.send_text(message_json)
                    
                    success_count += 1
                    self._update_connection_metrics(task_id, websocket, len(message_bytes))
                    
                except Exception as e:
                    logger.error(f"Failed to send message to connection: {e}")
                    connections_to_remove.append(websocket)
            
            # Clean up failed connections
            for websocket in connections_to_remove:
                await self.disconnect(websocket, task_id)
            
            # Update performance stats
            self.performance_stats["total_messages_sent"] += success_count
            self.performance_stats["total_bytes_sent"] += len(message_bytes) * success_count
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending immediate message for task_id {task_id}: {e}")
            return False
    
    def _update_connection_metrics(self, task_id: str, websocket: WebSocket, bytes_sent: int):
        """Update connection metrics."""
        try:
            connection_key = f"{task_id}_{id(websocket)}"
            
            if connection_key in self.connection_metrics:
                metrics = self.connection_metrics[connection_key]
                metrics.last_message_at = datetime.now(timezone.utc)
                metrics.messages_sent += 1
                metrics.bytes_sent += bytes_sent
                
        except Exception as e:
            logger.error(f"Error updating connection metrics: {e}")
    
    def _update_compression_stats(self, compression_ratio: float):
        """Update compression statistics."""
        try:
            current_avg = self.performance_stats["average_compression_ratio"]
            total_messages = self.performance_stats["total_messages_sent"]
            
            if total_messages > 0:
                self.performance_stats["average_compression_ratio"] = (
                    (current_avg * (total_messages - 1) + compression_ratio) / total_messages
                )
            else:
                self.performance_stats["average_compression_ratio"] = compression_ratio
                
        except Exception as e:
            logger.error(f"Error updating compression stats: {e}")
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status to all connections."""
        try:
            system_message = {
                "type": MessageType.SYSTEM_STATUS.value,
                "data": status_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Send to all active connections
            for task_id in list(self.active_connections.keys()):
                await self.send_json_message(task_id, system_message, MessageType.SYSTEM_STATUS)
            
        except Exception as e:
            logger.error(f"Error broadcasting system status: {e}")
    
    async def flush_all_batches(self):
        """Flush all pending message batches."""
        try:
            for task_id in list(self.message_batches.keys()):
                if self.message_batches[task_id]:
                    await self._send_batch(task_id)
                    
        except Exception as e:
            logger.error(f"Error flushing batches: {e}")
    
    def get_connection_count(self, task_id: Optional[str] = None) -> int:
        """Get connection count for task or total."""
        if task_id:
            return len(self.active_connections.get(task_id, []))
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.performance_stats,
            "active_tasks": len(self.active_connections),
            "pending_batches": sum(len(batch) for batch in self.message_batches.values()),
            "active_timers": len(self.batch_timers)
        }
    
    def get_connection_metrics(self, task_id: str) -> List[Dict[str, Any]]:
        """Get connection metrics for a task."""
        metrics = []
        
        for key, metric in self.connection_metrics.items():
            if key.startswith(f"{task_id}_"):
                metrics.append({
                    "connected_at": metric.connected_at.isoformat(),
                    "last_message_at": metric.last_message_at.isoformat(),
                    "messages_sent": metric.messages_sent,
                    "messages_failed": metric.messages_failed,
                    "bytes_sent": metric.bytes_sent,
                    "compression_ratio": metric.compression_ratio,
                    "average_latency": metric.average_latency
                })
        
        return metrics
    
    async def cleanup(self):
        """Clean up all connections and resources."""
        try:
            # Cancel all batch timers
            for timer in self.batch_timers.values():
                timer.cancel()
            
            # Close all connections
            for task_id, connections in self.active_connections.items():
                for websocket in connections:
                    try:
                        await websocket.close()
                    except Exception as e:
                        logger.error(f"Error closing websocket: {e}")
            
            # Clear all data
            self.active_connections.clear()
            self.connection_metrics.clear()
            self.message_batches.clear()
            self.batch_timers.clear()
            
            logger.info("BinaryWebSocketManager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global manager instance
binary_websocket_manager = BinaryWebSocketManager()