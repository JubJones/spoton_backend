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
        
        # Frame buffer for new connections (store last 3 frames per task)
        self.frame_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.max_buffered_frames = 3
        
        # Message buffer for tracking updates when no WebSocket connected (race condition fix)
        self.pending_message_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.max_buffered_messages = 10
        
        # Message size limits (WebSocket standard limit is ~1MB)
        self.max_message_size = 800 * 1024  # 800KB to leave room for headers
        self.large_message_compression_threshold = 100 * 1024  # 100KB
        
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
            logger.info(f"🔍 WS DEBUG: Accepting WebSocket connection for task_id: {task_id}")
            logger.info(f"🔍 WS DEBUG: Current active_connections before connect: {list(self.active_connections.keys())}")
            
            # Accept the WebSocket connection
            await websocket.accept()
            
            # Wait for WebSocket to be fully ready - prevents race condition
            await asyncio.sleep(0.1)  # Increased delay to ensure connection is stable
            
            # Verify WebSocket state after accept with retry
            from fastapi.websockets import WebSocketState
            for attempt in range(3):
                if websocket.client_state == WebSocketState.CONNECTED:
                    break
                logger.warning(f"WebSocket not ready, attempt {attempt + 1}/3, state: {websocket.client_state}")
                await asyncio.sleep(0.05)
            
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.error(f"WebSocket not in CONNECTED state after accept: {websocket.client_state}")
                return False
            
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
            
            logger.info(f"✅ WS DEBUG: WebSocket connected successfully for task_id: {task_id}")
            logger.info(f"🔍 WS DEBUG: Active connections after connect: {list(self.active_connections.keys())}")
            logger.info(f"🔍 WS DEBUG: Connection count for task {task_id}: {len(self.active_connections[task_id])}")
            
            # Deliver any buffered messages for this task
            await self._deliver_buffered_messages(task_id)
            
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
            logger.info(f"🔍 WS DEBUG: Disconnecting WebSocket for task_id: {task_id}")
            logger.info(f"🔍 WS DEBUG: Active connections before disconnect: {list(self.active_connections.keys())}")
            
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
                        
                        # Clear pending message buffer
                        if task_id in self.pending_message_buffer:
                            logger.info(f"🧹 WS DEBUG: Clearing pending message buffer for task_id: {task_id}")
                            del self.pending_message_buffer[task_id]
            
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
            # Check if task has active connections
            if task_id not in self.active_connections:
                logger.debug(f"No active connections for task_id: {task_id}, skipping frame")
                return False
            
            # Check if there are actually connected websockets
            if not self.active_connections[task_id]:
                logger.debug(f"Empty connection list for task_id: {task_id}, skipping frame")
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
            logger.info(f"🔍 WS DEBUG: send_json_message called for task_id: {task_id}, message_type: {message_type.value}")
            logger.info(f"🔍 WS DEBUG: Current active_connections keys: {list(self.active_connections.keys())}")
            
            # Check if task has active connections
            if task_id not in self.active_connections or not self.active_connections[task_id]:
                # Buffer tracking updates for later delivery when WebSocket connects
                if message_type == MessageType.TRACKING_UPDATE:
                    logger.info(f"📦 WS DEBUG: Buffering tracking update for task_id: {task_id} (no active connections)")
                    await self._buffer_tracking_update(task_id, message)
                    return True  # Consider buffering as successful
                else:
                    logger.warning(f"❌ WS DEBUG: No active connections for task_id: {task_id}, skipping non-tracking message")
                    logger.info(f"🔍 WS DEBUG: Available task_ids: {list(self.active_connections.keys())}")
                    return False
            
            connection_count = len(self.active_connections[task_id])
            logger.info(f"✅ WS DEBUG: Found {connection_count} active connections for task_id: {task_id}")
            
            # Add message type and timestamp
            message["type"] = message_type.value
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Log message size for debugging
            message_size = len(json.dumps(message).encode('utf-8'))
            logger.info(f"📏 WS DEBUG: Message size: {message_size} bytes ({message_size/1024:.1f} KB) for task_id: {task_id}")
            
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
            
            # Send batch - uses _send_immediate_message which now always sends text
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
        """Send message immediately to all connections with enhanced state validation."""
        try:
            # Check if task has active connections
            if task_id not in self.active_connections:
                logger.debug(f"No active connections for task_id: {task_id}")
                return False
            
            # Check if there are actually connected websockets
            if not self.active_connections[task_id]:
                logger.debug(f"Empty connection list for task_id: {task_id}")
                return False
            
            # Convert to JSON
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            
            # Check message size limits
            if len(message_bytes) > self.max_message_size:
                logger.warning(f"⚠️ WS DEBUG: Message too large ({len(message_bytes)} bytes) for task_id: {task_id}, attempting compression")
                
                # Try aggressive compression for large messages
                try:
                    compressed_data = gzip.compress(message_bytes, compresslevel=9)
                    if len(compressed_data) > self.max_message_size:
                        logger.error(f"❌ WS DEBUG: Message still too large after compression ({len(compressed_data)} bytes) for task_id: {task_id}")
                        # Try to reduce frame data or split message
                        message = await self._reduce_message_size(message)
                        message_json = json.dumps(message)
                        message_bytes = message_json.encode('utf-8')
                        logger.info(f"🔧 WS DEBUG: Reduced message size to {len(message_bytes)} bytes for task_id: {task_id}")
                    else:
                        message_bytes = compressed_data
                        logger.info(f"✅ WS DEBUG: Compressed large message from {len(message_json.encode('utf-8'))} to {len(compressed_data)} bytes for task_id: {task_id}")
                except Exception as e:
                    logger.error(f"❌ WS DEBUG: Failed to compress large message for task_id: {task_id}: {e}")
                    return False
            
            # Determine if compression should be applied for normal messages
            use_compression = False
            compressed_data = message_bytes
            
            if self.enable_compression and len(message_bytes) > self.compression_threshold and len(message_bytes) <= self.max_message_size:
                test_compressed = gzip.compress(message_bytes, compresslevel=self.compression_level)
                compression_ratio = len(test_compressed) / len(message_bytes)
                
                if compression_ratio < 0.9:
                    compressed_data = test_compressed
                    use_compression = True
                    self._update_compression_stats(compression_ratio)
            
            # Send to all connections with enhanced state validation
            success_count = 0
            connections_to_remove = []
            
            for websocket in self.active_connections[task_id]:
                try:
                    # Enhanced WebSocket state validation
                    if not await self._is_websocket_ready(websocket):
                        logger.debug(f"WebSocket not ready, marking for removal")
                        connections_to_remove.append(websocket)
                        continue
                    
                    # Send message with retry logic
                    if await self._send_with_retry(websocket, message_json):
                        success_count += 1
                        self._update_connection_metrics(task_id, websocket, len(message_bytes))
                    else:
                        logger.warning(f"Failed to send message after retries")
                        connections_to_remove.append(websocket)
                    
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
    
    async def _is_websocket_ready(self, websocket: WebSocket) -> bool:
        """Check if WebSocket is ready for communication."""
        try:
            from fastapi.websockets import WebSocketState
            
            # Check client state
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.debug(f"WebSocket client state not CONNECTED: {websocket.client_state}")
                return False
            
            # Check application state if available
            if hasattr(websocket, 'application_state'):
                if websocket.application_state != WebSocketState.CONNECTED:
                    logger.debug(f"WebSocket application state not CONNECTED: {websocket.application_state}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking WebSocket readiness: {e}")
            return False
    
    async def _send_with_retry(self, websocket: WebSocket, message: str, max_retries: int = 2) -> bool:
        """Send message with retry logic to handle race conditions."""
        for attempt in range(max_retries + 1):
            try:
                # Validate WebSocket is still ready before each attempt
                if not await self._is_websocket_ready(websocket):
                    logger.debug(f"WebSocket not ready on attempt {attempt + 1}")
                    if attempt < max_retries:
                        await asyncio.sleep(0.01 * (attempt + 1))  # Exponential backoff
                        continue
                    return False
                
                # Send the message
                await websocket.send_text(message)
                return True
                
            except Exception as e:
                logger.debug(f"Send attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.01 * (attempt + 1))  # Exponential backoff
                    continue
                return False
        
        return False
    
    async def _buffer_tracking_update(self, task_id: str, message: Dict[str, Any]):
        """Buffer tracking update for later delivery when WebSocket connects."""
        try:
            # Initialize buffer if not exists
            if task_id not in self.pending_message_buffer:
                self.pending_message_buffer[task_id] = []
            
            # Add message to buffer
            self.pending_message_buffer[task_id].append(message)
            
            # Limit buffer size to prevent memory issues
            if len(self.pending_message_buffer[task_id]) > self.max_buffered_messages:
                # Remove oldest message
                removed_msg = self.pending_message_buffer[task_id].pop(0)
                logger.info(f"📦 WS DEBUG: Buffer full for task_id: {task_id}, removed oldest message")
            
            buffer_size = len(self.pending_message_buffer[task_id])
            logger.info(f"📦 WS DEBUG: Message buffered for task_id: {task_id} (buffer size: {buffer_size})")
            
        except Exception as e:
            logger.error(f"Error buffering tracking update for task_id {task_id}: {e}")
    
    async def _deliver_buffered_messages(self, task_id: str):
        """Deliver all buffered messages when WebSocket connects."""
        try:
            if task_id not in self.pending_message_buffer:
                return
                
            buffered_messages = self.pending_message_buffer[task_id]
            if not buffered_messages:
                return
            
            logger.info(f"📤 WS DEBUG: Delivering {len(buffered_messages)} buffered messages for task_id: {task_id}")
            
            # Send all buffered messages (preemptively reduce size for compatibility)
            for i, message in enumerate(buffered_messages):
                # Preemptively reduce message size for buffered messages to prevent failures
                message_size = len(json.dumps(message).encode('utf-8'))
                if message_size > self.large_message_compression_threshold:
                    logger.info(f"📦 WS DEBUG: Preemptively reducing buffered message size ({message_size} bytes) for task_id: {task_id}")
                    message = await self._reduce_message_size(message)
                
                success = await self._send_immediate_message(task_id, message)
                if success:
                    logger.info(f"📤 WS DEBUG: Delivered buffered message {i+1}/{len(buffered_messages)} for task_id: {task_id}")
                else:
                    logger.warning(f"⚠️ WS DEBUG: Failed to deliver buffered message {i+1}/{len(buffered_messages)} for task_id: {task_id}")
            
            # Clear the buffer after delivery
            self.pending_message_buffer[task_id] = []
            logger.info(f"✅ WS DEBUG: Cleared message buffer for task_id: {task_id}")
            
        except Exception as e:
            logger.error(f"Error delivering buffered messages for task_id {task_id}: {e}")
    
    async def _reduce_message_size(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce message size by removing or compressing frame data."""
        try:
            reduced_message = message.copy()
            
            # If this is a tracking update with camera frames, reduce frame data
            if "cameras" in reduced_message and isinstance(reduced_message["cameras"], dict):
                for camera_id, camera_data in reduced_message["cameras"].items():
                    if isinstance(camera_data, dict) and "frame_image" in camera_data:
                        # Remove frame images to reduce size (keep tracking data)
                        camera_data["frame_image"] = "removed_for_size"  # Keep key but remove data
                        camera_data["frame_size_reduced"] = True
                
                logger.info(f"🔧 WS DEBUG: Removed frame images to reduce message size")
            
            return reduced_message
            
        except Exception as e:
            logger.error(f"Error reducing message size: {e}")
            return message
    
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
            self.pending_message_buffer.clear()
            
            logger.info("BinaryWebSocketManager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global manager instance
binary_websocket_manager = BinaryWebSocketManager()