"""
Enhanced Visualization Notification Service

Provides comprehensive WebSocket messaging for frontend visualization including:
- Multi-camera synchronized tracking updates
- Focus tracking notifications
- Real-time analytics streaming
- Performance monitoring
- Adaptive quality control
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
from collections import deque, defaultdict
from dataclasses import dataclass, field

from app.api.websockets.connection_manager import ConnectionManager
from app.api.v1.visualization_schemas import (
    MultiCameraVisualizationUpdate,
    FocusTrackState,
    LiveAnalyticsData,
    PlaybackControls,
    EnhancedTrackingUpdate,
    AnalyticsUpdateMessage,
    create_tracking_update_message
)
from app.domains.visualization.entities.visual_frame import VisualFrame

logger = logging.getLogger(__name__)


@dataclass
class NotificationMetrics:
    """Metrics for notification performance tracking."""
    messages_sent: int = 0
    bytes_transmitted: int = 0
    average_latency_ms: float = 0.0
    error_count: int = 0
    connection_count: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)
    peak_connections: int = 0
    
    def update_metrics(self, message_size: int, latency_ms: float, error: bool = False):
        """Update notification metrics."""
        self.messages_sent += 1
        self.bytes_transmitted += message_size
        
        # Update average latency using exponential moving average
        alpha = 0.1
        self.average_latency_ms = (alpha * latency_ms + 
                                  (1 - alpha) * self.average_latency_ms)
        
        if error:
            self.error_count += 1
            
        self.last_update = datetime.utcnow()


@dataclass
class QualityControlSettings:
    """Adaptive quality control settings."""
    max_fps: float = 30.0
    min_fps: float = 5.0
    target_latency_ms: float = 100.0
    max_latency_ms: float = 500.0
    
    # Frame quality
    high_quality: int = 95
    medium_quality: int = 80
    low_quality: int = 65
    
    # Compression settings
    enable_compression: bool = True
    compression_threshold_kb: int = 100
    
    # Adaptive behavior
    auto_adjust_quality: bool = True
    auto_adjust_fps: bool = True


class VisualizationNotificationService:
    """Enhanced notification service for frontend visualization."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        
        # Metrics and monitoring
        self.metrics = NotificationMetrics()
        self.quality_settings = QualityControlSettings()
        
        # Message queues for different types
        self.tracking_queue = asyncio.Queue(maxsize=1000)
        self.focus_queue = asyncio.Queue(maxsize=100)
        self.analytics_queue = asyncio.Queue(maxsize=500)
        
        # Active subscriptions by task_id
        self.tracking_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # task_id -> {connection_ids}
        self.focus_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.analytics_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.latency_history: deque = deque(maxlen=100)
        self.fps_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # State management
        self.current_focus_states: Dict[str, FocusTrackState] = {}
        self.last_tracking_updates: Dict[str, MultiCameraVisualizationUpdate] = {}
        
        # Background task references
        self._notification_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        logger.info("VisualizationNotificationService initialized")
    
    async def start_service(self):
        """Start the notification service background tasks."""
        if self._running:
            logger.warning("Notification service already running")
            return
        
        self._running = True
        
        # Start background processing tasks
        tasks = [
            asyncio.create_task(self._process_tracking_queue()),
            asyncio.create_task(self._process_focus_queue()),
            asyncio.create_task(self._process_analytics_queue()),
            asyncio.create_task(self._monitor_performance()),
            asyncio.create_task(self._adaptive_quality_control())
        ]
        
        self._notification_tasks.update(tasks)
        logger.info(f"Started {len(tasks)} notification service background tasks")
    
    async def stop_service(self):
        """Stop the notification service and cleanup."""
        self._running = False
        
        # Cancel all background tasks
        for task in self._notification_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._notification_tasks:
            await asyncio.gather(*self._notification_tasks, return_exceptions=True)
        
        self._notification_tasks.clear()
        logger.info("Visualization notification service stopped")
    
    # --- Tracking Updates ---
    
    async def send_tracking_update(
        self,
        task_id: str,
        update_data: MultiCameraVisualizationUpdate
    ) -> bool:
        """Send enhanced tracking update to all subscribers."""
        try:
            # Store latest update
            self.last_tracking_updates[task_id] = update_data
            
            # Queue the update for processing
            await self.tracking_queue.put({
                'task_id': task_id,
                'data': update_data,
                'timestamp': time.time()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error queuing tracking update for task {task_id}: {e}")
            return False
    
    async def _process_tracking_queue(self):
        """Background task to process tracking updates."""
        while self._running:
            try:
                # Get update from queue with timeout
                update_item = await asyncio.wait_for(
                    self.tracking_queue.get(), 
                    timeout=1.0
                )
                
                await self._send_tracking_update_to_subscribers(update_item)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing tracking queue: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_tracking_update_to_subscribers(self, update_item: Dict[str, Any]):
        """Send tracking update to all task subscribers."""
        task_id = update_item['task_id']
        update_data = update_item['data']
        queue_time = update_item['timestamp']
        
        # Get current focus state
        focus_state = self.current_focus_states.get(task_id)
        
        # Create WebSocket message
        message = create_tracking_update_message(update_data, focus_state)
        
        # Get subscribers for this task
        subscribers = self.tracking_subscriptions.get(task_id, set())
        
        if not subscribers:
            return
        
        # Send to all subscribers
        start_time = time.time()
        message_json = json.dumps(message)
        message_size = len(message_json.encode('utf-8'))
        
        successful_sends = 0
        failed_sends = 0
        
        for connection_id in subscribers.copy():  # Copy to avoid modification during iteration
            try:
                await self.connection_manager.send_json_to_connection(
                    connection_id, message
                )
                successful_sends += 1
                
            except Exception as e:
                logger.warning(f"Failed to send tracking update to connection {connection_id}: {e}")
                failed_sends += 1
                
                # Remove failed connection
                subscribers.discard(connection_id)
        
        # Update metrics
        processing_latency = (time.time() - queue_time) * 1000  # ms
        self.latency_history.append(processing_latency)
        self.metrics.update_metrics(
            message_size, processing_latency, error=(failed_sends > 0)
        )
        
        # Update FPS tracking
        current_time = time.time()
        self.fps_history[task_id].append(current_time)
        
        if successful_sends > 0:
            logger.debug(
                f"Sent tracking update to {successful_sends} subscribers for task {task_id} "
                f"(latency: {processing_latency:.1f}ms)"
            )
    
    # --- Focus Tracking ---
    
    async def send_focus_update(
        self,
        task_id: str,
        focus_state: FocusTrackState
    ) -> bool:
        """Send focus tracking update."""
        try:
            # Store current focus state
            self.current_focus_states[task_id] = focus_state
            
            # Queue the focus update
            await self.focus_queue.put({
                'task_id': task_id,
                'focus_state': focus_state,
                'timestamp': time.time()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error queuing focus update for task {task_id}: {e}")
            return False
    
    async def _process_focus_queue(self):
        """Background task to process focus updates."""
        while self._running:
            try:
                # Get focus update from queue
                focus_item = await asyncio.wait_for(
                    self.focus_queue.get(),
                    timeout=1.0
                )
                
                await self._send_focus_update_to_subscribers(focus_item)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing focus queue: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_focus_update_to_subscribers(self, focus_item: Dict[str, Any]):
        """Send focus update to subscribers."""
        task_id = focus_item['task_id']
        focus_state = focus_item['focus_state']
        
        # Create focus update message
        message = {
            "type": "focus_update",
            "payload": {
                "focused_person_id": focus_state.focused_person_id,
                "focus_mode": focus_state.focus_mode.value,
                "focus_start_time": (
                    focus_state.focus_start_time.isoformat() 
                    if focus_state.focus_start_time else None
                ),
                "highlight_color": focus_state.highlight_color,
                "cross_camera_sync": focus_state.cross_camera_sync,
                "show_trajectory": focus_state.show_trajectory,
                "auto_follow": focus_state.auto_follow,
                "person_details": (
                    focus_state.person_details.dict() 
                    if focus_state.person_details else None
                )
            }
        }
        
        # Send to focus subscribers
        subscribers = self.focus_subscriptions.get(task_id, set())
        
        for connection_id in subscribers.copy():
            try:
                await self.connection_manager.send_json_to_connection(
                    connection_id, message
                )
                
            except Exception as e:
                logger.warning(f"Failed to send focus update to connection {connection_id}: {e}")
                subscribers.discard(connection_id)
    
    # --- Analytics Updates ---
    
    async def send_analytics_update(
        self,
        environment_id: str,
        analytics_data: LiveAnalyticsData
    ) -> bool:
        """Send real-time analytics update."""
        try:
            await self.analytics_queue.put({
                'environment_id': environment_id,
                'data': analytics_data,
                'timestamp': time.time()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error queuing analytics update: {e}")
            return False
    
    async def _process_analytics_queue(self):
        """Background task to process analytics updates."""
        while self._running:
            try:
                # Get analytics update from queue
                analytics_item = await asyncio.wait_for(
                    self.analytics_queue.get(),
                    timeout=1.0
                )
                
                await self._send_analytics_update_to_subscribers(analytics_item)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing analytics queue: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_analytics_update_to_subscribers(self, analytics_item: Dict[str, Any]):
        """Send analytics update to subscribers."""
        environment_id = analytics_item['environment_id']
        analytics_data = analytics_item['data']
        
        # Create analytics update message
        message = AnalyticsUpdateMessage(
            payload=analytics_data
        ).dict()
        
        # Send to all analytics subscribers
        # Note: For now, sending to all analytics subscribers regardless of environment
        # Could be filtered by environment_id if needed
        
        all_analytics_subscribers = set()
        for subscribers in self.analytics_subscriptions.values():
            all_analytics_subscribers.update(subscribers)
        
        for connection_id in all_analytics_subscribers.copy():
            try:
                await self.connection_manager.send_json_to_connection(
                    connection_id, message
                )
                
            except Exception as e:
                logger.warning(f"Failed to send analytics update to connection {connection_id}: {e}")
                # Remove from all subscriber sets
                for subscribers in self.analytics_subscriptions.values():
                    subscribers.discard(connection_id)
    
    # --- Subscription Management ---
    
    def subscribe_to_tracking(self, task_id: str, connection_id: str) -> bool:
        """Subscribe connection to tracking updates for a task."""
        try:
            self.tracking_subscriptions[task_id].add(connection_id)
            logger.info(f"Connection {connection_id} subscribed to tracking for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to tracking: {e}")
            return False
    
    def unsubscribe_from_tracking(self, task_id: str, connection_id: str) -> bool:
        """Unsubscribe connection from tracking updates."""
        try:
            self.tracking_subscriptions[task_id].discard(connection_id)
            logger.info(f"Connection {connection_id} unsubscribed from tracking for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error unsubscribing from tracking: {e}")
            return False
    
    def subscribe_to_focus(self, task_id: str, connection_id: str) -> bool:
        """Subscribe connection to focus updates for a task."""
        try:
            self.focus_subscriptions[task_id].add(connection_id)
            logger.info(f"Connection {connection_id} subscribed to focus for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to focus: {e}")
            return False
    
    def unsubscribe_from_focus(self, task_id: str, connection_id: str) -> bool:
        """Unsubscribe connection from focus updates."""
        try:
            self.focus_subscriptions[task_id].discard(connection_id)
            logger.info(f"Connection {connection_id} unsubscribed from focus for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error unsubscribing from focus: {e}")
            return False
    
    def subscribe_to_analytics(self, task_id: str, connection_id: str) -> bool:
        """Subscribe connection to analytics updates."""
        try:
            self.analytics_subscriptions[task_id].add(connection_id)
            logger.info(f"Connection {connection_id} subscribed to analytics for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to analytics: {e}")
            return False
    
    def unsubscribe_from_analytics(self, task_id: str, connection_id: str) -> bool:
        """Unsubscribe connection from analytics updates."""
        try:
            self.analytics_subscriptions[task_id].discard(connection_id)
            logger.info(f"Connection {connection_id} unsubscribed from analytics for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error unsubscribing from analytics: {e}")
            return False
    
    def cleanup_connection_subscriptions(self, connection_id: str):
        """Remove connection from all subscriptions."""
        # Remove from tracking subscriptions
        for subscribers in self.tracking_subscriptions.values():
            subscribers.discard(connection_id)
        
        # Remove from focus subscriptions
        for subscribers in self.focus_subscriptions.values():
            subscribers.discard(connection_id)
        
        # Remove from analytics subscriptions
        for subscribers in self.analytics_subscriptions.values():
            subscribers.discard(connection_id)
        
        logger.info(f"Cleaned up all subscriptions for connection {connection_id}")
    
    # --- Performance Monitoring ---
    
    async def _monitor_performance(self):
        """Background task to monitor performance metrics."""
        while self._running:
            try:
                # Update connection count
                self.metrics.connection_count = len(self.connection_manager.get_all_connections())
                self.metrics.peak_connections = max(
                    self.metrics.peak_connections, 
                    self.metrics.connection_count
                )
                
                # Calculate current FPS for each task
                current_time = time.time()
                for task_id, timestamps in self.fps_history.items():
                    # Remove timestamps older than 1 second
                    while timestamps and (current_time - timestamps[0]) > 1.0:
                        timestamps.popleft()
                
                # Log performance metrics periodically
                if int(current_time) % 30 == 0:  # Every 30 seconds
                    await self._log_performance_metrics()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5.0)
    
    async def _adaptive_quality_control(self):
        """Background task for adaptive quality control."""
        while self._running:
            try:
                # Check average latency
                if len(self.latency_history) > 10:
                    avg_latency = sum(self.latency_history) / len(self.latency_history)
                    
                    # Adjust quality based on latency
                    if avg_latency > self.quality_settings.max_latency_ms:
                        # Reduce quality
                        self.quality_settings.high_quality = max(60, self.quality_settings.high_quality - 5)
                        logger.info(f"Reduced quality due to high latency: {avg_latency:.1f}ms")
                    
                    elif avg_latency < self.quality_settings.target_latency_ms:
                        # Increase quality
                        self.quality_settings.high_quality = min(95, self.quality_settings.high_quality + 2)
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in adaptive quality control: {e}")
                await asyncio.sleep(10.0)
    
    async def _log_performance_metrics(self):
        """Log current performance metrics."""
        avg_latency = (
            sum(self.latency_history) / len(self.latency_history)
            if self.latency_history else 0.0
        )
        
        # Calculate current FPS
        current_fps = {}
        current_time = time.time()
        for task_id, timestamps in self.fps_history.items():
            fps = len([t for t in timestamps if (current_time - t) <= 1.0])
            current_fps[task_id] = fps
        
        logger.info(
            f"Notification service metrics - "
            f"Connections: {self.metrics.connection_count}, "
            f"Messages sent: {self.metrics.messages_sent}, "
            f"Avg latency: {avg_latency:.1f}ms, "
            f"Error rate: {self.metrics.error_count / max(1, self.metrics.messages_sent) * 100:.1f}%, "
            f"FPS: {current_fps}"
        )
    
    # --- Status and Diagnostics ---
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        current_time = time.time()
        
        # Calculate current FPS
        current_fps = {}
        for task_id, timestamps in self.fps_history.items():
            fps = len([t for t in timestamps if (current_time - t) <= 1.0])
            current_fps[task_id] = fps
        
        # Calculate average latency
        avg_latency = (
            sum(self.latency_history) / len(self.latency_history)
            if self.latency_history else 0.0
        )
        
        return {
            "service_running": self._running,
            "background_tasks": len(self._notification_tasks),
            "metrics": {
                "messages_sent": self.metrics.messages_sent,
                "bytes_transmitted": self.metrics.bytes_transmitted,
                "average_latency_ms": avg_latency,
                "error_count": self.metrics.error_count,
                "error_rate_percent": (
                    self.metrics.error_count / max(1, self.metrics.messages_sent) * 100
                ),
                "connection_count": self.metrics.connection_count,
                "peak_connections": self.metrics.peak_connections
            },
            "subscriptions": {
                "tracking_tasks": len(self.tracking_subscriptions),
                "focus_tasks": len(self.focus_subscriptions),
                "analytics_tasks": len(self.analytics_subscriptions),
                "total_tracking_subscribers": sum(
                    len(subs) for subs in self.tracking_subscriptions.values()
                ),
                "total_focus_subscribers": sum(
                    len(subs) for subs in self.focus_subscriptions.values()
                ),
                "total_analytics_subscribers": sum(
                    len(subs) for subs in self.analytics_subscriptions.values()
                )
            },
            "performance": {
                "current_fps": current_fps,
                "queue_sizes": {
                    "tracking": self.tracking_queue.qsize(),
                    "focus": self.focus_queue.qsize(),
                    "analytics": self.analytics_queue.qsize()
                },
                "quality_settings": {
                    "high_quality": self.quality_settings.high_quality,
                    "auto_adjust_enabled": self.quality_settings.auto_adjust_quality
                }
            },
            "last_update": self.metrics.last_update.isoformat()
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = NotificationMetrics()
        self.latency_history.clear()
        for fps_history in self.fps_history.values():
            fps_history.clear()
        logger.info("Notification service metrics reset")


# Global service instance
_notification_service: Optional[VisualizationNotificationService] = None


def get_notification_service() -> Optional[VisualizationNotificationService]:
    """Get the global notification service instance."""
    return _notification_service


def initialize_notification_service(connection_manager: ConnectionManager) -> VisualizationNotificationService:
    """Initialize the global notification service."""
    global _notification_service
    if _notification_service is None:
        _notification_service = VisualizationNotificationService(connection_manager)
    return _notification_service