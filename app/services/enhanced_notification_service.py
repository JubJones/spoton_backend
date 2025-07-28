"""
Enhanced notification service with binary WebSocket support.

Handles:
- Binary frame transmission notifications
- Tracking update notifications  
- System status notifications
- Performance-optimized messaging
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np

from app.api.websockets import (
    binary_websocket_manager,
    frame_handler,
    tracking_handler,
    status_handler,
    MessageType
)
from app.services.performance_monitor import performance_monitor
from app.services.network_optimizer import network_optimizer
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.mapping.entities.trajectory import Trajectory
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class EnhancedNotificationService:
    """
    Enhanced notification service with binary WebSocket support.
    
    Features:
    - Binary frame transmission
    - Real-time tracking updates
    - System status monitoring
    - Performance optimization
    """
    
    def __init__(self):
        # Service state
        self.active_subscriptions: Dict[str, List[str]] = {}  # task_id -> [subscription_types]
        self.notification_stats = {
            "total_notifications_sent": 0,
            "frame_notifications_sent": 0,
            "tracking_notifications_sent": 0,
            "status_notifications_sent": 0,
            "failed_notifications": 0
        }
        
        # Performance monitoring
        self.performance_enabled = True
        
        logger.info("EnhancedNotificationService initialized")
    
    async def initialize(self):
        """Initialize the notification service."""
        try:
            # Start performance monitoring
            if self.performance_enabled:
                await performance_monitor.start_monitoring()
                await network_optimizer.start_monitoring()
                await status_handler.start_monitoring()
            
            logger.info("EnhancedNotificationService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing EnhancedNotificationService: {e}")
            raise
    
    async def cleanup(self):
        """Clean up the notification service."""
        try:
            # Stop monitoring
            await performance_monitor.stop_monitoring()
            await network_optimizer.stop_monitoring()
            await status_handler.stop_monitoring()
            
            # Clear subscriptions
            self.active_subscriptions.clear()
            
            logger.info("EnhancedNotificationService cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up EnhancedNotificationService: {e}")
    
    async def send_binary_frame_notification(
        self,
        task_id: str,
        camera_frames: Dict[str, np.ndarray],
        frame_metadata: Dict[str, Any]
    ) -> bool:
        """
        Send binary frame notification with performance optimization.
        
        Args:
            task_id: Task identifier
            camera_frames: Frames from multiple cameras
            frame_metadata: Frame metadata
            
        Returns:
            True if sent successfully
        """
        try:
            # Check if frame should be skipped for performance
            if performance_monitor.should_skip_frame():
                performance_monitor.record_frame_processing(0.0, frame_dropped=True)
                return True
            
            # Send synchronized frames
            success = await frame_handler.send_synchronized_frames(
                task_id, camera_frames, frame_metadata
            )
            
            if success:
                self.notification_stats["frame_notifications_sent"] += 1
                logger.debug(f"Binary frame notification sent for task {task_id}")
            else:
                self.notification_stats["failed_notifications"] += 1
                logger.warning(f"Failed to send binary frame notification for task {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending binary frame notification: {e}")
            self.notification_stats["failed_notifications"] += 1
            return False
    
    async def send_tracking_update_notification(
        self,
        task_id: str,
        person_identity: PersonIdentity,
        current_position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None
    ) -> bool:
        """
        Send tracking update notification.
        
        Args:
            task_id: Task identifier
            person_identity: Person identity object
            current_position: Current position coordinate
            trajectory: Person trajectory
            
        Returns:
            True if sent successfully
        """
        try:
            # Send tracking update
            success = await tracking_handler.send_tracking_update(
                task_id, person_identity, current_position, trajectory
            )
            
            if success:
                self.notification_stats["tracking_notifications_sent"] += 1
                logger.debug(f"Tracking update notification sent for person {person_identity.global_id}")
            else:
                self.notification_stats["failed_notifications"] += 1
                logger.warning(f"Failed to send tracking update for person {person_identity.global_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending tracking update notification: {e}")
            self.notification_stats["failed_notifications"] += 1
            return False
    
    async def send_batch_tracking_updates(
        self,
        task_id: str,
        person_updates: List[Dict[str, Any]]
    ) -> bool:
        """
        Send batch tracking update notifications.
        
        Args:
            task_id: Task identifier
            person_updates: List of person update data
            
        Returns:
            True if sent successfully
        """
        try:
            # Send batch tracking updates
            success = await tracking_handler.send_batch_tracking_updates(
                task_id, person_updates
            )
            
            if success:
                self.notification_stats["tracking_notifications_sent"] += len(person_updates)
                logger.debug(f"Batch tracking updates sent for {len(person_updates)} persons")
            else:
                self.notification_stats["failed_notifications"] += 1
                logger.warning(f"Failed to send batch tracking updates for task {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending batch tracking updates: {e}")
            self.notification_stats["failed_notifications"] += 1
            return False
    
    async def send_status_update_notification(
        self,
        task_id: str,
        status_data: Dict[str, Any]
    ) -> bool:
        """
        Send status update notification.
        
        Args:
            task_id: Task identifier
            status_data: Status data
            
        Returns:
            True if sent successfully
        """
        try:
            # Send status update
            success = await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "status_update",
                    "data": status_data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                MessageType.CONTROL_MESSAGE
            )
            
            if success:
                self.notification_stats["status_notifications_sent"] += 1
                logger.debug(f"Status update notification sent for task {task_id}")
            else:
                self.notification_stats["failed_notifications"] += 1
                logger.warning(f"Failed to send status update for task {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending status update notification: {e}")
            self.notification_stats["failed_notifications"] += 1
            return False
    
    async def send_system_status_notification(
        self,
        status_data: Dict[str, Any]
    ) -> bool:
        """
        Send system status notification to all connections.
        
        Args:
            status_data: System status data
            
        Returns:
            True if sent successfully
        """
        try:
            # Broadcast system status
            await binary_websocket_manager.broadcast_system_status(status_data)
            
            self.notification_stats["status_notifications_sent"] += 1
            logger.debug("System status notification broadcast")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending system status notification: {e}")
            self.notification_stats["failed_notifications"] += 1
            return False
    
    async def send_performance_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning"
    ) -> bool:
        """
        Send performance alert notification.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
            
        Returns:
            True if sent successfully
        """
        try:
            alert_data = {
                "type": "performance_alert",
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Broadcast to all connections
            await binary_websocket_manager.broadcast_system_status(alert_data)
            
            logger.info(f"Performance alert sent: {alert_type} - {message}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
            return False
    
    async def subscribe_to_notifications(
        self,
        task_id: str,
        subscription_types: List[str]
    ) -> bool:
        """
        Subscribe to notification types.
        
        Args:
            task_id: Task identifier
            subscription_types: List of subscription types
            
        Returns:
            True if subscribed successfully
        """
        try:
            self.active_subscriptions[task_id] = subscription_types
            
            logger.info(f"Subscribed to notifications for task {task_id}: {subscription_types}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to notifications: {e}")
            return False
    
    async def unsubscribe_from_notifications(
        self,
        task_id: str,
        subscription_types: Optional[List[str]] = None
    ) -> bool:
        """
        Unsubscribe from notification types.
        
        Args:
            task_id: Task identifier
            subscription_types: List of subscription types to unsubscribe from
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            if subscription_types is None:
                # Unsubscribe from all
                if task_id in self.active_subscriptions:
                    del self.active_subscriptions[task_id]
            else:
                # Unsubscribe from specific types
                if task_id in self.active_subscriptions:
                    current_subs = self.active_subscriptions[task_id]
                    self.active_subscriptions[task_id] = [
                        sub for sub in current_subs if sub not in subscription_types
                    ]
            
            logger.info(f"Unsubscribed from notifications for task {task_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from notifications: {e}")
            return False
    
    def is_subscribed(self, task_id: str, notification_type: str) -> bool:
        """
        Check if task is subscribed to notification type.
        
        Args:
            task_id: Task identifier
            notification_type: Notification type
            
        Returns:
            True if subscribed
        """
        try:
            return (task_id in self.active_subscriptions and 
                    notification_type in self.active_subscriptions[task_id])
            
        except Exception as e:
            logger.error(f"Error checking subscription: {e}")
            return False
    
    async def send_connection_status(
        self,
        task_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send connection status notification.
        
        Args:
            task_id: Task identifier
            status: Connection status
            details: Additional details
            
        Returns:
            True if sent successfully
        """
        try:
            status_data = {
                "type": "connection_status",
                "status": status,
                "details": details or {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            success = await binary_websocket_manager.send_json_message(
                task_id, status_data, MessageType.CONTROL_MESSAGE
            )
            
            if success:
                logger.debug(f"Connection status sent for task {task_id}: {status}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending connection status: {e}")
            return False
    
    async def flush_pending_notifications(self):
        """Flush all pending notifications."""
        try:
            # Flush WebSocket batches
            await binary_websocket_manager.flush_all_batches()
            
            logger.debug("Flushed pending notifications")
            
        except Exception as e:
            logger.error(f"Error flushing pending notifications: {e}")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        try:
            return {
                **self.notification_stats,
                "total_notifications": sum([
                    self.notification_stats["frame_notifications_sent"],
                    self.notification_stats["tracking_notifications_sent"],
                    self.notification_stats["status_notifications_sent"]
                ]),
                "active_subscriptions": len(self.active_subscriptions),
                "websocket_stats": binary_websocket_manager.get_performance_stats(),
                "performance_stats": performance_monitor.get_performance_metrics(),
                "network_stats": network_optimizer.get_network_stats(),
                "success_rate": (
                    self.notification_stats["total_notifications_sent"] /
                    max(1, self.notification_stats["total_notifications_sent"] + 
                        self.notification_stats["failed_notifications"])
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return {"error": str(e)}
    
    def reset_stats(self):
        """Reset notification statistics."""
        try:
            self.notification_stats = {
                "total_notifications_sent": 0,
                "frame_notifications_sent": 0,
                "tracking_notifications_sent": 0,
                "status_notifications_sent": 0,
                "failed_notifications": 0
            }
            
            logger.info("Notification statistics reset")
            
        except Exception as e:
            logger.error(f"Error resetting notification stats: {e}")
    
    def enable_performance_monitoring(self, enabled: bool):
        """Enable or disable performance monitoring."""
        try:
            self.performance_enabled = enabled
            logger.info(f"Performance monitoring {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Error setting performance monitoring: {e}")


# Global enhanced notification service instance
enhanced_notification_service = EnhancedNotificationService()