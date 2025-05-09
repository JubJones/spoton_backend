import logging
from typing import Dict, Any

from app.api.websockets import ConnectionManager

logger = logging.getLogger(__name__)

class NotificationService:
    """Service responsible for sending notifications via WebSockets."""

    def __init__(self, manager: ConnectionManager):
        """
        Initializes the NotificationService.

        Args:
            manager: An instance of ConnectionManager to handle WebSocket connections.
        """
        self.manager = manager
        logger.info("NotificationService initialized.")

    async def send_tracking_update(self, task_id: str, update_payload: Dict[str, Any]):
        """
        Sends tracking updates to all clients connected for a specific task.

        Args:
            task_id: The ID of the processing task.
            update_payload: The dictionary containing the tracking data for the frame
                            (e.g., camera_id, timestamp, frame_path, tracking_data list).
        """
        if not task_id:
            logger.warning("Attempted to send tracking update with empty task_id.")
            return

        message_to_send = {
            "type": "tracking_update",
            "payload": update_payload
        }
        # logger.debug(f"Broadcasting tracking update for task {task_id}")
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting tracking update for task {task_id}: {e}", exc_info=True)

    async def send_status_update(self, task_id: str, status_payload: Dict[str, Any]):
        """
        Sends status updates (e.g., progress, current step) to clients for a task.

        Args:
            task_id: The ID of the processing task.
            status_payload: The dictionary containing the status information
                            (e.g., status, progress, current_step, details).
        """
        if not task_id:
            logger.warning("Attempted to send status update with empty task_id.")
            return

        message_to_send = {
            "type": "status_update",
            "payload": status_payload # Send the raw status dict as payload
        }
        # logger.info(f"Broadcasting status update for task {task_id}: {status_payload.get('status')} - {status_payload.get('current_step')}")
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting status update for task {task_id}: {e}", exc_info=True)