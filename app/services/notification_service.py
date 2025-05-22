import logging
from typing import Dict, Any, List

from app.api.websockets import ConnectionManager
from app.api.v1.schemas import MediaURLEntry

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
            update_payload: The dictionary containing the tracking data for the frame.
        """
        if not task_id:
            logger.warning("Attempted to send tracking update with empty task_id.")
            return

        message_to_send = {
            "type": "tracking_update",
            "payload": update_payload
        }
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting tracking update for task {task_id}: {e}", exc_info=True)

    async def send_status_update(self, task_id: str, status_payload: Dict[str, Any]):
        """
        Sends status updates (e.g., progress, current step) to clients for a task.

        Args:
            task_id: The ID of the processing task.
            status_payload: The dictionary containing the status information.
        """
        if not task_id:
            logger.warning("Attempted to send status update with empty task_id.")
            return

        message_to_send = {
            "type": "status_update",
            "payload": status_payload
        }
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting status update for task {task_id}: {e}", exc_info=True)

    async def send_media_available_notification(
        self,
        task_id: str,
        sub_video_batch_index: int,
        media_entries: List[MediaURLEntry]
    ):
        """
        Sends a notification that a new batch of sub-videos is available for streaming.

        Args:
            task_id: The ID of the processing task.
            sub_video_batch_index: The 0-indexed identifier for this batch of sub-videos.
            media_entries: A list of MediaURLEntry objects, each containing camera_id and its backend URL.
        """
        if not task_id:
            logger.warning("Attempted to send media available notification with empty task_id.")
            return

        # Convert MediaURLEntry objects to dictionaries for JSON serialization
        media_urls_payload = [entry.model_dump() for entry in media_entries]

        message_to_send = {
            "type": "media_available",
            "payload": {
                "sub_video_batch_index": sub_video_batch_index,
                "media_urls": media_urls_payload
            }
        }
        logger.info(f"Broadcasting media_available for task {task_id}, batch index {sub_video_batch_index}, URLs: {len(media_urls_payload)}")
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting media_available for task {task_id}: {e}", exc_info=True)