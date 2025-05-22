import logging
from typing import Dict, Any, List

from app.api.websockets import ConnectionManager
from app.api.v1.schemas import MediaURLEntry, WebSocketMediaAvailablePayload, WebSocketBatchProcessingCompletePayload

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

    async def send_tracking_update(self, task_id: str, update_payload_dict: Dict[str, Any]):
        """
        Sends tracking updates to all clients connected for a specific task.

        Args:
            task_id: The ID of the processing task.
            update_payload_dict: The dictionary containing the tracking data, matching WebSocketTrackingMessagePayload.
        """
        if not task_id:
            logger.warning("Attempted to send tracking update with empty task_id.")
            return

        message_to_send = {
            "type": "tracking_update",
            "payload": update_payload_dict
        }
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting tracking update for task {task_id}: {e}", exc_info=True)

    async def send_status_update(self, task_id: str, status_payload_dict: Dict[str, Any]):
        """
        Sends status updates (e.g., progress, current step) to clients for a task.

        Args:
            task_id: The ID of the processing task.
            status_payload_dict: The dictionary containing the status information, matching TaskStatusResponse.
        """
        if not task_id:
            logger.warning("Attempted to send status update with empty task_id.")
            return

        message_to_send = {
            "type": "status_update",
            "payload": status_payload_dict
        }
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting status update for task {task_id}: {e}", exc_info=True)

    async def send_media_available_notification(
        self,
        task_id: str,
        payload: WebSocketMediaAvailablePayload # Use the Pydantic model directly
    ):
        """
        Sends a notification that a new batch of sub-videos is available for streaming.

        Args:
            task_id: The ID of the processing task.
            payload: The WebSocketMediaAvailablePayload object.
        """
        if not task_id:
            logger.warning("Attempted to send media available notification with empty task_id.")
            return

        message_to_send = {
            "type": "media_available",
            "payload": payload.model_dump(exclude_none=True) # Convert Pydantic model to dict
        }
        logger.info(
            f"Broadcasting media_available for task {task_id}, "
            f"batch index {payload.sub_video_batch_index}, URLs: {len(payload.media_urls)}"
        )
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting media_available for task {task_id}: {e}", exc_info=True)

    async def send_batch_processing_complete_notification(
        self,
        task_id: str,
        payload: WebSocketBatchProcessingCompletePayload # Use the Pydantic model
    ):
        """
        Sends a notification that a sub-video batch processing is complete.

        Args:
            task_id: The ID of the processing task.
            payload: The WebSocketBatchProcessingCompletePayload object.
        """
        if not task_id:
            logger.warning("Attempted to send batch processing complete notification with empty task_id.")
            return

        message_to_send = {
            "type": "batch_processing_complete",
            "payload": payload.model_dump(exclude_none=True) # Convert Pydantic model to dict
        }
        logger.info(
            f"Broadcasting batch_processing_complete for task {task_id}, "
            f"batch index {payload.sub_video_batch_index}."
        )
        try:
            await self.manager.broadcast_to_task(task_id, message_to_send)
        except Exception as e:
            logger.error(f"Error broadcasting batch_processing_complete for task {task_id}: {e}", exc_info=True)