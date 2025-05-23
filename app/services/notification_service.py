import logging
from typing import Dict, Any, List

from app.api.websockets import ConnectionManager
# Removed MediaURLEntry as it's no longer part of WebSocketMediaAvailablePayload
from app.api.v1.schemas import WebSocketBatchProcessingCompletePayload # WebSocketMediaAvailablePayload removed

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
        The payload now includes base64 encoded frame images.

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

    # The send_media_available_notification method is removed as per the new design.
    # async def send_media_available_notification(
    #     self,
    #     task_id: str,
    #     payload: WebSocketMediaAvailablePayload # Use the Pydantic model directly
    # ):
    #     """
    #     DEPRECATED: Sends a notification that a new batch of sub-videos is available for streaming.
    #     Frame images are now sent directly in 'tracking_update' messages.
    #     """
    #     # logger.warning(f"send_media_available_notification called for task {task_id}, but it's deprecated.")
    #     pass # Or raise an error, or log more prominently. For now, just pass.

    async def send_batch_processing_complete_notification(
        self,
        task_id: str,
        payload: WebSocketBatchProcessingCompletePayload # Use the Pydantic model
    ):
        """
        Sends a notification that a sub-video batch processing is complete.
        Its utility for frontend is reduced, but can be used for backend logic or advanced clients.

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