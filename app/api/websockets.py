# FILE: app/api/websockets.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import asyncio
import logging # Import the logging module

# Initialize logger for this module.
# Logs will be handled by the basicConfig in main.py or Uvicorn's logger.
logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    """Manages active WebSocket connections for different tasks."""
    def __init__(self):
        """Initializes the ConnectionManager."""
        # task_id maps to a list of active WebSockets for that task
        self.active_connections: Dict[str, List[WebSocket]] = {}
        logger.info("ConnectionManager initialized.")

    async def connect(self, websocket: WebSocket, task_id: str):
        """
        Accepts a new WebSocket connection and stores it for the given task_id.

        Args:
            websocket: The WebSocket connection object.
            task_id: The identifier of the task this WebSocket is associated with.

        Raises:
            Exception: If `websocket.accept()` fails.
        """
        logger.info(
            f"MANAGER: Attempting to accept WebSocket connection for task_id '{task_id}', client {websocket.client}"
        )
        try:
            await websocket.accept()  # Critical call for handshake completion
            logger.info(
                f"MANAGER: WebSocket accepted successfully for task_id '{task_id}', client {websocket.client}"
            )
        except Exception as e_accept:
            logger.error(
                f"MANAGER: Error during websocket.accept() for task_id '{task_id}', client {websocket.client}: {e_accept}",
                exc_info=True
            )
            # If accept fails, the connection should not be added or considered active.
            # The endpoint handler's error handling should manage this.
            raise  # Re-raise to let the endpoint handler know accept failed.

        # Proceed only if accept was successful
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logger.info(
            f"MANAGER: WebSocket stored for task_id '{task_id}', client {websocket.client}. "
            f"Total connections for this task: {len(self.active_connections[task_id])}"
        )

    def disconnect(self, websocket: WebSocket, task_id: str):
        """
        Removes a WebSocket connection from the active list for a task_id.

        Args:
            websocket: The WebSocket connection object to remove.
            task_id: The identifier of the task.
        """
        logger.info(
            f"MANAGER: Disconnecting WebSocket for task_id '{task_id}', client {websocket.client}"
        )
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
                logger.debug(
                    f"MANAGER: Removed WebSocket from task_id '{task_id}'. "
                    f"Sockets remaining for task: {len(self.active_connections[task_id])}"
                )
                if not self.active_connections[task_id]:  # No connections left for this task
                    del self.active_connections[task_id]
                    logger.debug(
                        f"MANAGER: Removed task_id '{task_id}' from active_connections as it has no more sockets."
                    )
            else:
                logger.warning(
                    f"MANAGER: WebSocket client {websocket.client} not found in active list "
                    f"for task_id '{task_id}' during disconnect."
                )
        else:
            logger.warning(
                f"MANAGER: task_id '{task_id}' not found in active_connections "
                f"during disconnect for client {websocket.client}."
            )

    async def broadcast_to_task(self, task_id: str, message: dict):
        """
        Broadcasts a JSON message to all WebSockets connected for a specific task_id.

        Args:
            task_id: The identifier of the task whose clients should receive the message.
            message: The JSON serializable dictionary to send.
        """
        if task_id in self.active_connections:
            # Create a copy of the list for safe iteration if disconnections occur during broadcast
            connections_to_broadcast = list(self.active_connections.get(task_id, []))
            disconnected_sockets = []
            # logger.debug(f"MANAGER: Broadcasting to {len(connections_to_broadcast)} client(s) for task '{task_id}'")
            for connection in connections_to_broadcast:
                try:
                    await connection.send_json(message)
                except RuntimeError as e_runtime:  # Can happen if client closed connection abruptly
                    logger.warning(
                        f"MANAGER: RuntimeError sending to client {connection.client} for task '{task_id}': {e_runtime}. Marking for disconnect."
                    )
                    disconnected_sockets.append(connection)
                except Exception as e_send: # Catch other send errors
                    logger.error(
                        f"MANAGER: Error sending message to client {connection.client} for task '{task_id}': {e_send}",
                        exc_info=True
                    )
                    disconnected_sockets.append(connection)  # Assume problematic, mark for disconnect

            # Clean up any sockets that failed during broadcast
            for ws_to_remove in disconnected_sockets:
                self.disconnect(ws_to_remove, task_id)


# Global manager instance. For larger apps, consider FastAPI's dependency injection.
manager = ConnectionManager()

@router.websocket("/tracking/{task_id}") # Path relative to the router's prefix in main.py ('/ws')
async def websocket_tracking_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time tracking updates for a given task_id.
    Clients connect to `/ws/tracking/{task_id}`.
    """
    logger.info(
        f"ENDPOINT: WebSocket connection attempt received for task_id: '{task_id}', client: {websocket.client}"
    )
    connected_successfully = False # Flag to track if manager.connect succeeded
    try:
        await manager.connect(websocket, task_id)
        connected_successfully = True
        logger.info(
            f"ENDPOINT: WebSocket connection established and registered for task_id: '{task_id}', client: {websocket.client}"
        )

        # Main loop to keep the connection alive and handle incoming messages (if any)
        while True:
            # This endpoint primarily listens for disconnects or server-pushed messages.
            # If client needs to send control messages, they would be received here.
            data = await websocket.receive_text()  # Or receive_json
            logger.debug(
                f"ENDPOINT: Received data from task_id '{task_id}', client {websocket.client}: {data}"
            )
            # Example: Handle client messages if needed
            # await manager.broadcast_to_task(task_id, {"echo_from_server": data})

    except WebSocketDisconnect:
        # This is a graceful disconnect initiated by the client or server (e.g. on normal task completion)
        logger.info(
            f"ENDPOINT: WebSocket client {websocket.client} (task '{task_id}') disconnected gracefully (WebSocketDisconnect)."
        )
    except Exception as e:
        # This catches errors from manager.connect (if accept failed) or within the receive loop
        logger.error(
            f"ENDPOINT: Exception for WebSocket task_id '{task_id}', client {websocket.client}: {e}",
            exc_info=True # Include traceback for unexpected errors
        )
    finally:
        logger.info(
            f"ENDPOINT: Cleaning up WebSocket connection for task_id '{task_id}', client {websocket.client}. "
            f"Was connection previously successful: {connected_successfully}"
        )
        # Ensure disconnect is called only if the connection was successfully registered with the manager
        if connected_successfully:
            manager.disconnect(websocket, task_id)
        logger.info(
            f"ENDPOINT: Finished handling WebSocket connection for task_id '{task_id}', client {websocket.client}"
        )