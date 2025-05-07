from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List
import asyncio # asyncio might be needed for more complex async operations within handlers

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        # task_id maps to a list of active WebSockets for that task
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        print(f"WebSocket connected for task_id {task_id}: {websocket.client}")

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
                if not self.active_connections[task_id]: # Remove task_id if no connections left
                    del self.active_connections[task_id]
        print(f"WebSocket disconnected for task_id {task_id}: {websocket.client}")

    async def broadcast_to_task(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            # Create a copy of the list for safe iteration if disconnections occur during broadcast
            connections_to_broadcast = list(self.active_connections.get(task_id, []))
            disconnected_sockets = []
            for connection in connections_to_broadcast:
                try:
                    await connection.send_json(message)
                except RuntimeError as e: # Can happen if client closed connection abruptly
                    print(f"RuntimeError sending to {connection.client} for task {task_id}: {e}")
                    disconnected_sockets.append(connection)
                except Exception as e:
                    print(f"Error sending message to {connection.client} for task {task_id}: {e}")
                    disconnected_sockets.append(connection) # Assume problematic
            
            for ws in disconnected_sockets:
                self.disconnect(ws, task_id) # Use the disconnect method for cleanup

# Global manager instance for simplicity. For larger apps, consider FastAPI's dependency injection for this.
manager = ConnectionManager()

# Example of how to use with Depends if you prefer:
# def get_connection_manager():
#     return manager

@router.websocket("/ws/tracking/{task_id}")
async def websocket_tracking_endpoint(websocket: WebSocket, task_id: str):
    # manager_instance = Depends(get_connection_manager) # If using dependency injection
    await manager.connect(websocket, task_id)
    try:
        while True:
            # This endpoint primarily listens for disconnects.
            # It can also handle incoming messages from the client if needed (e.g., control messages).
            data = await websocket.receive_text() # Or receive_json if client sends structured data
            print(f"Received from {task_id} client {websocket.client}: {data}")
            # Example: Handle client messages, e.g., to request focus on a track
            # await manager.broadcast_to_task(task_id, {"echo_from_server": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)
        print(f"WebSocket {websocket.client} for task {task_id} disconnected explicitly.")
    except Exception as e:
        # Catch any other exceptions during the WebSocket lifecycle for this client
        print(f"WebSocket error for task {task_id} client {websocket.client}: {e}")
        manager.disconnect(websocket, task_id) # Ensure cleanup
