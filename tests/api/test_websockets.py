# tests/api/test_websockets.py
"""
Unit tests for WebSocket components in app.api.websockets.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from app.api.websockets import ConnectionManager, websocket_tracking_endpoint, manager as global_manager
from fastapi import WebSocket, WebSocketDisconnect # For type hinting and raising

@pytest.fixture
def connection_manager_instance():
    """Provides a clean ConnectionManager instance for each test."""
    manager = ConnectionManager()
    # Clear active connections from previous tests if any (though fixture should handle this)
    manager.active_connections.clear()
    return manager

@pytest.fixture
def mock_websocket(mocker):
    """Mocks a FastAPI WebSocket object."""
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.accept = AsyncMock()
    mock_ws.send_json = AsyncMock()
    mock_ws.receive_text = AsyncMock() # For the endpoint loop if it were to process client messages
    mock_ws.client = ("testclient", 12345) # Mock client host/port
    return mock_ws

# --- ConnectionManager Tests ---

@pytest.mark.asyncio
async def test_cm_connect_successful(connection_manager_instance: ConnectionManager, mock_websocket: AsyncMock):
    """Tests successful WebSocket connection and registration."""
    task_id = "task_abc"
    await connection_manager_instance.connect(mock_websocket, task_id)
    
    mock_websocket.accept.assert_called_once()
    assert task_id in connection_manager_instance.active_connections
    assert mock_websocket in connection_manager_instance.active_connections[task_id]

@pytest.mark.asyncio
async def test_cm_connect_accept_fails(connection_manager_instance: ConnectionManager, mock_websocket: AsyncMock, mocker):
    """Tests connect when websocket.accept() fails."""
    task_id = "task_fail_accept"
    mock_websocket.accept.side_effect = RuntimeError("Accept handshake failed")
    mock_logger_error = mocker.patch("app.api.websockets.logger.error")

    with pytest.raises(RuntimeError, match="Accept handshake failed"):
        await connection_manager_instance.connect(mock_websocket, task_id)
    
    assert task_id not in connection_manager_instance.active_connections
    mock_logger_error.assert_called_once()


@pytest.mark.asyncio
async def test_cm_disconnect_existing_connection(connection_manager_instance: ConnectionManager, mock_websocket: AsyncMock):
    """Tests disconnecting an existing WebSocket connection."""
    task_id = "task_def"
    await connection_manager_instance.connect(mock_websocket, task_id) # Connect first
    
    connection_manager_instance.disconnect(mock_websocket, task_id)
    
    assert task_id not in connection_manager_instance.active_connections # Task key removed if list becomes empty

@pytest.mark.asyncio
async def test_cm_disconnect_multiple_connections_for_task(connection_manager_instance: ConnectionManager, mocker):
    """Tests disconnecting one of multiple WebSockets for the same task."""
    task_id = "task_multi"
    ws1 = AsyncMock(spec=WebSocket); ws1.accept = AsyncMock(); ws1.client = ("c1",1)
    ws2 = AsyncMock(spec=WebSocket); ws2.accept = AsyncMock(); ws2.client = ("c2",2)

    await connection_manager_instance.connect(ws1, task_id)
    await connection_manager_instance.connect(ws2, task_id)
    assert len(connection_manager_instance.active_connections[task_id]) == 2

    connection_manager_instance.disconnect(ws1, task_id)
    assert task_id in connection_manager_instance.active_connections
    assert len(connection_manager_instance.active_connections[task_id]) == 1
    assert ws2 in connection_manager_instance.active_connections[task_id]
    assert ws1 not in connection_manager_instance.active_connections[task_id]


def test_cm_disconnect_non_existent_websocket(connection_manager_instance: ConnectionManager, mock_websocket: AsyncMock, mocker):
    """Tests disconnecting a WebSocket not in the list for a task."""
    task_id = "task_xyz"
    # Add a different websocket to the task
    other_ws = AsyncMock(spec=WebSocket); other_ws.client=("other",0)
    connection_manager_instance.active_connections[task_id] = [other_ws]
    mock_logger_warning = mocker.patch("app.api.websockets.logger.warning")

    connection_manager_instance.disconnect(mock_websocket, task_id) # mock_websocket was never added
    
    assert mock_websocket not in connection_manager_instance.active_connections[task_id]
    mock_logger_warning.assert_called_with(
        f"MANAGER: WebSocket client {mock_websocket.client} not found in active list "
        f"for task_id '{task_id}' during disconnect."
    )

def test_cm_disconnect_non_existent_task(connection_manager_instance: ConnectionManager, mock_websocket: AsyncMock, mocker):
    """Tests disconnecting from a task_id that doesn't exist."""
    mock_logger_warning = mocker.patch("app.api.websockets.logger.warning")
    connection_manager_instance.disconnect(mock_websocket, "non_existent_task")
    mock_logger_warning.assert_called_with(
         f"MANAGER: task_id 'non_existent_task' not found in active_connections "
         f"during disconnect for client {mock_websocket.client}."
    )


@pytest.mark.asyncio
async def test_cm_broadcast_to_task_successful(connection_manager_instance: ConnectionManager, mocker):
    """Tests successful broadcast to all WebSockets for a task."""
    task_id = "task_broadcast"
    ws1 = AsyncMock(spec=WebSocket); ws1.send_json = AsyncMock(); ws1.client=("c1",1)
    ws2 = AsyncMock(spec=WebSocket); ws2.send_json = AsyncMock(); ws2.client=("c2",2)
    
    # Manually add to active_connections for this test as connect involves accept()
    connection_manager_instance.active_connections[task_id] = [ws1, ws2]
    
    message = {"data": "hello world"}
    await connection_manager_instance.broadcast_to_task(task_id, message)
    
    ws1.send_json.assert_called_once_with(message)
    ws2.send_json.assert_called_once_with(message)

@pytest.mark.asyncio
async def test_cm_broadcast_to_task_one_fails_and_disconnects(connection_manager_instance: ConnectionManager, mocker):
    """Tests broadcast where one WebSocket send fails, leading to its disconnect."""
    task_id = "task_broadcast_fail"
    ws_ok = AsyncMock(spec=WebSocket); ws_ok.send_json = AsyncMock(); ws_ok.client=("ok",1)
    ws_fail = AsyncMock(spec=WebSocket); ws_fail.send_json = AsyncMock(side_effect=RuntimeError("Send failed")); ws_fail.client = ("fail",2)
    
    connection_manager_instance.active_connections[task_id] = [ws_ok, ws_fail]
    mock_logger_warning = mocker.patch("app.api.websockets.logger.warning")
    # Mock the disconnect method within the manager for this test, to verify it's called.
    # This is a bit of self-mocking, but useful to check internal calls if disconnect logic is complex.
    # Alternatively, check the state of active_connections after.
    mock_disconnect_method = mocker.patch.object(connection_manager_instance, "disconnect")

    message = {"data": "test"}
    await connection_manager_instance.broadcast_to_task(task_id, message)

    ws_ok.send_json.assert_called_once_with(message)
    ws_fail.send_json.assert_called_once_with(message) # Attempted send
    
    # Check that disconnect was called for ws_fail
    mock_disconnect_method.assert_called_once_with(ws_fail, task_id)
    mock_logger_warning.assert_called_with(
        f"MANAGER: RuntimeError sending to client {ws_fail.client} for task '{task_id}': Send failed. Marking for disconnect."
    )
    # Verify that ws_fail is actually removed (if not mocking disconnect method)
    # assert task_id in connection_manager_instance.active_connections
    # assert ws_fail not in connection_manager_instance.active_connections[task_id]
    # assert ws_ok in connection_manager_instance.active_connections[task_id]


# --- websocket_tracking_endpoint Tests ---
# These are more like integration tests for the endpoint logic itself.

@pytest.mark.asyncio
async def test_websocket_endpoint_connect_disconnect_flow(mock_websocket: AsyncMock, mocker):
    """Tests the typical connect-disconnect flow of the endpoint."""
    task_id = "endpoint_task_1"
    
    # Mock the global manager used by the endpoint
    mock_global_manager_instance = MagicMock(spec=ConnectionManager)
    mock_global_manager_instance.connect = AsyncMock()
    mock_global_manager_instance.disconnect = MagicMock() # Sync mock for disconnect in finally
    mocker.patch("app.api.websockets.manager", mock_global_manager_instance)

    # Simulate client disconnecting after some time
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()

    await websocket_tracking_endpoint(mock_websocket, task_id)

    mock_global_manager_instance.connect.assert_called_once_with(mock_websocket, task_id)
    mock_global_manager_instance.disconnect.assert_called_once_with(mock_websocket, task_id)

@pytest.mark.asyncio
async def test_websocket_endpoint_connect_fails_in_manager(mock_websocket: AsyncMock, mocker):
    """Tests when manager.connect itself raises an exception (e.g., accept failed)."""
    task_id = "endpoint_task_fail_connect"
    mock_global_manager_instance = MagicMock(spec=ConnectionManager)
    mock_global_manager_instance.connect = AsyncMock(side_effect=RuntimeError("Manager connect error"))
    mock_global_manager_instance.disconnect = MagicMock() # Disconnect should still be attempted if flag not set
    mocker.patch("app.api.websockets.manager", mock_global_manager_instance)
    mock_logger_error = mocker.patch("app.api.websockets.logger.error")

    await websocket_tracking_endpoint(mock_websocket, task_id)
    
    mock_global_manager_instance.connect.assert_called_once_with(mock_websocket, task_id)
    # Disconnect should NOT be called if connected_successfully flag was False
    mock_global_manager_instance.disconnect.assert_not_called()
    mock_logger_error.assert_called_once()
    assert "Manager connect error" in mock_logger_error.call_args[0][0]


@pytest.mark.asyncio
async def test_websocket_endpoint_unexpected_exception_in_loop(mock_websocket: AsyncMock, mocker):
    """Tests an unexpected error during the receive_text loop."""
    task_id = "endpoint_task_loop_error"
    mock_global_manager_instance = MagicMock(spec=ConnectionManager)
    mock_global_manager_instance.connect = AsyncMock()
    mock_global_manager_instance.disconnect = MagicMock()
    mocker.patch("app.api.websockets.manager", mock_global_manager_instance)
    mock_logger_error = mocker.patch("app.api.websockets.logger.error")

    # Simulate an error after successful connection, during receive_text
    mock_websocket.receive_text.side_effect = Exception("Unexpected loop error")

    await websocket_tracking_endpoint(mock_websocket, task_id)

    mock_global_manager_instance.connect.assert_called_once_with(mock_websocket, task_id)
    mock_global_manager_instance.disconnect.assert_called_once_with(mock_websocket, task_id) # Disconnect in finally
    mock_logger_error.assert_called_once()
    assert "Unexpected loop error" in mock_logger_error.call_args[0][0] 