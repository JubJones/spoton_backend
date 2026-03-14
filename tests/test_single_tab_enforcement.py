
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from starlette.websockets import WebSocketState
from app.api.websockets.connection_manager import BinaryWebSocketManager

@pytest.fixture
def manager():
    return BinaryWebSocketManager()

@pytest.fixture
def mock_websocket():
    ws = AsyncMock()
    ws.client_state.value = 1 # CONNECTED
    return ws

@pytest.mark.asyncio
async def test_single_tab_enforcement(manager):
    task_id = "task_123"
    tab_id_1 = "tab_1"
    tab_id_2 = "tab_2"

    # Mock WebSockets
    ws1 = AsyncMock()
    ws1.client_state.value = 1 # CONNECTED
    ws2 = AsyncMock()
    ws2.client_state.value = 1 # CONNECTED

    # Connect first tab
    await manager.connect(ws1, task_id, channels=["tracking"], tab_id=tab_id_1)
    
    assert ws1 in manager.active_connections[task_id]
    assert manager.connection_tab_ids[ws1] == tab_id_1
    assert len(manager.active_connections[task_id]) == 1

    # Connect second tab (should disconnect first tab)
    await manager.connect(ws2, task_id, channels=["tracking"], tab_id=tab_id_2)

    # Verify second tab is connected
    assert ws2 in manager.active_connections[task_id]
    assert manager.connection_tab_ids[ws2] == tab_id_2
    
    # Verify first tab was closed
    ws1.close.assert_called_with(code=1000, reason="New tab connected")
    
    # Verify first tab is removed from manager (disconnect is called)
    # Note: In the real implementation, disconnect is called after close. 
    # Since we strictly mocked generic Close, we need to ensure disconnect logic ran.
    # The connect method calls ws.close() then await self.disconnect(ws, task_id)
    # so we should check if ws1 is no longer in active_connections
    assert ws1 not in manager.active_connections[task_id]
    assert len(manager.active_connections[task_id]) == 1

@pytest.mark.asyncio
async def test_same_tab_multiple_connections(manager):
    task_id = "task_456"
    tab_id = "tab_A"

    # Mock WebSockets
    ws_tracking = AsyncMock()
    ws_tracking.client_state.value = 1
    ws_frames = AsyncMock()
    ws_frames.client_state.value = 1

    # Connect tracking from tab A
    await manager.connect(ws_tracking, task_id, channels=["tracking"], tab_id=tab_id)
    
    assert len(manager.active_connections[task_id]) == 1
    assert manager.connection_tab_ids[ws_tracking] == tab_id

    # Connect frames from SAME tab A (should NOT disconnect tracking)
    await manager.connect(ws_frames, task_id, channels=["frames"], tab_id=tab_id)

    assert len(manager.active_connections[task_id]) == 2
    assert ws_tracking in manager.active_connections[task_id]
    assert ws_frames in manager.active_connections[task_id]
    
    # Verify no close calls
    ws_tracking.close.assert_not_called()

@pytest.mark.asyncio
async def test_channel_takeover_same_tab(manager):
    task_id = "task_789"
    tab_id = "tab_B"

    # Mock WebSockets
    ws1 = AsyncMock()
    ws1.client_state.value = 1
    ws2 = AsyncMock()
    ws2.client_state.value = 1

    # Connect tracking from tab B
    await manager.connect(ws1, task_id, channels=["tracking"], tab_id=tab_id)
    
    # Connect tracking again from SAME tab B (should replace valid connection)
    await manager.connect(ws2, task_id, channels=["tracking"], tab_id=tab_id)

    # ws1 should be disconnected because it's a channel conflict
    ws1.close.assert_called_with(code=1000, reason="New connection took over channel")
    assert ws1 not in manager.active_connections[task_id]
    assert ws2 in manager.active_connections[task_id]
