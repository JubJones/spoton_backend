# tests/services/test_notification_service.py
"""
Unit tests for the NotificationService in app.services.notification_service.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock

from app.services.notification_service import NotificationService
# from app.api.websockets import ConnectionManager # Not needed directly if mocking

@pytest.fixture
def mock_connection_manager(mocker):
    """Mocks the ConnectionManager."""
    mock_manager = MagicMock()
    # broadcast_to_task is an async method
    mock_manager.broadcast_to_task = AsyncMock()
    return mock_manager

@pytest.fixture
def notification_service_instance(mock_connection_manager):
    """Provides an instance of NotificationService with a mocked ConnectionManager."""
    return NotificationService(manager=mock_connection_manager)

@pytest.mark.asyncio
async def test_notification_service_init(mock_connection_manager):
    """Tests NotificationService initialization."""
    service = NotificationService(manager=mock_connection_manager)
    assert service.manager == mock_connection_manager

@pytest.mark.asyncio
async def test_send_tracking_update(notification_service_instance: NotificationService, mock_connection_manager: MagicMock):
    """Tests sending a tracking update."""
    task_id = "test_task_123"
    payload = {"frame": 1, "data": "some_tracking_info"}

    await notification_service_instance.send_tracking_update(task_id, payload)

    expected_message = {
        "type": "tracking_update",
        "payload": payload
    }
    mock_connection_manager.broadcast_to_task.assert_called_once_with(task_id, expected_message)

@pytest.mark.asyncio
async def test_send_tracking_update_empty_task_id(notification_service_instance: NotificationService, mock_connection_manager: MagicMock, mocker):
    """Tests send_tracking_update with an empty task_id."""
    mock_logger_warning = mocker.patch("app.services.notification_service.logger.warning")
    await notification_service_instance.send_tracking_update("", {"data": "info"})
    mock_connection_manager.broadcast_to_task.assert_not_called()
    mock_logger_warning.assert_called_with("Attempted to send tracking update with empty task_id.")


@pytest.mark.asyncio
async def test_send_status_update(notification_service_instance: NotificationService, mock_connection_manager: MagicMock):
    """Tests sending a status update."""
    task_id = "test_task_456"
    payload = {"status": "PROCESSING", "progress": 0.5}

    await notification_service_instance.send_status_update(task_id, payload)

    expected_message = {
        "type": "status_update",
        "payload": payload
    }
    mock_connection_manager.broadcast_to_task.assert_called_once_with(task_id, expected_message)

@pytest.mark.asyncio
async def test_send_status_update_empty_task_id(notification_service_instance: NotificationService, mock_connection_manager: MagicMock, mocker):
    """Tests send_status_update with an empty task_id."""
    mock_logger_warning = mocker.patch("app.services.notification_service.logger.warning")
    await notification_service_instance.send_status_update("", {"status": "QUEUED"})
    mock_connection_manager.broadcast_to_task.assert_not_called()
    mock_logger_warning.assert_called_with("Attempted to send status update with empty task_id.")

@pytest.mark.asyncio
async def test_send_methods_handle_broadcast_exception(notification_service_instance: NotificationService, mock_connection_manager: MagicMock, mocker):
    """Tests that send methods handle exceptions from broadcast_to_task."""
    task_id = "test_task_789"
    payload = {"data": "info"}
    mock_connection_manager.broadcast_to_task.side_effect = Exception("Broadcast failed")
    mock_logger_error = mocker.patch("app.services.notification_service.logger.error")

    await notification_service_instance.send_tracking_update(task_id, payload)
    mock_logger_error.assert_called_once_with(
        f"Error broadcasting tracking update for task {task_id}: Broadcast failed", exc_info=True
    )
    mock_logger_error.reset_mock()

    await notification_service_instance.send_status_update(task_id, payload)
    mock_logger_error.assert_called_once_with(
        f"Error broadcasting status update for task {task_id}: Broadcast failed", exc_info=True
    ) 