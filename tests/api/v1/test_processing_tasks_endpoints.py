# tests/api/v1/test_processing_tasks_endpoints.py
"""
Unit tests for API endpoints related to processing tasks.
"""
import pytest
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import BackgroundTasks, HTTPException

# Functions to test
from app.api.v1.endpoints.processing_tasks import start_processing_task_endpoint, get_processing_task_status_endpoint
from app.api.v1 import schemas as api_schemas
from app.services.pipeline_orchestrator import PipelineOrchestratorService
# mock_settings is available from conftest.py

@pytest.fixture
def mock_orchestrator_service(mocker):
    """Mocks PipelineOrchestratorService."""
    mock = MagicMock(spec=PipelineOrchestratorService)
    mock.initialize_task = AsyncMock(return_value=uuid.uuid4()) # Returns a dummy task_id
    mock.get_task_status = AsyncMock(return_value=None) # Default: task not found
    mock.run_processing_pipeline = AsyncMock() # The background task
    return mock

@pytest.fixture
def mock_background_tasks(mocker):
    """Mocks FastAPI's BackgroundTasks."""
    mock = MagicMock(spec=BackgroundTasks)
    mock.add_task = MagicMock()
    return mock


@pytest.mark.asyncio
async def test_start_processing_task_endpoint_success(
    mock_orchestrator_service: MagicMock, # Use the specific type
    mock_background_tasks: MagicMock,
    mock_settings # For API_V1_PREFIX
):
    """Tests successful initiation of a processing task."""
    env_id = "factory_test"
    request_payload = api_schemas.ProcessingTaskStartRequest(environment_id=env_id)
    
    # Configure the mock orchestrator for this specific call
    test_task_id = uuid.uuid4()
    mock_orchestrator_service.initialize_task.return_value = test_task_id

    response = await start_processing_task_endpoint(
        params=request_payload,
        background_tasks=mock_background_tasks,
        orchestrator=mock_orchestrator_service
    )

    mock_orchestrator_service.initialize_task.assert_called_once_with(env_id)
    mock_background_tasks.add_task.assert_called_once_with(
        mock_orchestrator_service.run_processing_pipeline,
        task_id=test_task_id,
        environment_id=env_id
    )
    
    assert response.task_id == test_task_id
    assert response.message == f"Processing task for environment '{env_id}' initiated."
    assert response.status_url == f"{mock_settings.API_V1_PREFIX}/processing-tasks/{test_task_id}/status"
    assert response.websocket_url == f"/ws/tracking/{test_task_id}"

@pytest.mark.asyncio
async def test_start_processing_task_endpoint_init_value_error(
    mock_orchestrator_service: MagicMock,
    mock_background_tasks: MagicMock,
):
    """Tests when orchestrator.initialize_task raises a ValueError."""
    env_id = "invalid_env"
    request_payload = api_schemas.ProcessingTaskStartRequest(environment_id=env_id)
    mock_orchestrator_service.initialize_task.side_effect = ValueError("Invalid environment specified")

    with pytest.raises(HTTPException) as exc_info:
        await start_processing_task_endpoint(
            params=request_payload,
            background_tasks=mock_background_tasks,
            orchestrator=mock_orchestrator_service
        )
    
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid environment specified"
    mock_background_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_start_processing_task_endpoint_unexpected_error(
    mock_orchestrator_service: MagicMock,
    mock_background_tasks: MagicMock,
):
    """Tests an unexpected error during task initiation."""
    env_id = "error_env"
    request_payload = api_schemas.ProcessingTaskStartRequest(environment_id=env_id)
    mock_orchestrator_service.initialize_task.side_effect = Exception("Unexpected boom")

    with pytest.raises(HTTPException) as exc_info:
        await start_processing_task_endpoint(
            params=request_payload,
            background_tasks=mock_background_tasks,
            orchestrator=mock_orchestrator_service
        )
    
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error starting task."


@pytest.mark.asyncio
async def test_get_processing_task_status_endpoint_found(
    mock_orchestrator_service: MagicMock,
):
    """Tests getting status for an existing task."""
    task_id = uuid.uuid4()
    status_info_from_service = {
        "status": "PROCESSING",
        "progress": 0.75,
        "current_step": "Analyzing frames",
        "details": "Almost done"
    }
    mock_orchestrator_service.get_task_status.return_value = status_info_from_service

    response = await get_processing_task_status_endpoint(task_id, orchestrator=mock_orchestrator_service)

    mock_orchestrator_service.get_task_status.assert_called_once_with(task_id)
    assert response.task_id == task_id
    assert response.status == status_info_from_service["status"]
    assert response.progress == status_info_from_service["progress"]
    assert response.current_step == status_info_from_service["current_step"]
    assert response.details == status_info_from_service["details"]


@pytest.mark.asyncio
async def test_get_processing_task_status_endpoint_not_found(
    mock_orchestrator_service: MagicMock,
):
    """Tests getting status for a non-existent task."""
    task_id = uuid.uuid4()
    mock_orchestrator_service.get_task_status.return_value = None # Simulate task not found

    with pytest.raises(HTTPException) as exc_info:
        await get_processing_task_status_endpoint(task_id, orchestrator=mock_orchestrator_service)
        
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Processing task not found." 