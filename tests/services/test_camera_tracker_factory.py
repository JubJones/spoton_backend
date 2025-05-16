# tests/services/test_camera_tracker_factory.py
"""
Unit tests for the CameraTrackerFactory in app.services.camera_tracker_factory.
"""
import pytest
import uuid
import torch
from unittest.mock import MagicMock, AsyncMock, PropertyMock, patch

from app.services.camera_tracker_factory import CameraTrackerFactory
# Assuming mock_settings is available from conftest.py
# We will also need to mock BotSortTracker

@pytest.fixture
def mock_botsort_tracker_class_for_factory(mocker):
    """Mocks the BotSortTracker class used by the factory."""
    mock_tracker_instance = MagicMock()
    mock_tracker_instance.load_model = AsyncMock()
    mock_tracker_instance.warmup = AsyncMock()
    mock_tracker_instance.reset = AsyncMock()

    mock_class = MagicMock(return_value=mock_tracker_instance) # Constructor returns our instance
    mocker.patch("app.services.camera_tracker_factory.BotSortTracker", mock_class)
    return mock_class, mock_tracker_instance


@pytest.fixture
def camera_tracker_factory_instance(mock_settings, mocker):
    """Provides an instance of CameraTrackerFactory."""
    # Patch settings in the camera_tracker_factory module if it imports it directly
    # (It doesn't seem to, it gets config via BotSortTracker which gets it from its module)
    # mocker.patch("app.services.camera_tracker_factory.settings", mock_settings)
    test_device = torch.device("cpu") # Use CPU for these tests
    return CameraTrackerFactory(device=test_device)


def test_camera_tracker_factory_init(camera_tracker_factory_instance: CameraTrackerFactory):
    """Tests CameraTrackerFactory initialization."""
    assert camera_tracker_factory_instance.device.type == "cpu"
    assert not camera_tracker_factory_instance._prototype_tracker_loaded
    assert camera_tracker_factory_instance._tracker_instances == {}

@pytest.mark.asyncio
async def test_preload_prototype_tracker_success(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory
):
    """Tests successful preloading of the prototype tracker."""
    mock_constructor, mock_instance = mock_botsort_tracker_class_for_factory

    await camera_tracker_factory_instance.preload_prototype_tracker()

    mock_constructor.assert_called_once() # BotSortTracker()
    mock_instance.load_model.assert_called_once()
    mock_instance.warmup.assert_called_once()
    assert camera_tracker_factory_instance._prototype_tracker_loaded is True

@pytest.mark.asyncio
async def test_preload_prototype_tracker_failure(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory,
    mocker
):
    """Tests failure during prototype tracker preloading."""
    mock_constructor, mock_instance = mock_botsort_tracker_class_for_factory
    mock_instance.load_model.side_effect = RuntimeError("Failed to load ReID")
    mock_logger_error = mocker.patch("app.services.camera_tracker_factory.logger.error")

    await camera_tracker_factory_instance.preload_prototype_tracker()

    assert camera_tracker_factory_instance._prototype_tracker_loaded is False
    mock_logger_error.assert_called_once_with(
        "Failed to preload prototype tracker (ReID model): Failed to load ReID", exc_info=True
    )

@pytest.mark.asyncio
async def test_preload_prototype_already_loaded(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory,
    mocker
):
    """Tests that preload does nothing if already done."""
    camera_tracker_factory_instance._prototype_tracker_loaded = True
    mock_constructor, _ = mock_botsort_tracker_class_for_factory
    mock_logger_info = mocker.patch("app.services.camera_tracker_factory.logger.info")


    await camera_tracker_factory_instance.preload_prototype_tracker()
    mock_constructor.assert_not_called()
    mock_logger_info.assert_called_with("Prototype tracker (and its ReID model) already preloaded.")


@pytest.mark.asyncio
async def test_get_tracker_new_creation_and_cache(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory
):
    """Tests creation of a new tracker and its caching."""
    task_id = uuid.uuid4()
    camera_id = "cam01"
    mock_constructor, mock_instance = mock_botsort_tracker_class_for_factory

    # First call: creates and caches
    tracker1 = await camera_tracker_factory_instance.get_tracker(task_id, camera_id)
    mock_constructor.assert_called_once()
    mock_instance.load_model.assert_called_once()
    assert tracker1 == mock_instance
    assert (task_id, camera_id) in camera_tracker_factory_instance._tracker_instances
    assert camera_tracker_factory_instance._tracker_instances[(task_id, camera_id)] == mock_instance

    # Second call: returns cached
    mock_constructor.reset_mock() # Reset for the next assertion
    mock_instance.load_model.reset_mock()
    tracker2 = await camera_tracker_factory_instance.get_tracker(task_id, camera_id)
    assert tracker2 == tracker1
    mock_constructor.assert_not_called() # Should not be called again
    mock_instance.load_model.assert_not_called() # load_model should not be called again

@pytest.mark.asyncio
async def test_get_tracker_load_failure(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory,
    mocker
):
    """Tests get_tracker when the tracker's load_model fails."""
    task_id = uuid.uuid4()
    camera_id = "cam_fail"
    _, mock_instance = mock_botsort_tracker_class_for_factory
    mock_instance.load_model.side_effect = RuntimeError("Tracker load failed")

    with pytest.raises(RuntimeError, match="Tracker load failed"):
        await camera_tracker_factory_instance.get_tracker(task_id, camera_id)

    assert (task_id, camera_id) not in camera_tracker_factory_instance._tracker_instances


@pytest.mark.asyncio
async def test_reset_tracker_existing(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory
):
    """Tests resetting an existing tracker."""
    task_id = uuid.uuid4()
    camera_id = "cam_to_reset"
    _, mock_instance = mock_botsort_tracker_class_for_factory

    # Get a tracker to populate the cache
    await camera_tracker_factory_instance.get_tracker(task_id, camera_id)
    mock_instance.reset.reset_mock() # Reset call count from get_tracker if it calls reset

    await camera_tracker_factory_instance.reset_tracker(task_id, camera_id)
    mock_instance.reset.assert_called_once()

@pytest.mark.asyncio
async def test_reset_tracker_non_existent(camera_tracker_factory_instance: CameraTrackerFactory, mocker):
    """Tests attempting to reset a non-existent tracker."""
    mock_logger_warning = mocker.patch("app.services.camera_tracker_factory.logger.warning")
    task_id = uuid.uuid4()
    camera_id = "cam_not_exist"
    await camera_tracker_factory_instance.reset_tracker(task_id, camera_id)
    mock_logger_warning.assert_called_once_with(
        f"Attempted to reset non-existent tracker for task '{task_id}', camera '{camera_id}'."
    )

@pytest.mark.asyncio
async def test_clear_trackers_for_task(camera_tracker_factory_instance: CameraTrackerFactory, mock_botsort_tracker_class_for_factory):
    """Tests clearing all trackers associated with a task."""
    task_id1 = uuid.uuid4()
    task_id2 = uuid.uuid4()

    # Populate with trackers for two tasks
    await camera_tracker_factory_instance.get_tracker(task_id1, "camA")
    await camera_tracker_factory_instance.get_tracker(task_id1, "camB")
    await camera_tracker_factory_instance.get_tracker(task_id2, "camC")

    assert len(camera_tracker_factory_instance._tracker_instances) == 3

    await camera_tracker_factory_instance.clear_trackers_for_task(task_id1)

    assert len(camera_tracker_factory_instance._tracker_instances) == 1
    assert (task_id1, "camA") not in camera_tracker_factory_instance._tracker_instances
    assert (task_id1, "camB") not in camera_tracker_factory_instance._tracker_instances
    assert (task_id2, "camC") in camera_tracker_factory_instance._tracker_instances


@pytest.mark.asyncio
async def test_reset_all_trackers_for_task(camera_tracker_factory_instance: CameraTrackerFactory, mock_botsort_tracker_class_for_factory):
    """Tests resetting all trackers for a specific task."""
    task_id1 = uuid.uuid4()
    task_id2 = uuid.uuid4() # Another task, its trackers should not be reset

    # Use a list to store the mock instances created for task_id1
    task1_mock_instances = []

    # Custom side effect for the BotSortTracker constructor mock
    def mock_constructor_side_effect_for_reset_all(*args, **kwargs):
        new_mock_instance = MagicMock()
        new_mock_instance.load_model = AsyncMock()
        new_mock_instance.reset = AsyncMock()
        # Check if this is for task_id1 based on some implicit context or just store all
        # For simplicity, we'll just check based on which ones are retrieved.
        # This part is a bit tricky without knowing exactly which instance corresponds to which task
        # if we are re-mocking the constructor for each get_tracker.
        # A better way is to have mock_botsort_tracker_class_for_factory return a list of instances
        # or have get_tracker return distinct mocks.

        # Simpler: The factory stores actual instances. We check if reset is called on them.
        # The mock_botsort_tracker_class_for_factory already gives us a mock instance.
        # We need to ensure that for each call to get_tracker, it returns a "new" mock BotSortTracker instance
        # if we want to check individual reset calls.
        # Or, if it returns the *same* mock instance (due to how mock_botsort_tracker_class_for_factory is set up),
        # we count total calls to reset.

        # Let's re-mock the main BotSortTracker class to return fresh mocks each time for this specific test
        fresh_mock_instances_task1 = [MagicMock(reset=AsyncMock()), MagicMock(reset=AsyncMock())]
        fresh_mock_instance_task2 = MagicMock(reset=AsyncMock())

        # Side effect for BotSortTracker constructor
        call_count = 0
        def constructor_side_effect(*args, **kwargs):
            nonlocal call_count
            if call_count < 2: # For task_id1's two trackers
                instance_to_return = fresh_mock_instances_task1[call_count]
            else: # For task_id2's tracker
                instance_to_return = fresh_mock_instance_task2
            instance_to_return.load_model = AsyncMock() # Ensure it has load_model
            call_count += 1
            return instance_to_return

        mock_botsort_tracker_class_for_factory[0].side_effect = constructor_side_effect


    # Populate trackers
    tracker_t1_cA = await camera_tracker_factory_instance.get_tracker(task_id1, "camA")
    tracker_t1_cB = await camera_tracker_factory_instance.get_tracker(task_id1, "camB")
    tracker_t2_cC = await camera_tracker_factory_instance.get_tracker(task_id2, "camC")

    # Reset mocks for task1 instances before calling reset_all
    tracker_t1_cA.reset.reset_mock() # type: ignore
    tracker_t1_cB.reset.reset_mock() # type: ignore
    tracker_t2_cC.reset.reset_mock() # type: ignore


    await camera_tracker_factory_instance.reset_all_trackers_for_task(task_id1)

    tracker_t1_cA.reset.assert_called_once() # type: ignore
    tracker_t1_cB.reset.assert_called_once() # type: ignore
    tracker_t2_cC.reset.assert_not_called() # type: ignore Tracker for task_id2 should not be reset 