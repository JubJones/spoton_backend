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
    
    # This function will be the side_effect for the BotSortTracker constructor
    def constructor_side_effect(*args, **kwargs):
        mock_tracker_instance = MagicMock(spec=True) # Use spec=True for stricter mocking
        mock_tracker_instance.load_model = AsyncMock()
        mock_tracker_instance.warmup = AsyncMock()
        mock_tracker_instance.reset = AsyncMock()
        return mock_tracker_instance

    mock_class = MagicMock(side_effect=constructor_side_effect) # Constructor returns a new mock each time
    mocker.patch("app.services.camera_tracker_factory.BotSortTracker", mock_class)
    return mock_class


@pytest.fixture
def camera_tracker_factory_instance(mock_settings, mocker):
    """Provides an instance of CameraTrackerFactory."""
    test_device = torch.device("cpu") 
    return CameraTrackerFactory(device=test_device)


def test_camera_tracker_factory_init(camera_tracker_factory_instance: CameraTrackerFactory):
    """Tests CameraTrackerFactory initialization."""
    assert camera_tracker_factory_instance.device.type == "cpu"
    assert not camera_tracker_factory_instance._prototype_tracker_loaded
    assert camera_tracker_factory_instance._tracker_instances == {}

@pytest.mark.asyncio
async def test_preload_prototype_tracker_success(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory # This fixture now ensures constructor returns a mock
):
    """Tests successful preloading of the prototype tracker."""
    # The mock_botsort_tracker_class_for_factory is the mocked class constructor
    # We need to get the instance it *will* return for assertions
    
    # To get the instance that was created, we can check the return_value of the first call
    # to the constructor mock, but since the side_effect creates new ones, this is tricky.
    # It's easier to check that the methods (load_model, warmup) were called on *an* instance.
    
    # Let's capture the instance created by preload_prototype_tracker
    created_instances = []
    original_side_effect = mock_botsort_tracker_class_for_factory.side_effect
    def side_effect_capturing_instance(*args, **kwargs):
        instance = original_side_effect(*args, **kwargs)
        created_instances.append(instance)
        return instance
    mock_botsort_tracker_class_for_factory.side_effect = side_effect_capturing_instance

    await camera_tracker_factory_instance.preload_prototype_tracker()

    mock_botsort_tracker_class_for_factory.assert_called_once() # BotSortTracker() constructor
    assert len(created_instances) == 1
    mock_prototype_instance = created_instances[0]
    
    mock_prototype_instance.load_model.assert_called_once()
    mock_prototype_instance.warmup.assert_called_once()
    assert camera_tracker_factory_instance._prototype_tracker_loaded is True

@pytest.mark.asyncio
async def test_preload_prototype_tracker_failure(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory,
    mocker
):
    """Tests failure during prototype tracker preloading."""
    # Configure the constructor mock to return an instance whose load_model fails
    failing_instance = MagicMock()
    failing_instance.load_model = AsyncMock(side_effect=RuntimeError("Failed to load ReID"))
    failing_instance.warmup = AsyncMock() # Not reached
    mock_botsort_tracker_class_for_factory.side_effect = lambda *a, **kw: failing_instance
    
    mock_logger_error = mocker.patch("app.services.camera_tracker_factory.logger.error")

    await camera_tracker_factory_instance.preload_prototype_tracker()

    assert camera_tracker_factory_instance._prototype_tracker_loaded is False
    failing_instance.load_model.assert_called_once()
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
    mock_logger_info = mocker.patch("app.services.camera_tracker_factory.logger.info")


    await camera_tracker_factory_instance.preload_prototype_tracker()
    mock_botsort_tracker_class_for_factory.assert_not_called()
    mock_logger_info.assert_called_with("Prototype tracker (and its ReID model) already preloaded.")


@pytest.mark.asyncio
async def test_get_tracker_new_creation_and_cache(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory # Mocked constructor
):
    """Tests creation of a new tracker and its caching."""
    task_id = uuid.uuid4()
    camera_id = "cam01"
    
    # Store instances created by the factory for assertion
    created_instances_map = {}
    original_side_effect = mock_botsort_tracker_class_for_factory.side_effect
    def side_effect_capturing(*args, **kwargs):
        instance = original_side_effect(*args, **kwargs)
        # A way to identify which call this instance belongs to, if needed,
        # but for this test, we just need to ensure methods are called on the right one.
        return instance
    mock_botsort_tracker_class_for_factory.side_effect = side_effect_capturing


    # First call: creates and caches
    tracker1 = await camera_tracker_factory_instance.get_tracker(task_id, camera_id)
    assert mock_botsort_tracker_class_for_factory.call_count == 1
    tracker1.load_model.assert_called_once() # Instance method check
    assert (task_id, camera_id) in camera_tracker_factory_instance._tracker_instances
    assert camera_tracker_factory_instance._tracker_instances[(task_id, camera_id)] == tracker1


    # Second call: returns cached
    tracker1.load_model.reset_mock() # Reset mock on the specific instance
    call_count_before_second_get = mock_botsort_tracker_class_for_factory.call_count

    tracker2 = await camera_tracker_factory_instance.get_tracker(task_id, camera_id)
    assert tracker2 == tracker1
    assert mock_botsort_tracker_class_for_factory.call_count == call_count_before_second_get # Constructor not called again
    tracker1.load_model.assert_not_called() # load_model should not be called again on cached instance

@pytest.mark.asyncio
async def test_get_tracker_load_failure(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory,
    mocker
):
    """Tests get_tracker when the tracker's load_model fails."""
    task_id = uuid.uuid4()
    camera_id = "cam_fail"

    failing_instance = MagicMock()
    failing_instance.load_model = AsyncMock(side_effect=RuntimeError("Tracker load failed"))
    mock_botsort_tracker_class_for_factory.side_effect = lambda *a, **kw: failing_instance

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

    # Get a tracker to populate the cache
    tracker_instance = await camera_tracker_factory_instance.get_tracker(task_id, camera_id)
    tracker_instance.reset.reset_mock() # Reset call count from get_tracker if it calls reset

    await camera_tracker_factory_instance.reset_tracker(task_id, camera_id)
    tracker_instance.reset.assert_called_once()

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
async def test_reset_all_trackers_for_task(
    camera_tracker_factory_instance: CameraTrackerFactory,
    mock_botsort_tracker_class_for_factory # This is the mock of the BotSortTracker class constructor
):
    """Tests resetting all trackers for a specific task."""
    task_id1 = uuid.uuid4()
    task_id2 = uuid.uuid4() # Another task, its trackers should not be reset

    # mock_botsort_tracker_class_for_factory is already configured to return a new MagicMock
    # with an AsyncMock for 'reset' each time BotSortTracker() is called.

    # Populate trackers - each call to get_tracker will result in BotSortTracker() being called,
    # which in turn uses the side_effect of mock_botsort_tracker_class_for_factory.
    tracker_t1_cA = await camera_tracker_factory_instance.get_tracker(task_id1, "camA")
    tracker_t1_cB = await camera_tracker_factory_instance.get_tracker(task_id1, "camB")
    tracker_t2_cC = await camera_tracker_factory_instance.get_tracker(task_id2, "camC")

    # Reset call counts on the specific mock instances before calling the method under test
    tracker_t1_cA.reset.reset_mock()
    tracker_t1_cB.reset.reset_mock()
    tracker_t2_cC.reset.reset_mock()

    await camera_tracker_factory_instance.reset_all_trackers_for_task(task_id1)

    tracker_t1_cA.reset.assert_called_once()
    tracker_t1_cB.reset.assert_called_once()
    tracker_t2_cC.reset.assert_not_called()