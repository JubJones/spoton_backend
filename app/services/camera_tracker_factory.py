"""
Factory for creating and managing per-camera tracker instances.
This ensures that each camera view has its own tracker state.
"""
import logging
from typing import Dict, Optional
import uuid

import torch

from app.models.trackers import BotSortTracker
from app.models.base_models import AbstractTracker
from app.core.config import settings

logger = logging.getLogger(__name__)

class CameraTrackerFactory:
    """
    Manages instances of trackers, providing a unique tracker for each camera
    within a given processing task.
    """

    def __init__(self, device: torch.device):
        """
        Initializes the CameraTrackerFactory.

        Args:
            device: The torch device (e.g., 'cuda', 'cpu') to load models onto.
        """
        self.device = device
        # Cache: {(task_id, camera_id): tracker_instance}
        self._tracker_instances: Dict[tuple[uuid.UUID, str], AbstractTracker] = {}
        logger.info(f"CameraTrackerFactory initialized for device: {self.device}.")

    async def get_tracker(self, task_id: uuid.UUID, camera_id: str) -> AbstractTracker:
        """
        Provides a tracker instance for the specified camera and task.
        If an instance for this combination doesn't exist, it creates,
        loads, and caches one.

        Args:
            task_id: The unique identifier for the processing task.
            camera_id: The identifier for the camera.

        Returns:
            An initialized AbstractTracker instance.
        """
        cache_key = (task_id, camera_id)
        if cache_key not in self._tracker_instances:
            logger.info(f"Creating new BotSortTracker for task '{task_id}', camera '{camera_id}'.")
            # Assuming BotSortTracker is the primary implementation
            # It should internally handle loading its ReID model via settings
            tracker = BotSortTracker() # This uses settings for ReID weights path etc.
            
            # The tracker's load_model method should handle device placement
            # and ReID model loading as per its implementation.
            try:
                await tracker.load_model() # load_model should use self.device
                logger.info(f"BotSortTracker model loaded successfully for task '{task_id}', camera '{camera_id}'.")
            except Exception as e:
                logger.error(
                    f"Failed to load model for BotSortTracker (task '{task_id}', camera '{camera_id}'): {e}",
                    exc_info=True
                )
                raise # Re-raise to signal failure to the caller

            self._tracker_instances[cache_key] = tracker
        
        return self._tracker_instances[cache_key]

    async def reset_tracker(self, task_id: uuid.UUID, camera_id: str):
        """
        Resets the state of a specific tracker. This might involve re-initializing it.
        """
        cache_key = (task_id, camera_id)
        if cache_key in self._tracker_instances:
            logger.info(f"Resetting tracker for task '{task_id}', camera '{camera_id}'.")
            tracker = self._tracker_instances[cache_key]
            await tracker.reset() # Call the tracker's reset method
            # Depending on reset implementation, model might need reloading or state just cleared
            # If reset implies full re-initialization, you might re-create it:
            # del self._tracker_instances[cache_key]
            # await self.get_tracker(task_id, camera_id) # This will re-create and load
        else:
            logger.warning(f"Attempted to reset non-existent tracker for task '{task_id}', camera '{camera_id}'.")

    async def clear_trackers_for_task(self, task_id: uuid.UUID):
        """
        Removes all tracker instances associated with a completed or failed task
        to free up resources.
        """
        keys_to_remove = [key for key in self._tracker_instances if key[0] == task_id]
        if keys_to_remove:
            logger.info(f"Clearing {len(keys_to_remove)} tracker instances for task '{task_id}'.")
            for key in keys_to_remove:
                # Here you might also call any specific cleanup on the tracker if needed
                del self._tracker_instances[key]
        # Optionally, trigger garbage collection or CUDA cache clearing if memory is critical
        # import gc
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    async def reset_all_trackers_for_task(self, task_id: uuid.UUID):
        """
        Resets all tracker instances associated with a task, e.g., before processing a new batch of sub-videos.
        """
        logger.info(f"Resetting all trackers for task '{task_id}'.")
        for (current_task_id, camera_id), tracker_instance in self._tracker_instances.items():
            if current_task_id == task_id:
                logger.debug(f"Calling reset for tracker of camera '{camera_id}' in task '{task_id}'.")
                await tracker_instance.reset()