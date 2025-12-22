"""
Factory for creating and managing per-camera tracker instances.
This ensures that each camera view has its own tracker state.
"""
import logging
from typing import Dict, Optional
import uuid

import torch
import numpy as np # For dummy warmup data

from app.models.trackers import ByteTrackTracker
from app.models.base_models import AbstractTracker
from app.core.config import settings

logger = logging.getLogger(__name__)

class CameraTrackerFactory:
    """
    Manages instances of trackers, providing a unique tracker for each camera
    within a given processing task.
    Can also preload a prototype tracker to warm up model code paths.
    """

    def __init__(self, device: torch.device):
        """
        Initializes the CameraTrackerFactory.

        Args:
            device: The torch device (e.g., 'cuda', 'cpu') to load models onto.
                    Note: Trackers themselves might also determine their device internally.
        """
        self.device = device # May not be directly used if trackers handle their own device
        # Cache: {(task_id, camera_id): tracker_instance}
        self._tracker_instances: Dict[tuple[uuid.UUID, str], AbstractTracker] = {}
        self._prototype_tracker_loaded = False
        logger.info(f"CameraTrackerFactory initialized for device: {self.device}.")

    async def preload_prototype_tracker(self):
        """
        Creates and loads a 'prototype' tracker (configured via settings.TRACKER_TYPE).
        The primary purpose is to ensure relevant code paths are warmed up during startup.
        The prototype itself isn't typically used for actual tracking tasks.
        """
        if self._prototype_tracker_loaded:
            logger.info("Prototype tracker already preloaded.")
            return

        logger.info("Preloading prototype ByteTrack tracker...")
        try:
            # Create a temporary tracker instance for warmup
            prototype_tracker = ByteTrackTracker()
            await prototype_tracker.load_model()
            await prototype_tracker.warmup() # Warm up the loaded prototype
            
            self._prototype_tracker_loaded = True
            logger.info("Prototype ByteTrack tracker preloaded and warmed up successfully.")
            # The prototype_tracker instance can be discarded here if its only role was preloading.
        except Exception as e:
            logger.error(f"Failed to preload prototype tracker: {e}", exc_info=True)
            # Depending on severity, could raise to halt startup or just log.
            # For now, just log, subsequent get_tracker calls will try to load individually.
            self._prototype_tracker_loaded = False # Explicitly mark as not successful


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
            # logger.info(f"Creating new ByteTrack tracker for task '{task_id}', camera '{camera_id}'.")
            try:
                tracker = ByteTrackTracker()
            except ImportError as e:
                logger.error(f"Failed to import/instantiate ByteTrackTracker: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error instantiating ByteTrackTracker: {e}")
                raise
            
            try:
                # The tracker's load_model method should handle device placement.
                await tracker.load_model()
                # No separate warmup here for task-specific trackers unless desired for absolute first frame.
                # The prototype warmup should cover general JIT.
                # logger.info(f"ByteTrack tracker model loaded successfully for task '{task_id}', camera '{camera_id}'.")
            except Exception as e:
                logger.error(
                    f"Failed to load model for ByteTrack tracker (task '{task_id}', camera '{camera_id}'): {e}",
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
            await tracker.reset()
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
                # Any specific cleanup on the tracker object itself before del?
                # Python's GC will handle it if no other references.
                del self._tracker_instances[key]
        # Optionally, trigger GC or CUDA cache clear if memory is critical
        # import gc; gc.collect()
        # if torch.cuda.is_available(): torch.cuda.empty_cache()

    async def reset_all_trackers_for_task(self, task_id: uuid.UUID):
        """
        Resets all tracker instances associated with a task, e.g., before processing a new batch of sub-videos.
        """
        logger.info(f"Resetting all trackers for task '{task_id}'.")
        for (current_task_id, camera_id), tracker_instance in self._tracker_instances.items():
            if current_task_id == task_id:
                pass # logger.debug(f"Calling reset for tracker of camera '{camera_id}' in task '{task_id}'.")
                await tracker_instance.reset()
