"""Playback control orchestration service."""

from __future__ import annotations

from typing import Optional

import logging

from fastapi import HTTPException, status

from app.api.v1.schemas import PlaybackState
from app.services.playback_status_store import PlaybackStatus, PlaybackStatusStore
from app.services.task_runtime_registry import TaskRuntimeRegistry, TaskRuntime
from app.utils.metrics_collector import production_metrics_collector


logger = logging.getLogger(__name__)


class PlaybackControlService:
    """Coordinate playback state transitions for processing tasks."""

    def __init__(
        self,
        status_store: PlaybackStatusStore,
        runtime_registry: TaskRuntimeRegistry,
        timeout_seconds: float = 1.0,
    ) -> None:
        self._status_store = status_store
        self._runtime_registry = runtime_registry
        self._control_timeout_seconds = timeout_seconds

    async def pause(self, task_id: str) -> PlaybackStatus:
        """Transition the task to PAUSED state."""

        runtime = self._ensure_runtime(task_id)
        await self._runtime_registry.pause_task(task_id)
        previous, updated = await self._status_store.transition(
            task_id,
            PlaybackState.PAUSED,
            last_frame_index=runtime.last_frame_index,
        )
        self._emit_transition(task_id, previous, updated)
        return updated if previous.state != PlaybackState.PAUSED else previous

    async def resume(self, task_id: str) -> PlaybackStatus:
        """Transition the task to PLAYING state."""

        runtime = self._ensure_runtime(task_id)
        await self._runtime_registry.resume_task(task_id)
        previous, updated = await self._status_store.transition(
            task_id,
            PlaybackState.PLAYING,
            last_frame_index=runtime.last_frame_index,
        )
        self._emit_transition(task_id, previous, updated)
        return updated if previous.state != PlaybackState.PLAYING else previous

    async def get_status(self, task_id: str) -> PlaybackStatus:
        """Fetch the current playback status for a task."""

        return await self._status_store.get_status(task_id)

    async def mark_error(
        self,
        task_id: str,
        *,
        message: str,
        last_frame_index: Optional[int] = None,
    ) -> PlaybackStatus:
        """Attach an error message to the current playback state."""

        return await self._status_store.set_error(
            task_id,
            error_message=message,
            last_frame_index=last_frame_index,
        )

    async def clear(self, task_id: str) -> None:
        """Remove task state—typically when tasks end."""

        await self._status_store.remove(task_id)
        await self._runtime_registry.remove(task_id)

    def _ensure_runtime(self, task_id: str) -> TaskRuntime:
        runtime = self._runtime_registry.get_runtime(task_id)
        if runtime is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task '{task_id}' is not active or has completed.",
            )
        return runtime

    def _emit_transition(
        self,
        task_id: str,
        previous: PlaybackStatus,
        updated: PlaybackStatus,
    ) -> None:
        if previous.state == updated.state:
            pass # logger.debug(
                #     "Playback state unchanged",
                #     extra={
                #         "event": "playback_state_transition_noop",
                #         "task_id": task_id,
                #         "state": previous.state.value,
                #     },
                # )
            return

        transition = {
            "event": "playback_state_transition",
            "task_id": task_id,
            "from_state": previous.state.value,
            "to_state": updated.state.value,
            "last_frame_index": updated.last_frame_index,
        }
        logger.info("Playback state transition", extra=transition)
        try:
            production_metrics_collector.record_playback_transition(
                previous_state=previous.state.value,
                next_state=updated.state.value,
            )
        except Exception as metrics_error:
            pass # logger.debug(f"Unable to record playback transition metric: {metrics_error}")
