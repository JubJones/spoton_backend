"""Runtime registry tracking playback execution state per task."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.api.v1.schemas import PlaybackState


@dataclass
class TaskRuntime:
    """Holds per-task runtime metadata used for pause/resume coordination."""

    task_id: str
    pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    state: PlaybackState = PlaybackState.PLAYING
    last_frame_index: Optional[int] = None

    def __post_init__(self) -> None:
        # Tasks start in playing state by default.
        self.pause_event.set()


class TaskRuntimeRegistry:
    """Registry coordinating playback state across worker loops."""

    def __init__(self) -> None:
        self._runtimes: Dict[str, TaskRuntime] = {}
        self._lock = asyncio.Lock()

    async def register(self, task_id: str) -> TaskRuntime:
        """Ensure a runtime entry exists for the given task."""

        async with self._lock:
            runtime = self._runtimes.get(task_id)
            if runtime is None:
                runtime = TaskRuntime(task_id=task_id)
                self._runtimes[task_id] = runtime
            return runtime

    async def remove(self, task_id: str) -> None:
        """Remove runtime entry when task is complete."""

        async with self._lock:
            self._runtimes.pop(task_id, None)

    def get_runtime(self, task_id: str) -> Optional[TaskRuntime]:
        return self._runtimes.get(task_id)

    async def pause_task(self, task_id: str) -> TaskRuntime:
        runtime = self._runtimes.get(task_id)
        if runtime is None:
            raise KeyError(task_id)
        if runtime.state != PlaybackState.PAUSED:
            runtime.state = PlaybackState.PAUSED
            runtime.pause_event.clear()
        return runtime

    async def resume_task(self, task_id: str) -> TaskRuntime:
        runtime = self._runtimes.get(task_id)
        if runtime is None:
            raise KeyError(task_id)
        if runtime.state != PlaybackState.PLAYING:
            runtime.state = PlaybackState.PLAYING
            runtime.pause_event.set()
        return runtime

    async def wait_until_playing(self, task_id: str) -> None:
        runtime = self._runtimes.get(task_id)
        if runtime is None:
            return
        await runtime.pause_event.wait()

    def update_frame_index(self, task_id: str, frame_index: int) -> None:
        runtime = self._runtimes.get(task_id)
        if runtime is not None:
            runtime.last_frame_index = frame_index
