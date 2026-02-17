"""In-memory storage and synchronization helpers for playback control state."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from app.api.v1.schemas import PlaybackState


def _utcnow() -> datetime:
    """Return timezone-aware UTC timestamps for consistent auditing."""

    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class PlaybackStatus:
    """Application-level representation of the playback state for a task."""

    task_id: str
    state: PlaybackState
    last_transition_at: datetime
    last_frame_index: Optional[int] = None
    last_error: Optional[str] = None
    total_frames: Optional[int] = None

    def with_updates(
        self,
        *,
        state: Optional[PlaybackState] = None,
        last_frame_index: Optional[int] = None,
        last_error: Optional[str] = None,
        transition_time: Optional[datetime] = None,
        total_frames: Optional[int] = None,
    ) -> "PlaybackStatus":
        """Return a copy with updated fields while preserving immutability."""

        return replace(
            self,
            state=state or self.state,
            last_transition_at=transition_time or self.last_transition_at,
            last_frame_index=(
                last_frame_index
                if last_frame_index is not None
                else self.last_frame_index
            ),
            last_error=last_error,
            total_frames=(
                total_frames
                if total_frames is not None
                else self.total_frames
            ),
        )


class PlaybackStatusStore:
    """Lightweight in-memory registry for per-task playback state."""

    def __init__(self) -> None:
        self._state: Dict[str, PlaybackStatus] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _ensure_default_status(self, task_id: str) -> PlaybackStatus:
        status = self._state.get(task_id)
        if status is None:
            status = PlaybackStatus(
                task_id=task_id,
                state=PlaybackState.PLAYING,
                last_transition_at=_utcnow(),
            )
            self._state[task_id] = status
        return status

    async def get_status(self, task_id: str) -> PlaybackStatus:
        """Return the current status for a task, creating defaults if needed."""

        lock = self._locks[task_id]
        async with lock:
            status = self._ensure_default_status(task_id)
            return status

    async def transition(
        self,
        task_id: str,
        next_state: PlaybackState,
        *,
        last_frame_index: Optional[int] = None,
        last_error: Optional[str] = None,
    ) -> Tuple[PlaybackStatus, PlaybackStatus]:
        """Persist a state transition and return (previous, updated) snapshots."""

        lock = self._locks[task_id]
        async with lock:
            previous = self._ensure_default_status(task_id)
            if (
                previous.state == next_state
                and last_frame_index is None
                and last_error is None
            ):
                return previous, previous

            transition_time = _utcnow()
            updated = previous.with_updates(
                state=next_state,
                last_frame_index=last_frame_index,
                last_error=last_error,
                transition_time=transition_time,
            )
            self._state[task_id] = updated
            return previous, updated

    async def remove(self, task_id: str) -> None:
        """Remove a task from the store, typically when the task completes."""

        lock = self._locks[task_id]
        async with lock:
            self._state.pop(task_id, None)

    async def set_error(
        self,
        task_id: str,
        *,
        error_message: str,
        last_frame_index: Optional[int] = None,
    ) -> PlaybackStatus:
        """Attach an error to the current status without altering state."""

        lock = self._locks[task_id]
        async with lock:
            current = self._ensure_default_status(task_id)
            updated = current.with_updates(
                last_error=error_message,
                last_frame_index=last_frame_index,
                transition_time=_utcnow(),
            )
            self._state[task_id] = updated
            return updated

    async def update_last_frame_index(
        self,
        task_id: str,
        frame_index: int,
    ) -> PlaybackStatus:
        """Record the latest processed frame without altering playback state."""

        lock = self._locks[task_id]
        async with lock:
            current = self._ensure_default_status(task_id)
            updated = current.with_updates(
                last_frame_index=frame_index,
                transition_time=current.last_transition_at,
            )
            self._state[task_id] = updated
            return updated

    async def set_total_frames(
        self,
        task_id: str,
        total_frames: int,
    ) -> PlaybackStatus:
        """Store the total frame count so clients can display progress."""

        lock = self._locks[task_id]
        async with lock:
            current = self._ensure_default_status(task_id)
            updated = current.with_updates(
                total_frames=total_frames,
                transition_time=current.last_transition_at,
            )
            self._state[task_id] = updated
            return updated
