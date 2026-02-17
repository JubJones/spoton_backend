"""Playback control REST endpoints."""

from fastapi import APIRouter, Depends

from app.api.v1.schemas import PlaybackStatusResponse, SeekRequest
from app.dependencies import get_playback_control_service
from app.services.playback_control_service import PlaybackControlService
from app.services.playback_status_store import PlaybackStatus


router = APIRouter(tags=["V1 - Playback Controls"])


def _map_status(status: PlaybackStatus) -> PlaybackStatusResponse:
    return PlaybackStatusResponse(
        task_id=status.task_id,
        state=status.state,
        last_transition_at=status.last_transition_at,
        last_frame_index=status.last_frame_index,
        last_error=status.last_error,
        total_frames=status.total_frames,
    )


@router.post("/{task_id}/pause", response_model=PlaybackStatusResponse)
async def pause_task(
    task_id: str,
    service: PlaybackControlService = Depends(get_playback_control_service),
) -> PlaybackStatusResponse:
    status = await service.pause(task_id)
    return _map_status(status)


@router.post("/{task_id}/resume", response_model=PlaybackStatusResponse)
async def resume_task(
    task_id: str,
    service: PlaybackControlService = Depends(get_playback_control_service),
) -> PlaybackStatusResponse:
    status = await service.resume(task_id)
    return _map_status(status)


@router.get("/{task_id}/status", response_model=PlaybackStatusResponse)
async def get_task_status(
    task_id: str,
    service: PlaybackControlService = Depends(get_playback_control_service),
) -> PlaybackStatusResponse:
    status = await service.get_status(task_id)
    return _map_status(status)


@router.post("/{task_id}/seek", response_model=PlaybackStatusResponse)
async def seek_task(
    task_id: str,
    body: SeekRequest,
    service: PlaybackControlService = Depends(get_playback_control_service),
) -> PlaybackStatusResponse:
    status = await service.seek(task_id, body.frame_index)
    return _map_status(status)
