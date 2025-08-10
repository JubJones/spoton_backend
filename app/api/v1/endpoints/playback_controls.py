"""
Playback Control API Endpoints

Provides playback control functionality for recorded video analysis
and historical data exploration.
"""
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class PlaybackState:
    """Represents the playback state for a task."""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.is_playing = False
        self.current_position = 0.0  # 0.0 to 1.0
        self.playback_speed = 1.0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_timestamp: Optional[datetime] = None
        self.duration_seconds = 0.0
        self.can_seek = True
        self.loop_enabled = False
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class PlayRequest(BaseModel):
    """Request model for play control."""
    speed: Optional[float] = 1.0


class SeekRequest(BaseModel):
    """Request model for seek control."""
    position: Optional[float] = None  # 0.0 to 1.0
    timestamp: Optional[str] = None  # ISO format timestamp


class PlaybackConfigRequest(BaseModel):
    """Request model for playback configuration."""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    loop_enabled: Optional[bool] = None


# TODO: This would be replaced with a proper dependency injection system
_playback_states: Dict[str, PlaybackState] = {}


def get_playback_state(task_id: str) -> Optional[PlaybackState]:
    """Get playback state for a task."""
    return _playback_states.get(task_id)


def create_or_update_playback_state(task_id: str, state: PlaybackState) -> PlaybackState:
    """Create or update playback state for a task."""
    _playback_states[task_id] = state
    return state


@router.post(
    "/controls/{task_id}/play",
    summary="Start or resume playback"
)
async def play_video(
    task_id: uuid.UUID,
    request: PlayRequest
):
    """
    Start or resume video playback for a task.
    
    This enables playback of recorded video data with configurable speed.
    """
    try:
        task_id_str = str(task_id)
        
        # Get or create playback state
        playback_state = get_playback_state(task_id_str)
        if playback_state is None:
            playback_state = PlaybackState(task_id_str)
            
            # TODO: Initialize with actual video duration and time range
            playback_state.duration_seconds = 3600.0  # 1 hour placeholder
            playback_state.start_time = datetime.utcnow() - timedelta(hours=1)
            playback_state.end_time = datetime.utcnow()
        
        # Start playback
        playback_state.is_playing = True
        playback_state.playback_speed = max(0.1, min(4.0, request.speed))
        playback_state.updated_at = datetime.utcnow()
        
        # Calculate current timestamp based on position
        if playback_state.start_time and playback_state.end_time:
            duration = (playback_state.end_time - playback_state.start_time).total_seconds()
            current_offset = duration * playback_state.current_position
            playback_state.current_timestamp = playback_state.start_time + timedelta(seconds=current_offset)
        
        # Save state
        create_or_update_playback_state(task_id_str, playback_state)
        
        logger.info(f"Started playback for task {task_id} at speed {request.speed}x")
        
        return {
            "message": "Playback started",
            "task_id": task_id_str,
            "is_playing": True,
            "playback_speed": playback_state.playback_speed,
            "current_position": playback_state.current_position,
            "current_timestamp": playback_state.current_timestamp.isoformat() if playback_state.current_timestamp else None,
            "duration_seconds": playback_state.duration_seconds
        }
        
    except Exception as e:
        logger.error(f"Error starting playback for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error starting playback"
        )


@router.post(
    "/controls/{task_id}/pause",
    summary="Pause playback"
)
async def pause_video(
    task_id: uuid.UUID
):
    """
    Pause video playback for a task.
    
    This stops playback while maintaining the current position.
    """
    try:
        task_id_str = str(task_id)
        playback_state = get_playback_state(task_id_str)
        
        if playback_state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No playback session found for this task"
            )
        
        playback_state.is_playing = False
        playback_state.updated_at = datetime.utcnow()
        
        # Save state
        create_or_update_playback_state(task_id_str, playback_state)
        
        logger.info(f"Paused playback for task {task_id}")
        
        return {
            "message": "Playback paused",
            "task_id": task_id_str,
            "is_playing": False,
            "current_position": playback_state.current_position,
            "current_timestamp": playback_state.current_timestamp.isoformat() if playback_state.current_timestamp else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing playback for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error pausing playback"
        )


@router.post(
    "/controls/{task_id}/stop",
    summary="Stop playback"
)
async def stop_video(
    task_id: uuid.UUID
):
    """
    Stop video playback and reset to beginning.
    
    This stops playback and resets the position to the start.
    """
    try:
        task_id_str = str(task_id)
        playback_state = get_playback_state(task_id_str)
        
        if playback_state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No playback session found for this task"
            )
        
        # Stop and reset
        playback_state.is_playing = False
        playback_state.current_position = 0.0
        playback_state.current_timestamp = playback_state.start_time
        playback_state.updated_at = datetime.utcnow()
        
        # Save state
        create_or_update_playback_state(task_id_str, playback_state)
        
        logger.info(f"Stopped playback for task {task_id}")
        
        return {
            "message": "Playback stopped and reset",
            "task_id": task_id_str,
            "is_playing": False,
            "current_position": 0.0,
            "current_timestamp": playback_state.start_time.isoformat() if playback_state.start_time else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping playback for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error stopping playback"
        )


@router.post(
    "/controls/{task_id}/seek",
    summary="Seek to specific position or timestamp"
)
async def seek_video(
    task_id: uuid.UUID,
    request: SeekRequest
):
    """
    Seek to a specific position or timestamp in the video.
    
    This allows jumping to any point in the recorded data.
    """
    try:
        task_id_str = str(task_id)
        playback_state = get_playback_state(task_id_str)
        
        if playback_state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No playback session found for this task"
            )
        
        if not playback_state.can_seek:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Seeking is not available for this session"
            )
        
        # Handle position-based seek
        if request.position is not None:
            position = max(0.0, min(1.0, request.position))
            playback_state.current_position = position
            
            # Calculate timestamp from position
            if playback_state.start_time and playback_state.end_time:
                duration = (playback_state.end_time - playback_state.start_time).total_seconds()
                current_offset = duration * position
                playback_state.current_timestamp = playback_state.start_time + timedelta(seconds=current_offset)
        
        # Handle timestamp-based seek
        elif request.timestamp is not None:
            try:
                target_timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
                
                # Calculate position from timestamp
                if playback_state.start_time and playback_state.end_time:
                    if target_timestamp < playback_state.start_time:
                        target_timestamp = playback_state.start_time
                    elif target_timestamp > playback_state.end_time:
                        target_timestamp = playback_state.end_time
                    
                    duration = (playback_state.end_time - playback_state.start_time).total_seconds()
                    offset = (target_timestamp - playback_state.start_time).total_seconds()
                    playback_state.current_position = offset / duration if duration > 0 else 0.0
                    playback_state.current_timestamp = target_timestamp
                    
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid timestamp format"
                )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either position or timestamp must be provided"
            )
        
        playback_state.updated_at = datetime.utcnow()
        
        # Save state
        create_or_update_playback_state(task_id_str, playback_state)
        
        logger.info(f"Seeked to position {playback_state.current_position} for task {task_id}")
        
        return {
            "message": "Seek completed",
            "task_id": task_id_str,
            "current_position": playback_state.current_position,
            "current_timestamp": playback_state.current_timestamp.isoformat() if playback_state.current_timestamp else None,
            "is_playing": playback_state.is_playing
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error seeking for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error performing seek operation"
        )


@router.get(
    "/controls/{task_id}/status",
    summary="Get playback status"
)
async def get_playback_status(
    task_id: uuid.UUID
):
    """
    Get the current playback status for a task.
    
    Returns detailed information about playback state and controls.
    """
    try:
        task_id_str = str(task_id)
        playback_state = get_playback_state(task_id_str)
        
        if playback_state is None:
            return {
                "task_id": task_id_str,
                "has_session": False,
                "message": "No playback session found"
            }
        
        return {
            "task_id": task_id_str,
            "has_session": True,
            "is_playing": playback_state.is_playing,
            "playback_speed": playback_state.playback_speed,
            "current_position": playback_state.current_position,
            "current_timestamp": playback_state.current_timestamp.isoformat() if playback_state.current_timestamp else None,
            "start_time": playback_state.start_time.isoformat() if playback_state.start_time else None,
            "end_time": playback_state.end_time.isoformat() if playback_state.end_time else None,
            "duration_seconds": playback_state.duration_seconds,
            "can_seek": playback_state.can_seek,
            "loop_enabled": playback_state.loop_enabled,
            "updated_at": playback_state.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting playback status for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving playback status"
        )


@router.post(
    "/controls/{task_id}/configure",
    summary="Configure playback settings"
)
async def configure_playback(
    task_id: uuid.UUID,
    config: PlaybackConfigRequest
):
    """
    Configure playback settings including time range and options.
    
    This allows setting the playback window and behavior options.
    """
    try:
        task_id_str = str(task_id)
        
        # Get or create playback state
        playback_state = get_playback_state(task_id_str)
        if playback_state is None:
            playback_state = PlaybackState(task_id_str)
        
        # Update time range
        if config.start_time:
            try:
                playback_state.start_time = datetime.fromisoformat(config.start_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid start_time format"
                )
        
        if config.end_time:
            try:
                playback_state.end_time = datetime.fromisoformat(config.end_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid end_time format"
                )
        
        # Update loop setting
        if config.loop_enabled is not None:
            playback_state.loop_enabled = config.loop_enabled
        
        # Calculate duration
        if playback_state.start_time and playback_state.end_time:
            playback_state.duration_seconds = (
                playback_state.end_time - playback_state.start_time
            ).total_seconds()
        
        playback_state.updated_at = datetime.utcnow()
        
        # Save state
        create_or_update_playback_state(task_id_str, playback_state)
        
        logger.info(f"Configured playback for task {task_id}")
        
        return {
            "message": "Playback configured successfully",
            "task_id": task_id_str,
            "start_time": playback_state.start_time.isoformat() if playback_state.start_time else None,
            "end_time": playback_state.end_time.isoformat() if playback_state.end_time else None,
            "duration_seconds": playback_state.duration_seconds,
            "loop_enabled": playback_state.loop_enabled
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring playback for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error configuring playback"
        )