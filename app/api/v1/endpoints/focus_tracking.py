"""
Focus Tracking API Endpoints

Provides focus track functionality for person-centric viewing
and interactive user control.
"""
import logging
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from app.domains.interaction.entities.focus_state import FocusState, PersonDetails

logger = logging.getLogger(__name__)
router = APIRouter()


class FocusPersonRequest(BaseModel):
    """Request model for setting focus on a person."""
    global_person_id: str
    cross_camera_sync: Optional[bool] = True
    highlight_settings: Optional[Dict[str, Any]] = None


class HighlightSettingsRequest(BaseModel):
    """Request model for updating highlight settings."""
    enabled: Optional[bool] = None
    intensity: Optional[float] = None
    border_thickness: Optional[int] = None
    border_color: Optional[list] = None
    glow_effect: Optional[bool] = None
    darken_background: Optional[bool] = None


# TODO: This would be replaced with a proper dependency injection system
_focus_states: Dict[str, FocusState] = {}


def get_focus_state(task_id: str) -> Optional[FocusState]:
    """Get focus state for a task."""
    return _focus_states.get(task_id)


def create_or_update_focus_state(task_id: str, focus_state: FocusState) -> FocusState:
    """Create or update focus state for a task."""
    _focus_states[task_id] = focus_state
    return focus_state


@router.post(
    "/focus/{task_id}",
    summary="Set focus on specific person"
)
async def set_focus_person(
    task_id: uuid.UUID,
    request: FocusPersonRequest
):
    """
    Set focus on a specific person for tracking across all camera views.
    
    This enables person-centric viewing where the selected person is
    highlighted across all cameras and detailed information is provided.
    """
    try:
        task_id_str = str(task_id)
        
        # Get or create focus state
        focus_state = get_focus_state(task_id_str)
        if focus_state is None:
            focus_state = FocusState(task_id=task_id_str)
        
        # TODO: Fetch person details from tracking system
        # For now, create placeholder person details
        person_details = PersonDetails(
            global_id=request.global_person_id,
            first_detected=focus_state.created_at,
            tracking_duration=0.0,
            current_camera="unknown",
            position_history=[],
            movement_metrics={}
        )
        
        # Set focus person
        focus_state.set_focus_person(request.global_person_id, person_details)
        focus_state.cross_camera_sync = request.cross_camera_sync
        
        # Update highlight settings if provided
        if request.highlight_settings:
            focus_state.update_highlight_settings(request.highlight_settings)
        
        # Save focus state
        create_or_update_focus_state(task_id_str, focus_state)
        
        logger.info(f"Focus set on person {request.global_person_id} for task {task_id}")
        
        return {
            "message": "Focus set successfully",
            "task_id": task_id_str,
            "focused_person_id": request.global_person_id,
            "cross_camera_sync": request.cross_camera_sync,
            "focus_state": focus_state.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error setting focus for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting focus on person"
        )


@router.get(
    "/focus/{task_id}",
    summary="Get current focus state"
)
async def get_focus_state_endpoint(
    task_id: uuid.UUID
):
    """
    Get the current focus state for a task.
    
    Returns information about which person is currently focused,
    their tracking details, and highlight settings.
    """
    try:
        task_id_str = str(task_id)
        focus_state = get_focus_state(task_id_str)
        
        if focus_state is None:
            return {
                "task_id": task_id_str,
                "focused_person_id": None,
                "has_active_focus": False,
                "focus_state": None
            }
        
        return {
            "task_id": task_id_str,
            "focused_person_id": focus_state.focused_person_id,
            "has_active_focus": focus_state.has_active_focus(),
            "focus_duration": focus_state.get_focus_duration(),
            "focus_state": focus_state.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting focus state for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving focus state"
        )


@router.delete(
    "/focus/{task_id}",
    summary="Clear focus"
)
async def clear_focus(
    task_id: uuid.UUID
):
    """
    Clear the current focus for a task.
    
    This removes person highlighting and returns to normal viewing mode.
    """
    try:
        task_id_str = str(task_id)
        focus_state = get_focus_state(task_id_str)
        
        if focus_state is None:
            return {
                "message": "No active focus to clear",
                "task_id": task_id_str
            }
        
        previous_person_id = focus_state.focused_person_id
        focus_state.clear_focus()
        
        # Save updated state
        create_or_update_focus_state(task_id_str, focus_state)
        
        logger.info(f"Cleared focus for task {task_id} (was {previous_person_id})")
        
        return {
            "message": "Focus cleared successfully",
            "task_id": task_id_str,
            "previous_person_id": previous_person_id,
            "focus_state": focus_state.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error clearing focus for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error clearing focus"
        )


@router.put(
    "/focus/{task_id}/highlight-settings",
    summary="Update highlight settings"
)
async def update_highlight_settings(
    task_id: uuid.UUID,
    settings: HighlightSettingsRequest
):
    """
    Update highlight settings for the focused person.
    
    This allows customization of visual effects applied to the focused person.
    """
    try:
        task_id_str = str(task_id)
        focus_state = get_focus_state(task_id_str)
        
        if focus_state is None or not focus_state.has_active_focus():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active focus to update"
            )
        
        # Build settings update
        settings_update = {}
        if settings.enabled is not None:
            settings_update["enabled"] = settings.enabled
        if settings.intensity is not None:
            settings_update["intensity"] = max(0.0, min(1.0, settings.intensity))
        if settings.border_thickness is not None:
            settings_update["border_thickness"] = max(1, settings.border_thickness)
        if settings.border_color is not None:
            if len(settings.border_color) == 3:
                settings_update["border_color"] = settings.border_color
        if settings.glow_effect is not None:
            settings_update["glow_effect"] = settings.glow_effect
        if settings.darken_background is not None:
            settings_update["darken_background"] = settings.darken_background
        
        # Update settings
        focus_state.update_highlight_settings(settings_update)
        
        # Save updated state
        create_or_update_focus_state(task_id_str, focus_state)
        
        logger.info(f"Updated highlight settings for task {task_id}")
        
        return {
            "message": "Highlight settings updated successfully",
            "task_id": task_id_str,
            "focused_person_id": focus_state.focused_person_id,
            "highlight_settings": focus_state.highlight_settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating highlight settings for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating highlight settings"
        )


@router.get(
    "/persons/{global_person_id}/details",
    summary="Get detailed person information"
)
async def get_person_details(
    global_person_id: str,
    include_history: Optional[bool] = False,
    history_limit: Optional[int] = 50
):
    """
    Get detailed information about a specific person.
    
    This endpoint provides comprehensive tracking statistics,
    position history, and movement analysis for a person.
    """
    try:
        # TODO: Implement person detail retrieval from tracking system
        # This would integrate with the ReID service and tracking repository
        
        # For now, return placeholder data
        person_details = {
            "global_person_id": global_person_id,
            "first_detected": "2025-01-01T10:00:00Z",
            "last_seen": "2025-01-01T10:30:00Z",
            "total_tracking_duration": 1800.0,
            "cameras_visited": ["c01", "c02", "c03"],
            "current_camera": "c03",
            "current_status": "active",
            "confidence_score": 0.89,
            "movement_metrics": {
                "average_speed": 1.2,
                "total_distance": 45.6,
                "dwell_time": 120.5,
                "direction_changes": 8
            }
        }
        
        if include_history:
            person_details["position_history"] = [
                {
                    "timestamp": "2025-01-01T10:00:00Z",
                    "camera": "c01",
                    "bbox": [100, 200, 150, 300],
                    "map_coords": [10.5, 20.3],
                    "confidence": 0.92
                }
                # TODO: Add actual position history from database
            ]
        
        return {
            "person_details": person_details,
            "timestamp": "2025-01-01T10:30:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting person details for {global_person_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving person details"
        )


@router.get(
    "/focus/active",
    summary="Get all active focus states"
)
async def get_all_active_focus_states():
    """
    Get all currently active focus states across all tasks.
    
    Useful for monitoring and debugging focus tracking.
    """
    try:
        active_states = []
        
        for task_id, focus_state in _focus_states.items():
            if focus_state.has_active_focus():
                active_states.append({
                    "task_id": task_id,
                    "focused_person_id": focus_state.focused_person_id,
                    "focus_duration": focus_state.get_focus_duration(),
                    "active_cameras": focus_state.active_cameras,
                    "last_seen_camera": focus_state.last_seen_camera
                })
        
        return {
            "active_focus_count": len(active_states),
            "active_states": active_states,
            "timestamp": "2025-01-01T10:30:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting active focus states: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving active focus states"
        )