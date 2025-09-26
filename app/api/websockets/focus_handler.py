"""
Focus Track WebSocket Handler

Handles WebSocket connections and messaging for focus track functionality.
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.api.websockets.connection_manager import binary_websocket_manager
from app.domains.interaction.entities.focus_state import FocusState, PersonDetails

logger = logging.getLogger(__name__)


class FocusTrackingHandler:
    """Handles focus tracking WebSocket communications."""
    
    def __init__(self):
        self.active_focus_states: Dict[str, FocusState] = {}
        self.focus_update_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("FocusTrackingHandler initialized")
    
    async def handle_focus_connection(self, task_id: str):
        """Handle new focus tracking connection."""
        try:
            # Initialize focus state if not exists
            if task_id not in self.active_focus_states:
                self.active_focus_states[task_id] = FocusState(task_id=task_id)
            
            # Send initial focus state
            focus_state = self.active_focus_states[task_id]
            await self.send_focus_update(task_id, focus_state)
            
            logger.info(f"Focus connection established for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error handling focus connection for task {task_id}: {e}")
    
    async def handle_focus_disconnection(self, task_id: str):
        """Handle focus tracking disconnection."""
        try:
            # Cancel update task if running
            if task_id in self.focus_update_tasks:
                self.focus_update_tasks[task_id].cancel()
                del self.focus_update_tasks[task_id]
            
            logger.info(f"Focus connection closed for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error handling focus disconnection for task {task_id}: {e}")
    
    async def set_focus_person(
        self, 
        task_id: str, 
        person_id: str,
        person_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set focus on a specific person."""
        try:
            # Get or create focus state
            if task_id not in self.active_focus_states:
                self.active_focus_states[task_id] = FocusState(task_id=task_id)
            
            focus_state = self.active_focus_states[task_id]
            
            # Set focus person
            if person_details:
                raw_first_detected = person_details.get('first_detected')
                if raw_first_detected:
                    try:
                        first_detected = datetime.fromisoformat(raw_first_detected)
                    except ValueError:
                        first_detected = datetime.now(timezone.utc)
                else:
                    first_detected = datetime.now(timezone.utc)

                person_detail_obj = PersonDetails(
                    global_id=person_id,
                    first_detected=first_detected,
                    tracking_duration=person_details.get('tracking_duration', 0.0),
                    current_camera=person_details.get('current_camera'),
                    position_history=person_details.get('position_history', []),
                    movement_metrics=person_details.get('movement_metrics', {}),
                    camera_id=person_details.get('camera_id'),
                    track_id=person_details.get('track_id'),
                    detection_id=person_details.get('detection_id'),
                    bbox=person_details.get('bbox'),
                    confidence=person_details.get('confidence'),
                )
            else:
                person_detail_obj = PersonDetails(
                    global_id=person_id,
                    first_detected=datetime.now(timezone.utc),
                    tracking_duration=0.0,
                    current_camera=None,
                )

            focus_state.set_focus_person(person_id, person_detail_obj)
            if person_details:
                focus_state.record_observation(
                    camera_id=person_details.get('camera_id', person_detail_obj.current_camera or 'unknown'),
                    bbox=person_details.get('bbox'),
                    confidence=person_details.get('confidence'),
                    detection_id=person_details.get('detection_id'),
                    track_id=person_details.get('track_id'),
                )
            
            # Send focus update
            await self.send_focus_update(task_id, focus_state)
            
            # Start continuous updates if not already running
            await self.start_focus_updates(task_id)
            
            logger.info(f"Focus set on person {person_id} for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting focus person for task {task_id}: {e}")
            return False
    
    async def clear_focus(self, task_id: str) -> bool:
        """Clear focus for a task."""
        try:
            if task_id in self.active_focus_states:
                focus_state = self.active_focus_states[task_id]
                focus_state.clear_focus()
                
                # Send focus update
                await self.send_focus_update(task_id, focus_state)
                
                # Stop continuous updates
                await self.stop_focus_updates(task_id)
                
                logger.info(f"Focus cleared for task {task_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error clearing focus for task {task_id}: {e}")
            return False
    
    async def update_person_location(
        self, 
        task_id: str, 
        person_id: str,
        camera_id: str,
        position: Dict[str, Any]
    ) -> bool:
        """Update the location of a focused person."""
        try:
            if task_id not in self.active_focus_states:
                return False
            
            focus_state = self.active_focus_states[task_id]
            
            if focus_state.focused_person_id == person_id:
                focus_state.update_person_location(camera_id, position)
                
                # Send focus update
                await self.send_focus_update(task_id, focus_state)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating person location for task {task_id}: {e}")
            return False
    
    async def send_focus_update(self, task_id: str, focus_state: FocusState):
        """Send focus update via WebSocket."""
        try:
            message = {
                "type": "focus_update",
                "payload": focus_state.get_focus_update_payload()
            }
            
            await binary_websocket_manager.send_json_message(
                task_id, 
                message,
                message_type=binary_websocket_manager.MessageType.CONTROL_MESSAGE
            )
            
            logger.debug(f"Sent focus update for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error sending focus update for task {task_id}: {e}")
    
    async def start_focus_updates(self, task_id: str):
        """Start continuous focus updates for a task."""
        try:
            # Cancel existing task if running
            if task_id in self.focus_update_tasks:
                self.focus_update_tasks[task_id].cancel()
            
            # Start new update task
            self.focus_update_tasks[task_id] = asyncio.create_task(
                self._focus_update_loop(task_id)
            )
            
            logger.debug(f"Started focus updates for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error starting focus updates for task {task_id}: {e}")
    
    async def stop_focus_updates(self, task_id: str):
        """Stop continuous focus updates for a task."""
        try:
            if task_id in self.focus_update_tasks:
                self.focus_update_tasks[task_id].cancel()
                del self.focus_update_tasks[task_id]
                
            logger.debug(f"Stopped focus updates for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error stopping focus updates for task {task_id}: {e}")
    
    async def _focus_update_loop(self, task_id: str):
        """Continuous focus update loop."""
        try:
            while True:
                if task_id in self.active_focus_states:
                    focus_state = self.active_focus_states[task_id]
                    
                    # Only send updates if there's an active focus
                    if focus_state.has_active_focus():
                        await self.send_focus_update(task_id, focus_state)
                    else:
                        break  # No active focus, stop loop
                
                # Wait before next update
                await asyncio.sleep(1.0)  # Update every 1 second
                
        except asyncio.CancelledError:
            logger.debug(f"Focus update loop cancelled for task {task_id}")
        except Exception as e:
            logger.error(f"Error in focus update loop for task {task_id}: {e}")
    
    def get_focus_state(self, task_id: str) -> Optional[FocusState]:
        """Get current focus state for a task."""
        return self.active_focus_states.get(task_id)
    
    def get_all_active_focus_states(self) -> Dict[str, FocusState]:
        """Get all active focus states."""
        return {
            task_id: state 
            for task_id, state in self.active_focus_states.items()
            if state.has_active_focus()
        }
    
    async def handle_client_message(self, task_id: str, message: Dict[str, Any]):
        """Handle messages from focus WebSocket clients."""
        try:
            message_type = message.get("type")
            
            if message_type == "set_focus":
                person_id = message.get("person_id")
                person_details = message.get("person_details")
                
                if person_id:
                    await self.set_focus_person(task_id, person_id, person_details)
                
            elif message_type == "clear_focus":
                await self.clear_focus(task_id)
                
            elif message_type == "update_highlight_settings":
                settings = message.get("settings", {})
                
                if task_id in self.active_focus_states:
                    focus_state = self.active_focus_states[task_id]
                    focus_state.update_highlight_settings(settings)
                    await self.send_focus_update(task_id, focus_state)
                
            elif message_type == "ping":
                # Respond to ping
                pong_message = {
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                }
                
                await binary_websocket_manager.send_json_message(
                    task_id,
                    pong_message,
                    message_type=binary_websocket_manager.MessageType.CONTROL_MESSAGE
                )
            
            else:
                logger.warning(f"Unknown focus message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling focus client message: {e}")
    
    def get_focus_statistics(self) -> Dict[str, Any]:
        """Get focus tracking statistics."""
        active_focus_count = len(self.get_all_active_focus_states())
        total_sessions = len(self.active_focus_states)
        running_updates = len(self.focus_update_tasks)
        
        return {
            "total_focus_sessions": total_sessions,
            "active_focus_count": active_focus_count,
            "running_update_tasks": running_updates,
            "focus_sessions": [
                {
                    "task_id": task_id,
                    "focused_person_id": state.focused_person_id,
                    "has_active_focus": state.has_active_focus(),
                    "focus_duration": state.get_focus_duration()
                }
                for task_id, state in self.active_focus_states.items()
            ]
        }
    
    async def cleanup(self):
        """Cleanup focus tracking handler resources."""
        try:
            # Cancel all update tasks
            for task in self.focus_update_tasks.values():
                task.cancel()
            
            # Wait for all tasks to be cancelled
            if self.focus_update_tasks:
                await asyncio.gather(
                    *self.focus_update_tasks.values(),
                    return_exceptions=True
                )
            
            # Clear all state
            self.focus_update_tasks.clear()
            self.active_focus_states.clear()
            
            logger.info("FocusTrackingHandler cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up focus tracking handler: {e}")


# Global focus handler instance
focus_tracking_handler = FocusTrackingHandler()
