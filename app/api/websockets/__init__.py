"""
WebSocket handlers for binary frame transmission and real-time updates.

This module provides:
- Binary WebSocket message protocol
- Frame encoding and transmission
- Tracking update messages
- System status messages
- Connection management
"""

from .connection_manager import BinaryWebSocketManager, MessageType, binary_websocket_manager
from .frame_handler import FrameHandler, frame_handler
from .tracking_handler import TrackingHandler, tracking_handler, CameraTransition, TrackingUpdate
from .status_handler import StatusHandler, status_handler

__all__ = [
    'BinaryWebSocketManager',
    'MessageType',
    'binary_websocket_manager',
    'FrameHandler',
    'frame_handler',
    'TrackingHandler',
    'tracking_handler',
    'CameraTransition',
    'TrackingUpdate',
    'StatusHandler',
    'status_handler'
]