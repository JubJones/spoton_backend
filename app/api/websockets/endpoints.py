"""
Enhanced WebSocket endpoints with binary frame support.

Provides:
- Binary frame streaming endpoints
- Tracking update endpoints
- System status monitoring endpoints
- Connection management
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState
import json

from app.api.websockets import (
    binary_websocket_manager,
    frame_handler,
    tracking_handler,
    status_handler,
    MessageType
)
from app.api.websockets.focus_handler import focus_tracking_handler
from app.api.websockets.analytics_handler import analytics_handler
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/tracking/{task_id}")
async def websocket_tracking_endpoint(websocket: WebSocket, task_id: str):
    """
    Enhanced WebSocket endpoint for real-time tracking updates.
    
    Supports:
    - Binary frame transmission
    - JSON tracking updates
    - Message compression
    - Connection health monitoring
    """
    logger.info(f"WebSocket tracking connection request for task_id: {task_id}")
    
    connected = False
    
    try:
        # Connect to binary WebSocket manager
        connected = await binary_websocket_manager.connect(websocket, task_id)
        
        if not connected:
            logger.error(f"Failed to connect WebSocket for task_id: {task_id}")
            await websocket.close(code=1011, reason="Connection failed")
            return
        
        logger.info(f"WebSocket tracking connection established for task_id: {task_id}")
        
        # Wait briefly to ensure WebSocket is fully ready
        await asyncio.sleep(0.05)
        
        # Start status monitoring if not already running
        if not status_handler.monitoring_active:
            await status_handler.start_monitoring()
        
        # Send initial connection message with retry logic
        connection_message_sent = False
        for attempt in range(3):
            try:
                success = await binary_websocket_manager.send_json_message(
                    task_id,
                    {
                        "type": "connection_established",
                        "task_id": task_id,
                        "capabilities": [
                            "binary_frames",
                            "tracking_updates",
                            "system_status",
                            "message_compression"
                        ]
                    },
                    MessageType.CONTROL_MESSAGE
                )
                if success:
                    connection_message_sent = True
                    break
                else:
                    logger.warning(f"Failed to send connection message, attempt {attempt + 1}/3")
                    await asyncio.sleep(0.02 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Error sending connection message attempt {attempt + 1}: {e}")
                await asyncio.sleep(0.02 * (attempt + 1))
        
        if not connection_message_sent:
            logger.error(f"Failed to send initial connection message after 3 attempts for task_id: {task_id}")
        
        # Main message loop - handle both interactive and listen-only clients
        while True:
            try:
                # Check WebSocket state before receiving
                from fastapi.websockets import WebSocketState
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.info(f"WebSocket no longer connected (state: {websocket.client_state}) for task_id: {task_id}")
                    break
                
                # Receive message from client with timeout, but handle listen-only clients gracefully
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Parse client message
                    try:
                        message = json.loads(data)
                        await handle_client_message(task_id, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from client: {data}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Handle listen-only clients - no message received is normal for streaming clients
                    logger.debug(f"No client message received in 30s for task_id {task_id} (listen-only client)")
                    # Keep connection alive for listen-only clients
                    continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected gracefully for task_id: {task_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message loop for task_id {task_id}: {e}")
                # Check if it's the specific race condition error
                if "Need to call 'accept' first" in str(e):
                    logger.error(f"WebSocket race condition detected for task_id {task_id}, terminating connection")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket tracking endpoint for task_id {task_id}: {e}")
        
    finally:
        # Cleanup connection
        if connected:
            await binary_websocket_manager.disconnect(websocket, task_id)
        
        logger.info(f"WebSocket tracking connection closed for task_id: {task_id}")


@router.websocket("/frames/{task_id}")
async def websocket_frames_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint specifically for binary frame streaming.
    
    Optimized for:
    - High-frequency frame transmission
    - Binary data streaming
    - Adaptive quality control
    - Frame synchronization
    """
    logger.info(f"WebSocket frames connection request for task_id: {task_id}")
    
    connected = False
    
    try:
        # Connect to binary WebSocket manager
        connected = await binary_websocket_manager.connect(websocket, task_id)
        
        if not connected:
            logger.error(f"Failed to connect frames WebSocket for task_id: {task_id}")
            await websocket.close(code=1011, reason="Connection failed")
            return
        
        logger.info(f"WebSocket frames connection established for task_id: {task_id}")
        
        # Wait briefly to ensure WebSocket is fully ready
        await asyncio.sleep(0.05)
        
        # Send initial frame capabilities with retry logic
        for attempt in range(3):
            try:
                success = await binary_websocket_manager.send_json_message(
                    task_id,
                    {
                        "type": "frame_capabilities",
                        "task_id": task_id,
                        "supported_formats": ["jpeg", "png"],
                        "compression_enabled": True,
                        "adaptive_quality": True,
                        "frame_sync": True
                    },
                    MessageType.CONTROL_MESSAGE
                )
                if success:
                    break
                else:
                    logger.warning(f"Failed to send frame capabilities, attempt {attempt + 1}/3")
                    await asyncio.sleep(0.02 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Error sending frame capabilities attempt {attempt + 1}: {e}")
                await asyncio.sleep(0.02 * (attempt + 1))
        
        # Frame streaming loop - handle both interactive and listen-only clients
        while True:
            try:
                # Check WebSocket state before receiving
                from fastapi.websockets import WebSocketState
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.info(f"WebSocket frames no longer connected for task_id: {task_id}")
                    break
                
                # Receive control messages from client with timeout handling
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Parse client message
                    try:
                        message = json.loads(data)
                        await handle_frame_control_message(task_id, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from frames client: {data}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Handle listen-only clients - no message received is normal for streaming clients
                    logger.debug(f"No frame control message received in 30s for task_id {task_id} (listen-only client)")
                    # Keep connection alive for listen-only clients
                    continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket frames client disconnected gracefully for task_id: {task_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket frames loop for task_id {task_id}: {e}")
                if "Need to call 'accept' first" in str(e):
                    logger.error(f"WebSocket race condition detected in frames for task_id {task_id}")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket frames endpoint for task_id {task_id}: {e}")
        
    finally:
        # Cleanup connection
        if connected:
            await binary_websocket_manager.disconnect(websocket, task_id)
        
        logger.info(f"WebSocket frames connection closed for task_id: {task_id}")


@router.websocket("/system")
async def websocket_system_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for system status monitoring.
    
    Provides:
    - Real-time system metrics
    - Health status updates
    - Performance monitoring
    - Alert notifications
    """
    logger.info("WebSocket system connection request")
    
    # Generate system connection ID
    system_task_id = "system_monitoring"
    connected = False
    
    try:
        # Connect to binary WebSocket manager
        connected = await binary_websocket_manager.connect(websocket, system_task_id)
        
        if not connected:
            logger.error("Failed to connect system WebSocket")
            await websocket.close(code=1011, reason="Connection failed")
            return
        
        logger.info("WebSocket system connection established")
        
        # Wait briefly to ensure WebSocket is fully ready
        await asyncio.sleep(0.05)
        
        # Start system monitoring
        await status_handler.start_monitoring()
        
        # Send initial system status with retry logic
        for attempt in range(3):
            try:
                system_status = await status_handler.get_system_status()
                success = await binary_websocket_manager.send_json_message(
                    system_task_id,
                    {
                        "type": "system_status",
                        "data": system_status
                    },
                    MessageType.SYSTEM_STATUS
                )
                if success:
                    break
                else:
                    logger.warning(f"Failed to send system status, attempt {attempt + 1}/3")
                    await asyncio.sleep(0.02 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Error sending system status attempt {attempt + 1}: {e}")
                await asyncio.sleep(0.02 * (attempt + 1))
        
        # System monitoring loop - handle both interactive and listen-only clients  
        while True:
            try:
                # Check WebSocket state before receiving
                from fastapi.websockets import WebSocketState
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.info("WebSocket system no longer connected")
                    break
                
                # Receive control messages from client with timeout handling
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Parse client message
                    try:
                        message = json.loads(data)
                        await handle_system_control_message(message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from system client: {data}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Handle listen-only clients - no message received is normal for monitoring clients
                    logger.debug("No system control message received in 30s (listen-only client)")
                    # Keep connection alive for listen-only clients
                    continue
                
            except WebSocketDisconnect:
                logger.info("WebSocket system client disconnected gracefully")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket system loop: {e}")
                if "Need to call 'accept' first" in str(e):
                    logger.error("WebSocket race condition detected in system endpoint")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket system endpoint: {e}")
        
    finally:
        # Cleanup connection
        if connected:
            await binary_websocket_manager.disconnect(websocket, system_task_id)
        
        logger.info("WebSocket system connection closed")


@router.websocket("/focus/{task_id}")
async def websocket_focus_tracking_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for focus tracking functionality.
    
    Provides:
    - Real-time focus updates
    - Person highlight controls
    - Focus state management
    - Interactive person selection
    """
    logger.info(f"WebSocket focus tracking connection request for task_id: {task_id}")
    
    connected = False
    
    try:
        # Connect to binary WebSocket manager
        connected = await binary_websocket_manager.connect(websocket, task_id)
        
        if not connected:
            logger.error(f"Failed to connect focus WebSocket for task_id: {task_id}")
            await websocket.close(code=1011, reason="Connection failed")
            return
        
        logger.info(f"WebSocket focus tracking connection established for task_id: {task_id}")
        
        # Wait briefly to ensure WebSocket is fully ready
        await asyncio.sleep(0.05)
        
        # Initialize focus tracking for this task
        await focus_tracking_handler.handle_focus_connection(task_id)
        
        # Main message loop
        while True:
            try:
                # Check WebSocket state before receiving
                from fastapi.websockets import WebSocketState
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.info(f"WebSocket focus no longer connected for task_id: {task_id}")
                    break
                
                # Receive message from client with extended timeout for slow processing
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Parse client message
                    try:
                        message = json.loads(data)
                        await focus_tracking_handler.handle_client_message(task_id, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from focus client: {data}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Handle listen-only clients - no message received is normal for focus clients
                    logger.debug(f"No focus message received in 30s for task_id {task_id} (listen-only client)")
                    # Keep connection alive for listen-only clients
                    continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket focus client disconnected gracefully for task_id: {task_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket focus loop for task_id {task_id}: {e}")
                if "Need to call 'accept' first" in str(e):
                    logger.error(f"WebSocket race condition detected in focus for task_id {task_id}")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket focus endpoint for task_id {task_id}: {e}")
        
    finally:
        # Cleanup connection
        if connected:
            await binary_websocket_manager.disconnect(websocket, task_id)
            await focus_tracking_handler.handle_focus_disconnection(task_id)
        
        logger.info(f"WebSocket focus tracking connection closed for task_id: {task_id}")


@router.websocket("/detection-tracking/{task_id}")
async def websocket_detection_tracking_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for RT-DETR detection processing (Phase 1).
    
    Provides:
    - Real-time detection results from RT-DETR model
    - Person detection bounding boxes and confidence scores
    - Frame streaming with detection annotations
    - Detection processing status updates
    """
    logger.info(f"WebSocket detection tracking connection request for task_id: {task_id}")
    
    connected = False
    
    try:
        # Connect to binary WebSocket manager
        connected = await binary_websocket_manager.connect(websocket, task_id)
        
        if not connected:
            logger.error(f"Failed to connect detection tracking WebSocket for task_id: {task_id}")
            await websocket.close(code=1011, reason="Connection failed")
            return
        
        logger.info(f"WebSocket detection tracking connection established for task_id: {task_id}")
        
        # Wait briefly to ensure WebSocket is fully ready
        await asyncio.sleep(0.05)
        
        # Send initial connection message with detection capabilities
        connection_message_sent = False
        for attempt in range(3):
            try:
                success = await binary_websocket_manager.send_json_message(
                    task_id,
                    {
                        "type": "detection_connection_established",
                        "task_id": task_id,
                        "mode": "detection_processing",
                        "capabilities": [
                            "rtdetr_detection",
                            "person_detection",
                            "detection_frames",
                            "detection_metadata",
                            "confidence_scoring"
                        ],
                        "model_info": {
                            "model_type": "RT-DETR",
                            "model_variant": "rtdetr-l",
                            "confidence_threshold": 0.5,
                            "supported_classes": ["person"]
                        }
                    },
                    MessageType.CONTROL_MESSAGE
                )
                if success:
                    connection_message_sent = True
                    break
                else:
                    logger.warning(f"Failed to send detection connection message, attempt {attempt + 1}/3")
                    await asyncio.sleep(0.02 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Error sending detection connection message attempt {attempt + 1}: {e}")
                await asyncio.sleep(0.02 * (attempt + 1))
        
        if not connection_message_sent:
            logger.error(f"Failed to send initial detection connection message after 3 attempts for task_id: {task_id}")
        
        # Main message loop - handle both interactive and listen-only clients
        while True:
            try:
                # Check WebSocket state before receiving
                from fastapi.websockets import WebSocketState
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.info(f"WebSocket detection tracking no longer connected (state: {websocket.client_state}) for task_id: {task_id}")
                    break
                
                # Receive message from client with timeout, but handle listen-only clients gracefully
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Parse client message
                    try:
                        message = json.loads(data)
                        await handle_detection_client_message(task_id, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from detection client: {data}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Handle listen-only clients - no message received is normal for streaming clients
                    logger.debug(f"No detection client message received in 30s for task_id {task_id} (listen-only client)")
                    # Keep connection alive for listen-only clients
                    continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket detection client disconnected gracefully for task_id: {task_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket detection message loop for task_id {task_id}: {e}")
                # Check if it's the specific race condition error
                if "Need to call 'accept' first" in str(e):
                    logger.error(f"WebSocket race condition detected for detection task_id {task_id}, terminating connection")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket detection tracking endpoint for task_id {task_id}: {e}")
        
    finally:
        # Cleanup connection
        if connected:
            await binary_websocket_manager.disconnect(websocket, task_id)
        
        logger.info(f"WebSocket detection tracking connection closed for task_id: {task_id}")


@router.websocket("/raw-tracking/{task_id}")
async def websocket_raw_tracking_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for raw video streaming without AI processing.
    
    Provides:
    - Raw frame streaming from multiple cameras
    - No AI processing (detection, tracking, re-identification)
    - Direct video-to-frontend streaming
    - Basic frame metadata
    """
    logger.info(f"WebSocket raw tracking connection request for task_id: {task_id}")
    
    connected = False
    
    try:
        # Connect to binary WebSocket manager
        connected = await binary_websocket_manager.connect(websocket, task_id)
        
        if not connected:
            logger.error(f"Failed to connect raw tracking WebSocket for task_id: {task_id}")
            await websocket.close(code=1011, reason="Connection failed")
            return
        
        logger.info(f"WebSocket raw tracking connection established for task_id: {task_id}")
        
        # Wait briefly to ensure WebSocket is fully ready
        await asyncio.sleep(0.05)
        
        # Send initial connection message with raw streaming capabilities
        connection_message_sent = False
        for attempt in range(3):
            try:
                success = await binary_websocket_manager.send_json_message(
                    task_id,
                    {
                        "type": "raw_connection_established",
                        "task_id": task_id,
                        "mode": "raw_streaming",
                        "capabilities": [
                            "raw_frames",
                            "no_ai_processing",
                            "multi_camera_streaming",
                            "basic_metadata"
                        ]
                    },
                    MessageType.CONTROL_MESSAGE
                )
                if success:
                    connection_message_sent = True
                    break
                else:
                    logger.warning(f"Failed to send raw connection message, attempt {attempt + 1}/3")
                    await asyncio.sleep(0.02 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Error sending raw connection message attempt {attempt + 1}: {e}")
                await asyncio.sleep(0.02 * (attempt + 1))
        
        if not connection_message_sent:
            logger.error(f"Failed to send initial raw connection message after 3 attempts for task_id: {task_id}")
        
        # Main message loop - handle both interactive and listen-only clients
        while True:
            try:
                # Check WebSocket state before receiving
                from fastapi.websockets import WebSocketState
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.info(f"WebSocket raw tracking no longer connected (state: {websocket.client_state}) for task_id: {task_id}")
                    break
                
                # Receive message from client with timeout, but handle listen-only clients gracefully
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Parse client message
                    try:
                        message = json.loads(data)
                        await handle_raw_client_message(task_id, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from raw client: {data}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Handle listen-only clients - no message received is normal for streaming clients
                    logger.debug(f"No raw client message received in 30s for task_id {task_id} (listen-only client)")
                    # Keep connection alive for listen-only clients
                    continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket raw client disconnected gracefully for task_id: {task_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket raw message loop for task_id {task_id}: {e}")
                # Check if it's the specific race condition error
                if "Need to call 'accept' first" in str(e):
                    logger.error(f"WebSocket race condition detected for raw task_id {task_id}, terminating connection")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket raw tracking endpoint for task_id {task_id}: {e}")
        
    finally:
        # Cleanup connection
        if connected:
            await binary_websocket_manager.disconnect(websocket, task_id)
        
        logger.info(f"WebSocket raw tracking connection closed for task_id: {task_id}")


@router.websocket("/analytics/{task_id}")
async def websocket_analytics_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time analytics data.
    
    Provides:
    - Live performance metrics
    - System health monitoring
    - Processing statistics
    - Camera analytics
    """
    logger.info(f"WebSocket analytics connection request for task_id: {task_id}")
    
    connected = False
    
    try:
        # Connect to binary WebSocket manager
        connected = await binary_websocket_manager.connect(websocket, task_id)
        
        if not connected:
            logger.error(f"Failed to connect analytics WebSocket for task_id: {task_id}")
            await websocket.close(code=1011, reason="Connection failed")
            return
        
        logger.info(f"WebSocket analytics connection established for task_id: {task_id}")
        
        # Wait briefly to ensure WebSocket is fully ready
        await asyncio.sleep(0.05)
        
        # Initialize analytics for this task
        await analytics_handler.handle_analytics_connection(task_id)
        
        # Main message loop
        while True:
            try:
                # Check WebSocket state before receiving
                from fastapi.websockets import WebSocketState
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.info(f"WebSocket analytics no longer connected for task_id: {task_id}")
                    break
                
                # Receive message from client with extended timeout for slow processing
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Parse client message
                    try:
                        message = json.loads(data)
                        await analytics_handler.handle_client_message(task_id, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from analytics client: {data}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Handle listen-only clients - no message received is normal for analytics clients
                    logger.debug(f"No analytics message received in 30s for task_id {task_id} (listen-only client)")
                    # Keep connection alive for listen-only clients
                    continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket analytics client disconnected gracefully for task_id: {task_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket analytics loop for task_id {task_id}: {e}")
                if "Need to call 'accept' first" in str(e):
                    logger.error(f"WebSocket race condition detected in analytics for task_id {task_id}")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket analytics endpoint for task_id {task_id}: {e}")
        
    finally:
        # Cleanup connection
        if connected:
            await binary_websocket_manager.disconnect(websocket, task_id)
            await analytics_handler.handle_analytics_disconnection(task_id)
        
        logger.info(f"WebSocket analytics connection closed for task_id: {task_id}")


async def handle_client_message(task_id: str, message: Dict[str, Any]):
    """Handle messages from tracking WebSocket clients."""
    try:
        message_type = message.get("type")
        
        if message_type == "ping":
            # Respond to ping
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                },
                MessageType.CONTROL_MESSAGE
            )
            
        elif message_type == "subscribe_tracking":
            # Subscribe to tracking updates
            logger.info(f"Client subscribed to tracking updates for task_id: {task_id}")
            
        elif message_type == "unsubscribe_tracking":
            # Unsubscribe from tracking updates
            logger.info(f"Client unsubscribed from tracking updates for task_id: {task_id}")
            
        elif message_type == "request_status":
            # Send current tracking status
            active_persons = tracking_handler.get_active_persons(task_id)
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "tracking_status",
                    "active_persons": active_persons,
                    "stats": tracking_handler.get_tracking_stats()
                },
                MessageType.TRACKING_UPDATE
            )
            
        else:
            logger.warning(f"Unknown message type received: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling client message: {e}")


async def handle_frame_control_message(task_id: str, message: Dict[str, Any]):
    """Handle control messages from frame WebSocket clients."""
    try:
        message_type = message.get("type")
        
        if message_type == "set_quality":
            # Set frame quality
            quality = message.get("quality", 85)
            logger.info(f"Setting frame quality to {quality} for task_id: {task_id}")
            # This would be implemented to adjust frame handler quality
            
        elif message_type == "enable_compression":
            # Enable/disable compression
            enabled = message.get("enabled", True)
            logger.info(f"Setting compression to {enabled} for task_id: {task_id}")
            
        elif message_type == "request_frame_stats":
            # Send frame statistics
            stats = frame_handler.get_encoding_stats()
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "frame_stats",
                    "stats": stats
                },
                MessageType.CONTROL_MESSAGE
            )
            
        else:
            logger.warning(f"Unknown frame control message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling frame control message: {e}")


async def handle_system_control_message(message: Dict[str, Any]):
    """Handle control messages from system WebSocket clients."""
    try:
        message_type = message.get("type")
        
        if message_type == "set_alert_threshold":
            # Set alert threshold
            metric = message.get("metric")
            threshold = message.get("threshold")
            
            if metric and threshold is not None:
                status_handler.set_alert_threshold(metric, threshold)
                logger.info(f"Updated alert threshold for {metric}: {threshold}")
            
        elif message_type == "request_performance_history":
            # Send performance history
            history = status_handler.get_performance_history()
            await binary_websocket_manager.send_json_message(
                "system_monitoring",
                {
                    "type": "performance_history",
                    "history": history
                },
                MessageType.SYSTEM_STATUS
            )
            
        elif message_type == "reset_stats":
            # Reset performance statistics
            status_handler.reset_performance_history()
            logger.info("Performance statistics reset")
            
        else:
            logger.warning(f"Unknown system control message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling system control message: {e}")


async def handle_detection_client_message(task_id: str, message: Dict[str, Any]):
    """Handle messages from detection tracking WebSocket clients."""
    try:
        message_type = message.get("type")
        
        if message_type == "ping":
            # Respond to ping with detection mode
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "pong",
                    "timestamp": message.get("timestamp"),
                    "mode": "detection_processing",
                    "model": "RT-DETR-l"
                },
                MessageType.CONTROL_MESSAGE
            )
            
        elif message_type == "subscribe_detection_updates":
            # Subscribe to detection updates
            logger.info(f"Client subscribed to detection updates for task_id: {task_id}")
            
        elif message_type == "unsubscribe_detection_updates":
            # Unsubscribe from detection updates
            logger.info(f"Client unsubscribed from detection updates for task_id: {task_id}")
            
        elif message_type == "request_detection_status":
            # Send current detection processing status
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "detection_status",
                    "task_id": task_id,
                    "mode": "detection_processing",
                    "model": "RT-DETR-l",
                    "status": "active",
                    "message": "RT-DETR person detection active",
                    "capabilities": [
                        "person_detection",
                        "confidence_scoring", 
                        "bounding_boxes",
                        "detection_metadata",
                        "frame_annotation",
                        "base64_encoding",
                        "realtime_streaming",
                        "progress_tracking"
                    ],
                    "phase": "Phase 2: Core Detection Pipeline"
                },
                MessageType.STATUS_UPDATE
            )
            
        elif message_type == "set_detection_confidence":
            # Set detection confidence threshold
            confidence = message.get("confidence", 0.5)
            logger.info(f"Setting detection confidence threshold to {confidence} for task_id: {task_id}")
            
        elif message_type == "request_detection_stats":
            # Send detection statistics (Phase 2: Enhanced stats)
            from app.core.dependencies import get_detection_video_service
            detection_service = get_detection_video_service()
            stats = detection_service.get_detection_stats()
            
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "detection_statistics",
                    "task_id": task_id,
                    "phase": "Phase 2: Core Detection Pipeline",
                    "stats": {
                        "model_type": "RT-DETR-l",
                        "confidence_threshold": 0.5,
                        "total_frames_processed": stats.get("total_frames_processed", 0),
                        "total_detections_found": stats.get("total_detections_found", 0),
                        "frames_annotated": stats.get("frames_annotated", 0),
                        "websocket_messages_sent": stats.get("websocket_messages_sent", 0),
                        "average_detection_time_ms": stats.get("average_detection_time", 0.0),
                        "average_annotation_time_ms": stats.get("annotation_time", 0.0),
                        "successful_detections": stats.get("successful_detections", 0),
                        "failed_detections": stats.get("failed_detections", 0),
                        "detector_loaded": stats.get("detector_loaded", False)
                    }
                },
                MessageType.CONTROL_MESSAGE
            )
            
        elif message_type == "request_frame_overlay":
            # Phase 2: Request frame overlay capabilities
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "frame_overlay_info",
                    "task_id": task_id,
                    "overlay_capabilities": {
                        "original_frame_encoding": "base64_jpeg",
                        "annotated_frame_encoding": "base64_jpeg",
                        "annotation_features": [
                            "bounding_boxes",
                            "confidence_scores",
                            "person_labels",
                            "detection_ids"
                        ],
                        "supported_qualities": [30, 50, 70, 85, 95],
                        "default_quality": 85
                    },
                    "phase": "Phase 2: Core Detection Pipeline"
                },
                MessageType.CONTROL_MESSAGE
            )
            
        else:
            logger.warning(f"Unknown detection client message type received: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling detection client message: {e}")


async def handle_raw_client_message(task_id: str, message: Dict[str, Any]):
    """Handle messages from raw tracking WebSocket clients."""
    try:
        message_type = message.get("type")
        
        if message_type == "ping":
            # Respond to ping
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "pong",
                    "timestamp": message.get("timestamp"),
                    "mode": "raw_streaming"
                },
                MessageType.CONTROL_MESSAGE
            )
            
        elif message_type == "subscribe_raw_frames":
            # Subscribe to raw frame updates
            logger.info(f"Client subscribed to raw frame updates for task_id: {task_id}")
            
        elif message_type == "unsubscribe_raw_frames":
            # Unsubscribe from raw frame updates
            logger.info(f"Client unsubscribed from raw frame updates for task_id: {task_id}")
            
        elif message_type == "request_raw_status":
            # Send current raw streaming status
            await binary_websocket_manager.send_json_message(
                task_id,
                {
                    "type": "raw_streaming_status",
                    "task_id": task_id,
                    "mode": "raw_streaming",
                    "status": "active",
                    "message": "Raw video streaming active - no AI processing"
                },
                MessageType.STATUS_UPDATE
            )
            
        elif message_type == "set_raw_quality":
            # Set raw frame quality
            quality = message.get("quality", 85)
            logger.info(f"Setting raw frame quality to {quality} for task_id: {task_id}")
            
        else:
            logger.warning(f"Unknown raw client message type received: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling raw client message: {e}")


# Health check endpoint for WebSocket status
@router.get("/health")
async def websocket_health():
    """Get WebSocket system health status."""
    try:
        performance_stats = binary_websocket_manager.get_performance_stats()
        system_status = await status_handler.get_system_status()
        
        return {
            "status": "healthy",
            "websocket_connections": performance_stats.get("active_connections", 0),
            "system_health": system_status.get("health_status", "unknown"),
            "monitoring_active": status_handler.monitoring_active,
            "frame_handler_stats": frame_handler.get_encoding_stats(),
            "tracking_handler_stats": tracking_handler.get_tracking_stats(),
            "focus_handler_stats": focus_tracking_handler.get_focus_statistics(),
            "analytics_handler_stats": analytics_handler.get_analytics_statistics()
        }
        
    except Exception as e:
        logger.error(f"Error getting WebSocket health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoint
@router.get("/stats")
async def websocket_stats():
    """Get comprehensive WebSocket statistics."""
    try:
        return {
            "websocket_manager": binary_websocket_manager.get_performance_stats(),
            "frame_handler": frame_handler.get_encoding_stats(),
            "tracking_handler": tracking_handler.get_tracking_stats(),
            "focus_handler": focus_tracking_handler.get_focus_statistics(),
            "analytics_handler": analytics_handler.get_analytics_statistics(),
            "system_status": await status_handler.get_system_status()
        }
        
    except Exception as e:
        logger.error(f"Error getting WebSocket statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))