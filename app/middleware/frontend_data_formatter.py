"""
Frontend Data Format Standardization Middleware

Ensures consistent data formatting for frontend consumption including:
- Standardized response formats
- Date/time normalization
- Numeric precision control
- Error message formatting
- Metadata injection
- Performance metrics inclusion
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from decimal import Decimal
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class FrontendDataFormatter:
    """Utility class for standardizing data formats for frontend consumption."""
    
    @staticmethod
    def format_datetime(dt: Union[datetime, str, None]) -> Optional[str]:
        """Standardize datetime formatting to ISO format with UTC timezone."""
        if dt is None:
            return None
        
        if isinstance(dt, str):
            try:
                # Try to parse the string and reformat
                parsed_dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                return parsed_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
            except ValueError:
                # Return as-is if parsing fails
                return dt
        
        if isinstance(dt, datetime):
            # Convert to UTC and format
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        return str(dt)
    
    @staticmethod
    def format_numeric(value: Union[int, float, Decimal, None], precision: int = 2) -> Optional[float]:
        """Standardize numeric formatting with consistent precision."""
        if value is None:
            return None
        
        if isinstance(value, (int, float, Decimal)):
            try:
                return round(float(value), precision)
            except (ValueError, TypeError):
                return None
        
        try:
            return round(float(value), precision)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def format_coordinates(coords: Union[List, tuple, None]) -> Optional[List[float]]:
        """Standardize coordinate formatting."""
        if coords is None:
            return None
        
        try:
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                return [
                    FrontendDataFormatter.format_numeric(coords[0], precision=3),
                    FrontendDataFormatter.format_numeric(coords[1], precision=3)
                ]
        except (IndexError, TypeError, ValueError):
            pass
        
        return None
    
    @staticmethod
    def format_bbox(bbox: Union[List, tuple, None]) -> Optional[List[float]]:
        """Standardize bounding box formatting [x1, y1, x2, y2]."""
        if bbox is None:
            return None
        
        try:
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return [
                    FrontendDataFormatter.format_numeric(bbox[0], precision=1),
                    FrontendDataFormatter.format_numeric(bbox[1], precision=1),
                    FrontendDataFormatter.format_numeric(bbox[2], precision=1),
                    FrontendDataFormatter.format_numeric(bbox[3], precision=1)
                ]
        except (IndexError, TypeError, ValueError):
            pass
        
        return None
    
    @staticmethod
    def format_confidence(confidence: Union[float, int, None]) -> Optional[float]:
        """Standardize confidence score formatting (0.0 to 1.0)."""
        if confidence is None:
            return None
        
        try:
            conf_value = float(confidence)
            # Clamp between 0.0 and 1.0
            return max(0.0, min(1.0, round(conf_value, 3)))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def format_person_track(track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize person track data for frontend."""
        formatted_track = {}
        
        # Required fields with standardization
        formatted_track['track_id'] = track_data.get('track_id', 0)
        formatted_track['global_id'] = str(track_data.get('global_id', ''))
        
        # Bounding box
        bbox = track_data.get('bbox_xyxy') or track_data.get('bbox')
        formatted_track['bbox_xyxy'] = FrontendDataFormatter.format_bbox(bbox)
        
        # Confidence
        formatted_track['confidence'] = FrontendDataFormatter.format_confidence(
            track_data.get('confidence')
        )
        
        # Map coordinates
        map_coords = track_data.get('map_coords') or track_data.get('map_coordinates')
        formatted_track['map_coords'] = FrontendDataFormatter.format_coordinates(map_coords)
        
        # Timestamps
        formatted_track['detection_time'] = FrontendDataFormatter.format_datetime(
            track_data.get('detection_time')
        )
        
        # Optional fields
        formatted_track['is_focused'] = bool(track_data.get('is_focused', False))
        formatted_track['tracking_duration'] = FrontendDataFormatter.format_numeric(
            track_data.get('tracking_duration', 0.0)
        )
        
        # Class information
        if 'class_id' in track_data:
            formatted_track['class_id'] = int(track_data['class_id'])
        
        return formatted_track
    
    @staticmethod
    def format_camera_data(camera_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize camera data for frontend."""
        formatted_camera = {}
        
        # Basic camera information
        formatted_camera['camera_id'] = str(camera_data.get('camera_id', ''))
        formatted_camera['image_source'] = str(camera_data.get('image_source', ''))
        
        # Frame data
        if 'frame_image_base64' in camera_data:
            formatted_camera['frame_image_base64'] = camera_data['frame_image_base64']
        
        # Cropped persons
        cropped_persons = camera_data.get('cropped_persons', {})
        if isinstance(cropped_persons, dict):
            formatted_camera['cropped_persons'] = cropped_persons
        else:
            formatted_camera['cropped_persons'] = {}
        
        # Tracks
        tracks = camera_data.get('tracks', [])
        formatted_camera['tracks'] = [
            FrontendDataFormatter.format_person_track(track)
            for track in tracks
            if isinstance(track, dict)
        ]
        
        # Performance metrics
        formatted_camera['processing_time_ms'] = FrontendDataFormatter.format_numeric(
            camera_data.get('processing_time_ms', 0.0)
        )
        
        formatted_camera['fps'] = FrontendDataFormatter.format_numeric(
            camera_data.get('fps', 0.0)
        )
        
        return formatted_camera
    
    @staticmethod
    def format_tracking_update(update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize complete tracking update for frontend."""
        formatted_update = {
            'type': 'tracking_update',
            'payload': {}
        }
        
        payload = update_data.get('payload', {})
        formatted_payload = {}
        
        # Frame information
        formatted_payload['global_frame_index'] = int(payload.get('global_frame_index', 0))
        formatted_payload['scene_id'] = str(payload.get('scene_id', ''))
        formatted_payload['timestamp_processed_utc'] = FrontendDataFormatter.format_datetime(
            payload.get('timestamp_processed_utc')
        )
        
        # Camera data
        cameras = payload.get('cameras', {})
        formatted_cameras = {}
        for camera_id, camera_data in cameras.items():
            if isinstance(camera_data, dict):
                formatted_cameras[str(camera_id)] = FrontendDataFormatter.format_camera_data(camera_data)
        formatted_payload['cameras'] = formatted_cameras
        
        # Person count per camera
        person_count = payload.get('person_count_per_camera', {})
        formatted_payload['person_count_per_camera'] = {
            str(k): int(v) for k, v in person_count.items()
            if isinstance(v, (int, float))
        }
        
        # Focus information
        focused_person_id = payload.get('focus_person_id') or payload.get('focused_person_id')
        formatted_payload['focus_person_id'] = str(focused_person_id) if focused_person_id else None
        
        # Performance metrics
        formatted_payload['total_processing_time_ms'] = FrontendDataFormatter.format_numeric(
            payload.get('total_processing_time_ms', 0.0)
        )
        
        formatted_payload['synchronization_offset_ms'] = FrontendDataFormatter.format_numeric(
            payload.get('synchronization_offset_ms', 0.0)
        )
        
        formatted_update['payload'] = formatted_payload
        return formatted_update
    
    @staticmethod
    def format_analytics_data(analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize analytics data for frontend."""
        formatted_analytics = {}
        
        # Basic information
        formatted_analytics['environment_id'] = str(analytics_data.get('environment_id', ''))
        formatted_analytics['timestamp'] = FrontendDataFormatter.format_datetime(
            analytics_data.get('timestamp')
        )
        
        # Occupancy metrics
        if 'occupancy' in analytics_data:
            occupancy = analytics_data['occupancy']
            formatted_analytics['occupancy'] = {
                'total_persons': int(occupancy.get('total_persons', 0)),
                'persons_per_camera': {
                    str(k): int(v) for k, v in occupancy.get('persons_per_camera', {}).items()
                },
                'zone_occupancy': {
                    str(k): int(v) for k, v in occupancy.get('zone_occupancy', {}).items()
                },
                'occupancy_trend': str(occupancy.get('occupancy_trend', 'stable')),
                'peak_occupancy': int(occupancy.get('peak_occupancy', 0))
            }
        
        # Movement metrics
        if 'movement' in analytics_data:
            movement = analytics_data['movement']
            formatted_analytics['movement'] = {
                'average_speed': FrontendDataFormatter.format_numeric(
                    movement.get('average_speed', 0.0)
                ),
                'movement_density': FrontendDataFormatter.format_numeric(
                    movement.get('movement_density', 0.0)
                ),
                'congestion_areas': movement.get('congestion_areas', []),
                'flow_patterns': movement.get('flow_patterns', {})
            }
        
        # Performance metrics
        if 'performance' in analytics_data:
            performance = analytics_data['performance']
            formatted_analytics['performance'] = {
                'avg_processing_time_ms': FrontendDataFormatter.format_numeric(
                    performance.get('avg_processing_time_ms', 0.0)
                ),
                'total_frames_processed': int(performance.get('total_frames_processed', 0)),
                'frames_per_second': FrontendDataFormatter.format_numeric(
                    performance.get('frames_per_second', 0.0)
                ),
                'memory_usage_mb': FrontendDataFormatter.format_numeric(
                    performance.get('memory_usage_mb', 0.0)
                ),
                'gpu_utilization': FrontendDataFormatter.format_numeric(
                    performance.get('gpu_utilization', 0.0)
                ),
                'active_connections': int(performance.get('active_connections', 0))
            }
        
        # Alerts and warnings
        formatted_analytics['alerts'] = analytics_data.get('alerts', [])
        formatted_analytics['warnings'] = analytics_data.get('warnings', [])
        formatted_analytics['trend_data'] = analytics_data.get('trend_data', {})
        
        return formatted_analytics
    
    @staticmethod
    def format_error_response(
        error: Exception,
        request_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Standardize error response format for frontend."""
        error_response = {
            'error': True,
            'error_type': error.__class__.__name__,
            'message': str(error),
            'timestamp': FrontendDataFormatter.format_datetime(datetime.now(timezone.utc)),
            'request_id': request_id
        }
        
        if additional_context:
            error_response['context'] = additional_context
        
        return error_response
    
    @staticmethod
    def format_success_response(
        data: Any,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Standardize success response format for frontend."""
        response = {
            'success': True,
            'data': data,
            'timestamp': FrontendDataFormatter.format_datetime(datetime.now(timezone.utc))
        }
        
        if message:
            response['message'] = message
        
        if metadata:
            response['metadata'] = metadata
        
        return response
    
    @staticmethod
    def recursively_format_data(data: Any) -> Any:
        """Recursively format data structure for frontend consistency."""
        if isinstance(data, dict):
            formatted_dict = {}
            for key, value in data.items():
                # Format datetime fields
                if key.endswith('_time') or key.endswith('_timestamp') or key == 'timestamp':
                    formatted_dict[key] = FrontendDataFormatter.format_datetime(value)
                # Format coordinate fields
                elif key.endswith('_coords') or key.endswith('_coordinates'):
                    formatted_dict[key] = FrontendDataFormatter.format_coordinates(value)
                # Format bounding box fields
                elif key.endswith('_bbox') or key == 'bbox' or key == 'bbox_xyxy':
                    formatted_dict[key] = FrontendDataFormatter.format_bbox(value)
                # Format confidence fields
                elif key.endswith('_confidence') or key == 'confidence':
                    formatted_dict[key] = FrontendDataFormatter.format_confidence(value)
                # Format numeric fields
                elif key.endswith('_ms') or key.endswith('_seconds') or key.endswith('_percent'):
                    formatted_dict[key] = FrontendDataFormatter.format_numeric(value)
                else:
                    formatted_dict[key] = FrontendDataFormatter.recursively_format_data(value)
            return formatted_dict
        
        elif isinstance(data, list):
            return [FrontendDataFormatter.recursively_format_data(item) for item in data]
        
        elif isinstance(data, datetime):
            return FrontendDataFormatter.format_datetime(data)
        
        elif isinstance(data, (int, float, Decimal)):
            return FrontendDataFormatter.format_numeric(data)
        
        else:
            return data


class FrontendDataFormatterMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically format API responses for frontend consumption."""
    
    def __init__(self, app, include_performance_metrics: bool = True):
        super().__init__(app)
        self.include_performance_metrics = include_performance_metrics
        
        # Paths that should be formatted
        self.format_paths = {
            '/api/v1/',
            '/ws/',
            '/analytics/',
            '/tracking/',
            '/media/'
        }
        
        # Paths that should be excluded from formatting
        self.exclude_paths = {
            '/health',
            '/metrics',
            '/docs',
            '/openapi.json'
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request and format response for frontend."""
        request_start_time = time.time()
        request_id = f"req_{int(request_start_time * 1000)}"
        
        try:
            # Check if this path should be formatted
            should_format = self._should_format_response(request.url.path)
            
            # Process the request
            response = await call_next(request)
            
            # Only format JSON responses
            if (should_format and 
                isinstance(response, (Response, JSONResponse)) and
                response.headers.get('content-type', '').startswith('application/json')):
                
                # Get response body
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                
                try:
                    # Parse JSON response
                    response_data = json.loads(response_body.decode())
                    
                    # Format the response data
                    formatted_data = self._format_response_data(
                        response_data, request, request_start_time, request_id
                    )
                    
                    # Create new response with formatted data
                    new_response = JSONResponse(
                        content=formatted_data,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
                    
                    # Add custom headers
                    new_response.headers["X-Request-ID"] = request_id
                    new_response.headers["X-Formatted"] = "true"
                    
                    return new_response
                
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    # Return original response if parsing fails
                    return response
            
            return response
            
        except Exception as e:
            logger.error(f"Error in FrontendDataFormatterMiddleware: {e}")
            
            # Return standardized error response
            error_response = FrontendDataFormatter.format_error_response(
                error=e,
                request_id=request_id,
                additional_context={
                    'path': request.url.path,
                    'method': request.method
                }
            )
            
            return JSONResponse(
                content=error_response,
                status_code=500,
                headers={"X-Request-ID": request_id}
            )
    
    def _should_format_response(self, path: str) -> bool:
        """Check if response should be formatted."""
        # Check excluded paths first
        for excluded in self.exclude_paths:
            if path.startswith(excluded):
                return False
        
        # Check included paths
        for included in self.format_paths:
            if path.startswith(included):
                return True
        
        return False
    
    def _format_response_data(
        self,
        response_data: Dict[str, Any],
        request: Request,
        request_start_time: float,
        request_id: str
    ) -> Dict[str, Any]:
        """Format response data for frontend consumption."""
        try:
            # Check if this is already a formatted response
            if isinstance(response_data, dict) and 'success' in response_data:
                formatted_data = response_data
            else:
                # Wrap in success response format
                formatted_data = FrontendDataFormatter.format_success_response(response_data)
            
            # Apply recursive formatting
            formatted_data['data'] = FrontendDataFormatter.recursively_format_data(
                formatted_data.get('data', response_data)
            )
            
            # Add metadata
            if 'metadata' not in formatted_data:
                formatted_data['metadata'] = {}
            
            formatted_data['metadata'].update({
                'request_id': request_id,
                'path': request.url.path,
                'method': request.method,
                'formatted_at': FrontendDataFormatter.format_datetime(datetime.now(timezone.utc))
            })
            
            # Add performance metrics if enabled
            if self.include_performance_metrics:
                processing_time_ms = (time.time() - request_start_time) * 1000
                formatted_data['metadata']['performance'] = {
                    'processing_time_ms': FrontendDataFormatter.format_numeric(processing_time_ms),
                    'formatted_response': True
                }
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error formatting response data: {e}")
            # Return original data if formatting fails
            return response_data
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return {
            'middleware_name': 'FrontendDataFormatterMiddleware',
            'include_performance_metrics': self.include_performance_metrics,
            'format_paths': list(self.format_paths),
            'exclude_paths': list(self.exclude_paths)
        }


class WebSocketDataFormatter:
    """Formatter specifically for WebSocket messages."""
    
    @staticmethod
    def format_websocket_message(message: Dict[str, Any]) -> Dict[str, Any]:
        """Format WebSocket message for frontend consumption."""
        try:
            message_type = message.get('type', '')
            
            # Format based on message type
            if message_type == 'tracking_update':
                return FrontendDataFormatter.format_tracking_update(message)
            
            elif message_type == 'analytics_update':
                formatted_message = {
                    'type': 'analytics_update',
                    'payload': FrontendDataFormatter.format_analytics_data(
                        message.get('payload', {})
                    )
                }
                return formatted_message
            
            elif message_type == 'focus_update':
                payload = message.get('payload', {})
                formatted_payload = {
                    'focused_person_id': str(payload.get('focused_person_id', '')),
                    'focus_mode': str(payload.get('focus_mode', '')),
                    'focus_start_time': FrontendDataFormatter.format_datetime(
                        payload.get('focus_start_time')
                    ),
                    'highlight_color': str(payload.get('highlight_color', '#FF0000')),
                    'cross_camera_sync': bool(payload.get('cross_camera_sync', True)),
                    'show_trajectory': bool(payload.get('show_trajectory', True)),
                    'auto_follow': bool(payload.get('auto_follow', True))
                }
                
                if 'person_details' in payload:
                    formatted_payload['person_details'] = FrontendDataFormatter.recursively_format_data(
                        payload['person_details']
                    )
                
                return {
                    'type': 'focus_update',
                    'payload': formatted_payload
                }
            
            else:
                # Generic formatting for other message types
                return {
                    'type': message_type,
                    'payload': FrontendDataFormatter.recursively_format_data(
                        message.get('payload', {})
                    ),
                    'timestamp': FrontendDataFormatter.format_datetime(
                        message.get('timestamp', datetime.now(timezone.utc))
                    )
                }
        
        except Exception as e:
            logger.error(f"Error formatting WebSocket message: {e}")
            # Return original message if formatting fails
            return message
    
    @staticmethod
    def add_websocket_metadata(
        message: Dict[str, Any],
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add metadata to WebSocket message."""
        if 'metadata' not in message:
            message['metadata'] = {}
        
        message['metadata'].update({
            'formatted_at': FrontendDataFormatter.format_datetime(datetime.now(timezone.utc)),
            'websocket_formatted': True
        })
        
        if connection_id:
            message['metadata']['connection_id'] = connection_id
        
        return message


# Utility functions for manual formatting
def format_api_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
    """Utility function to manually format API response."""
    return FrontendDataFormatter.format_success_response(data, message)


def format_error_response(error: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Utility function to manually format error response."""
    return FrontendDataFormatter.format_error_response(error, request_id)


def format_websocket_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Utility function to manually format WebSocket message."""
    return WebSocketDataFormatter.format_websocket_message(message)