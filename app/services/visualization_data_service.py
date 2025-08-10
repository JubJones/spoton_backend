"""
Comprehensive Visualization Data Processing Service

Orchestrates all visualization data processing for frontend integration including:
- Multi-camera frame synchronization and composition
- Real-time tracking data aggregation
- Focus tracking state management
- Performance optimization and caching
- Data format standardization
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from app.api.v1.visualization_schemas import (
    MultiCameraVisualizationUpdate,
    CameraVisualizationData,
    EnhancedPersonTrack,
    BoundingBoxData,
    MapCoordinates,
    PersonAppearanceData,
    PersonMovementMetrics,
    FocusTrackState,
    LiveAnalyticsData,
    OccupancyMetrics,
    MovementMetrics,
    PerformanceMetrics
)
from app.domains.visualization.entities.visual_frame import VisualFrame
from app.domains.visualization.entities.overlay_config import OverlayConfig
from app.domains.visualization.services.frame_composition_service import FrameCompositionService
from app.domains.visualization.services.image_caching_service import ImageCachingService
from app.domains.detection.entities.detection import Detection
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.reid.entities.person_identity import PersonIdentity
from app.infrastructure.cache.tracking_cache import TrackingCache

logger = logging.getLogger(__name__)


class VisualizationDataService:
    """Comprehensive service for processing and managing visualization data."""
    
    def __init__(
        self,
        frame_composition_service: FrameCompositionService,
        image_caching_service: ImageCachingService,
        tracking_cache: TrackingCache
    ):
        self.frame_service = frame_composition_service
        self.image_cache = image_caching_service
        self.tracking_cache = tracking_cache
        
        # Active tracking state
        self.active_tracking_sessions: Dict[str, Dict[str, Any]] = {}
        self.focus_states: Dict[str, FocusTrackState] = {}
        
        # Performance tracking
        self.processing_metrics = defaultdict(list)
        self.frame_sync_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Real-time analytics data
        self.analytics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.person_movement_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.sync_tolerance_ms = 100.0  # Frame synchronization tolerance
        self.max_person_history = 500  # Max history per person
        
        logger.info("VisualizationDataService initialized")
    
    # --- Multi-Camera Frame Processing ---
    
    async def process_multi_camera_frame(
        self,
        task_id: str,
        frame_data: Dict[str, Any],
        detections: Dict[str, List[Detection]],
        person_identities: Dict[str, List[PersonIdentity]],
        map_coordinates: Dict[str, List[Coordinate]]
    ) -> MultiCameraVisualizationUpdate:
        """Process multi-camera frame data into visualization format."""
        start_time = time.time()
        
        try:
            # Get or create tracking session
            if task_id not in self.active_tracking_sessions:
                self.active_tracking_sessions[task_id] = {
                    'start_time': datetime.utcnow(),
                    'frame_count': 0,
                    'person_registry': {},
                    'last_update': datetime.utcnow()
                }
            
            session = self.active_tracking_sessions[task_id]
            session['frame_count'] += 1
            session['last_update'] = datetime.utcnow()
            
            # Get frame index and timestamp
            global_frame_index = frame_data.get('global_frame_index', session['frame_count'])
            timestamp_processed = datetime.utcnow()
            
            # Process each camera's data
            camera_viz_data = {}
            total_processing_time = 0.0
            
            for camera_id in frame_data.get('cameras', {}):
                camera_frame_data = frame_data['cameras'][camera_id]
                camera_detections = detections.get(camera_id, [])
                camera_identities = person_identities.get(camera_id, [])
                camera_coordinates = map_coordinates.get(camera_id, [])
                
                # Process individual camera
                viz_data = await self._process_camera_frame(
                    task_id=task_id,
                    camera_id=camera_id,
                    frame_data=camera_frame_data,
                    detections=camera_detections,
                    person_identities=camera_identities,
                    map_coordinates=camera_coordinates,
                    global_frame_index=global_frame_index
                )
                
                camera_viz_data[camera_id] = viz_data
                total_processing_time += viz_data.processing_time_ms
            
            # Synchronize frames across cameras
            await self._synchronize_camera_frames(task_id, camera_viz_data)
            
            # Get current focus state
            focus_state = self.focus_states.get(task_id)
            focused_person_id = focus_state.focused_person_id if focus_state else None
            
            # Count total persons and active cameras
            total_persons = sum(len(cam_data.tracks) for cam_data in camera_viz_data.values())
            active_cameras = list(camera_viz_data.keys())
            
            # Calculate synchronization offset
            sync_offset = await self._calculate_sync_offset(camera_viz_data)
            
            # Create multi-camera update
            multi_camera_update = MultiCameraVisualizationUpdate(
                global_frame_index=global_frame_index,
                scene_id=frame_data.get('scene_id', 'default'),
                timestamp_processed_utc=timestamp_processed,
                cameras=camera_viz_data,
                total_person_count=total_persons,
                focused_person_id=focused_person_id,
                active_cameras=active_cameras,
                total_processing_time_ms=total_processing_time,
                synchronization_offset_ms=sync_offset
            )
            
            # Update analytics data
            await self._update_analytics_data(task_id, multi_camera_update)
            
            # Track performance
            total_time_ms = (time.time() - start_time) * 1000
            self.processing_metrics[task_id].append(total_time_ms)
            if len(self.processing_metrics[task_id]) > 100:
                self.processing_metrics[task_id].pop(0)
            
            logger.debug(
                f"Processed multi-camera frame for task {task_id}: "
                f"{len(camera_viz_data)} cameras, {total_persons} persons, "
                f"{total_time_ms:.1f}ms total"
            )
            
            return multi_camera_update
            
        except Exception as e:
            logger.error(f"Error processing multi-camera frame: {e}")
            raise
    
    async def _process_camera_frame(
        self,
        task_id: str,
        camera_id: str,
        frame_data: Dict[str, Any],
        detections: List[Detection],
        person_identities: List[PersonIdentity],
        map_coordinates: List[Coordinate],
        global_frame_index: int
    ) -> CameraVisualizationData:
        """Process individual camera frame data."""
        try:
            # Get frame information
            frame_source = frame_data.get('image_source', f'frame_{global_frame_index}.jpg')
            frame_bytes = frame_data.get('frame_data', b'')
            frame_timestamp = datetime.fromisoformat(
                frame_data.get('timestamp', datetime.utcnow().isoformat())
            )
            
            # Get focus state
            focus_state = self.focus_states.get(task_id)
            focused_person_id = focus_state.focused_person_id if focus_state else None
            
            # Create overlay config
            overlay_config = await self._get_overlay_config(task_id, camera_id)
            
            # Compose visual frame
            visual_frame = self.frame_service.compose_visual_frame(
                camera_id=camera_id,
                frame_index=global_frame_index,
                frame_data=frame_bytes,
                detections=detections,
                overlay_config=overlay_config,
                focused_person_id=focused_person_id
            )
            
            # Cache the visual frame
            await self.image_cache.cache_visual_frame(visual_frame)
            
            # Convert detections to enhanced person tracks
            enhanced_tracks = []
            for i, detection in enumerate(detections):
                # Find corresponding identity and coordinates
                person_identity = None
                map_coord = None
                
                if i < len(person_identities):
                    person_identity = person_identities[i]
                
                if i < len(map_coordinates):
                    map_coord = map_coordinates[i]
                
                # Create enhanced track
                enhanced_track = await self._create_enhanced_person_track(
                    detection=detection,
                    person_identity=person_identity,
                    map_coordinate=map_coord,
                    visual_frame=visual_frame,
                    camera_id=camera_id,
                    focused_person_id=focused_person_id
                )
                
                enhanced_tracks.append(enhanced_track)
            
            # Calculate performance metrics
            fps = await self._calculate_camera_fps(task_id, camera_id)
            quality_score = await self._calculate_frame_quality(visual_frame)
            
            # Create camera visualization data
            camera_viz_data = CameraVisualizationData(
                camera_id=camera_id,
                image_source=frame_source,
                frame_image_base64=visual_frame.to_base64(),
                frame_width=visual_frame.original_width,
                frame_height=visual_frame.original_height,
                frame_timestamp=frame_timestamp,
                tracks=enhanced_tracks,
                person_count=len(enhanced_tracks),
                processing_time_ms=visual_frame.processing_time_ms,
                fps=fps,
                quality_score=quality_score,
                active_overlays=overlay_config.get_active_overlay_types(),
                overlay_opacity=overlay_config.overlay_opacity
            )
            
            return camera_viz_data
            
        except Exception as e:
            logger.error(f"Error processing camera frame for {camera_id}: {e}")
            raise
    
    async def _create_enhanced_person_track(
        self,
        detection: Detection,
        person_identity: Optional[PersonIdentity],
        map_coordinate: Optional[Coordinate],
        visual_frame: VisualFrame,
        camera_id: str,
        focused_person_id: Optional[str]
    ) -> EnhancedPersonTrack:
        """Create enhanced person track with all visualization data."""
        try:
            # Get global person ID
            global_id = (
                person_identity.global_id if person_identity 
                else f"person_{detection.track_id}_{camera_id}"
            )
            
            # Create bounding box data
            bbox_data = BoundingBoxData(
                x1=detection.bbox[0],
                y1=detection.bbox[1],
                x2=detection.bbox[2],
                y2=detection.bbox[3],
                confidence=detection.confidence,
                highlight=(global_id == focused_person_id)
            )
            
            # Create map coordinates
            map_coords = None
            if map_coordinate:
                map_coords = MapCoordinates(
                    x=map_coordinate.x,
                    y=map_coordinate.y,
                    confidence=getattr(map_coordinate, 'confidence', 1.0)
                )
            
            # Get appearance data
            appearance_data = None
            cropped_image = visual_frame.cropped_persons.get(global_id)
            if cropped_image:
                appearance_data = PersonAppearanceData(
                    cropped_image_base64=cropped_image.to_data_uri(),
                    image_format=cropped_image.image_format,
                    image_quality=cropped_image.image_quality,
                    appearance_confidence=cropped_image.confidence
                )
            
            # Get movement metrics
            movement_metrics = await self._get_person_movement_metrics(global_id)
            
            # Get tracking duration and cameras seen
            tracking_duration = 0.0
            cameras_seen = [camera_id]
            last_seen_time = detection.timestamp
            
            if person_identity:
                # Calculate tracking duration
                if hasattr(person_identity, 'first_seen'):
                    duration_delta = detection.timestamp - person_identity.first_seen
                    tracking_duration = duration_delta.total_seconds()
                
                # Get cameras seen
                if hasattr(person_identity, 'cameras_seen'):
                    cameras_seen = list(person_identity.cameras_seen)
                
                # Get last seen time
                if hasattr(person_identity, 'last_seen'):
                    last_seen_time = person_identity.last_seen
            
            # Create enhanced person track
            enhanced_track = EnhancedPersonTrack(
                track_id=detection.track_id,
                global_id=global_id,
                bbox=bbox_data,
                map_coords=map_coords,
                detection_time=detection.timestamp,
                last_seen_time=last_seen_time,
                tracking_duration=tracking_duration,
                is_focused=(global_id == focused_person_id),
                is_active=True,
                current_camera=camera_id,
                appearance=appearance_data,
                movement_metrics=movement_metrics,
                cameras_seen=cameras_seen,
                handoff_confidence=getattr(person_identity, 'confidence', 1.0) if person_identity else 1.0
            )
            
            # Update person movement data
            await self._update_person_movement_data(global_id, enhanced_track)
            
            return enhanced_track
            
        except Exception as e:
            logger.error(f"Error creating enhanced person track: {e}")
            raise
    
    # --- Focus Tracking Management ---
    
    async def set_focus_person(
        self,
        task_id: str,
        person_id: Optional[str],
        focus_mode: str = "single_person",
        highlight_color: str = "#FF0000"
    ) -> FocusTrackState:
        """Set focus on a specific person."""
        try:
            # Create or update focus state
            focus_state = FocusTrackState(
                task_id=task_id,
                focused_person_id=person_id,
                focus_mode=focus_mode,
                focus_start_time=datetime.utcnow() if person_id else None,
                highlight_color=highlight_color,
                cross_camera_sync=True,
                show_trajectory=True,
                auto_follow=True
            )
            
            # Get person details if focusing
            if person_id:
                person_details = await self._get_person_detailed_info(person_id)
                focus_state.person_details = person_details
            
            # Store focus state
            self.focus_states[task_id] = focus_state
            
            logger.info(f"Set focus person {person_id} for task {task_id}")
            return focus_state
            
        except Exception as e:
            logger.error(f"Error setting focus person: {e}")
            raise
    
    async def clear_focus(self, task_id: str) -> FocusTrackState:
        """Clear focus for a task."""
        return await self.set_focus_person(task_id, None)
    
    async def get_focus_state(self, task_id: str) -> Optional[FocusTrackState]:
        """Get current focus state for a task."""
        return self.focus_states.get(task_id)
    
    # --- Real-Time Analytics ---
    
    async def get_live_analytics(self, environment_id: str) -> LiveAnalyticsData:
        """Get current live analytics data."""
        try:
            current_time = datetime.utcnow()
            
            # Aggregate occupancy metrics from all active sessions
            occupancy_metrics = await self._calculate_occupancy_metrics(environment_id)
            
            # Calculate movement metrics
            movement_metrics = await self._calculate_movement_metrics(environment_id)
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()
            
            # Get alerts and warnings
            alerts = await self._generate_alerts(environment_id)
            warnings = await self._generate_warnings(environment_id)
            
            # Create live analytics data
            analytics_data = LiveAnalyticsData(
                environment_id=environment_id,
                timestamp=current_time,
                occupancy=occupancy_metrics,
                movement=movement_metrics,
                performance=performance_metrics,
                alerts=alerts,
                warnings=warnings,
                trend_data=await self._calculate_trend_data(environment_id)
            )
            
            # Store in history
            self.analytics_history[environment_id].append({
                'timestamp': current_time,
                'data': analytics_data.dict()
            })
            
            return analytics_data
            
        except Exception as e:
            logger.error(f"Error getting live analytics: {e}")
            raise
    
    # --- Helper Methods ---
    
    async def _synchronize_camera_frames(
        self,
        task_id: str,
        camera_data: Dict[str, CameraVisualizationData]
    ):
        """Synchronize frames across cameras."""
        if len(camera_data) <= 1:
            return
        
        # Get timestamps
        timestamps = [
            data.frame_timestamp.timestamp() 
            for data in camera_data.values()
        ]
        
        # Calculate sync offset
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)
        sync_offset = (max_timestamp - min_timestamp) * 1000  # ms
        
        # Store sync data
        self.frame_sync_data[task_id] = {
            'sync_offset_ms': sync_offset,
            'camera_count': len(camera_data),
            'timestamp': datetime.utcnow()
        }
        
        if sync_offset > self.sync_tolerance_ms:
            logger.warning(
                f"Frame sync offset {sync_offset:.1f}ms exceeds tolerance "
                f"{self.sync_tolerance_ms}ms for task {task_id}"
            )
    
    async def _calculate_sync_offset(
        self,
        camera_data: Dict[str, CameraVisualizationData]
    ) -> float:
        """Calculate synchronization offset between cameras."""
        if len(camera_data) <= 1:
            return 0.0
        
        timestamps = [
            data.frame_timestamp.timestamp() 
            for data in camera_data.values()
        ]
        
        return (max(timestamps) - min(timestamps)) * 1000  # ms
    
    async def _get_overlay_config(self, task_id: str, camera_id: str) -> OverlayConfig:
        """Get overlay configuration for a camera."""
        # This would typically load from database or cache
        # For now, return default config
        return OverlayConfig(
            show_bounding_boxes=True,
            show_person_id=True,
            show_confidence=False,
            show_tracking_duration=False,
            overlay_opacity=0.8,
            overlay_quality=85
        )
    
    async def _calculate_camera_fps(self, task_id: str, camera_id: str) -> float:
        """Calculate current FPS for a camera."""
        # This would track frame timestamps and calculate FPS
        # For now, return default value
        return 25.0
    
    async def _calculate_frame_quality(self, visual_frame: VisualFrame) -> float:
        """Calculate frame quality score."""
        # This would analyze frame quality metrics
        # For now, return based on processing success
        return 0.95 if visual_frame.processing_time_ms < 100 else 0.8
    
    async def _get_person_movement_metrics(self, person_id: str) -> PersonMovementMetrics:
        """Get movement metrics for a person."""
        person_data = self.person_movement_data.get(person_id, {})
        
        return PersonMovementMetrics(
            total_distance=person_data.get('total_distance', 0.0),
            average_speed=person_data.get('average_speed', 0.0),
            dwell_time=person_data.get('dwell_time', 0.0),
            direction_changes=person_data.get('direction_changes', 0),
            trajectory_smoothness=person_data.get('trajectory_smoothness', 1.0)
        )
    
    async def _update_person_movement_data(
        self,
        person_id: str,
        track: EnhancedPersonTrack
    ):
        """Update movement data for a person."""
        if person_id not in self.person_movement_data:
            self.person_movement_data[person_id] = {
                'positions': deque(maxlen=self.max_person_history),
                'total_distance': 0.0,
                'last_position': None,
                'direction_changes': 0,
                'last_direction': None
            }
        
        person_data = self.person_movement_data[person_id]
        
        # Add current position
        current_pos = {
            'timestamp': track.detection_time,
            'camera': track.current_camera,
            'bbox_center': [
                (track.bbox.x1 + track.bbox.x2) / 2,
                (track.bbox.y1 + track.bbox.y2) / 2
            ]
        }
        
        if track.map_coords:
            current_pos['map_coords'] = [track.map_coords.x, track.map_coords.y]
        
        person_data['positions'].append(current_pos)
        
        # Calculate movement metrics
        if len(person_data['positions']) > 1:
            await self._calculate_person_movement_metrics(person_id, person_data)
    
    async def _calculate_person_movement_metrics(self, person_id: str, person_data: Dict[str, Any]):
        """Calculate detailed movement metrics for a person."""
        positions = list(person_data['positions'])
        
        if len(positions) < 2:
            return
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(positions)):
            if 'map_coords' in positions[i] and 'map_coords' in positions[i-1]:
                curr_pos = positions[i]['map_coords']
                prev_pos = positions[i-1]['map_coords']
                
                # Euclidean distance
                distance = ((curr_pos[0] - prev_pos[0]) ** 2 + 
                           (curr_pos[1] - prev_pos[1]) ** 2) ** 0.5
                total_distance += distance
        
        person_data['total_distance'] = total_distance
        
        # Calculate average speed
        if len(positions) > 1:
            time_diff = (positions[-1]['timestamp'] - positions[0]['timestamp']).total_seconds()
            if time_diff > 0:
                person_data['average_speed'] = total_distance / time_diff
    
    async def _get_person_detailed_info(self, person_id: str):
        """Get detailed information for a person."""
        # This would query the database for comprehensive person info
        # For now, return None (would be populated by tracking service)
        return None
    
    async def _update_analytics_data(
        self,
        task_id: str,
        multi_camera_update: MultiCameraVisualizationUpdate
    ):
        """Update analytics data with latest frame information."""
        # Update occupancy metrics
        total_persons = multi_camera_update.total_person_count
        
        # Store analytics point
        analytics_point = {
            'timestamp': multi_camera_update.timestamp_processed_utc,
            'task_id': task_id,
            'total_persons': total_persons,
            'active_cameras': len(multi_camera_update.active_cameras),
            'processing_time': multi_camera_update.total_processing_time_ms
        }
        
        # Add to history
        environment_id = multi_camera_update.scene_id
        if environment_id not in self.analytics_history:
            self.analytics_history[environment_id] = deque(maxlen=1000)
        
        self.analytics_history[environment_id].append(analytics_point)
    
    async def _calculate_occupancy_metrics(self, environment_id: str) -> OccupancyMetrics:
        """Calculate occupancy metrics for environment."""
        # Get recent data
        recent_data = list(self.analytics_history.get(environment_id, []))
        
        if not recent_data:
            return OccupancyMetrics()
        
        # Get latest data
        latest = recent_data[-1] if recent_data else {}
        total_persons = latest.get('total_persons', 0)
        
        # Calculate peak occupancy
        peak_occupancy = max(
            (point.get('total_persons', 0) for point in recent_data),
            default=0
        )
        
        return OccupancyMetrics(
            total_persons=total_persons,
            persons_per_camera={},  # Would be populated from session data
            zone_occupancy={},      # Would be populated from zone analysis
            occupancy_trend="stable",
            peak_occupancy=peak_occupancy
        )
    
    async def _calculate_movement_metrics(self, environment_id: str) -> MovementMetrics:
        """Calculate movement metrics for environment."""
        # Calculate from person movement data
        total_speed = 0.0
        person_count = 0
        
        for person_data in self.person_movement_data.values():
            if 'average_speed' in person_data:
                total_speed += person_data['average_speed']
                person_count += 1
        
        avg_speed = total_speed / person_count if person_count > 0 else 0.0
        
        return MovementMetrics(
            average_speed=avg_speed,
            movement_density=0.5,  # Would be calculated from occupancy and movement
            congestion_areas=[],   # Would be calculated from zone analysis
            flow_patterns={}       # Would be calculated from trajectory analysis
        )
    
    async def _get_performance_metrics(self) -> PerformanceMetrics:
        """Get system performance metrics."""
        # Calculate average processing time across all tasks
        all_times = []
        for task_times in self.processing_metrics.values():
            all_times.extend(task_times)
        
        avg_processing_time = sum(all_times) / len(all_times) if all_times else 0.0
        
        return PerformanceMetrics(
            avg_processing_time_ms=avg_processing_time,
            total_frames_processed=sum(len(times) for times in self.processing_metrics.values()),
            frames_per_second=25.0,  # Would be calculated from actual FPS
            memory_usage_mb=0.0,     # Would be retrieved from system metrics
            gpu_utilization=0.0,     # Would be retrieved from GPU metrics
            active_connections=len(self.active_tracking_sessions)
        )
    
    async def _generate_alerts(self, environment_id: str) -> List[Dict[str, Any]]:
        """Generate system alerts."""
        alerts = []
        
        # Check for high processing times
        recent_times = []
        for task_times in self.processing_metrics.values():
            recent_times.extend(task_times[-10:])  # Last 10 samples
        
        if recent_times and sum(recent_times) / len(recent_times) > 200:
            alerts.append({
                'type': 'high_processing_time',
                'severity': 'warning',
                'message': 'High frame processing time detected',
                'value': sum(recent_times) / len(recent_times)
            })
        
        return alerts
    
    async def _generate_warnings(self, environment_id: str) -> List[Dict[str, Any]]:
        """Generate system warnings."""
        warnings = []
        
        # Check for sync issues
        for task_id, sync_data in self.frame_sync_data.items():
            if sync_data.get('sync_offset_ms', 0) > self.sync_tolerance_ms:
                warnings.append({
                    'type': 'frame_sync_issue',
                    'task_id': task_id,
                    'message': f'Frame synchronization offset: {sync_data["sync_offset_ms"]:.1f}ms',
                    'value': sync_data['sync_offset_ms']
                })
        
        return warnings
    
    async def _calculate_trend_data(self, environment_id: str) -> Dict[str, Any]:
        """Calculate trend data for analytics."""
        recent_data = list(self.analytics_history.get(environment_id, []))
        
        if len(recent_data) < 2:
            return {}
        
        # Calculate occupancy trend
        recent_occupancy = [point.get('total_persons', 0) for point in recent_data[-10:]]
        if len(recent_occupancy) >= 2:
            trend = "increasing" if recent_occupancy[-1] > recent_occupancy[0] else "decreasing"
            if abs(recent_occupancy[-1] - recent_occupancy[0]) <= 1:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            'occupancy_trend': trend,
            'data_points': len(recent_data),
            'time_range_minutes': (
                (recent_data[-1]['timestamp'] - recent_data[0]['timestamp']).total_seconds() / 60
                if len(recent_data) > 1 else 0
            )
        }
    
    # --- Status and Cleanup ---
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'active_sessions': len(self.active_tracking_sessions),
            'focus_states': len(self.focus_states),
            'analytics_environments': len(self.analytics_history),
            'person_tracking': len(self.person_movement_data),
            'performance_metrics': {
                'tasks_with_metrics': len(self.processing_metrics),
                'total_processing_samples': sum(
                    len(times) for times in self.processing_metrics.values()
                ),
                'frame_sync_issues': len(self.frame_sync_data)
            },
            'cache_status': self.image_cache.get_cache_statistics() if self.image_cache else {},
            'frame_service_stats': self.frame_service.get_performance_stats()
        }
    
    async def cleanup_session(self, task_id: str):
        """Clean up data for a completed session."""
        # Remove from active sessions
        if task_id in self.active_tracking_sessions:
            del self.active_tracking_sessions[task_id]
        
        # Clear focus state
        if task_id in self.focus_states:
            del self.focus_states[task_id]
        
        # Clear performance metrics
        if task_id in self.processing_metrics:
            del self.processing_metrics[task_id]
        
        # Clear frame sync data
        if task_id in self.frame_sync_data:
            del self.frame_sync_data[task_id]
        
        logger.info(f"Cleaned up visualization data for session {task_id}")


# Global service instance
_visualization_data_service: Optional[VisualizationDataService] = None


def get_visualization_data_service() -> Optional[VisualizationDataService]:
    """Get the global visualization data service instance."""
    return _visualization_data_service


def initialize_visualization_data_service(
    frame_composition_service: FrameCompositionService,
    image_caching_service: ImageCachingService,
    tracking_cache: TrackingCache
) -> VisualizationDataService:
    """Initialize the global visualization data service."""
    global _visualization_data_service
    if _visualization_data_service is None:
        _visualization_data_service = VisualizationDataService(
            frame_composition_service, image_caching_service, tracking_cache
        )
    return _visualization_data_service