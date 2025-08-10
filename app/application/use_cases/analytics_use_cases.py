"""
Analytics use cases for application layer.

Business logic for analytics operations including behavioral analysis,
crowd dynamics, and historical data analysis.
Maximum 350 lines per plan.
"""
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.time_range import TimeRange
from app.domain.tracking.entities.track import Track
from app.domain.mapping.value_objects.coordinates import WorldCoordinates
from app.domain.analytics.services.behavioral_analysis_service import BehavioralAnalysisService
from app.domain.analytics.services.crowd_dynamics_service import CrowdDynamicsService
from app.domain.analytics.value_objects.behavioral_metrics import (
    BehavioralAnomalyType, SocialInteractionType
)

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsRequest:
    """Request for analytics processing."""
    camera_ids: List[CameraID]
    time_range: TimeRange
    analysis_types: List[str]
    parameters: Dict[str, Any]


@dataclass
class AnalyticsResult:
    """Result of analytics operation."""
    request_id: str
    camera_ids: List[CameraID]
    time_range: TimeRange
    behavioral_analysis: Optional[Dict[str, Any]] = None
    crowd_analysis: Optional[Dict[str, Any]] = None
    anomaly_detection: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = None


class BehavioralAnalyticsUseCase:
    """
    Behavioral analytics use case for application layer.
    
    Orchestrates behavioral analysis operations including individual behavior
    patterns, social interactions, and anomaly detection.
    """
    
    def __init__(
        self,
        behavioral_service: BehavioralAnalysisService,
        crowd_service: CrowdDynamicsService
    ):
        """
        Initialize behavioral analytics use case.
        
        Args:
            behavioral_service: Behavioral analysis domain service
            crowd_service: Crowd dynamics domain service
        """
        self.behavioral_service = behavioral_service
        self.crowd_service = crowd_service
        
        # Analytics statistics
        self._analytics_stats = {
            'analyses_performed': 0,
            'behaviors_analyzed': 0,
            'anomalies_detected': 0,
            'social_interactions_found': 0,
            'avg_processing_time_ms': 0.0
        }
        
        logger.debug("BehavioralAnalyticsUseCase initialized")
    
    async def analyze_behavioral_patterns(
        self,
        request: AnalyticsRequest
    ) -> AnalyticsResult:
        """
        Analyze behavioral patterns from tracking data.
        
        Args:
            request: Analytics request with parameters
            
        Returns:
            Analytics result with behavior analysis
        """
        start_time = datetime.utcnow()
        request_id = f"analytics_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            result = AnalyticsResult(
                request_id=request_id,
                camera_ids=request.camera_ids,
                time_range=request.time_range,
                timestamp=start_time
            )
            
            # Collect tracking data for analysis
            tracks_data = await self._collect_tracking_data(
                request.camera_ids,
                request.time_range
            )
            
            # Perform behavioral analysis if requested
            if 'behavioral' in request.analysis_types:
                result.behavioral_analysis = await self._analyze_individual_behaviors(
                    tracks_data,
                    request.parameters
                )
            
            # Perform crowd analysis if requested
            if 'crowd' in request.analysis_types:
                result.crowd_analysis = await self._analyze_crowd_dynamics(
                    tracks_data,
                    request.parameters
                )
            
            # Perform anomaly detection if requested
            if 'anomaly' in request.analysis_types:
                result.anomaly_detection = await self._detect_behavioral_anomalies(
                    tracks_data,
                    request.parameters
                )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            # Update statistics
            self._update_analytics_statistics(result, processing_time)
            
            logger.info(f"Completed analytics request {request_id} in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Analytics processing failed for request {request_id}: {e}")
            raise
    
    async def analyze_social_interactions(
        self,
        camera_ids: List[CameraID],
        time_window: timedelta = timedelta(minutes=5)
    ) -> Dict[str, Any]:
        """
        Analyze social interactions between people.
        
        Args:
            camera_ids: List of cameras to analyze
            time_window: Time window for analysis
            
        Returns:
            Social interaction analysis results
        """
        try:
            # Collect current tracking data
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            time_range = TimeRange(start_time=start_time, end_time=end_time)
            
            tracks_data = await self._collect_tracking_data(camera_ids, time_range)
            
            # Analyze social interactions
            interactions = {}
            
            for camera_id, camera_tracks in tracks_data.items():
                if len(camera_tracks) < 2:
                    continue
                
                # Get world coordinates for tracks
                world_coords = await self._get_world_coordinates(camera_id, camera_tracks)
                
                # Detect social interactions
                camera_interactions = self.behavioral_service.analyze_social_interactions(
                    camera_tracks, world_coords
                )
                
                interactions[str(camera_id)] = camera_interactions
                
                # Update statistics
                self._analytics_stats['social_interactions_found'] += len(camera_interactions)
            
            return {
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'cameras_analyzed': len(camera_ids),
                'interactions_by_camera': interactions,
                'total_interactions': sum(len(interactions) for interactions in interactions.values())
            }
            
        except Exception as e:
            logger.error(f"Social interaction analysis failed: {e}")
            return {'error': str(e)}
    
    async def detect_crowd_anomalies(
        self,
        camera_ids: List[CameraID],
        density_threshold: Optional[int] = None,
        historical_baseline: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect crowd-level behavioral anomalies.
        
        Args:
            camera_ids: List of cameras to analyze
            density_threshold: Optional crowd density threshold
            historical_baseline: Optional historical baseline for comparison
            
        Returns:
            Crowd anomaly detection results
        """
        try:
            anomalies_by_camera = {}
            total_anomalies = set()
            
            for camera_id in camera_ids:
                # Get current active tracks
                current_time = datetime.utcnow()
                time_range = TimeRange(
                    start_time=current_time - timedelta(minutes=1),
                    end_time=current_time
                )
                
                tracks_data = await self._collect_tracking_data([camera_id], time_range)
                camera_tracks = tracks_data.get(camera_id, [])
                
                if not camera_tracks:
                    continue
                
                # Get world coordinates
                world_coords = await self._get_world_coordinates(camera_id, camera_tracks)
                
                # Detect anomalies
                anomalies = self.crowd_service.detect_crowd_anomalies(
                    camera_tracks, world_coords, historical_baseline
                )
                
                if anomalies:
                    anomalies_by_camera[str(camera_id)] = [anomaly.value for anomaly in anomalies]
                    total_anomalies.update(anomalies)
            
            # Update statistics
            self._analytics_stats['anomalies_detected'] += len(total_anomalies)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cameras_analyzed': len(camera_ids),
                'anomalies_by_camera': anomalies_by_camera,
                'total_anomaly_types': list(anomaly.value for anomaly in total_anomalies),
                'anomaly_count': len(total_anomalies)
            }
            
        except Exception as e:
            logger.error(f"Crowd anomaly detection failed: {e}")
            return {'error': str(e)}
    
    async def analyze_traffic_patterns(
        self,
        camera_id: CameraID,
        analysis_duration: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """
        Analyze traffic flow patterns for a camera.
        
        Args:
            camera_id: Camera to analyze
            analysis_duration: Duration of analysis
            
        Returns:
            Traffic pattern analysis
        """
        try:
            # Define time range
            end_time = datetime.utcnow()
            start_time = end_time - analysis_duration
            time_range = TimeRange(start_time=start_time, end_time=end_time)
            
            # Collect tracking data
            tracks_data = await self._collect_tracking_data([camera_id], time_range)
            camera_tracks = tracks_data.get(camera_id, [])
            
            if not camera_tracks:
                return {'error': 'No tracking data available'}
            
            # Get world coordinates
            world_coords = await self._get_world_coordinates(camera_id, camera_tracks)
            
            # Analyze flow patterns
            flow_analysis = self.crowd_service.analyze_flow_patterns(
                camera_tracks, world_coords, timedelta(minutes=5)
            )
            
            # Analyze density over time
            density_analysis = self.crowd_service.analyze_crowd_density(
                camera_tracks, world_coords
            )
            
            return {
                'camera_id': str(camera_id),
                'analysis_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': analysis_duration.total_seconds() / 3600
                },
                'total_people_detected': len(camera_tracks),
                'flow_patterns': flow_analysis,
                'density_analysis': density_analysis
            }
            
        except Exception as e:
            logger.error(f"Traffic pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get analytics processing statistics."""
        return self._analytics_stats.copy()
    
    async def _collect_tracking_data(
        self,
        camera_ids: List[CameraID],
        time_range: TimeRange
    ) -> Dict[CameraID, List[Track]]:
        """Collect tracking data for analysis (placeholder implementation)."""
        # This is a placeholder - in real implementation, this would
        # fetch actual tracking data from repositories or services
        
        tracks_data = {}
        for camera_id in camera_ids:
            # Simulate tracking data collection
            tracks_data[camera_id] = []
        
        return tracks_data
    
    async def _get_world_coordinates(
        self,
        camera_id: CameraID,
        tracks: List[Track]
    ) -> List[WorldCoordinates]:
        """Get world coordinates for tracks (placeholder implementation)."""
        # This is a placeholder - in real implementation, this would
        # convert camera coordinates to world coordinates using homography
        
        world_coords = []
        for track in tracks:
            if track.position_history:
                latest_pos = track.position_history[-1]
                # Placeholder coordinate conversion
                world_coords.append(WorldCoordinates(
                    x=latest_pos.bbox.x,
                    y=latest_pos.bbox.y
                ))
        
        return world_coords
    
    async def _analyze_individual_behaviors(
        self,
        tracks_data: Dict[CameraID, List[Track]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze individual behavioral patterns."""
        behavior_analysis = {
            'cameras_analyzed': len(tracks_data),
            'total_behaviors_analyzed': 0,
            'behavior_patterns': {},
            'summary_statistics': {}
        }
        
        for camera_id, tracks in tracks_data.items():
            if not tracks:
                continue
            
            # Analyze behaviors for this camera
            camera_behaviors = []
            for track in tracks:
                world_coords = [WorldCoordinates(x=0, y=0)]  # Placeholder
                behavior = self.behavioral_service.analyze_person_behavior(
                    track, world_coords
                )
                camera_behaviors.append(behavior)
            
            behavior_analysis['behavior_patterns'][str(camera_id)] = camera_behaviors
            behavior_analysis['total_behaviors_analyzed'] += len(camera_behaviors)
        
        return behavior_analysis
    
    async def _analyze_crowd_dynamics(
        self,
        tracks_data: Dict[CameraID, List[Track]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze crowd dynamics patterns."""
        crowd_analysis = {
            'cameras_analyzed': len(tracks_data),
            'density_analysis': {},
            'flow_analysis': {},
            'group_formations': {}
        }
        
        for camera_id, tracks in tracks_data.items():
            if not tracks:
                continue
            
            world_coords = [WorldCoordinates(x=0, y=0) for _ in tracks]  # Placeholder
            
            # Density analysis
            density = self.crowd_service.analyze_crowd_density(tracks, world_coords)
            crowd_analysis['density_analysis'][str(camera_id)] = density
            
            # Flow analysis
            flow = self.crowd_service.analyze_flow_patterns(tracks, world_coords)
            crowd_analysis['flow_analysis'][str(camera_id)] = flow
            
            # Group formations
            groups = self.crowd_service.detect_group_formations(tracks, world_coords)
            crowd_analysis['group_formations'][str(camera_id)] = groups
        
        return crowd_analysis
    
    async def _detect_behavioral_anomalies(
        self,
        tracks_data: Dict[CameraID, List[Track]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect behavioral anomalies."""
        anomaly_detection = {
            'cameras_analyzed': len(tracks_data),
            'anomalies_by_camera': {},
            'total_anomalies': 0
        }
        
        for camera_id, tracks in tracks_data.items():
            if not tracks:
                continue
            
            world_coords = [WorldCoordinates(x=0, y=0) for _ in tracks]  # Placeholder
            
            # Detect anomalies
            anomalies = self.behavioral_service.detect_behavioral_anomalies(
                tracks, world_coords
            )
            
            if anomalies:
                anomaly_detection['anomalies_by_camera'][str(camera_id)] = [
                    anomaly.value for anomaly in anomalies
                ]
                anomaly_detection['total_anomalies'] += len(anomalies)
        
        return anomaly_detection
    
    def _update_analytics_statistics(
        self,
        result: AnalyticsResult,
        processing_time_ms: float
    ) -> None:
        """Update analytics processing statistics."""
        self._analytics_stats['analyses_performed'] += 1
        
        # Update average processing time
        current_avg = self._analytics_stats['avg_processing_time_ms']
        analysis_count = self._analytics_stats['analyses_performed']
        
        self._analytics_stats['avg_processing_time_ms'] = (
            (current_avg * (analysis_count - 1) + processing_time_ms) / analysis_count
        )
        
        # Count behaviors analyzed
        if result.behavioral_analysis:
            behavior_count = result.behavioral_analysis.get('total_behaviors_analyzed', 0)
            self._analytics_stats['behaviors_analyzed'] += behavior_count
        
        # Count anomalies detected
        if result.anomaly_detection:
            anomaly_count = result.anomaly_detection.get('total_anomalies', 0)
            self._analytics_stats['anomalies_detected'] += anomaly_count