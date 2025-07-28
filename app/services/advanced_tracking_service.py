"""
Advanced tracking service with enhanced person following and zone analysis.

Handles:
- Advanced person tracking across cameras
- Zone-based analytics and monitoring
- Person following with notifications
- Cross-camera handoff optimization
- Crowd analysis and density monitoring
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from app.infrastructure.database.integrated_database_service import integrated_db_service
from app.infrastructure.cache.tracking_cache import tracking_cache
from app.services.enhanced_notification_service import enhanced_notification_service
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.mapping.entities.trajectory import Trajectory
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Zone definition for area-based analytics."""
    zone_id: str
    name: str
    camera_id: str
    polygon: List[Tuple[float, float]]
    zone_type: str = "general"  # general, entrance, exit, restricted, interest
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonFollowingSession:
    """Person following session data."""
    session_id: str
    person_id: str
    user_id: str
    start_time: datetime
    is_active: bool = True
    notifications_enabled: bool = True
    follow_distance: float = 100.0  # pixels
    cameras_to_track: Set[str] = field(default_factory=set)
    alert_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoneEvent:
    """Zone event data structure."""
    event_id: str
    zone_id: str
    person_id: str
    event_type: str  # enter, exit, dwell, violation
    timestamp: datetime
    position: Coordinate
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrowdAnalysis:
    """Crowd analysis data structure."""
    camera_id: str
    timestamp: datetime
    person_count: int
    density_score: float
    occupancy_rate: float
    crowd_state: str  # low, medium, high, critical
    hotspots: List[Dict[str, Any]]
    flow_patterns: Dict[str, Any]


class AdvancedTrackingService:
    """
    Advanced tracking service with enhanced features.
    
    Features:
    - Person following with real-time notifications
    - Zone-based analytics and monitoring
    - Cross-camera handoff optimization
    - Crowd analysis and density monitoring
    - Advanced trajectory analysis
    """
    
    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        self.following_sessions: Dict[str, PersonFollowingSession] = {}
        self.zone_events: deque = deque(maxlen=10000)
        self.crowd_analyses: Dict[str, CrowdAnalysis] = {}
        
        # Configuration
        self.zone_dwell_threshold = 30.0  # seconds
        self.crowd_density_threshold = 0.7
        self.handoff_confidence_threshold = 0.8
        self.notification_cooldown = 5.0  # seconds
        
        # Performance tracking
        self.tracking_stats = {
            'total_handoffs': 0,
            'successful_handoffs': 0,
            'zone_events_processed': 0,
            'following_sessions_active': 0,
            'crowd_analyses_performed': 0,
            'notifications_sent': 0
        }
        
        # Notification cooldown tracking
        self.notification_cooldowns: Dict[str, datetime] = {}
        
        logger.info("AdvancedTrackingService initialized")
    
    async def initialize(self):
        """Initialize the advanced tracking service."""
        try:
            # Start background processing tasks
            asyncio.create_task(self._zone_monitoring_loop())
            asyncio.create_task(self._crowd_analysis_loop())
            asyncio.create_task(self._person_following_loop())
            
            # Load predefined zones
            await self._load_predefined_zones()
            
            logger.info("AdvancedTrackingService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AdvancedTrackingService: {e}")
            raise
    
    # Zone Management
    async def create_zone(
        self,
        zone_id: str,
        name: str,
        camera_id: str,
        polygon: List[Tuple[float, float]],
        zone_type: str = "general",
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new zone for monitoring."""
        try:
            zone = Zone(
                zone_id=zone_id,
                name=name,
                camera_id=camera_id,
                polygon=polygon,
                zone_type=zone_type,
                properties=properties or {}
            )
            
            self.zones[zone_id] = zone
            
            logger.info(f"Created zone {zone_id} ({name}) for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating zone: {e}")
            return False
    
    async def delete_zone(self, zone_id: str) -> bool:
        """Delete a zone."""
        try:
            if zone_id in self.zones:
                del self.zones[zone_id]
                logger.info(f"Deleted zone {zone_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting zone: {e}")
            return False
    
    async def get_zones(self, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get zones, optionally filtered by camera."""
        try:
            zones = []
            for zone in self.zones.values():
                if camera_id is None or zone.camera_id == camera_id:
                    zones.append({
                        'zone_id': zone.zone_id,
                        'name': zone.name,
                        'camera_id': zone.camera_id,
                        'polygon': zone.polygon,
                        'zone_type': zone.zone_type,
                        'properties': zone.properties
                    })
            return zones
            
        except Exception as e:
            logger.error(f"Error getting zones: {e}")
            return []
    
    def _is_point_in_zone(self, point: Coordinate, zone: Zone) -> bool:
        """Check if a point is inside a zone."""
        try:
            shapely_point = Point(point.x, point.y)
            shapely_polygon = Polygon(zone.polygon)
            return shapely_polygon.contains(shapely_point)
            
        except Exception as e:
            logger.error(f"Error checking point in zone: {e}")
            return False
    
    async def _load_predefined_zones(self):
        """Load predefined zones from configuration."""
        try:
            # Example predefined zones - in production, these would be loaded from database
            predefined_zones = [
                {
                    'zone_id': 'entrance_c01',
                    'name': 'Main Entrance',
                    'camera_id': 'c01',
                    'polygon': [(0, 0), (100, 0), (100, 100), (0, 100)],
                    'zone_type': 'entrance'
                },
                {
                    'zone_id': 'exit_c01',
                    'name': 'Main Exit',
                    'camera_id': 'c01',
                    'polygon': [(500, 0), (600, 0), (600, 100), (500, 100)],
                    'zone_type': 'exit'
                }
            ]
            
            for zone_data in predefined_zones:
                await self.create_zone(**zone_data)
            
            logger.info(f"Loaded {len(predefined_zones)} predefined zones")
            
        except Exception as e:
            logger.error(f"Error loading predefined zones: {e}")
    
    # Zone Monitoring
    async def _zone_monitoring_loop(self):
        """Main zone monitoring loop."""
        try:
            while True:
                await self._process_zone_events()
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("Zone monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in zone monitoring loop: {e}")
    
    async def _process_zone_events(self):
        """Process zone events for all active persons."""
        try:
            # Get active persons from cache
            active_persons = await tracking_cache.get_active_persons()
            
            for person in active_persons:
                if person.current_position:
                    position = Coordinate(
                        x=person.current_position['x'],
                        y=person.current_position['y']
                    )
                    
                    # Check zone interactions
                    await self._check_zone_interactions(
                        person.global_id,
                        person.last_seen_camera,
                        position
                    )
            
        except Exception as e:
            logger.error(f"Error processing zone events: {e}")
    
    async def _check_zone_interactions(
        self,
        person_id: str,
        camera_id: str,
        position: Coordinate
    ):
        """Check if person is interacting with zones."""
        try:
            # Get zones for this camera
            camera_zones = [zone for zone in self.zones.values() if zone.camera_id == camera_id]
            
            for zone in camera_zones:
                is_in_zone = self._is_point_in_zone(position, zone)
                
                # Check for zone events
                await self._detect_zone_event(person_id, zone, position, is_in_zone)
            
        except Exception as e:
            logger.error(f"Error checking zone interactions: {e}")
    
    async def _detect_zone_event(
        self,
        person_id: str,
        zone: Zone,
        position: Coordinate,
        is_in_zone: bool
    ):
        """Detect zone entry/exit events."""
        try:
            # This would typically track previous states to detect entry/exit
            # For now, we'll create a simple event if person is in zone
            if is_in_zone:
                event = ZoneEvent(
                    event_id=f"{person_id}_{zone.zone_id}_{int(datetime.now(timezone.utc).timestamp())}",
                    zone_id=zone.zone_id,
                    person_id=person_id,
                    event_type="presence",
                    timestamp=datetime.now(timezone.utc),
                    position=position,
                    confidence=0.9,
                    metadata={
                        'zone_name': zone.name,
                        'zone_type': zone.zone_type,
                        'camera_id': zone.camera_id
                    }
                )
                
                self.zone_events.append(event)
                self.tracking_stats['zone_events_processed'] += 1
                
                # Store zone event in database
                await integrated_db_service.store_tracking_event(
                    global_person_id=person_id,
                    camera_id=zone.camera_id,
                    environment_id="default",
                    event_type="zone_presence",
                    position=position,
                    metadata={
                        'zone_id': zone.zone_id,
                        'zone_name': zone.name,
                        'zone_type': zone.zone_type
                    }
                )
                
                # Send notification if person is being followed
                await self._check_following_notifications(person_id, zone, position)
            
        except Exception as e:
            logger.error(f"Error detecting zone event: {e}")
    
    # Person Following
    async def start_person_following(
        self,
        session_id: str,
        person_id: str,
        user_id: str,
        follow_distance: float = 100.0,
        cameras_to_track: Optional[Set[str]] = None,
        alert_conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Start following a person."""
        try:
            session = PersonFollowingSession(
                session_id=session_id,
                person_id=person_id,
                user_id=user_id,
                start_time=datetime.now(timezone.utc),
                follow_distance=follow_distance,
                cameras_to_track=cameras_to_track or set(),
                alert_conditions=alert_conditions or {}
            )
            
            self.following_sessions[session_id] = session
            self.tracking_stats['following_sessions_active'] += 1
            
            logger.info(f"Started following person {person_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting person following: {e}")
            return False
    
    async def stop_person_following(self, session_id: str) -> bool:
        """Stop following a person."""
        try:
            if session_id in self.following_sessions:
                session = self.following_sessions[session_id]
                session.is_active = False
                
                # Send final notification
                await enhanced_notification_service.send_status_update_notification(
                    task_id=session.user_id,
                    status_data={
                        'type': 'following_stopped',
                        'person_id': session.person_id,
                        'session_id': session_id,
                        'duration': (datetime.now(timezone.utc) - session.start_time).total_seconds()
                    }
                )
                
                del self.following_sessions[session_id]
                self.tracking_stats['following_sessions_active'] -= 1
                
                logger.info(f"Stopped following session {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error stopping person following: {e}")
            return False
    
    async def _person_following_loop(self):
        """Main person following loop."""
        try:
            while True:
                await self._process_person_following()
                await asyncio.sleep(2.0)
                
        except asyncio.CancelledError:
            logger.info("Person following loop cancelled")
        except Exception as e:
            logger.error(f"Error in person following loop: {e}")
    
    async def _process_person_following(self):
        """Process person following notifications."""
        try:
            active_sessions = [
                session for session in self.following_sessions.values()
                if session.is_active
            ]
            
            for session in active_sessions:
                await self._check_person_following_status(session)
            
        except Exception as e:
            logger.error(f"Error processing person following: {e}")
    
    async def _check_person_following_status(self, session: PersonFollowingSession):
        """Check status of a person being followed."""
        try:
            # Get person state from cache
            person_state = await tracking_cache.get_person_state(session.person_id)
            
            if person_state:
                # Check if person moved to new camera
                if session.cameras_to_track and person_state.last_seen_camera not in session.cameras_to_track:
                    await self._send_following_notification(
                        session,
                        "camera_change",
                        {
                            'new_camera': person_state.last_seen_camera,
                            'position': person_state.current_position
                        }
                    )
                
                # Check alert conditions
                await self._check_alert_conditions(session, person_state)
            else:
                # Person lost
                await self._send_following_notification(
                    session,
                    "person_lost",
                    {'last_seen_time': person_state.last_seen_time if person_state else None}
                )
            
        except Exception as e:
            logger.error(f"Error checking person following status: {e}")
    
    async def _check_following_notifications(
        self,
        person_id: str,
        zone: Zone,
        position: Coordinate
    ):
        """Check if zone event should trigger following notification."""
        try:
            # Find active following sessions for this person
            active_sessions = [
                session for session in self.following_sessions.values()
                if session.person_id == person_id and session.is_active
            ]
            
            for session in active_sessions:
                # Check notification cooldown
                cooldown_key = f"{session.session_id}_{zone.zone_id}"
                now = datetime.now(timezone.utc)
                
                if cooldown_key in self.notification_cooldowns:
                    if (now - self.notification_cooldowns[cooldown_key]).total_seconds() < self.notification_cooldown:
                        continue
                
                # Send zone notification
                await self._send_following_notification(
                    session,
                    "zone_interaction",
                    {
                        'zone_id': zone.zone_id,
                        'zone_name': zone.name,
                        'zone_type': zone.zone_type,
                        'position': {'x': position.x, 'y': position.y}
                    }
                )
                
                # Update cooldown
                self.notification_cooldowns[cooldown_key] = now
            
        except Exception as e:
            logger.error(f"Error checking following notifications: {e}")
    
    async def _check_alert_conditions(
        self,
        session: PersonFollowingSession,
        person_state
    ):
        """Check alert conditions for followed person."""
        try:
            # Check custom alert conditions
            for condition_name, condition_params in session.alert_conditions.items():
                if condition_name == "speed_threshold":
                    # Check if person is moving too fast
                    # This would require calculating current speed
                    pass
                elif condition_name == "restricted_zone":
                    # Check if person entered restricted zone
                    # This would be handled by zone monitoring
                    pass
                elif condition_name == "dwell_time":
                    # Check if person has been in same location too long
                    # This would require tracking position history
                    pass
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    async def _send_following_notification(
        self,
        session: PersonFollowingSession,
        notification_type: str,
        data: Dict[str, Any]
    ):
        """Send notification for person following."""
        try:
            if not session.notifications_enabled:
                return
            
            notification_data = {
                'type': 'person_following',
                'notification_type': notification_type,
                'person_id': session.person_id,
                'session_id': session.session_id,
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await enhanced_notification_service.send_status_update_notification(
                task_id=session.user_id,
                status_data=notification_data
            )
            
            self.tracking_stats['notifications_sent'] += 1
            
            logger.debug(f"Sent following notification: {notification_type}")
            
        except Exception as e:
            logger.error(f"Error sending following notification: {e}")
    
    # Crowd Analysis
    async def _crowd_analysis_loop(self):
        """Main crowd analysis loop."""
        try:
            while True:
                await self._perform_crowd_analysis()
                await asyncio.sleep(10.0)
                
        except asyncio.CancelledError:
            logger.info("Crowd analysis loop cancelled")
        except Exception as e:
            logger.error(f"Error in crowd analysis loop: {e}")
    
    async def _perform_crowd_analysis(self):
        """Perform crowd analysis for all cameras."""
        try:
            # Get active persons by camera
            active_persons = await tracking_cache.get_active_persons()
            
            # Group by camera
            camera_persons = defaultdict(list)
            for person in active_persons:
                camera_persons[person.last_seen_camera].append(person)
            
            # Analyze each camera
            for camera_id, persons in camera_persons.items():
                if len(persons) > 0:
                    analysis = await self._analyze_camera_crowd(camera_id, persons)
                    if analysis:
                        self.crowd_analyses[camera_id] = analysis
                        self.tracking_stats['crowd_analyses_performed'] += 1
            
        except Exception as e:
            logger.error(f"Error performing crowd analysis: {e}")
    
    async def _analyze_camera_crowd(
        self,
        camera_id: str,
        persons: List[Any]
    ) -> Optional[CrowdAnalysis]:
        """Analyze crowd for a specific camera."""
        try:
            person_count = len(persons)
            
            if person_count == 0:
                return None
            
            # Calculate density (simplified)
            # In production, this would consider camera field of view and actual area
            density_score = min(person_count / 10.0, 1.0)  # Normalize to 0-1
            
            # Determine crowd state
            if density_score < 0.3:
                crowd_state = "low"
            elif density_score < 0.6:
                crowd_state = "medium"
            elif density_score < 0.8:
                crowd_state = "high"
            else:
                crowd_state = "critical"
            
            # Calculate occupancy rate (simplified)
            occupancy_rate = min(person_count / 20.0, 1.0)  # Assuming max 20 persons
            
            # Identify hotspots (simplified clustering)
            hotspots = await self._identify_hotspots(persons)
            
            # Analyze flow patterns (simplified)
            flow_patterns = await self._analyze_flow_patterns(persons)
            
            analysis = CrowdAnalysis(
                camera_id=camera_id,
                timestamp=datetime.now(timezone.utc),
                person_count=person_count,
                density_score=density_score,
                occupancy_rate=occupancy_rate,
                crowd_state=crowd_state,
                hotspots=hotspots,
                flow_patterns=flow_patterns
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing camera crowd: {e}")
            return None
    
    async def _identify_hotspots(self, persons: List[Any]) -> List[Dict[str, Any]]:
        """Identify crowd hotspots."""
        try:
            positions = []
            for person in persons:
                if person.current_position:
                    positions.append([
                        person.current_position['x'],
                        person.current_position['y']
                    ])
            
            if len(positions) < 3:
                return []
            
            # Simple hotspot detection (would use more sophisticated clustering in production)
            hotspots = []
            positions_array = np.array(positions)
            
            # Find center of mass
            center = np.mean(positions_array, axis=0)
            
            # Count persons within radius of center
            radius = 50.0  # pixels
            distances = np.linalg.norm(positions_array - center, axis=1)
            persons_in_hotspot = np.sum(distances <= radius)
            
            if persons_in_hotspot >= 3:
                hotspots.append({
                    'center': {'x': float(center[0]), 'y': float(center[1])},
                    'radius': radius,
                    'person_count': int(persons_in_hotspot),
                    'density': persons_in_hotspot / (np.pi * radius**2)
                })
            
            return hotspots
            
        except Exception as e:
            logger.error(f"Error identifying hotspots: {e}")
            return []
    
    async def _analyze_flow_patterns(self, persons: List[Any]) -> Dict[str, Any]:
        """Analyze crowd flow patterns."""
        try:
            # This would typically analyze movement vectors and directions
            # For now, return basic statistics
            return {
                'total_persons': len(persons),
                'movement_detected': True,  # Would be calculated based on trajectory analysis
                'dominant_direction': 'unknown',  # Would be calculated from movement vectors
                'flow_rate': 0.0  # Persons per second
            }
            
        except Exception as e:
            logger.error(f"Error analyzing flow patterns: {e}")
            return {}
    
    # Cross-Camera Handoff
    async def optimize_camera_handoff(
        self,
        person_id: str,
        source_camera: str,
        target_camera: str,
        position: Coordinate,
        confidence: float
    ) -> bool:
        """Optimize cross-camera handoff."""
        try:
            self.tracking_stats['total_handoffs'] += 1
            
            # Check handoff confidence
            if confidence < self.handoff_confidence_threshold:
                logger.warning(f"Low confidence handoff: {confidence}")
                return False
            
            # Store handoff event
            await integrated_db_service.store_tracking_event(
                global_person_id=person_id,
                camera_id=target_camera,
                environment_id="default",
                event_type="camera_handoff",
                position=position,
                reid_confidence=confidence,
                metadata={
                    'source_camera': source_camera,
                    'target_camera': target_camera,
                    'handoff_confidence': confidence
                }
            )
            
            # Update person state in cache
            await tracking_cache.update_person_activity(person_id, CameraID(target_camera))
            
            self.tracking_stats['successful_handoffs'] += 1
            
            logger.info(f"Successfully handed off person {person_id} from {source_camera} to {target_camera}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing camera handoff: {e}")
            return False
    
    # Analytics and Reporting
    async def get_zone_analytics(
        self,
        zone_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Get zone analytics."""
        try:
            # Filter zone events
            filtered_events = []
            for event in self.zone_events:
                if zone_id and event.zone_id != zone_id:
                    continue
                if time_range:
                    if event.timestamp < time_range[0] or event.timestamp > time_range[1]:
                        continue
                filtered_events.append(event)
            
            # Calculate analytics
            analytics = {
                'total_events': len(filtered_events),
                'event_types': defaultdict(int),
                'persons_involved': set(),
                'zones_active': set()
            }
            
            for event in filtered_events:
                analytics['event_types'][event.event_type] += 1
                analytics['persons_involved'].add(event.person_id)
                analytics['zones_active'].add(event.zone_id)
            
            # Convert sets to lists for JSON serialization
            analytics['persons_involved'] = list(analytics['persons_involved'])
            analytics['zones_active'] = list(analytics['zones_active'])
            analytics['event_types'] = dict(analytics['event_types'])
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting zone analytics: {e}")
            return {}
    
    async def get_crowd_analytics(
        self,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get crowd analytics."""
        try:
            if camera_id:
                analysis = self.crowd_analyses.get(camera_id)
                if analysis:
                    return {
                        'camera_id': analysis.camera_id,
                        'timestamp': analysis.timestamp.isoformat(),
                        'person_count': analysis.person_count,
                        'density_score': analysis.density_score,
                        'occupancy_rate': analysis.occupancy_rate,
                        'crowd_state': analysis.crowd_state,
                        'hotspots': analysis.hotspots,
                        'flow_patterns': analysis.flow_patterns
                    }
                return {}
            else:
                # Return all crowd analyses
                return {
                    camera_id: {
                        'camera_id': analysis.camera_id,
                        'timestamp': analysis.timestamp.isoformat(),
                        'person_count': analysis.person_count,
                        'density_score': analysis.density_score,
                        'occupancy_rate': analysis.occupancy_rate,
                        'crowd_state': analysis.crowd_state,
                        'hotspots': analysis.hotspots,
                        'flow_patterns': analysis.flow_patterns
                    }
                    for camera_id, analysis in self.crowd_analyses.items()
                }
            
        except Exception as e:
            logger.error(f"Error getting crowd analytics: {e}")
            return {}
    
    async def get_following_sessions(
        self,
        user_id: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get person following sessions."""
        try:
            sessions = []
            for session in self.following_sessions.values():
                if active_only and not session.is_active:
                    continue
                if user_id and session.user_id != user_id:
                    continue
                
                sessions.append({
                    'session_id': session.session_id,
                    'person_id': session.person_id,
                    'user_id': session.user_id,
                    'start_time': session.start_time.isoformat(),
                    'is_active': session.is_active,
                    'notifications_enabled': session.notifications_enabled,
                    'follow_distance': session.follow_distance,
                    'cameras_to_track': list(session.cameras_to_track),
                    'alert_conditions': session.alert_conditions
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting following sessions: {e}")
            return []
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            return {
                'tracking_stats': self.tracking_stats,
                'zones_configured': len(self.zones),
                'active_following_sessions': len([
                    s for s in self.following_sessions.values() if s.is_active
                ]),
                'recent_zone_events': len(self.zone_events),
                'crowd_analyses_active': len(self.crowd_analyses),
                'notification_cooldowns_active': len(self.notification_cooldowns)
            }
            
        except Exception as e:
            logger.error(f"Error getting service statistics: {e}")
            return {}
    
    def reset_statistics(self):
        """Reset service statistics."""
        self.tracking_stats = {
            'total_handoffs': 0,
            'successful_handoffs': 0,
            'zone_events_processed': 0,
            'following_sessions_active': 0,
            'crowd_analyses_performed': 0,
            'notifications_sent': 0
        }
        logger.info("Advanced tracking statistics reset")


# Global advanced tracking service instance
advanced_tracking_service = AdvancedTrackingService()