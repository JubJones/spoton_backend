"""
Redis implementation of tracking repository.

Concrete implementation of tracking repository using Redis
for high-performance caching and real-time track management.
"""
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
import logging
import json
import redis.asyncio as redis
from dataclasses import asdict

from app.application.repositories.tracking_repository import TrackingRepository
from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.tracking.entities.track import Track
from app.domain.tracking.value_objects.track_id import TrackID
from app.domain.shared.value_objects.time_range import TimeRange

logger = logging.getLogger(__name__)


class RedisTrackingRepository(TrackingRepository):
    """
    Redis implementation of tracking repository.
    
    Uses Redis for high-performance real-time track storage and retrieval
    with proper expiration policies and optimized data structures.
    """
    
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        """
        Initialize Redis tracking repository.
        
        Args:
            redis_client: Redis client instance
            ttl_seconds: Time-to-live for track data in seconds
        """
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        
        # Redis key patterns
        self.track_key_pattern = "track:{camera_id}:{track_id}"
        self.active_tracks_key = "active_tracks:{camera_id}"
        self.camera_tracks_key = "camera_tracks:{camera_id}"
        self.track_stats_key = "track_stats:{camera_id}"
        
        logger.debug("RedisTrackingRepository initialized")
    
    async def save_track(self, track: Track) -> bool:
        """Save a track to Redis."""
        try:
            # Serialize track data
            track_data = self._track_to_dict(track)
            track_json = json.dumps(track_data)
            
            # Create Redis keys
            track_key = self.track_key_pattern.format(
                camera_id=track.camera_id,
                track_id=track.track_id
            )
            active_key = self.active_tracks_key.format(camera_id=track.camera_id)
            
            # Save track data with TTL
            await self.redis_client.setex(track_key, self.ttl_seconds, track_json)
            
            # Add to active tracks set if track is active
            if track.is_active():
                await self.redis_client.sadd(active_key, str(track.track_id))
                await self.redis_client.expire(active_key, self.ttl_seconds)
            
            logger.debug(f"Saved track {track.track_id} for camera {track.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save track: {e}")
            return False
    
    async def update_track(self, track: Track) -> bool:
        """Update an existing track in Redis."""
        # For Redis, update is same as save due to key-value nature
        return await self.save_track(track)
    
    async def get_track_by_id(
        self,
        camera_id: CameraID,
        track_id: TrackID
    ) -> Optional[Track]:
        """Get track by camera and track ID."""
        try:
            track_key = self.track_key_pattern.format(
                camera_id=camera_id,
                track_id=track_id
            )
            
            track_json = await self.redis_client.get(track_key)
            if not track_json:
                return None
            
            track_data = json.loads(track_json)
            return self._dict_to_track(track_data)
            
        except Exception as e:
            logger.error(f"Failed to get track by ID: {e}")
            return None
    
    async def get_active_tracks(
        self,
        camera_id: CameraID
    ) -> Dict[TrackID, Track]:
        """Get all currently active tracks for a camera."""
        try:
            active_key = self.active_tracks_key.format(camera_id=camera_id)
            
            # Get active track IDs
            track_ids = await self.redis_client.smembers(active_key)
            
            active_tracks = {}
            for track_id_bytes in track_ids:
                track_id_str = track_id_bytes.decode('utf-8')
                track_id = TrackID(track_id_str)
                
                track = await self.get_track_by_id(camera_id, track_id)
                if track and track.is_active():
                    active_tracks[track_id] = track
            
            return active_tracks
            
        except Exception as e:
            logger.error(f"Failed to get active tracks: {e}")
            return {}
    
    async def get_tracks_by_camera(
        self,
        camera_id: CameraID,
        time_range: Optional[TimeRange] = None,
        include_ended: bool = False
    ) -> Dict[TrackID, Track]:
        """Get tracks for specific camera."""
        try:
            # Get all track keys for camera
            pattern = self.track_key_pattern.format(
                camera_id=camera_id,
                track_id="*"
            )
            
            track_keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                track_keys.append(key)
            
            tracks = {}
            for track_key in track_keys:
                try:
                    track_json = await self.redis_client.get(track_key)
                    if not track_json:
                        continue
                    
                    track_data = json.loads(track_json)
                    track = self._dict_to_track(track_data)
                    
                    if not track:
                        continue
                    
                    # Apply filters
                    if not include_ended and not track.is_active():
                        continue
                    
                    if time_range and not self._track_in_time_range(track, time_range):
                        continue
                    
                    tracks[track.track_id] = track
                    
                except Exception as e:
                    logger.warning(f"Failed to process track key {track_key}: {e}")
                    continue
            
            return tracks
            
        except Exception as e:
            logger.error(f"Failed to get tracks by camera: {e}")
            return {}
    
    async def get_tracks_by_time_range(
        self,
        time_range: TimeRange,
        camera_ids: Optional[List[CameraID]] = None
    ) -> List[Track]:
        """Get tracks within time range across cameras."""
        try:
            if camera_ids:
                cameras_to_check = camera_ids
            else:
                # Get all camera IDs from active tracks
                pattern = "active_tracks:*"
                camera_keys = []
                async for key in self.redis_client.scan_iter(match=pattern):
                    camera_keys.append(key)
                
                cameras_to_check = [
                    CameraID(key.decode('utf-8').split(':')[1])
                    for key in camera_keys
                ]
            
            all_tracks = []
            for camera_id in cameras_to_check:
                camera_tracks = await self.get_tracks_by_camera(
                    camera_id, time_range, include_ended=True
                )
                all_tracks.extend(camera_tracks.values())
            
            return all_tracks
            
        except Exception as e:
            logger.error(f"Failed to get tracks by time range: {e}")
            return []
    
    async def mark_track_ended(
        self,
        camera_id: CameraID,
        track_id: TrackID,
        end_time: datetime
    ) -> bool:
        """Mark a track as ended."""
        try:
            track = await self.get_track_by_id(camera_id, track_id)
            if not track:
                return False
            
            # Update track state to ended
            track.mark_ended(end_time)
            
            # Save updated track
            await self.save_track(track)
            
            # Remove from active tracks
            active_key = self.active_tracks_key.format(camera_id=camera_id)
            await self.redis_client.srem(active_key, str(track_id))
            
            logger.debug(f"Marked track {track_id} as ended")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark track as ended: {e}")
            return False
    
    async def get_track_count(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None,
        active_only: bool = False
    ) -> int:
        """Get count of tracks matching criteria."""
        try:
            if camera_id:
                if active_only:
                    # Count active tracks for specific camera
                    active_key = self.active_tracks_key.format(camera_id=camera_id)
                    return await self.redis_client.scard(active_key)
                else:
                    # Count all tracks for specific camera
                    tracks = await self.get_tracks_by_camera(
                        camera_id, time_range, include_ended=True
                    )
                    return len(tracks)
            else:
                # Count tracks across all cameras
                if active_only:
                    pattern = "active_tracks:*"
                    total_count = 0
                    async for key in self.redis_client.scan_iter(match=pattern):
                        count = await self.redis_client.scard(key)
                        total_count += count
                    return total_count
                else:
                    # This is expensive operation - scan all track keys
                    pattern = "track:*"
                    count = 0
                    async for _ in self.redis_client.scan_iter(match=pattern):
                        count += 1
                    return count
            
        except Exception as e:
            logger.error(f"Failed to get track count: {e}")
            return 0
    
    async def delete_old_tracks(
        self,
        older_than: datetime,
        batch_size: int = 1000
    ) -> int:
        """Delete tracks older than specified date."""
        try:
            # Scan for all track keys
            pattern = "track:*"
            deleted_count = 0
            batch_keys = []
            
            async for key in self.redis_client.scan_iter(match=pattern):
                try:
                    track_json = await self.redis_client.get(key)
                    if not track_json:
                        continue
                    
                    track_data = json.loads(track_json)
                    created_at = datetime.fromisoformat(track_data['created_at'])
                    
                    if created_at < older_than:
                        batch_keys.append(key)
                        
                        if len(batch_keys) >= batch_size:
                            # Delete batch
                            if batch_keys:
                                await self.redis_client.delete(*batch_keys)
                                deleted_count += len(batch_keys)
                                batch_keys = []
                
                except Exception as e:
                    logger.warning(f"Failed to process track key during cleanup: {e}")
                    continue
            
            # Delete remaining keys
            if batch_keys:
                await self.redis_client.delete(*batch_keys)
                deleted_count += len(batch_keys)
            
            logger.info(f"Deleted {deleted_count} old tracks")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete old tracks: {e}")
            return 0
    
    async def get_track_statistics(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None
    ) -> Dict[str, Any]:
        """Get tracking statistics for analysis."""
        try:
            if camera_id:
                tracks = await self.get_tracks_by_camera(
                    camera_id, time_range, include_ended=True
                )
                track_list = list(tracks.values())
            else:
                track_list = await self.get_tracks_by_time_range(
                    time_range or TimeRange(
                        start_time=datetime.now() - timedelta(hours=24),
                        end_time=datetime.now()
                    )
                )
            
            if not track_list:
                return {}
            
            # Calculate statistics
            active_count = len([t for t in track_list if t.is_active()])
            ended_count = len(track_list) - active_count
            
            # Calculate average duration for ended tracks
            ended_tracks = [t for t in track_list if not t.is_active()]
            avg_duration = 0.0
            if ended_tracks:
                total_duration = sum(
                    (t.last_seen - t.created_at).total_seconds()
                    for t in ended_tracks
                    if t.last_seen
                )
                avg_duration = total_duration / len(ended_tracks)
            
            return {
                'total_tracks': len(track_list),
                'active_tracks': active_count,
                'ended_tracks': ended_count,
                'avg_track_duration_seconds': avg_duration,
                'unique_cameras': len(set(t.camera_id for t in track_list))
            }
            
        except Exception as e:
            logger.error(f"Failed to get track statistics: {e}")
            return {}
    
    async def get_cross_camera_associations(
        self,
        person_id: str,
        time_range: Optional[TimeRange] = None
    ) -> List[Dict[str, Any]]:
        """Get cross-camera track associations for a person."""
        # This would be implemented with additional Redis data structures
        # for cross-camera associations - placeholder implementation
        return []
    
    async def save_cross_camera_association(
        self,
        person_id: str,
        camera_associations: List[Dict[str, Any]]
    ) -> bool:
        """Save cross-camera track associations."""
        # This would be implemented with additional Redis data structures
        # for cross-camera associations - placeholder implementation
        return True
    
    def _track_to_dict(self, track: Track) -> Dict[str, Any]:
        """Convert Track entity to dictionary for serialization."""
        return {
            'track_id': str(track.track_id),
            'camera_id': str(track.camera_id),
            'state': track.state.value,
            'created_at': track.created_at.isoformat(),
            'last_seen': track.last_seen.isoformat() if track.last_seen else None,
            'position_history': [
                {
                    'timestamp': pos.timestamp.isoformat(),
                    'bbox': {
                        'x': pos.bbox.x,
                        'y': pos.bbox.y,
                        'width': pos.bbox.width,
                        'height': pos.bbox.height,
                        'normalized': pos.bbox.normalized
                    },
                    'confidence': pos.confidence.value
                }
                for pos in track.position_history
            ],
            'velocity': {
                'vx': track.velocity.vx,
                'vy': track.velocity.vy,
                'speed': track.velocity.speed,
                'direction': track.velocity.direction
            } if track.velocity else None
        }
    
    def _dict_to_track(self, data: Dict[str, Any]) -> Optional[Track]:
        """Convert dictionary to Track entity."""
        try:
            # This is a simplified conversion - in real implementation
            # you would need to reconstruct all the value objects properly
            
            track_id = TrackID(data['track_id'])
            camera_id = CameraID(data['camera_id'])
            
            # For this example, create a basic track structure
            # Real implementation would need full reconstruction
            
            return None  # Placeholder - would need full Track reconstruction
            
        except Exception as e:
            logger.error(f"Failed to convert dict to track: {e}")
            return None
    
    def _track_in_time_range(self, track: Track, time_range: TimeRange) -> bool:
        """Check if track falls within time range."""
        return (track.created_at >= time_range.start_time and 
                track.created_at <= time_range.end_time)