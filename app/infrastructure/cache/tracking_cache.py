"""
Redis-based caching service for real-time tracking state management.

Handles:
- Person tracking state caching
- Session state management
- Multi-camera tracking state
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
from redis.asyncio import Redis as AsyncRedis

from app.infrastructure.cache.redis_client import get_redis_async
from app.shared.types import CameraID
from typing import Any
from app.domains.mapping.entities.coordinate import Coordinate
from app.domains.mapping.entities.trajectory import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class CachedPersonState:
    """Cached person state for real-time tracking."""
    global_id: str
    last_seen_camera: str
    last_seen_time: datetime
    current_position: Optional[Dict[str, float]]
    confidence: float
    track_id: Optional[str]
    embedding: Optional[List[float]]
    trajectory_points: List[Dict[str, Any]]
    is_active: bool


@dataclass
class CachedSessionState:
    """Cached session state for multi-user tracking."""
    session_id: str
    user_id: Optional[str]
    camera_ids: List[str]
    active_persons: Set[str]
    start_time: datetime
    last_activity: datetime
    settings: Dict[str, Any]


class TrackingCache:
    """
    Redis-based cache for real-time tracking state management.
    
    Features:
    - Person tracking state caching
    - Session state management
    - Multi-camera tracking coordination
    """
    
    def __init__(self):
        self.redis: Optional[AsyncRedis] = None
        
        # Cache configuration
        self.person_state_ttl = 300  # 5 minutes
        self.embedding_ttl = 1800  # 30 minutes
        self.session_ttl = 7200  # 2 hours
        self.trajectory_ttl = 3600  # 1 hour
        
        # Key prefixes
        self.person_key_prefix = "person_state"
        self.embedding_key_prefix = "embedding"
        self.session_key_prefix = "session"
        self.trajectory_key_prefix = "trajectory"
        self.camera_key_prefix = "camera_state"
        self.global_key_prefix = "global_state"
        
        # Performance metrics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "deletes": 0,
            "errors": 0
        }
        
        logger.info("TrackingCache initialized")
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = await get_redis_async()
            logger.info("TrackingCache Redis connection established")
        except Exception as e:
            logger.error(f"Failed to initialize TrackingCache: {e}")
            raise
    
    async def cleanup(self):
        """Clean up Redis connection."""
        try:
            if self.redis:
                await self.redis.aclose()
                self.redis = None
            logger.info("TrackingCache cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up TrackingCache: {e}")
    
    # Person State Management
    async def cache_person_state(
        self,
        person_identity: Any,
        camera_id: CameraID,
        position: Optional[Coordinate] = None,
        trajectory: Optional[Trajectory] = None,
    ) -> bool:
        """Cache person tracking state."""
        try:
            if not self.redis:
                await self.initialize()
            
            # Convert to cached state format
            # Support either object with attrs or dict-like
            gid = getattr(person_identity, 'global_id', None)
            if gid is None and isinstance(person_identity, dict):
                gid = person_identity.get('global_id')
            confidence = getattr(person_identity, 'confidence', 0.0)
            if isinstance(person_identity, dict):
                confidence = person_identity.get('confidence', confidence)
            track_id = getattr(person_identity, 'track_id', None)
            if isinstance(person_identity, dict):
                track_id = person_identity.get('track_id', track_id)
            fv = getattr(person_identity, 'feature_vector', None)
            if isinstance(person_identity, dict):
                fv = person_identity.get('feature_vector', fv)
            fv_vec = None
            if fv is not None:
                # Support object with vector attr or dict with 'vector'
                fv_vec = getattr(fv, 'vector', None)
                if fv_vec is None and isinstance(fv, dict):
                    fv_vec = fv.get('vector')
                if isinstance(fv_vec, np.ndarray):
                    fv_vec = fv_vec.tolist()

            cached_state = CachedPersonState(
                global_id=str(gid) if gid is not None else "",
                last_seen_camera=str(camera_id),
                last_seen_time=datetime.now(timezone.utc),
                current_position=position.to_dict() if position else None,
                confidence=float(confidence),
                track_id=str(track_id) if track_id is not None else None,
                embedding=fv_vec,
                trajectory_points=(trajectory.to_dict()['points'] if trajectory else []),
                is_active=bool(getattr(person_identity, 'is_active', True))
            )
            
            # Cache with TTL
            key = f"{self.person_key_prefix}:{cached_state.global_id}"
            await self.redis.setex(
                key,
                self.person_state_ttl,
                json.dumps(asdict(cached_state), default=str)
            )
            
            # Update camera-specific index
            await self._update_camera_person_index(camera_id, cached_state.global_id)
            
            self.cache_stats["writes"] += 1
            logger.debug(f"Cached person state for {cached_state.global_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching person state: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def get_person_state(self, global_id: str) -> Optional[CachedPersonState]:
        """Get cached person state."""
        try:
            if not self.redis:
                await self.initialize()
            
            key = f"{self.person_key_prefix}:{global_id}"
            cached_data = await self.redis.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                self.cache_stats["hits"] += 1
                return CachedPersonState(**data)
            else:
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting person state: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    async def get_active_persons(self, camera_id: Optional[CameraID] = None) -> List[CachedPersonState]:
        """Get all active persons, optionally filtered by camera."""
        try:
            if not self.redis:
                await self.initialize()
            
            if camera_id:
                # Get persons for specific camera
                camera_key = f"{self.camera_key_prefix}:{camera_id}"
                person_ids = await self.redis.smembers(camera_key)
            else:
                # Iterate keys using SCAN to avoid blocking
                person_ids = []
                async for key in self.redis.scan_iter(match=f"{self.person_key_prefix}:*"):
                    try:
                        person_ids.append(key.decode('utf-8').split(':', 1)[1])
                    except Exception:
                        continue
            
            # Fetch all person states
            active_persons = []
            for person_id in person_ids:
                person_state = await self.get_person_state(person_id)
                if person_state and person_state.is_active:
                    active_persons.append(person_state)
            
            return active_persons
            
        except Exception as e:
            logger.error(f"Error getting active persons: {e}")
            return []
    
    async def update_person_activity(self, global_id: str, camera_id: CameraID) -> bool:
        """Update person's last activity timestamp."""
        try:
            if not self.redis:
                await self.initialize()
            
            person_state = await self.get_person_state(global_id)
            if person_state:
                person_state.last_seen_time = datetime.now(timezone.utc)
                person_state.last_seen_camera = str(camera_id)
                
                key = f"{self.person_key_prefix}:{global_id}"
                await self.redis.setex(
                    key,
                    self.person_state_ttl,
                    json.dumps(asdict(person_state), default=str)
                )
                
                await self._update_camera_person_index(camera_id, global_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating person activity: {e}")
            return False
    
    async def deactivate_person(self, global_id: str) -> bool:
        """Deactivate person in cache."""
        try:
            if not self.redis:
                await self.initialize()
            
            person_state = await self.get_person_state(global_id)
            if person_state:
                person_state.is_active = False
                
                key = f"{self.person_key_prefix}:{global_id}"
                await self.redis.setex(
                    key,
                    self.person_state_ttl,
                    json.dumps(asdict(person_state), default=str)
                )
                
                # Remove from camera indices
                await self._remove_person_from_camera_indices(global_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deactivating person: {e}")
            return False
    
    # Embedding Management
    async def cache_embedding(
        self,
        person_id: str,
        feature_vector: Any,
        camera_id: CameraID,
    ) -> bool:
        """Cache embedding with metadata."""
        try:
            if not self.redis:
                await self.initialize()
            
            # Support object or dict feature vector
            vec = getattr(feature_vector, 'vector', None)
            if vec is None and isinstance(feature_vector, dict):
                vec = feature_vector.get('vector')
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            confidence = getattr(feature_vector, 'confidence', None)
            if isinstance(feature_vector, dict):
                confidence = feature_vector.get('confidence', confidence)
            model_version = getattr(feature_vector, 'model_version', None)
            if isinstance(feature_vector, dict):
                model_version = feature_vector.get('model_version', model_version)

            embedding_data = {
                "person_id": person_id,
                "vector": vec,
                "camera_id": str(camera_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": confidence,
                "model_version": model_version,
            }
            
            key = f"{self.embedding_key_prefix}:{person_id}:{camera_id}"
            await self.redis.setex(
                key,
                self.embedding_ttl,
                json.dumps(embedding_data)
            )
            
            # Update embedding index
            await self._update_embedding_index(person_id, camera_id)
            
            self.cache_stats["writes"] += 1
            logger.debug(f"Cached embedding for person {person_id} from camera {camera_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def get_person_embeddings(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all cached embeddings for a person."""
        try:
            if not self.redis:
                await self.initialize()
            
            # Iterate keys using SCAN to avoid blocking
            keys = []
            async for key in self.redis.scan_iter(match=f"{self.embedding_key_prefix}:{person_id}:*"):
                keys.append(key)
            embeddings = []
            
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    embedding_data = json.loads(data)
                    embeddings.append(embedding_data)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting person embeddings: {e}")
            return []
    
    async def get_camera_embeddings(self, camera_id: CameraID) -> List[Dict[str, Any]]:
        """Get all cached embeddings from a specific camera."""
        try:
            if not self.redis:
                await self.initialize()
            
            embeddings = []
            async for key in self.redis.scan_iter(match=f"{self.embedding_key_prefix}:*:{camera_id}"):
                data = await self.redis.get(key)
                if data:
                    embedding_data = json.loads(data)
                    embeddings.append(embedding_data)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting camera embeddings: {e}")
            return []
    
    # Session Management
    async def create_session(
        self,
        session_id: str,
        camera_ids: List[CameraID],
        user_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new tracking session."""
        try:
            if not self.redis:
                await self.initialize()
            
            session_state = CachedSessionState(
                session_id=session_id,
                user_id=user_id,
                camera_ids=[str(cam_id) for cam_id in camera_ids],
                active_persons=set(),
                start_time=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                settings=settings or {}
            )
            
            key = f"{self.session_key_prefix}:{session_id}"
            await self.redis.setex(
                key,
                self.session_ttl,
                json.dumps(asdict(session_state), default=str)
            )
            
            # Update session index
            await self._update_session_index(session_id, camera_ids)
            
            self.cache_stats["writes"] += 1
            logger.info(f"Created session {session_id} for cameras {camera_ids}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def get_session_state(self, session_id: str) -> Optional[CachedSessionState]:
        """Get cached session state."""
        try:
            if not self.redis:
                await self.initialize()
            
            key = f"{self.session_key_prefix}:{session_id}"
            cached_data = await self.redis.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                # Convert active_persons back to set
                data['active_persons'] = set(data.get('active_persons', []))
                self.cache_stats["hits"] += 1
                return CachedSessionState(**data)
            else:
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    async def update_session_activity(self, session_id: str, person_id: str) -> bool:
        """Update session activity with new person."""
        try:
            if not self.redis:
                await self.initialize()
            
            session_state = await self.get_session_state(session_id)
            if session_state:
                session_state.active_persons.add(person_id)
                session_state.last_activity = datetime.now(timezone.utc)
                
                key = f"{self.session_key_prefix}:{session_id}"
                # Convert set to list for JSON serialization
                session_dict = asdict(session_state)
                session_dict['active_persons'] = list(session_state.active_persons)
                
                await self.redis.setex(
                    key,
                    self.session_ttl,
                    json.dumps(session_dict, default=str)
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating session activity: {e}")
            return False
    
    async def end_session(self, session_id: str) -> bool:
        """End and cleanup session."""
        try:
            if not self.redis:
                await self.initialize()
            
            # Remove session
            key = f"{self.session_key_prefix}:{session_id}"
            await self.redis.delete(key)
            
            # Remove from session index
            await self._remove_session_from_index(session_id)
            
            self.cache_stats["deletes"] += 1
            logger.info(f"Ended session {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False
    
    # Trajectory Management
    async def cache_trajectory(
        self,
        person_id: str,
        trajectory: Trajectory,
        camera_id: CameraID
    ) -> bool:
        """Cache person trajectory."""
        try:
            if not self.redis:
                await self.initialize()
            
            trajectory_data = {
                "person_id": person_id,
                "camera_id": str(camera_id),
                "points": trajectory.to_dict()['points'],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": trajectory.confidence if hasattr(trajectory, 'confidence') else 1.0
            }
            
            key = f"{self.trajectory_key_prefix}:{person_id}:{camera_id}"
            await self.redis.setex(
                key,
                self.trajectory_ttl,
                json.dumps(trajectory_data)
            )
            
            self.cache_stats["writes"] += 1
            logger.debug(f"Cached trajectory for person {person_id} from camera {camera_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching trajectory: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def get_person_trajectories(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all cached trajectories for a person."""
        try:
            if not self.redis:
                await self.initialize()
            
            # Iterate keys using SCAN to avoid blocking
            keys = []
            async for key in self.redis.scan_iter(match=f"{self.trajectory_key_prefix}:{person_id}:*"):
                keys.append(key)
            trajectories = []
            
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    trajectory_data = json.loads(data)
                    trajectories.append(trajectory_data)
            
            return trajectories
            
        except Exception as e:
            logger.error(f"Error getting person trajectories: {e}")
            return []
    
    # Cache Maintenance
    async def cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            if not self.redis:
                await self.initialize()
            
            # Clean up expired person states
            await self._cleanup_expired_persons()
            
            # Clean up expired sessions
            await self._cleanup_expired_sessions()
            
            # Clean up expired embeddings
            await self._cleanup_expired_embeddings()
            
            logger.debug("Completed cache cleanup")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        try:
            if not self.redis:
                await self.initialize()
            
            # Get Redis info
            info = await self.redis.info()
            
            # Count keys by type
            key_counts = {}
            for prefix in [self.person_key_prefix, self.embedding_key_prefix, 
                          self.session_key_prefix, self.trajectory_key_prefix]:
                count = 0
                async for _ in self.redis.scan_iter(match=f"{prefix}:*"):
                    count += 1
                key_counts[prefix] = count
            
            return {
                "cache_stats": self.cache_stats,
                "key_counts": key_counts,
                "redis_info": {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def reset_cache_stats(self):
        """Reset cache statistics."""
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "deletes": 0,
            "errors": 0
        }
        logger.info("Cache statistics reset")
    
    # Helper methods
    async def _update_camera_person_index(self, camera_id: CameraID, person_id: str):
        """Update camera-specific person index."""
        try:
            camera_key = f"{self.camera_key_prefix}:{camera_id}"
            await self.redis.sadd(camera_key, person_id)
            await self.redis.expire(camera_key, self.person_state_ttl)
        except Exception as e:
            logger.error(f"Error updating camera person index: {e}")
    
    async def _remove_person_from_camera_indices(self, person_id: str):
        """Remove person from all camera indices."""
        try:
            async for key in self.redis.scan_iter(match=f"{self.camera_key_prefix}:*"):
                await self.redis.srem(key, person_id)
        except Exception as e:
            logger.error(f"Error removing person from camera indices: {e}")
    
    async def _update_embedding_index(self, person_id: str, camera_id: CameraID):
        """Update embedding index."""
        try:
            index_key = f"embedding_index:{person_id}"
            await self.redis.sadd(index_key, str(camera_id))
            await self.redis.expire(index_key, self.embedding_ttl)
        except Exception as e:
            logger.error(f"Error updating embedding index: {e}")
    
    async def _update_session_index(self, session_id: str, camera_ids: List[CameraID]):
        """Update session index."""
        try:
            for camera_id in camera_ids:
                index_key = f"session_index:{camera_id}"
                await self.redis.sadd(index_key, session_id)
                await self.redis.expire(index_key, self.session_ttl)
        except Exception as e:
            logger.error(f"Error updating session index: {e}")
    
    async def _remove_session_from_index(self, session_id: str):
        """Remove session from all indices."""
        try:
            async for key in self.redis.scan_iter(match="session_index:*"):
                await self.redis.srem(key, session_id)
        except Exception as e:
            logger.error(f"Error removing session from index: {e}")
    
    async def _cleanup_expired_persons(self):
        """Clean up expired person states."""
        try:
            # This is handled by Redis TTL, but we can clean up indices
            async for key in self.redis.scan_iter(match=f"{self.camera_key_prefix}:*"):
                # Get all persons in this camera
                person_ids = await self.redis.smembers(key)
                for person_id in person_ids:
                    # Check if person state still exists
                    person_key = f"{self.person_key_prefix}:{person_id}"
                    if not await self.redis.exists(person_key):
                        await self.redis.srem(key, person_id)
        except Exception as e:
            logger.error(f"Error cleaning up expired persons: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired session indices."""
        try:
            async for key in self.redis.scan_iter(match="session_index:*"):
                session_ids = await self.redis.smembers(key)
                for session_id in session_ids:
                    session_key = f"{self.session_key_prefix}:{session_id}"
                    if not await self.redis.exists(session_key):
                        await self.redis.srem(key, session_id)
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
    
    async def _cleanup_expired_embeddings(self):
        """Clean up expired embedding indices."""
        try:
            async for key in self.redis.scan_iter(match="embedding_index:*"):
                person_id = key.split(':')[1]
                camera_ids = await self.redis.smembers(key)
                for camera_id in camera_ids:
                    embedding_key = f"{self.embedding_key_prefix}:{person_id}:{camera_id}"
                    if not await self.redis.exists(embedding_key):
                        await self.redis.srem(key, camera_id)
        except Exception as e:
            logger.error(f"Error cleaning up expired embeddings: {e}")


# Global tracking cache instance
tracking_cache = TrackingCache()

def get_tracking_cache() -> TrackingCache:
    """Get the global tracking cache instance."""
    return tracking_cache
