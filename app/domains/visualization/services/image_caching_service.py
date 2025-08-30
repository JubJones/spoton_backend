"""
Image Caching Service

Handles caching of processed frames and cropped person images.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import hashlib

from app.domains.visualization.entities.visual_frame import VisualFrame
from app.domains.visualization.entities.cropped_image import CroppedImage
from app.infrastructure.cache.redis_client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


class ImageCachingService:
    """Service for caching visual frames and cropped images."""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        
        # Cache configuration
        self.frame_cache_ttl = 3600  # 1 hour
        self.cropped_image_cache_ttl = 7200  # 2 hours
        self.thumbnail_cache_ttl = 1800  # 30 minutes
        
        # Cache key prefixes
        self.frame_prefix = "visual_frame"
        self.cropped_prefix = "cropped_image"
        self.thumbnail_prefix = "thumbnail"
        self.metadata_prefix = "frame_metadata"
        
        logger.info("ImageCachingService initialized")
    
    def cache_visual_frame(self, visual_frame: VisualFrame) -> bool:
        """Cache a visual frame."""
        try:
            cache_key = self._get_frame_cache_key(
                visual_frame.camera_id, 
                visual_frame.frame_index,
                visual_frame.focused_person_id
            )
            
            # Cache frame data
            frame_data_key = f"{cache_key}:frame_data"
            self.redis_client.setex(
                frame_data_key,
                self.frame_cache_ttl,
                visual_frame.processed_frame_data
            )
            
            # Cache metadata
            metadata_key = f"{cache_key}:metadata"
            metadata = visual_frame.to_dict()
            # Remove binary data from metadata
            metadata.pop('original_frame_data', None)
            metadata.pop('processed_frame_data', None)
            
            self.redis_client.setex(
                metadata_key,
                self.frame_cache_ttl,
                json.dumps(metadata, default=str)
            )
            
            # Cache cropped images separately
            for person_id, cropped_image in visual_frame.cropped_persons.items():
                self.cache_cropped_image(cropped_image)
            
            visual_frame.cached_at = datetime.utcnow()
            
            logger.debug(f"Cached visual frame: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching visual frame: {e}")
            return False
    
    def get_cached_visual_frame(
        self,
        camera_id: str,
        frame_index: int,
        focused_person_id: Optional[str] = None
    ) -> Optional[VisualFrame]:
        """Get cached visual frame."""
        try:
            cache_key = self._get_frame_cache_key(camera_id, frame_index, focused_person_id)
            
            # Get frame data
            frame_data_key = f"{cache_key}:frame_data"
            frame_data = self.redis_client.get(frame_data_key)
            
            # Get metadata
            metadata_key = f"{cache_key}:metadata"
            metadata_json = self.redis_client.get(metadata_key)
            
            if frame_data is None or metadata_json is None:
                return None
            
            # Parse metadata
            metadata = json.loads(metadata_json)
            
            # Reconstruct visual frame (simplified version)
            # Note: This would need to be enhanced to fully reconstruct the VisualFrame object
            logger.debug(f"Retrieved cached visual frame: {cache_key}")
            return None  # TODO: Implement full reconstruction
            
        except Exception as e:
            logger.error(f"Error getting cached visual frame: {e}")
            return None
    
    def cache_cropped_image(self, cropped_image: CroppedImage) -> bool:
        """Cache a cropped person image."""
        try:
            cache_key = self._get_cropped_image_cache_key(cropped_image)
            
            # Cache image data
            image_data_key = f"{cache_key}:data"
            self.redis_client.setex(
                image_data_key,
                self.cropped_image_cache_ttl,
                cropped_image.image_data
            )
            
            # Cache metadata
            metadata_key = f"{cache_key}:metadata"
            metadata = cropped_image.to_dict()
            metadata.pop('image_data', None)  # Remove binary data
            
            self.redis_client.setex(
                metadata_key,
                self.cropped_image_cache_ttl,
                json.dumps(metadata, default=str)
            )
            
            cropped_image.cached_at = datetime.utcnow()
            cropped_image.expires_at = datetime.utcnow() + timedelta(seconds=self.cropped_image_cache_ttl)
            
            logger.debug(f"Cached cropped image: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching cropped image: {e}")
            return False
    
    def get_cached_cropped_image(self, global_person_id: str, camera_id: str, frame_index: int) -> Optional[CroppedImage]:
        """Get cached cropped image."""
        try:
            # Try to find cached image with this combination
            pattern = f"{self.cropped_prefix}:{global_person_id}:{camera_id}:{frame_index}:*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return None
            
            # Use the first matching key
            cache_key = keys[0].decode('utf-8').replace(':data', '').replace(':metadata', '')
            
            # Get image data
            image_data_key = f"{cache_key}:data"
            image_data = self.redis_client.get(image_data_key)
            
            # Get metadata
            metadata_key = f"{cache_key}:metadata"
            metadata_json = self.redis_client.get(metadata_key)
            
            if image_data is None or metadata_json is None:
                return None
            
            # Parse metadata and reconstruct
            metadata = json.loads(metadata_json)
            cropped_image = CroppedImage.from_dict(metadata, image_data)
            
            logger.debug(f"Retrieved cached cropped image: {cache_key}")
            return cropped_image
            
        except Exception as e:
            logger.error(f"Error getting cached cropped image: {e}")
            return None
    
    def cache_thumbnail(
        self,
        camera_id: str,
        frame_index: int,
        thumbnail_data: bytes,
        size: Tuple[int, int]
    ) -> bool:
        """Cache a thumbnail image."""
        try:
            cache_key = f"{self.thumbnail_prefix}:{camera_id}:{frame_index}:{size[0]}x{size[1]}"
            
            self.redis_client.setex(
                cache_key,
                self.thumbnail_cache_ttl,
                thumbnail_data
            )
            
            logger.debug(f"Cached thumbnail: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching thumbnail: {e}")
            return False
    
    def get_cached_thumbnail(
        self,
        camera_id: str,
        frame_index: int,
        size: Tuple[int, int]
    ) -> Optional[bytes]:
        """Get cached thumbnail."""
        try:
            cache_key = f"{self.thumbnail_prefix}:{camera_id}:{frame_index}:{size[0]}x{size[1]}"
            thumbnail_data = self.redis_client.get(cache_key)
            
            if thumbnail_data:
                logger.debug(f"Retrieved cached thumbnail: {cache_key}")
            
            return thumbnail_data
            
        except Exception as e:
            logger.error(f"Error getting cached thumbnail: {e}")
            return None
    
    def invalidate_frame_cache(self, camera_id: str, frame_index: int) -> bool:
        """Invalidate all cached data for a specific frame."""
        try:
            # Find all keys related to this frame
            pattern = f"{self.frame_prefix}:{camera_id}:{frame_index}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} cache entries for {camera_id}:{frame_index}")
            
            # Also invalidate thumbnails
            thumbnail_pattern = f"{self.thumbnail_prefix}:{camera_id}:{frame_index}:*"
            thumbnail_keys = self.redis_client.keys(thumbnail_pattern)
            
            if thumbnail_keys:
                self.redis_client.delete(*thumbnail_keys)
                logger.debug(f"Invalidated {len(thumbnail_keys)} thumbnail entries for {camera_id}:{frame_index}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating frame cache: {e}")
            return False
    
    def invalidate_person_cache(self, global_person_id: str) -> bool:
        """Invalidate all cached data for a specific person."""
        try:
            pattern = f"{self.cropped_prefix}:{global_person_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} cached images for person {global_person_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating person cache: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        try:
            stats = {}
            
            # Count different types of cached items
            for prefix in [self.frame_prefix, self.cropped_prefix, self.thumbnail_prefix]:
                pattern = f"{prefix}:*"
                keys = self.redis_client.keys(pattern)
                stats[f"{prefix}_count"] = len(keys)
            
            # Get Redis info
            redis_info = self.redis_client.info('memory')
            stats['redis_memory_used'] = redis_info.get('used_memory_human', 'Unknown')
            stats['redis_memory_peak'] = redis_info.get('used_memory_peak_human', 'Unknown')
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            cleaned_count = 0
            
            # Redis handles TTL automatically, but we can clean up entries
            # that might have been missed or need manual cleanup
            
            # Check cropped images for expiration
            pattern = f"{self.cropped_prefix}:*:metadata"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    metadata_json = self.redis_client.get(key)
                    if metadata_json:
                        metadata = json.loads(metadata_json)
                        expires_at = metadata.get('expires_at')
                        
                        if expires_at:
                            expire_time = datetime.fromisoformat(expires_at)
                            if datetime.utcnow() > expire_time:
                                # Delete both metadata and data
                                base_key = key.decode('utf-8').replace(':metadata', '')
                                self.redis_client.delete(key, f"{base_key}:data")
                                cleaned_count += 1
                                
                except Exception as inner_e:
                    logger.warning(f"Error processing cache key {key}: {inner_e}")
                    continue
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")
            return 0
    
    def _get_frame_cache_key(
        self,
        camera_id: str,
        frame_index: int,
        focused_person_id: Optional[str] = None
    ) -> str:
        """Generate cache key for visual frame."""
        focus_part = f":{focused_person_id}" if focused_person_id else ":none"
        return f"{self.frame_prefix}:{camera_id}:{frame_index}{focus_part}"
    
    def _get_cropped_image_cache_key(self, cropped_image: CroppedImage) -> str:
        """Generate cache key for cropped image."""
        timestamp_hash = hashlib.md5(
            cropped_image.timestamp.isoformat().encode()
        ).hexdigest()[:8]
        
        return (f"{self.cropped_prefix}:{cropped_image.global_person_id}:"
                f"{cropped_image.camera_id}:{cropped_image.frame_index}:{timestamp_hash}")
    
    def preload_cache(self, cache_entries: List[Dict[str, Any]]) -> int:
        """Preload cache with multiple entries."""
        try:
            loaded_count = 0
            
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            
            for entry in cache_entries:
                try:
                    if entry['type'] == 'visual_frame':
                        # Add visual frame to pipeline
                        cache_key = self._get_frame_cache_key(
                            entry['camera_id'],
                            entry['frame_index'],
                            entry.get('focused_person_id')
                        )
                        pipe.setex(f"{cache_key}:frame_data", self.frame_cache_ttl, entry['frame_data'])
                        pipe.setex(f"{cache_key}:metadata", self.frame_cache_ttl, entry['metadata'])
                        loaded_count += 1
                        
                    elif entry['type'] == 'cropped_image':
                        # Add cropped image to pipeline
                        cache_key = entry['cache_key']
                        pipe.setex(f"{cache_key}:data", self.cropped_image_cache_ttl, entry['image_data'])
                        pipe.setex(f"{cache_key}:metadata", self.cropped_image_cache_ttl, entry['metadata'])
                        loaded_count += 1
                        
                except Exception as inner_e:
                    logger.warning(f"Error preparing cache entry: {inner_e}")
                    continue
            
            # Execute pipeline
            pipe.execute()
            
            logger.info(f"Preloaded {loaded_count} cache entries")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error preloading cache: {e}")
            return 0