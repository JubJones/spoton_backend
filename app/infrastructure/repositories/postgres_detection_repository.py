"""
PostgreSQL implementation of detection repository.

Concrete implementation of detection repository using TimescaleDB
for time-series optimization with proper database patterns.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncpg
import json

from app.application.repositories.detection_repository import DetectionRepository
from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.frame_id import FrameID
from app.domain.detection.entities.detection import Detection
from app.domain.shared.value_objects.time_range import TimeRange
from app.domain.detection.value_objects.confidence import Confidence
from app.domain.detection.value_objects.detection_class import DetectionClass
from app.domain.shared.value_objects.bounding_box import BoundingBox

logger = logging.getLogger(__name__)


class PostgreSQLDetectionRepository(DetectionRepository):
    """
    PostgreSQL implementation of detection repository.
    
    Uses TimescaleDB for optimized time-series storage and querying
    of detection data with proper ACID properties.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize PostgreSQL detection repository.
        
        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
        logger.debug("PostgreSQLDetectionRepository initialized")
    
    async def save_detection(self, detection: Detection) -> bool:
        """Save a detection to PostgreSQL."""
        query = """
        INSERT INTO detections (
            camera_id, frame_id, timestamp, bbox_x, bbox_y, bbox_width, bbox_height,
            confidence, detection_class, normalized_coords, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
        ON CONFLICT (camera_id, frame_id, bbox_x, bbox_y) DO NOTHING
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    str(detection.camera_id),
                    str(detection.frame_id),
                    detection.timestamp,
                    detection.bbox.x,
                    detection.bbox.y,
                    detection.bbox.width,
                    detection.bbox.height,
                    detection.confidence.value,
                    detection.detection_class.value,
                    detection.bbox.normalized
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to save detection: {e}")
            return False
    
    async def save_detections_batch(self, detections: List[Detection]) -> int:
        """Save multiple detections in batch operation."""
        if not detections:
            return 0
        
        query = """
        INSERT INTO detections (
            camera_id, frame_id, timestamp, bbox_x, bbox_y, bbox_width, bbox_height,
            confidence, detection_class, normalized_coords, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
        ON CONFLICT (camera_id, frame_id, bbox_x, bbox_y) DO NOTHING
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                # Prepare batch data
                batch_data = []
                for detection in detections:
                    batch_data.append((
                        str(detection.camera_id),
                        str(detection.frame_id),
                        detection.timestamp,
                        detection.bbox.x,
                        detection.bbox.y,
                        detection.bbox.width,
                        detection.bbox.height,
                        detection.confidence.value,
                        detection.detection_class.value,
                        detection.bbox.normalized
                    ))
                
                # Execute batch insert
                result = await conn.executemany(query, batch_data)
                saved_count = len([r for r in result if r == 'INSERT 0 1'])
                
                logger.debug(f"Batch saved {saved_count}/{len(detections)} detections")
                return saved_count
                
        except Exception as e:
            logger.error(f"Failed to batch save detections: {e}")
            return 0
    
    async def get_detections_by_camera(
        self,
        camera_id: CameraID,
        time_range: Optional[TimeRange] = None,
        limit: int = 1000
    ) -> List[Detection]:
        """Get detections for specific camera."""
        base_query = """
        SELECT camera_id, frame_id, timestamp, bbox_x, bbox_y, bbox_width, bbox_height,
               confidence, detection_class, normalized_coords
        FROM detections
        WHERE camera_id = $1
        """
        
        params = [str(camera_id)]
        param_count = 1
        
        if time_range:
            param_count += 1
            base_query += f" AND timestamp >= ${param_count}"
            params.append(time_range.start_time)
            
            param_count += 1
            base_query += f" AND timestamp <= ${param_count}"
            params.append(time_range.end_time)
        
        param_count += 1
        base_query += f" ORDER BY timestamp DESC LIMIT ${param_count}"
        params.append(limit)
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(base_query, *params)
                
                detections = []
                for row in rows:
                    detection = self._row_to_detection(row)
                    if detection:
                        detections.append(detection)
                
                return detections
                
        except Exception as e:
            logger.error(f"Failed to get detections by camera: {e}")
            return []
    
    async def get_detections_by_frame(
        self,
        camera_id: CameraID,
        frame_id: FrameID
    ) -> List[Detection]:
        """Get all detections for a specific frame."""
        query = """
        SELECT camera_id, frame_id, timestamp, bbox_x, bbox_y, bbox_width, bbox_height,
               confidence, detection_class, normalized_coords
        FROM detections
        WHERE camera_id = $1 AND frame_id = $2
        ORDER BY confidence DESC
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, str(camera_id), str(frame_id))
                
                detections = []
                for row in rows:
                    detection = self._row_to_detection(row)
                    if detection:
                        detections.append(detection)
                
                return detections
                
        except Exception as e:
            logger.error(f"Failed to get detections by frame: {e}")
            return []
    
    async def get_detection_count(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None
    ) -> int:
        """Get count of detections matching criteria."""
        base_query = "SELECT COUNT(*) FROM detections WHERE 1=1"
        params = []
        param_count = 0
        
        if camera_id:
            param_count += 1
            base_query += f" AND camera_id = ${param_count}"
            params.append(str(camera_id))
        
        if time_range:
            param_count += 1
            base_query += f" AND timestamp >= ${param_count}"
            params.append(time_range.start_time)
            
            param_count += 1
            base_query += f" AND timestamp <= ${param_count}"
            params.append(time_range.end_time)
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(base_query, *params)
                return result or 0
                
        except Exception as e:
            logger.error(f"Failed to get detection count: {e}")
            return 0
    
    async def delete_old_detections(
        self,
        older_than: datetime,
        batch_size: int = 1000
    ) -> int:
        """Delete detections older than specified date."""
        delete_query = """
        DELETE FROM detections
        WHERE id IN (
            SELECT id FROM detections
            WHERE timestamp < $1
            ORDER BY timestamp
            LIMIT $2
        )
        """
        
        total_deleted = 0
        
        try:
            async with self.db_pool.acquire() as conn:
                while True:
                    result = await conn.execute(delete_query, older_than, batch_size)
                    deleted_count = int(result.split()[-1])
                    
                    if deleted_count == 0:
                        break
                    
                    total_deleted += deleted_count
                    logger.debug(f"Deleted {deleted_count} old detections")
                
                logger.info(f"Deleted total of {total_deleted} old detections")
                return total_deleted
                
        except Exception as e:
            logger.error(f"Failed to delete old detections: {e}")
            return total_deleted
    
    async def get_detection_statistics(
        self,
        camera_id: Optional[CameraID] = None,
        time_range: Optional[TimeRange] = None
    ) -> Dict[str, Any]:
        """Get detection statistics for analysis."""
        base_query = """
        SELECT 
            COUNT(*) as total_detections,
            AVG(confidence) as avg_confidence,
            MIN(confidence) as min_confidence,
            MAX(confidence) as max_confidence,
            COUNT(DISTINCT camera_id) as unique_cameras,
            COUNT(DISTINCT frame_id) as unique_frames,
            COUNT(DISTINCT DATE(timestamp)) as unique_days
        FROM detections
        WHERE 1=1
        """
        
        params = []
        param_count = 0
        
        if camera_id:
            param_count += 1
            base_query += f" AND camera_id = ${param_count}"
            params.append(str(camera_id))
        
        if time_range:
            param_count += 1
            base_query += f" AND timestamp >= ${param_count}"
            params.append(time_range.start_time)
            
            param_count += 1
            base_query += f" AND timestamp <= ${param_count}"
            params.append(time_range.end_time)
        
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(base_query, *params)
                
                if row:
                    return {
                        'total_detections': row['total_detections'],
                        'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0.0,
                        'min_confidence': float(row['min_confidence']) if row['min_confidence'] else 0.0,
                        'max_confidence': float(row['max_confidence']) if row['max_confidence'] else 0.0,
                        'unique_cameras': row['unique_cameras'],
                        'unique_frames': row['unique_frames'],
                        'unique_days': row['unique_days']
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get detection statistics: {e}")
            return {}
    
    async def exists_detection(
        self,
        camera_id: CameraID,
        frame_id: FrameID,
        detection_bbox: Dict[str, float]
    ) -> bool:
        """Check if similar detection already exists."""
        # Check for detection within small tolerance
        tolerance = 0.01  # 1% tolerance for bbox coordinates
        
        query = """
        SELECT EXISTS(
            SELECT 1 FROM detections
            WHERE camera_id = $1 
            AND frame_id = $2
            AND ABS(bbox_x - $3) < $4
            AND ABS(bbox_y - $5) < $4
            AND ABS(bbox_width - $6) < $4
            AND ABS(bbox_height - $7) < $4
        )
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    query,
                    str(camera_id),
                    str(frame_id),
                    detection_bbox['x'],
                    tolerance,
                    detection_bbox['y'],
                    detection_bbox['width'],
                    detection_bbox['height']
                )
                return bool(result)
                
        except Exception as e:
            logger.error(f"Failed to check detection existence: {e}")
            return False
    
    def _row_to_detection(self, row: asyncpg.Record) -> Optional[Detection]:
        """Convert database row to Detection entity."""
        try:
            # Create bounding box
            bbox = BoundingBox(
                x=float(row['bbox_x']),
                y=float(row['bbox_y']),
                width=float(row['bbox_width']),
                height=float(row['bbox_height']),
                normalized=row['normalized_coords']
            )
            
            # Create detection entity
            detection = Detection(
                camera_id=CameraID(row['camera_id']),
                frame_id=FrameID(row['frame_id']),
                bbox=bbox,
                confidence=Confidence(float(row['confidence'])),
                detection_class=DetectionClass(row['detection_class']),
                timestamp=row['timestamp']
            )
            
            return detection
            
        except Exception as e:
            logger.error(f"Failed to convert row to detection: {e}")
            return None