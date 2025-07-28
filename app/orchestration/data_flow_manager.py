"""
Data flow manager for coordinating pipeline data flow.

Manages:
- Data flow between pipeline stages
- Data format validation and conversion
- Pipeline data buffering and caching
- Performance optimization for data transfers
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict, deque
import time

from app.domains.detection.entities.detection import Detection, DetectionBatch
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.trajectory import Trajectory
from app.orchestration.feature_integrator import FeatureIntegrator
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class DataFlowManager:
    """
    Manages data flow coordination between pipeline stages.
    
    Features:
    - Stage-to-stage data flow coordination
    - Data format validation and conversion
    - Pipeline data buffering and caching
    - Performance optimization for data transfers
    """
    
    def __init__(self):
        self.feature_integrator = FeatureIntegrator()
        
        # Data flow buffers
        self.detection_buffer: deque = deque(maxlen=100)
        self.reid_buffer: deque = deque(maxlen=100)
        self.mapping_buffer: deque = deque(maxlen=100)
        
        # Data flow statistics
        self.flow_stats = {
            "total_flows": 0,
            "successful_flows": 0,
            "failed_flows": 0,
            "detection_to_reid_flows": 0,
            "reid_to_mapping_flows": 0,
            "end_to_end_flows": 0,
            "average_flow_time": 0.0,
            "buffer_overflows": 0
        }
        
        # Performance tracking
        self.flow_times: List[float] = []
        
        # Data validation rules
        self.validation_rules = {
            "detection_batch": {
                "required_fields": ["detections", "camera_frames", "batch_id", "timestamp"],
                "detection_fields": ["id", "camera_id", "bbox", "confidence", "timestamp"]
            },
            "reid_results": {
                "required_fields": ["identities", "processing_time"],
                "identity_fields": ["global_id", "cameras_seen", "track_ids_by_camera"]
            },
            "mapping_results": {
                "required_fields": ["transformed_detections", "trajectories"],
                "trajectory_fields": ["global_id", "path_points", "cameras_traversed"]
            }
        }
        
        logger.info("DataFlowManager initialized")
    
    async def coordinate_pipeline_flow(
        self,
        detection_results: Dict[str, Any],
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate complete pipeline data flow from detection to mapping.
        
        Args:
            detection_results: Results from detection stage
            pipeline_config: Configuration for pipeline flow
            
        Returns:
            Complete pipeline results with integrated data
        """
        flow_start_time = time.time()
        
        try:
            self.flow_stats["total_flows"] += 1
            
            # Stage 1: Detection to ReID flow
            logger.info("Coordinating detection to ReID flow")
            
            reid_input = await self._coordinate_detection_to_reid_flow(detection_results)
            
            if not reid_input:
                raise ValueError("Failed to coordinate detection to ReID flow")
            
            # Stage 2: ReID to Mapping flow
            logger.info("Coordinating ReID to mapping flow")
            
            # Simulate ReID processing (in real implementation, this would call ReID service)
            reid_results = await self._simulate_reid_processing(reid_input)
            
            mapping_input = await self._coordinate_reid_to_mapping_flow(
                reid_results, detection_results
            )
            
            if not mapping_input:
                raise ValueError("Failed to coordinate ReID to mapping flow")
            
            # Stage 3: Mapping processing
            logger.info("Coordinating mapping processing")
            
            # Simulate mapping processing (in real implementation, this would call mapping service)
            mapping_results = await self._simulate_mapping_processing(mapping_input)
            
            # Stage 4: Integration
            logger.info("Coordinating result integration")
            
            integrated_results = await self.feature_integrator.integrate_pipeline_results(
                detection_results, reid_results, mapping_results
            )
            
            # Calculate flow time
            flow_time = time.time() - flow_start_time
            self.flow_times.append(flow_time)
            
            # Update statistics
            self.flow_stats["successful_flows"] += 1
            self.flow_stats["end_to_end_flows"] += 1
            
            if len(self.flow_times) > 100:
                self.flow_times = self.flow_times[-100:]
            
            # Calculate average flow time
            if self.flow_times:
                self.flow_stats["average_flow_time"] = sum(self.flow_times) / len(self.flow_times)
            
            # Add flow metadata
            integrated_results["flow_metadata"] = {
                "total_flow_time": flow_time,
                "flow_timestamp": datetime.now(timezone.utc).isoformat(),
                "stages_completed": ["detection", "reid", "mapping", "integration"],
                "data_flow_stats": self.get_flow_stats()
            }
            
            logger.info(f"Pipeline flow completed successfully in {flow_time:.3f}s")
            
            return integrated_results
            
        except Exception as e:
            self.flow_stats["failed_flows"] += 1
            logger.error(f"Error coordinating pipeline flow: {e}")
            raise
    
    async def _coordinate_detection_to_reid_flow(
        self,
        detection_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Coordinate data flow from detection to ReID stage."""
        try:
            # Validate detection results
            if not self._validate_detection_results(detection_results):
                logger.error("Detection results validation failed")
                return None
            
            # Convert detection format for ReID
            detection_batch = detection_results.get("detection_batch")
            if not detection_batch:
                logger.error("No detection batch found in detection results")
                return None
            
            # Convert to ReID format
            reid_format = self.feature_integrator.convert_detection_to_reid_format(detection_batch)
            
            if "error" in reid_format:
                logger.error(f"Error converting detection to ReID format: {reid_format['error']}")
                return None
            
            # Buffer detection data
            self._buffer_detection_data(detection_results)
            
            # Update flow statistics
            self.flow_stats["detection_to_reid_flows"] += 1
            
            logger.info(f"Detection to ReID flow coordinated: {len(reid_format.get('person_detections', []))} detections")
            
            return {
                "reid_input": reid_format,
                "original_detection_results": detection_results,
                "flow_stage": "detection_to_reid",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error coordinating detection to ReID flow: {e}")
            return None
    
    async def _coordinate_reid_to_mapping_flow(
        self,
        reid_results: Dict[str, Any],
        detection_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Coordinate data flow from ReID to mapping stage."""
        try:
            # Validate ReID results
            if not self._validate_reid_results(reid_results):
                logger.error("ReID results validation failed")
                return None
            
            # Convert ReID format for mapping
            detection_batch = detection_results.get("detection_batch")
            if not detection_batch:
                logger.error("No detection batch found for mapping conversion")
                return None
            
            # Convert to mapping format
            mapping_format = self.feature_integrator.convert_reid_to_mapping_format(
                reid_results, detection_batch
            )
            
            if "error" in mapping_format:
                logger.error(f"Error converting ReID to mapping format: {mapping_format['error']}")
                return None
            
            # Buffer ReID data
            self._buffer_reid_data(reid_results)
            
            # Update flow statistics
            self.flow_stats["reid_to_mapping_flows"] += 1
            
            logger.info(f"ReID to mapping flow coordinated: {len(mapping_format.get('identity_mappings', {}))} identities")
            
            return {
                "mapping_input": mapping_format,
                "original_reid_results": reid_results,
                "flow_stage": "reid_to_mapping",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error coordinating ReID to mapping flow: {e}")
            return None
    
    async def _simulate_reid_processing(self, reid_input: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate ReID processing for flow coordination."""
        try:
            # Simulate processing delay
            await asyncio.sleep(0.1)
            
            # Extract person detections
            person_detections = reid_input.get("reid_input", {}).get("person_detections", [])
            
            # Create mock identities
            identities = {}
            for i, detection in enumerate(person_detections):
                identity_id = f"person_{i}"
                identities[identity_id] = {
                    "global_id": identity_id,
                    "cameras_seen": [detection.get("camera_id")],
                    "track_ids_by_camera": {detection.get("camera_id"): f"track_{i}"},
                    "identity_confidence": 0.8,
                    "first_seen": datetime.now(timezone.utc),
                    "last_seen": datetime.now(timezone.utc)
                }
            
            return {
                "identities": identities,
                "processing_time": 0.1,
                "identity_count": len(identities),
                "stage": "reid"
            }
            
        except Exception as e:
            logger.error(f"Error simulating ReID processing: {e}")
            return {"error": str(e)}
    
    async def _simulate_mapping_processing(self, mapping_input: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate mapping processing for flow coordination."""
        try:
            # Simulate processing delay
            await asyncio.sleep(0.1)
            
            # Extract mapping data
            mapping_data = mapping_input.get("mapping_input", {})
            identity_mappings = mapping_data.get("identity_mappings", {})
            
            # Create mock trajectories
            trajectories = []
            transformed_detections = []
            
            for identity_id, identity_data in identity_mappings.items():
                # Create mock trajectory
                trajectory = {
                    "global_id": identity_id,
                    "path_points": [
                        {"x": 100.0, "y": 200.0, "timestamp": datetime.now(timezone.utc)},
                        {"x": 105.0, "y": 205.0, "timestamp": datetime.now(timezone.utc)}
                    ],
                    "cameras_traversed": identity_data.get("cameras_seen", []),
                    "smoothness_score": 0.9,
                    "completeness_score": 0.8
                }
                trajectories.append(trajectory)
                
                # Create mock transformed detection
                transformed_detection = {
                    "detection": {
                        "id": f"det_{identity_id}",
                        "camera_id": identity_data.get("cameras_seen", ["camera_1"])[0],
                        "track_id": identity_id
                    },
                    "map_coordinates": {"x": 100.0, "y": 200.0},
                    "transformation_confidence": 0.9
                }
                transformed_detections.append(transformed_detection)
            
            return {
                "mapping_results": {
                    "transformed_detections": transformed_detections,
                    "coordinate_system": "MAP"
                },
                "trajectories": trajectories,
                "processing_time": 0.1,
                "trajectory_count": len(trajectories),
                "stage": "mapping"
            }
            
        except Exception as e:
            logger.error(f"Error simulating mapping processing: {e}")
            return {"error": str(e)}
    
    def _validate_detection_results(self, detection_results: Dict[str, Any]) -> bool:
        """Validate detection results format."""
        try:
            rules = self.validation_rules["detection_batch"]
            
            # Check required fields
            detection_batch = detection_results.get("detection_batch")
            if not detection_batch:
                return False
            
            # Check if detection_batch has required attributes
            if not hasattr(detection_batch, 'detections'):
                return False
            
            # Validate individual detections
            for detection in detection_batch.detections:
                if not hasattr(detection, 'id') or not hasattr(detection, 'camera_id'):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating detection results: {e}")
            return False
    
    def _validate_reid_results(self, reid_results: Dict[str, Any]) -> bool:
        """Validate ReID results format."""
        try:
            rules = self.validation_rules["reid_results"]
            
            # Check required fields
            if "identities" not in reid_results:
                return False
            
            # Validate identities
            identities = reid_results["identities"]
            if not isinstance(identities, dict):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ReID results: {e}")
            return False
    
    def _buffer_detection_data(self, detection_data: Dict[str, Any]):
        """Buffer detection data for flow coordination."""
        try:
            buffer_entry = {
                "data": detection_data,
                "timestamp": datetime.now(timezone.utc),
                "stage": "detection"
            }
            
            self.detection_buffer.append(buffer_entry)
            
            # Check for buffer overflow
            if len(self.detection_buffer) >= self.detection_buffer.maxlen:
                self.flow_stats["buffer_overflows"] += 1
                
        except Exception as e:
            logger.error(f"Error buffering detection data: {e}")
    
    def _buffer_reid_data(self, reid_data: Dict[str, Any]):
        """Buffer ReID data for flow coordination."""
        try:
            buffer_entry = {
                "data": reid_data,
                "timestamp": datetime.now(timezone.utc),
                "stage": "reid"
            }
            
            self.reid_buffer.append(buffer_entry)
            
            # Check for buffer overflow
            if len(self.reid_buffer) >= self.reid_buffer.maxlen:
                self.flow_stats["buffer_overflows"] += 1
                
        except Exception as e:
            logger.error(f"Error buffering ReID data: {e}")
    
    def _buffer_mapping_data(self, mapping_data: Dict[str, Any]):
        """Buffer mapping data for flow coordination."""
        try:
            buffer_entry = {
                "data": mapping_data,
                "timestamp": datetime.now(timezone.utc),
                "stage": "mapping"
            }
            
            self.mapping_buffer.append(buffer_entry)
            
            # Check for buffer overflow
            if len(self.mapping_buffer) >= self.mapping_buffer.maxlen:
                self.flow_stats["buffer_overflows"] += 1
                
        except Exception as e:
            logger.error(f"Error buffering mapping data: {e}")
    
    def get_flow_stats(self) -> Dict[str, Any]:
        """Get data flow statistics."""
        return {
            **self.flow_stats,
            "success_rate": (
                self.flow_stats["successful_flows"] / 
                max(1, self.flow_stats["total_flows"])
            ),
            "buffer_utilization": {
                "detection": len(self.detection_buffer) / self.detection_buffer.maxlen,
                "reid": len(self.reid_buffer) / self.reid_buffer.maxlen,
                "mapping": len(self.mapping_buffer) / self.mapping_buffer.maxlen
            },
            "flow_performance": {
                "avg_flow_time": self.flow_stats["average_flow_time"],
                "total_flows": len(self.flow_times)
            }
        }
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information."""
        return {
            "detection_buffer": {
                "size": len(self.detection_buffer),
                "max_size": self.detection_buffer.maxlen,
                "utilization": len(self.detection_buffer) / self.detection_buffer.maxlen
            },
            "reid_buffer": {
                "size": len(self.reid_buffer),
                "max_size": self.reid_buffer.maxlen,
                "utilization": len(self.reid_buffer) / self.reid_buffer.maxlen
            },
            "mapping_buffer": {
                "size": len(self.mapping_buffer),
                "max_size": self.mapping_buffer.maxlen,
                "utilization": len(self.mapping_buffer) / self.mapping_buffer.maxlen
            }
        }
    
    def clear_buffers(self):
        """Clear all data flow buffers."""
        self.detection_buffer.clear()
        self.reid_buffer.clear()
        self.mapping_buffer.clear()
        logger.info("All data flow buffers cleared")
    
    def reset_stats(self):
        """Reset flow statistics."""
        self.flow_stats = {
            "total_flows": 0,
            "successful_flows": 0,
            "failed_flows": 0,
            "detection_to_reid_flows": 0,
            "reid_to_mapping_flows": 0,
            "end_to_end_flows": 0,
            "average_flow_time": 0.0,
            "buffer_overflows": 0
        }
        self.flow_times.clear()
        logger.info("Data flow statistics reset")