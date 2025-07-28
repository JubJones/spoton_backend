"""
Feature integrator for seamless data flow between detection, ReID, and mapping.

Provides integration logic for:
- Data format conversion between domains
- Result aggregation and combination
- Cross-domain data validation
- Performance optimization
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from app.domains.detection.entities.detection import Detection, DetectionBatch
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.reid.entities.track import Track
from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem
from app.domains.mapping.entities.trajectory import Trajectory, TrajectoryPoint
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class FeatureIntegrator:
    """
    Integrates detection, ReID, and mapping features for seamless data flow.
    
    Features:
    - Cross-domain data conversion
    - Result aggregation and validation
    - Performance optimization
    - Data consistency checks
    """
    
    def __init__(self):
        # Integration statistics
        self.integration_stats = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "detection_to_reid_conversions": 0,
            "reid_to_mapping_conversions": 0,
            "trajectory_aggregations": 0,
            "validation_errors": 0
        }
        
        logger.info("FeatureIntegrator initialized")
    
    async def integrate_pipeline_results(
        self,
        detection_results: Dict[str, Any],
        reid_results: Dict[str, Any],
        mapping_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate results from all three pipeline stages.
        
        Args:
            detection_results: Detection stage results
            reid_results: ReID stage results
            mapping_results: Mapping stage results
            
        Returns:
            Integrated results with unified data structure
        """
        try:
            self.integration_stats["total_integrations"] += 1
            
            # Extract core data from each stage
            detection_batch = detection_results.get("detection_batch")
            reid_data = reid_results.get("reid_results", {})
            mapping_data = mapping_results.get("mapping_results", {})
            trajectories = mapping_results.get("trajectories", [])
            
            # Validate input data
            validation_result = self._validate_integration_inputs(
                detection_batch, reid_data, mapping_data
            )
            
            if not validation_result["valid"]:
                self.integration_stats["validation_errors"] += 1
                logger.warning(f"Integration validation failed: {validation_result['errors']}")
            
            # Build integrated person objects
            integrated_persons = await self._build_integrated_persons(
                detection_batch, reid_data, mapping_data, trajectories
            )
            
            # Aggregate statistics
            aggregated_stats = self._aggregate_stage_statistics(
                detection_results, reid_results, mapping_results
            )
            
            # Create final integrated result
            integrated_result = {
                "integrated_persons": integrated_persons,
                "stage_results": {
                    "detection": detection_results,
                    "reid": reid_results,
                    "mapping": mapping_results
                },
                "aggregated_stats": aggregated_stats,
                "integration_metadata": {
                    "integration_timestamp": datetime.now(timezone.utc).isoformat(),
                    "person_count": len(integrated_persons),
                    "trajectory_count": len(trajectories),
                    "validation_result": validation_result
                },
                "performance_summary": self._calculate_performance_summary(
                    detection_results, reid_results, mapping_results
                )
            }
            
            self.integration_stats["successful_integrations"] += 1
            
            logger.info(f"Successfully integrated pipeline results: {len(integrated_persons)} persons")
            
            return integrated_result
            
        except Exception as e:
            self.integration_stats["failed_integrations"] += 1
            logger.error(f"Error integrating pipeline results: {e}")
            raise
    
    def _validate_integration_inputs(
        self,
        detection_batch: Optional[DetectionBatch],
        reid_data: Dict[str, Any],
        mapping_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate integration inputs for consistency."""
        errors = []
        
        try:
            # Validate detection batch
            if detection_batch is None:
                errors.append("Missing detection batch")
            elif not hasattr(detection_batch, 'detections'):
                errors.append("Invalid detection batch format")
            
            # Validate ReID data
            if not isinstance(reid_data, dict):
                errors.append("Invalid ReID data format")
            elif "identities" not in reid_data:
                errors.append("Missing identities in ReID data")
            
            # Validate mapping data
            if not isinstance(mapping_data, dict):
                errors.append("Invalid mapping data format")
            elif "transformed_detections" not in mapping_data:
                errors.append("Missing transformed detections in mapping data")
            
            # Cross-validation checks
            if detection_batch and reid_data.get("identities"):
                # Check if ReID identities match detection count expectations
                detection_count = len(detection_batch.detections) if hasattr(detection_batch, 'detections') else 0
                identity_count = len(reid_data["identities"])
                
                if identity_count > detection_count:
                    errors.append(f"More identities ({identity_count}) than detections ({detection_count})")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating integration inputs: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _build_integrated_persons(
        self,
        detection_batch: Optional[DetectionBatch],
        reid_data: Dict[str, Any],
        mapping_data: Dict[str, Any],
        trajectories: List[Trajectory]
    ) -> List[Dict[str, Any]]:
        """Build integrated person objects from all stages."""
        integrated_persons = []
        
        try:
            # Get identities from ReID
            identities = reid_data.get("identities", {})
            
            # Get transformed detections from mapping
            transformed_detections = mapping_data.get("transformed_detections", [])
            
            # Build trajectory lookup
            trajectory_lookup = {
                trajectory.global_id: trajectory 
                for trajectory in trajectories
            }
            
            # Process each identity
            for identity_id, identity in identities.items():
                try:
                    # Get associated detections
                    person_detections = self._get_detections_for_identity(
                        identity, detection_batch, transformed_detections
                    )
                    
                    # Get trajectory
                    trajectory = trajectory_lookup.get(identity_id)
                    
                    # Build integrated person object
                    integrated_person = {
                        "global_id": identity_id,
                        "identity": identity,
                        "detections": person_detections,
                        "trajectory": trajectory,
                        "cameras_seen": list(identity.cameras_seen) if hasattr(identity, 'cameras_seen') else [],
                        "track_count": len(identity.track_ids_by_camera) if hasattr(identity, 'track_ids_by_camera') else 0,
                        "detection_count": len(person_detections),
                        "first_seen": identity.first_seen.isoformat() if hasattr(identity, 'first_seen') else None,
                        "last_seen": identity.last_seen.isoformat() if hasattr(identity, 'last_seen') else None,
                        "confidence": getattr(identity, 'identity_confidence', 0.0),
                        "trajectory_quality": {
                            "smoothness": trajectory.smoothness_score if trajectory else 0.0,
                            "completeness": trajectory.completeness_score if trajectory else 0.0,
                            "point_count": len(trajectory.path_points) if trajectory else 0
                        }
                    }
                    
                    integrated_persons.append(integrated_person)
                    
                except Exception as e:
                    logger.error(f"Error building integrated person for identity {identity_id}: {e}")
                    continue
            
            return integrated_persons
            
        except Exception as e:
            logger.error(f"Error building integrated persons: {e}")
            return []
    
    def _get_detections_for_identity(
        self,
        identity: PersonIdentity,
        detection_batch: Optional[DetectionBatch],
        transformed_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get detections associated with an identity."""
        person_detections = []
        
        try:
            if not detection_batch or not hasattr(detection_batch, 'detections'):
                return person_detections
            
            # Get identity track IDs
            identity_track_ids = getattr(identity, 'track_ids_by_camera', {})
            
            # Find detections for this identity
            for detection in detection_batch.detections:
                # Check if detection belongs to this identity
                if hasattr(detection, 'track_id') and detection.track_id:
                    if detection.track_id in identity_track_ids.values():
                        # Find corresponding transformed detection
                        transformed_detection = None
                        for trans_det in transformed_detections:
                            if trans_det.get("detection") and trans_det["detection"].id == detection.id:
                                transformed_detection = trans_det
                                break
                        
                        person_detections.append({
                            "detection": detection,
                            "transformed_detection": transformed_detection,
                            "camera_id": detection.camera_id,
                            "timestamp": detection.timestamp.isoformat(),
                            "confidence": detection.confidence,
                            "bbox": {
                                "x": detection.bbox.x,
                                "y": detection.bbox.y,
                                "width": detection.bbox.width,
                                "height": detection.bbox.height
                            } if hasattr(detection, 'bbox') else None
                        })
            
            return person_detections
            
        except Exception as e:
            logger.error(f"Error getting detections for identity: {e}")
            return []
    
    def _aggregate_stage_statistics(
        self,
        detection_results: Dict[str, Any],
        reid_results: Dict[str, Any],
        mapping_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate statistics from all pipeline stages."""
        try:
            aggregated_stats = {
                "detection": {
                    "processing_time": detection_results.get("processing_time", 0.0),
                    "detection_count": detection_results.get("detection_count", 0),
                    "stage": detection_results.get("stage", "unknown")
                },
                "reid": {
                    "processing_time": reid_results.get("processing_time", 0.0),
                    "identity_count": reid_results.get("identity_count", 0),
                    "stage": reid_results.get("stage", "unknown")
                },
                "mapping": {
                    "processing_time": mapping_results.get("processing_time", 0.0),
                    "trajectory_count": mapping_results.get("trajectory_count", 0),
                    "stage": mapping_results.get("stage", "unknown")
                },
                "totals": {
                    "total_processing_time": (
                        detection_results.get("processing_time", 0.0) +
                        reid_results.get("processing_time", 0.0) +
                        mapping_results.get("processing_time", 0.0)
                    ),
                    "total_detections": detection_results.get("detection_count", 0),
                    "total_identities": reid_results.get("identity_count", 0),
                    "total_trajectories": mapping_results.get("trajectory_count", 0)
                }
            }
            
            return aggregated_stats
            
        except Exception as e:
            logger.error(f"Error aggregating stage statistics: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_summary(
        self,
        detection_results: Dict[str, Any],
        reid_results: Dict[str, Any],
        mapping_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance summary across all stages."""
        try:
            # Extract stage times
            stage_times = {
                "detection": detection_results.get("processing_time", 0.0),
                "reid": reid_results.get("processing_time", 0.0),
                "mapping": mapping_results.get("processing_time", 0.0)
            }
            
            total_time = sum(stage_times.values())
            
            # Calculate performance metrics
            performance_summary = {
                "total_processing_time": total_time,
                "stage_times": stage_times,
                "stage_percentages": {
                    stage: (time / total_time * 100) if total_time > 0 else 0
                    for stage, time in stage_times.items()
                },
                "throughput_metrics": {
                    "detections_per_second": (
                        detection_results.get("detection_count", 0) / 
                        max(0.001, stage_times["detection"])
                    ),
                    "identities_per_second": (
                        reid_results.get("identity_count", 0) / 
                        max(0.001, stage_times["reid"])
                    ),
                    "trajectories_per_second": (
                        mapping_results.get("trajectory_count", 0) / 
                        max(0.001, stage_times["mapping"])
                    )
                },
                "efficiency_metrics": {
                    "detection_efficiency": (
                        detection_results.get("detection_count", 0) / 
                        max(1, len(detection_results.get("detection_batch", {}).get("camera_frames", {})))
                    ),
                    "reid_efficiency": (
                        reid_results.get("identity_count", 0) / 
                        max(1, detection_results.get("detection_count", 1))
                    ),
                    "mapping_efficiency": (
                        mapping_results.get("trajectory_count", 0) / 
                        max(1, reid_results.get("identity_count", 1))
                    )
                }
            }
            
            return performance_summary
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {"error": str(e)}
    
    def convert_detection_to_reid_format(
        self,
        detection_batch: DetectionBatch
    ) -> Dict[str, Any]:
        """Convert detection batch to ReID-compatible format."""
        try:
            self.integration_stats["detection_to_reid_conversions"] += 1
            
            # Create ReID-compatible format
            reid_format = {
                "person_detections": [],
                "batch_metadata": {
                    "batch_id": detection_batch.batch_id,
                    "timestamp": detection_batch.timestamp.isoformat(),
                    "camera_count": len(detection_batch.camera_frames)
                }
            }
            
            # Convert person detections
            for detection in detection_batch.detections:
                if hasattr(detection, 'class_id') and detection.class_id == 0:  # Person class
                    reid_format["person_detections"].append({
                        "detection_id": detection.id,
                        "camera_id": detection.camera_id,
                        "bbox": detection.bbox,
                        "confidence": detection.confidence,
                        "timestamp": detection.timestamp,
                        "frame_index": detection.frame_index
                    })
            
            return reid_format
            
        except Exception as e:
            logger.error(f"Error converting detection to ReID format: {e}")
            return {"error": str(e)}
    
    def convert_reid_to_mapping_format(
        self,
        reid_results: Dict[str, Any],
        detection_batch: DetectionBatch
    ) -> Dict[str, Any]:
        """Convert ReID results to mapping-compatible format."""
        try:
            self.integration_stats["reid_to_mapping_conversions"] += 1
            
            # Create mapping-compatible format
            mapping_format = {
                "detection_batch": detection_batch,
                "identity_mappings": {},
                "track_detections": {}
            }
            
            # Convert identity mappings
            identities = reid_results.get("identities", {})
            
            for identity_id, identity in identities.items():
                mapping_format["identity_mappings"][identity_id] = {
                    "global_id": identity_id,
                    "cameras_seen": list(identity.cameras_seen) if hasattr(identity, 'cameras_seen') else [],
                    "track_ids_by_camera": getattr(identity, 'track_ids_by_camera', {}),
                    "confidence": getattr(identity, 'identity_confidence', 0.0)
                }
            
            return mapping_format
            
        except Exception as e:
            logger.error(f"Error converting ReID to mapping format: {e}")
            return {"error": str(e)}
    
    def aggregate_trajectories(
        self,
        trajectories: List[Trajectory],
        time_window_seconds: float = 60.0
    ) -> Dict[str, Any]:
        """Aggregate trajectories for analysis."""
        try:
            self.integration_stats["trajectory_aggregations"] += 1
            
            # Group trajectories by person
            person_trajectories = {}
            
            for trajectory in trajectories:
                if trajectory.global_id not in person_trajectories:
                    person_trajectories[trajectory.global_id] = []
                person_trajectories[trajectory.global_id].append(trajectory)
            
            # Calculate aggregated metrics
            aggregated_metrics = {
                "total_persons": len(person_trajectories),
                "total_trajectories": len(trajectories),
                "person_metrics": {}
            }
            
            for person_id, person_trajs in person_trajectories.items():
                # Calculate person-specific metrics
                total_points = sum(len(traj.path_points) for traj in person_trajs)
                avg_smoothness = sum(traj.smoothness_score or 0 for traj in person_trajs) / len(person_trajs)
                avg_completeness = sum(traj.completeness_score or 0 for traj in person_trajs) / len(person_trajs)
                
                aggregated_metrics["person_metrics"][person_id] = {
                    "trajectory_count": len(person_trajs),
                    "total_points": total_points,
                    "avg_smoothness": avg_smoothness,
                    "avg_completeness": avg_completeness,
                    "cameras_traversed": len(set().union(*[traj.cameras_traversed for traj in person_trajs]))
                }
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Error aggregating trajectories: {e}")
            return {"error": str(e)}
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self.integration_stats,
            "success_rate": (
                self.integration_stats["successful_integrations"] / 
                max(1, self.integration_stats["total_integrations"])
            ),
            "conversion_counts": {
                "detection_to_reid": self.integration_stats["detection_to_reid_conversions"],
                "reid_to_mapping": self.integration_stats["reid_to_mapping_conversions"]
            }
        }
    
    def reset_stats(self):
        """Reset integration statistics."""
        self.integration_stats = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "detection_to_reid_conversions": 0,
            "reid_to_mapping_conversions": 0,
            "trajectory_aggregations": 0,
            "validation_errors": 0
        }
        logger.info("Integration statistics reset")
    
    def validate_data_consistency(
        self,
        integrated_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate consistency of integrated data."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            integrated_persons = integrated_result.get("integrated_persons", [])
            
            # Check for duplicate person IDs
            person_ids = [person["global_id"] for person in integrated_persons]
            if len(person_ids) != len(set(person_ids)):
                validation_result["errors"].append("Duplicate person IDs found")
                validation_result["valid"] = False
            
            # Check trajectory consistency
            for person in integrated_persons:
                trajectory = person.get("trajectory")
                if trajectory:
                    # Check if trajectory points match detection count
                    detection_count = person.get("detection_count", 0)
                    trajectory_points = len(trajectory.path_points) if hasattr(trajectory, 'path_points') else 0
                    
                    if abs(trajectory_points - detection_count) > detection_count * 0.5:
                        validation_result["warnings"].append(
                            f"Person {person['global_id']}: Large discrepancy between "
                            f"trajectory points ({trajectory_points}) and detections ({detection_count})"
                        )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating data consistency: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "warnings": []
            }