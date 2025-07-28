"""
Result aggregator for pipeline result collection and analysis.

Handles:
- Result collection from all pipeline stages
- Result analysis and quality assessment
- Performance metrics aggregation
- Historical result tracking
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import statistics
import json

from app.domains.detection.entities.detection import Detection, DetectionBatch
from app.domains.reid.entities.person_identity import PersonIdentity
from app.domains.mapping.entities.trajectory import Trajectory
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates and analyzes pipeline results.
    
    Features:
    - Multi-stage result collection
    - Performance metrics aggregation
    - Quality assessment and validation
    - Historical result tracking
    - Trend analysis
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # Result history storage
        self.result_history: deque = deque(maxlen=history_size)
        self.performance_history: deque = deque(maxlen=history_size)
        
        # Aggregation statistics
        self.aggregation_stats = {
            "total_results_processed": 0,
            "successful_aggregations": 0,
            "failed_aggregations": 0,
            "quality_assessments": 0,
            "trend_analyses": 0
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "detection_confidence": 0.7,
            "reid_confidence": 0.6,
            "trajectory_smoothness": 0.5,
            "trajectory_completeness": 0.7,
            "processing_time_threshold": 2.0,  # seconds
            "minimum_detections_per_frame": 1
        }
        
        # Performance metrics
        self.performance_metrics = {
            "processing_times": defaultdict(list),
            "throughput_rates": defaultdict(list),
            "quality_scores": defaultdict(list),
            "error_rates": defaultdict(list)
        }
        
        logger.info("ResultAggregator initialized")
    
    async def aggregate_pipeline_results(
        self,
        pipeline_results: Dict[str, Any],
        include_quality_assessment: bool = True
    ) -> Dict[str, Any]:
        """
        Aggregate results from complete pipeline execution.
        
        Args:
            pipeline_results: Results from all pipeline stages
            include_quality_assessment: Whether to include quality assessment
            
        Returns:
            Aggregated results with analysis
        """
        try:
            self.aggregation_stats["total_results_processed"] += 1
            
            # Extract stage results
            detection_results = pipeline_results.get("detection_results", {})
            reid_results = pipeline_results.get("reid_results", {})
            mapping_results = pipeline_results.get("mapping_results", {})
            
            # Aggregate stage-specific metrics
            stage_metrics = await self._aggregate_stage_metrics(
                detection_results, reid_results, mapping_results
            )
            
            # Aggregate performance metrics
            performance_metrics = self._aggregate_performance_metrics(pipeline_results)
            
            # Quality assessment
            quality_assessment = None
            if include_quality_assessment:
                quality_assessment = await self._assess_result_quality(pipeline_results)
                self.aggregation_stats["quality_assessments"] += 1
            
            # Create aggregated result
            aggregated_result = {
                "aggregation_timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_results": pipeline_results,
                "stage_metrics": stage_metrics,
                "performance_metrics": performance_metrics,
                "quality_assessment": quality_assessment,
                "aggregation_metadata": {
                    "total_processing_time": pipeline_results.get("total_processing_time", 0.0),
                    "stages_completed": self._get_completed_stages(pipeline_results),
                    "result_counts": self._get_result_counts(pipeline_results)
                }
            }
            
            # Store in history
            self._store_result_in_history(aggregated_result)
            
            # Update performance tracking
            self._update_performance_tracking(aggregated_result)
            
            self.aggregation_stats["successful_aggregations"] += 1
            
            logger.info(f"Pipeline results aggregated successfully")
            
            return aggregated_result
            
        except Exception as e:
            self.aggregation_stats["failed_aggregations"] += 1
            logger.error(f"Error aggregating pipeline results: {e}")
            raise
    
    async def _aggregate_stage_metrics(
        self,
        detection_results: Dict[str, Any],
        reid_results: Dict[str, Any],
        mapping_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate metrics from individual pipeline stages."""
        try:
            stage_metrics = {
                "detection": {
                    "processing_time": detection_results.get("processing_time", 0.0),
                    "detection_count": detection_results.get("detection_count", 0),
                    "average_confidence": 0.0,
                    "cameras_processed": 0
                },
                "reid": {
                    "processing_time": reid_results.get("processing_time", 0.0),
                    "identity_count": reid_results.get("identity_count", 0),
                    "average_confidence": 0.0,
                    "cross_camera_matches": 0
                },
                "mapping": {
                    "processing_time": mapping_results.get("processing_time", 0.0),
                    "trajectory_count": mapping_results.get("trajectory_count", 0),
                    "average_smoothness": 0.0,
                    "average_completeness": 0.0
                }
            }
            
            # Calculate detection metrics
            detection_batch = detection_results.get("detection_batch")
            if detection_batch and hasattr(detection_batch, 'detections'):
                detections = detection_batch.detections
                if detections:
                    confidences = [d.confidence for d in detections if hasattr(d, 'confidence')]
                    if confidences:
                        stage_metrics["detection"]["average_confidence"] = statistics.mean(confidences)
                    
                    # Count unique cameras
                    cameras = set(d.camera_id for d in detections if hasattr(d, 'camera_id'))
                    stage_metrics["detection"]["cameras_processed"] = len(cameras)
            
            # Calculate ReID metrics
            reid_data = reid_results.get("reid_results", {})
            if reid_data and "identities" in reid_data:
                identities = reid_data["identities"]
                if identities:
                    # Calculate average confidence
                    confidences = []
                    cross_camera_count = 0
                    
                    for identity in identities.values():
                        if hasattr(identity, 'identity_confidence'):
                            confidences.append(identity.identity_confidence)
                        
                        # Count cross-camera matches
                        if hasattr(identity, 'cameras_seen') and len(identity.cameras_seen) > 1:
                            cross_camera_count += 1
                    
                    if confidences:
                        stage_metrics["reid"]["average_confidence"] = statistics.mean(confidences)
                    
                    stage_metrics["reid"]["cross_camera_matches"] = cross_camera_count
            
            # Calculate mapping metrics
            trajectories = mapping_results.get("trajectories", [])
            if trajectories:
                smoothness_scores = []
                completeness_scores = []
                
                for trajectory in trajectories:
                    if hasattr(trajectory, 'smoothness_score') and trajectory.smoothness_score:
                        smoothness_scores.append(trajectory.smoothness_score)
                    
                    if hasattr(trajectory, 'completeness_score') and trajectory.completeness_score:
                        completeness_scores.append(trajectory.completeness_score)
                
                if smoothness_scores:
                    stage_metrics["mapping"]["average_smoothness"] = statistics.mean(smoothness_scores)
                
                if completeness_scores:
                    stage_metrics["mapping"]["average_completeness"] = statistics.mean(completeness_scores)
            
            return stage_metrics
            
        except Exception as e:
            logger.error(f"Error aggregating stage metrics: {e}")
            return {
                "detection": {"processing_time": 0.0, "detection_count": 0},
                "reid": {"processing_time": 0.0, "identity_count": 0},
                "mapping": {"processing_time": 0.0, "trajectory_count": 0}
            }
    
    def _aggregate_performance_metrics(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate performance metrics from pipeline results."""
        try:
            stage_times = pipeline_results.get("stage_times", {})
            total_time = pipeline_results.get("total_processing_time", 0.0)
            
            # Calculate throughput metrics
            detection_count = pipeline_results.get("detection_results", {}).get("detection_count", 0)
            identity_count = pipeline_results.get("reid_results", {}).get("identity_count", 0)
            trajectory_count = pipeline_results.get("mapping_results", {}).get("trajectory_count", 0)
            
            throughput_metrics = {
                "overall_throughput": (detection_count + identity_count + trajectory_count) / max(0.001, total_time),
                "detection_throughput": detection_count / max(0.001, stage_times.get("detection", 0.001)),
                "reid_throughput": identity_count / max(0.001, stage_times.get("reid", 0.001)),
                "mapping_throughput": trajectory_count / max(0.001, stage_times.get("mapping", 0.001))
            }
            
            # Calculate efficiency metrics
            efficiency_metrics = {
                "detection_efficiency": stage_times.get("detection", 0.0) / max(0.001, total_time),
                "reid_efficiency": stage_times.get("reid", 0.0) / max(0.001, total_time),
                "mapping_efficiency": stage_times.get("mapping", 0.0) / max(0.001, total_time),
                "pipeline_efficiency": total_time / max(0.001, sum(stage_times.values()))
            }
            
            # Resource utilization
            resource_metrics = {
                "memory_usage": pipeline_results.get("memory_usage", 0.0),
                "gpu_utilization": pipeline_results.get("gpu_utilization", 0.0),
                "cpu_utilization": pipeline_results.get("cpu_utilization", 0.0)
            }
            
            return {
                "throughput_metrics": throughput_metrics,
                "efficiency_metrics": efficiency_metrics,
                "resource_metrics": resource_metrics,
                "timing_metrics": {
                    "total_time": total_time,
                    "stage_times": stage_times,
                    "stage_percentages": {
                        stage: (time / total_time * 100) if total_time > 0 else 0
                        for stage, time in stage_times.items()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error aggregating performance metrics: {e}")
            return {"error": str(e)}
    
    async def _assess_result_quality(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of pipeline results."""
        try:
            quality_assessment = {
                "overall_quality_score": 0.0,
                "stage_quality_scores": {},
                "quality_issues": [],
                "quality_recommendations": []
            }
            
            # Assess detection quality
            detection_quality = await self._assess_detection_quality(
                pipeline_results.get("detection_results", {})
            )
            quality_assessment["stage_quality_scores"]["detection"] = detection_quality
            
            # Assess ReID quality
            reid_quality = await self._assess_reid_quality(
                pipeline_results.get("reid_results", {})
            )
            quality_assessment["stage_quality_scores"]["reid"] = reid_quality
            
            # Assess mapping quality
            mapping_quality = await self._assess_mapping_quality(
                pipeline_results.get("mapping_results", {})
            )
            quality_assessment["stage_quality_scores"]["mapping"] = mapping_quality
            
            # Calculate overall quality score
            stage_scores = [
                detection_quality.get("quality_score", 0.0),
                reid_quality.get("quality_score", 0.0),
                mapping_quality.get("quality_score", 0.0)
            ]
            quality_assessment["overall_quality_score"] = statistics.mean(stage_scores)
            
            # Collect quality issues
            for stage_quality in [detection_quality, reid_quality, mapping_quality]:
                quality_assessment["quality_issues"].extend(stage_quality.get("issues", []))
                quality_assessment["quality_recommendations"].extend(stage_quality.get("recommendations", []))
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error assessing result quality: {e}")
            return {"error": str(e)}
    
    async def _assess_detection_quality(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess detection stage quality."""
        try:
            quality_score = 1.0
            issues = []
            recommendations = []
            
            detection_batch = detection_results.get("detection_batch")
            processing_time = detection_results.get("processing_time", 0.0)
            
            # Check processing time
            if processing_time > self.quality_thresholds["processing_time_threshold"]:
                quality_score -= 0.2
                issues.append(f"Detection processing time ({processing_time:.2f}s) exceeds threshold")
                recommendations.append("Consider GPU acceleration or model optimization")
            
            # Check detection count
            detection_count = detection_results.get("detection_count", 0)
            if detection_count < self.quality_thresholds["minimum_detections_per_frame"]:
                quality_score -= 0.3
                issues.append(f"Low detection count: {detection_count}")
                recommendations.append("Check input frame quality or detection model sensitivity")
            
            # Check detection confidence
            if detection_batch and hasattr(detection_batch, 'detections'):
                confidences = [d.confidence for d in detection_batch.detections if hasattr(d, 'confidence')]
                if confidences:
                    avg_confidence = statistics.mean(confidences)
                    if avg_confidence < self.quality_thresholds["detection_confidence"]:
                        quality_score -= 0.2
                        issues.append(f"Low average detection confidence: {avg_confidence:.2f}")
                        recommendations.append("Consider model retraining or confidence threshold adjustment")
            
            return {
                "quality_score": max(0.0, quality_score),
                "issues": issues,
                "recommendations": recommendations,
                "metrics": {
                    "processing_time": processing_time,
                    "detection_count": detection_count,
                    "average_confidence": statistics.mean(confidences) if confidences else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing detection quality: {e}")
            return {"quality_score": 0.0, "issues": [str(e)], "recommendations": []}
    
    async def _assess_reid_quality(self, reid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ReID stage quality."""
        try:
            quality_score = 1.0
            issues = []
            recommendations = []
            
            reid_data = reid_results.get("reid_results", {})
            processing_time = reid_results.get("processing_time", 0.0)
            
            # Check processing time
            if processing_time > self.quality_thresholds["processing_time_threshold"]:
                quality_score -= 0.2
                issues.append(f"ReID processing time ({processing_time:.2f}s) exceeds threshold")
                recommendations.append("Consider batch processing optimization")
            
            # Check identity count
            identity_count = reid_results.get("identity_count", 0)
            if identity_count == 0:
                quality_score -= 0.5
                issues.append("No identities found")
                recommendations.append("Check detection input quality or ReID model performance")
            
            # Check ReID confidence
            if reid_data and "identities" in reid_data:
                identities = reid_data["identities"]
                confidences = []
                
                for identity in identities.values():
                    if hasattr(identity, 'identity_confidence'):
                        confidences.append(identity.identity_confidence)
                
                if confidences:
                    avg_confidence = statistics.mean(confidences)
                    if avg_confidence < self.quality_thresholds["reid_confidence"]:
                        quality_score -= 0.2
                        issues.append(f"Low average ReID confidence: {avg_confidence:.2f}")
                        recommendations.append("Consider feature extraction optimization")
            
            return {
                "quality_score": max(0.0, quality_score),
                "issues": issues,
                "recommendations": recommendations,
                "metrics": {
                    "processing_time": processing_time,
                    "identity_count": identity_count,
                    "average_confidence": statistics.mean(confidences) if confidences else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing ReID quality: {e}")
            return {"quality_score": 0.0, "issues": [str(e)], "recommendations": []}
    
    async def _assess_mapping_quality(self, mapping_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess mapping stage quality."""
        try:
            quality_score = 1.0
            issues = []
            recommendations = []
            
            processing_time = mapping_results.get("processing_time", 0.0)
            trajectories = mapping_results.get("trajectories", [])
            
            # Check processing time
            if processing_time > self.quality_thresholds["processing_time_threshold"]:
                quality_score -= 0.2
                issues.append(f"Mapping processing time ({processing_time:.2f}s) exceeds threshold")
                recommendations.append("Consider coordinate transformation optimization")
            
            # Check trajectory count
            trajectory_count = len(trajectories)
            if trajectory_count == 0:
                quality_score -= 0.3
                issues.append("No trajectories generated")
                recommendations.append("Check ReID input quality or trajectory building parameters")
            
            # Check trajectory quality
            if trajectories:
                smoothness_scores = []
                completeness_scores = []
                
                for trajectory in trajectories:
                    if hasattr(trajectory, 'smoothness_score') and trajectory.smoothness_score:
                        smoothness_scores.append(trajectory.smoothness_score)
                    
                    if hasattr(trajectory, 'completeness_score') and trajectory.completeness_score:
                        completeness_scores.append(trajectory.completeness_score)
                
                # Check smoothness
                if smoothness_scores:
                    avg_smoothness = statistics.mean(smoothness_scores)
                    if avg_smoothness < self.quality_thresholds["trajectory_smoothness"]:
                        quality_score -= 0.15
                        issues.append(f"Low trajectory smoothness: {avg_smoothness:.2f}")
                        recommendations.append("Consider trajectory smoothing parameters")
                
                # Check completeness
                if completeness_scores:
                    avg_completeness = statistics.mean(completeness_scores)
                    if avg_completeness < self.quality_thresholds["trajectory_completeness"]:
                        quality_score -= 0.15
                        issues.append(f"Low trajectory completeness: {avg_completeness:.2f}")
                        recommendations.append("Check camera coverage or trajectory gap handling")
            
            return {
                "quality_score": max(0.0, quality_score),
                "issues": issues,
                "recommendations": recommendations,
                "metrics": {
                    "processing_time": processing_time,
                    "trajectory_count": trajectory_count,
                    "average_smoothness": statistics.mean(smoothness_scores) if smoothness_scores else 0.0,
                    "average_completeness": statistics.mean(completeness_scores) if completeness_scores else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing mapping quality: {e}")
            return {"quality_score": 0.0, "issues": [str(e)], "recommendations": []}
    
    def _get_completed_stages(self, pipeline_results: Dict[str, Any]) -> List[str]:
        """Get list of completed pipeline stages."""
        completed_stages = []
        
        if pipeline_results.get("detection_results"):
            completed_stages.append("detection")
        
        if pipeline_results.get("reid_results"):
            completed_stages.append("reid")
        
        if pipeline_results.get("mapping_results"):
            completed_stages.append("mapping")
        
        return completed_stages
    
    def _get_result_counts(self, pipeline_results: Dict[str, Any]) -> Dict[str, int]:
        """Get counts of results from each stage."""
        return {
            "detections": pipeline_results.get("detection_results", {}).get("detection_count", 0),
            "identities": pipeline_results.get("reid_results", {}).get("identity_count", 0),
            "trajectories": pipeline_results.get("mapping_results", {}).get("trajectory_count", 0)
        }
    
    def _store_result_in_history(self, aggregated_result: Dict[str, Any]):
        """Store aggregated result in history."""
        try:
            history_entry = {
                "timestamp": aggregated_result["aggregation_timestamp"],
                "result_counts": aggregated_result["aggregation_metadata"]["result_counts"],
                "processing_time": aggregated_result["aggregation_metadata"]["total_processing_time"],
                "quality_score": aggregated_result.get("quality_assessment", {}).get("overall_quality_score", 0.0),
                "stages_completed": aggregated_result["aggregation_metadata"]["stages_completed"]
            }
            
            self.result_history.append(history_entry)
            
        except Exception as e:
            logger.error(f"Error storing result in history: {e}")
    
    def _update_performance_tracking(self, aggregated_result: Dict[str, Any]):
        """Update performance tracking metrics."""
        try:
            perf_metrics = aggregated_result.get("performance_metrics", {})
            
            # Update processing times
            timing_metrics = perf_metrics.get("timing_metrics", {})
            stage_times = timing_metrics.get("stage_times", {})
            
            for stage, time_value in stage_times.items():
                self.performance_metrics["processing_times"][stage].append(time_value)
            
            # Update throughput rates
            throughput_metrics = perf_metrics.get("throughput_metrics", {})
            for metric, value in throughput_metrics.items():
                self.performance_metrics["throughput_rates"][metric].append(value)
            
            # Update quality scores
            quality_assessment = aggregated_result.get("quality_assessment", {})
            if quality_assessment:
                overall_score = quality_assessment.get("overall_quality_score", 0.0)
                self.performance_metrics["quality_scores"]["overall"].append(overall_score)
                
                stage_scores = quality_assessment.get("stage_quality_scores", {})
                for stage, score_data in stage_scores.items():
                    score = score_data.get("quality_score", 0.0)
                    self.performance_metrics["quality_scores"][stage].append(score)
            
            # Maintain history size
            for metric_type in self.performance_metrics:
                for metric_name in self.performance_metrics[metric_type]:
                    if len(self.performance_metrics[metric_type][metric_name]) > self.history_size:
                        self.performance_metrics[metric_type][metric_name] = \
                            self.performance_metrics[metric_type][metric_name][-self.history_size:]
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            **self.aggregation_stats,
            "success_rate": (
                self.aggregation_stats["successful_aggregations"] / 
                max(1, self.aggregation_stats["total_results_processed"])
            ),
            "history_size": len(self.result_history),
            "performance_metrics_size": {
                metric_type: len(self.performance_metrics[metric_type]) 
                for metric_type in self.performance_metrics
            }
        }
    
    def get_trend_analysis(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get trend analysis for recent results."""
        try:
            self.aggregation_stats["trend_analyses"] += 1
            
            # Filter recent results
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            recent_results = [
                result for result in self.result_history
                if datetime.fromisoformat(result["timestamp"].replace('Z', '+00:00')) > cutoff_time
            ]
            
            if not recent_results:
                return {"message": "No recent results for trend analysis"}
            
            # Calculate trends
            processing_times = [r["processing_time"] for r in recent_results]
            quality_scores = [r["quality_score"] for r in recent_results]
            
            trend_analysis = {
                "time_window_hours": time_window_hours,
                "sample_size": len(recent_results),
                "processing_time_trend": {
                    "average": statistics.mean(processing_times),
                    "median": statistics.median(processing_times),
                    "std_dev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                    "min": min(processing_times),
                    "max": max(processing_times)
                },
                "quality_trend": {
                    "average": statistics.mean(quality_scores),
                    "median": statistics.median(quality_scores),
                    "std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                    "min": min(quality_scores),
                    "max": max(quality_scores)
                },
                "throughput_trend": {
                    "results_per_hour": len(recent_results) / time_window_hours
                }
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {e}")
            return {"error": str(e)}
    
    def reset_stats(self):
        """Reset all aggregation statistics."""
        self.aggregation_stats = {
            "total_results_processed": 0,
            "successful_aggregations": 0,
            "failed_aggregations": 0,
            "quality_assessments": 0,
            "trend_analyses": 0
        }
        
        self.result_history.clear()
        self.performance_history.clear()
        
        for metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].clear()
        
        logger.info("Result aggregation statistics reset")