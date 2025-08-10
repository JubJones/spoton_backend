"""
Enhanced Analytics API Endpoints for Frontend Integration

Provides comprehensive analytics endpoints including:
- Real-time analytics with live data streaming
- Historical analytics with flexible time ranges
- Person journey analysis and behavior patterns
- Zone occupancy and heatmap generation
- Movement path analysis and prediction
- Environment configuration management
- Export and reporting capabilities
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from app.core.dependencies import get_current_user
from app.services.analytics_engine import AnalyticsEngine
from app.services.visualization_data_service import get_visualization_data_service
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.api.v1.visualization_schemas import (
    LiveAnalyticsData,
    PersonJourneyResponse,
    ExportJobResponse,
    EnvironmentConfigurationResponse,
    CameraConfigurationResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


# --- Real-Time Analytics Endpoints ---

@router.get("/real-time/metrics")
async def get_real_time_analytics_metrics(
    environment_id: Optional[str] = Query(None, description="Environment ID to filter metrics"),
    current_user: dict = Depends(get_current_user)
) -> LiveAnalyticsData:
    """
    Get comprehensive real-time analytics metrics including:
    - Current person count across all cameras
    - Movement and behavior metrics
    - System performance metrics
    - Zone occupancy data
    - Alerts and warnings
    """
    try:
        viz_service = get_visualization_data_service()
        if not viz_service:
            raise HTTPException(status_code=503, detail="Visualization service not available")
        
        # Use default environment if not specified
        env_id = environment_id or "default"
        
        # Get live analytics data
        analytics_data = await viz_service.get_live_analytics(env_id)
        
        logger.info(f"Retrieved real-time analytics for environment {env_id}")
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error fetching real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching real-time metrics: {str(e)}")


@router.get("/real-time/occupancy")
async def get_real_time_occupancy(
    environment_id: Optional[str] = Query(None, description="Environment ID"),
    include_zones: bool = Query(True, description="Include zone-specific occupancy"),
    include_cameras: bool = Query(True, description="Include camera-specific counts"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get real-time occupancy metrics with camera and zone breakdowns.
    """
    try:
        viz_service = get_visualization_data_service()
        if not viz_service:
            raise HTTPException(status_code=503, detail="Visualization service not available")
        
        env_id = environment_id or "default"
        analytics_data = await viz_service.get_live_analytics(env_id)
        
        occupancy_data = {
            "environment_id": env_id,
            "timestamp": analytics_data.timestamp.isoformat(),
            "total_persons": analytics_data.occupancy.total_persons,
            "occupancy_trend": analytics_data.occupancy.occupancy_trend,
            "peak_occupancy": analytics_data.occupancy.peak_occupancy
        }
        
        if include_cameras:
            occupancy_data["persons_per_camera"] = analytics_data.occupancy.persons_per_camera
        
        if include_zones:
            occupancy_data["zone_occupancy"] = analytics_data.occupancy.zone_occupancy
        
        return occupancy_data
        
    except Exception as e:
        logger.error(f"Error fetching real-time occupancy: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching occupancy: {str(e)}")


# --- Historical Analytics Endpoints ---

@router.get("/historical/{environment_id}/summary")
async def get_historical_analytics_summary(
    environment_id: str = Path(..., description="Environment identifier"),
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    include_patterns: bool = Query(True, description="Include movement patterns"),
    include_zones: bool = Query(True, description="Include zone analytics"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive historical analytics summary for a specific environment and date range.
    
    Includes:
    - Total person detections and unique visitors
    - Peak occupancy times and capacity analysis
    - Average dwell times and movement patterns
    - Zone-based analytics and heatmap data
    - Behavior pattern analysis and trends
    """
    try:
        analytics_engine = AnalyticsEngine()
        tracking_repo = TrackingRepository()
        
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Check if date range is reasonable (max 30 days)
        if (end_date - start_date).days > 30:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 30 days")
        
        # Get historical data
        historical_data = await tracking_repo.get_tracking_data_in_range(
            environment_id=environment_id,
            start_time=start_date,
            end_time=end_date
        )
        
        # Analyze historical data
        summary = await analytics_engine.analyze_historical_data(
            historical_data,
            include_patterns=include_patterns,
            include_zones=include_zones
        )
        
        # Add metadata
        summary.update({
            "environment_id": environment_id,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "duration_hours": (end_date - start_date).total_seconds() / 3600
            },
            "data_points": len(historical_data),
            "analysis_timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(
            f"Generated historical analytics summary for {environment_id}: "
            f"{len(historical_data)} data points from {start_date} to {end_date}"
        )
        
        return {
            "summary": summary,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching historical summary: {str(e)}")


# --- Person Journey and Behavior Analysis ---

@router.get("/persons/{global_person_id}/journey")
async def get_person_journey_analysis(
    global_person_id: str = Path(..., description="Global person identifier"),
    include_trajectory: bool = Query(True, description="Include detailed trajectory data"),
    include_behavior: bool = Query(True, description="Include behavior analysis"),
    current_user: dict = Depends(get_current_user)
) -> PersonJourneyResponse:
    """
    Get comprehensive journey analysis for a specific person.
    
    Includes:
    - Complete movement path across all cameras
    - Dwell times and zone interactions
    - Speed, direction, and movement pattern analysis
    - Camera transitions and handoff confidence
    - Behavioral insights and anomaly detection
    """
    try:
        analytics_engine = AnalyticsEngine()
        tracking_repo = TrackingRepository()
        
        # Get person tracking data
        person_data = await tracking_repo.get_person_tracking_data(global_person_id)
        
        if not person_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Person with ID {global_person_id} not found"
            )
        
        # Analyze journey with options
        journey_analysis = await analytics_engine.analyze_person_journey(
            person_data,
            include_trajectory=include_trajectory,
            include_behavior=include_behavior
        )
        
        # Create response
        journey_response = PersonJourneyResponse(
            global_person_id=global_person_id,
            journey_start=journey_analysis.get('start_time'),
            journey_end=journey_analysis.get('end_time'),
            total_duration=journey_analysis.get('duration_seconds', 0.0),
            total_distance=journey_analysis.get('total_distance', 0.0),
            average_speed=journey_analysis.get('average_speed', 0.0),
            cameras_visited=journey_analysis.get('cameras_visited', []),
            zones_visited=journey_analysis.get('zones_visited', []),
            trajectory_points=journey_analysis.get('trajectory_points', []),
            camera_transitions=journey_analysis.get('camera_transitions', []),
            dwell_times=journey_analysis.get('dwell_times', {}),
            movement_patterns=journey_analysis.get('movement_patterns', {})
        )
        
        logger.info(f"Generated journey analysis for person {global_person_id}")
        return journey_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing person journey: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing person journey: {str(e)}")


@router.get("/behavior/{global_person_id}/analysis")
async def get_person_behavior_analysis(
    global_person_id: str = Path(..., description="Global person identifier"),
    include_patterns: bool = Query(True, description="Include movement pattern analysis"),
    include_interactions: bool = Query(True, description="Include interaction analysis"),
    include_anomalies: bool = Query(True, description="Include anomaly detection"),
    time_window_hours: int = Query(24, ge=1, le=720, description="Time window for analysis (max 30 days)"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive behavioral analysis for a specific person.
    
    Includes:
    - Movement patterns and trajectory analysis
    - Zone interaction patterns and preferences
    - Temporal behavior patterns (time of day, day of week)
    - Anomaly detection and unusual behavior identification
    - Behavioral scoring and classification
    - Speed and dwell time analysis
    
    Parameters:
    - time_window_hours: Hours of historical data to analyze (default 24h, max 30 days)
    - include_patterns: Whether to perform movement pattern analysis
    - include_interactions: Whether to analyze zone interactions
    - include_anomalies: Whether to detect behavioral anomalies
    """
    try:
        analytics_engine = AnalyticsEngine()
        tracking_repo = TrackingRepository()
        
        # Calculate time range for analysis
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        # Get person data within time window
        person_data = await tracking_repo.get_person_tracking_data(
            global_person_id, 
            start_time=start_time,
            end_time=end_time
        )
        
        if not person_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Person with ID {global_person_id} not found or no data in specified time window"
            )
        
        # Analyze behavior with enhanced options
        behavior_analysis = await analytics_engine.analyze_person_behavior(
            person_data,
            include_patterns=include_patterns,
            include_interactions=include_interactions,
            include_anomalies=include_anomalies,
            time_window_hours=time_window_hours
        )
        
        # Calculate behavior summary statistics
        summary_stats = {
            "total_detections": len(person_data),
            "time_span_hours": time_window_hours,
            "unique_cameras": len(set(d.get('camera_id', '') for d in person_data)),
            "unique_zones": len(behavior_analysis.get('zones_visited', [])),
            "total_distance_traveled": behavior_analysis.get('total_distance', 0.0),
            "average_speed": behavior_analysis.get('average_speed', 0.0),
            "total_dwell_time_minutes": behavior_analysis.get('total_dwell_time_minutes', 0.0)
        }
        
        response_data = {
            "global_person_id": global_person_id,
            "analysis_configuration": {
                "time_window": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": time_window_hours
                },
                "analysis_options": {
                    "include_patterns": include_patterns,
                    "include_interactions": include_interactions,
                    "include_anomalies": include_anomalies
                }
            },
            "summary_statistics": summary_stats,
            "behavior_analysis": behavior_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Generated behavior analysis for person {global_person_id}: "
            f"{len(person_data)} detections over {time_window_hours}h"
        )
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing person behavior: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing person behavior: {str(e)}")


# --- Zone and Occupancy Analysis ---

@router.get("/zones/{zone_id}/occupancy")
async def get_zone_occupancy_analysis(
    zone_id: str = Path(..., description="Zone identifier"),
    hours: int = Query(24, ge=1, le=168, description="Hours of historical data (max 168 = 1 week)"),
    granularity: str = Query("hour", regex="^(minute|hour|day)$", description="Data granularity"),
    include_predictions: bool = Query(False, description="Include occupancy predictions"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive occupancy analysis for a specific zone.
    
    Includes:
    - Current occupancy count and status
    - Historical occupancy trends with configurable granularity
    - Peak occupancy times and capacity analysis
    - Average dwell time and visitor patterns
    - Optional: Predictive occupancy modeling
    """
    try:
        analytics_engine = AnalyticsEngine()
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get zone occupancy data with enhanced options
        occupancy_data = await analytics_engine.get_zone_occupancy(
            zone_id=zone_id,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            include_predictions=include_predictions
        )
        
        # Calculate additional metrics
        current_occupancy = occupancy_data.get('current_occupancy', 0)
        peak_occupancy = occupancy_data.get('peak_occupancy', 0)
        avg_occupancy = occupancy_data.get('average_occupancy', 0.0)
        
        # Determine occupancy status
        occupancy_status = "normal"
        if current_occupancy > peak_occupancy * 0.8:
            occupancy_status = "high"
        elif current_occupancy < avg_occupancy * 0.3:
            occupancy_status = "low"
        
        response_data = {
            "zone_id": zone_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours,
                "granularity": granularity
            },
            "current_status": {
                "occupancy_count": current_occupancy,
                "status": occupancy_status,
                "last_updated": end_time.isoformat()
            },
            "metrics": {
                "peak_occupancy": peak_occupancy,
                "average_occupancy": avg_occupancy,
                "total_visitors": occupancy_data.get('total_visitors', 0),
                "average_dwell_time_minutes": occupancy_data.get('avg_dwell_time_minutes', 0.0)
            },
            "occupancy_data": occupancy_data,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Generated zone occupancy analysis for {zone_id}: "
            f"{current_occupancy} current, {peak_occupancy} peak"
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching zone occupancy: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching zone occupancy: {str(e)}")


# --- Heatmap and Movement Analysis ---

@router.get("/heatmap/{environment_id}")
async def generate_heatmap_data(
    environment_id: str = Path(..., description="Environment identifier"),
    hours: int = Query(24, ge=1, le=168, description="Hours of historical data"),
    resolution: int = Query(50, ge=10, le=200, description="Grid resolution (10-200)"),
    heatmap_type: str = Query("occupancy", regex="^(occupancy|movement|dwell_time)$", description="Type of heatmap"),
    normalize: bool = Query(True, description="Normalize heatmap values"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate comprehensive heatmap data for person movement and occupancy analysis.
    
    Supports different heatmap types:
    - occupancy: Person presence density
    - movement: Movement frequency and paths
    - dwell_time: Average time spent in areas
    
    Parameters:
    - environment_id: Environment to analyze
    - hours: Hours of historical data to include (max 1 week)
    - resolution: Grid resolution for heatmap analysis
    - heatmap_type: Type of analysis to perform
    - normalize: Whether to normalize values to 0-1 range
    """
    try:
        analytics_engine = AnalyticsEngine()
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Generate heatmap data with enhanced options
        heatmap_data = await analytics_engine.generate_heatmap(
            environment_id=environment_id,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution,
            heatmap_type=heatmap_type,
            normalize=normalize
        )
        
        # Calculate heatmap statistics
        values = heatmap_data.get('data', [])
        flat_values = [v for row in values for v in row] if values else []
        
        statistics = {
            "max_value": max(flat_values) if flat_values else 0.0,
            "min_value": min(flat_values) if flat_values else 0.0,
            "avg_value": sum(flat_values) / len(flat_values) if flat_values else 0.0,
            "data_points": len(flat_values),
            "non_zero_points": len([v for v in flat_values if v > 0])
        }
        
        response_data = {
            "environment_id": environment_id,
            "configuration": {
                "heatmap_type": heatmap_type,
                "resolution": resolution,
                "normalized": normalize,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours
                }
            },
            "statistics": statistics,
            "heatmap_data": heatmap_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Generated {heatmap_type} heatmap for {environment_id}: "
            f"{resolution}x{resolution} resolution, {len(flat_values)} data points"
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating heatmap: {str(e)}")


@router.get("/paths/{environment_id}")
async def get_movement_path_analysis(
    environment_id: str = Path(..., description="Environment identifier"),
    hours: int = Query(24, ge=1, le=168, description="Hours of historical data"),
    min_path_length: int = Query(3, ge=2, le=20, description="Minimum path length for analysis"),
    cluster_paths: bool = Query(True, description="Enable path clustering analysis"),
    include_flows: bool = Query(True, description="Include flow direction analysis"),
    path_smoothing: bool = Query(True, description="Apply path smoothing"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive movement path analysis for an environment.
    
    Includes:
    - Common movement patterns and trajectories
    - Path clustering and similarity analysis
    - Entry/exit point identification and statistics
    - Flow direction analysis and congestion points
    - Speed analysis and dwell time correlations
    
    Parameters:
    - min_path_length: Minimum number of detection points to consider a path
    - cluster_paths: Whether to group similar paths together
    - include_flows: Whether to analyze directional flows
    - path_smoothing: Whether to smooth trajectory data
    """
    try:
        analytics_engine = AnalyticsEngine()
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get movement path analysis with enhanced options
        path_analysis = await analytics_engine.analyze_movement_paths(
            environment_id=environment_id,
            start_time=start_time,
            end_time=end_time,
            min_path_length=min_path_length,
            cluster_paths=cluster_paths,
            include_flows=include_flows,
            path_smoothing=path_smoothing
        )
        
        # Calculate path statistics
        total_paths = path_analysis.get('total_paths', 0)
        avg_path_length = path_analysis.get('average_path_length', 0.0)
        unique_persons = path_analysis.get('unique_persons', 0)
        
        response_data = {
            "environment_id": environment_id,
            "analysis_configuration": {
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours
                },
                "parameters": {
                    "min_path_length": min_path_length,
                    "cluster_paths": cluster_paths,
                    "include_flows": include_flows,
                    "path_smoothing": path_smoothing
                }
            },
            "summary_statistics": {
                "total_paths_analyzed": total_paths,
                "unique_persons": unique_persons,
                "average_path_length": avg_path_length,
                "paths_per_person": total_paths / max(1, unique_persons)
            },
            "path_analysis": path_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Generated movement path analysis for {environment_id}: "
            f"{total_paths} paths, {unique_persons} unique persons"
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error analyzing movement paths: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing movement paths: {str(e)}")


# --- Environment Configuration Endpoints ---

@router.get("/environments")
async def get_available_environments(
    current_user: dict = Depends(get_current_user)
) -> List[EnvironmentConfigurationResponse]:
    """
    Get list of all available environments for analytics.
    
    Returns configuration details for each environment including:
    - Camera configurations and capabilities
    - Zone definitions and layouts
    - Available date ranges for historical data
    """
    try:
        analytics_engine = AnalyticsEngine()
        environments = await analytics_engine.get_available_environments()
        
        logger.info(f"Retrieved {len(environments)} available environments")
        return environments
        
    except Exception as e:
        logger.error(f"Error fetching available environments: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching environments: {str(e)}")


@router.get("/environments/{environment_id}")
async def get_environment_details(
    environment_id: str = Path(..., description="Environment identifier"),
    current_user: dict = Depends(get_current_user)
) -> EnvironmentConfigurationResponse:
    """
    Get detailed configuration for a specific environment.
    """
    try:
        analytics_engine = AnalyticsEngine()
        environment_config = await analytics_engine.get_environment_configuration(environment_id)
        
        if not environment_config:
            raise HTTPException(status_code=404, detail=f"Environment {environment_id} not found")
        
        logger.info(f"Retrieved configuration for environment {environment_id}")
        return environment_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching environment details: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching environment details: {str(e)}")


# --- Report Generation and Export ---

@router.post("/reports/generate")
async def generate_custom_analytics_report(
    report_config: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> ExportJobResponse:
    """
    Generate a custom analytics report based on specified parameters.
    
    Report can include:
    - Comprehensive time-based analysis
    - Zone-specific metrics and comparisons
    - Person journey and behavior analysis
    - Multi-environment comparative analysis
    - Custom KPI calculations
    
    Report Configuration:
    {
        "environment_id": "campus",
        "report_type": "comprehensive|occupancy|movement|behavior",
        "time_range": {
            "start": "2023-10-01T00:00:00Z",
            "end": "2023-10-31T23:59:59Z"
        },
        "options": {
            "include_heatmaps": true,
            "include_trajectories": true,
            "zone_analysis": ["zone_1", "zone_2"],
            "export_format": "pdf|json|csv"
        }
    }
    """
    try:
        analytics_engine = AnalyticsEngine()
        
        # Validate report configuration
        required_fields = ["environment_id", "report_type", "time_range"]
        for field in required_fields:
            if field not in report_config:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Validate time range
        time_range = report_config["time_range"]
        if "start" not in time_range or "end" not in time_range:
            raise HTTPException(status_code=400, detail="Invalid time_range: missing start or end")
        
        try:
            start_time = datetime.fromisoformat(time_range["start"].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(time_range["end"].replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid time_range format. Use ISO 8601 format.")
        
        if start_time >= end_time:
            raise HTTPException(status_code=400, detail="Start time must be before end time")
        
        # Validate report type
        valid_report_types = ["comprehensive", "occupancy", "movement", "behavior"]
        if report_config["report_type"] not in valid_report_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid report_type. Must be one of: {', '.join(valid_report_types)}"
            )
        
        # Generate custom report (async job)
        report_job = await analytics_engine.generate_custom_report_async(report_config)
        
        # Create export job response
        export_response = ExportJobResponse(
            job_id=report_job["job_id"],
            status="queued",
            progress=0.0,
            export_type="analytics_report",
            file_format=report_config.get("options", {}).get("export_format", "pdf"),
            estimated_size_mb=report_job.get("estimated_size_mb"),
            estimated_completion=report_job.get("estimated_completion")
        )
        
        logger.info(
            f"Queued custom analytics report job {report_job['job_id']} "
            f"for {report_config['environment_id']}"
        )
        
        return export_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating custom report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating custom report: {str(e)}")


@router.get("/export/jobs/{job_id}/status")
async def get_export_job_status(
    job_id: str = Path(..., description="Export job identifier"),
    current_user: dict = Depends(get_current_user)
) -> ExportJobResponse:
    """
    Get status of an export job.
    """
    try:
        analytics_engine = AnalyticsEngine()
        job_status = await analytics_engine.get_export_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Export job {job_id} not found")
        
        return ExportJobResponse(**job_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching export job status: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching export job status: {str(e)}")


@router.get("/export/jobs/{job_id}/download")
async def download_export_file(
    job_id: str = Path(..., description="Export job identifier"),
    current_user: dict = Depends(get_current_user)
):
    """
    Download completed export file.
    """
    try:
        analytics_engine = AnalyticsEngine()
        
        # Get job status
        job_status = await analytics_engine.get_export_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Export job {job_id} not found")
        
        if job_status["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Export job not completed. Current status: {job_status['status']}"
            )
        
        # Get download response
        download_response = await analytics_engine.get_export_download(job_id)
        return download_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading export file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading export file: {str(e)}")


# --- System Performance and Monitoring ---

@router.get("/system/performance")
async def get_system_performance_metrics(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive system performance metrics.
    
    Includes:
    - Processing performance statistics
    - Memory and resource utilization
    - Analytics computation performance
    - Database query performance
    """
    try:
        viz_service = get_visualization_data_service()
        analytics_engine = AnalyticsEngine()
        
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "visualization_service": viz_service.get_service_status() if viz_service else {},
            "analytics_engine": await analytics_engine.get_performance_metrics(),
            "system_resources": await analytics_engine.get_system_resource_usage()
        }
        
        logger.info("Retrieved system performance metrics")
        return performance_data
        
    except Exception as e:
        logger.error(f"Error fetching system performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching performance metrics: {str(e)}")


# --- Health Check ---

@router.get("/health")
async def analytics_health_check(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Comprehensive health check for analytics services.
    """
    try:
        viz_service = get_visualization_data_service()
        analytics_engine = AnalyticsEngine()
        
        health_status = {
            "analytics_api": "healthy",
            "visualization_service": "healthy" if viz_service else "unavailable",
            "analytics_engine": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add detailed service status if available
        if viz_service:
            health_status["visualization_service_details"] = viz_service.get_service_status()
        
        return {
            "status": "healthy",
            "services": health_status
        }
        
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")