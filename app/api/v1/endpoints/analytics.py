"""
Advanced analytics API endpoints.

Handles:
- Real-time analytics queries
- Behavior analysis requests
- Historical data analysis
- Performance metrics
- Custom reports
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from app.services.analytics_engine import analytics_engine
from app.infrastructure.integration.database_integration_service import database_integration_service
from app.shared.types import CameraID

router = APIRouter()


class AnalyticsQueryRequest(BaseModel):
    """Request model for analytics queries."""
    environment_id: str = Field(..., description="Environment ID")
    start_time: Optional[datetime] = Field(None, description="Start time for analysis")
    end_time: Optional[datetime] = Field(None, description="End time for analysis")
    camera_ids: Optional[List[str]] = Field(None, description="Specific camera IDs to analyze")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional query parameters")


class BehaviorAnalysisRequest(BaseModel):
    """Request model for behavior analysis."""
    person_id: str = Field(..., description="Person ID to analyze")
    time_range_hours: Optional[int] = Field(24, description="Time range in hours to analyze")
    include_predictions: bool = Field(False, description="Include path predictions")
    analysis_depth: str = Field("standard", description="Analysis depth: basic, standard, detailed")


class ReportRequest(BaseModel):
    """Request model for report generation."""
    report_type: str = Field(..., description="Type of report: summary, behavior, performance")
    environment_id: str = Field(..., description="Environment ID")
    time_range_hours: Optional[int] = Field(24, description="Time range in hours")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Report parameters")


class PathPredictionRequest(BaseModel):
    """Request model for path prediction."""
    person_id: str = Field(..., description="Person ID")
    current_position: Dict[str, float] = Field(..., description="Current position (x, y)")
    prediction_horizon: int = Field(30, description="Prediction horizon in seconds")


# Dashboard Endpoint
@router.get("/dashboard")
async def get_analytics_dashboard(
    environment_id: str = Query("default", description="Environment identifier"),
    window_hours: int = Query(24, ge=1, le=168, description="Hours of history to include"),
    uptime_days: int = Query(7, ge=1, le=30, description="Days of uptime history"),
):
    """Return consolidated analytics snapshot for dashboard rendering."""
    from app.utils.mock_analytics_data import get_mock_dashboard_data, merge_with_real_data
    
    # Get real data from database
    real_snapshot = await database_integration_service.get_dashboard_snapshot(
        environment_id=environment_id,
        window_hours=window_hours,
        uptime_history_days=uptime_days,
    )
    
    # Get mock data as base (with proper time distribution)
    mock_snapshot = get_mock_dashboard_data(environment_id, window_hours, uptime_days)
    
    # Merge: real data takes priority over mock
    merged_snapshot = merge_with_real_data(mock_snapshot, real_snapshot)

    return {
        "status": "success",
        "data": merged_snapshot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Real-Time Analytics Endpoints
@router.get("/real-time/metrics")
async def get_real_time_metrics():
    """Get current real-time analytics metrics."""
    try:
        metrics = await analytics_engine.get_real_time_metrics()
        
        return {
            "status": "success",
            "data": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting real-time metrics: {str(e)}")


@router.get("/real-time/active-persons")
async def get_active_persons(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    environment_id: Optional[str] = Query(None, description="Filter by environment ID")
):
    """Get currently active persons with real-time data."""
    try:
        camera_id_typed = CameraID(camera_id) if camera_id else None
        
        active_persons = await database_integration_service.get_active_persons(
            camera_id=camera_id_typed,
            environment_id=environment_id,
            prefer_cache=True
        )
        
        return {
            "status": "success",
            "data": {
                "active_persons": active_persons,
                "total_count": len(active_persons),
                "camera_id": camera_id,
                "environment_id": environment_id
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting active persons: {str(e)}")


@router.get("/real-time/camera-loads")
async def get_camera_loads(environment_id: Optional[str] = Query(None)):
    """Get current load distribution across cameras."""
    try:
        metrics = await analytics_engine.get_real_time_metrics()
        camera_loads = metrics.get('camera_loads', {})
        
        return {
            "status": "success",
            "data": {
                "camera_loads": camera_loads,
                "total_active": sum(camera_loads.values()),
                "environment_id": environment_id
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting camera loads: {str(e)}")


# Behavior Analysis Endpoints
@router.post("/behavior/analyze")
async def analyze_person_behavior(request: BehaviorAnalysisRequest):
    """Analyze person behavior patterns."""
    try:
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=request.time_range_hours)
        
        # Perform behavior analysis
        behavior_profile = await analytics_engine.analyze_person_behavior(
            person_id=request.person_id,
            time_range=(start_time, end_time)
        )
        
        if not behavior_profile:
            raise HTTPException(
                status_code=404,
                detail=f"No behavior data found for person {request.person_id}"
            )
        
        # Convert to dict for response
        profile_data = {
            "person_id": behavior_profile.person_id,
            "total_detections": behavior_profile.total_detections,
            "total_time_tracked": behavior_profile.total_time_tracked,
            "average_speed": behavior_profile.average_speed,
            "dwell_time": behavior_profile.dwell_time,
            "path_complexity": behavior_profile.path_complexity,
            "cameras_visited": behavior_profile.cameras_visited,
            "frequent_areas": behavior_profile.frequent_areas,
            "activity_patterns": behavior_profile.activity_patterns,
            "anomaly_score": behavior_profile.anomaly_score
        }
        
        # Add predictions if requested
        predictions = None
        if request.include_predictions and behavior_profile.cameras_visited:
            # Get current position from latest trajectory
            trajectory_data = await database_integration_service.get_person_trajectory(
                person_id=request.person_id,
                limit=1
            )
            
            if trajectory_data:
                from app.domains.mapping.entities.coordinate import Coordinate
                current_pos = Coordinate(
                    x=trajectory_data[0]['position_x'],
                    y=trajectory_data[0]['position_y']
                )
                
                predictions = await analytics_engine.predict_person_path(
                    person_id=request.person_id,
                    current_position=current_pos
                )
        
        return {
            "status": "success",
            "data": {
                "behavior_profile": profile_data,
                "predictions": predictions,
                "analysis_depth": request.analysis_depth,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": request.time_range_hours
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing behavior: {str(e)}")


@router.get("/behavior/profile/{person_id}")
async def get_behavior_profile(person_id: str):
    """Get cached behavior profile for a person."""
    try:
        profile = await analytics_engine.get_behavior_profile(person_id)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"No behavior profile found for person {person_id}"
            )
        
        profile_data = {
            "person_id": profile.person_id,
            "total_detections": profile.total_detections,
            "total_time_tracked": profile.total_time_tracked,
            "average_speed": profile.average_speed,
            "dwell_time": profile.dwell_time,
            "path_complexity": profile.path_complexity,
            "cameras_visited": profile.cameras_visited,
            "frequent_areas": profile.frequent_areas,
            "activity_patterns": profile.activity_patterns,
            "anomaly_score": profile.anomaly_score
        }
        
        return {
            "status": "success",
            "data": profile_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting behavior profile: {str(e)}")


@router.post("/behavior/anomalies")
async def detect_anomalies(request: AnalyticsQueryRequest):
    """Detect behavioral anomalies in the specified time range."""
    try:
        # Set default time range if not provided
        end_time = request.end_time or datetime.now(timezone.utc)
        start_time = request.start_time or (end_time - timedelta(hours=24))
        
        # Get active persons
        active_persons = await database_integration_service.get_active_persons(
            environment_id=request.environment_id,
            prefer_cache=True
        )
        
        anomalies = []
        
        # Analyze each person for anomalies
        for person in active_persons:
            profile = await analytics_engine.analyze_person_behavior(
                person_id=person['global_id'],
                time_range=(start_time, end_time)
            )
            
            if profile and profile.anomaly_score > 0.7:  # Threshold for anomaly
                anomalies.append({
                    "person_id": profile.person_id,
                    "anomaly_score": profile.anomaly_score,
                    "anomaly_factors": {
                        "path_complexity": profile.path_complexity,
                        "speed_variance": profile.activity_patterns.get('speed_patterns', {}).get('std', 0),
                        "cameras_visited": len(profile.cameras_visited)
                    },
                    "last_seen_camera": person.get('last_seen_camera'),
                    "last_seen_time": person.get('last_seen_time')
                })
        
        return {
            "status": "success",
            "data": {
                "anomalies": anomalies,
                "total_anomalies": len(anomalies),
                "total_persons_analyzed": len(active_persons),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")


# Path Prediction Endpoints
@router.post("/prediction/path")
async def predict_person_path(request: PathPredictionRequest):
    """Predict future path for a person."""
    try:
        from app.domains.mapping.entities.coordinate import Coordinate
        
        current_position = Coordinate(
            x=request.current_position['x'],
            y=request.current_position['y']
        )
        
        predictions = await analytics_engine.predict_person_path(
            person_id=request.person_id,
            current_position=current_position,
            prediction_horizon=request.prediction_horizon
        )
        
        if not predictions:
            raise HTTPException(
                status_code=404,
                detail=f"Cannot generate predictions for person {request.person_id}"
            )
        
        return {
            "status": "success",
            "data": {
                "person_id": request.person_id,
                "current_position": request.current_position,
                "predictions": predictions,
                "prediction_horizon": request.prediction_horizon,
                "confidence_note": "Confidence decreases over time"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting path: {str(e)}")


# Historical Analytics Endpoints
@router.post("/historical/summary")
async def get_historical_summary(request: AnalyticsQueryRequest):
    """Get historical analytics summary."""
    try:
        # Set default time range if not provided
        end_time = request.end_time or datetime.now(timezone.utc)
        start_time = request.start_time or (end_time - timedelta(hours=24))
        
        # Get historical summary from database integration service
        summary = await database_integration_service.get_analytics_summary(
            environment_id=request.environment_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting historical summary: {str(e)}")


@router.get("/historical/trajectory/{person_id}")
async def get_person_trajectory(
    person_id: str,
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    hours: int = Query(24, description="Hours of history to retrieve"),
    limit: int = Query(1000, description="Maximum number of trajectory points")
):
    """Get historical trajectory for a person."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        camera_id_typed = CameraID(camera_id) if camera_id else None
        
        trajectory_data = await database_integration_service.get_person_trajectory(
            person_id=person_id,
            camera_id=camera_id_typed,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return {
            "status": "success",
            "data": {
                "person_id": person_id,
                "camera_id": camera_id,
                "trajectory": trajectory_data,
                "total_points": len(trajectory_data),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trajectory: {str(e)}")


# Reporting Endpoints
@router.post("/reports/generate")
async def generate_analytics_report(request: ReportRequest):
    """Generate comprehensive analytics report."""
    try:
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=request.time_range_hours or 24)
        
        # Generate report
        report = await analytics_engine.generate_analytics_report(
            report_type=request.report_type,
            environment_id=request.environment_id,
            time_range=(start_time, end_time),
            parameters=request.parameters
        )
        
        if not report:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to generate {request.report_type} report"
            )
        
        # Convert report to dict
        report_data = {
            "report_id": report.report_id,
            "report_type": report.report_type,
            "environment_id": report.environment_id,
            "time_range": {
                "start": report.time_range['start'].isoformat(),
                "end": report.time_range['end'].isoformat()
            },
            "metrics": report.metrics,
            "insights": report.insights,
            "recommendations": report.recommendations,
            "generated_at": report.generated_at.isoformat()
        }
        
        return {
            "status": "success",
            "data": report_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.get("/reports/types")
async def get_available_report_types():
    """Get available report types and their descriptions."""
    try:
        report_types = {
            "summary": {
                "description": "Comprehensive summary of tracking performance and statistics",
                "parameters": ["detection_threshold", "confidence_threshold"],
                "time_range": "Configurable (default: 24 hours)"
            },
            "behavior": {
                "description": "Detailed behavior analysis including patterns and anomalies",
                "parameters": ["anomaly_threshold", "clustering_parameters"],
                "time_range": "Configurable (default: 24 hours)"
            },
            "performance": {
                "description": "System performance metrics and optimization recommendations",
                "parameters": ["performance_thresholds"],
                "time_range": "Configurable (default: 24 hours)"
            }
        }
        
        return {
            "status": "success",
            "data": report_types,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting report types: {str(e)}")


# System Analytics Endpoints
@router.get("/system/statistics")
async def get_system_statistics():
    """Get comprehensive system analytics statistics."""
    from app.utils.mock_analytics_data import get_mock_system_statistics, merge_with_real_data
    
    try:
        # Get analytics engine statistics
        analytics_stats = await analytics_engine.get_analytics_statistics()
        
        # Get database integration service health
        service_health = await database_integration_service.get_service_health()
        
        real_data = {
            "analytics_engine": analytics_stats,
            "database_service": service_health,
            "system_uptime": "N/A",  # Would be implemented with system monitoring
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Get mock data as base
        mock_data = get_mock_system_statistics()
        
        # Merge: real data takes priority over mock
        merged_data = merge_with_real_data(mock_data, real_data)
        
        return {
            "status": "success",
            "data": merged_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system statistics: {str(e)}")


@router.post("/system/reset-statistics")
async def reset_system_statistics():
    """Reset all analytics statistics."""
    try:
        # Reset analytics engine statistics
        analytics_engine.reset_statistics()
        
        # Reset database integration service statistics
        database_integration_service.reset_statistics()
        
        return {
            "status": "success",
            "message": "Analytics statistics reset successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting statistics: {str(e)}")


# Health Check Endpoint
@router.get("/health")
async def analytics_health_check():
    """Health check for analytics service."""
    try:
        # Check analytics engine status
        analytics_stats = await analytics_engine.get_analytics_statistics()
        
        # Check real-time metrics availability
        real_time_metrics = await analytics_engine.get_real_time_metrics()
        
        health_status = {
            "analytics_engine": "healthy",
            "real_time_metrics": "healthy" if real_time_metrics else "degraded",
            "behavior_analysis": "healthy",
            "prediction_models": "healthy",
            "statistics": analytics_stats
        }
        
        return {
            "status": "success",
            "data": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics health check failed: {str(e)}")
