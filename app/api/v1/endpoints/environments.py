"""
Environment Management API Endpoints

Comprehensive API endpoints for environment management providing:
- Environment listing and selection for landing page
- Available date ranges for each environment
- Environment metadata and configuration serving
- User preference storage for environment settings
- Multi-environment data access and management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field

from app.services.environment_configuration_service import (
    get_environment_configuration_service,
    EnvironmentConfigurationService,
    EnvironmentType
)
from app.services.historical_data_service import (
    get_historical_data_service,
    HistoricalDataService,
    TimeRange,
    HistoricalQueryFilter
)
from app.infrastructure.cache.tracking_cache import get_tracking_cache

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/environments", tags=["environments"])


# --- Request/Response Models ---

class EnvironmentListItem(BaseModel):
    """Environment list item for landing page."""
    environment_id: str
    name: str
    environment_type: str
    description: str
    is_active: bool
    camera_count: int
    zone_count: int
    has_data: bool
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class DateRangeInfo(BaseModel):
    """Date range information for environment."""
    earliest_date: Optional[datetime]
    latest_date: Optional[datetime]
    total_days: int
    has_data: bool
    data_gaps: List[Dict[str, str]] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat() if dt else None
        }


class EnvironmentMetadata(BaseModel):
    """Complete environment metadata."""
    environment_id: str
    name: str
    environment_type: str
    description: str
    is_active: bool
    cameras: Dict[str, Any]
    zones: Dict[str, Any]
    data_availability: DateRangeInfo
    settings: Dict[str, Any]
    validation_status: Dict[str, Any]
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class UserEnvironmentPreferences(BaseModel):
    """User preferences for environment settings."""
    user_id: Optional[str] = None
    preferred_environment: Optional[str] = None
    default_date_range_hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week
    favorite_environments: List[str] = Field(default_factory=list)
    view_settings: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    """Request to create analysis session."""
    environment_id: str
    start_time: datetime
    end_time: datetime
    session_name: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class EnvironmentSessionResponse(BaseModel):
    """Response for environment session creation."""
    session_id: str
    environment_id: str
    time_range: Dict[str, str]
    status: str
    data_points_available: int
    estimated_processing_time_ms: float
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# --- Dependency Functions ---

def get_environment_service() -> EnvironmentConfigurationService:
    """Get environment configuration service dependency."""
    service = get_environment_configuration_service()
    if not service:
        raise HTTPException(status_code=503, detail="Environment configuration service not available")
    return service


def get_historical_service() -> HistoricalDataService:
    """Get historical data service dependency."""
    service = get_historical_data_service()
    if not service:
        raise HTTPException(status_code=503, detail="Historical data service not available")
    return service


# --- Environment Listing & Selection ---

@router.get("/", response_model=List[EnvironmentListItem])
async def list_environments(
    environment_type: Optional[str] = Query(None, description="Filter by environment type"),
    active_only: bool = Query(True, description="Show only active environments"),
    include_data_check: bool = Query(True, description="Check data availability"),
    env_service: EnvironmentConfigurationService = Depends(get_environment_service),
    historical_service: HistoricalDataService = Depends(get_historical_service)
):
    """
    Get list of all available environments for landing page selection.
    
    This endpoint provides the primary data needed for the frontend landing page
    environment selection interface.
    """
    try:
        # Parse environment type
        env_type_filter = None
        if environment_type:
            try:
                env_type_filter = EnvironmentType(environment_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid environment type: {environment_type}")
        
        # Get environments from service
        environments = await env_service.list_environments(env_type_filter, active_only)
        
        # Build response list
        environment_list = []
        
        for env in environments:
            # Check data availability if requested
            has_data = False
            if include_data_check:
                try:
                    date_ranges = await env_service.get_available_date_ranges(env.environment_id)
                    env_range = date_ranges.get(env.environment_id, {})
                    has_data = env_range.get('has_data', False)
                except Exception as e:
                    logger.warning(f"Could not check data availability for {env.environment_id}: {e}")
                    has_data = False
            
            environment_item = EnvironmentListItem(
                environment_id=env.environment_id,
                name=env.name,
                environment_type=env.environment_type.value,
                description=env.description,
                is_active=env.is_active,
                camera_count=len(env.cameras),
                zone_count=len(env.zones),
                has_data=has_data,
                last_updated=env.last_updated
            )
            
            environment_list.append(environment_item)
        
        # Sort by name for consistent ordering
        environment_list.sort(key=lambda x: x.name)
        
        logger.info(f"Retrieved {len(environment_list)} environments (type: {environment_type}, active: {active_only})")
        return environment_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing environments: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve environments")


@router.get("/{environment_id}", response_model=EnvironmentMetadata)
async def get_environment_details(
    environment_id: str = Path(..., description="Environment ID"),
    include_validation: bool = Query(True, description="Include configuration validation"),
    env_service: EnvironmentConfigurationService = Depends(get_environment_service)
):
    """
    Get detailed information about a specific environment.
    
    Provides comprehensive metadata needed for environment-specific operations
    and configuration display.
    """
    try:
        # Get environment metadata
        metadata = await env_service.get_environment_metadata(environment_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Environment {environment_id} not found")
        
        # Convert to response format
        date_range_data = metadata.get('data_availability', {})
        date_range = DateRangeInfo(
            earliest_date=datetime.fromisoformat(date_range_data['earliest_date']) if date_range_data.get('earliest_date') else None,
            latest_date=datetime.fromisoformat(date_range_data['latest_date']) if date_range_data.get('latest_date') else None,
            total_days=date_range_data.get('total_days', 0),
            has_data=date_range_data.get('has_data', False),
            data_gaps=date_range_data.get('data_gaps', [])
        )
        
        validation_status = metadata.get('validation', {}) if include_validation else {}
        
        response = EnvironmentMetadata(
            environment_id=metadata['environment_id'],
            name=metadata['name'],
            environment_type=metadata['type'],
            description=metadata['description'],
            is_active=metadata['is_active'],
            cameras=metadata['cameras'],
            zones=metadata['zones'],
            data_availability=date_range,
            settings=metadata['settings'],
            validation_status=validation_status,
            last_updated=datetime.fromisoformat(metadata['last_updated'])
        )
        
        logger.info(f"Retrieved metadata for environment {environment_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environment details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve environment details")


@router.get("/{environment_id}/date-ranges", response_model=DateRangeInfo)
async def get_environment_date_ranges(
    environment_id: str = Path(..., description="Environment ID"),
    detailed: bool = Query(False, description="Include detailed gap analysis"),
    env_service: EnvironmentConfigurationService = Depends(get_environment_service),
    historical_service: HistoricalDataService = Depends(get_historical_service)
):
    """
    Get available data date ranges for a specific environment.
    
    Essential for frontend date/time selection components to show
    valid date ranges and data availability.
    """
    try:
        # Verify environment exists
        environment = await env_service.get_environment(environment_id)
        if not environment:
            raise HTTPException(status_code=404, detail=f"Environment {environment_id} not found")
        
        # Get date ranges from service
        date_ranges = await env_service.get_available_date_ranges(environment_id)
        env_range = date_ranges.get(environment_id, {})
        
        if not env_range:
            # Return empty range
            return DateRangeInfo(
                earliest_date=None,
                latest_date=None,
                total_days=0,
                has_data=False,
                data_gaps=[]
            )
        
        # Parse dates
        earliest_date = None
        latest_date = None
        
        if env_range.get('earliest_date'):
            earliest_date = datetime.fromisoformat(env_range['earliest_date'])
        if env_range.get('latest_date'):
            latest_date = datetime.fromisoformat(env_range['latest_date'])
        
        # Calculate data gaps if detailed analysis requested
        data_gaps = []
        if detailed and earliest_date and latest_date:
            try:
                # Query for data gaps (simplified implementation)
                gaps = await _analyze_data_gaps(
                    historical_service, environment_id, earliest_date, latest_date
                )
                data_gaps = gaps
            except Exception as e:
                logger.warning(f"Could not analyze data gaps for {environment_id}: {e}")
        
        response = DateRangeInfo(
            earliest_date=earliest_date,
            latest_date=latest_date,
            total_days=env_range.get('total_days', 0),
            has_data=env_range.get('has_data', False),
            data_gaps=data_gaps
        )
        
        logger.info(f"Retrieved date ranges for environment {environment_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environment date ranges: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve date ranges")


@router.get("/{environment_id}/cameras")
async def get_environment_cameras(
    environment_id: str = Path(..., description="Environment ID"),
    active_only: bool = Query(True, description="Show only active cameras"),
    env_service: EnvironmentConfigurationService = Depends(get_environment_service)
):
    """
    Get list of cameras available in the environment.
    
    Used for camera-specific operations and selection interfaces.
    """
    try:
        environment = await env_service.get_environment(environment_id)
        if not environment:
            raise HTTPException(status_code=404, detail=f"Environment {environment_id} not found")
        
        # Get cameras based on filter
        if active_only:
            cameras = environment.get_active_cameras()
        else:
            cameras = list(environment.cameras.values())
        
        # Convert to response format
        camera_list = []
        for camera in cameras:
            camera_dict = camera.to_dict()
            camera_list.append(camera_dict)
        
        logger.info(f"Retrieved {len(camera_list)} cameras for environment {environment_id}")
        return {
            "environment_id": environment_id,
            "cameras": camera_list,
            "total_cameras": len(camera_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environment cameras: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cameras")


@router.get("/{environment_id}/zones")
async def get_environment_zones(
    environment_id: str = Path(..., description="Environment ID"),
    zone_type: Optional[str] = Query(None, description="Filter by zone type"),
    env_service: EnvironmentConfigurationService = Depends(get_environment_service)
):
    """
    Get zone definitions for the environment.
    
    Used for zone-based analytics and spatial analysis configuration.
    """
    try:
        environment = await env_service.get_environment(environment_id)
        if not environment:
            raise HTTPException(status_code=404, detail=f"Environment {environment_id} not found")
        
        # Get zones based on filter
        zones = list(environment.zones.values())
        
        if zone_type:
            try:
                from app.services.environment_configuration_service import ZoneType
                zone_type_enum = ZoneType(zone_type.lower())
                zones = environment.get_zones_by_type(zone_type_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid zone type: {zone_type}")
        
        # Convert to response format
        zone_list = []
        for zone in zones:
            zone_dict = zone.to_dict()
            zone_list.append(zone_dict)
        
        logger.info(f"Retrieved {len(zone_list)} zones for environment {environment_id}")
        return {
            "environment_id": environment_id,
            "zones": zone_list,
            "total_zones": len(zone_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environment zones: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve zones")


# --- Session Management ---

@router.post("/{environment_id}/sessions", response_model=EnvironmentSessionResponse)
async def create_analysis_session(
    session_request: CreateSessionRequest,
    environment_id: str = Path(..., description="Environment ID"),
    env_service: EnvironmentConfigurationService = Depends(get_environment_service),
    historical_service: HistoricalDataService = Depends(get_historical_service)
):
    """
    Create a new analysis session for the environment.
    
    Sets up a session for historical data analysis with specified time range
    and validates data availability.
    """
    try:
        # Verify environment exists
        environment = await env_service.get_environment(environment_id)
        if not environment:
            raise HTTPException(status_code=404, detail=f"Environment {environment_id} not found")
        
        # Validate time range
        if session_request.start_time >= session_request.end_time:
            raise HTTPException(status_code=400, detail="Start time must be before end time")
        
        # Check if time range is within available data
        date_ranges = await env_service.get_available_date_ranges(environment_id)
        env_range = date_ranges.get(environment_id, {})
        
        if not env_range.get('has_data', False):
            raise HTTPException(status_code=400, detail="No data available for this environment")
        
        earliest = datetime.fromisoformat(env_range['earliest_date']) if env_range.get('earliest_date') else None
        latest = datetime.fromisoformat(env_range['latest_date']) if env_range.get('latest_date') else None
        
        if earliest and session_request.start_time < earliest:
            raise HTTPException(
                status_code=400, 
                detail=f"Start time is before available data (earliest: {earliest.isoformat()})"
            )
        
        if latest and session_request.end_time > latest:
            raise HTTPException(
                status_code=400, 
                detail=f"End time is after available data (latest: {latest.isoformat()})"
            )
        
        # Create session
        import uuid
        session_id = str(uuid.uuid4())
        
        # Estimate data points and processing time
        time_range = TimeRange(
            start_time=session_request.start_time,
            end_time=session_request.end_time
        )
        
        query_filter = HistoricalQueryFilter(
            time_range=time_range,
            environment_id=environment_id
        )
        
        # Get count estimate (simplified - would use count query in production)
        try:
            sample_data = await historical_service.query_historical_data(query_filter, limit=10)
            estimated_points = len(sample_data) * 10  # Rough estimate
        except Exception:
            estimated_points = 1000  # Default estimate
        
        estimated_processing_time = max(100.0, estimated_points * 0.1)  # Rough estimate
        
        # Cache session information
        cache = get_tracking_cache()
        if cache:
            session_data = {
                'session_id': session_id,
                'environment_id': environment_id,
                'start_time': session_request.start_time.isoformat(),
                'end_time': session_request.end_time.isoformat(),
                'name': session_request.session_name,
                'description': session_request.description,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'created'
            }
            
            await cache.set_json(f"session_{session_id}", session_data, ttl=3600)  # 1 hour TTL
        
        response = EnvironmentSessionResponse(
            session_id=session_id,
            environment_id=environment_id,
            time_range={
                'start': session_request.start_time.isoformat(),
                'end': session_request.end_time.isoformat()
            },
            status='created',
            data_points_available=estimated_points,
            estimated_processing_time_ms=estimated_processing_time
        )
        
        logger.info(f"Created analysis session {session_id} for environment {environment_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating analysis session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create analysis session")


@router.get("/{environment_id}/sessions/{session_id}")
async def get_session_status(
    environment_id: str = Path(..., description="Environment ID"),
    session_id: str = Path(..., description="Session ID")
):
    """
    Get status of an analysis session.
    
    Provides current status and progress information for session monitoring.
    """
    try:
        cache = get_tracking_cache()
        if not cache:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        # Get session data from cache
        session_data = await cache.get_json(f"session_{session_id}")
        
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if session_data.get('environment_id') != environment_id:
            raise HTTPException(status_code=404, detail="Session not found in this environment")
        
        logger.info(f"Retrieved session status for {session_id}")
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session status")


# --- User Preferences ---

@router.get("/preferences/{user_id}", response_model=UserEnvironmentPreferences)
async def get_user_preferences(
    user_id: str = Path(..., description="User ID")
):
    """
    Get user environment preferences.
    
    Retrieves saved user preferences for environment selection and settings.
    """
    try:
        cache = get_tracking_cache()
        if not cache:
            # Return default preferences if cache not available
            return UserEnvironmentPreferences(user_id=user_id)
        
        # Get preferences from cache
        preferences_data = await cache.get_json(f"user_preferences_{user_id}")
        
        if not preferences_data:
            return UserEnvironmentPreferences(user_id=user_id)
        
        preferences = UserEnvironmentPreferences(**preferences_data)
        logger.info(f"Retrieved preferences for user {user_id}")
        return preferences
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        return UserEnvironmentPreferences(user_id=user_id)


@router.put("/preferences/{user_id}", response_model=UserEnvironmentPreferences)
async def update_user_preferences(
    preferences: UserEnvironmentPreferences,
    user_id: str = Path(..., description="User ID")
):
    """
    Update user environment preferences.
    
    Saves user preferences for environment selection and settings.
    """
    try:
        preferences.user_id = user_id  # Ensure user_id matches
        
        cache = get_tracking_cache()
        if cache:
            # Save to cache
            await cache.set_json(
                f"user_preferences_{user_id}", 
                preferences.dict(), 
                ttl=86400 * 30  # 30 days TTL
            )
        
        logger.info(f"Updated preferences for user {user_id}")
        return preferences
        
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user preferences")


# --- Utility Functions ---

async def _analyze_data_gaps(
    historical_service: HistoricalDataService,
    environment_id: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, str]]:
    """
    Analyze data gaps in the specified time range.
    
    This is a simplified implementation. In production, this would
    perform more sophisticated gap analysis.
    """
    try:
        # Sample implementation - would use proper gap analysis in production
        gaps = []
        
        # Check for major gaps (>6 hours without data)
        current_date = start_date
        gap_threshold = timedelta(hours=6)
        
        while current_date < end_date:
            next_date = min(current_date + timedelta(days=1), end_date)
            
            # Query for data in this day
            time_range = TimeRange(start_time=current_date, end_time=next_date)
            query_filter = HistoricalQueryFilter(
                time_range=time_range,
                environment_id=environment_id
            )
            
            try:
                data_points = await historical_service.query_historical_data(query_filter, limit=1)
                if not data_points:
                    gaps.append({
                        'start': current_date.isoformat(),
                        'end': next_date.isoformat(),
                        'type': 'no_data'
                    })
            except Exception:
                # Assume gap if query fails
                gaps.append({
                    'start': current_date.isoformat(),
                    'end': next_date.isoformat(),
                    'type': 'unknown'
                })
            
            current_date = next_date
        
        return gaps[:10]  # Limit to 10 gaps for performance
        
    except Exception as e:
        logger.error(f"Error analyzing data gaps: {e}")
        return []


# --- Health Check ---

@router.get("/health")
async def environment_api_health():
    """Health check for environment API."""
    try:
        env_service = get_environment_configuration_service()
        service_status = env_service.get_service_status() if env_service else None
        
        return {
            "status": "healthy" if env_service else "degraded",
            "service_available": env_service is not None,
            "service_status": service_status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }