"""
Analytics repository interface for application layer.

Abstract interface defining analytics data access operations.
Follows repository pattern with clean separation of concerns.
"""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.time_range import TimeRange


class AnalyticsRepository(ABC):
    """
    Abstract analytics repository interface.
    
    Defines the contract for analytics data persistence and retrieval
    without coupling to specific database implementations.
    """
    
    @abstractmethod
    async def save_analytics_session(
        self,
        session_id: str,
        session_data: Dict[str, Any]
    ) -> bool:
        """
        Save analytics session data.
        
        Args:
            session_id: Session identifier
            session_data: Session data to save
            
        Returns:
            True if save successful
        """
        pass
    
    @abstractmethod
    async def get_analytics_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get analytics session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        pass
    
    @abstractmethod
    async def save_behavioral_metrics(
        self,
        camera_id: CameraID,
        timestamp: datetime,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Save behavioral metrics data.
        
        Args:
            camera_id: Camera identifier
            timestamp: Metrics timestamp
            metrics: Behavioral metrics data
            
        Returns:
            True if save successful
        """
        pass
    
    @abstractmethod
    async def get_behavioral_metrics(
        self,
        camera_id: CameraID,
        time_range: TimeRange,
        metric_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get behavioral metrics for analysis.
        
        Args:
            camera_id: Camera identifier
            time_range: Time range for metrics
            metric_types: Optional filter for specific metric types
            
        Returns:
            List of behavioral metrics
        """
        pass
    
    @abstractmethod
    async def save_crowd_analytics(
        self,
        camera_id: CameraID,
        timestamp: datetime,
        crowd_data: Dict[str, Any]
    ) -> bool:
        """
        Save crowd analytics data.
        
        Args:
            camera_id: Camera identifier
            timestamp: Analytics timestamp
            crowd_data: Crowd analytics data
            
        Returns:
            True if save successful
        """
        pass
    
    @abstractmethod
    async def get_crowd_analytics(
        self,
        camera_id: CameraID,
        time_range: TimeRange,
        aggregation_interval: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get crowd analytics data.
        
        Args:
            camera_id: Camera identifier
            time_range: Time range for analytics
            aggregation_interval: Optional aggregation interval (hourly, daily, etc.)
            
        Returns:
            List of crowd analytics data
        """
        pass
    
    @abstractmethod
    async def save_anomaly_detection(
        self,
        camera_id: CameraID,
        timestamp: datetime,
        anomaly_data: Dict[str, Any]
    ) -> bool:
        """
        Save anomaly detection results.
        
        Args:
            camera_id: Camera identifier
            timestamp: Anomaly timestamp
            anomaly_data: Anomaly detection data
            
        Returns:
            True if save successful
        """
        pass
    
    @abstractmethod
    async def get_anomalies(
        self,
        camera_ids: Optional[List[CameraID]] = None,
        time_range: Optional[TimeRange] = None,
        anomaly_types: Optional[List[str]] = None,
        severity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get anomaly detection results.
        
        Args:
            camera_ids: Optional camera filters
            time_range: Optional time range filter
            anomaly_types: Optional anomaly type filters
            severity_threshold: Optional severity threshold
            
        Returns:
            List of anomaly detection results
        """
        pass
    
    @abstractmethod
    async def save_performance_metrics(
        self,
        component: str,
        metrics: Dict[str, Union[int, float, str]],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Save system performance metrics.
        
        Args:
            component: Component name (detection, tracking, analytics)
            metrics: Performance metrics data
            timestamp: Optional custom timestamp
            
        Returns:
            True if save successful
        """
        pass
    
    @abstractmethod
    async def get_performance_metrics(
        self,
        component: Optional[str] = None,
        time_range: Optional[TimeRange] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get system performance metrics.
        
        Args:
            component: Optional component filter
            time_range: Optional time range filter
            metric_names: Optional metric name filters
            
        Returns:
            List of performance metrics
        """
        pass
    
    @abstractmethod
    async def get_analytics_summary(
        self,
        camera_ids: Optional[List[CameraID]] = None,
        time_range: Optional[TimeRange] = None,
        summary_type: str = "daily"
    ) -> Dict[str, Any]:
        """
        Get analytics summary data.
        
        Args:
            camera_ids: Optional camera filters
            time_range: Optional time range filter
            summary_type: Summary aggregation type
            
        Returns:
            Analytics summary data
        """
        pass
    
    @abstractmethod
    async def delete_old_analytics(
        self,
        older_than: datetime,
        data_types: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        Delete old analytics data.
        
        Args:
            older_than: Delete data before this timestamp
            data_types: Optional data type filters
            batch_size: Number of records to delete per batch
            
        Returns:
            Dictionary with deletion counts per data type
        """
        pass
    
    @abstractmethod
    async def get_available_date_ranges(
        self,
        camera_id: Optional[CameraID] = None,
        data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get available data date ranges.
        
        Args:
            camera_id: Optional camera filter
            data_type: Optional data type filter
            
        Returns:
            Dictionary with earliest and latest available dates
        """
        pass