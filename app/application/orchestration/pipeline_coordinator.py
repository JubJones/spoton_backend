"""
Pipeline coordinator for application layer orchestration.

Modern replacement for the monolithic pipeline orchestrator using clean architecture.
Coordinates detection, tracking, and analytics use cases with proper separation of concerns.
Maximum 400 lines per plan.
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum

from app.domain.shared.value_objects.camera_id import CameraID
from app.domain.shared.value_objects.frame_id import FrameID
from app.application.use_cases.detection_use_cases import DetectionUseCase, DetectionRequest
from app.application.use_cases.tracking_use_cases import SingleCameraTrackingUseCase, TrackingRequest
from app.application.use_cases.analytics_use_cases import BehavioralAnalyticsUseCase, AnalyticsRequest
from app.application.services.configuration_manager import ConfigurationManager
# from app.infrastructure.services.notification_service import NotificationService

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INITIALIZATION = "initialization"
    DETECTION = "detection"
    TRACKING = "tracking"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"
    COMPLETION = "completion"


@dataclass
class PipelineTask:
    """Pipeline processing task."""
    task_id: str
    camera_id: CameraID
    frame_id: FrameID
    frame_data: bytes
    timestamp: datetime
    processing_stages: List[PipelineStage]
    
    # Task state
    current_stage: PipelineStage
    completed_stages: List[PipelineStage]
    processing_results: Dict[str, Any]
    
    # Performance tracking
    stage_start_time: Optional[datetime] = None
    total_processing_time_ms: float = 0.0


@dataclass
class PipelineConfiguration:
    """Pipeline configuration parameters."""
    enabled_cameras: List[CameraID]
    detection_confidence_threshold: float = 0.5
    tracking_enabled: bool = True
    analytics_enabled: bool = True
    notifications_enabled: bool = True
    
    # Performance settings
    max_concurrent_tasks: int = 10
    batch_processing_size: int = 5
    stage_timeout_seconds: int = 30


class PipelineCoordinator:
    """
    Pipeline coordinator for application layer orchestration.
    
    Coordinates the complete processing pipeline using clean architecture
    with proper separation between detection, tracking, and analytics use cases.
    """
    
    def __init__(
        self,
        detection_use_case: DetectionUseCase,
        tracking_use_case: SingleCameraTrackingUseCase,
        analytics_use_case: BehavioralAnalyticsUseCase,
        configuration_manager: ConfigurationManager,
        notification_service  # NotificationService
    ):
        """
        Initialize pipeline coordinator.
        
        Args:
            detection_use_case: Detection business logic
            tracking_use_case: Tracking business logic
            analytics_use_case: Analytics business logic
            configuration_manager: Configuration management service
            notification_service: Notification service
        """
        self.detection_use_case = detection_use_case
        self.tracking_use_case = tracking_use_case
        self.analytics_use_case = analytics_use_case
        self.config_manager = configuration_manager
        self.notification_service = notification_service
        
        # Pipeline state
        self.active_tasks: Dict[str, PipelineTask] = {}
        self.task_futures: Dict[str, asyncio.Task] = {}
        self.pipeline_config = PipelineConfiguration()
        
        # Performance metrics
        self.pipeline_stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'avg_processing_time_ms': 0.0,
            'total_detections': 0,
            'total_tracks_updated': 0,
            'total_analytics_performed': 0
        }
        
        # Processing control
        self._running = False
        self._semaphore = asyncio.Semaphore(self.pipeline_config.max_concurrent_tasks)
        
        logger.debug("PipelineCoordinator initialized")
    
    async def start_pipeline(self, configuration: Optional[PipelineConfiguration] = None) -> None:
        """
        Start the pipeline coordinator.
        
        Args:
            configuration: Optional pipeline configuration
        """
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        if configuration:
            self.pipeline_config = configuration
        
        self._running = True
        
        logger.info("Pipeline coordinator started")
    
    async def stop_pipeline(self) -> None:
        """Stop the pipeline coordinator."""
        self._running = False
        
        # Cancel all active tasks
        for task_id, future in self.task_futures.items():
            if not future.done():
                future.cancel()
        
        # Wait for tasks to complete
        if self.task_futures:
            await asyncio.gather(*self.task_futures.values(), return_exceptions=True)
        
        self.task_futures.clear()
        self.active_tasks.clear()
        
        logger.info("Pipeline coordinator stopped")
    
    async def process_frame(
        self,
        camera_id: CameraID,
        frame_id: FrameID,
        frame_data: bytes,
        timestamp: Optional[datetime] = None,
        custom_stages: Optional[List[PipelineStage]] = None
    ) -> str:
        """
        Process a single frame through the pipeline.
        
        Args:
            camera_id: Camera identifier
            frame_id: Frame identifier
            frame_data: Frame image data
            timestamp: Frame timestamp (defaults to current time)
            custom_stages: Optional custom processing stages
            
        Returns:
            Task ID for tracking processing progress
        """
        if not self._running:
            raise RuntimeError("Pipeline not running")
        
        # Generate task ID
        task_id = f"task_{camera_id}_{frame_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine processing stages
        stages = custom_stages or self._determine_processing_stages()
        
        # Create pipeline task
        task = PipelineTask(
            task_id=task_id,
            camera_id=camera_id,
            frame_id=frame_id,
            frame_data=frame_data,
            timestamp=timestamp or datetime.utcnow(),
            processing_stages=stages,
            current_stage=stages[0],
            completed_stages=[],
            processing_results={}
        )
        
        # Store task
        self.active_tasks[task_id] = task
        
        # Start processing
        future = asyncio.create_task(self._process_pipeline_task(task))
        self.task_futures[task_id] = future
        
        logger.debug(f"Started pipeline processing for task {task_id}")
        return task_id
    
    async def process_batch_frames(
        self,
        frame_batch: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Process multiple frames in batch.
        
        Args:
            frame_batch: List of frame data dictionaries
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        # Process frames in parallel batches
        batch_size = self.pipeline_config.batch_processing_size
        
        for i in range(0, len(frame_batch), batch_size):
            batch = frame_batch[i:i + batch_size]
            batch_tasks = []
            
            for frame_data in batch:
                task_id = await self.process_frame(
                    camera_id=CameraID(frame_data['camera_id']),
                    frame_id=FrameID(frame_data['frame_id']),
                    frame_data=frame_data['frame_data'],
                    timestamp=frame_data.get('timestamp')
                )
                batch_tasks.append(task_id)
            
            task_ids.extend(batch_tasks)
            
            # Small delay between batches to prevent overwhelming
            await asyncio.sleep(0.01)
        
        logger.info(f"Started batch processing for {len(task_ids)} frames")
        return task_ids
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing status for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information or None if not found
        """
        task = self.active_tasks.get(task_id)
        if not task:
            return None
        
        return {
            'task_id': task_id,
            'camera_id': str(task.camera_id),
            'frame_id': str(task.frame_id),
            'current_stage': task.current_stage.value,
            'completed_stages': [stage.value for stage in task.completed_stages],
            'total_stages': len(task.processing_stages),
            'progress_percentage': len(task.completed_stages) / len(task.processing_stages) * 100,
            'processing_time_ms': task.total_processing_time_ms,
            'results': task.processing_results
        }
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        active_tasks = len(self.active_tasks)
        
        # Combine statistics from use cases
        detection_stats = self.detection_use_case.get_detection_statistics()
        tracking_stats = self.tracking_use_case.get_tracking_statistics()
        analytics_stats = self.analytics_use_case.get_analytics_statistics()
        
        return {
            'pipeline': {
                **self.pipeline_stats,
                'active_tasks': active_tasks,
                'max_concurrent_tasks': self.pipeline_config.max_concurrent_tasks,
                'is_running': self._running
            },
            'detection': detection_stats,
            'tracking': tracking_stats,
            'analytics': analytics_stats
        }
    
    async def _process_pipeline_task(self, task: PipelineTask) -> None:
        """
        Process a pipeline task through all stages.
        
        Args:
            task: Pipeline task to process
        """
        async with self._semaphore:  # Limit concurrent tasks
            try:
                task_start_time = datetime.utcnow()
                
                for stage in task.processing_stages:
                    task.current_stage = stage
                    task.stage_start_time = datetime.utcnow()
                    
                    # Process stage
                    await self._process_stage(task, stage)
                    
                    # Mark stage as completed
                    task.completed_stages.append(stage)
                    
                    # Update total processing time
                    stage_time = (datetime.utcnow() - task.stage_start_time).total_seconds() * 1000
                    task.total_processing_time_ms += stage_time
                
                # Task completed successfully
                total_time = (datetime.utcnow() - task_start_time).total_seconds() * 1000
                task.total_processing_time_ms = total_time
                
                # Update statistics
                self._update_pipeline_statistics(task, success=True)
                
                logger.debug(f"Pipeline task {task.task_id} completed in {total_time:.2f}ms")
                
            except Exception as e:
                logger.error(f"Pipeline task {task.task_id} failed: {e}")
                
                # Update statistics
                self._update_pipeline_statistics(task, success=False)
                
                # Store error in results
                task.processing_results['error'] = str(e)
                
            finally:
                # Clean up task
                self._cleanup_task(task.task_id)
    
    async def _process_stage(self, task: PipelineTask, stage: PipelineStage) -> None:
        """
        Process a specific pipeline stage.
        
        Args:
            task: Pipeline task
            stage: Current stage to process
        """
        if stage == PipelineStage.DETECTION:
            await self._process_detection_stage(task)
        elif stage == PipelineStage.TRACKING:
            await self._process_tracking_stage(task)
        elif stage == PipelineStage.ANALYTICS:
            await self._process_analytics_stage(task)
        elif stage == PipelineStage.NOTIFICATION:
            await self._process_notification_stage(task)
        # INITIALIZATION and COMPLETION are handled implicitly
    
    async def _process_detection_stage(self, task: PipelineTask) -> None:
        """Process detection stage."""
        detection_request = DetectionRequest(
            camera_id=task.camera_id,
            frame_id=task.frame_id,
            frame_data=task.frame_data,
            timestamp=task.timestamp,
            confidence_threshold=self.pipeline_config.detection_confidence_threshold
        )
        
        detection_result = await self.detection_use_case.process_frame_for_detection(
            detection_request
        )
        
        task.processing_results['detection'] = {
            'detections_count': len(detection_result.detections),
            'processing_time_ms': detection_result.processing_time_ms,
            'detections': [
                {
                    'bbox': {
                        'x': d.bbox.x, 'y': d.bbox.y,
                        'width': d.bbox.width, 'height': d.bbox.height
                    },
                    'confidence': d.confidence.value,
                    'class': d.detection_class.value
                }
                for d in detection_result.detections
            ]
        }
        
        # Store detections for next stages
        task.processing_results['_detections'] = detection_result.detections
        
        self.pipeline_stats['total_detections'] += len(detection_result.detections)
    
    async def _process_tracking_stage(self, task: PipelineTask) -> None:
        """Process tracking stage."""
        detections = task.processing_results.get('_detections', [])
        
        if not detections:
            task.processing_results['tracking'] = {'message': 'No detections to track'}
            return
        
        tracking_request = TrackingRequest(
            camera_id=task.camera_id,
            frame_id=task.frame_id,
            detections=detections,
            timestamp=task.timestamp
        )
        
        tracking_result = await self.tracking_use_case.update_tracks_with_detections(
            tracking_request
        )
        
        task.processing_results['tracking'] = {
            'active_tracks': len(tracking_result.active_tracks),
            'new_tracks': len(tracking_result.new_tracks),
            'ended_tracks': len(tracking_result.ended_tracks),
            'processing_time_ms': tracking_result.processing_time_ms
        }
        
        # Store tracks for analytics
        task.processing_results['_tracks'] = tracking_result.active_tracks
        
        self.pipeline_stats['total_tracks_updated'] += 1
    
    async def _process_analytics_stage(self, task: PipelineTask) -> None:
        """Process analytics stage."""
        tracks = task.processing_results.get('_tracks', [])
        
        if not tracks:
            task.processing_results['analytics'] = {'message': 'No tracks for analytics'}
            return
        
        # Simple analytics for individual frame
        task.processing_results['analytics'] = {
            'people_count': len(tracks),
            'timestamp': task.timestamp.isoformat(),
            'analytics_performed': True
        }
        
        self.pipeline_stats['total_analytics_performed'] += 1
    
    async def _process_notification_stage(self, task: PipelineTask) -> None:
        """Process notification stage."""
        # Send notifications if configured
        if self.pipeline_config.notifications_enabled:
            await self.notification_service.send_processing_update(
                task.task_id,
                task.processing_results
            )
        
        task.processing_results['notification'] = {'sent': self.pipeline_config.notifications_enabled}
    
    def _determine_processing_stages(self) -> List[PipelineStage]:
        """Determine processing stages based on configuration."""
        stages = [PipelineStage.INITIALIZATION, PipelineStage.DETECTION]
        
        if self.pipeline_config.tracking_enabled:
            stages.append(PipelineStage.TRACKING)
        
        if self.pipeline_config.analytics_enabled:
            stages.append(PipelineStage.ANALYTICS)
        
        if self.pipeline_config.notifications_enabled:
            stages.append(PipelineStage.NOTIFICATION)
        
        stages.append(PipelineStage.COMPLETION)
        return stages
    
    def _update_pipeline_statistics(self, task: PipelineTask, success: bool) -> None:
        """Update pipeline statistics."""
        self.pipeline_stats['tasks_processed'] += 1
        
        if success:
            self.pipeline_stats['tasks_successful'] += 1
        else:
            self.pipeline_stats['tasks_failed'] += 1
        
        # Update average processing time
        if task.total_processing_time_ms > 0:
            current_avg = self.pipeline_stats['avg_processing_time_ms']
            task_count = self.pipeline_stats['tasks_processed']
            
            self.pipeline_stats['avg_processing_time_ms'] = (
                (current_avg * (task_count - 1) + task.total_processing_time_ms) / task_count
            )
    
    def _cleanup_task(self, task_id: str) -> None:
        """Clean up completed task."""
        self.active_tasks.pop(task_id, None)
        self.task_futures.pop(task_id, None)