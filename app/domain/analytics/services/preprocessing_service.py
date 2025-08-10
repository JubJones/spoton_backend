"""
Preprocessing domain service for historical data preparation.

Focused service for data preprocessing, temporal indexing, and analytics preparation.
Maximum 300 lines per plan.
"""
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum

from app.domain.shared.value_objects.time_range import TimeRange

logger = logging.getLogger(__name__)


class PreprocessingStage(Enum):
    """Stages of data preprocessing."""
    DATA_VALIDATION = "data_validation"
    TEMPORAL_INDEXING = "temporal_indexing"
    FEATURE_EXTRACTION = "feature_extraction"
    TRAJECTORY_BUILDING = "trajectory_building"
    ANALYTICS_PREPARATION = "analytics_preparation"
    CACHING_OPTIMIZATION = "caching_optimization"
    COMPLETION = "completion"


@dataclass
class PreprocessingProgress:
    """Progress tracking for data preprocessing."""
    current_stage: PreprocessingStage
    stage_progress: float  # 0.0 - 1.0
    overall_progress: float  # 0.0 - 1.0
    
    # Stage details
    stages_completed: List[PreprocessingStage] = field(default_factory=list)
    current_stage_message: str = ""
    estimated_completion_time: Optional[datetime] = None
    
    # Data processing stats
    data_points_processed: int = 0
    total_data_points: int = 0
    trajectories_built: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'current_stage': self.current_stage.value,
            'stage_progress': self.stage_progress,
            'overall_progress': self.overall_progress,
            'stages_completed': [stage.value for stage in self.stages_completed],
            'current_stage_message': self.current_stage_message,
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'data_points_processed': self.data_points_processed,
            'total_data_points': self.total_data_points,
            'trajectories_built': self.trajectories_built
        }


@dataclass
class PreprocessingTask:
    """Represents a preprocessing task."""
    task_id: str
    session_id: str
    environment_id: str
    time_range: TimeRange
    stages_enabled: List[PreprocessingStage]
    
    # Task state
    started_at: datetime
    progress: PreprocessingProgress
    batch_size: int = 1000
    
    # Callbacks
    progress_callback: Optional[Callable[[PreprocessingProgress], None]] = None
    completion_callback: Optional[Callable[[bool, Optional[str]], None]] = None


class PreprocessingService:
    """
    Preprocessing domain service.
    
    Handles data preprocessing, temporal indexing, and preparation
    of historical data for efficient analysis.
    """
    
    def __init__(self, max_concurrent_tasks: int = 5):
        """
        Initialize preprocessing service.
        
        Args:
            max_concurrent_tasks: Maximum concurrent preprocessing tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task management
        self.active_tasks: Dict[str, PreprocessingTask] = {}
        self.task_futures: Dict[str, asyncio.Task] = {}
        
        # Stage configuration
        self.stage_weights = {
            PreprocessingStage.DATA_VALIDATION: 0.1,
            PreprocessingStage.TEMPORAL_INDEXING: 0.2,
            PreprocessingStage.FEATURE_EXTRACTION: 0.15,
            PreprocessingStage.TRAJECTORY_BUILDING: 0.25,
            PreprocessingStage.ANALYTICS_PREPARATION: 0.2,
            PreprocessingStage.CACHING_OPTIMIZATION: 0.1
        }
        
        # Processing statistics
        self._preprocessing_stats = {
            'tasks_started': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_data_points_processed': 0,
            'total_trajectories_built': 0,
            'avg_processing_time_seconds': 0.0
        }
        
        logger.debug("PreprocessingService initialized")
    
    async def start_preprocessing(
        self,
        session_id: str,
        environment_id: str,
        time_range: TimeRange,
        enabled_stages: Optional[List[PreprocessingStage]] = None,
        batch_size: int = 1000,
        progress_callback: Optional[Callable[[PreprocessingProgress], None]] = None,
        completion_callback: Optional[Callable[[bool, Optional[str]], None]] = None
    ) -> str:
        """
        Start preprocessing task.
        
        Args:
            session_id: Session identifier
            environment_id: Environment identifier
            time_range: Time range for preprocessing
            enabled_stages: Stages to enable (defaults to all)
            batch_size: Processing batch size
            progress_callback: Progress update callback
            completion_callback: Task completion callback
            
        Returns:
            Task identifier
            
        Raises:
            ValueError: If max concurrent tasks reached
        """
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            raise ValueError(f"Maximum concurrent tasks ({self.max_concurrent_tasks}) reached")
        
        # Generate task ID
        task_id = f"preproc_{session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Default stages
        if enabled_stages is None:
            enabled_stages = list(PreprocessingStage)
            enabled_stages.remove(PreprocessingStage.COMPLETION)
        
        # Create initial progress
        initial_progress = PreprocessingProgress(
            current_stage=enabled_stages[0],
            stage_progress=0.0,
            overall_progress=0.0,
            current_stage_message="Initializing preprocessing..."
        )
        
        # Create preprocessing task
        task = PreprocessingTask(
            task_id=task_id,
            session_id=session_id,
            environment_id=environment_id,
            time_range=time_range,
            stages_enabled=enabled_stages,
            started_at=datetime.utcnow(),
            progress=initial_progress,
            batch_size=batch_size,
            progress_callback=progress_callback,
            completion_callback=completion_callback
        )
        
        # Store task
        self.active_tasks[task_id] = task
        
        # Start async processing
        future = asyncio.create_task(self._execute_preprocessing(task))
        self.task_futures[task_id] = future
        
        self._preprocessing_stats['tasks_started'] += 1
        
        logger.info(f"Started preprocessing task {task_id} for session {session_id}")
        return task_id
    
    def get_preprocessing_progress(self, task_id: str) -> Optional[PreprocessingProgress]:
        """
        Get preprocessing progress.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Progress information or None if task not found
        """
        task = self.active_tasks.get(task_id)
        return task.progress if task else None
    
    def cancel_preprocessing(self, task_id: str) -> bool:
        """
        Cancel preprocessing task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancellation successful
        """
        if task_id not in self.task_futures:
            return False
        
        future = self.task_futures[task_id]
        if not future.done():
            future.cancel()
        
        # Clean up task
        self._cleanup_task(task_id)
        
        logger.info(f"Cancelled preprocessing task {task_id}")
        return True
    
    def get_active_task_count(self) -> int:
        """Get number of active preprocessing tasks."""
        return len(self.active_tasks)
    
    def get_preprocessing_statistics(self) -> Dict[str, Any]:
        """Get preprocessing service statistics."""
        return {
            **self._preprocessing_stats,
            'active_tasks': len(self.active_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks
        }
    
    async def _execute_preprocessing(self, task: PreprocessingTask) -> None:
        """
        Execute preprocessing task.
        
        Args:
            task: Preprocessing task to execute
        """
        try:
            logger.info(f"Executing preprocessing task {task.task_id}")
            
            # Process each enabled stage
            for i, stage in enumerate(task.stages_enabled):
                await self._process_stage(task, stage, i)
                
                # Update overall progress
                task.progress.overall_progress = (i + 1) / len(task.stages_enabled)
                task.progress.stages_completed.append(stage)
                
                # Call progress callback
                if task.progress_callback:
                    task.progress_callback(task.progress)
            
            # Mark completion
            task.progress.current_stage = PreprocessingStage.COMPLETION
            task.progress.stage_progress = 1.0
            task.progress.overall_progress = 1.0
            task.progress.current_stage_message = "Preprocessing completed successfully"
            
            # Final progress callback
            if task.progress_callback:
                task.progress_callback(task.progress)
            
            # Completion callback
            if task.completion_callback:
                task.completion_callback(True, None)
            
            self._preprocessing_stats['tasks_completed'] += 1
            
            logger.info(f"Completed preprocessing task {task.task_id}")
            
        except asyncio.CancelledError:
            logger.info(f"Preprocessing task {task.task_id} was cancelled")
            if task.completion_callback:
                task.completion_callback(False, "Task cancelled")
            
        except Exception as e:
            error_message = f"Preprocessing failed: {str(e)}"
            logger.error(f"Preprocessing task {task.task_id} failed: {e}")
            
            task.progress.current_stage_message = error_message
            
            if task.completion_callback:
                task.completion_callback(False, error_message)
            
            self._preprocessing_stats['tasks_failed'] += 1
            
        finally:
            self._cleanup_task(task.task_id)
    
    async def _process_stage(
        self,
        task: PreprocessingTask,
        stage: PreprocessingStage,
        stage_index: int
    ) -> None:
        """
        Process a specific preprocessing stage.
        
        Args:
            task: Preprocessing task
            stage: Current stage to process
            stage_index: Index of current stage
        """
        task.progress.current_stage = stage
        task.progress.stage_progress = 0.0
        
        # Stage-specific processing
        if stage == PreprocessingStage.DATA_VALIDATION:
            await self._validate_data(task)
        elif stage == PreprocessingStage.TEMPORAL_INDEXING:
            await self._build_temporal_index(task)
        elif stage == PreprocessingStage.FEATURE_EXTRACTION:
            await self._extract_features(task)
        elif stage == PreprocessingStage.TRAJECTORY_BUILDING:
            await self._build_trajectories(task)
        elif stage == PreprocessingStage.ANALYTICS_PREPARATION:
            await self._prepare_analytics(task)
        elif stage == PreprocessingStage.CACHING_OPTIMIZATION:
            await self._optimize_caching(task)
        
        task.progress.stage_progress = 1.0
        
        logger.debug(f"Completed stage {stage.value} for task {task.task_id}")
    
    async def _validate_data(self, task: PreprocessingTask) -> None:
        """Validate historical data."""
        task.progress.current_stage_message = "Validating historical data..."
        
        # Simulate data validation processing
        for i in range(10):
            await asyncio.sleep(0.1)  # Simulate processing
            task.progress.stage_progress = (i + 1) / 10
            if task.progress_callback:
                task.progress_callback(task.progress)
    
    async def _build_temporal_index(self, task: PreprocessingTask) -> None:
        """Build temporal index for efficient querying."""
        task.progress.current_stage_message = "Building temporal index..."
        
        # Simulate temporal indexing
        for i in range(20):
            await asyncio.sleep(0.05)
            task.progress.stage_progress = (i + 1) / 20
            if task.progress_callback:
                task.progress_callback(task.progress)
    
    async def _extract_features(self, task: PreprocessingTask) -> None:
        """Extract features from historical data."""
        task.progress.current_stage_message = "Extracting features..."
        
        # Simulate feature extraction
        for i in range(15):
            await asyncio.sleep(0.1)
            task.progress.stage_progress = (i + 1) / 15
            if task.progress_callback:
                task.progress_callback(task.progress)
    
    async def _build_trajectories(self, task: PreprocessingTask) -> None:
        """Build person trajectories."""
        task.progress.current_stage_message = "Building trajectories..."
        
        # Simulate trajectory building
        for i in range(25):
            await asyncio.sleep(0.08)
            task.progress.stage_progress = (i + 1) / 25
            task.progress.trajectories_built = i * 10  # Simulate trajectory count
            if task.progress_callback:
                task.progress_callback(task.progress)
        
        self._preprocessing_stats['total_trajectories_built'] += task.progress.trajectories_built
    
    async def _prepare_analytics(self, task: PreprocessingTask) -> None:
        """Prepare analytics data structures."""
        task.progress.current_stage_message = "Preparing analytics..."
        
        # Simulate analytics preparation
        for i in range(20):
            await asyncio.sleep(0.1)
            task.progress.stage_progress = (i + 1) / 20
            if task.progress_callback:
                task.progress_callback(task.progress)
    
    async def _optimize_caching(self, task: PreprocessingTask) -> None:
        """Optimize data caching."""
        task.progress.current_stage_message = "Optimizing cache..."
        
        # Simulate cache optimization
        for i in range(10):
            await asyncio.sleep(0.1)
            task.progress.stage_progress = (i + 1) / 10
            if task.progress_callback:
                task.progress_callback(task.progress)
    
    def _cleanup_task(self, task_id: str) -> None:
        """Clean up completed or cancelled task."""
        self.active_tasks.pop(task_id, None)
        self.task_futures.pop(task_id, None)