"""
Orchestration layer for system coordination.

This layer coordinates the three core features:
- pipeline_orchestrator: Main processing pipeline coordination
- camera_manager: Multi-camera processing coordination
- real_time_processor: Real-time data flow management
- feature_integrator: Seamless data flow between detection, ReID, and mapping
- data_flow_manager: Data flow coordination between pipeline stages
- result_aggregator: Pipeline result collection and analysis
"""

from .pipeline_orchestrator import PipelineOrchestrator, orchestrator
from .camera_manager import CameraManager, camera_manager
from .real_time_processor import RealTimeProcessor, real_time_processor
from .feature_integrator import FeatureIntegrator
from .data_flow_manager import DataFlowManager
from .result_aggregator import ResultAggregator

__all__ = [
    'PipelineOrchestrator',
    'orchestrator',
    'CameraManager', 
    'camera_manager',
    'RealTimeProcessor',
    'real_time_processor',
    'FeatureIntegrator',
    'DataFlowManager',
    'ResultAggregator'
]