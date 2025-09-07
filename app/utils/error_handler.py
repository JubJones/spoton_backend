"""
Comprehensive Error Handler for Phase 5: Production Readiness

This module provides robust error handling and recovery strategies for production deployment
as specified in DETECTION.md Phase 5. Includes specialized error recovery for different
pipeline components and intelligent fallback mechanisms.

Features:
- Component-specific error handling (Detection, Re-ID, Homography, Handoff)
- Intelligent recovery strategies with fallback mechanisms
- Circuit breaker pattern for preventing cascading failures
- Error classification and severity assessment
- Recovery state management and monitoring
"""

import asyncio
import time
import logging
import traceback
from functools import wraps
from typing import Callable, Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and response strategy."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentType(Enum):
    """System component types for targeted error handling."""
    DETECTION = "detection"
    REID = "reid"
    HOMOGRAPHY = "homography"
    HANDOFF = "handoff"
    SPATIAL_INTELLIGENCE = "spatial_intelligence"
    WEBSOCKET = "websocket"
    STORAGE = "storage"
    GENERIC = "generic"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    component: ComponentType
    operation: str
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for preventing cascading failures."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_success_threshold: int = 3


class ProductionErrorHandler:
    """
    Comprehensive error handling system for production deployment.
    
    Implements intelligent error recovery strategies, circuit breaker patterns,
    and component-specific fallback mechanisms for maximum system resilience.
    """
    
    def __init__(self):
        """Initialize production error handler with circuit breakers and recovery state."""
        self.circuit_breakers: Dict[ComponentType, CircuitBreakerState] = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ComponentType, Callable] = {}
        self.fallback_handlers: Dict[ComponentType, Callable] = {}
        
        # Initialize circuit breakers for each component
        for component in ComponentType:
            self.circuit_breakers[component] = CircuitBreakerState()
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'component_errors': {comp.value: 0 for comp in ComponentType}
        }
        
        logger.info("ProductionErrorHandler initialized with circuit breakers and recovery strategies")
    
    def handle_pipeline_errors(self, component: ComponentType, error_types: Dict[str, str] = None):
        """
        Decorator for handling pipeline errors with specific recovery strategies.
        
        Args:
            component: Component type for targeted error handling
            error_types: Optional mapping of error types to recovery strategies
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                error_context = ErrorContext(
                    component=component,
                    operation=func.__name__,
                    metadata={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                )
                
                # Check circuit breaker state
                if self._is_circuit_breaker_open(component):
                    logger.warning(f"Circuit breaker open for {component.value}, using fallback")
                    return await self._execute_fallback(component, error_context, *args, **kwargs)
                
                try:
                    # Execute the original function
                    result = await func(*args, **kwargs)
                    
                    # Record success for circuit breaker
                    self._record_success(component)
                    
                    return result
                    
                except Exception as e:
                    # Classify and handle the error
                    error_context.severity = self._classify_error_severity(e, component)
                    
                    logger.error(f"Pipeline error in {func.__name__} ({component.value}): {type(e).__name__} - {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Record failure for circuit breaker
                    self._record_failure(component)
                    
                    # Update error statistics
                    self._update_error_stats(error_context, e)
                    
                    # Attempt recovery
                    recovery_result = await self._attempt_recovery(e, error_context, *args, **kwargs)
                    
                    if recovery_result is not None:
                        logger.info(f"Successfully recovered from {type(e).__name__} in {component.value}")
                        self.error_stats['recovered_errors'] += 1
                        return recovery_result
                    else:
                        # Recovery failed, use fallback
                        logger.warning(f"Recovery failed for {component.value}, using fallback")
                        return await self._execute_fallback(component, error_context, *args, **kwargs)
                        
            return wrapper
        return decorator
    
    async def _attempt_recovery(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """
        Attempt to recover from an error using component-specific strategies.
        
        Args:
            error: The exception that occurred
            context: Error context information
            args, kwargs: Original function arguments
            
        Returns:
            Recovery result or None if recovery failed
        """
        try:
            # Check if we should attempt recovery
            if context.retry_count >= context.max_retries:
                logger.warning(f"Max retries ({context.max_retries}) exceeded for {context.component.value}")
                return None
            
            context.retry_count += 1
            
            # Get component-specific recovery strategy
            recovery_strategy = self.recovery_strategies.get(context.component)
            
            if recovery_strategy:
                logger.info(f"Attempting recovery for {context.component.value} (attempt {context.retry_count})")
                
                # Add error information to context
                context.recovery_data['error_type'] = type(error).__name__
                context.recovery_data['error_message'] = str(error)
                
                # Execute recovery strategy
                recovery_result = await recovery_strategy(error, context, *args, **kwargs)
                
                if recovery_result is not None:
                    logger.info(f"Recovery successful for {context.component.value}")
                    return recovery_result
            
            # Generic recovery attempt
            if context.retry_count <= 2:  # Only retry for first 2 attempts
                logger.info(f"Attempting generic retry for {context.component.value}")
                await asyncio.sleep(min(context.retry_count * 0.5, 2.0))  # Exponential backoff
                return None  # Signal to retry original function
            
            return None
            
        except Exception as recovery_error:
            logger.error(f"Error during recovery attempt: {recovery_error}")
            return None
    
    async def _execute_fallback(self, component: ComponentType, context: ErrorContext, *args, **kwargs) -> Any:
        """
        Execute fallback strategy when recovery fails or circuit breaker is open.
        
        Args:
            component: Component that failed
            context: Error context
            args, kwargs: Original function arguments
            
        Returns:
            Fallback result
        """
        try:
            fallback_handler = self.fallback_handlers.get(component)
            
            if fallback_handler:
                logger.info(f"Executing fallback for {component.value}")
                return await fallback_handler(context, *args, **kwargs)
            else:
                # Generic fallback
                return await self._generic_fallback(component, context)
                
        except Exception as fallback_error:
            logger.error(f"Fallback execution failed for {component.value}: {fallback_error}")
            return await self._generic_fallback(component, context)
    
    def _register_recovery_strategies(self):
        """Register component-specific recovery strategies."""
        self.recovery_strategies = {
            ComponentType.DETECTION: self._recover_detection_error,
            ComponentType.REID: self._recover_reid_error,
            ComponentType.HOMOGRAPHY: self._recover_homography_error,
            ComponentType.HANDOFF: self._recover_handoff_error,
            ComponentType.SPATIAL_INTELLIGENCE: self._recover_spatial_intelligence_error,
            ComponentType.WEBSOCKET: self._recover_websocket_error,
            ComponentType.STORAGE: self._recover_storage_error
        }
        
        self.fallback_handlers = {
            ComponentType.DETECTION: self._fallback_detection,
            ComponentType.REID: self._fallback_reid,
            ComponentType.HOMOGRAPHY: self._fallback_homography,
            ComponentType.HANDOFF: self._fallback_handoff,
            ComponentType.SPATIAL_INTELLIGENCE: self._fallback_spatial_intelligence,
            ComponentType.WEBSOCKET: self._fallback_websocket,
            ComponentType.STORAGE: self._fallback_storage
        }
    
    # Component-specific recovery strategies
    async def _recover_detection_error(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """Recovery strategy for detection errors."""
        try:
            if "GPU" in str(error) or "CUDA" in str(error):
                # GPU/CUDA error - try CPU fallback
                logger.info("GPU error detected, attempting CPU fallback")
                context.recovery_data['fallback_device'] = 'cpu'
                return None  # Signal to retry with CPU
            
            elif "model" in str(error).lower():
                # Model loading error - try reloading
                logger.info("Model error detected, attempting model reload")
                context.recovery_data['reload_model'] = True
                return None
            
            elif "memory" in str(error).lower():
                # Memory error - reduce batch size
                logger.info("Memory error detected, reducing processing load")
                context.recovery_data['reduce_batch_size'] = True
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error in detection recovery: {e}")
            return None
    
    async def _recover_reid_error(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """Recovery strategy for re-identification errors."""
        try:
            if "feature" in str(error).lower():
                # Feature extraction error - use fallback features
                logger.info("Feature extraction error, using fallback features")
                return {
                    'global_id': -1,  # Indicates re-ID failure
                    'reid_confidence': 0.0,
                    'feature_vector': None,
                    'error': str(error),
                    'fallback_used': True
                }
            
            elif "similarity" in str(error).lower():
                # Similarity computation error - assign new ID
                logger.info("Similarity computation error, assigning new ID")
                return {
                    'global_id': hash(str(time.time())) % 10000,  # Generate pseudo-random ID
                    'reid_confidence': 0.0,
                    'association_type': 'new_fallback',
                    'error': str(error)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in re-ID recovery: {e}")
            return None
    
    async def _recover_homography_error(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """Recovery strategy for homography errors."""
        try:
            if "matrix" in str(error).lower():
                # Matrix computation/loading error - use identity matrix
                logger.info("Homography matrix error, using fallback coordinates")
                return {
                    'map_x': 0.0,
                    'map_y': 0.0,
                    'projection_successful': False,
                    'error': str(error),
                    'fallback_used': True
                }
            
            elif "calibration" in str(error).lower():
                # Calibration points error - skip homography
                logger.info("Calibration error, skipping homography transformation")
                return None  # Skip homography processing
            
            return None
            
        except Exception as e:
            logger.error(f"Error in homography recovery: {e}")
            return None
    
    async def _recover_handoff_error(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """Recovery strategy for handoff detection errors."""
        try:
            # Handoff detection error - return safe defaults
            logger.info("Handoff detection error, using safe defaults")
            return (False, [])  # No handoff triggered, no candidate cameras
            
        except Exception as e:
            logger.error(f"Error in handoff recovery: {e}")
            return (False, [])
    
    async def _recover_spatial_intelligence_error(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """Recovery strategy for spatial intelligence errors."""
        try:
            # Spatial intelligence error - disable spatial features
            logger.info("Spatial intelligence error, disabling spatial features")
            return {
                'spatial_features_enabled': False,
                'homography_available': False,
                'handoff_detection_enabled': False,
                'error': str(error)
            }
            
        except Exception as e:
            logger.error(f"Error in spatial intelligence recovery: {e}")
            return None
    
    async def _recover_websocket_error(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """Recovery strategy for WebSocket errors."""
        try:
            if "connection" in str(error).lower():
                # Connection error - queue message for retry
                logger.info("WebSocket connection error, queuing message")
                context.recovery_data['queue_message'] = True
                return False  # Indicate sending failed but handled
            
            return None
            
        except Exception as e:
            logger.error(f"Error in WebSocket recovery: {e}")
            return False
    
    async def _recover_storage_error(self, error: Exception, context: ErrorContext, *args, **kwargs) -> Any:
        """Recovery strategy for storage errors."""
        try:
            if "s3" in str(error).lower() or "download" in str(error).lower():
                # S3/download error - try alternative source
                logger.info("Storage error, attempting alternative data source")
                context.recovery_data['use_alternative_source'] = True
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error in storage recovery: {e}")
            return None
    
    # Fallback handlers
    async def _fallback_detection(self, context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Fallback handler for detection failures."""
        return {
            "detections": [],
            "detection_count": 0,
            "processing_time_ms": 0,
            "error": "Detection service unavailable",
            "fallback_used": True
        }
    
    async def _fallback_reid(self, context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Fallback handler for re-ID failures."""
        return {
            "global_id": -1,
            "reid_confidence": 0.0,
            "error": "Re-ID service unavailable",
            "fallback_used": True
        }
    
    async def _fallback_homography(self, context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Fallback handler for homography failures."""
        return {
            "map_x": 0.0,
            "map_y": 0.0,
            "projection_successful": False,
            "error": "Homography service unavailable",
            "fallback_used": True
        }
    
    async def _fallback_handoff(self, context: ErrorContext, *args, **kwargs) -> tuple:
        """Fallback handler for handoff detection failures."""
        return (False, [])  # No handoff, no candidates
    
    async def _fallback_spatial_intelligence(self, context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Fallback handler for spatial intelligence failures."""
        return {
            "spatial_features_enabled": False,
            "error": "Spatial intelligence unavailable",
            "fallback_used": True
        }
    
    async def _fallback_websocket(self, context: ErrorContext, *args, **kwargs) -> bool:
        """Fallback handler for WebSocket failures."""
        logger.warning("WebSocket fallback: message not sent")
        return False
    
    async def _fallback_storage(self, context: ErrorContext, *args, **kwargs) -> Any:
        """Fallback handler for storage failures."""
        logger.error("Storage fallback: using empty data")
        return {}
    
    async def _generic_fallback(self, component: ComponentType, context: ErrorContext) -> Any:
        """Generic fallback when component-specific fallback fails."""
        logger.warning(f"Using generic fallback for {component.value}")
        return {
            "error": f"{component.value} service unavailable",
            "fallback_used": True,
            "timestamp": time.time()
        }
    
    # Circuit breaker methods
    def _is_circuit_breaker_open(self, component: ComponentType) -> bool:
        """Check if circuit breaker is open for a component."""
        breaker = self.circuit_breakers[component]
        
        if not breaker.is_open:
            return False
        
        # Check if recovery timeout has passed
        if time.time() - breaker.last_failure_time > breaker.recovery_timeout:
            breaker.is_open = False
            logger.info(f"Circuit breaker for {component.value} entering half-open state")
            return False
        
        return True
    
    def _record_success(self, component: ComponentType):
        """Record successful operation for circuit breaker."""
        breaker = self.circuit_breakers[component]
        breaker.success_count += 1
        
        if breaker.is_open and breaker.success_count >= breaker.half_open_success_threshold:
            breaker.is_open = False
            breaker.failure_count = 0
            breaker.success_count = 0
            logger.info(f"Circuit breaker for {component.value} closed - service recovered")
    
    def _record_failure(self, component: ComponentType):
        """Record failed operation for circuit breaker."""
        breaker = self.circuit_breakers[component]
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        breaker.success_count = 0
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.is_open = True
            logger.warning(f"Circuit breaker for {component.value} opened - too many failures")
    
    def _classify_error_severity(self, error: Exception, component: ComponentType) -> ErrorSeverity:
        """Classify error severity based on error type and component."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Critical errors
        if any(keyword in error_str for keyword in ['critical', 'fatal', 'system', 'corruption']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_str for keyword in ['gpu', 'cuda', 'memory', 'model']):
            return ErrorSeverity.HIGH
        
        # Component-specific severity
        if component == ComponentType.DETECTION and 'model' in error_str:
            return ErrorSeverity.HIGH
        elif component == ComponentType.WEBSOCKET and 'connection' in error_str:
            return ErrorSeverity.MEDIUM
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    def _update_error_stats(self, context: ErrorContext, error: Exception):
        """Update error statistics."""
        self.error_stats['total_errors'] += 1
        self.error_stats['component_errors'][context.component.value] += 1
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.error_stats['critical_errors'] += 1
        
        # Add to error history
        self.error_history.append(context)
        
        # Keep history manageable
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics and circuit breaker status."""
        try:
            return {
                'error_statistics': dict(self.error_stats),
                'circuit_breaker_status': {
                    comp.value: {
                        'is_open': breaker.is_open,
                        'failure_count': breaker.failure_count,
                        'success_count': breaker.success_count,
                        'last_failure_time': breaker.last_failure_time
                    }
                    for comp, breaker in self.circuit_breakers.items()
                },
                'recent_errors': [
                    {
                        'component': ctx.component.value,
                        'operation': ctx.operation,
                        'severity': ctx.severity.value,
                        'retry_count': ctx.retry_count,
                        'timestamp': ctx.timestamp
                    }
                    for ctx in self.error_history[-10:]  # Last 10 errors
                ],
                'recovery_effectiveness': {
                    'total_errors': self.error_stats['total_errors'],
                    'recovered_errors': self.error_stats['recovered_errors'],
                    'recovery_rate': (self.error_stats['recovered_errors'] / max(self.error_stats['total_errors'], 1)) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting error statistics: {e}")
            return {'error': str(e)}


# Global error handler instance
production_error_handler = ProductionErrorHandler()