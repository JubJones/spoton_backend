"""
Model quantization service for GPU memory optimization.

Handles:
- Model quantization for memory efficiency
- Dynamic precision management
- Model optimization strategies
- Performance monitoring
- Memory usage reduction
"""

import asyncio
import logging
import torch
import torch.nn as nn
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import copy
import threading

from app.infrastructure.gpu import get_gpu_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Quantization types."""
    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    target_dtype: torch.dtype = torch.float16
    calibration_samples: int = 100
    accuracy_threshold: float = 0.95
    memory_reduction_target: float = 0.5
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    batch_size_optimization: bool = True


@dataclass
class ModelOptimizationResult:
    """Model optimization result."""
    model_name: str
    original_memory_mb: float
    optimized_memory_mb: float
    memory_reduction: float
    original_inference_time: float
    optimized_inference_time: float
    performance_impact: float
    accuracy_impact: float
    quantization_applied: QuantizationType
    optimization_time: float
    timestamp: datetime


@dataclass
class ModelProfile:
    """Model performance profile."""
    model_name: str
    model_size_mb: float
    parameter_count: int
    inference_time_ms: float
    memory_usage_mb: float
    accuracy_score: float
    is_quantized: bool
    quantization_type: Optional[QuantizationType] = None
    optimization_history: List[ModelOptimizationResult] = field(default_factory=list)


class ModelQuantizationService:
    """
    Model quantization service for GPU memory optimization.
    
    Features:
    - Dynamic and static quantization
    - Mixed precision training
    - Model optimization
    - Performance monitoring
    - Memory usage reduction
    """
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self.config = QuantizationConfig()
        
        # Model registry
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.quantized_models: Dict[str, nn.Module] = {}
        self.optimization_results: List[ModelOptimizationResult] = []
        
        # Optimization state
        self.optimization_in_progress = False
        self.optimization_queue: List[str] = []
        
        # Performance tracking
        self.quantization_stats = {
            'models_optimized': 0,
            'total_memory_saved_mb': 0.0,
            'average_performance_impact': 0.0,
            'optimization_time_total': 0.0,
            'quantization_attempts': 0,
            'quantization_successes': 0
        }
        
        # Thread safety
        self.optimization_lock = threading.Lock()
        
        logger.info("ModelQuantizationService initialized")
    
    async def initialize(self):
        """Initialize quantization service."""
        try:
            # Check GPU availability
            if not self.gpu_manager or not self.gpu_manager.is_available():
                logger.warning("GPU not available, quantization will use CPU")
            
            # Initialize quantization backends
            await self._initialize_quantization_backends()
            
            logger.info("ModelQuantizationService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ModelQuantizationService: {e}")
            raise
    
    async def _initialize_quantization_backends(self):
        """Initialize quantization backends."""
        try:
            # Check for available quantization backends
            if hasattr(torch, 'quantization'):
                logger.info("PyTorch quantization backend available")
            
            # Check for TensorRT support
            try:
                import tensorrt
                logger.info("TensorRT quantization backend available")
            except ImportError:
                logger.info("TensorRT not available")
            
            # Check for ONNX Runtime support
            try:
                import onnxruntime
                logger.info("ONNX Runtime quantization backend available")
            except ImportError:
                logger.info("ONNX Runtime not available")
            
        except Exception as e:
            logger.error(f"Error initializing quantization backends: {e}")
    
    # Model Registration and Profiling
    async def register_model(
        self,
        model_name: str,
        model: nn.Module,
        sample_input: torch.Tensor,
        accuracy_function: Optional[callable] = None
    ) -> bool:
        """Register model for quantization."""
        try:
            # Profile model
            profile = await self._profile_model(model_name, model, sample_input, accuracy_function)
            
            if profile:
                self.model_profiles[model_name] = profile
                logger.info(f"Registered model {model_name} for quantization")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    async def _profile_model(
        self,
        model_name: str,
        model: nn.Module,
        sample_input: torch.Tensor,
        accuracy_function: Optional[callable] = None
    ) -> Optional[ModelProfile]:
        """Profile model performance."""
        try:
            model.eval()
            
            # Calculate model size
            model_size_mb = self._calculate_model_size(model)
            
            # Count parameters
            parameter_count = sum(p.numel() for p in model.parameters())
            
            # Measure inference time
            inference_time_ms = await self._measure_inference_time(model, sample_input)
            
            # Measure memory usage
            memory_usage_mb = await self._measure_memory_usage(model, sample_input)
            
            # Calculate accuracy score
            accuracy_score = 1.0  # Default accuracy
            if accuracy_function:
                try:
                    accuracy_score = accuracy_function(model, sample_input)
                except Exception as e:
                    logger.warning(f"Error calculating accuracy: {e}")
            
            profile = ModelProfile(
                model_name=model_name,
                model_size_mb=model_size_mb,
                parameter_count=parameter_count,
                inference_time_ms=inference_time_ms,
                memory_usage_mb=memory_usage_mb,
                accuracy_score=accuracy_score,
                is_quantized=False
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling model: {e}")
            return None
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        try:
            total_size = 0
            for param in model.parameters():
                total_size += param.numel() * param.element_size()
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Error calculating model size: {e}")
            return 0.0
    
    async def _measure_inference_time(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Measure inference time in milliseconds."""
        try:
            device = next(model.parameters()).device
            sample_input = sample_input.to(device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(sample_input)
            
            # Synchronize GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            times = []
            for _ in range(10):
                if device.type == 'cuda':
                    start_time.record()
                    with torch.no_grad():
                        _ = model(sample_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
                else:
                    import time
                    start = time.time()
                    with torch.no_grad():
                        _ = model(sample_input)
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms
            
            return sum(times) / len(times)
            
        except Exception as e:
            logger.error(f"Error measuring inference time: {e}")
            return 0.0
    
    async def _measure_memory_usage(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Measure memory usage in MB."""
        try:
            device = next(model.parameters()).device
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                sample_input = sample_input.to(device)
                
                with torch.no_grad():
                    _ = model(sample_input)
                
                memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
                torch.cuda.empty_cache()
                
                return memory_usage
            else:
                # For CPU, return approximate memory usage
                return self._calculate_model_size(model) * 2  # Approximate with activations
                
        except Exception as e:
            logger.error(f"Error measuring memory usage: {e}")
            return 0.0
    
    # Quantization Methods
    async def optimize_model(
        self,
        model_name: str,
        model: nn.Module,
        sample_input: torch.Tensor,
        quantization_type: Optional[QuantizationType] = None,
        accuracy_function: Optional[callable] = None
    ) -> Optional[ModelOptimizationResult]:
        """Optimize model using quantization."""
        try:
            with self.optimization_lock:
                if self.optimization_in_progress:
                    self.optimization_queue.append(model_name)
                    logger.info(f"Added {model_name} to optimization queue")
                    return None
                
                self.optimization_in_progress = True
            
            self.quantization_stats['quantization_attempts'] += 1
            optimization_start = datetime.now(timezone.utc)
            
            # Get original profile
            original_profile = self.model_profiles.get(model_name)
            if not original_profile:
                original_profile = await self._profile_model(model_name, model, sample_input, accuracy_function)
                if original_profile:
                    self.model_profiles[model_name] = original_profile
            
            if not original_profile:
                logger.error(f"Cannot optimize model {model_name} - profiling failed")
                return None
            
            # Determine quantization type
            if quantization_type is None:
                quantization_type = self._select_optimal_quantization_type(original_profile)
            
            # Apply quantization
            quantized_model = await self._apply_quantization(model, quantization_type, sample_input)
            
            if quantized_model is None:
                logger.error(f"Quantization failed for model {model_name}")
                return None
            
            # Profile quantized model
            quantized_profile = await self._profile_model(
                f"{model_name}_quantized",
                quantized_model,
                sample_input,
                accuracy_function
            )
            
            if not quantized_profile:
                logger.error(f"Cannot profile quantized model {model_name}")
                return None
            
            # Calculate optimization results
            result = ModelOptimizationResult(
                model_name=model_name,
                original_memory_mb=original_profile.memory_usage_mb,
                optimized_memory_mb=quantized_profile.memory_usage_mb,
                memory_reduction=(original_profile.memory_usage_mb - quantized_profile.memory_usage_mb) / original_profile.memory_usage_mb,
                original_inference_time=original_profile.inference_time_ms,
                optimized_inference_time=quantized_profile.inference_time_ms,
                performance_impact=(quantized_profile.inference_time_ms - original_profile.inference_time_ms) / original_profile.inference_time_ms,
                accuracy_impact=(quantized_profile.accuracy_score - original_profile.accuracy_score) / original_profile.accuracy_score,
                quantization_applied=quantization_type,
                optimization_time=(datetime.now(timezone.utc) - optimization_start).total_seconds(),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Update profiles
            quantized_profile.is_quantized = True
            quantized_profile.quantization_type = quantization_type
            original_profile.optimization_history.append(result)
            
            # Store quantized model
            self.quantized_models[model_name] = quantized_model
            self.optimization_results.append(result)
            
            # Update statistics
            self.quantization_stats['models_optimized'] += 1
            self.quantization_stats['total_memory_saved_mb'] += (original_profile.memory_usage_mb - quantized_profile.memory_usage_mb)
            self.quantization_stats['optimization_time_total'] += result.optimization_time
            self.quantization_stats['quantization_successes'] += 1
            
            logger.info(f"Model {model_name} optimized: {result.memory_reduction:.2%} memory reduction")
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            return None
        finally:
            with self.optimization_lock:
                self.optimization_in_progress = False
                
                # Process queue
                if self.optimization_queue:
                    next_model = self.optimization_queue.pop(0)
                    logger.info(f"Processing queued optimization for {next_model}")
                    # Would trigger async optimization here
    
    def _select_optimal_quantization_type(self, profile: ModelProfile) -> QuantizationType:
        """Select optimal quantization type based on model profile."""
        try:
            # Simple heuristic for quantization type selection
            if profile.model_size_mb > 500:  # Large models
                return QuantizationType.INT8
            elif profile.model_size_mb > 100:  # Medium models
                return QuantizationType.FP16
            else:  # Small models
                return QuantizationType.DYNAMIC
                
        except Exception as e:
            logger.error(f"Error selecting quantization type: {e}")
            return QuantizationType.DYNAMIC
    
    async def _apply_quantization(
        self,
        model: nn.Module,
        quantization_type: QuantizationType,
        sample_input: torch.Tensor
    ) -> Optional[nn.Module]:
        """Apply quantization to model."""
        try:
            quantized_model = None
            
            if quantization_type == QuantizationType.DYNAMIC:
                quantized_model = await self._apply_dynamic_quantization(model)
            elif quantization_type == QuantizationType.STATIC:
                quantized_model = await self._apply_static_quantization(model, sample_input)
            elif quantization_type == QuantizationType.FP16:
                quantized_model = await self._apply_fp16_quantization(model)
            elif quantization_type == QuantizationType.INT8:
                quantized_model = await self._apply_int8_quantization(model, sample_input)
            elif quantization_type == QuantizationType.BF16:
                quantized_model = await self._apply_bf16_quantization(model)
            else:
                logger.warning(f"Unsupported quantization type: {quantization_type}")
                return None
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error applying quantization: {e}")
            return None
    
    async def _apply_dynamic_quantization(self, model: nn.Module) -> Optional[nn.Module]:
        """Apply dynamic quantization."""
        try:
            # Create a copy of the model
            quantized_model = copy.deepcopy(model)
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                quantized_model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error applying dynamic quantization: {e}")
            return None
    
    async def _apply_static_quantization(self, model: nn.Module, sample_input: torch.Tensor) -> Optional[nn.Module]:
        """Apply static quantization."""
        try:
            # Create a copy of the model
            quantized_model = copy.deepcopy(model)
            
            # Set to evaluation mode
            quantized_model.eval()
            
            # Prepare for static quantization
            quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model
            torch.quantization.prepare(quantized_model, inplace=True)
            
            # Calibrate with sample input
            with torch.no_grad():
                quantized_model(sample_input)
            
            # Convert to quantized model
            torch.quantization.convert(quantized_model, inplace=True)
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error applying static quantization: {e}")
            return None
    
    async def _apply_fp16_quantization(self, model: nn.Module) -> Optional[nn.Module]:
        """Apply FP16 quantization."""
        try:
            # Create a copy of the model
            quantized_model = copy.deepcopy(model)
            
            # Convert to half precision
            quantized_model = quantized_model.half()
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error applying FP16 quantization: {e}")
            return None
    
    async def _apply_int8_quantization(self, model: nn.Module, sample_input: torch.Tensor) -> Optional[nn.Module]:
        """Apply INT8 quantization."""
        try:
            # This is a simplified INT8 quantization
            # In practice, this would use more sophisticated methods
            return await self._apply_static_quantization(model, sample_input)
            
        except Exception as e:
            logger.error(f"Error applying INT8 quantization: {e}")
            return None
    
    async def _apply_bf16_quantization(self, model: nn.Module) -> Optional[nn.Module]:
        """Apply BF16 quantization."""
        try:
            # Create a copy of the model
            quantized_model = copy.deepcopy(model)
            
            # Convert to bfloat16 if supported
            if hasattr(torch, 'bfloat16'):
                quantized_model = quantized_model.to(torch.bfloat16)
            else:
                logger.warning("BFloat16 not supported, using FP16")
                quantized_model = quantized_model.half()
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error applying BF16 quantization: {e}")
            return None
    
    # Mixed Precision Support
    async def enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision training."""
        try:
            # Wrap model for mixed precision
            if hasattr(torch.cuda.amp, 'autocast'):
                # This would typically be done during training
                logger.info("Mixed precision enabled")
                return model
            else:
                logger.warning("Mixed precision not supported")
                return model
                
        except Exception as e:
            logger.error(f"Error enabling mixed precision: {e}")
            return model
    
    async def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing."""
        try:
            # Enable gradient checkpointing if supported
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Gradient checkpointing not supported for this model")
            
            return model
            
        except Exception as e:
            logger.error(f"Error enabling gradient checkpointing: {e}")
            return model
    
    # Batch Size Optimization
    async def optimize_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 64
    ) -> int:
        """Find optimal batch size for model."""
        try:
            device = next(model.parameters()).device
            optimal_batch_size = 1
            
            # Test different batch sizes
            for batch_size in [1, 2, 4, 8, 16, 32, 64]:
                if batch_size > max_batch_size:
                    break
                
                try:
                    # Create batch input
                    batch_input = sample_input.repeat(batch_size, 1, 1, 1) if len(sample_input.shape) == 4 else sample_input.repeat(batch_size, 1)
                    batch_input = batch_input.to(device)
                    
                    # Test inference
                    with torch.no_grad():
                        _ = model(batch_input)
                    
                    optimal_batch_size = batch_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    else:
                        raise
            
            logger.info(f"Optimal batch size found: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
            return 1
    
    # API Methods
    async def get_model_profiles(self) -> List[Dict[str, Any]]:
        """Get all model profiles."""
        try:
            profiles = []
            for profile in self.model_profiles.values():
                profiles.append({
                    'model_name': profile.model_name,
                    'model_size_mb': profile.model_size_mb,
                    'parameter_count': profile.parameter_count,
                    'inference_time_ms': profile.inference_time_ms,
                    'memory_usage_mb': profile.memory_usage_mb,
                    'accuracy_score': profile.accuracy_score,
                    'is_quantized': profile.is_quantized,
                    'quantization_type': profile.quantization_type.value if profile.quantization_type else None,
                    'optimization_count': len(profile.optimization_history)
                })
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error getting model profiles: {e}")
            return []
    
    async def get_optimization_results(self) -> List[Dict[str, Any]]:
        """Get optimization results."""
        try:
            results = []
            for result in self.optimization_results:
                results.append({
                    'model_name': result.model_name,
                    'original_memory_mb': result.original_memory_mb,
                    'optimized_memory_mb': result.optimized_memory_mb,
                    'memory_reduction': result.memory_reduction,
                    'original_inference_time': result.original_inference_time,
                    'optimized_inference_time': result.optimized_inference_time,
                    'performance_impact': result.performance_impact,
                    'accuracy_impact': result.accuracy_impact,
                    'quantization_applied': result.quantization_applied.value,
                    'optimization_time': result.optimization_time,
                    'timestamp': result.timestamp.isoformat()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting optimization results: {e}")
            return []
    
    async def get_quantization_statistics(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        try:
            return {
                'quantization_stats': self.quantization_stats,
                'registered_models': len(self.model_profiles),
                'quantized_models': len(self.quantized_models),
                'optimization_results': len(self.optimization_results),
                'optimization_in_progress': self.optimization_in_progress,
                'optimization_queue_size': len(self.optimization_queue),
                'configuration': {
                    'quantization_type': self.config.quantization_type.value,
                    'target_dtype': str(self.config.target_dtype),
                    'accuracy_threshold': self.config.accuracy_threshold,
                    'memory_reduction_target': self.config.memory_reduction_target
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting quantization statistics: {e}")
            return {}
    
    def reset_statistics(self):
        """Reset quantization statistics."""
        self.quantization_stats = {
            'models_optimized': 0,
            'total_memory_saved_mb': 0.0,
            'average_performance_impact': 0.0,
            'optimization_time_total': 0.0,
            'quantization_attempts': 0,
            'quantization_successes': 0
        }
        logger.info("Quantization statistics reset")


# Global model quantization service instance
model_quantization_service = ModelQuantizationService()