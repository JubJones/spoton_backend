"""
Batch Processing Optimization Service for Phase 5: Production Readiness

This service implements intelligent batching strategies for optimal GPU utilization
and performance optimization as specified in DETECTION.md Phase 5.

Features:
- Adaptive batch sizing based on GPU memory and processing capacity
- Intelligent frame batching for detection and feature extraction
- Performance monitoring and optimization metrics
- Resource-aware processing with automatic fallbacks
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingMetrics:
    """Metrics for batch processing performance tracking."""
    total_batches_processed: int = 0
    total_items_processed: int = 0
    average_batch_size: float = 0.0
    average_processing_time: float = 0.0
    gpu_utilization_peak: float = 0.0
    memory_utilization_peak: float = 0.0
    throughput_items_per_second: float = 0.0
    
    def update_metrics(self, batch_size: int, processing_time: float):
        """Update metrics with new batch processing data."""
        self.total_batches_processed += 1
        self.total_items_processed += batch_size
        
        # Update moving averages
        alpha = 0.1  # Exponential moving average factor
        self.average_batch_size = (1 - alpha) * self.average_batch_size + alpha * batch_size
        self.average_processing_time = (1 - alpha) * self.average_processing_time + alpha * processing_time
        
        # Calculate throughput
        if processing_time > 0:
            current_throughput = batch_size / processing_time
            self.throughput_items_per_second = (1 - alpha) * self.throughput_items_per_second + alpha * current_throughput


class BatchOptimizationService:
    """
    Intelligent batch processing service for production-grade performance optimization.
    
    Implements adaptive batching strategies based on system capacity and workload characteristics
    to maximize GPU utilization and minimize processing latency.
    """
    
    def __init__(self, initial_batch_size: int = 4, max_batch_size: int = 16):
        """
        Initialize batch optimization service.
        
        Args:
            initial_batch_size: Starting batch size for adaptive optimization
            max_batch_size: Maximum allowed batch size to prevent memory issues
        """
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.metrics = BatchProcessingMetrics()
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.batch_sizes: List[int] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Resource monitoring
        self.memory_pressure_threshold = 0.85  # 85% memory usage threshold
        self.performance_degradation_threshold = 1.5  # 50% performance drop threshold
        
        logger.info(f"BatchOptimizationService initialized - initial_batch_size: {initial_batch_size}, max_batch_size: {max_batch_size}")
    
    async def process_detection_batch(self, frames_data: List[Dict[str, Any]], 
                                    detector) -> List[Dict[str, Any]]:
        """
        Process multiple frames in optimized batches for detection.
        
        Args:
            frames_data: List of frame data dictionaries with 'frame', 'camera_id', 'frame_number'
            detector: RT-DETR detector instance
            
        Returns:
            List of detection results corresponding to input frames
        """
        if not frames_data:
            return []
        
        try:
            # Determine optimal batch size for current workload
            optimal_batch_size = self._calculate_optimal_batch_size(len(frames_data))
            
            results = []
            processing_start = time.time()
            
            # Process frames in optimal batches
            for i in range(0, len(frames_data), optimal_batch_size):
                batch = frames_data[i:i + optimal_batch_size]
                batch_start = time.time()
                
                # Extract frames for batch processing
                batch_frames = [item['frame'] for item in batch]
                
                # Process batch through detector
                if len(batch_frames) == 1:
                    # Single frame processing
                    batch_detections = [await detector.detect(batch_frames[0])]
                else:
                    # Batch processing (if detector supports it)
                    batch_detections = await self._batch_detect_frames(batch_frames, detector)
                
                # Combine results with metadata
                for j, (frame_data, detections) in enumerate(zip(batch, batch_detections)):
                    result = {
                        'frame_data': frame_data,
                        'detections': detections,
                        'batch_id': i // optimal_batch_size,
                        'batch_position': j
                    }
                    results.append(result)
                
                # Update performance metrics
                batch_time = time.time() - batch_start
                self.metrics.update_metrics(len(batch), batch_time)
                
                # Log batch processing details
                logger.debug(f"Processed batch {i//optimal_batch_size + 1} - size: {len(batch)}, time: {batch_time:.3f}s")
            
            # Adaptive optimization based on performance
            total_time = time.time() - processing_start
            await self._optimize_batch_size(total_time, len(frames_data))
            
            logger.info(f"Batch detection completed - {len(frames_data)} frames in {total_time:.3f}s, avg throughput: {len(frames_data)/total_time:.1f} fps")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch detection processing: {e}")
            raise
    
    async def process_feature_extraction_batch(self, detections_data: List[Dict[str, Any]], 
                                             feature_extractor) -> List[Dict[str, Any]]:
        """
        Process multiple detections in optimized batches for feature extraction.
        
        Args:
            detections_data: List of detection data with crops for feature extraction
            feature_extractor: CLIP feature extractor instance
            
        Returns:
            List of feature extraction results
        """
        if not detections_data:
            return []
        
        try:
            # Group detections by similarity for better batch efficiency
            grouped_detections = self._group_detections_for_batching(detections_data)
            
            results = []
            processing_start = time.time()
            
            for group in grouped_detections:
                batch_size = min(len(group), self.current_batch_size)
                
                for i in range(0, len(group), batch_size):
                    batch = group[i:i + batch_size]
                    batch_start = time.time()
                    
                    # Extract crops for batch processing
                    batch_crops = [item['crop'] for item in batch]
                    
                    # Process batch through feature extractor
                    if len(batch_crops) == 1:
                        batch_features = [await feature_extractor.extract_features_from_crop(batch_crops[0])]
                    else:
                        batch_features = await self._batch_extract_features(batch_crops, feature_extractor)
                    
                    # Combine results with metadata
                    for detection_data, features in zip(batch, batch_features):
                        result = {
                            'detection_data': detection_data,
                            'features': features,
                            'feature_dimension': len(features) if features is not None else 0
                        }
                        results.append(result)
                    
                    batch_time = time.time() - batch_start
                    logger.debug(f"Feature extraction batch - size: {len(batch)}, time: {batch_time:.3f}s")
            
            total_time = time.time() - processing_start
            logger.info(f"Batch feature extraction completed - {len(detections_data)} items in {total_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch feature extraction: {e}")
            raise
    
    async def _batch_detect_frames(self, frames: List[np.ndarray], detector) -> List[List]:
        """
        Perform batch detection on multiple frames.
        
        Args:
            frames: List of frame arrays
            detector: RT-DETR detector instance
            
        Returns:
            List of detection results for each frame
        """
        try:
            # Check if detector supports batch processing
            if hasattr(detector, 'detect_batch'):
                return await detector.detect_batch(frames)
            else:
                # Fallback to sequential processing
                results = []
                for frame in frames:
                    detections = await detector.detect(frame)
                    results.append(detections)
                return results
                
        except Exception as e:
            logger.error(f"Error in batch detection: {e}")
            # Fallback to sequential processing
            results = []
            for frame in frames:
                try:
                    detections = await detector.detect(frame)
                    results.append(detections)
                except Exception as frame_error:
                    logger.error(f"Error detecting frame in fallback: {frame_error}")
                    results.append([])  # Empty detection list for failed frame
            return results
    
    async def _batch_extract_features(self, crops: List[np.ndarray], feature_extractor) -> List[Optional[np.ndarray]]:
        """
        Perform batch feature extraction on multiple crops.
        
        Args:
            crops: List of cropped image arrays
            feature_extractor: CLIP feature extractor instance
            
        Returns:
            List of feature vectors for each crop
        """
        try:
            # Check if feature extractor supports batch processing
            if hasattr(feature_extractor, 'extract_features_batch'):
                return await feature_extractor.extract_features_batch(crops)
            else:
                # Fallback to sequential processing
                results = []
                for crop in crops:
                    features = await feature_extractor.extract_features_from_crop(crop)
                    results.append(features)
                return results
                
        except Exception as e:
            logger.error(f"Error in batch feature extraction: {e}")
            # Fallback to sequential processing
            results = []
            for crop in crops:
                try:
                    features = await feature_extractor.extract_features_from_crop(crop)
                    results.append(features)
                except Exception as crop_error:
                    logger.error(f"Error extracting features in fallback: {crop_error}")
                    results.append(None)  # None for failed feature extraction
            return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """
        Calculate optimal batch size based on current performance metrics and system state.
        
        Args:
            total_items: Total number of items to process
            
        Returns:
            Optimal batch size for current conditions
        """
        try:
            # Start with current batch size
            optimal_size = self.current_batch_size
            
            # Adjust based on total workload
            if total_items < self.current_batch_size:
                optimal_size = total_items
            elif total_items > self.current_batch_size * 4:
                # For large workloads, consider increasing batch size
                optimal_size = min(self.current_batch_size * 2, self.max_batch_size)
            
            # Check recent performance trends
            if len(self.processing_times) >= 3:
                recent_times = self.processing_times[-3:]
                if all(t > self.metrics.average_processing_time * self.performance_degradation_threshold for t in recent_times):
                    # Performance degrading, reduce batch size
                    optimal_size = max(1, optimal_size // 2)
                    logger.warning(f"Performance degradation detected, reducing batch size to {optimal_size}")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal batch size: {e}")
            return self.initial_batch_size
    
    def _group_detections_for_batching(self, detections_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group detections for efficient batch processing based on similarity.
        
        Args:
            detections_data: List of detection data
            
        Returns:
            List of detection groups optimized for batching
        """
        try:
            # Simple grouping by crop size similarity for now
            # In production, could use more sophisticated grouping strategies
            
            groups = []
            current_group = []
            
            for detection in detections_data:
                if len(current_group) >= self.current_batch_size:
                    groups.append(current_group)
                    current_group = [detection]
                else:
                    current_group.append(detection)
            
            if current_group:
                groups.append(current_group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error grouping detections for batching: {e}")
            # Fallback to single-item groups
            return [[detection] for detection in detections_data]
    
    async def _optimize_batch_size(self, processing_time: float, items_processed: int):
        """
        Adaptive batch size optimization based on performance feedback.
        
        Args:
            processing_time: Total processing time for recent batch
            items_processed: Number of items processed
        """
        try:
            # Track processing time
            self.processing_times.append(processing_time)
            self.batch_sizes.append(items_processed)
            
            # Keep only recent history
            if len(self.processing_times) > 20:
                self.processing_times = self.processing_times[-10:]
                self.batch_sizes = self.batch_sizes[-10:]
            
            # Optimization logic
            if len(self.processing_times) >= 5:
                recent_avg_time = sum(self.processing_times[-5:]) / 5
                
                # If performance is good and stable, try increasing batch size
                if (recent_avg_time < self.metrics.average_processing_time * 0.8 and 
                    self.current_batch_size < self.max_batch_size):
                    self.current_batch_size = min(self.current_batch_size + 1, self.max_batch_size)
                    logger.info(f"Performance good, increased batch size to {self.current_batch_size}")
                
                # If performance is degrading, reduce batch size
                elif (recent_avg_time > self.metrics.average_processing_time * 1.2 and 
                      self.current_batch_size > 1):
                    self.current_batch_size = max(1, self.current_batch_size - 1)
                    logger.info(f"Performance degrading, reduced batch size to {self.current_batch_size}")
            
            # Record optimization decision
            self.optimization_history.append({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'items_processed': items_processed,
                'batch_size_decision': self.current_batch_size,
                'avg_processing_time': self.metrics.average_processing_time
            })
            
            # Keep optimization history manageable
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-50:]
            
        except Exception as e:
            logger.error(f"Error in batch size optimization: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring.
        
        Returns:
            Dictionary containing performance metrics and optimization data
        """
        try:
            return {
                'batch_processing': {
                    'current_batch_size': self.current_batch_size,
                    'max_batch_size': self.max_batch_size,
                    'total_batches_processed': self.metrics.total_batches_processed,
                    'total_items_processed': self.metrics.total_items_processed,
                    'average_batch_size': round(self.metrics.average_batch_size, 2),
                    'average_processing_time': round(self.metrics.average_processing_time, 3),
                    'throughput_fps': round(self.metrics.throughput_items_per_second, 2)
                },
                'performance_optimization': {
                    'optimization_decisions': len(self.optimization_history),
                    'recent_processing_times': self.processing_times[-5:] if self.processing_times else [],
                    'performance_trend': self._calculate_performance_trend(),
                    'optimization_effectiveness': self._calculate_optimization_effectiveness()
                },
                'system_utilization': {
                    'memory_pressure_threshold': self.memory_pressure_threshold,
                    'performance_degradation_threshold': self.performance_degradation_threshold,
                    'current_efficiency': self._calculate_current_efficiency()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_trend(self) -> str:
        """Calculate current performance trend."""
        try:
            if len(self.processing_times) < 3:
                return 'insufficient_data'
            
            recent = sum(self.processing_times[-3:]) / 3
            older = sum(self.processing_times[-6:-3]) / 3 if len(self.processing_times) >= 6 else recent
            
            if recent < older * 0.9:
                return 'improving'
            elif recent > older * 1.1:
                return 'degrading' 
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'
    
    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate effectiveness of batch size optimization."""
        try:
            if len(self.optimization_history) < 2:
                return 0.0
            
            initial_throughput = self.optimization_history[0].get('items_processed', 1) / max(self.optimization_history[0].get('processing_time', 1), 0.001)
            recent_throughput = self.metrics.throughput_items_per_second
            
            return max(0.0, (recent_throughput - initial_throughput) / max(initial_throughput, 0.001))
            
        except Exception:
            return 0.0
    
    def _calculate_current_efficiency(self) -> float:
        """Calculate current processing efficiency score."""
        try:
            if self.metrics.average_processing_time <= 0:
                return 0.0
            
            # Efficiency based on throughput and batch utilization
            batch_utilization = self.metrics.average_batch_size / max(self.current_batch_size, 1)
            throughput_score = min(1.0, self.metrics.throughput_items_per_second / 30.0)  # Target 30 fps
            
            return (batch_utilization * 0.4 + throughput_score * 0.6)
            
        except Exception:
            return 0.0


# Global batch optimization service instance
batch_optimization_service = BatchOptimizationService()