"""
AI model configuration service for infrastructure layer.

Handles AI model settings, weights management, and model parameters.
Maximum 150 lines per plan.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of AI models."""
    DETECTOR = "detector"
    TRACKER = "tracker" 
    REID = "reid"
    FEATURE_EXTRACTOR = "feature_extractor"


class AIModelConfigService:
    """
    AI model configuration infrastructure service.
    
    Manages AI model configurations, weights paths, and model parameters.
    """
    
    def __init__(self, config_file_path: Optional[Path] = None):
        """
        Initialize AI model configuration service.
        
        Args:
            config_file_path: Optional path to configuration file
        """
        self.config_file_path = config_file_path or Path("config/ai_models.json")
        self._model_configs = {}
        
        self._load_model_configurations()
        logger.debug("AIModelConfigService initialized")
    
    def get_ai_model_configs(self) -> Dict[str, Any]:
        """Get all AI model configurations."""
        return self._model_configs.copy()
    
    def get_model_config(self, model_type: ModelType) -> Optional[Dict[str, Any]]:
        """
        Get configuration for specific model type.
        
        Args:
            model_type: Type of AI model
            
        Returns:
            Model configuration dictionary or None
        """
        model_key = model_type.value
        return self._model_configs.get(model_key)
    
    def update_model_config(
        self,
        model_type: ModelType,
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update model configuration.
        
        Args:
            model_type: Type of AI model
            config_updates: Configuration updates to apply
            
        Returns:
            True if update successful
        """
        model_key = model_type.value
        
        try:
            # Update or create configuration
            if model_key not in self._model_configs:
                self._model_configs[model_key] = {}
            
            self._model_configs[model_key].update(config_updates)
            
            # Save to file
            self._save_model_configurations()
            
            logger.info(f"Updated configuration for model type {model_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model {model_type.value} configuration: {e}")
            return False
    
    def validate_ai_configurations(self) -> Dict[str, Any]:
        """
        Validate all AI model configurations.
        
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'models_checked': 0
        }
        
        for model_key, config in self._model_configs.items():
            validation_result['models_checked'] += 1
            
            # Validate model path exists
            model_path = config.get('model_path')
            if model_path and not Path(model_path).exists():
                validation_result['errors'].append(
                    f"Model {model_key}: Model file not found at {model_path}"
                )
                validation_result['is_valid'] = False
            
            # Validate required parameters
            if model_key == 'detector':
                required_fields = ['model_path', 'confidence_threshold']
                for field in required_fields:
                    if field not in config:
                        validation_result['errors'].append(
                            f"Detector: Missing required field '{field}'"
                        )
                        validation_result['is_valid'] = False
            
            elif model_key == 'tracker':
                required_fields = ['tracker_type', 'max_age', 'n_init']
                for field in required_fields:
                    if field not in config:
                        validation_result['errors'].append(
                            f"Tracker: Missing required field '{field}'"
                        )
                        validation_result['is_valid'] = False
            
            elif model_key == 'reid':
                required_fields = ['model_path', 'feature_dimension']
                for field in required_fields:
                    if field not in config:
                        validation_result['errors'].append(
                            f"ReID: Missing required field '{field}'"
                        )
                        validation_result['is_valid'] = False
            
            # Validate thresholds are in valid range
            confidence_threshold = config.get('confidence_threshold')
            if confidence_threshold is not None:
                if not (0.0 <= confidence_threshold <= 1.0):
                    validation_result['warnings'].append(
                        f"Model {model_key}: Confidence threshold {confidence_threshold} outside valid range [0,1]"
                    )
        
        return validation_result
    
    def reload_configurations(self) -> bool:
        """
        Reload model configurations from file.
        
        Returns:
            True if reload successful
        """
        try:
            self._model_configs.clear()
            self._load_model_configurations()
            
            logger.info("AI model configurations reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload AI model configurations: {e}")
            return False
    
    def _load_model_configurations(self) -> None:
        """Load model configurations from file."""
        try:
            if not self.config_file_path.exists():
                logger.warning(f"AI model config file not found: {self.config_file_path}")
                self._create_default_configuration()
                return
            
            with open(self.config_file_path, 'r') as f:
                self._model_configs = json.load(f)
            
            logger.info(f"Loaded {len(self._model_configs)} AI model configurations")
            
        except Exception as e:
            logger.error(f"Failed to load AI model configurations: {e}")
            self._create_default_configuration()
    
    def _save_model_configurations(self) -> None:
        """Save model configurations to file."""
        try:
            # Ensure directory exists
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                **self._model_configs,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            with open(self.config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.debug(f"Saved AI model configurations to {self.config_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save AI model configurations: {e}")
    
    def _create_default_configuration(self) -> None:
        """Create default AI model configuration."""
        self._model_configs = {
            'detector': {
                'model_type': 'faster_rcnn',
                'model_path': 'weights/fasterrcnn_resnet50_fpn.pth',
                'confidence_threshold': 0.5,
                'device': 'cuda',
                'input_size': [640, 640],
                'nms_threshold': 0.5
            },
            'tracker': {
                'tracker_type': 'boxmot',
                'max_age': 30,
                'n_init': 3,
                'max_iou_distance': 0.7,
                'max_cosine_distance': 0.2
            },
            'reid': {
                'model_type': 'clip',
                'model_path': 'weights/clip_market1501.pt',
                'feature_dimension': 512,
                'similarity_threshold': 0.7
            },
            'feature_extractor': {
                'model_type': 'clip',
                'input_resolution': [224, 224],
                'normalize_features': True
            }
        }
        
        # Save default configuration
        self._save_model_configurations()
        
        logger.info("Created default AI model configuration")