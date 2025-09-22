"""
Calibration service for camera calibration management and validation.

Provides business logic for:
- Camera calibration data management
- Calibration validation and quality assessment
- Calibration environment switching
- Calibration data persistence
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from app.domains.mapping.entities.camera_view import CameraView, CameraViewManager
from app.domains.mapping.models.calibration_loader import (
    CalibrationLoader,
    CalibrationData,
    CalibrationManifest,
    CameraIntrinsics,
    DistortionCoefficients
)
from app.domains.mapping.models.coordinate_transformer import CoordinateTransformer
from app.domains.mapping.models.homography_model import HomographyModel
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


class CalibrationService:
    """
    Service for camera calibration management.
    
    Features:
    - Load and manage calibration data
    - Validate calibration quality
    - Switch between calibration environments
    - Coordinate transformer integration
    - Calibration data persistence
    """
    
    def __init__(
        self,
        calibration_root: str = "data/calibration",
        coordinate_transformer: Optional[CoordinateTransformer] = None
    ):
        """
        Initialize calibration service.
        
        Args:
            calibration_root: Root directory for calibration data
            coordinate_transformer: Coordinate transformer instance
        """
        self.calibration_root = calibration_root
        self.coordinate_transformer = coordinate_transformer or CoordinateTransformer()
        
        # Calibration loader
        self.calibration_loader = CalibrationLoader(calibration_root)
        
        # Current calibration state
        self.current_environment: Optional[str] = None
        self.current_manifest: Optional[CalibrationManifest] = None
        self.camera_view_manager = CameraViewManager()
        
        # Homography models for validation
        self.homography_models: Dict[CameraID, HomographyModel] = {}
        
        # Service statistics
        self.service_stats = {
            "environments_loaded": 0,
            "calibrations_validated": 0,
            "calibration_errors": 0,
            "camera_views_created": 0,
            "transformers_registered": 0
        }
        
        logger.info("CalibrationService initialized")
    
    async def load_calibration_environment(self, environment: str) -> bool:
        """
        Load calibration environment.
        
        Args:
            environment: Environment name to load
            
        Returns:
            True if environment loaded successfully
        """
        try:
            logger.info(f"Loading calibration environment: {environment}")
            
            # Load calibration manifest
            manifest = self.calibration_loader.load_calibration_manifest(environment)
            
            if manifest is None:
                logger.error(f"Failed to load calibration manifest for environment: {environment}")
                return False
            
            # Validate all calibrations
            validation_results = manifest.validate_all()
            
            # Count validation errors
            validation_errors = sum(1 for valid, _ in validation_results.values() if not valid)
            
            if validation_errors > 0:
                logger.warning(f"Found {validation_errors} validation errors in environment {environment}")
                
                # Log specific errors
                for camera_id, (valid, error) in validation_results.items():
                    if not valid:
                        logger.error(f"Camera {camera_id} validation failed: {error}")
                        self.service_stats["calibration_errors"] += 1
            
            # Create camera views
            await self._create_camera_views(manifest)
            
            # Register homography models with coordinate transformer
            await self._register_coordinate_transformers(manifest)
            
            # Update current state
            self.current_environment = environment
            self.current_manifest = manifest
            
            self.service_stats["environments_loaded"] += 1
            self.service_stats["calibrations_validated"] += len(manifest.cameras)
            
            logger.info(f"Successfully loaded calibration environment: {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration environment {environment}: {e}")
            self.service_stats["calibration_errors"] += 1
            return False
    
    async def _create_camera_views(self, manifest: CalibrationManifest):
        """Create camera views from calibration manifest."""
        try:
            # Clear existing camera views
            self.camera_view_manager.clear_camera_views()
            
            # Create camera views for each camera
            for camera_id, calibration_data in manifest.cameras.items():
                camera_view = self.calibration_loader.create_camera_view(camera_id)
                
                if camera_view:
                    self.camera_view_manager.add_camera_view(camera_view)
                    self.service_stats["camera_views_created"] += 1
                    logger.debug(f"Created camera view for {camera_id}")
                else:
                    logger.warning(f"Failed to create camera view for {camera_id}")
            
            logger.info(f"Created {len(self.camera_view_manager.camera_views)} camera views")
            
        except Exception as e:
            logger.error(f"Error creating camera views: {e}")
    
    async def _register_coordinate_transformers(self, manifest: CalibrationManifest):
        """Register homography models with coordinate transformer."""
        try:
            # Clear existing homography models
            self.homography_models.clear()
            
            # Register each camera's homography with coordinate transformer
            for camera_id, calibration_data in manifest.cameras.items():
                # Create homography model
                homography_model = HomographyModel(camera_id=camera_id)
                
                # Load homography matrix
                if homography_model.load_homography_matrix(calibration_data.homography_matrix):
                    self.homography_models[camera_id] = homography_model
                    
                    # Register with coordinate transformer
                    success = self.coordinate_transformer.register_camera_homography(
                        camera_id=camera_id,
                        homography_matrix=calibration_data.homography_matrix
                    )
                    
                    if success:
                        self.service_stats["transformers_registered"] += 1
                        logger.debug(f"Registered coordinate transformer for {camera_id}")
                    else:
                        logger.warning(f"Failed to register coordinate transformer for {camera_id}")
                else:
                    logger.warning(f"Failed to load homography model for {camera_id}")
            
            logger.info(f"Registered {len(self.homography_models)} coordinate transformers")
            
        except Exception as e:
            logger.error(f"Error registering coordinate transformers: {e}")
    
    def get_calibration_data(self, camera_id: CameraID) -> Optional[CalibrationData]:
        """Get calibration data for a specific camera."""
        if self.current_manifest is None:
            return None
        
        return self.current_manifest.cameras.get(camera_id)
    
    def get_homography_matrix(self, camera_id: CameraID) -> Optional[np.ndarray]:
        """Get homography matrix for a camera."""
        calibration_data = self.get_calibration_data(camera_id)
        if calibration_data is None:
            return None
        
        return calibration_data.homography_matrix.copy()
    
    def get_camera_intrinsics(self, camera_id: CameraID) -> Optional[CameraIntrinsics]:
        """Get camera intrinsics for a camera."""
        calibration_data = self.get_calibration_data(camera_id)
        if calibration_data is None:
            return None
        
        return calibration_data.intrinsics
    
    def get_camera_view(self, camera_id: CameraID) -> Optional[CameraView]:
        """Get camera view for a camera."""
        return self.camera_view_manager.get_camera_view(camera_id)
    
    def get_all_camera_views(self) -> List[CameraView]:
        """Get all camera views."""
        return self.camera_view_manager.get_all_camera_views()
    
    def get_camera_ids(self) -> List[CameraID]:
        """Get all camera IDs in current environment."""
        if self.current_manifest is None:
            return []
        
        return self.current_manifest.get_camera_ids()
    
    async def validate_camera_calibration(self, camera_id: CameraID) -> Dict[str, Any]:
        """
        Validate calibration for a specific camera.
        
        Args:
            camera_id: Camera to validate
            
        Returns:
            Validation results
        """
        try:
            calibration_data = self.get_calibration_data(camera_id)
            
            if calibration_data is None:
                return {
                    "valid": False,
                    "error": f"No calibration data for camera {camera_id}"
                }
            
            # Validate calibration data
            valid, error = calibration_data.validate()
            
            if not valid:
                return {
                    "valid": False,
                    "error": error
                }
            
            # Validate homography model
            homography_model = self.homography_models.get(camera_id)
            
            if homography_model is None:
                return {
                    "valid": False,
                    "error": f"No homography model for camera {camera_id}"
                }
            
            validation_result = homography_model.get_validation_result()
            
            if validation_result is None:
                return {
                    "valid": False,
                    "error": "No validation result available"
                }
            
            # Validate coordinate transformer
            transformer_validation = self.coordinate_transformer.validate_camera_transformation(camera_id)
            
            # Combine validation results
            return {
                "valid": valid and validation_result.is_valid and transformer_validation["valid"],
                "calibration_validation": {
                    "valid": valid,
                    "error": error
                },
                "homography_validation": {
                    "valid": validation_result.is_valid,
                    "error": validation_result.error_message,
                    "condition_number": validation_result.condition_number,
                    "determinant": validation_result.determinant,
                    "test_success": validation_result.test_transformation_success
                },
                "transformer_validation": transformer_validation
            }
            
        except Exception as e:
            logger.error(f"Error validating camera calibration for {camera_id}: {e}")
            return {
                "valid": False,
                "error": f"Validation error: {e}"
            }
    
    async def validate_all_calibrations(self) -> Dict[CameraID, Dict[str, Any]]:
        """Validate all camera calibrations in current environment."""
        results = {}
        
        for camera_id in self.get_camera_ids():
            results[camera_id] = await self.validate_camera_calibration(camera_id)
        
        return results
    
    async def get_calibration_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive calibration quality report."""
        try:
            if self.current_manifest is None:
                return {
                    "environment": None,
                    "error": "No calibration environment loaded"
                }
            
            # Validate all calibrations
            validation_results = await self.validate_all_calibrations()
            
            # Calculate quality metrics
            total_cameras = len(self.current_manifest.cameras)
            valid_cameras = sum(1 for result in validation_results.values() if result["valid"])
            
            quality_score = valid_cameras / total_cameras if total_cameras > 0 else 0.0
            
            # Get transformation statistics
            transformer_stats = self.coordinate_transformer.get_transformation_stats()
            
            # Get camera overlaps
            camera_overlaps = self.camera_view_manager.get_camera_overlaps()
            
            return {
                "environment": self.current_environment,
                "manifest": {
                    "version": self.current_manifest.version,
                    "created_at": self.current_manifest.created_at.isoformat(),
                    "updated_at": self.current_manifest.updated_at.isoformat(),
                    "coordinate_system": self.current_manifest.coordinate_system,
                    "units": self.current_manifest.units
                },
                "quality_metrics": {
                    "total_cameras": total_cameras,
                    "valid_cameras": valid_cameras,
                    "invalid_cameras": total_cameras - valid_cameras,
                    "quality_score": quality_score
                },
                "validation_results": validation_results,
                "transformer_stats": transformer_stats,
                "camera_overlaps": [
                    {
                        "camera1": overlap[0],
                        "camera2": overlap[1],
                        "overlap_area": overlap[2]
                    }
                    for overlap in camera_overlaps
                ],
                "coverage_area": self.camera_view_manager.get_total_coverage_area()
            }
            
        except Exception as e:
            logger.error(f"Error generating calibration quality report: {e}")
            return {
                "environment": self.current_environment,
                "error": f"Report generation error: {e}"
            }
    
    def get_available_environments(self) -> List[str]:
        """Get list of available calibration environments."""
        return self.calibration_loader.list_environments()
    
    async def save_calibration_data(
        self,
        camera_id: CameraID,
        homography_matrix: np.ndarray,
        intrinsics: Optional[CameraIntrinsics] = None,
        distortion: Optional[DistortionCoefficients] = None,
        notes: str = ""
    ) -> bool:
        """
        Save calibration data for a camera.
        
        Args:
            camera_id: Camera identifier
            homography_matrix: Homography matrix
            intrinsics: Camera intrinsics (optional)
            distortion: Distortion coefficients (optional)
            notes: Additional notes
            
        Returns:
            True if save successful
        """
        try:
            if self.current_manifest is None:
                logger.error("No calibration environment loaded")
                return False
            
            # Create calibration data
            calibration_data = CalibrationData(
                camera_id=camera_id,
                homography_matrix=homography_matrix,
                intrinsics=intrinsics,
                distortion=distortion,
                calibration_date=datetime.now(timezone.utc),
                calibration_method="api_update",
                notes=notes
            )
            
            # Validate calibration data
            valid, error = calibration_data.validate()
            
            if not valid:
                logger.error(f"Invalid calibration data for {camera_id}: {error}")
                return False
            
            # Add to manifest
            self.current_manifest.add_camera(calibration_data)
            
            # Save manifest
            success = self.calibration_loader.save_calibration_manifest(self.current_manifest)
            
            if success:
                # Update coordinate transformer
                await self._register_coordinate_transformers(self.current_manifest)
                
                # Update camera views
                await self._create_camera_views(self.current_manifest)
                
                logger.info(f"Saved calibration data for camera {camera_id}")
                return True
            else:
                logger.error(f"Failed to save calibration data for camera {camera_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error saving calibration data for {camera_id}: {e}")
            return False
    
    async def create_new_environment(self, environment_name: str) -> bool:
        """
        Create a new calibration environment.
        
        Args:
            environment_name: Name for new environment
            
        Returns:
            True if environment created successfully
        """
        try:
            # Create new manifest
            new_manifest = CalibrationManifest(
                environment=environment_name,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Save empty manifest
            success = self.calibration_loader.save_calibration_manifest(new_manifest, environment_name)
            
            if success:
                logger.info(f"Created new calibration environment: {environment_name}")
                return True
            else:
                logger.error(f"Failed to create calibration environment: {environment_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating calibration environment {environment_name}: {e}")
            return False
    
    def get_current_environment(self) -> Optional[str]:
        """Get current calibration environment."""
        return self.current_environment
    
    def get_current_manifest(self) -> Optional[CalibrationManifest]:
        """Get current calibration manifest."""
        return self.current_manifest
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get calibration service statistics."""
        return {
            **self.service_stats,
            "current_environment": self.current_environment,
            "loaded_cameras": len(self.homography_models),
            "camera_views": len(self.camera_view_manager.camera_views),
            "calibration_root": self.calibration_root,
            "loader_stats": self.calibration_loader.get_loader_stats()
        }
    
    def reset_stats(self):
        """Reset service statistics."""
        self.service_stats = {
            "environments_loaded": 0,
            "calibrations_validated": 0,
            "calibration_errors": 0,
            "camera_views_created": 0,
            "transformers_registered": 0
        }
        
        self.calibration_loader.reset_stats()
        self.coordinate_transformer.reset_stats()
        
        logger.info("Calibration service statistics reset")
    
    async def cleanup(self):
        """Clean up calibration service resources."""
        try:
            # Clean up homography models
            for camera_id, homography_model in self.homography_models.items():
                homography_model.cleanup()
            
            self.homography_models.clear()
            
            # Clean up coordinate transformer
            self.coordinate_transformer.cleanup()
            
            # Clean up camera view manager
            self.camera_view_manager.clear_camera_views()
            
            # Clean up calibration loader
            self.calibration_loader.cleanup()
            
            # Reset state
            self.current_environment = None
            self.current_manifest = None
            
            logger.info("Calibration service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during calibration service cleanup: {e}")