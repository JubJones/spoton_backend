"""
Calibration data loader for camera homography matrices and calibration parameters.

Handles loading, validation, and management of camera calibration data including:
- Homography matrices
- Camera intrinsic parameters
- Distortion coefficients
- Calibration metadata
"""

import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from app.domains.mapping.entities.camera_view import CameraView
from app.domains.mapping.entities.coordinate import CoordinateSystem, CoordinateTransformation
from app.shared.types import CameraID

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float
    skew: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """Convert to camera matrix."""
        return np.array([
            [self.focal_length_x, self.skew, self.principal_point_x],
            [0, self.focal_length_y, self.principal_point_y],
            [0, 0, 1]
        ])


@dataclass
class DistortionCoefficients:
    """Camera distortion coefficients."""
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to coefficient array."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


@dataclass
class CalibrationData:
    """Complete camera calibration data."""
    camera_id: CameraID
    homography_matrix: np.ndarray
    intrinsics: Optional[CameraIntrinsics] = None
    distortion: Optional[DistortionCoefficients] = None
    image_size: Tuple[int, int] = (1920, 1080)
    calibration_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calibration_method: str = "manual"
    reprojection_error: float = 0.0
    notes: str = ""
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate calibration data."""
        # Check homography matrix
        if self.homography_matrix.shape != (3, 3):
            return False, "Homography matrix must be 3x3"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(self.homography_matrix)) or np.any(np.isinf(self.homography_matrix)):
            return False, "Homography matrix contains NaN or infinite values"
        
        # Check determinant
        det = np.linalg.det(self.homography_matrix)
        if abs(det) < 1e-10:
            return False, "Homography matrix is singular"
        
        # Check image size
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            return False, "Invalid image size"
        
        return True, None


@dataclass
class CalibrationManifest:
    """Calibration manifest with metadata."""
    cameras: Dict[CameraID, CalibrationData] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    environment: str = "default"
    coordinate_system: str = "map"
    units: str = "meters"
    
    def add_camera(self, calibration_data: CalibrationData):
        """Add camera calibration data."""
        self.cameras[calibration_data.camera_id] = calibration_data
        self.updated_at = datetime.now(timezone.utc)
    
    def get_camera_ids(self) -> List[CameraID]:
        """Get all camera IDs."""
        return list(self.cameras.keys())
    
    def validate_all(self) -> Dict[CameraID, Tuple[bool, Optional[str]]]:
        """Validate all camera calibrations."""
        results = {}
        for camera_id, calibration_data in self.cameras.items():
            results[camera_id] = calibration_data.validate()
        return results


class CalibrationLoader:
    """
    Loader for camera calibration data.
    
    Features:
    - Load calibration data from multiple formats (JSON, NPZ, etc.)
    - Validate homography matrices and camera parameters
    - Generate CameraView objects from calibration data
    - Handle calibration data versioning and migration
    - Support for multiple calibration environments
    """
    
    def __init__(
        self,
        calibration_root: str = "data/calibration",
        default_environment: str = "default"
    ):
        """
        Initialize calibration loader.
        
        Args:
            calibration_root: Root directory for calibration data
            default_environment: Default environment name
        """
        self.calibration_root = Path(calibration_root)
        self.default_environment = default_environment
        
        # Loaded calibration data
        self.calibration_manifest: Optional[CalibrationManifest] = None
        self.camera_views: Dict[CameraID, CameraView] = {}
        
        # Statistics
        self.loader_stats = {
            "total_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "validation_errors": 0,
            "cameras_loaded": 0
        }
        
        # Create calibration directory if it doesn't exist
        self.calibration_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CalibrationLoader initialized with root: {self.calibration_root}")
    
    def load_calibration_manifest(
        self, 
        environment: Optional[str] = None
    ) -> Optional[CalibrationManifest]:
        """
        Load calibration manifest for environment.
        
        Args:
            environment: Environment name (uses default if None)
            
        Returns:
            Loaded calibration manifest or None if failed
        """
        env_name = environment or self.default_environment
        
        try:
            self.loader_stats["total_loads"] += 1
            
            # Look for calibration files in order of preference
            manifest_path = self.calibration_root / env_name / "manifest.json"
            
            if manifest_path.exists():
                manifest = self._load_json_manifest(manifest_path)
            else:
                # Try to build manifest from individual files
                manifest = self._build_manifest_from_files(env_name)
            
            if manifest is None:
                self.loader_stats["failed_loads"] += 1
                logger.error(f"Failed to load calibration manifest for environment {env_name}")
                return None
            
            # Validate all calibrations
            validation_results = manifest.validate_all()
            validation_errors = sum(1 for valid, _ in validation_results.values() if not valid)
            
            if validation_errors > 0:
                self.loader_stats["validation_errors"] += validation_errors
                logger.warning(f"Found {validation_errors} validation errors in calibration data")
                
                # Log validation errors
                for camera_id, (valid, error) in validation_results.items():
                    if not valid:
                        logger.error(f"Camera {camera_id} validation failed: {error}")
            
            self.calibration_manifest = manifest
            self.loader_stats["successful_loads"] += 1
            self.loader_stats["cameras_loaded"] += len(manifest.cameras)
            
            logger.info(f"Loaded calibration manifest for {env_name} with {len(manifest.cameras)} cameras")
            return manifest
            
        except Exception as e:
            self.loader_stats["failed_loads"] += 1
            logger.error(f"Error loading calibration manifest: {e}")
            return None
    
    def _load_json_manifest(self, manifest_path: Path) -> Optional[CalibrationManifest]:
        """Load calibration manifest from JSON file."""
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            
            # Parse manifest metadata
            manifest = CalibrationManifest(
                created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
                updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
                version=data.get("version", "1.0"),
                environment=data.get("environment", "default"),
                coordinate_system=data.get("coordinate_system", "map"),
                units=data.get("units", "meters")
            )
            
            # Parse camera calibrations
            for camera_id, cam_data in data.get("cameras", {}).items():
                calibration_data = self._parse_camera_calibration(camera_id, cam_data)
                if calibration_data:
                    manifest.add_camera(calibration_data)
            
            return manifest
            
        except Exception as e:
            logger.error(f"Error loading JSON manifest: {e}")
            return None
    
    def _parse_camera_calibration(self, camera_id: CameraID, cam_data: Dict[str, Any]) -> Optional[CalibrationData]:
        """Parse camera calibration data from JSON."""
        try:
            # Parse homography matrix
            homography_list = cam_data.get("homography_matrix")
            if homography_list is None:
                logger.error(f"No homography matrix for camera {camera_id}")
                return None
            
            homography_matrix = np.array(homography_list, dtype=np.float64)
            
            # Parse intrinsics (optional)
            intrinsics = None
            if "intrinsics" in cam_data:
                intrinsics_data = cam_data["intrinsics"]
                intrinsics = CameraIntrinsics(
                    focal_length_x=intrinsics_data.get("fx", 800.0),
                    focal_length_y=intrinsics_data.get("fy", 800.0),
                    principal_point_x=intrinsics_data.get("cx", 960.0),
                    principal_point_y=intrinsics_data.get("cy", 540.0),
                    skew=intrinsics_data.get("skew", 0.0)
                )
            
            # Parse distortion coefficients (optional)
            distortion = None
            if "distortion" in cam_data:
                distortion_data = cam_data["distortion"]
                distortion = DistortionCoefficients(
                    k1=distortion_data.get("k1", 0.0),
                    k2=distortion_data.get("k2", 0.0),
                    p1=distortion_data.get("p1", 0.0),
                    p2=distortion_data.get("p2", 0.0),
                    k3=distortion_data.get("k3", 0.0)
                )
            
            # Parse other metadata
            image_size = tuple(cam_data.get("image_size", [1920, 1080]))
            calibration_date = datetime.fromisoformat(
                cam_data.get("calibration_date", datetime.now(timezone.utc).isoformat())
            )
            
            return CalibrationData(
                camera_id=camera_id,
                homography_matrix=homography_matrix,
                intrinsics=intrinsics,
                distortion=distortion,
                image_size=image_size,
                calibration_date=calibration_date,
                calibration_method=cam_data.get("calibration_method", "manual"),
                reprojection_error=cam_data.get("reprojection_error", 0.0),
                notes=cam_data.get("notes", "")
            )
            
        except Exception as e:
            logger.error(f"Error parsing camera calibration for {camera_id}: {e}")
            return None
    
    def _build_manifest_from_files(self, environment: str) -> Optional[CalibrationManifest]:
        """Build calibration manifest from individual files."""
        try:
            env_path = self.calibration_root / environment
            
            if not env_path.exists():
                logger.warning(f"Environment directory does not exist: {env_path}")
                return None
            
            manifest = CalibrationManifest(environment=environment)
            
            # Look for individual homography files
            for file_path in env_path.glob("*.npz"):
                camera_id = file_path.stem
                
                try:
                    # Load NPZ file
                    data = np.load(file_path)
                    
                    if "homography" in data:
                        homography_matrix = data["homography"]
                        
                        # Create calibration data
                        calibration_data = CalibrationData(
                            camera_id=camera_id,
                            homography_matrix=homography_matrix,
                            calibration_method="file_load",
                            notes=f"Loaded from {file_path.name}"
                        )
                        
                        # Add intrinsics if available
                        if "camera_matrix" in data:
                            camera_matrix = data["camera_matrix"]
                            calibration_data.intrinsics = CameraIntrinsics(
                                focal_length_x=camera_matrix[0, 0],
                                focal_length_y=camera_matrix[1, 1],
                                principal_point_x=camera_matrix[0, 2],
                                principal_point_y=camera_matrix[1, 2],
                                skew=camera_matrix[0, 1]
                            )
                        
                        # Add distortion if available
                        if "distortion" in data:
                            dist_coeffs = data["distortion"]
                            calibration_data.distortion = DistortionCoefficients(
                                k1=dist_coeffs[0] if len(dist_coeffs) > 0 else 0.0,
                                k2=dist_coeffs[1] if len(dist_coeffs) > 1 else 0.0,
                                p1=dist_coeffs[2] if len(dist_coeffs) > 2 else 0.0,
                                p2=dist_coeffs[3] if len(dist_coeffs) > 3 else 0.0,
                                k3=dist_coeffs[4] if len(dist_coeffs) > 4 else 0.0
                            )
                        
                        manifest.add_camera(calibration_data)
                        
                except Exception as e:
                    logger.error(f"Error loading calibration file {file_path}: {e}")
                    continue
            
            if len(manifest.cameras) == 0:
                logger.warning(f"No valid calibration files found in {env_path}")
                return None
            
            return manifest
            
        except Exception as e:
            logger.error(f"Error building manifest from files: {e}")
            return None
    
    def save_calibration_manifest(
        self, 
        manifest: CalibrationManifest,
        environment: Optional[str] = None
    ) -> bool:
        """
        Save calibration manifest to file.
        
        Args:
            manifest: Calibration manifest to save
            environment: Environment name (uses manifest.environment if None)
            
        Returns:
            True if save successful
        """
        env_name = environment or manifest.environment
        
        try:
            env_path = self.calibration_root / env_name
            env_path.mkdir(parents=True, exist_ok=True)
            
            manifest_path = env_path / "manifest.json"
            
            # Convert to JSON-serializable format
            data = {
                "created_at": manifest.created_at.isoformat(),
                "updated_at": manifest.updated_at.isoformat(),
                "version": manifest.version,
                "environment": manifest.environment,
                "coordinate_system": manifest.coordinate_system,
                "units": manifest.units,
                "cameras": {}
            }
            
            for camera_id, calibration_data in manifest.cameras.items():
                cam_data = {
                    "homography_matrix": calibration_data.homography_matrix.tolist(),
                    "image_size": list(calibration_data.image_size),
                    "calibration_date": calibration_data.calibration_date.isoformat(),
                    "calibration_method": calibration_data.calibration_method,
                    "reprojection_error": calibration_data.reprojection_error,
                    "notes": calibration_data.notes
                }
                
                # Add intrinsics if available
                if calibration_data.intrinsics:
                    cam_data["intrinsics"] = {
                        "fx": calibration_data.intrinsics.focal_length_x,
                        "fy": calibration_data.intrinsics.focal_length_y,
                        "cx": calibration_data.intrinsics.principal_point_x,
                        "cy": calibration_data.intrinsics.principal_point_y,
                        "skew": calibration_data.intrinsics.skew
                    }
                
                # Add distortion if available
                if calibration_data.distortion:
                    cam_data["distortion"] = {
                        "k1": calibration_data.distortion.k1,
                        "k2": calibration_data.distortion.k2,
                        "p1": calibration_data.distortion.p1,
                        "p2": calibration_data.distortion.p2,
                        "k3": calibration_data.distortion.k3
                    }
                
                data["cameras"][camera_id] = cam_data
            
            # Save to file
            with open(manifest_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved calibration manifest to {manifest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration manifest: {e}")
            return False
    
    def get_camera_calibration(self, camera_id: CameraID) -> Optional[CalibrationData]:
        """Get calibration data for a specific camera."""
        if self.calibration_manifest is None:
            return None
        
        return self.calibration_manifest.cameras.get(camera_id)
    
    def get_homography_matrix(self, camera_id: CameraID) -> Optional[np.ndarray]:
        """Get homography matrix for a camera."""
        calibration_data = self.get_camera_calibration(camera_id)
        if calibration_data is None:
            return None
        
        return calibration_data.homography_matrix.copy()
    
    def create_camera_view(self, camera_id: CameraID) -> Optional[CameraView]:
        """Create CameraView object from calibration data."""
        calibration_data = self.get_camera_calibration(camera_id)
        if calibration_data is None:
            return None
        
        try:
            # Create coordinate transformation
            transformation = CoordinateTransformation(
                source_system=CoordinateSystem.IMAGE,
                target_system=CoordinateSystem.MAP,
                transformation_matrix=calibration_data.homography_matrix.tolist(),
                camera_id=camera_id,
                calibration_date=calibration_data.calibration_date
            )
            
            # Create camera view
            camera_view = CameraView(
                camera_id=camera_id,
                resolution=calibration_data.image_size,
                homography_transformation=transformation,
                calibration_date=calibration_data.calibration_date
            )
            
            self.camera_views[camera_id] = camera_view
            return camera_view
            
        except Exception as e:
            logger.error(f"Error creating camera view for {camera_id}: {e}")
            return None
    
    def get_all_camera_views(self) -> List[CameraView]:
        """Get all camera views from loaded calibration data."""
        if self.calibration_manifest is None:
            return []
        
        camera_views = []
        for camera_id in self.calibration_manifest.cameras:
            camera_view = self.create_camera_view(camera_id)
            if camera_view:
                camera_views.append(camera_view)
        
        return camera_views
    
    def validate_environment(self, environment: str) -> Dict[str, Any]:
        """Validate calibration environment."""
        try:
            env_path = self.calibration_root / environment
            
            if not env_path.exists():
                return {
                    "valid": False,
                    "error": f"Environment directory does not exist: {env_path}"
                }
            
            # Try to load manifest
            manifest = self.load_calibration_manifest(environment)
            if manifest is None:
                return {
                    "valid": False,
                    "error": "Failed to load calibration manifest"
                }
            
            # Validate all cameras
            validation_results = manifest.validate_all()
            
            valid_cameras = [camera_id for camera_id, (valid, _) in validation_results.items() if valid]
            invalid_cameras = [camera_id for camera_id, (valid, _) in validation_results.items() if not valid]
            
            return {
                "valid": len(invalid_cameras) == 0,
                "total_cameras": len(manifest.cameras),
                "valid_cameras": valid_cameras,
                "invalid_cameras": invalid_cameras,
                "validation_results": validation_results
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {e}"
            }
    
    def get_loader_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            **self.loader_stats,
            "calibration_root": str(self.calibration_root),
            "default_environment": self.default_environment,
            "manifest_loaded": self.calibration_manifest is not None,
            "camera_views_created": len(self.camera_views)
        }
    
    def list_environments(self) -> List[str]:
        """List available calibration environments."""
        try:
            environments = []
            for item in self.calibration_root.iterdir():
                if item.is_dir():
                    environments.append(item.name)
            return sorted(environments)
            
        except Exception as e:
            logger.error(f"Error listing environments: {e}")
            return []
    
    def reset_stats(self):
        """Reset loader statistics."""
        self.loader_stats = {
            "total_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "validation_errors": 0,
            "cameras_loaded": 0
        }
        logger.info("Calibration loader statistics reset")
    
    def cleanup(self):
        """Clean up loader resources."""
        self.calibration_manifest = None
        self.camera_views.clear()
        logger.info("CalibrationLoader cleaned up")