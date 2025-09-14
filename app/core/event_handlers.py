import logging
from fastapi import FastAPI
import torch # For device selection

from app.core.config import settings # For passing to services
from app.services.camera_tracker_factory import CameraTrackerFactory
from app.services.homography_service import HomographyService
from app.services.environment_configuration_service import initialize_environment_configuration_service
from app.services.historical_data_service import initialize_historical_data_service
from app.infrastructure.cache.tracking_cache import get_tracking_cache
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.infrastructure.database.base import get_db
from app.utils.device_utils import get_selected_device
from app.infrastructure.security.jwt_service import jwt_service

logger = logging.getLogger(__name__)

# Define keys for app.state for consistency
DETECTOR_KEY = "detector_instance"
TRACKER_FACTORY_KEY = "tracker_factory_instance"
HOMOGRAPHY_SERVICE_KEY = "homography_service_instance"
COMPUTE_DEVICE_KEY = "compute_device"


async def on_startup(app: FastAPI):
    """
    Actions to perform when the application starts.
    - Determine compute device.
    - Initialize and preload detector model.
    - Initialize CameraTrackerFactory and preload prototype tracker (for ReID model).
    - Initialize HomographyService and precompute all homography matrices.
    - Perform model warm-ups.
    """
    logger.info("--- SpotOn Backend: Executing Application Startup Tasks ---")

    # 0. Initialize Authentication Service
    logger.info("Initializing JWT authentication service...")
    try:
        await jwt_service.initialize()
        logger.info("JWT authentication service initialized with default users")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize JWT service: {e}", exc_info=True)
        # JWT service failure should not prevent app startup, but log the issue

    # 1. Determine Compute Device (once for the app)
    try:
        compute_device = get_selected_device(requested_device="auto") # Or from settings
        app.state.compute_device = compute_device
        logger.info(f"Selected compute device for application: {app.state.compute_device}")
    except Exception as e:
        logger.error(f"Failed to determine compute device during startup: {e}", exc_info=True)
        # Fallback to CPU or raise error to prevent startup
        app.state.compute_device = torch.device("cpu")
        logger.warning(f"Defaulting to CPU due to error in device selection.")


    # 2. Skip legacy detector preload (Faster R-CNN). RT-DETR is initialized per detection task.
    logger.info("Skipping legacy Faster R-CNN preload; RT-DETR loads within detection pipeline.")
    app.state.detector = None


    # 3. Initialize CameraTrackerFactory and Preload Prototype Tracker
    logger.info("Initializing CameraTrackerFactory and preloading prototype tracker (for ReID model)...")
    try:
        # Pass the determined compute_device to the factory
        tracker_factory = CameraTrackerFactory(device=app.state.compute_device)
        await tracker_factory.preload_prototype_tracker()
        app.state.tracker_factory = tracker_factory # Store instance
        logger.info("CameraTrackerFactory initialized and prototype tracker preloaded.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to preload prototype tracker: {e}", exc_info=True)
        app.state.tracker_factory = None


    # 4. Initialize HomographyService and Precompute Matrices
    logger.info("Initializing HomographyService and precomputing all homography matrices...")
    try:
        homography_service = HomographyService(settings=settings)
        await homography_service.preload_all_homography_matrices()
        app.state.homography_service = homography_service # Store instance
        logger.info("HomographyService initialized and matrices precomputed.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to precompute homography matrices: {e}", exc_info=True)
        app.state.homography_service = None

    # 5. Initialize Environment Configuration Service
    logger.info("Initializing Environment Configuration Service...")
    try:
        tracking_cache = get_tracking_cache()
        # Get database session for tracking repository
        db_session = next(get_db())
        tracking_repository = TrackingRepository(db_session)
        environment_service = initialize_environment_configuration_service(
            tracking_cache=tracking_cache,
            tracking_repository=tracking_repository
        )
        logger.info("Environment Configuration Service initialized successfully.")
    except Exception as e:
        logger.error(f"WARNING: Failed to initialize Environment Configuration Service: {e}", exc_info=True)
        # This is not critical for basic operations, so don't stop startup

    # 6. Initialize Historical Data Service
    logger.info("Initializing Historical Data Service...")
    try:
        tracking_cache = get_tracking_cache()
        # Get database session for tracking repository
        db_session = next(get_db())
        tracking_repository = TrackingRepository(db_session)
        historical_service = initialize_historical_data_service(
            tracking_cache=tracking_cache,
            tracking_repository=tracking_repository
        )
        logger.info("Historical Data Service initialized successfully.")
    except Exception as e:
        logger.error(f"WARNING: Failed to initialize Historical Data Service: {e}", exc_info=True)
        # This is not critical for basic operations, so don't stop startup

    logger.info("--- SpotOn Backend: Application Startup Tasks Completed ---")


async def on_shutdown(app: FastAPI):
    """
    Actions to perform when the application shuts down.
    - Release resources, close connections (if any were explicitly managed).
    """
    logger.info("--- SpotOn Backend: Executing Application Shutdown Tasks ---")
    # Example: If any services stored in app.state need explicit cleanup
    # if hasattr(app.state, TRACKER_FACTORY_KEY) and app.state.tracker_factory:
    #     logger.info("Cleaning up tracker factory resources (if any)...")
    #     # app.state.tracker_factory.cleanup() # If such a method existed

    # Models loaded into PyTorch will generally be released when Python process exits.
    # CUDA context might need explicit release if managed carefully.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared PyTorch CUDA cache.")

    logger.info("--- SpotOn Backend: Application Shutdown Tasks Completed ---")