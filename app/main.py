from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Callable # For ASGIApp type hint
import asyncio
import os

from app.core.config import settings
from app.core.security_config import configure_security_middleware, get_cors_config
from app.api.v1.endpoints import detection_processing_tasks
from app.api.v1.endpoints import environments
from app.api.v1.endpoints import analytics as analytics_endpoints
from app.api.v1.endpoints import playback_controls
from app.api.v1.endpoints import stream
from app.api.websockets import endpoints as ws_router
from app.api import health as health_router
from app.infrastructure.cache.tracking_cache import get_tracking_cache
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository
from app.services.environment_configuration_service import (
    initialize_environment_configuration_service,
    get_environment_configuration_service,
)
from app.services.analytics_engine import analytics_engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeaderLoggingMiddleware:
    def __init__(self, app: Callable):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            headers_decoded = {k.decode('utf-8'): v.decode('utf-8') for k, v in headers.items()}
            client_host, client_port = scope.get("client", (None, None))
            log_message = (
                f"Incoming Request Scope:\n"
                f"  Type: {scope['type']}\n"
                f"  Path: {scope.get('path')}\n"
                f"  Client: {client_host}:{client_port}\n"
                f"  HTTP Version: {scope.get('http_version')}\n"
                f"  Scheme: {scope.get('scheme')}\n"
                f"  Headers: {headers_decoded}\n"
            )
            if scope["type"] == "websocket":
                logger.info(f"--- WebSocket Connection Attempt --- \n{log_message}")
            elif scope["type"] == "http" and scope.get("path", "").startswith("/ws/"):
                logger.info(f"--- HTTP to WebSocket Upgrade Attempt --- \n{log_message}")

        await self.app(scope, receive, send)

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Application startup sequence initiated...")
    Path(settings.LOCAL_VIDEO_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.LOCAL_FRAME_EXTRACTION_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured local video download dir: {settings.LOCAL_VIDEO_DOWNLOAD_DIR}")
    logger.info(f"Ensured local frame extraction dir: {settings.LOCAL_FRAME_EXTRACTION_DIR}")

    # Minimal built-in startup/shutdown to replace removed event_handlers
    try:
        import torch
        from app.services.camera_tracker_factory import CameraTrackerFactory
        from app.services.homography_service import HomographyService
        from app.services.trail_management_service import TrailManagementService
        from app.services.detection_video_service import detection_video_service
        # Select compute device (CPU by default in CPU images)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        app_instance.state.compute_device = device
        logger.info(f"Compute device set to: {device}")

        # Preload tracker factory (ByteTrack prototype)
        tracker_factory = CameraTrackerFactory(device=device)
        if settings.PRELOAD_TRACKER_FACTORY:
            try:
                await tracker_factory.preload_prototype_tracker()
            except Exception as e:
                logger.warning(f"Tracker prototype preload failed (non-fatal): {e}")
        app_instance.state.tracker_factory = tracker_factory

        # Preload homography matrices
        homography_service = HomographyService(settings)
        if settings.PRELOAD_HOMOGRAPHY:
            try:
                await homography_service.preload_all_homography_matrices()
            except Exception as e:
                logger.warning(f"Homography preload failed (non-fatal): {e}")
        # Startup validation hints for homography and weights
        try:
            homography_json = Path(settings.HOMOGRAPHY_FILE_PATH)
            if not homography_json.exists():
                logger.warning(
                    f"Homography JSON file not found: {homography_json}. "
                    f"Ensure valid calibration file is present."
                )
            else:
                logger.info(f"Homography configuration found: {homography_json}")
            weights_dir = Path(settings.WEIGHTS_DIR).resolve()
            if not weights_dir.exists():
                logger.warning(
                    f"Weights directory not found: {weights_dir}. Ensure model weights are available or mounted."
                )
            else:
                logger.info(f"Weights directory present: {weights_dir}")
        except Exception as e:
            logger.debug(f"Startup validation checks skipped due to error: {e}")
        app_instance.state.homography_service = homography_service
        # Log RT-DETR model path configuration and a resolved sample (campus)
        try:
            logger.info(
                "RT-DETR configuration: default=%s, campus_override=%s, factory_override=%s",
                settings.RTDETR_MODEL_PATH,
                settings.RTDETR_MODEL_PATH_CAMPUS or "None",
                settings.RTDETR_MODEL_PATH_FACTORY or "None",
            )
            try:
                resolved_campus = detection_video_service._resolve_rtdetr_weights_for_environment("campus")
                exists_str = "present" if Path(resolved_campus).exists() else "missing"
                logger.info(
                    "RT-DETR resolved weights for 'campus': %s (%s)",
                    resolved_campus,
                    exists_str,
                )
            except Exception as e:
                logger.warning(f"Could not resolve RT-DETR weights for 'campus': {e}")
        except Exception as e:
            logger.debug(f"RT-DETR configuration logging skipped: {e}")

        # Initialize Environment Configuration Service so camera/env endpoints work
        try:
            # Best-effort DB session; service degrades when DB disabled
            db_session = None
            if getattr(settings, "DB_ENABLED", True):
                try:
                    from app.infrastructure.database.base import get_session_factory, setup_database
                    
                    # Ensure database tables exist
                    logger.info("Setting up database tables and indexes...")
                    await setup_database()
                    
                    SessionLocal = get_session_factory()
                    if SessionLocal is not None:
                        db_session = SessionLocal()
                except Exception as e:
                    logger.warning(f"Could not create DB session for environment service (continuing without DB): {e}")

            tracking_cache = get_tracking_cache()
            tracking_repository = TrackingRepository(db_session) if db_session is not None else TrackingRepository(None)
            initialize_environment_configuration_service(tracking_cache, tracking_repository)
            env_svc = get_environment_configuration_service()
            if env_svc:
                logger.info("EnvironmentConfigurationService initialized")
        except Exception as e:
            logger.warning(f"Environment service initialization encountered issues (non-fatal): {e}")

        # Global trail management service and cleanup loop
        trail_service = TrailManagementService(trail_length=settings.TRAIL_LENGTH)
        app_instance.state.trail_service = trail_service
        app_instance.state._trail_cleanup_task = None
        if settings.START_TRAIL_CLEANUP:
            async def _trail_cleanup_loop():
                while True:
                    try:
                        await trail_service.cleanup_old_trails(max_age_seconds=settings.TRAIL_MAX_AGE_SECONDS)
                        await asyncio.sleep(settings.TRAIL_CLEANUP_INTERVAL_SECONDS)
                    except Exception as e:
                        logger.warning(f"Trail cleanup loop error: {e}")
                        await asyncio.sleep(max(30, settings.TRAIL_CLEANUP_INTERVAL_SECONDS))
            app_instance.state._trail_cleanup_task = asyncio.create_task(_trail_cleanup_loop())
            logger.info("Started global trail cleanup background task")

        # Inject preloaded services into detection video service singleton
        try:
            detection_video_service.tracker_factory = tracker_factory
            detection_video_service.homography_service = homography_service
            detection_video_service.trail_service = trail_service
            logger.info("Injected preloaded services into detection video service")
        except Exception as e:
            logger.debug(f"Could not inject services into detection video service: {e}")

        # Initialize analytics engine so real-time metrics are populated
        try:
            await analytics_engine.initialize()
        except Exception as e:
            logger.warning(f"Analytics engine initialization encountered issues (non-fatal): {e}")

        # Detector is initialized on-demand by services; keep None for now
        app_instance.state.detector = None
    except Exception as e:
        logger.warning(f"Startup initialization encountered issues: {e}")

    try:
        yield
    finally:
        logger.info("Application shutdown sequence initiated...")
        try:
            await analytics_engine.shutdown()
        except Exception as e:
            logger.warning(f"Analytics engine shutdown encountered issues: {e}")
        # Graceful shutdown: cancel background tasks
        try:
            cleanup_task = getattr(app_instance.state, '_trail_cleanup_task', None)
            if cleanup_task:
                cleanup_task.cancel()
            # Close any DB session we might have opened for env service
            try:
                svc = get_environment_configuration_service()
                if svc and getattr(svc, 'repository', None) is not None and getattr(svc.repository, 'db', None) is not None:
                    svc.repository.db.close()
            except Exception:
                pass
        except Exception:
            pass

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    version="0.1.0",
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"/docs",
    redoc_url=f"/redoc"
)

# Configure security middleware (includes rate limiting, security headers, etc.)
configure_security_middleware(app)

# Add header logging middleware after security middleware
app.add_middleware(HeaderLoggingMiddleware)

# Configure secure CORS based on environment
cors_config = get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

api_v1_router_prefix = settings.API_V1_PREFIX
app.include_router(
    detection_processing_tasks.router,
    prefix=f"{api_v1_router_prefix}/detection-processing-tasks",
    tags=["V1 - RT-DETR Detection Tasks (Phase 1)"]
)
app.include_router(
    environments.router,
    prefix=f"{api_v1_router_prefix}",
    tags=["V1 - Environment Management"]
)
app.include_router(
    analytics_endpoints.router,
    prefix=f"{api_v1_router_prefix}/analytics",
    tags=["V1 - Analytics"]
)
app.include_router(
    stream.router,
    prefix=f"{api_v1_router_prefix}/stream",
    tags=["V1 - MJPEG Stream"]
)
if settings.ENABLE_PLAYBACK_CONTROL:
    app.include_router(
        playback_controls.router,
        prefix=f"{api_v1_router_prefix}/controls"
    )
else:
    logger.info("Playback control endpoints are disabled via settings.ENABLE_PLAYBACK_CONTROL")

app.include_router(ws_router.router, prefix="/ws", tags=["WebSockets"])

# Health Check System
app.include_router(health_router.router, tags=["Health Checks"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.APP_NAME} - Version {app.version}"}

@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    detector_state = getattr(request.app.state, 'detector', None)
    tracker_factory_state = getattr(request.app.state, 'tracker_factory', None)
    homography_service_state = getattr(request.app.state, 'homography_service', None)

    # Detector is initialized per task (RT-DETR). Do not require preload for health.
    detector_ready = True
    tracker_factory_ready = tracker_factory_state and tracker_factory_state._prototype_tracker_loaded
    homography_ready = False
    if homography_service_state is not None:
        try:
            homography_ready = bool(homography_service_state.json_homography_matrices)
        except Exception:
            homography_ready = False

    status_report = {
        "status": "healthy",
        "detector_model_loaded": bool(detector_ready),
        "prototype_tracker_loaded": bool(tracker_factory_ready),
        "homography_matrices_precomputed": bool(homography_ready)
    }
    if not all([detector_ready, tracker_factory_ready, homography_ready]):
        status_report["status"] = "degraded" # Or "starting_up" if more granular states are desired
        logger.warning(f"Health check: One or more components not fully ready: {status_report}")

    return status_report
