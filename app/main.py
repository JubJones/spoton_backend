from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Callable # For ASGIApp type hint

from app.core.config import settings
from app.core import event_handlers
from app.api.v1.endpoints import processing_tasks
from app.api.v1.endpoints import media as media_endpoints
from app.api import websockets as ws_router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    await event_handlers.on_startup(app_instance)
    yield
    logger.info("Application shutdown sequence initiated...")
    await event_handlers.on_shutdown(app_instance)

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    version="0.1.0",
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"/docs",
    redoc_url=f"/redoc"
)

app.add_middleware(HeaderLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1_router_prefix = settings.API_V1_PREFIX
app.include_router(
    processing_tasks.router,
    prefix=f"{api_v1_router_prefix}/processing-tasks",
    tags=["V1 - Processing Tasks"]
)
app.include_router(
    media_endpoints.router,
    prefix=f"{api_v1_router_prefix}/media",
    tags=["V1 - Media Content"]
)
app.include_router(ws_router.router, prefix="/ws", tags=["WebSockets"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.APP_NAME} - Version {app.version}"}

@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    detector_state = getattr(request.app.state, 'detector', None)
    tracker_factory_state = getattr(request.app.state, 'tracker_factory', None)
    homography_service_state = getattr(request.app.state, 'homography_service', None)

    detector_ready = detector_state and detector_state._model_loaded_flag
    tracker_factory_ready = tracker_factory_state and tracker_factory_state._prototype_tracker_loaded
    homography_ready = homography_service_state and homography_service_state._preloaded

    status_report = {
        "status": "healthy",
        "detector_model_loaded": bool(detector_ready),
        "prototype_tracker_loaded (reid_model)": bool(tracker_factory_ready),
        "homography_matrices_precomputed": bool(homography_ready)
    }
    if not all([detector_ready, tracker_factory_ready, homography_ready]):
        status_report["status"] = "degraded" # Or "starting_up" if more granular states are desired
        logger.warning(f"Health check: One or more components not fully ready: {status_report}")

    return status_report