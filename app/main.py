from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from app.core.config import settings
from app.core import event_handlers
# Import routers
from app.api.v1.endpoints import processing_tasks
from app.api import websockets as ws_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Custom Debugging Middleware ---
class HeaderLoggingMiddleware:
    """
    Middleware to log headers of incoming requests, especially for debugging WebSocket connections.
    """
    def __init__(self, app):
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
                # For WebSockets, log this before proceeding, as rejection might happen early
                logger.info(f"--- WebSocket Connection Attempt --- \n{log_message}")
            elif scope["type"] == "http" and scope.get("path", "").startswith("/ws/"): # Log HTTP upgrade requests too
                logger.info(f"--- HTTP to WebSocket Upgrade Attempt --- \n{log_message}")
            # else:
            #     logger.debug(f"--- HTTP Request --- \n{log_message}") # Too verbose for every HTTP

        await self.app(scope, receive, send)
        


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    logger.info("Application startup sequence initiated...")
    # Ensure local directories exist
    Path(settings.LOCAL_VIDEO_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.LOCAL_FRAME_EXTRACTION_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured local video download dir: {settings.LOCAL_VIDEO_DOWNLOAD_DIR}")
    logger.info(f"Ensured local frame extraction dir: {settings.LOCAL_FRAME_EXTRACTION_DIR}")
    logger.info("Model loading deferred to first request/usage (via dependencies).")
    # ---------------------------------------------

    await event_handlers.on_startup(app)
    yield
    logger.info("Application shutdown sequence initiated...")
    await event_handlers.on_shutdown(app)

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    version="0.1.0",
    lifespan=lifespan, # Use the lifespan context manager
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    # Ensure docs URLs use the prefix if desired
    docs_url=f"/docs", # Standard /docs
    redoc_url=f"/redoc" # Standard /redoc
)

# --- Middleware Stack Order Matters ---
# Add custom debugging middleware first to see rawest input if needed
app.add_middleware(HeaderLoggingMiddleware)

# --- CORS Middleware ---
# This is crucial for allowing requests from different origins,
# including WebSocket connections, especially during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, restrict this to specific domains.
    allow_credentials=True, # Allows cookies to be included in cross-origin requests.
    allow_methods=["*"],  # Allows all HTTP methods.
    allow_headers=["*"],  # Allows all headers.
)


# --- API Routers ---
api_v1_router_prefix = settings.API_V1_PREFIX

app.include_router(
    processing_tasks.router,
    prefix=f"{api_v1_router_prefix}/processing-tasks", # Note: hyphenated path
    tags=["V1 - Processing Tasks"]
)
# app.include_router(
#     analytics_data.router,
#     prefix=f"{api_v1_router_prefix}/analytics",
#     tags=["V1 - Analytics Data"]
# )

# WebSocket Router (typically at root or specific path)
app.include_router(ws_router.router, prefix="/ws", tags=["WebSockets"])


@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint for health check or basic info."""
    return {"message": f"Welcome to {settings.APP_NAME} - Version {app.version}"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    # TODO: Add more sophisticated checks (DB, Redis, Model status)
    return {"status": "healthy"}