from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from pathlib import Path # Import Path

from app.core.config import settings
from app.core import event_handlers
# Import routers
from app.api.v1.endpoints import processing_tasks, analytics_data
from app.api import websockets as ws_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    await event_handlers.on_startup(app) # Other startup tasks (DB connections etc.)
    yield
    logger.info("Application shutdown sequence initiated...")
    await event_handlers.on_shutdown(app) # Clean up resources

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

# --- API Routers ---
api_v1_router_prefix = settings.API_V1_PREFIX

app.include_router(
    processing_tasks.router,
    prefix=f"{api_v1_router_prefix}/processing-tasks", # Note: hyphenated path
    tags=["V1 - Processing Tasks"]
)
app.include_router(
    analytics_data.router,
    prefix=f"{api_v1_router_prefix}/analytics",
    tags=["V1 - Analytics Data"]
)

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