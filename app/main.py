from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core import event_handlers
# Import routers
from app.api.v1.endpoints import processing_tasks, analytics_data
# from app.api.v1.endpoints import ingestion_tasks # REMOVED
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
    # Example: Create local directories if they don't exist
    Path(settings.LOCAL_VIDEO_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.LOCAL_FRAME_EXTRACTION_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured local video download dir: {settings.LOCAL_VIDEO_DOWNLOAD_DIR}")
    logger.info(f"Ensured local frame extraction dir: {settings.LOCAL_FRAME_EXTRACTION_DIR}")
    
    await event_handlers.on_startup(app) # Load AI models, connect to DBs etc.
    yield
    logger.info("Application shutdown sequence initiated...")
    await event_handlers.on_shutdown(app) # Clean up resources

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    version="0.1.0",
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc"
)

# --- API Routers ---
api_v1_router_prefix = settings.API_V1_PREFIX

app.include_router(
    processing_tasks.router,
    prefix=f"{api_v1_router_prefix}/processing-tasks",
    tags=["V1 - Processing Tasks"]
)
app.include_router(
    analytics_data.router, 
    prefix=f"{api_v1_router_prefix}/analytics",
    tags=["V1 - Analytics Data"]
)
# Ingestion tasks router is removed.

# WebSocket Router
app.include_router(ws_router.router, tags=["WebSockets"])


@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint for health check or basic info."""
    return {"message": f"Welcome to {settings.APP_NAME} - Version {app.version}"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy"}

# Need to import Path for main.py
from pathlib import Path