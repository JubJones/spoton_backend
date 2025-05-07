from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core import event_handlers
# from app.api.v1 import api_router_v1 # Assuming you create an aggregator router
from app.api.v1.endpoints import processing_tasks, analytics_data # analytics_data.py is empty but imported here as per the plan.
from app.api import websockets as ws_router # ws_router implies the entire websockets.py module is the router.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    print("Application startup...")
    await event_handlers.on_startup(app) # Load models, connect to DBs etc.
    yield
    print("Application shutdown...")
    await event_handlers.on_shutdown(app) # Clean up resources

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    version="0.1.0",
    lifespan=lifespan
)

# Include API routers
# A more organized way is to have an api_router_v1 in app/api/v1/__init__.py
# that includes all endpoint routers from app/api/v1/endpoints/
# For simplicity here:
app.include_router(processing_tasks.router, prefix=f"{settings.API_V1_PREFIX}/processing_tasks", tags=["Processing Tasks"])
app.include_router(analytics_data.router, prefix=f"{settings.API_V1_PREFIX}/analytics_data", tags=["Analytics Data"])
app.include_router(ws_router.router) # WebSocket router typically at root or /ws

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint for health check or basic info."""
    return {"message": f"Welcome to {settings.APP_NAME}"}

# Example: If you have a main API router for v1
# from app.api.v1.api import api_router as api_router_v1 # This file api.py is not in the plan
# app.include_router(api_router_v1, prefix=settings.API_V1_PREFIX)
