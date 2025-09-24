# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create virtual environment using uv
uv venv .venv --python 3.9
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
uv pip install ".[dev]"

# For PyTorch CPU version (local development)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Docker Commands
```bash
# Build and start all services
docker-compose up --build -d

# Start only Redis and TimescaleDB for local development
docker-compose up -d redis timescaledb

# View backend logs
docker-compose logs -f backend

# Stop all services
docker-compose down

# Force rebuild without cache
docker-compose build --no-cache
```

### Running the Application
```bash
# Local development (requires Redis/TimescaleDB running)
uvicorn app.main:app --host 0.0.0.0 --port 3847 --reload

# Docker (recommended)
docker-compose up --build -d
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/models/test_detectors.py

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/services/test_pipeline_orchestrator.py::test_initialize_task
```

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix linting issues
ruff check . --fix
```

## Architecture Overview

This is a multi-camera person tracking system with AI-powered detection, tracking, and re-identification capabilities. The system processes video feeds from multiple cameras, tracks people across different camera views, and provides real-time analytics.

### Core Components

**Services Layer** (`app/services/`):
- `PipelineOrchestratorService` - Main processing pipeline coordinator
- `MultiCameraFrameProcessor` - Processes frames from multiple cameras simultaneously
- `VideoDataManagerService` - Manages video data from S3 storage
- `ReIDStateManager` - Handles person re-identification across cameras using FAISS
- `CameraTrackerFactory` - Creates and manages per-camera tracker instances
- `HomographyService` - Transforms image coordinates to map coordinates
- `NotificationService` - Manages WebSocket communications

**Models Layer** (`app/models/`):
- `RTDETRDetector` - Person detection using RT-DETR (Real-Time Detection Transformer) via Ultralytics
- `BoxMOTTracker` - Multi-object tracking with re-identification features
- `CLIPFeatureExtractor` - Extract features for person re-identification

**API Layer** (`app/api/`):
- REST endpoints for task management and status
- WebSocket endpoints for real-time tracking updates
- Health check endpoint for system readiness

### Key Design Patterns

- **Strategy Pattern**: Interchangeable AI models (detector, tracker, feature extractor)
- **Factory Pattern**: `CameraTrackerFactory` for per-camera instances
- **Service Layer**: Business logic encapsulation
- **Dependency Injection**: FastAPI's dependency system for services
- **Pipeline Pattern**: Sequential processing in `PipelineOrchestratorService`

### Data Flow

1. **Initialization**: AI models loaded during app startup (`app/core/event_handlers.py`)
2. **Task Start**: Client requests processing via REST API
3. **Video Processing**: 
   - Download sub-videos from S3 storage
   - Extract frames using OpenCV
   - Run detection (RT-DETR) and tracking (BotSort)
   - Perform cross-camera re-identification using CLIP features
   - Apply homography transformations for map coordinates
4. **Real-time Updates**: Stream tracking data and frame images via WebSocket
5. **Storage**: Cache recent data in Redis, historical data in TimescaleDB

### Configuration

The system uses environment variables configured in `.env` file:
- **S3 Configuration**: `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `S3_BUCKET_NAME`
- **Database**: `POSTGRES_*` and `REDIS_*` settings
- **AI Models**: `DETECTOR_TYPE`, `TRACKER_TYPE`, `REID_MODEL_TYPE`
- **Video Processing**: `TARGET_FPS`, `FRAME_JPEG_QUALITY`

Video sets and camera configurations are defined in `app/core/config.py` under `VIDEO_SETS` and `CAMERA_HANDOFF_DETAILS`.

### WebSocket Communication

The system sends two main message types:
- `tracking_update`: Contains bounding boxes, person IDs, map coordinates, and base64-encoded frame images
- `status_update`: Reports processing progress and current step

### Testing Strategy

- Unit tests for individual components (detectors, trackers, services)
- Integration tests for multi-component workflows
- Test client script: `scripts/websocket_client_test.py`
- Health check endpoint: `/health` for system readiness

### Frontend Simulation Testing

The project includes a comprehensive frontend simulation client for end-to-end testing:

#### Quick Start
```bash
# 1. Start backend services
docker-compose -f docker-compose.cpu.yml up -d

# 2. Install dependencies (if needed)
python setup_frontend_simulation.py

# 3. Run comprehensive test
python frontend_simulation_client.py
```

#### Features Tested
- ✅ Health endpoints (system, auth, analytics)
- ✅ Authentication flow (login, user info, permissions)  
- ✅ Processing task lifecycle (start, monitor, status)
- ✅ WebSocket connections (tracking, frames, system status)
- ✅ Media endpoints (video file serving)
- ✅ Analytics endpoints (real-time metrics, reports)

#### Dependencies
- `aiohttp>=3.8.0` - HTTP client for REST API testing
- `websockets>=10.0` - WebSocket client for real-time testing

If you encounter `ModuleNotFoundError: No module named 'aiohttp'`, run:
```bash
pip install aiohttp>=3.8.0 websockets>=10.0
```
Or use the automated setup: `python setup_frontend_simulation.py`

### Key Dependencies

- **FastAPI**: Web framework and WebSocket support
- **PyTorch**: AI model inference (detection, tracking, re-identification)
- **OpenCV**: Video processing and frame manipulation
- **BoxMOT**: Multi-object tracking library
- **Redis**: Real-time data caching
- **TimescaleDB**: Historical data storage
- **FAISS**: Efficient similarity search for re-identification

## Important Notes

- AI model weights must be downloaded manually (see `weights/note.txt`)
- Homography data files are required for coordinate transformations
- The system requires S3-compatible storage for video data
- GPU support available via CUDA-enabled Docker builds
- Health check endpoint should be polled before WebSocket connections
- Remember "Always check the checkbox in the planning file if it existed when you implemented the task"
