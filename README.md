# SpotOn Backend

Intelligent multi-camera person tracking and analytics system for processing video feeds with AI-driven detection, tracking, and re-identification capabilities.

## Overview

SpotOn processes video data from multiple cameras, performs person detection and tracking, maintains identity across camera views using CLIP-based re-identification, and provides real-time analytics through REST APIs and WebSockets.

**Core Features:**
- **Multi-View Person Detection** - Identifies people in each camera independently
- **Cross-Camera Re-Identification** - Tracks the same person across multiple cameras
- **Unified Spatial Mapping** - Maps all detections to a common 2D coordinate system
- **Real-Time Analytics** - Live tracking updates and historical data analysis

## Quick Start

### Prerequisites
- **Docker & Docker Compose**
- **Model weights**: Download `clip_market1501.pt` (~600MB) from the [Google Drive link](https://drive.google.com/uc?id=1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7)

### Setup Options

#### CPU Version (Development & Testing)
```bash
# Clone repository
git clone <your-repo-url>
cd spoton_backend

# Download model weights to ./weights/clip_market1501.pt

# Setup and start (CPU-optimized)
cp .env.cpu .env
docker-compose -f docker-compose.cpu.yml up --build -d

# Verify setup
curl http://localhost:8000/health
```

#### GPU Version (Production Performance)
```bash
# Prerequisites: NVIDIA GPU + NVIDIA Container Toolkit

# Setup and start (GPU-optimized)
cp .env.gpu .env
docker-compose -f docker-compose.gpu.yml up --build -d

# Verify setup
curl http://localhost:8000/health
```

### Expected Health Response
```json
{
  "status": "healthy",
  "gpu_available": false,  // true for GPU version
  "device": "cpu",         // "cuda" for GPU version
  "models_loaded": true,
  "database_connected": true,
  "redis_connected": true
}
```

## Performance Expectations

| Version | Detection Speed | Processing FPS | Memory Usage | Startup Time |
|---------|----------------|----------------|--------------|---------------|
| **CPU** | 300-500ms/frame | 2-4 FPS | 4-8GB RAM | 60-120s |
| **GPU** | 30-50ms/frame | 10-15 FPS | 2-4GB GPU | 30-60s |

## Usage

### Basic Operations
```bash
# View logs
docker-compose -f docker-compose.cpu.yml logs -f backend  # CPU
docker-compose -f docker-compose.gpu.yml logs -f backend  # GPU

# Stop services
docker-compose -f docker-compose.cpu.yml down  # CPU
docker-compose -f docker-compose.gpu.yml down  # GPU

# Monitor resources
docker stats
```

### API Testing
```bash
# Check available endpoints
curl http://localhost:8000/api/v1/

# Start a tracking session
curl -X POST http://localhost:8000/api/v1/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"video_set": "demo", "cameras": ["cam1"]}'

# Test WebSocket connection
python scripts/websocket_client_test.py
```

## Configuration

### Environment Files
- **`.env.cpu`** - CPU-optimized settings (3 FPS, reduced quality)
- **`.env.gpu`** - GPU-optimized settings (15 FPS, high quality, AMP enabled)

### Key Settings
```bash
# Performance tuning
TARGET_FPS=3           # CPU: 3, GPU: 15
FRAME_JPEG_QUALITY=70  # CPU: 70, GPU: 90
DETECTION_CONFIDENCE_THRESHOLD=0.7  # CPU: 0.7, GPU: 0.5

# Model configuration
DETECTOR_TYPE="fasterrcnn"
TRACKER_TYPE="botsort"
REID_MODEL_TYPE="clip"
```

## Architecture

```
app/
├── domains/           # Core AI features
│   ├── detection/     # Person detection (Faster R-CNN, YOLO)
│   ├── reid/          # Re-identification (CLIP-based)
│   └── mapping/       # Spatial mapping & homography
├── infrastructure/    # Database, cache, GPU management
├── orchestration/     # Pipeline coordination
└── api/              # REST & WebSocket endpoints
```

## API Endpoints

- **Health**: `GET /health`
- **Sessions**: `POST /api/v1/sessions/`
- **Tracking**: `GET /api/v1/tracking/{session_id}`
- **Analytics**: `GET /api/v1/analytics/`
- **WebSocket**: `ws://localhost:8000/ws/tracking/`

## Troubleshooting

### Common Issues

**Slow Performance (CPU):**
```bash
# Monitor resources
docker stats
htop

# Reduce processing load
echo "TARGET_FPS=2" >> .env
echo "DETECTION_CONFIDENCE_THRESHOLD=0.8" >> .env
```

**Model Loading Errors:**
```bash
# Verify weights file exists
ls -la weights/clip_market1501.pt  # Should be ~600MB

# Check logs
docker-compose logs backend
```

**Memory Issues:**
```bash
# Check available memory
free -h

# Restart with cleanup
docker-compose restart
```

### GPU-Specific Issues
- Ensure NVIDIA Container Toolkit is installed
- Check `nvidia-smi` for GPU availability
- Verify CUDA compatibility (CUDA 12.1 required)

## Dependencies

**Core Technologies:**
- **FastAPI** - Web framework and WebSocket support
- **PyTorch** - AI model inference (CPU/GPU)
- **Redis** - Real-time data caching
- **TimescaleDB** - Historical data storage
- **BoxMOT** - Multi-object tracking
- **FAISS** - Similarity search for re-identification

**AI Models:**
- **Detection**: Faster R-CNN, YOLO
- **Re-ID**: CLIP-based feature extraction
- **Tracking**: BotSort with cross-camera fusion

## Development

### Local Development
```bash
# Install dependencies
uv venv .venv --python 3.9
source .venv/bin/activate
uv pip install ".[dev]"

# Run tests
pytest

# Code quality
ruff format .
ruff check .
```

### Project Structure
```
spoton_backend/
├── docker-compose.cpu.yml  # CPU deployment
├── docker-compose.gpu.yml  # GPU deployment  
├── .env.cpu                # CPU configuration
├── .env.gpu                # GPU configuration
├── weights/                # AI model weights
│   └── clip_market1501.pt  # Required ReID model
├── app/                    # Application source
└── tests/                  # Test suite
```

This system provides full person tracking capabilities with CPU fallback for development and GPU acceleration for production workloads.