# SpotOn Backend

**Status**: ✅ **Production Ready** (Phase 11 Complete)

Multi-camera person tracking system with real-time AI detection and analytics.

## 🎉 Phase 11: Final Production Enablement - COMPLETED

The SpotOn backend has reached **100% production readiness** with comprehensive security hardening, performance monitoring, and full API enablement.

### ✨ New Features Added

- **🔐 Security Hardening**: Comprehensive security middleware, rate limiting, CORS protection
- **📊 Performance Monitoring**: Real-time system metrics, health monitoring, maintenance endpoints
- **🔧 Production Configuration**: Environment-based endpoint control, security settings
- **🛡️ Authentication**: Full JWT implementation with role-based access control
- **📈 Analytics Endpoints**: Real-time metrics, behavior analysis, system statistics
- **📦 Export Capabilities**: Data export, report generation, video export with overlays
- **🔍 System Monitoring**: Performance dashboard, diagnostics, alerts, maintenance operations

## Quick Start

### Requirements
- Docker & Docker Compose
- Model weights: Download [clip_market1501.pt](https://drive.google.com/uc?id=1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7) (~600MB) to `./weights/`

### Start the System
```bash
# Backend only (fastest for frontend integration)
docker compose up -d backend

# Full stack (backend + Redis + TimescaleDB)
docker compose --profile infra up --build -d

# CPU stack (profiled compose)
docker compose -f docker-compose.cpu.yml --profile infra up --build -d

# GPU stack (requires NVIDIA GPU)
docker compose -f docker-compose.gpu.yml --profile infra up --build -d

# Health check
curl http://localhost:3847/health
```

### Database Setup (optional)
```bash
# If DB is enabled and TimescaleDB is available, run migrations:
python scripts/setup_db.py

# DB-specific readiness probe
curl http://localhost:3847/health/database
```

## API Endpoints

### Core System
```bash
GET  /health                           # System health and model status
GET  /                                 # Welcome message
```

### Processing Control
```bash
POST /api/v1/detection-processing-tasks/start    # Start detection session
GET  /api/v1/detection-processing-tasks/{id}/status  # Get task status
DELETE /api/v1/detection-processing-tasks/{id}/stop  # Stop task
```

### Real-time Data
```bash
# WebSocket connections for live data
ws://localhost:3847/ws/tracking/{task_id}      # Live tracking updates
ws://localhost:3847/ws/frames/{task_id}        # Live camera frames
ws://localhost:3847/ws/system                  # System status updates
```

### Media & Images
```bash
GET  /api/v1/media/frames/{task_id}/{camera}   # Camera frame with overlays
GET  /api/v1/media/persons/{person_id}/image   # Cropped person image
GET  /api/v1/media/frames/{task_id}/raw        # Raw camera frame
```

### Analytics & Tracking
```bash
GET  /api/v1/analytics/real-time/metrics       # Live person counts
GET  /api/v1/analytics/real-time/active-persons # Currently tracked people
GET  /api/v1/analytics/system/statistics       # System performance stats
```

### Environment Management
```bash
GET  /api/v1/environments                       # Available environments (Campus, Factory)
GET  /api/v1/environments/{env_id}/cameras      # Cameras in environment
GET  /api/v1/environments/{env_id}/date-ranges  # Available data dates
```

### Data Export
```bash
POST /api/v1/export/tracking-data               # Export tracking data (CSV/JSON)
POST /api/v1/export/analytics-report            # Export analytics report
POST /api/v1/export/video-with-overlays         # Export video with person boxes
GET  /api/v1/export/jobs/{job_id}/status        # Check export job status
GET  /api/v1/export/jobs/{job_id}/download      # Download completed export
```

### 🆕 Authentication & Security
```bash
POST /api/v1/auth/login                         # User authentication
GET  /api/v1/auth/me                           # Get current user info  
POST /api/v1/auth/refresh                      # Refresh JWT token
POST /api/v1/auth/logout                       # Logout user
GET  /api/v1/auth/health                       # Authentication service health
```

### 🆕 System Monitoring & Performance
```bash
GET  /api/v1/system/performance/dashboard       # Comprehensive performance dashboard
GET  /api/v1/system/performance/metrics        # Real-time system metrics
GET  /api/v1/system/performance/history        # Historical performance data
GET  /api/v1/system/health/comprehensive       # Detailed system health status  
GET  /api/v1/system/diagnostics                # System diagnostics information
GET  /api/v1/system/alerts                     # Current system alerts
POST /api/v1/system/maintenance/clear-cache    # Clear system caches
POST /api/v1/system/maintenance/garbage-collect # Force garbage collection
```

### Authentication
```bash
POST /api/v1/auth/login                         # Login (returns JWT token)
GET  /api/v1/auth/me                           # User profile
GET  /api/v1/auth/permissions/test             # Check permissions
```

## Usage Examples

### Start Tracking Session
```bash
curl -X POST http://localhost:3847/api/v1/detection-processing-tasks/start \
  -H "Content-Type: application/json" \
  -d '{
    "environment_id": "campus",
    "scene_id": "main_entrance", 
    "camera_ids": ["c09", "c12", "c13", "c16"]
  }'
```

### Get Live Analytics
```bash
curl http://localhost:3847/api/v1/analytics/real-time/metrics
# Returns: {"active_persons": 5, "total_cameras": 4, "processing_fps": 12.3}
```

### Export Tracking Data
```bash
curl -X POST http://localhost:3847/api/v1/export/tracking-data \
  -H "Content-Type: application/json" \
  -d '{
    "environment_id": "campus",
    "start_time": "2024-01-01T10:00:00Z",
    "end_time": "2024-01-01T11:00:00Z",
    "format": "csv",
    "camera_ids": ["c09", "c12"]
  }'
```

## WebSocket Data Format

### Tracking Updates
```json
{
  "type": "tracking_update",
  "payload": {
    "global_frame_index": 123,
    "timestamp": "2024-01-01T10:30:05.456Z",
    "cameras": {
      "c09": {
        "frame_image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
        "tracks": [{
          "track_id": 5,
          "global_id": "person-uuid-abc-123",
          "bbox_xyxy": [110.2, 220.5, 160.0, 330.8],
          "confidence": 0.92,
          "map_coords": [12.3, 45.6]
        }]
      }
    },
    "person_count_per_camera": {"c09": 3, "c12": 1}
  }
}
```

## System Configuration

### Performance Settings
```bash
# .env file settings
TARGET_FPS=3                           # Processing frame rate
FRAME_JPEG_QUALITY=70                  # Image quality (1-100)
DETECTION_CONFIDENCE_THRESHOLD=0.7     # Detection sensitivity
```

### Camera Configuration
```bash
# Available environments and cameras
Campus: c09, c12, c13, c16
Factory: c01, c02, c03, c04
```

## Troubleshooting

### Check System Status
```bash
# View logs
docker-compose logs -f backend

# Check health
curl http://localhost:3847/health

# Monitor resources
docker stats
```

### Common Issues
- **Slow performance**: Lower TARGET_FPS in .env file
- **Out of memory**: Reduce FRAME_JPEG_QUALITY or restart containers
- **Model not loaded**: Ensure clip_market1501.pt is in ./weights/ directory

## Development

### Local Setup
```bash
# Python environment
uv venv .venv --python 3.9
source .venv/bin/activate
uv pip install ".[dev]"

# Run locally (requires Redis/TimescaleDB)
uvicorn app.main:app --reload

# Run tests
pytest
```

### API Documentation
- **Swagger UI**: http://localhost:3847/docs
- **ReDoc**: http://localhost:3847/redoc
