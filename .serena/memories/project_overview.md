# SpotOn Backend Project Overview

## Purpose
SpotOn is an intelligent multi-camera person tracking and analytics system designed for real-time surveillance and monitoring. The backend provides AI-powered person detection, cross-camera re-identification, and spatial mapping capabilities for security and operational analysis.

## Current Status
✅ **Production Ready** (Phase 11 Complete) - The backend has reached 100% production readiness with comprehensive security hardening, performance monitoring, and full API enablement.

## Tech Stack

### Core Framework
- **FastAPI** - Modern async web framework with automatic API documentation
- **Python 3.9+** - Primary programming language
- **Pydantic** - Data validation and settings management
- **Uvicorn** - ASGI server for production deployment

### AI/ML Components  
- **PyTorch** - Deep learning framework for model inference
- **torchvision** - Computer vision models and utilities
- **BoxMOT** - Multi-object tracking library
- **FAISS** - Efficient similarity search for re-identification
- **ultralytics** - YOLOv8 implementation
- **transformers** - CLIP model support for feature extraction
- **OpenCV** - Video processing and frame manipulation

### Storage & Caching
- **Redis** - Real-time data caching and session management
- **TimescaleDB** - PostgreSQL extension for time-series data
- **PostgreSQL** - Primary database with vector search capabilities
- **AWS S3** - Object storage for video data

### Communication
- **WebSockets** - Real-time communication for live tracking updates
- **aiohttp** - Async HTTP client for external services
- **websockets** - WebSocket server implementation

### Security & Authentication
- **PyJWT** - JSON Web Token handling
- **cryptography** - Cryptographic operations
- **aioredis** - Async Redis client for JWT blacklisting

### Monitoring & Analytics
- **psutil** - System monitoring
- **matplotlib** - Data visualization
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Analytics clustering (DBSCAN)

### Development Tools
- **pytest** - Testing framework with async support
- **ruff** - Modern Python linter and formatter
- **Docker** - Containerization for deployment

## Architecture Overview

### Domain-Driven Design
The codebase follows domain-driven design principles with clear separation of concerns:

- **Domains** (`app/domains/`) - Business logic organized by capabilities
- **Services** (`app/services/`) - Application services and orchestration
- **Infrastructure** (`app/infrastructure/`) - External dependencies and adapters
- **API** (`app/api/`) - HTTP endpoints and WebSocket handlers
- **Models** (`app/models/`) - AI model implementations

### Key Design Patterns
- **Strategy Pattern** - Interchangeable AI models (detector, tracker, feature extractor)
- **Factory Pattern** - `CameraTrackerFactory` for per-camera instances
- **Service Layer** - Business logic encapsulation  
- **Dependency Injection** - FastAPI's dependency system
- **Pipeline Pattern** - Sequential processing in orchestration layer

### Core Data Flow
1. **Video Processing** - Download from S3 → Extract frames → AI detection/tracking
2. **Cross-Camera Re-ID** - CLIP feature extraction → FAISS similarity search
3. **Spatial Mapping** - Homography transformation for map coordinates
4. **Real-time Streaming** - WebSocket updates to frontend clients
5. **Data Storage** - Redis caching + TimescaleDB historical storage

## Key Features

### AI Capabilities
- Multi-camera person detection using RT-DETR or YOLO
- Cross-camera person re-identification using CLIP embeddings
- Real-time multi-object tracking with BotSort algorithm
- Spatial coordinate transformation using homography matrices

### API Endpoints
- **Processing Control** - Start/stop tracking sessions
- **Real-time Data** - WebSocket streams for live updates
- **Analytics** - Performance metrics and system statistics
- **Authentication** - JWT-based security with role-based access
- **Media Serving** - Camera frames with tracking overlays
- **Data Export** - CSV/JSON export and video generation

### Production Features
- Comprehensive security middleware with rate limiting
- Performance monitoring and health checks
- Environment-based configuration management
- Docker containerization with CPU/GPU support
- Comprehensive test coverage (80%+ requirement)

## Development Environment
The project supports both local development and containerized deployment with automatic dependency management using `uv` package manager.