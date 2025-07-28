# SpotOn Backend System Design

## 1. Introduction

This document outlines the high-level architecture for the SpotOn backend system. SpotOn is an intelligent multi-camera person tracking and analytics system designed for **real-time analysis** of recorded video feeds. The backend is responsible for processing video data, performing AI-driven detection, tracking, re-identification, and streaming tracking metadata and frame images to a frontend client.

The primary goals of this backend design are:
*   **Performance:** Optimized for real-time processing at <10 FPS across 4 cameras with GPU acceleration
*   **Modularity:** Components should be loosely coupled and independently manageable
*   **Scalability:** The system should be able to handle increasing data loads and user requests
*   **Maintainability:** Code should be clean, well-documented, and easy to understand and modify
*   **Reusability:** Leverage established design patterns and libraries effectively
*   **Robustness:** Implement proper error handling, logging, and ensure graceful client interaction
*   **Efficient Startup:** Minimize "cold start" latency with GPU model pre-loading and warm-up

## 2. High-Level Architecture

The backend follows a service-oriented architecture, containerized using Docker and optimized for local development with GPU acceleration.

Key backend responsibilities:
1.  **API Endpoints:** Expose RESTful APIs for control, configuration, and historical data retrieval
2.  **Real-Time WebSocket Communication:** Provide optimized binary streaming of frame data with JSON metadata
3.  **GPU-Accelerated Processing Pipeline:**
    *   Fetch video segments (sub-videos) from S3 and cache them locally
    *   Extract individual frames from cached video segments for AI processing
    *   **GPU-accelerated object detection** using Faster R-CNN with CUDA support
    *   **GPU-accelerated tracking** with BotSort via BoxMOT on GPU
    *   **GPU-accelerated Re-ID feature extraction** using CLIP model on GPU
    *   Execute cross-camera Re-Identification with GPU-accelerated similarity search
    *   Apply perspective transformation (Homography) to get map coordinates
    *   **Binary frame encoding** and real-time streaming optimization
4.  **Optimized Data Management:**
    *   Redis for real-time state caching and frame buffering
    *   TimescaleDB for historical tracking events and analytics
    *   **GPU memory management** for efficient model switching and batch processing
5.  **GPU-Optimized Application Initialization:**
    *   **GPU model pre-loading:** All AI models loaded directly to GPU memory
    *   **CUDA warm-up:** GPU kernels pre-compiled with dummy inference runs
    *   **Memory optimization:** Efficient GPU memory allocation for 4-camera concurrent processing
    *   **Performance monitoring:** GPU utilization and memory usage tracking
6.  **Client Interaction During Startup and Processing:**
    *   The backend's startup phase (model loading, etc.) can take time. During this period, attempts to connect to WebSocket endpoints might be refused or fail.
    *   **Frontend Responsibility (Initial Connection):** The frontend should implement a robust connection strategy:
        1.  After initiating a task (via REST), the frontend receives a WebSocket URL.
        2.  Before attempting the WebSocket connection, or if an initial connection attempt fails, the frontend **must poll the backend's `/health` REST endpoint.**
        3.  The `/health` endpoint (see Section 7) reports the status of critical components (e.g., detector loaded, Re-ID model ready, homography service ready).
        4.  The frontend should only establish (or retry establishing) the WebSocket connection once the `/health` endpoint indicates that essential components are loaded and the backend is sufficiently ready (e.g., status is "healthy", or "degraded" but WebSocket-relevant components are up).
        5.  A retry mechanism with exponential backoff for WebSocket connection attempts, combined with `/health` checks, is recommended.
    *   **Frontend Responsibility (During Processing):** The frontend receives frame images and metadata bundled together in `tracking_update` messages, simplifying synchronization.

## 3. Backend Directory Structure

The backend codebase is organized as follows:

*   `spoton_backend/`: Root directory for the backend.
    *   `app/`: Main application package.
        *   `api/`: Contains API endpoint definitions using FastAPI routers.
            *   `v1/`: For API versioning (e.g., `/api/v1/...`).
                *   `endpoints/`: Specific modules for different resource groups (e.g., `processing_tasks.py`, `analytics_data.py`, `media.py` (potentially deprecated for primary use)).
                *   `schemas.py`: Pydantic models defining request/response data structures, including WebSocket message payloads.
            *   `websockets.py`: Manages WebSocket connections and message broadcasting.
        *   `core/`: Core application setup.
            *   `config.py`: Manages application settings, loaded from environment variables (Pydantic `BaseSettings`).
            *   `event_handlers.py`: Logic for application lifecycle events. Responsible for eager loading AI models, homography, and warm-ups.
        *   `models/`: Wrappers and interfaces for AI models.
        *   `services/`: Business logic layer.
            *   `pipeline_orchestrator.py`: Orchestrates the core processing pipeline. **Responsible for encoding processed frame images and packaging them with metadata for WebSocket transmission.**
            *   `reid_components.py`: Manages the Re-ID process.
            *   `video_data_manager_service.py`: Manages S3 interaction and local caching of sub-videos.
            *   `homography_service.py`: Manages homography matrices.
            *   `notification_service.py`: Sends `tracking_update` (now including frame images) and `status_update` messages via WebSockets.
        *   `utils/`: Common utility functions (e.g., image manipulation, timestamp conversions, **image encoding utilities**).
        *   `main.py`: FastAPI application entry point. Includes the `/health` endpoint.
    *   `tests/`: Unit and integration tests.
    *   `.env.example`: Template for environment variables.
    *   `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, `README.md`, `DESIGN.md`.

## 4. Core Technologies

*   **Programming Language:** Python 3.9+
*   **Web Framework:** FastAPI (for REST APIs and WebSocket handling)
*   **AI/CV Libraries:** 
    *   PyTorch with CUDA support for GPU acceleration
    *   OpenCV with GPU acceleration for video processing
    *   BoxMOT with GPU-enabled tracking
    *   FAISS-GPU for fast similarity search
*   **GPU Acceleration:** CUDA, cuDNN, TensorRT (optional optimization)
*   **Async HTTP Client:** `aiobotocore` (for S3), `httpx`
*   **Cache:** Redis with binary data optimization
*   **Database:** TimescaleDB on PostgreSQL
*   **Containerization:** Docker with GPU runtime support
*   **Binary Streaming:** WebSocket binary frames, optimized JPEG compression

## 5. Design Patterns Employed

*   **Strategy Pattern:** For AI model components (detector, tracker, feature extractor).
*   **Factory Pattern:** `CameraTrackerFactory` for per-camera tracker instances.
*   **Service Layer Pattern:** Encapsulates business logic (e.g., `PipelineOrchestratorService`).
*   **Repository Pattern (Conceptual):** `VideoDataManagerService` for data access.
*   **Observer Pattern (Conceptual, via WebSockets):** Backend (subject) pushes frame image and metadata updates to frontend clients (observers).
*   **Pipeline Pattern (Conceptual):** Core processing managed by `PipelineOrchestratorService`.
*   **Dependency Injection (FastAPI):** Provides services and pre-loaded components.
*   **Singleton Pattern (FastAPI Dependencies & App State):** Core services, AI models, pre-computed data managed as singletons.

## 6. Communication Protocols

*   **RESTful APIs (HTTP/S):**
    *   **Purpose:** Client-initiated request-response and configuration
    *   **Examples:**
        *   Starting a processing session
        *   Fetching historical analytics
        *   **`/health` endpoint:** Reports GPU status and backend readiness
    *   **Technology:** FastAPI
*   **Binary WebSocket Streaming:**
    *   **Purpose:** High-performance real-time streaming optimized for <10 FPS
    *   **Key Message Types:**
        *   `tracking_update`: **Binary frame data** + JSON metadata in separate messages
            *   Binary message: Raw JPEG bytes for each camera
            *   JSON message: Tracking metadata (bounding boxes, global IDs, map coordinates)
        *   `status_update`: Reports GPU utilization and processing performance
        *   `frame_sync`: Synchronization signals for multi-camera coordination
    *   **Performance Optimizations:**
        *   Binary frame transmission (no base64 encoding overhead)
        *   Adaptive JPEG compression based on network conditions
        *   Frame skipping for real-time performance
        *   GPU-to-WebSocket direct pipeline
    *   **Technology:** FastAPI WebSocket with binary message support
    *   **Client Connection Strategy:**
        1.  Client initiates task via REST, receives WebSocket URL
        2.  Client polls `/health` for GPU readiness
        3.  Client connects with binary message handler and frame synchronization

## 7. Key API Endpoints (High-Level)

All endpoints `/api/v1/...` unless specified.

*   **Processing Tasks (`/processing-tasks`):**
    *   `POST /start`: Initiates analysis. Response: `{ "task_id", "status_url", "websocket_url" }`.
    *   `GET /{task_id}/status`: Retrieves task status.
*   **Analytics Data (`/analytics_data`):** (Endpoints for historical data)
*   **Configuration/Metadata (`/config`):** (Endpoints for environments, cameras)
*   **Media Endpoints (`/media`):**
    *   `GET /tasks/{task_id}/environments/{environment_id}/cameras/{camera_id}/sub_videos/{sub_video_filename}`: **This endpoint is now considered deprecated for primary real-time frontend display.** It may be retained for debugging, direct sub-video downloads, or alternative use cases but is not part of the main frame-by-frame display flow.
*   **Health Endpoint (`/health`):**
    *   `GET /health`: Provides backend status. Crucial for frontend before WebSocket connection.
*   **WebSocket Endpoint (`/ws/tracking/{task_id}`):**
    *   Refer to Section 6 for key message types. `tracking_update` is now the primary carrier of both metadata and visual frame data.

## 8. GPU-Optimized Data Flow and Real-Time Streaming

This describes the optimized flow for <10 FPS real-time processing:

1.  **S3 Data Preparation:** Source videos pre-split into sub-videos and stored in S3
2.  **Initiation (REST API):** Frontend sends `POST /api/v1/processing_tasks/start`. Backend responds with `task_id`, `websocket_url`
3.  **GPU Readiness Check:** Frontend polls `/health` for GPU model loading status and memory availability
4.  **Real-Time Processing Pipeline (`PipelineOrchestratorService` - Backend):**
    *   **GPU-Accelerated Frame Processing:**
        *   `VideoDataManagerService` streams sub-videos to GPU memory
        *   `MultiCameraFrameProcessor` processes 4 cameras concurrently on GPU
        *   **GPU Pipeline:** Detection → Tracking → Re-ID → Coordinate Transform
        *   **Binary Frame Encoding:** Direct GPU-to-JPEG encoding without CPU transfer
    *   **Real-Time Streaming Strategy:**
        *   **Frame Skipping:** Drop frames if processing falls behind target FPS
        *   **Adaptive Quality:** Reduce JPEG quality under high load
        *   **Binary WebSocket Messages:**
            ```python
            # Binary message for each camera
            binary_frame_data = {
                "camera_id": "c01",
                "frame_index": 123,
                "jpeg_data": raw_jpeg_bytes  # No base64 encoding
            }
            
            # JSON metadata message
            metadata = {
                "type": "tracking_update",
                "global_frame_index": 123,
                "scene_id": "campus",
                "timestamp_processed_utc": "...",
                "cameras": {
                    "c01": {
                        "tracks": [/* TrackedPersonData objects */]
                    }
                }
            }
            ```
5.  **Frontend Real-Time Rendering:**
    *   **Binary Message Handler:** Receives raw JPEG bytes, creates image blobs
    *   **Immediate Display:** No buffering, displays frames as received
    *   **Frame Synchronization:** Uses frame_index for multi-camera sync
    *   **Performance Monitoring:** Tracks FPS and frame drop rates
6.  **GPU Memory Management:** 
    *   **Model Persistence:** Keep models loaded in GPU memory
    *   **Batch Processing:** Process multiple cameras simultaneously
    *   **Memory Monitoring:** Track GPU memory usage and optimize allocation
7.  **Performance Optimization:**
    *   **Target FPS:** <10 FPS with frame skipping if needed
    *   **Latency:** <100ms from frame capture to WebSocket transmission
    *   **Throughput:** Optimized for 4-camera concurrent processing

## 9. Performance and Maintainability

*   **Performance Optimizations:**
    *   **GPU-First Architecture:** All AI processing on GPU for maximum throughput
    *   **Binary Streaming:** Eliminates base64 encoding/decoding overhead
    *   **Reduced Network Load:** 40-60% reduction in WebSocket message size
    *   **Frame Skipping:** Maintains real-time performance under load
    *   **Concurrent Processing:** 4 cameras processed simultaneously on GPU
    *   **Memory Optimization:** Efficient GPU memory allocation and reuse
*   **Scalability Considerations:**
    *   **GPU Memory Limits:** Current design optimized for single GPU with 4 cameras
    *   **Network Bandwidth:** Binary streaming reduces bandwidth by ~50%
    *   **Real-Time Constraints:** <10 FPS target with adaptive quality control
    *   **Database Scaling:** Redis for real-time, TimescaleDB for historical data
*   **Maintainability:**
    *   **GPU Abstraction:** Clear separation between GPU and CPU operations
    *   **Performance Monitoring:** Built-in GPU utilization and memory tracking
    *   **Error Handling:** Graceful degradation when GPU memory is exhausted
    *   **Hot Reload:** Docker development environment with GPU support
    *   **Testing:** GPU-aware unit tests and performance benchmarks

## 10. Development and Deployment

*   **Local Development Environment:**
    *   **Docker GPU Support:** nvidia-docker for GPU acceleration in containers
    *   **Hot Reload:** FastAPI auto-reload with GPU model persistence
    *   **Performance Profiling:** Built-in GPU utilization monitoring
    *   **Development Database:** Simplified Redis + PostgreSQL setup
*   **GPU Requirements:**
    *   **Minimum:** NVIDIA GPU with 6GB VRAM (GTX 1060 or better)
    *   **Recommended:** NVIDIA RTX 30xx series with 8GB+ VRAM
    *   **CUDA Version:** 11.8+ with cuDNN 8.6+
*   **Performance Monitoring:**
    *   **GPU Metrics:** Memory usage, utilization, temperature
    *   **Inference Times:** Per-model performance tracking
    *   **Network Metrics:** WebSocket throughput and latency
    *   **Frame Metrics:** FPS, frame drops, processing delays
*   **Next Steps:**
    *   **GPU Memory Optimization:** Implement model quantization
    *   **TensorRT Integration:** Optimize inference performance
    *   **Multi-GPU Support:** Scale to multiple GPUs for higher throughput
    *   **Cloud GPU Deployment:** AWS/GCP GPU instance optimization
    *   **Performance Testing:** Load testing with multiple concurrent sessions