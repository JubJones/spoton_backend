# SpotOn Backend System Design

## 1. Introduction

This document outlines the high-level architecture for the SpotOn backend system. SpotOn is an intelligent multi-camera person tracking and analytics system designed for retrospective analysis of recorded video feeds. The backend is responsible for processing video data, performing AI-driven detection, tracking, re-identification, and pushing tracking metadata and frame images to a frontend client.

The primary goals of this backend design are:
*   **Modularity:** Components should be loosely coupled and independently manageable.
*   **Scalability:** The system should be able to handle increasing data loads and user requests.
*   **Maintainability:** Code should be clean, well-documented, and easy to understand and modify.
*   **Reusability:** Leverage established design patterns and libraries effectively.
*   **Robustness:** Implement proper error handling, logging, and ensure graceful client interaction, especially during backend startup.
*   **Efficient Startup:** Minimize "cold start" latency for core processing by performing intensive initializations during application startup.

## 2. High-Level Architecture

The backend follows a service-oriented architecture, containerized using Docker and intended for cloud deployment (potentially orchestrated by Kubernetes).

Key backend responsibilities:
1.  **API Endpoints:** Expose RESTful APIs for control, configuration, and historical data retrieval.
2.  **WebSocket Communication:** Provide real-time *tracking metadata and frame image updates* to the frontend and notify it about processing state.
3.  **Data Processing Pipeline:**
    *   Fetch video segments (sub-videos) from S3 and cache them locally, processed sequentially one batch of sub-videos (one per camera for a given time index) at a time.
    *   Locally extract individual frames from the cached video segments for AI processing.
    *   Perform object detection and intra-camera tracking on each frame (e.g., Faster R-CNN + BotSort via BoxMOT).
    *   Extract appearance features for Re-ID from detected persons (e.g., CLIP via BoxMOT).
    *   Execute cross-camera Re-Identification and global ID association.
    *   Apply perspective transformation (Homography) to get map coordinates using pre-calculated matrices.
    *   Compile tracking results (bounding boxes, global IDs, map coordinates) and encode frame images.
4.  **Data Management:**
    *   Interact with Redis for caching real-time state and recent Re-ID embeddings.
    *   Interact with TimescaleDB for storing historical tracking events and older embeddings.
5.  **Application Initialization (Startup Phase):**
    *   **Eager loading of AI models:** The primary detection model (e.g., FasterRCNN) and the core Re-ID model (e.g., `clip_market1501.pt`, typically by initializing a prototype tracker instance within the `CameraTrackerFactory`) are loaded into memory/GPU.
    *   **Model warm-up:** Dummy data is passed through loaded models to compile any JIT kernels and prepare them for inference.
    *   **Homography pre-computation:** Homography matrices for all configured cameras are calculated from their respective point files and cached by the `HomographyService`.
    This ensures that the system is immediately ready to process requests with minimal latency once the application is reported as "ready."
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
*   **AI/CV Libraries:** PyTorch, OpenCV (for video processing, frame extraction, **image encoding**), BoxMOT
*   **Async HTTP Client:** `aiobotocore` (for S3), `httpx`
*   **Cache:** Redis
*   **Database:** TimescaleDB on PostgreSQL
*   **Containerization:** Docker
*   **Image Encoding:** OpenCV (`cv2.imencode`), `base64` module.

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
    *   **Purpose:** Client-initiated request-response.
    *   **Examples:**
        *   Starting a processing session.
        *   Fetching historical analytics.
        *   **`/health` endpoint:** Polled by the frontend to check backend readiness.
    *   **Technology:** FastAPI.
*   **WebSockets:**
    *   **Purpose:** Real-time, bidirectional communication for backend to push updates.
    *   **Key Message Types:**
        *   `tracking_update`: Contains per-frame metadata (bounding boxes, global IDs, map coordinates, `global_frame_index`) AND **the base64 encoded frame image data for each camera**.
            *   `cameras.{camera_id}.frame_image_base64: Optional[str]` field added to the payload.
        *   `status_update`: Reports overall task progress and current operational step.
        *   **REMOVED:** `media_available` (no longer needed as images are in `tracking_update`).
        *   **REMOVED/INTERNAL:** `batch_processing_complete` (no longer directly relevant for frontend video synchronization).
    *   **Technology:** FastAPI's WebSocket support.
    *   **Client Connection Strategy:**
        1.  Client initiates a task via REST, receives WebSocket URL.
        2.  Client polls `/health` REST endpoint until backend is ready.
        3.  Client connects to WebSocket with retry logic.

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

## 8. Data Flow and Frontend Synchronization Strategy

This describes the flow once a processing task is initiated.

1.  **S3 Data Preparation (One-Time):** Source videos pre-split into sub-videos and stored in S3.
2.  **Initiation (REST API):** Frontend sends `POST /api/v1/processing_tasks/start`. Backend responds with `task_id`, `websocket_url`.
3.  **Frontend WebSocket Connection Attempt & Health Check:** Frontend polls `/health`. Once healthy, establishes WebSocket connection.
4.  **Task Creation & Sub-Video Batch Processing Loop (`PipelineOrchestratorService` - Backend):**
    *   The backend creates a task.
    *   Loop through sub-video batches (internal backend concept for fetching from S3):
        *   **(a) Backend Downloads Current Sub-Video Batch:** `VideoDataManagerService` downloads sub-videos for all relevant cameras to local cache.
        *   **(b) Backend Processes Frames and Streams Image+Metadata (Batch `idx`):**
            *   `MultiCameraFrameProcessor` processes frames from the locally downloaded sub-videos.
            *   For each processed multi-camera frame (or synchronized set of frames):
                *   The backend **encodes the raw frame image (e.g., to JPEG, then base64)** for each camera.
                *   `NotificationService` sends a `tracking_update` WebSocket message.
            *   **Payload of `tracking_update`:**
                ```json
                {
                  "type": "tracking_update",
                  "payload": {
                    "global_frame_index": 123,
                    "scene_id": "campus",
                    "timestamp_processed_utc": "...",
                    "cameras": {
                      "c01": {
                        "image_source": "000123.jpg", // Frame identifier
                        "frame_image_base64": "base64_encoded_jpeg_string_for_c01_frame_123", // << NEW
                        "tracks": [ /* list of TrackedPersonData objects for c01 */ ]
                      },
                      "c02": {
                        "image_source": "000123.jpg",
                        "frame_image_base64": "base64_encoded_jpeg_string_for_c02_frame_123", // << NEW
                        "tracks": [ /* list of TrackedPersonData objects for c02 */ ]
                      }
                      // ... other cameras
                    }
                  }
                }
                ```
        *   **(c) Frontend Receives Image+Metadata and Renders:**
            *   Frontend receives the `tracking_update` message.
            *   For each camera in the `cameras` dictionary:
                *   It decodes the `frame_image_base64` string into an image.
                *   Displays this image directly (this *is* the video frame).
                *   Uses the `tracks` data (bounding boxes, IDs, map_coords) from the *same message* to draw overlays on this displayed image.
            *   Synchronization is inherent because the image and its corresponding metadata arrive together.
        *   The loop then proceeds to the next sub-video batch internally on the backend.

5.  **Frontend Video Transition Management:** Not applicable in the same way. The frontend displays a continuous stream of images received via WebSockets. There are no separate "sub-video files" for the frontend to manage transitions between.
6.  **Data Storage & Caching (Backend - Ongoing):** Re-ID embeddings to Redis, historical data to TimescaleDB.
7.  **Task Completion:** Backend sends "COMPLETED" `status_update`. Backend cleans up cached sub-videos.

## 9. Scalability and Maintainability

*   **Scalability:**
    *   Stateless FastAPI application instances can be horizontally scaled.
    *   **Increased WebSocket Load:** Sending base64 encoded images significantly increases WebSocket message size and network bandwidth. This is a primary scalability concern.
        *   **Mitigation:**
            *   Optimize image encoding (JPEG quality, consider WebP if client support is good).
            *   Potentially offer scaled-down image resolutions for frontend display.
            *   Ensure WebSocket server infrastructure can handle larger message throughput.
    *   **Increased CPU Load:**
        *   Backend: Image encoding (e.g., `cv2.imencode`, `base64.b64encode`) adds CPU load per frame per camera.
        *   Frontend: Base64 decoding and image rendering add CPU load per frame per camera.
    *   Redis and TimescaleDB are scalable.
*   **Maintainability:**
    *   Modular design, clear service responsibilities.
    *   **Simplified Frontend Sync:** The new approach greatly simplifies frontend synchronization logic.
    *   Consistent coding style, type hinting, comprehensive testing.
    *   Detailed logging.
    *   Centralized initialization logic in `on_startup`.

## 10. Next Steps / Considerations

*   **Authentication & Authorization.**
*   **Detailed Error Handling & Retry Mechanisms.**
*   **Resource Management:** Monitor CPU, memory, and network for backend (especially image encoding) and WebSockets.
*   **Image Encoding Optimization:** Investigate optimal JPEG quality vs. size. Consider if frontend can request different resolutions/qualities.
*   **WebSocket Message Size Limits:** Be mindful of default limits in WebSocket libraries/proxies. May need configuration for larger image payloads.
*   **Client-Side Performance:** Ensure frontend can decode and render images at the target frame rate without lag.