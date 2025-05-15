# SpotOn Backend System Design

## 1. Introduction

This document outlines the high-level architecture for the SpotOn backend system. SpotOn is an intelligent multi-camera person tracking and analytics system designed for retrospective analysis of recorded video feeds. The backend is responsible for processing video data, performing AI-driven detection, tracking, re-identification, and serving this information to a frontend client.

The primary goals of this backend design are:
*   **Modularity:** Components should be loosely coupled and independently manageable.
*   **Scalability:** The system should be able to handle increasing data loads and user requests.
*   **Maintainability:** Code should be clean, well-documented, and easy to understand and modify.
*   **Reusability:** Leverage established design patterns and libraries effectively.
*   **Robustness:** Implement proper error handling and logging.
*   **Efficient Startup:** Minimize "cold start" latency by performing intensive initializations during application startup.

## 2. High-Level Architecture

The backend follows a service-oriented architecture, containerized using Docker and intended for cloud deployment (potentially orchestrated by Kubernetes).

Key backend responsibilities:
1.  **API Endpoints:** Expose RESTful APIs for control, configuration, and historical data retrieval.
2.  **WebSocket Communication:** Provide real-time *tracking metadata updates* to the frontend.
3.  **Data Processing Pipeline:**
    *   Fetch video segments (sub-videos) from S3 and cache them locally, processed sequentially one batch of sub-videos (one per camera for a given time index) at a time.
    *   Locally extract individual frames from the cached video segments.
    *   Perform object detection and intra-camera tracking on each frame (e.g., Faster R-CNN + BotSort via BoxMOT).
    *   Extract appearance features for Re-ID from detected persons (e.g., CLIP via BoxMOT).
    *   Execute cross-camera Re-Identification and global ID association.
    *   Apply perspective transformation (Homography) to get map coordinates using pre-calculated matrices.
    *   Compile tracking results (bounding boxes, global IDs, map coordinates). *Client handles video display and overlays based on this metadata.*
4.  **Data Management:**
    *   Interact with Redis for caching real-time state and recent Re-ID embeddings.
    *   Interact with TimescaleDB for storing historical tracking events and older embeddings.
5.  **Application Initialization (Startup Phase):**
    *   **Eager loading of AI models:** The primary detection model (e.g., FasterRCNN) and the core Re-ID model (e.g., `clip_market1501.pt`, potentially via a prototype tracker instance) are loaded into memory/GPU.
    *   **Model warm-up:** Dummy data is passed through loaded models to compile any JIT kernels and prepare them for inference.
    *   **Homography pre-computation:** Homography matrices for all configured cameras are calculated from their respective point files and cached.
    This ensures that the system is immediately ready to process requests with minimal latency once the application is reported as "ready."

## 3. Backend Directory Structure

The backend codebase is organized as follows:

*   `spoton_backend/`: Root directory for the backend.
    *   `app/`: Main application package.
        *   `api/`: Contains API endpoint definitions using FastAPI routers.
            *   `v1/`: For API versioning (e.g., `/api/v1/...`).
                *   `endpoints/`: Specific modules for different resource groups (e.g., processing, analytics).
                *   `schemas.py`: Pydantic models defining request/response data structures, ensuring validation.
            *   `websockets.py`: Manages WebSocket connections and message broadcasting.
        *   `core/`: Core application setup.
            *   `config.py`: Manages application settings, loaded from environment variables (Pydantic `BaseSettings`).
            *   `event_handlers.py`: Logic for application lifecycle events. Specifically, `on_startup` handlers are responsible for **eagerly loading and initializing core AI models (detector, Re-ID models via a prototype tracker or shared component), pre-calculating and caching all homography matrices, and performing model warm-ups.** This shifts initialization overhead from the first request to the application startup phase. `on_shutdown` handles resource cleanup.
        *   `models/`: Wrappers and interfaces for AI models.
            *   `base_models.py`: Abstract base classes for AI components (Detector, Tracker, FeatureExtractor) to implement the Strategy Pattern.
            *   `detectors.py`, `trackers.py`, `feature_extractors.py`: Concrete implementations of the AI strategies, likely wrapping libraries like BoxMOT. Models are designed to be loaded during application startup.
        *   `services/`: Business logic layer. Each service encapsulates a specific domain of functionality.
            *   `pipeline_orchestrator.py`: Orchestrates the sequence of operations in the core processing pipeline for each video segment and its frames.
            *   `reid_service.py` (or `reid_components.py`): Manages the Re-ID process, including embedding comparison and gallery management (Redis/TimescaleDB). Relies on pre-loaded Re-ID models.
            *   `storage_service.py`: Implements the Repository/Data Access Object (DAO) pattern for interacting with S3 (fetching video segments, managing local cache), Redis, and TimescaleDB.
            *   `homography_service.py` (or similar component, e.g., integrated into `MultiCameraFrameProcessor`): Handles homography transformations using pre-calculated matrices loaded at startup.
            *   `notification_service.py`: Responsible for sending data to connected frontend clients via WebSockets.
            *   `camera_tracker_factory.py`: Manages per-camera tracker instances, potentially aiding in the startup pre-loading of shared Re-ID models by initializing a prototype tracker.
        *   `tasks/`: For background, potentially long-running processing tasks.
            *   `frame_processor.py`: Contains the detailed logic for processing a single frame (extracted from a video segment) through the detection, tracking, Re-ID pipeline. This can be invoked by the `PipelineOrchestratorService`.
        *   `utils/`: Common utility functions (e.g., image manipulation, timestamp conversions, video processing utilities).
        *   `main.py`: The entry point for the FastAPI application, where the app instance is created and routers are included. Its `lifespan` manager triggers the `on_startup` events.
        *   `dependencies.py`: Defines FastAPI dependencies for injecting service instances, database sessions, etc., into endpoint handlers. These services expect their underlying models/data to be pre-loaded.
    *   `tests/`: Contains all unit and integration tests, mirroring the `app` structure.
    *   `.env.example`: Template for environment variables.
    *   `Dockerfile`: Instructions to build the Docker image for the backend.
    *   `docker-compose.yml`: For local development, sets up the backend service along with dependencies like Redis and TimescaleDB.
    *   `pyproject.toml`: Project metadata and Python package dependencies.
    *   `README.md`: General information about the backend project.
    *   `DESIGN.md`: This document.

## 4. Core Technologies

*   **Programming Language:** Python 3.9+
*   **Web Framework:** FastAPI (for REST APIs and WebSocket handling, leveraging Starlette and Pydantic)
*   **AI/CV Libraries:** PyTorch, OpenCV (for video processing and frame extraction), BoxMOT (for detection, tracking, Re-ID feature extraction)
*   **Async HTTP Client:** `aiobotocore` (for S3 video segment fetching), `httpx` (if needed for other async calls)
*   **Cache:** Redis (for real-time state, Re-ID gallery)
*   **Database:** TimescaleDB on PostgreSQL (for historical data, with `pgvector` for embedding similarity search)
*   **Containerization:** Docker
*   **Orchestration (Recommended for Prod):** Kubernetes (or Docker Compose for simpler setups/dev)
*   **Dependency Management:** Poetry or PDM (as per `pyproject.toml`)

## 5. Design Patterns Employed

*   **Strategy Pattern:**
    *   **Usage:** For AI model components (detection, tracking, feature extraction). Abstract base classes (e.g., `AbstractDetector`, `AbstractTracker`, `AbstractFeatureExtractor`) define a common interface. Concrete classes (e.g., `FasterRCNNDetector`, `BotSortTracker`, `CLIPExtractor`) implement these interfaces. These strategies are instantiated and their models pre-loaded at application startup.
    *   **Benefit:** Allows easy swapping or addition of new AI models without altering the core pipeline logic.
*   **Factory Pattern (or Abstract Factory):**
    *   **Usage:** To create instances of AI model strategies (if needed beyond simple instantiation at startup) or per-camera trackers (`CameraTrackerFactory`). The factory may also play a role in the startup sequence by creating a "prototype" tracker to ensure shared Re-ID models are loaded.
    *   **Benefit:** Decouples client code from concrete model classes, centralizes instantiation logic.
*   **Service Layer Pattern:**
    *   **Usage:** Business logic is encapsulated in service classes (e.g., `PipelineOrchestratorService`, `ReIDService`, `StorageService`). API handlers delegate to these services.
    *   **Benefit:** Promotes Separation of Concerns (SoC), improves code organization and testability.
*   **Repository Pattern (or Data Access Object - DAO):**
    *   **Usage:** The `StorageService` acts as a facade for data access related to S3 (video segments, local caching), Redis, and TimescaleDB.
    *   **Benefit:** Abstracts data storage details.
*   **Observer Pattern (Conceptual, via WebSockets):**
    *   **Usage:** Backend processes data (subject), frontend clients (observers) receive real-time *metadata* updates via WebSockets from `NotificationService`.
    *   **Benefit:** Enables real-time data streaming for client-side rendering and synchronization.
*   **Pipeline Pattern (Conceptual):**
    *   **Usage:** The core per-frame processing (Frame Extraction -> Preprocess -> Detect -> Track -> Feature Extract -> Re-ID -> Transform) is a pipeline managed by `PipelineOrchestratorService` and `FrameProcessorTask` (or `MultiCameraFrameProcessor`).
    *   **Benefit:** Structures complex processing into manageable steps.
*   **Dependency Injection (via FastAPI):**
    *   **Usage:** FastAPI's dependency injection provides services, database connections, etc., to components. These dependencies (like model wrappers or services holding pre-computed data) are available immediately after application startup.
    *   **Benefit:** Promotes loose coupling and testability.
*   **Singleton Pattern (via FastAPI Dependencies & Application State):** Core services, AI model instances (detector, Re-ID model handler), and pre-computed data (like all homography matrices) are managed as singletons, initialized during the application's startup phase and reused throughout its lifecycle.

## 6. Communication Protocols

*   **RESTful APIs (HTTP/S):**
    *   **Purpose:** Client-initiated request-response interactions.
    *   **Examples:**
        *   Starting a processing session for specified cameras within an environment.
        *   Fetching historical analytics data.
        *   Retrieving system configuration (e.g., available environments, cameras per environment).
    *   **Technology:** FastAPI.
*   **WebSockets:**
    *   **Purpose:** Real-time, bidirectional communication for backend to push updates.
    *   **Examples:**
        *   Streaming per-frame *tracking metadata* (bounding boxes, global IDs, map coordinates, frame timestamps) to the frontend. *The client separately fetches and plays the video stream (e.g., from S3) and uses this metadata to render overlays.*
    *   **Technology:** FastAPI's WebSocket support.

## 7. Key API Endpoints (High-Level)

All endpoints will be versioned (e.g., `/api/v1/...`). Pydantic schemas will define request and response bodies.

*   **Processing Tasks (`/processing_tasks`):**
    *   `POST /start`: Initiates a retrospective analysis task for specified cameras in an environment.
        *   Request: `{ "cameras": ["c01", "c02"], "environment_id": "campus" }`
        *   Response: `{ "task_id": "uuid", "status_url": "/api/v1/processing_tasks/{task_id}/status", "websocket_url": "/ws/tracking/{task_id}" }`
    *   `GET /{task_id}/status`: Retrieves the current status of a processing task.
    *   `POST /{task_id}/control`: Send control commands (e.g., pause, resume - if supported, focus_track).

*   **Analytics Data (`/analytics_data`):**
    *   `GET /trajectory/{global_person_id}`: Get historical trajectory for a person.
    *   `GET /heatmap`: Get data for generating a heatmap.

*   **Configuration/Metadata (`/config`):**
    *   `GET /environments`: List available environments.
    *   `GET /environments/{environment_id}/cameras`: List cameras available for a given environment.

*   **WebSocket Endpoint (`/ws/tracking/{task_id}`):**
    *   The backend pushes messages with the structure defined in `app.api.v1.schemas.WebSocketTrackingMessage`.

## 8. Data Flow (Retrospective Analysis - Using Pre-Split Sub-Videos)

This describes the flow once a processing task is initiated, assuming models and homography matrices are pre-loaded at application startup.

1.  **S3 Data Preparation (One-Time):**
    *   For each environment and its cameras, source videos are pre-split into sub-video segments (e.g., `sub_video_01.mp4` to `sub_video_04.mp4`).
    *   These sub-videos are stored in S3 with a defined naming convention.

2.  **Initiation (REST API):**
    *   Frontend sends a request to `POST /api/v1/processing_tasks/start`.
    *   The backend responds with task details, including the WebSocket URL.

3.  **Task Creation & Sub-Video Processing Loop (`PipelineOrchestratorService`):**
    *   Backend creates a processing task with a unique `task_id`.
    *   The system determines the total number of sub-video batches to process for the environment (e.g., if max sub-videos per camera is 4, then 4 batches).
    *   The `PipelineOrchestratorService` enters a loop, iterating through sub-video indices (e.g., `idx = 0, 1, 2, 3`).
        *   **(a) Download Current Sub-Video Batch (e.g., for index `idx`):**
            *   `VideoDataManagerService` downloads the sub-video corresponding to the current `idx` (e.g., `sub_video_{idx+1}.mp4`) for *all relevant cameras* in the specified environment.
            *   These videos are saved to a local, task-specific cache directory. This download for the current batch `idx` completes before processing of this batch begins.
        *   **(b) Process Current Sub-Video Batch:**
            *   A `BatchedFrameProvider` is created for the just-downloaded set of sub-videos (all having the same index `idx`).
            *   The system then processes frames from this batch:
                *   **(Frame Extraction):** `BatchedFrameProvider` reads synchronized frames (one from each sub-video in the current batch).
                *   **(Per-Frame AI Processing - `MultiCameraFrameProcessor`):** For each such multi-camera frame batch:
                    *   Detection, Tracking, Feature Extraction, Re-ID, and Homography transformation (using pre-loaded models and pre-calculated matrices) are performed.
                *   **(Real-time Metadata Update):** `NotificationService` sends tracking metadata via WebSocket.
            *   This inner frame processing continues until all frames from the current batch of sub-videos (index `idx`) are exhausted.
            *   The `BatchedFrameProvider` for batch `idx` is then closed.
        *   The loop then proceeds to the next `sub_video_idx` (e.g., `idx+1`), and steps (a) and (b) repeat for the next set of sub-videos.

4.  **Client-Side Video Access:**
    *   The frontend client, based on task details, constructs S3 URLs for relevant sub-video segments and handles their playback.

5.  **Data Storage & Caching (Backend - Ongoing):**
    *   `StorageService` (or `ReIDService`) updates Redis with recent Re-ID embeddings.
    *   `StorageService` (asynchronously) writes historical tracking data to TimescaleDB.

6.  **Client-Side Video Playback and Overlay Rendering:**
    *   Client plays video segments and uses WebSocket metadata to render synchronized overlays.

7.  **Task Completion:**
    *   Once all sub-video batches for all requested cameras are processed, the task is marked as complete.
    *   The local cache for the completed `task_id` is cleared by `StorageService`.

## 9. Scalability and Maintainability

*   **Scalability:**
    *   Stateless FastAPI application instances can be horizontally scaled.
    *   AI models and homography data are pre-loaded, improving response consistency.
    *   Video processing (frame extraction) is local to the worker after download.
    *   WebSocket communication is lightweight (metadata only).
    *   Video delivery to clients can leverage standard HTTP streaming and CDNs.
    *   Redis and TimescaleDB are scalable.
*   **Maintainability:**
    *   Modular design with clear service responsibilities.
    *   Consistent coding style, type hinting, and comprehensive testing.
    *   Detailed logging.
    *   Dependency injection.
    *   Centralized initialization logic in `on_startup` improves predictability.

## 10. Next Steps / Considerations

*   **Authentication & Authorization.**
*   **Detailed Error Handling & Retry Mechanisms.**
*   **Resource Management:** For GPU resources and local disk space. Monitor Docker startup time and resource consumption due to eager loading.
*   **CI/CD Pipeline.**
*   **Frame Timestamps:** Precise calculation and communication for client-side synchronization.
*   **Client-Side Video Rendering Strategy and Implementation.**
*   **Task Concurrency and Resource Allocation.**
*   **Explicit Prefetching of Sub-Video Batches:** While the current flow is sequential per batch, a future optimization could involve initiating the download of the *next* batch of sub-videos (`idx+1`) while the current batch (`idx`) is still undergoing frame-by-frame processing. This would further hide S3 download latency.