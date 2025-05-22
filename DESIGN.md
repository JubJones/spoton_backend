# SpotOn Backend System Design

## 1. Introduction

This document outlines the high-level architecture for the SpotOn backend system. SpotOn is an intelligent multi-camera person tracking and analytics system designed for retrospective analysis of recorded video feeds. The backend is responsible for processing video data, performing AI-driven detection, tracking, re-identification, serving video data, and pushing tracking metadata to a frontend client.

The primary goals of this backend design are:
*   **Modularity:** Components should be loosely coupled and independently manageable.
*   **Scalability:** The system should be able to handle increasing data loads and user requests, including serving media files.
*   **Maintainability:** Code should be clean, well-documented, and easy to understand and modify.
*   **Reusability:** Leverage established design patterns and libraries effectively.
*   **Robustness:** Implement proper error handling, logging, and ensure graceful client interaction, especially during backend startup.
*   **Efficient Startup:** Minimize "cold start" latency for core processing by performing intensive initializations during application startup.

## 2. High-Level Architecture

The backend follows a service-oriented architecture, containerized using Docker and intended for cloud deployment (potentially orchestrated by Kubernetes).

Key backend responsibilities:
1.  **API Endpoints:** Expose RESTful APIs for control, configuration, historical data retrieval, and **serving video media**.
2.  **WebSocket Communication:** Provide real-time *tracking metadata updates* to the frontend and notify it about available media and processing state.
3.  **Data Processing Pipeline:**
    *   Fetch video segments (sub-videos) from S3 and cache them locally, processed sequentially one batch of sub-videos (one per camera for a given time index) at a time.
    *   Locally extract individual frames from the cached video segments for AI processing.
    *   Perform object detection and intra-camera tracking on each frame (e.g., Faster R-CNN + BotSort via BoxMOT).
    *   Extract appearance features for Re-ID from detected persons (e.g., CLIP via BoxMOT).
    *   Execute cross-camera Re-Identification and global ID association.
    *   Apply perspective transformation (Homography) to get map coordinates using pre-calculated matrices.
    *   Compile tracking results (bounding boxes, global IDs, map coordinates).
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
    *   **Frontend Responsibility (During Processing):** The frontend must gracefully handle the timing differences between receiving `media_available` notifications (for new video batches) and the arrival of corresponding `tracking_update` metadata. It should buffer video playback or metadata as needed to ensure synchronization.

## 3. Backend Directory Structure

The backend codebase is organized as follows:

*   `spoton_backend/`: Root directory for the backend.
    *   `app/`: Main application package.
        *   `api/`: Contains API endpoint definitions using FastAPI routers.
            *   `v1/`: For API versioning (e.g., `/api/v1/...`).
                *   `endpoints/`: Specific modules for different resource groups (e.g., `processing_tasks.py`, `analytics_data.py`, **`media.py`**).
                *   `schemas.py`: Pydantic models defining request/response data structures, including WebSocket message payloads.
            *   `websockets.py`: Manages WebSocket connections and message broadcasting.
        *   `core/`: Core application setup.
            *   `config.py`: Manages application settings, loaded from environment variables (Pydantic `BaseSettings`).
            *   `event_handlers.py`: Logic for application lifecycle events. Specifically, `on_startup` handlers are responsible for **eagerly loading and initializing core AI models (detector, Re-ID models via a prototype tracker or shared component), pre-calculating and caching all homography matrices, and performing model warm-ups.** This shifts initialization overhead from the first request to the application startup phase. `on_shutdown` handles resource cleanup.
        *   `models/`: Wrappers and interfaces for AI models.
            *   `base_models.py`: Abstract base classes for AI components (Detector, Tracker, FeatureExtractor) to implement the Strategy Pattern.
            *   `detectors.py`, `trackers.py`, `feature_extractors.py`: Concrete implementations of the AI strategies, likely wrapping libraries like BoxMOT. Models are designed to be loaded during application startup.
        *   `services/`: Business logic layer. Each service encapsulates a specific domain of functionality.
            *   `pipeline_orchestrator.py`: Orchestrates the sequence of operations in the core processing pipeline for each video segment and its frames. **Responsible for constructing media URLs and notifying the frontend (via `NotificationService`) about available media with associated metadata (e.g., `start_global_frame_index`, `num_frames_in_sub_video`).**
            *   `reid_components.py`: Manages the Re-ID process, including embedding comparison and gallery management (Redis/TimescaleDB). Relies on pre-loaded Re-ID models.
            *   `video_data_manager_service.py`: Implements logic for interacting with S3 (fetching video segments, managing local cache of sub-videos). This service provides the sub-videos that will be served by the new media endpoints and can provide frame counts for sub-videos.
            *   `homography_service.py`: Dedicated service that loads homography point files, pre-calculates all homography matrices at application startup via `on_startup` event handlers, and provides cached access to these matrices.
            *   `notification_service.py`: Responsible for sending `tracking_update`, `status_update`, and **`media_available` notifications (with URLs and synchronization metadata)** to connected frontend clients via WebSockets.
            *   `camera_tracker_factory.py`: Manages per-camera tracker instances, and aids in the startup pre-loading of shared Re-ID models by initializing a prototype tracker.
        *   `tasks/` (Potentially less emphasis now that `PipelineOrchestratorService` handles the main background processing loop): May contain specific, separable background jobs if needed.
        *   `utils/`: Common utility functions (e.g., image manipulation, timestamp conversions, video processing utilities).
        *   `main.py`: The entry point for the FastAPI application, where the app instance is created and routers are included. Its `lifespan` manager triggers the `on_startup` events. Includes the `/health` endpoint.
        *   `dependencies.py`: Defines FastAPI dependencies for injecting service instances, database sessions, etc., into endpoint handlers. These services expect their underlying models/data to be pre-loaded (available via `app.state`).
    *   `tests/`: Contains all unit and integration tests, mirroring the `app` structure.
    *   `.env.example`: Template for environment variables.
    *   `Dockerfile`: Instructions to build the Docker image for the backend.
    *   `docker-compose.yml`: For local development, sets up the backend service along with dependencies like Redis and TimescaleDB.
    *   `pyproject.toml`: Project metadata and Python package dependencies.
    *   `README.md`: General information about the backend project.
    *   `DESIGN.md`: This document.

## 4. Core Technologies

*   **Programming Language:** Python 3.9+
*   **Web Framework:** FastAPI (for REST APIs, WebSocket handling, and **serving static/streamed media files**, leveraging Starlette and Pydantic)
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
    *   **Usage:** To create instances of AI model strategies (if needed beyond simple instantiation at startup) or per-camera trackers (`CameraTrackerFactory`). The factory also plays a role in the startup sequence by creating a "prototype" tracker to ensure shared Re-ID models are loaded.
    *   **Benefit:** Decouples client code from concrete model classes, centralizes instantiation logic.
*   **Service Layer Pattern:**
    *   **Usage:** Business logic is encapsulated in service classes (e.g., `PipelineOrchestratorService`, `ReIDService`, `VideoDataManagerService`). API handlers delegate to these services.
    *   **Benefit:** Promotes Separation of Concerns (SoC), improves code organization and testability.
*   **Repository Pattern (or Data Access Object - DAO):**
    *   **Usage:** The `VideoDataManagerService` acts as a facade for data access related to S3 (video segments, local caching). Other services might abstract Redis/TimescaleDB interactions.
    *   **Benefit:** Abstracts data storage details.
*   **Observer Pattern (Conceptual, via WebSockets):**
    *   **Usage:** Backend processes data (subject), frontend clients (observers) receive real-time *metadata* and **media availability notifications** via WebSockets from `NotificationService`.
    *   **Benefit:** Enables real-time data streaming for client-side rendering and synchronization.
*   **Pipeline Pattern (Conceptual):**
    *   **Usage:** The core per-frame processing (Frame Extraction -> Preprocess -> Detect -> Track -> Feature Extract -> Re-ID -> Transform) is a pipeline managed by `PipelineOrchestratorService` and `MultiCameraFrameProcessor`.
    *   **Benefit:** Structures complex processing into manageable steps.
*   **Dependency Injection (via FastAPI):**
    *   **Usage:** FastAPI's dependency injection provides services, database connections, etc., to components. These dependencies (like model wrappers or services holding pre-computed data) are available immediately after application startup via `app.state`.
    *   **Benefit:** Promotes loose coupling and testability.
*   **Singleton Pattern (via FastAPI Dependencies & Application State):** Core services, AI model instances (detector, Re-ID model handler), and pre-computed data (like all homography matrices) are managed as singletons, initialized during the application's startup phase (stored in `app.state`) and reused throughout its lifecycle.

## 6. Communication Protocols

*   **RESTful APIs (HTTP/S):**
    *   **Purpose:** Client-initiated request-response interactions.
    *   **Examples:**
        *   Starting a processing session for specified cameras within an environment.
        *   Fetching historical analytics data.
        *   Retrieving system configuration (e.g., available environments, cameras per environment).
        *   **Serving sub-video files to the frontend for playback.**
        *   **`/health` endpoint:** Polled by the frontend to check backend readiness before establishing WebSocket connections, especially during application startup. This endpoint indicates the status of critical components like AI models and services.
    *   **Technology:** FastAPI (using `FileResponse` or `StreamingResponse` for media).
*   **WebSockets:**
    *   **Purpose:** Real-time, bidirectional communication for backend to push updates.
    *   **Key Message Types:**
        *   `tracking_update`: Contains per-frame metadata (bounding boxes, global IDs, map coordinates, `global_frame_index`). Optionally may include `sub_video_batch_index_of_frame` and `frame_index_in_sub_video` for easier frontend synchronization.
        *   `status_update`: Reports overall task progress and current operational step.
        *   **`media_available`**: Notifies the frontend when a new batch of sub-videos is downloaded by the backend and ready to be streamed. This message includes a list of `MediaURLEntry` objects, each containing:
            *   `camera_id`
            *   `sub_video_filename`
            *   `url` (the backend-hosted HTTP URL for this sub-video)
            *   `start_global_frame_index` (the global frame index corresponding to the start of this sub-video)
            *   `num_frames_in_sub_video` (total frames in this sub-video at the processing FPS)
        *   **`batch_processing_complete` (Optional):** Explicitly signals the frontend that all frames for a particular `sub_video_batch_index` have been processed and no more `tracking_update` messages for that batch will be sent.
    *   **Technology:** FastAPI's WebSocket support.
    *   **Client Connection Strategy:**
        1.  Client initiates a task via REST and receives a WebSocket URL.
        2.  Client attempts to connect to the WebSocket.
        3.  If the connection fails (e.g., during backend startup):
            *   Client polls the `/health` REST endpoint.
            *   Once `/health` indicates readiness (critical components loaded), client retries WebSocket connection.
            *   Implement retry logic with backoff for WebSocket connection attempts.

## 7. Key API Endpoints (High-Level)

All endpoints will be versioned (e.g., `/api/v1/...`). Pydantic schemas will define request and response bodies.

*   **Processing Tasks (`/processing-tasks`):**
    *   `POST /start`: Initiates a retrospective analysis task for specified cameras in an environment.
        *   Request: `{ "environment_id": "campus" }`
        *   Response: `{ "task_id": "uuid", "status_url": "/api/v1/processing_tasks/{task_id}/status", "websocket_url": "/ws/tracking/{task_id}" }`
    *   `GET /{task_id}/status`: Retrieves the current status of a processing task.
    *   `POST /{task_id}/control`: Send control commands (e.g., pause, resume - if supported, focus_track).

*   **Analytics Data (`/analytics_data`):**
    *   `GET /trajectory/{global_person_id}`: Get historical trajectory for a person.
    *   `GET /heatmap`: Get data for generating a heatmap.

*   **Configuration/Metadata (`/config`):**
    *   `GET /environments`: List available environments.
    *   `GET /environments/{environment_id}/cameras`: List cameras available for a given environment.

*   **Media Endpoints (`/media`):**
    *   `GET /tasks/{task_id}/environments/{environment_id}/cameras/{camera_id}/sub_videos/{sub_video_filename}`: Serves a specific sub-video file.
        *   Example: `/api/v1/media/tasks/xyz-123/environments/campus/cameras/c01/sub_videos/sub_video_01.mp4`
        *   Response: The binary video file (e.g., MP4) streamed to the client.
        *   These URLs are provided to the frontend via the `media_available` WebSocket notification.

*   **Health Endpoint (`/health`):**
    *   `GET /health`: Provides a status report of the backend.
        *   Response: JSON object indicating overall status and readiness of key components.
        *   **Usage by Frontend:** Polled by the client during its startup or after WebSocket connection failures to determine if the backend's core services are initialized and ready.

*   **WebSocket Endpoint (`/ws/tracking/{task_id}`):**
    *   Refer to Section 6 for key message types and their payloads (`tracking_update`, `status_update`, `media_available`, `batch_processing_complete`).


## 8. Data Flow and Frontend Synchronization Strategy

This describes the flow once a processing task is initiated, emphasizing frontend-backend synchronization. It assumes models and homography matrices are pre-loaded at application startup.

1.  **S3 Data Preparation (One-Time):**
    *   For each environment and its cameras, source videos are pre-split into a fixed number of sub-video segments (e.g., each camera in an environment might have 4 sub-videos, named `sub_video_01.mp4` through `sub_video_04.mp4`).
    *   These sub-videos are stored in S3 with a defined naming convention.

2.  **Initiation (REST API):**
    *   Frontend sends a request to `POST /api/v1/processing_tasks/start` (e.g., for a specific `environment_id`).
    *   The backend responds with `task_id` and `websocket_url`. Frontend also notes the `/health` endpoint.

3.  **Frontend WebSocket Connection Attempt & Health Check:**
    *   Frontend polls the backend's `/health` REST endpoint.
    *   Once `/health` indicates the backend is "healthy" (critical components loaded), the frontend establishes the WebSocket connection using the `websocket_url` received in step 2. Retry logic with backoff is used for WebSocket connection attempts if the backend is still initializing.

4.  **Task Creation & Sub-Video Batch Processing Loop (`PipelineOrchestratorService` - Backend):**
    *   The backend creates a processing task associated with the `task_id`.
    *   The system determines the total number of sub-video batches to process for the environment (e.g., if cameras have up to 4 sub-videos each, there will be 4 batches).
    *   The `PipelineOrchestratorService` enters a loop, iterating through each sub-video batch. The `sub_video_batch_index` will range from `0` up to `N-1`, where `N` is the maximum number of sub-videos per camera in the environment. For example, if `N=4`, `sub_video_batch_index` will be `0, 1, 2, 3`.
        *   **(a) Backend Downloads Current Sub-Video Batch (e.g., for `sub_video_batch_index = idx`):**
            *   `VideoDataManagerService` downloads the sub-video corresponding to this index (e.g., `sub_video_{idx+1}.mp4`) for *all relevant cameras* in the specified environment.
            *   These videos are saved to a local, task-specific cache directory on the backend server.
        *   **(b) Backend Notifies Frontend of Media Availability (via WebSocket):**
            *   Once the sub-video batch `idx` is downloaded locally on the backend, the `NotificationService` sends a `media_available` WebSocket message to the connected frontend client for this `task_id`.
            *   **Payload of `media_available`:**
                ```json
                {
                  "type": "media_available",
                  "payload": {
                    "sub_video_batch_index": 0, // Current 0-indexed batch identifier
                    "media_urls": [
                      {
                        "camera_id": "c01",
                        "sub_video_filename": "sub_video_01.mp4", // The actual filename
                        "url": "/api/v1/media/tasks/.../sub_video_01.mp4", // Backend relative URL to stream this video
                        "start_global_frame_index": 0, // Global frame index for the task corresponding to the start of this sub-video
                        "num_frames_in_sub_video": 500 // Expected number of frames in this sub-video (at processing FPS)
                      },
                      // ... entries for other cameras active in this batch
                    ]
                  }
                }
                ```
        *   **(c) Frontend Loads and Prepares Sub-Videos (Batch `idx`):**
            *   The frontend receives the `media_available` message. For each `MediaURLEntry`, it constructs the full HTTP URL to the backend's media server (e.g., `http://<backend_host>` + `entry.url`).
            *   It loads these videos into its respective HTML `<video>` players (e.g., by setting the `src` attribute).
            *   The frontend waits for player "ready" events (e.g., `canplaythrough`, `loadeddata`) before considering a video ready for playback. Videos should not auto-play upon load; playback control is synchronized with metadata.
        *   **(d) Backend Processes Frames and Streams Metadata (Batch `idx`):**
            *   The backend's `MultiCameraFrameProcessor` processes frames from the locally downloaded sub-videos for batch `idx`.
            *   For each processed multi-camera frame batch, `NotificationService` sends a `tracking_update` WebSocket message.
            *   **Payload of `tracking_update`:**
                ```json
                {
                  "type": "tracking_update",
                  "payload": {
                    "global_frame_index": 123, // Absolute frame index for the task, increments across all sub-video batches
                    "scene_id": "campus", // Environment ID
                    "timestamp_processed_utc": "...",
                    "cameras": {
                      "c01": {
                        "image_source": "000123.jpg", // Identifier for the frame image (e.g., based on global_frame_index)
                        "tracks": [ /* list of TrackedPersonData objects */ ]
                      },
                      // ... data for other cameras
                    }
                  }
                }
                ```
        *   **(e) Frontend Synchronizes Video Playback with Metadata (Batch `idx`):**
            *   Once a video player for batch `idx` is "ready," the frontend can start its playback.
            *   The frontend uses the `start_global_frame_index` and `num_frames_in_sub_video` (received in `media_available` for the current batch) to map the `global_frame_index` (from incoming `tracking_update` messages) to the correct playback time or frame number within the currently playing sub-video.
            *   Overlays (bounding boxes, IDs) are rendered onto the video based on this synchronized timing.
            *   The frontend should buffer `tracking_update` messages if they arrive before the video player is ready to display the corresponding frame, ensuring no metadata is lost.
        *   **(f) Backend Signals Batch Processing Completion (via WebSocket):**
            *   After processing all frames for `sub_video_batch_index = idx`, the backend **will** send a `batch_processing_complete` message.
            *   **Payload of `batch_processing_complete`:**
                ```json
                {
                  "type": "batch_processing_complete",
                  "payload": {
                    "sub_video_batch_index": idx // The 0-indexed batch that just finished processing
                  }
                }
                ```
        *   The loop then proceeds to the next `sub_video_batch_index` (e.g., `idx+1`), and steps (a) through (f) repeat for the next set of sub-videos.

5.  **Frontend Video Transition Management:**
    *   When a sub-video for batch `idx` finishes playing on the frontend, AND/OR upon receiving the `batch_processing_complete` message for batch `idx` from the backend:
        *   The frontend checks if `media_available` for the next batch (`idx+1`) has already been received and if those videos are loaded and ready in their respective players.
        *   If the next batch's videos are ready, the frontend transitions playback to the videos from batch `idx+1`. This might involve smoothly switching video sources or player visibility.

6.  **Data Storage & Caching (Backend - Ongoing):**
    *   `ReIDService` (or equivalent logic within `ReIDStateManager` or `MultiCameraFrameProcessor`) updates Redis with recent Re-ID embeddings.
    *   Historical tracking data is (asynchronously) written to TimescaleDB.

7.  **Task Completion:**
    *   When all sub-video batches for the environment have been processed, the backend sends a final "COMPLETED" `status_update` via WebSocket.
    *   The backend's `VideoDataManagerService` cleans up the locally cached sub-videos for the completed `task_id`.

## 9. Scalability and Maintainability

*   **Scalability:**
    *   Stateless FastAPI application instances can be horizontally scaled.
    *   AI models and homography data are pre-loaded.
    *   Video processing is local to the worker after download.
    *   WebSocket communication is for lightweight metadata and notifications.
    *   **Video delivery from backend HTTP endpoints:** Scalable by horizontal scaling of backend instances. For very high loads, consider dedicated media servers or CDNs.
    *   Redis and TimescaleDB are scalable.
    *   **Increased Backend Load:** Serving video files increases backend CPU, memory, and network bandwidth. Account for this in resource allocation.
*   **Maintainability:**
    *   Modular design, clear service responsibilities.
    *   Consistent coding style, type hinting, comprehensive testing.
    *   Detailed logging.
    *   Dependency injection via FastAPI.
    *   Centralized initialization logic in `on_startup`.
    *   Clear frontend guidelines for handling backend startup (`/health`) and processing synchronization (enriched WebSocket messages).

## 10. Next Steps / Considerations

*   **Authentication & Authorization** (for all endpoints, including new media endpoints).
*   **Detailed Error Handling & Retry Mechanisms:** Especially for S3 operations and external service interactions.
*   **Resource Management:** For GPU resources and local disk space. Monitor Docker startup time and resource consumption. Monitor backend resource usage under video streaming load.
*   **CI/CD Pipeline.**
*   **Frame Timestamps:** Precise calculation and communication for client-side synchronization (complementary to `global_frame_index`).
*   **Client-Side Video Rendering Strategy and Implementation:** Including robust handling of backend startup state (polling `/health`, WebSocket retry logic), dynamic updates to video player sources based on `media_available` notifications, and efficient metadata buffering/application.
*   **Task Concurrency and Resource Allocation:** How many tasks can a single backend instance handle.
*   **Explicit Prefetching of Sub-Video Batches:** Consider initiating the download of the *next* batch of sub-videos (`idx+1`) while the current batch (`idx`) is still undergoing frame-by-frame processing to hide S3 download latency.
*   **Range Requests for HTTP Video Serving:** Ensure FastAPI's `FileResponse` (or `StreamingResponse` if manually implemented) correctly handles HTTP Range Requests to allow clients to seek within videos efficiently.