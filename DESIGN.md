# SpotOn Backend System Design

## 1. Introduction

This document outlines the high-level architecture for the SpotOn backend system. SpotOn is an intelligent multi-camera person tracking and analytics system designed for retrospective analysis of recorded video feeds. The backend is responsible for processing video data, performing AI-driven detection, tracking, re-identification, and serving this information to a frontend client.

The primary goals of this backend design are:
*   **Modularity:** Components should be loosely coupled and independently manageable.
*   **Scalability:** The system should be able to handle increasing data loads and user requests.
*   **Maintainability:** Code should be clean, well-documented, and easy to understand and modify.
*   **Reusability:** Leverage established design patterns and libraries effectively.
*   **Robustness:** Implement proper error handling and logging.

## 2. High-Level Architecture

The backend follows a service-oriented architecture, containerized using Docker and intended for cloud deployment (potentially orchestrated by Kubernetes).

Key backend responsibilities:
1.  **API Endpoints:** Expose RESTful APIs for control, configuration, and historical data retrieval.
2.  **WebSocket Communication:** Provide real-time *tracking metadata updates* to the frontend.
3.  **Data Processing Pipeline:**
    *   Fetch video segments (sub-videos) from S3 and cache them locally.
    *   Locally extract individual frames from the cached video segments.
    *   Perform object detection and intra-camera tracking on each frame (e.g., Faster R-CNN + BotSort via BoxMOT).
    *   Extract appearance features for Re-ID from detected persons (e.g., CLIP via BoxMOT).
    *   Execute cross-camera Re-Identification and global ID association.
    *   Apply perspective transformation (Homography) to get map coordinates.
    *   Compile tracking results (bounding boxes, global IDs, map coordinates). *Client handles video display and overlays based on this metadata.*
4.  **Data Management:**
    *   Interact with Redis for caching real-time state and recent Re-ID embeddings.
    *   Interact with TimescaleDB for storing historical tracking events and older embeddings.

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
            *   `event_handlers.py`: Logic for application lifecycle events (e.g., loading AI models on startup, closing database connections on shutdown).
        *   `models/`: Wrappers and interfaces for AI models.
            *   `base_models.py`: Abstract base classes for AI components (Detector, Tracker, FeatureExtractor) to implement the Strategy Pattern.
            *   `detectors.py`, `trackers.py`, `feature_extractors.py`: Concrete implementations of the AI strategies, likely wrapping libraries like BoxMOT.
        *   `services/`: Business logic layer. Each service encapsulates a specific domain of functionality.
            *   `pipeline_orchestrator.py`: Orchestrates the sequence of operations in the core processing pipeline for each video segment and its frames.
            *   `reid_service.py`: Manages the Re-ID process, including embedding comparison and gallery management (Redis/TimescaleDB).
            *   `storage_service.py`: Implements the Repository/Data Access Object (DAO) pattern for interacting with S3 (fetching video segments, managing local cache), Redis, and TimescaleDB.
            *   `homography_service.py`: Handles homography transformations.
            *   `notification_service.py`: Responsible for sending data to connected frontend clients via WebSockets.
        *   `tasks/`: For background, potentially long-running processing tasks.
            *   `frame_processor.py`: Contains the detailed logic for processing a single frame (extracted from a video segment) through the detection, tracking, Re-ID pipeline. This can be invoked by the `PipelineOrchestratorService`.
        *   `utils/`: Common utility functions (e.g., image manipulation, timestamp conversions, video processing utilities).
        *   `main.py`: The entry point for the FastAPI application, where the app instance is created and routers are included.
        *   `dependencies.py`: Defines FastAPI dependencies for injecting service instances, database sessions, etc., into endpoint handlers.
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
    *   **Usage:** For AI model components (detection, tracking, feature extraction). Abstract base classes (e.g., `AbstractDetector`, `AbstractTracker`, `AbstractFeatureExtractor`) define a common interface. Concrete classes (e.g., `FasterRCNNDetector`, `BotSortTracker`, `CLIPExtractor`) implement these interfaces.
    *   **Benefit:** Allows easy swapping or addition of new AI models without altering the core pipeline logic.
*   **Factory Pattern (or Abstract Factory):**
    *   **Usage:** To create instances of AI model strategies. A `ModelFactory` can instantiate and configure AI models based on application settings.
    *   **Benefit:** Decouples client code from concrete model classes.
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
    *   **Usage:** The core per-frame processing (Frame Extraction -> Preprocess -> Detect -> Track -> Feature Extract -> Re-ID -> Transform) is a pipeline managed by `PipelineOrchestratorService` and `FrameProcessorTask`.
    *   **Benefit:** Structures complex processing into manageable steps.
*   **Dependency Injection (via FastAPI):**
    *   **Usage:** FastAPI's dependency injection provides services, database connections, etc., to components.
    *   **Benefit:** Promotes loose coupling and testability.

## 6. Communication Protocols

*   **RESTful APIs (HTTP/S):**
    *   **Purpose:** Client-initiated request-response interactions.
    *   **Examples:**
        *   Starting a processing session for specified cameras within an environment.
        *   Fetching historical analytics data.
        *   Retrieving system configuration (e.g., available environments, cameras per environment).
        *   *Potentially serving video files or providing signed URLs for video access if not directly from S3.*
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
        *   Request: `{ "cameras": ["c01", "c02"], "environment_id": "campus" }` (Camera IDs are specific to the environment, e.g., "c01" for campus, "f01" for factory. Each camera implies processing its predefined set of 4 sub-videos.)
        *   Response: `{ "task_id": "uuid", "status_url": "/api/v1/processing_tasks/{task_id}/status", "websocket_url": "/ws/tracking/{task_id}" }`
            *   *Note: The client will need to derive the S3 paths for the sub-videos based on the `environment_id`, `camera_ids`, and sub-video sequence, or an additional mechanism might be needed for the client to discover these video URLs.*
    *   `GET /{task_id}/status`: Retrieves the current status of a processing task.
    *   `POST /{task_id}/control`: Send control commands (e.g., pause, resume - if supported, focus_track).
        *   Request: `{ "command": "focus_track", "global_person_id": "person_xyz" }`

*   **Analytics Data (`/analytics_data`):**
    *   `GET /trajectory/{global_person_id}`: Get historical trajectory for a person.
        *   Params: `start_time_utc` (optional, to filter results), `end_time_utc` (optional).
    *   `GET /heatmap`: Get data for generating a heatmap.
        *   Params: `camera_id` (optional), `start_time_utc` (optional), `end_time_utc` (optional).

*   **Configuration/Metadata (`/config`):**
    *   `GET /environments`: List available environments (e.g., "campus", "factory").
    *   `GET /environments/{environment_id}/cameras`: List cameras available for a given environment (e.g., for "campus": ["c01", "c02", "c03", "c04"]).
    *   *Potentially `GET /environments/{environment_id}/cameras/{camera_id}/video_urls` to get S3 URLs for sub-videos if client derivation is too complex.*

*   **WebSocket Endpoint (`/ws/tracking/{task_id}`):**
    *   The backend pushes messages with the structure defined in `app.api.v1.schemas.WebSocketTrackingMessage` (containing tracking metadata, not frame images).
    *   The video stream itself is not sent over WebSockets. The client is responsible for fetching and displaying the appropriate sub-video segments (e.g., constructing S3 URLs based on task details like environment, camera, and sub-video sequence) and then synchronizing the WebSocket metadata with the video playback.

## 8. Data Flow (Retrospective Analysis - Using Pre-Split Sub-Videos)

1.  **S3 Data Preparation (One-Time):**
    *   For each environment ("campus", "factory") and each of its 4 cameras, the ~5.5 min source video is pre-split into 4 sub-video segments (e.g., `sub_video_01.mp4` to `sub_video_04.mp4`).
    *   These sub-videos are stored in S3 with a defined naming convention (e.g., `s3://<bucket>/videos/campus/camera_c01/sub_video_01.mp4`).

2.  **Initiation (REST API):**
    *   Frontend sends a request to `POST /api/v1/processing_tasks/start` with `environment_id` and a list of `camera_ids`.
    *   The backend responds with task details, including the WebSocket URL for metadata.

3.  **Task Creation & Sub-Video List Generation (`PipelineOrchestratorService`):**
    *   Backend creates a processing task with a unique `task_id`.
    *   For each requested camera, the system identifies the S3 keys for its 4 sub-videos based on the `environment_id` and `camera_id`. This forms a master list of sub-videos for the task.
    *   The task is initiated to run in the background.

4.  **Sub-Video Fetching & Caching (Backend - `StorageService`):**
    *   For the current sub-video to be processed in the master list:
        *   `StorageService` checks its local task-specific cache (e.g., `/tmp/spoton_cache/<task_id>/<sub_video_filename>`).
        *   If not cached, it downloads the sub-video from S3 to the local cache.
    *   **Prefetching:** Asynchronously, `StorageService` starts downloading the *next* sub-video from the master list to the local cache to minimize waiting time.

5.  **Client-Side Video Access:**
    *   The frontend client, based on the `environment_id`, `camera_ids`, and the current sub-video sequence being processed by the backend (implicitly known or explicitly signaled), constructs the S3 URLs for the relevant sub-video segments.
    *   The client fetches and buffers these video segments for playback (e.g., using an HTML5 `<video>` tag).

6.  **Frame-by-Frame Processing Loop (Backend - Orchestrated by `PipelineOrchestratorService`, logic in `FrameProcessorTask` for a given sub-video):**
    *   Once a sub-video is locally cached on the backend, `PipelineOrchestratorService` instructs `FrameProcessorTask` (or similar component) to process it.
    *   **(a) Frame Extraction:** `FrameProcessorTask` opens the local sub-video file (e.g., using OpenCV's `cv2.VideoCapture`) and reads frames sequentially.
    *   **For each extracted frame:**
        *   **(b) Detection:** `DetectorStrategy` (e.g., `FasterRCNNDetector`) identifies persons in the frame.
        *   **(c) Tracking:** `TrackerStrategy` (e.g., `BotSortTracker` via BoxMOT) performs intra-camera tracking using detections.
        *   **(d) Feature Extraction:** `FeatureExtractorStrategy` (e.g., `CLIPExtractor` via BoxMOT) extracts appearance embeddings for new/updated tracks.
        *   **(e) Re-Identification:** `ReIDService` uses embeddings to assign or update `global_person_id` by comparing against a gallery (managed in Redis/TimescaleDB).
        *   **(f) Homography:** `HomographyService` transforms image coordinates to map coordinates.
        *   **(g) Data Aggregation:** Frame timestamp (derived from frame number and 23 FPS), camera ID, and all tracking results (bounding boxes, global IDs, map coordinates) are compiled.

7.  **Data Storage & Caching (Backend - Ongoing):**
    *   `StorageService` (or `ReIDService`) updates Redis with recent Re-ID embeddings for the gallery.
    *   `StorageService` (asynchronously) writes historical tracking data (trajectories, older embeddings) to TimescaleDB.

8.  **Real-time Metadata Update (Backend to Client via WebSocket - `NotificationService`):**
    *   After processing each frame (or a small batch), `NotificationService` is triggered.
    *   It sends a `WebSocketTrackingMessage` (containing frame timestamp, camera ID, tracking data with global IDs, bounding boxes, map coordinates) to clients subscribed to the `task_id`. *The message does not contain the frame image itself.*

9.  **Client-Side Video Playback and Overlay Rendering:**
    *   The frontend client plays the video segment fetched in Step 5.
    *   Upon receiving `WebSocketTrackingMessage`s, the client uses the `frame_timestamp` (or a derived frame index based on FPS) to synchronize the tracking metadata with the current video playback position.
    *   The client renders overlays (bounding boxes, global IDs, etc.) on top of the video display (e.g., using an HTML canvas).

10. **Iteration and Completion (Backend & Client):**
    *   The backend frame processing loop (Step 6) continues until all frames in the current sub-video are processed.
    *   The backend then moves to the next sub-video in the master list (which is ideally already prefetched by `StorageService`). The client also moves to playing the next sub-video segment.
    *   Once all sub-videos for all requested cameras are processed, the task is marked as complete on the backend.
    *   The local cache for the completed `task_id` is cleared by `StorageService` on the backend.

## 9. Scalability and Maintainability

*   **Scalability:**
    *   Stateless FastAPI application instances can be horizontally scaled.
    *   AI model inference can utilize `asyncio.to_thread` for CPU-bound tasks or be offloaded if necessary. GPU resource management will be key if GPUs are used.
    *   Video processing (frame extraction) is local to the worker after download, improving performance.
    *   Prefetching of video segments hides S3 latency.
    *   WebSocket communication is lightweight (metadata only), improving scalability for concurrent clients.
    *   Video delivery to clients can leverage standard HTTP streaming and CDNs.
    *   Redis and TimescaleDB are scalable.
*   **Maintainability:**
    *   Modular design with clear service responsibilities.
    *   Consistent coding style, type hinting, and comprehensive testing.
    *   Detailed logging.
    *   Dependency injection.

## 10. Next Steps / Considerations

*   **Authentication & Authorization:** Secure API endpoints and WebSocket connections.
*   **Detailed Error Handling & Retry Mechanisms:** For S3 access, video processing, database operations.
*   **Configuration of AI Models & Homography:** How homography matrices (per camera), model paths, and thresholds are managed and loaded (likely via `config.py` or a dedicated configuration service).
*   **Resource Management:** For GPU resources if AI models require them. Ensure efficient local disk space management for cached video segments.
*   **CI/CD Pipeline:** For automated testing and deployment.
*   **Frame Timestamps:** Precise calculation and communication of frame timestamps (or consistent frame indices) to ensure accurate client-side synchronization with video playback.
*   **Client-Side Video Rendering Strategy:** The current design assumes the client fetches video streams (e.g., sub-videos from S3 by deriving URLs) and synchronizes WebSocket metadata for overlays. Future considerations could include:
    *   An API endpoint for clients to discover video stream URLs if derivation logic becomes complex or S3 paths are not publicly patterned.
    *   Support for adaptive streaming protocols like HLS/DASH for video delivery to clients for improved user experience over varying network conditions.
    *   Detailed specifications for client-side synchronization mechanisms (e.g., handling video seeking, playback rate changes, buffering).
*   **Client-Side Implementation:** Significant logic will reside on the client for video playback, WebSocket data handling, timestamp synchronization, and rendering overlays.
*   **Task Concurrency:** How many sub-videos or cameras are processed in parallel by a single backend instance.