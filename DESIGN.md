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
2.  **WebSocket Communication:** Provide real-time tracking updates to the frontend.
3.  **Data Processing Pipeline:**
    *   Fetch image data from S3.
    *   Perform object detection and intra-camera tracking (e.g., Faster R-CNN + BotSort via BoxMOT).
    *   Extract appearance features for Re-ID (e.g., CLIP via BoxMOT).
    *   Execute cross-camera Re-Identification and global ID association.
    *   Apply perspective transformation (Homography).
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
            *   `pipeline_orchestrator.py`: Orchestrates the sequence of operations in the core processing pipeline for each frame/stream.
            *   `reid_service.py`: Manages the Re-ID process, including embedding comparison and gallery management (Redis/TimescaleDB).
            *   `storage_service.py`: Implements the Repository/Data Access Object (DAO) pattern for interacting with S3, Redis, and TimescaleDB.
            *   `homography_service.py`: Handles homography transformations.
            *   `notification_service.py`: Responsible for sending data to connected frontend clients via WebSockets.
        *   `tasks/`: For background, potentially long-running processing tasks.
            *   `frame_processor.py`: Contains the detailed logic for processing a single frame or a batch of frames through the detection, tracking, Re-ID pipeline. This can be invoked asynchronously.
        *   `utils/`: Common utility functions (e.g., image manipulation, timestamp conversions).
        *   `main.py`: The entry point for the FastAPI application, where the app instance is created and routers are included.
        *   `dependencies.py`: Defines FastAPI dependencies for injecting service instances, database sessions, etc., into endpoint handlers.
    *   `tests/`: Contains all unit and integration tests, mirroring the `app` structure.
    *   `.env.example`: Template for environment variables.
    *   `Dockerfile`: Instructions to build the Docker image for the backend.
    *   `docker-compose.yml`: For local development, sets up the backend service along with dependencies like Redis and TimescaleDB.
    *   `pyproject.toml`: Project metadata and Python package dependencies (e.g., using Poetry or PDM).
    *   `README.md`: General information about the backend project.
    *   `DESIGN.md`: This document.

## 4. Core Technologies

*   **Programming Language:** Python 3.9+
*   **Web Framework:** FastAPI (for REST APIs and WebSocket handling, leveraging Starlette and Pydantic)
*   **AI/CV Libraries:** PyTorch, OpenCV, BoxMOT (for detection, tracking, Re-ID feature extraction)
*   **Async HTTP Client:** `aiobotocore` (for S3), `httpx` (if needed for other async calls)
*   **Cache:** Redis (for real-time state, Re-ID gallery)
*   **Database:** TimescaleDB on PostgreSQL (for historical data, with `pgvector` for embedding similarity search)
*   **Containerization:** Docker
*   **Orchestration (Recommended for Prod):** Kubernetes (or Docker Compose for simpler setups/dev)
*   **Dependency Management:** Poetry or PDM (recommended for `pyproject.toml`)

## 5. Design Patterns Employed

*   **Strategy Pattern:**
    *   **Usage:** For AI model components (detection, tracking, feature extraction). Abstract base classes (e.g., `AbstractDetector`, `AbstractTracker`, `AbstractFeatureExtractor`) define a common interface. Concrete classes (e.g., `FasterRCNNDetector`, `BotSortTracker`, `CLIPExtractor`) implement these interfaces, wrapping specific models from libraries like BoxMOT.
    *   **Benefit:** Allows easy swapping or addition of new AI models without altering the core pipeline logic. Promotes flexibility and testability.
*   **Factory Pattern (or Abstract Factory):**
    *   **Usage:** To create instances of AI model strategies. A `ModelFactory` (potentially within `dependencies.py` or loaded at startup) can be responsible for instantiating and configuring the chosen AI models based on application settings.
    *   **Benefit:** Decouples client code from concrete model classes and centralizes model creation logic.
*   **Service Layer Pattern:**
    *   **Usage:** Business logic is encapsulated in service classes (e.g., `PipelineOrchestratorService`, `ReIDService`, `StorageService`). API endpoint handlers in `app/api/` delegate tasks to these services.
    *   **Benefit:** Promotes Separation of Concerns (SoC), improves code organization, testability, and reusability of business logic.
*   **Repository Pattern (or Data Access Object - DAO):**
    *   **Usage:** The `StorageService` will act as a facade or implement repository patterns for data access operations related to S3, Redis, and TimescaleDB.
    *   **Benefit:** Abstracts data storage details from the rest of the application, making it easier to change database technologies or mock data sources for testing.
*   **Observer Pattern (Conceptual, via WebSockets):**
    *   **Usage:** The backend acts as a "subject" that processes data. The frontend clients connect via WebSockets and act as "observers." When new tracking data is available, the `NotificationService` (triggered by the `PipelineOrchestratorService`) pushes updates to all relevant observers.
    *   **Benefit:** Enables real-time data streaming to the frontend efficiently.
*   **Pipeline Pattern (Conceptual):**
    *   **Usage:** The core per-frame processing (Fetch -> Preprocess -> Detect -> Track -> Feature Extract -> Re-ID -> Transform) is inherently a pipeline. The `PipelineOrchestratorService` and `FrameProcessorTask` will manage the flow through these stages.
    *   **Benefit:** Structures complex processing into manageable, sequential steps.
*   **Dependency Injection (via FastAPI):**
    *   **Usage:** FastAPI's built-in dependency injection is used to provide instances of services, database connections, and configuration objects to API endpoint handlers and other components.
    *   **Benefit:** Promotes loose coupling and makes components easier to test in isolation.

## 6. Communication Protocols

*   **RESTful APIs (HTTP/S):**
    *   **Purpose:** Used for client-initiated requests that are typically request-response in nature.
    *   **Examples:**
        *   Starting/stopping a processing session for specific cameras and time ranges.
        *   User interactions like selecting a person to "focus track."
        *   Fetching historical or aggregated analytics data.
        *   Retrieving system configuration or metadata (e.g., list of available cameras).
    *   **Technology:** FastAPI.
*   **WebSockets:**
    *   **Purpose:** Used for real-time, bidirectional communication, primarily for the backend to push continuous updates to the frontend.
    *   **Examples:**
        *   Streaming per-frame tracking results (bounding boxes, global IDs, map coordinates, image URLs) to the frontend.
    *   **Technology:** FastAPI's WebSocket support.

## 7. Key API Endpoints (High-Level)

All endpoints will be versioned (e.g., `/api/v1/...`). Pydantic schemas will define request and response bodies.

*   **Processing Tasks (`/processing_tasks`):**
    *   `POST /start`: Initiates a retrospective analysis task.
        *   Request: `{ "cameras": ["id1", "id2"], "time_range_utc": {"start": "iso_datetime", "end": "iso_datetime"}, "environment_id": "factory_A" }`
        *   Response: `{ "task_id": "uuid", "status_url": "/api/v1/processing_tasks/{task_id}/status", "websocket_url": "/ws/tracking/{task_id}" }`
    *   `GET /{task_id}/status`: Retrieves the current status of a processing task.
    *   `POST /{task_id}/control`: Send control commands (e.g., pause, resume - if supported, focus_track).
        *   Request: `{ "command": "focus_track", "global_person_id": "person_xyz" }`

*   **Analytics Data (`/analytics_data`):**
    *   `GET /trajectory/{global_person_id}`: Get historical trajectory for a person.
        *   Params: `start_time_utc`, `end_time_utc`.
    *   `GET /heatmap`: Get data for generating a heatmap.
        *   Params: `camera_id` (optional), `start_time_utc`, `end_time_utc`.

*   **Configuration/Metadata (`/config`):**
    *   `GET /environments`: List available environments (e.g., Campus, Factory).
    *   `GET /environments/{environment_id}/cameras`: List cameras for a given environment.

*   **WebSocket Endpoint (`/ws/tracking/{task_id}`):**
    *   The backend pushes messages with the structure defined in the project overview (Section 5, WebSocket Payload).

## 8. Data Flow (Retrospective Analysis - Simplified)

1.  **Initiation (REST API):** Frontend sends a request to `POST /api/v1/processing_tasks/start` with camera IDs and time range.
2.  **Task Creation:** Backend creates a processing task (e.g., managed by `PipelineOrchestratorService`), possibly as a background job using `tasks/frame_processor.py`.
3.  **Image Fetching (Async):** `StorageService` fetches images from S3 for the current frame/timestamp.
4.  **AI Processing Loop (per frame, orchestrated by `PipelineOrchestratorService` delegating to `FrameProcessorTask`):**
    *   **Detection:** `DetectorStrategy` (e.g., FasterRCNN via BoxMOT) identifies persons.
    *   **Tracking:** `TrackerStrategy` (e.g., BotSort via BoxMOT) performs intra-camera tracking.
    *   **Feature Extraction (Conditional):** `FeatureExtractorStrategy` (e.g., CLIP via BoxMOT) extracts embeddings.
    *   **Re-Identification:** `ReIDService` compares embeddings against gallery (Redis/TimescaleDB) and assigns/updates `global_person_id`.
    *   **Homography:** `HomographyService` transforms image coordinates to map coordinates.
5.  **Data Storage & Caching:**
    *   `StorageService` updates Redis with real-time state.
    *   `StorageService` (asynchronously) writes historical data to TimescaleDB.
6.  **Real-time Update (WebSocket):** `NotificationService` (called by the orchestrator) pushes processed frame data (image URL, bounding boxes, IDs, map coordinates) to the relevant frontend client via WebSocket.

## 9. Scalability and Maintainability

*   **Scalability:**
    *   Stateless FastAPI application instances can be horizontally scaled behind a load balancer.
    *   AI model inference (CPU/GPU intensive) can be handled by running blocking I/O in separate thread pools (`asyncio.to_thread` or `run_in_executor`) or by offloading to dedicated model serving workers/services if performance demands.
    *   Redis and TimescaleDB can be scaled independently as per their capabilities (e.g., clustering, read replicas).
*   **Maintainability:**
    *   Modular design with clear separation of concerns.
    *   Consistent coding style (enforced by linters like Flake8 and formatters like Black).
    *   Comprehensive unit and integration tests.
    *   Type hinting (leveraged by FastAPI and Pydantic) improves code clarity and reduces runtime errors.
    *   Detailed logging and monitoring hooks.
    *   Dependency injection simplifies component wiring and testing.

## 10. Next Steps / Considerations

*   **Authentication & Authorization:** Secure API endpoints and WebSocket connections (e.g., JWT).
*   **Detailed Error Handling & Retry Mechanisms:** For S3 access, database operations, etc.
*   **Configuration of AI Models:** How homography matrices, model paths, thresholds are managed and loaded.
*   **Resource Management:** For GPU resources if AI models require them.
*   **CI/CD Pipeline:** For automated testing and deployment. 