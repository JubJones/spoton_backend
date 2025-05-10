# SpotOn Backend

## Overview

SpotOn is an intelligent multi-camera person tracking and analytics system designed for retrospective analysis of recorded video feeds. The backend is responsible for processing video data, performing AI-driven detection, tracking, re-identification, applying homography transformations, and serving this metadata via REST APIs and WebSockets.

The system processes sub-videos from multiple cameras within a defined environment (e.g., "factory", "campus"). It extracts frames, performs detection (e.g., Faster R-CNN) and tracking (e.g., BotSort with Re-ID using CLIP features) on each frame. Detected persons are assigned global IDs, and their image coordinates can be transformed to map coordinates using pre-configured homography data. Real-time tracking metadata is streamed via WebSockets, while status and control are managed via REST APIs.

For a detailed system design, please refer to [DESIGN.md](DESIGN.md).

## Prerequisites

*   **Git**: For cloning the repository.
*   **Docker & Docker Compose**: Essential for running the application and its dependent services (Redis, TimescaleDB) in a containerized environment.
*   **Python 3.9+**: Required for local development if not using Docker for the application.
*   **`uv` (Python Package Installer)**: Used for managing Python environments and dependencies during local development. Install it by running:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **S3-Compatible Object Storage**: Access to an S3-compatible storage (e.g., AWS S3, MinIO, DagsHub) where the input video data (sub-videos) is stored.
*   **AI Model Weights**: Download the required Re-ID model weights (e.g., `clip_market1501.pt`). Links and filenames are typically specified in `weights/note.txt`. Place the downloaded `.pt` file(s) into the `spoton_backend/weights/` directory. This directory is copied into the Docker image during the build.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JubJones/spoton_backend # Replace with your repository URL if different
cd spoton_backend
```

### 2. Configure Environment Variables

The application uses environment variables for configuration. Copy the example file and customize it:

```bash
cp .env.example .env
```

Now, edit the `.env` file. Pay close attention to the following:
*   **S3/DagsHub Credentials & Configuration**:
    *   `S3_ENDPOINT_URL`: Your S3-compatible endpoint (e.g., `https://s3.amazonaws.com` or DagsHub's `https://s3.dagshub.com`).
    *   `AWS_ACCESS_KEY_ID`: Your S3 access key.
    *   `AWS_SECRET_ACCESS_KEY`: Your S3 secret key.
    *   `S3_BUCKET_NAME`: The name of the S3 bucket containing your video data.
    *   `DAGSHUB_REPO_OWNER` & `DAGSHUB_REPO_NAME`: If using DagsHub storage.
*   **Database Credentials** (if different from defaults in `docker-compose.yml`):
    *   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`.
*   **PyTorch Variant for Docker Build**:
    *   `PYTORCH_VARIANT_BUILD`: Set to `cpu` for CPU-only execution or `cu118` for CUDA 11.8 GPU support. If using `cu118`, ensure the NVIDIA Container Toolkit is installed on your Docker host and your Docker host has compatible NVIDIA drivers and GPUs.

### 3. Setup Options

You can set up the backend using Docker (recommended for most users and for running the service) or by setting up a local Python environment (suitable for active development of the backend code).

#### Option A: Docker Setup (Recommended for Running)

This method encapsulates the backend application and its dependencies (Redis, TimescaleDB) in Docker containers, providing a consistent and isolated environment.

1.  **Ensure AI Model Weights are in Place**:
    As mentioned in Prerequisites, download the necessary model weights (e.g., `clip_market1501.pt`) and place them in the `spoton_backend/weights/` directory. The `Dockerfile` is configured to copy the contents of this directory and `homography_data/` into the Docker image.

2.  **Build and Start Services using Docker Compose**:
    Navigate to the root directory of `spoton_backend` (where `docker-compose.yml` is located) and run:
    ```bash
    # Build images and start all services in detached mode
    docker-compose up --build -d
    ```
    *   To force a rebuild without using Docker's cache (e.g., if `Dockerfile` changes significantly):
        ```bash
        docker-compose build --no-cache
        docker-compose up -d
        ```
    *   The `PYTORCH_VARIANT_BUILD` variable from your `.env` file guides the PyTorch installation variant within the Docker image.

#### Option B: Local Development Setup

This setup is intended for developers who want to run and debug the Python application code directly on their host machine. It's still recommended to run Redis and TimescaleDB via Docker for simplicity.

1.  **Start External Services (Redis & TimescaleDB via Docker)**:
    From the `spoton_backend` root directory:
    ```bash
    # Start only Redis and TimescaleDB services in detached mode
    docker-compose up -d redis timescaledb
    ```
    Wait for these services to initialize. You can check their logs:
    ```bash
    docker-compose logs redis
    docker-compose logs timescaledb
    ```

2.  **Set up Python Virtual Environment and Install Dependencies**:
    Ensure you have Python 3.9+ and `uv` installed.
    ```bash
    # Create a Python 3.9 virtual environment (or your preferred 3.9+ version)
    uv venv .venv --python 3.9
    
    # Activate the virtual environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (PowerShell):
    # .\.venv\Scripts\Activate.ps1
    # On Windows (CMD):
    # .\.venv\Scripts\activate.bat

    # Install PyTorch. For local development, CPU version is often easier.
    # Ensure version compatibility with other libraries if changing.
    # Example for CPU PyTorch (uv will respect versions from pyproject.toml if compatible):
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install all project dependencies, including development tools like pytest and ruff
    uv pip install ".[dev]"
    ```

3.  **Ensure AI Model Weights and Homography Data are Accessible**:
    *   Confirm that the downloaded AI model weights are in the `spoton_backend/weights/` directory.
    *   The `spoton_backend/homography_data/` directory (containing `.npz` files) should be present.
    *   The application uses paths configured in your `.env` file (e.g., `WEIGHTS_DIR`, `HOMOGRAPHY_DATA_DIR`) to find these assets, which default to these locations relative to the app's execution directory.

## Running the Backend

### Using Docker (Recommended)

If you've set up using Docker (Option A):

*   **To Build and Start all services**:
    ```bash
    docker-compose up --build -d
    ```
*   **To Stop all services** (this will remove the containers but preserve volumes like database data):
    ```bash
    docker-compose down
    ```
*   **To Restart a specific service** (e.g., the backend if you mounted local code and made changes, though for production you'd rebuild image):
    ```bash
    docker-compose restart backend
    ```
*   **To View logs for the backend service**:
    ```bash
    docker-compose logs backend
    # To follow logs in real-time:
    docker-compose logs -f backend
    ```
    _Note: For the application's S3 data access (video downloads), the AWS credentials in your `.env` file are used by `boto3` and are sufficient._

### Local Development (Uvicorn)

If you've set up for local development (Option B):

1.  **Ensure Redis and TimescaleDB are running** (as started in local setup step 1).
2.  **Activate your Python virtual environment**:
    ```bash
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows: (use appropriate command for your shell)
    ```
3.  **Start the FastAPI application using Uvicorn**:
    From the `spoton_backend` root directory:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The backend API will be accessible at `http://localhost:8000`. The `--reload` flag enables automatic reloading of the server when Python code changes are detected, which is useful during development.

## Testing and Validation

Once the backend is running (either via Docker or locally):

### 1. API Tests (using `curl`)

You can perform quick tests of the API endpoints using `curl` or an API client like Postman or Insomnia.

*   **Start a processing task**:
    The `environment_id` (e.g., `"factory"`, `"campus"`) must match one configured in `app/core/config.py` under `VIDEO_SETS`.
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"environment_id": "factory"}' http://localhost:8000/api/v1/processing-tasks/start
    ```
    This command will return a JSON response. Note down the `task_id`, `status_url`, and `websocket_url` from the response. Example response:
    ```json
    {
      "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "message": "Processing task for environment 'factory' initiated.",
      "status_url": "/api/v1/processing-tasks/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/status",
      "websocket_url": "/ws/tracking/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    }
    ```

*   **Check task status**:
    Replace `{TASK_ID_FROM_PREVIOUS_STEP}` with the actual `task_id` you received.
    ```bash
    curl http://localhost:8000/api/v1/processing-tasks/{TASK_ID_FROM_PREVIOUS_STEP}/status
    ```
    This will show the current status, progress, and any details about the task.

### 2. WebSocket Test (using `scripts/websocket_client_test.py`)

The `scripts/websocket_client_test.py` script is a helpful tool for simulating a client that connects to the backend's WebSocket endpoint. It automates the process of starting a task and then listens for and prints messages received over the WebSocket. This is particularly useful for frontend developers to understand the real-time data format and flow.

*   **How to run the test client**:
    1.  Ensure the SpotOn backend is running and accessible at `http://localhost:8000`.
    2.  Open a new terminal.
    3.  If you set up the backend for local development (Option B) and activated its virtual environment, you can run the script from within that same environment, as it includes the necessary `httpx` and `websockets` packages. Otherwise, ensure these packages are installed in the Python environment you use to run the script:
        ```bash
        # If needed, in a separate environment or your global Python:
        uv pip install httpx websockets
        ```
    4.  Navigate to the `spoton_backend` root directory and execute the script:
        ```bash
        python scripts/websocket_client_test.py
        ```

*   **Understanding the WebSocket client output**:
    The client script will first log its attempt to start a processing task. If successful, it will connect to the WebSocket URL provided by the backend. You will then observe two main types of messages printed to your console:

    *   **`status_update` messages**: These provide updates on the overall progress and current stage of the processing task.
        Example log:
        ```
        INFO:websocket_client:[TASK {task_id}][STATUS_UPDATE]
        INFO:websocket_client:  Status: PROCESSING
        INFO:websocket_client:  Progress: 25.50%
        INFO:websocket_client:  Current Step: Processing sub-video 1, frame batch 80 (Global Index: 79)
        ```

    *   **`tracking_update` messages**: These are the core real-time data messages, sent for each processed batch of frames from the cameras. The structure of the payload (after JSON parsing) is crucial for frontend rendering.
        A typical `tracking_update` message payload looks like this (schema defined in `app.api.v1.schemas.WebSocketTrackingMessagePayload`):
        ```json
        {
          // "type": "tracking_update", // This is part of the outer message wrapper
          // "payload": { // This is the actual content
            "frame_index": 123, // 0-indexed global frame counter for the task
            "scene_id": "factory", // Corresponds to the environment_id
            "timestamp_processed_utc": "2023-10-27T10:30:05.456Z",
            "cameras": {
              "c09": { // Camera ID (string)
                "image_source": "000123.jpg", // A pseudo-filename for identification
                "tracks": [
                  {
                    "track_id": 5, // Intra-camera track ID (integer)
                    "global_id": "person-uuid-abc-123", // System-wide unique person ID (string, null if not identified)
                    "bbox_xyxy": [110.2, 220.5, 160.0, 330.8], // Bounding box [x1, y1, x2, y2] (list of floats)
                    "confidence": 0.92, // Detection confidence (float, optional)
                    "class_id": 1, // Object class ID (integer, e.g., 1 for person, optional)
                    "map_coords": [12.3, 45.6] // Projected [X, Y] on the map (list of floats, null if no homography or projection failed)
                  }
                  // ... more tracked persons in camera "c09"
                ]
              },
              "c12": {
                // ... data for camera "c12"
              }
              // ... other cameras active in this frame batch for the "factory" environment
            }
          // }
        }
        ```
        The `websocket_client_test.py` script logs a formatted version of this data. Frontend developers should particularly note:
        *   `frame_index`: Essential for synchronizing overlays with video playback.
        *   The `cameras` dictionary: This is keyed by camera ID string (e.g., "c09", "c12"). Each camera entry contains its `image_source` identifier and a list of `tracks`.
        *   The `tracks` list: Each item represents a person detected/tracked in that camera's view for the current `frame_index`.
        *   `bbox_xyxy`: Coordinates for drawing bounding boxes on the video.
        *   `global_id`: The persistent ID for a person across different cameras and over time.
        *   `map_coords`: The [X, Y] coordinates for plotting the person's location on a 2D map view.

This test client script effectively demonstrates the client-server interaction: initiating a task, receiving the WebSocket URL, establishing the connection, and processing incoming JSON messages containing tracking and status updates.

## Project Structure

The backend codebase is organized as follows:

*   `app/`: Main FastAPI application package.
    *   `api/`: API endpoint definitions (FastAPI routers, WebSocket handlers).
    *   `core/`: Core application setup (configuration, startup/shutdown events).
    *   `models/`: AI model wrappers and base abstractions (detectors, trackers).
    *   `services/`: Business logic layer (pipeline orchestration, Re-ID, data management).
    *   `tasks/`: Background processing logic (though much is now in services).
    *   `utils/`: Common utility functions.
    *   `main.py`: FastAPI application entry point.
*   `homography_data/`: Stores `.npz` files with points for homography calculations.
*   `scripts/`: Utility scripts, including `websocket_client_test.py`.
*   `tests/`: Unit and integration tests.
*   `weights/`: Directory for storing AI model weight files (e.g., `.pt` files).
*   `Dockerfile`: Instructions to build the Docker image for the backend.
*   `docker-compose.yml`: Defines services for local development and deployment (backend, Redis, TimescaleDB).
*   `pyproject.toml`: Project metadata and Python package dependencies.

## System Design

For a comprehensive understanding of the system architecture, components, data flow, and design patterns, please refer to the [DESIGN.md](DESIGN.md) document.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.