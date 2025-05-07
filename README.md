# SpotOn Backend

Backend for the Intelligent Multi-Camera Person Tracking and Analytics System (SpotOn).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start with Docker Compose](#quick-start-with-docker-compose)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Configure Environment](#2-configure-environment)
  - [3. Build and Run Services](#3-build-and-run-services)
  - [Stopping Services](#stopping-services)
  - [Viewing Logs](#viewing-logs)
- [Running Tests (with Docker Compose)](#running-tests-with-docker-compose)
- [GPU Support Notes](#gpu-support-notes)
- [API Access](#api-access)
- [Tech Stack Overview](#tech-stack-overview)

## Prerequisites

*   [Docker](https://www.docker.com/get-started)
*   [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)
*   **For GPU Support:**
    *   NVIDIA GPU with compatible drivers.
    *   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your Docker host.

## Quick Start with Docker Compose

These instructions will guide you through running the entire backend stack (application, Redis, TimescaleDB) using Docker Compose.

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/spoton_backend.git # Replace with your repo URL
cd spoton_backend
```

### 2. Configure Environment

Copy the example environment file and customize it if necessary. The defaults are generally suitable for local Docker Compose development.

```bash
cp .env.example .env
```
Edit `.env` if you need to change default database credentials or S3 settings.

**For GPU/CPU Build Selection:**
The Docker build process can create either a CPU or GPU-enabled image for the backend. You can control this by setting the `PYTORCH_VARIANT_BUILD` variable in your `.env` file or your shell environment before building.

*   In your `.env` file, add or modify:
    ```env
    # For GPU build (e.g., CUDA 11.8):
    PYTORCH_VARIANT_BUILD=cu118

    # For CPU build (default if not set or set to 'cpu'):
    # PYTORCH_VARIANT_BUILD=cpu
    ```

### 3. Build and Run Services

This command will build the Docker images (respecting `PYTORCH_VARIANT_BUILD`) and start all services.

```bash
docker-compose up --build -d
```

*   `--build`: Ensures images are rebuilt if `Dockerfile` or related files have changed.
*   `-d`: Runs containers in detached mode (in the background).

Once started, the backend API will typically be available at `http://localhost:8000`.

### Stopping Services

To stop all running services defined in `docker-compose.yml`:

```bash
docker-compose down
```

To stop services and remove associated data volumes (e.g., database data, Redis cache):
```bash
docker-compose down -v
```

### Viewing Logs

To view logs from all services in real-time:

```bash
docker-compose logs -f
```

To view logs for a specific service (e.g., `backend`):

```bash
docker-compose logs -f backend
```

## Running Tests (with Docker Compose)

You can execute the test suite within a Docker container managed by Docker Compose. This ensures tests run in an environment consistent with the application.

1.  Ensure your images are built (the `docker-compose up --build` command above would have done this).
2.  Run the tests:
    ```bash
    # This command starts a temporary container for the 'backend' service and runs pytest.
    # It's recommended to use a CPU build for tests unless GPU-specific tests are needed.
    # If PYTORCH_VARIANT_BUILD was set to 'cpu' in .env or shell for the build:
    docker-compose run --rm backend pytest
    ```
    If your `docker-compose.yml` defaults to a GPU build, you might want to explicitly build a CPU image for testing or ensure your tests can handle the GPU environment if needed. Typically, tests are run on CPU images.

## GPU Support Notes

*   To build an image with GPU support, set `PYTORCH_VARIANT_BUILD=cu118` (or your target CUDA version) in your `.env` file or shell environment before running `docker-compose up --build`.
*   To **run** the container with GPU access:
    1.  Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on your Docker host.
    2.  Uncomment the `deploy` section in the `docker-compose.yml` file for the `backend` service to grant GPU access to the container.
*   The application code (`app/`) is designed to automatically detect and use `cuda` if `torch.cuda.is_available()` is true within the container.

## API Access

*   **REST API Base URL (local Docker):** `http://localhost:8000`
*   **OpenAPI Docs (Swagger UI):** `http://localhost:8000/docs`
*   **ReDoc:** `http://localhost:8000/redoc`
*   **WebSocket Example Endpoint:** `ws://localhost:8000/ws/tracking/{task_id}` (after a processing task is started)

## Tech Stack Overview

*   **Framework:** FastAPI
*   **Language:** Python 3.9+
*   **Database:** TimescaleDB (on PostgreSQL)
*   **Cache:** Redis
*   **Containerization:** Docker, Docker Compose
*   **AI/CV Libraries:** BoxMOT (interfacing with PyTorch, OpenCV)
*   **Package Management & Build:** `uv`, `pyproject.toml`, `hatchling` (primarily for Docker build process)
*   **Testing:** Pytest
*   **Linting/Formatting:** Ruff
