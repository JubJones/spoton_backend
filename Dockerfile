# ---- Base Stage ----
# Using a specific Python version for better reproducibility
FROM python:3.9.18-bullseye as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK on
ENV PIP_DEFAULT_TIMEOUT 100

# Install uv globally
RUN apt-get update && apt-get install -y curl procps && \
    # The installer script will place uv in $HOME/.local/bin by default.
    # For the root user in Docker, $HOME is /root.
    echo "Installing uv..." && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo "--- In base stage: Verifying uv installation (expected in /root/.local/bin) ---" && \
    echo "Listing /root/.local/bin contents:" && \
    ls -la /root/.local/bin && \
    echo "Running uv --version using absolute path /root/.local/bin/uv:" && \
    /root/.local/bin/uv --version && \
    echo "--- End of uv verification in base stage ---" && \
    apt-get purge -y --auto-remove curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Add uv's directory to PATH.
# For uv calls within this Dockerfile, we'll use absolute paths to be more robust,
# but this PATH modification is good practice for general use.
ENV PATH="/root/.local/bin:$PATH"

# Create a non-root user and group for security
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

WORKDIR /app

# ---- PyTorch Installer Stage ----
# This stage handles the conditional installation of PyTorch
FROM base as pytorch_installer
ARG PYTORCH_VARIANT=cpu # Default to CPU. Can be 'cpu' or 'cu118' (or other CUDA versions)
ARG TORCH_VERSION="2.2.1" # Define torch versions here or pass as build args
ARG TORCHVISION_VERSION="0.17.1"
ARG TORCHAUDIO_VERSION="2.2.1"

# System dependencies that might be needed by PyTorch or other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libopenblas-dev \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*
# Removed cython from apt-get

# Create virtual environment
RUN uv venv /opt/venv --python $(which python)
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch using absolute path for uv
RUN echo "Installing PyTorch variant: ${PYTORCH_VARIANT} using /root/.local/bin/uv" && \
    if [ "${PYTORCH_VARIANT}" = "cu118" ]; then \
        /root/.local/bin/uv pip install --no-cache-dir \
            torch==${TORCH_VERSION}+cu118 \
            torchvision==${TORCHVISION_VERSION}+cu118 \
            torchaudio==${TORCHAUDIO_VERSION}+cu118 \
            --index-url https://download.pytorch.org/whl/cu118; \
    elif [ "${PYTORCH_VARIANT}" = "cpu" ]; then \
        /root/.local/bin/uv pip install --no-cache-dir \
            torch==${TORCH_VERSION} \
            torchvision==${TORCHVISION_VERSION} \
            torchaudio==${TORCHAUDIO_VERSION} \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Error: Invalid PYTORCH_VARIANT. Must be 'cpu' or 'cu118'." && exit 1; \
    fi

# ---- Builder Stage ----
# Installs application dependencies using the venv with PyTorch from the previous stage
FROM pytorch_installer as builder

# Copy project definition
COPY pyproject.toml ./
COPY README.md ./
COPY ./app ./app

# Install Cython if it's needed as a build dependency for packages in pyproject.toml
# If Cython is a direct dependency of your project, it should be in pyproject.toml.
# If it's a build-dependency for another package, that package's build system
# should ideally declare it. This is a fallback.
RUN uv pip install --no-cache-dir cython

# Install dependencies from pyproject.toml.
# PyTorch, torchvision, torchaudio should already be in /opt/venv from the previous stage,
# so uv should respect these existing versions if they satisfy constraints.
RUN uv pip install --no-cache-dir ".[dev]"

# ---- Runtime Stage ----
# Final image with the application and its dependencies
FROM base as runtime

# Install runtime dependencies for OpenCV and other libraries from the pytorch_installer stage
# These are needed for cv2 and potentially other packages to run correctly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY ./app /app/app

# Set PATH to include the virtual environment's bin directory for app execution.
# /root/.local/bin (from base) is also still in PATH, but /opt/venv/bin takes precedence
# for python and packages installed in the venv.
ENV PATH="/opt/venv/bin:$PATH"

# Set application directory as working directory
WORKDIR /app

ARG LOCAL_VIDEO_DOWNLOAD_DIR="./downloaded_videos"
ARG LOCAL_FRAME_EXTRACTION_DIR="./extracted_frames"

# Create directories as root
RUN mkdir -p "${LOCAL_VIDEO_DOWNLOAD_DIR}" && \
    mkdir -p "${LOCAL_FRAME_EXTRACTION_DIR}"

# Change ownership to appuser
RUN chown -R appuser:appgroup "${LOCAL_VIDEO_DOWNLOAD_DIR}" && \
    chown -R appuser:appgroup "${LOCAL_FRAME_EXTRACTION_DIR}"

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]