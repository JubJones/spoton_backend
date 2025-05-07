# ---- Base Stage ----
# Using a specific Python version for better reproducibility
FROM python:3.9.18-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off # For pip, uv manages its own cache effectively
ENV PIP_DISABLE_PIP_VERSION_CHECK on
ENV PIP_DEFAULT_TIMEOUT 100

# Install uv globally
RUN apt-get update && apt-get install -y curl procps && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:$PATH"

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
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN uv venv /opt/venv --python $(which python)
ENV PATH="/opt/venv/bin:$PATH"

RUN echo "Installing PyTorch variant: ${PYTORCH_VARIANT}" && \
    if [ "${PYTORCH_VARIANT}" = "cu118" ]; then \
        uv pip install --no-cache-dir \
            torch==${TORCH_VERSION}+cu118 \
            torchvision==${TORCHVISION_VERSION}+cu118 \
            torchaudio==${TORCHAUDIO_VERSION}+cu118 \
            --index-url https://download.pytorch.org/whl/cu118; \
    elif [ "${PYTORCH_VARIANT}" = "cpu" ]; then \
        uv pip install --no-cache-dir \
            torch==${TORCH_VERSION}+cpu \
            torchvision==${TORCHVISION_VERSION}+cpu \
            torchaudio==${TORCHAUDIO_VERSION}+cpu \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Error: Invalid PYTORCH_VARIANT. Must be 'cpu' or 'cu118'." && exit 1; \
    fi

# ---- Builder Stage ----
# Installs application dependencies using the venv with PyTorch from the previous stage
FROM pytorch_installer as builder

# Copy project definition
COPY pyproject.toml ./

# Install dependencies from pyproject.toml.
# PyTorch, torchvision, torchaudio should already be in /opt/venv from the previous stage,
# so uv should respect these existing versions if they satisfy constraints.
# Use --no-deps for torch, torchvision, torchaudio if strict control is needed here,
# but usually uv's resolution handles this well.
RUN uv pip install --no-cache-dir ".[dev]" # Installs project and its dev dependencies

# ---- Runtime Stage ----
# Final image with the application and its dependencies
FROM base as runtime

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY ./app /app/app

# Set PATH to include the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set application directory as working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# The host 0.0.0.0 makes it accessible from outside the container.
# Application code (e.g., in app.core.config or model loading) should use:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to dynamically select the device at runtime.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]