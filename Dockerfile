# ---- Base Stage ----
# Using a specific Python version for better reproducibility
# CUDA 12.1 compatible base image for GPU support
FROM python:3.9.18-bullseye as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK on
ENV PIP_DEFAULT_TIMEOUT 100

# Install uv globally
RUN apt-get update && apt-get install -y curl procps && \
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
ENV PATH="/root/.local/bin:$PATH"

# Create a non-root user and group for security
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

WORKDIR /app

# ---- PyTorch Installer Stage ----
FROM base as pytorch_installer
ARG PYTORCH_VARIANT=cpu
ARG TORCH_VERSION="2.2.1"
ARG TORCHVISION_VERSION="0.17.1"
ARG TORCHAUDIO_VERSION="2.2.1"

# Install CUDA 12.1 dependencies for GPU support
RUN if [ "${PYTORCH_VARIANT}" = "cu121" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg2 software-properties-common && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-runtime-12-1 \
    cuda-libraries-12-1 \
    cuda-libraries-dev-12-1 \
    libcudnn8 \
    libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libopenblas-dev \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables for GPU support
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN uv venv /opt/venv --python $(which python)
ENV PATH="/opt/venv/bin:$PATH"

# Copy and run GPU dependencies installation script
COPY scripts/install_gpu_deps.sh /tmp/install_gpu_deps.sh
RUN chmod +x /tmp/install_gpu_deps.sh

RUN echo "Installing PyTorch variant: ${PYTORCH_VARIANT} using /root/.local/bin/uv" && \
    if [ "${PYTORCH_VARIANT}" = "cu121" ]; then \
    /root/.local/bin/uv pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cu121 \
    torchvision==${TORCHVISION_VERSION}+cu121 \
    torchaudio==${TORCHAUDIO_VERSION}+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    /root/.local/bin/uv pip install --no-cache-dir faiss-gpu>=1.7.4,<1.8.0; \
    elif [ "${PYTORCH_VARIANT}" = "cu118" ]; then \
    /root/.local/bin/uv pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cu118 \
    torchvision==${TORCHVISION_VERSION}+cu118 \
    torchaudio==${TORCHAUDIO_VERSION}+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    /root/.local/bin/uv pip install --no-cache-dir faiss-gpu>=1.7.4,<1.8.0; \
    elif [ "${PYTORCH_VARIANT}" = "cpu" ]; then \
    /root/.local/bin/uv pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cpu && \
    /root/.local/bin/uv pip install --no-cache-dir faiss-cpu>=1.7.4,<1.8.0; \
    else \
    echo "Error: Invalid PYTORCH_VARIANT. Must be 'cpu', 'cu118', or 'cu121'." && exit 1; \
    fi

# ---- Builder Stage ----
FROM pytorch_installer as builder

COPY pyproject.toml ./
COPY README.md ./ 

RUN uv pip install --no-cache-dir ".[dev]"

# ---- Runtime Stage ----
FROM base as runtime

ARG PYTORCH_VARIANT=cpu

# Install runtime dependencies including CUDA runtime for GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas-base \
    && if [ "${PYTORCH_VARIANT}" = "cu121" ]; then \
    apt-get install -y --no-install-recommends \
    wget gnupg2 software-properties-common && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-runtime-12-1 \
    cuda-libraries-12-1 \
    libcudnn8; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables for runtime
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY ./app /app/app

RUN mkdir -p /home/appuser/.cache/gdown

ARG LOCAL_VIDEO_DOWNLOAD_DIR="./downloaded_videos"   
ARG LOCAL_FRAME_EXTRACTION_DIR="./extracted_frames" 

# Define container paths for weights and homography data relative to WORKDIR /app
ENV CONTAINER_WEIGHTS_DIR_REL="./weights"               
ENV CONTAINER_HOMOGRAPHY_DIR_REL="./homography_points"  

# Create all necessary directories as root BEFORE copying files into them
RUN mkdir -p "${CONTAINER_WEIGHTS_DIR_REL}" && \
    mkdir -p "${CONTAINER_HOMOGRAPHY_DIR_REL}" && \
    mkdir -p "${LOCAL_VIDEO_DOWNLOAD_DIR}" && \
    mkdir -p "${LOCAL_FRAME_EXTRACTION_DIR}"

# Copy model weights and homography files from HOST to CONTAINER
COPY ./weights/ ${CONTAINER_WEIGHTS_DIR_REL}/
# MODIFIED: Copy homography data to its new top-level directory
COPY ./homography_data/ ${CONTAINER_HOMOGRAPHY_DIR_REL}/

RUN chown -R appuser:appgroup /app && \
    chown -R appuser:appgroup /home/appuser/.cache

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]