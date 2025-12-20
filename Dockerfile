# ---- Base Stage ----
# Using a specific Python version for better reproducibility
# CUDA 12.1 compatible base image for GPU support
FROM python:3.9.18-bullseye AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK on
ENV PIP_DEFAULT_TIMEOUT 100

# Install uv globally
ARG VERBOSE_BUILD=false
RUN apt-get update && apt-get install -y --no-install-recommends curl procps ca-certificates && \
  curl -LsSf https://astral.sh/uv/install.sh | sh && \
  if [ "$VERBOSE_BUILD" = "true" ]; then \
  echo "uv installed at /root/.local/bin" && ls -la /root/.local/bin && /root/.local/bin/uv --version; \
  fi && \
  apt-get purge -y --auto-remove curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Add uv's directory to PATH.
ENV PATH="/root/.local/bin:$PATH"

# Create a non-root user and group for security
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

WORKDIR /app

# ---- PyTorch Installer Stage ----
FROM base AS pytorch_installer
ARG PYTORCH_VARIANT=cpu
ARG TORCH_VERSION="2.2.1"
ARG TORCHVISION_VERSION="0.17.1"
ARG TORCHAUDIO_VERSION="2.2.1"

ARG CUDA_VERSION=12.1
ARG CUDA_APT_VER=12-1
# Install CUDA dependencies for GPU support when requested
ENV DEBIAN_FRONTEND=noninteractive

RUN if [ "${PYTORCH_VARIANT}" = "cu121" ]; then \
  echo "--- Starting CUDA setup for cu121 ---" && \
  apt-get update && \
  echo "--- Installing initial dependencies ---" && \
  apt-get install -y --no-install-recommends wget gnupg2 software-properties-common && \
  echo "--- Transforming sources list ---" && \
  sed -i -e's/ main/ main contrib non-free/g' /etc/apt/sources.list && \
  echo "--- Fetching NVIDIA GPG key ---" && \
  wget -O - https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
  echo "--- Adding NVIDIA repository ---" && \
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
  echo "--- Updating apt cache with new repo ---" && \
  apt-get update && \
  echo "--- Installing CUDA packages ---" && \
  apt-get install -y --no-install-recommends \
  cuda-runtime-${CUDA_APT_VER} \
  cuda-libraries-${CUDA_APT_VER} \
  cuda-libraries-dev-${CUDA_APT_VER} \
  libcudnn8 \
  libcudnn8-dev && \
  echo "--- Cleaning up ---" && \
  rm -rf /var/lib/apt/lists/* && \
  echo "--- CUDA setup complete ---"; \
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
RUN /opt/venv/bin/python -m ensurepip --upgrade || true
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip bash -lc ' \
  echo "Installing PyTorch variant: ${PYTORCH_VARIANT} using pip in venv"; \
  PIP="/opt/venv/bin/python -m pip"; \
  if [ "${PYTORCH_VARIANT}" = "cu121" ]; then \
  $PIP install \
  torch==${TORCH_VERSION}+cu121 \
  torchvision==${TORCHVISION_VERSION}+cu121 \
  torchaudio==${TORCHAUDIO_VERSION}+cu121 \
  --index-url https://download.pytorch.org/whl/cu121 && \
  $PIP install "faiss-gpu>=1.7.4,<1.8.0"; \
  elif [ "${PYTORCH_VARIANT}" = "cu118" ]; then \
  $PIP install \
  torch==${TORCH_VERSION}+cu118 \
  torchvision==${TORCHVISION_VERSION}+cu118 \
  torchaudio==${TORCHAUDIO_VERSION}+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 && \
  $PIP install "faiss-gpu>=1.7.4,<1.8.0"; \
  elif [ "${PYTORCH_VARIANT}" = "cpu" ]; then \
  $PIP install \
  torch==${TORCH_VERSION} \
  torchvision==${TORCHVISION_VERSION} \
  torchaudio==${TORCHAUDIO_VERSION} \
  --index-url https://download.pytorch.org/whl/cpu && \
  $PIP install "faiss-cpu>=1.7.4,<1.8.0"; \
  else \
  echo "Error: Invalid PYTORCH_VARIANT. Must be \"cpu\", \"cu118\", or \"cu121\"." && exit 1; \
  fi'

# ---- Builder Stage ----
FROM pytorch_installer AS builder

# Install runtime dependencies explicitly to avoid GUI packages like PyQt5
COPY requirements/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip /opt/venv/bin/python -m pip install -r /tmp/requirements.txt && \
  /opt/venv/bin/python -m pip install boxmot==15.0.2 --no-deps

# ---- Runtime Stage ----
FROM base AS runtime

ARG PYTORCH_VARIANT=cpu
ARG CUDA_APT_VER=12-1

# Install runtime dependencies including CUDA runtime for GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libopenblas-base curl \
  && if [ "${PYTORCH_VARIANT}" = "cu121" ]; then \
  apt-get install -y --no-install-recommends wget gnupg2 software-properties-common && \
  sed -i -e's/ main/ main contrib non-free/g' /etc/apt/sources.list && \
  wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-runtime-${CUDA_APT_VER} \
  cuda-libraries-${CUDA_APT_VER} \
  libcudnn8; \
  fi && rm -rf /var/lib/apt/lists/*

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
COPY ./reid /app/reid

RUN mkdir -p /home/appuser/.cache/gdown

# Configure writable dirs for libraries that attempt to write under $HOME
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
RUN mkdir -p /tmp/matplotlib /tmp/ultralytics && chown -R appuser:appgroup /tmp/matplotlib /tmp/ultralytics

ARG LOCAL_VIDEO_DOWNLOAD_DIR="./downloaded_videos"   
ARG LOCAL_FRAME_EXTRACTION_DIR="./extracted_frames" 

# Define container paths for weights and homography data relative to WORKDIR /app
ENV CONTAINER_WEIGHTS_DIR_REL="./weights"
ENV CONTAINER_HOMOGRAPHY_DIR_REL="./homography_data"

# Create all necessary directories as root BEFORE copying files into them
RUN mkdir -p "${CONTAINER_WEIGHTS_DIR_REL}" \
  "${CONTAINER_HOMOGRAPHY_DIR_REL}" \
  "${LOCAL_VIDEO_DOWNLOAD_DIR}" \
  "${LOCAL_FRAME_EXTRACTION_DIR}"

RUN chown -R appuser:appgroup /app && \
  chown -R appuser:appgroup /home/appuser/.cache

USER appuser
EXPOSE 3847
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3847"]
