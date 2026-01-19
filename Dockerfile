# ---- Base Stage ----
FROM python:3.10-bullseye AS base

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
  apt-get purge -y --auto-remove curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Add uv's directory to PATH.
ENV PATH="/root/.local/bin:$PATH"

# Create a non-root user and group for security
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

WORKDIR /app

# ---- Builder Stage ----
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv --python $(which python)
RUN /opt/venv/bin/python -m ensurepip --upgrade || true
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU (No CUDA)
RUN /opt/venv/bin/python -m pip install \
  torch==2.2.1 \
  torchvision==0.17.1 \
  torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cpu && \
  /opt/venv/bin/python -m pip install "faiss-cpu>=1.7.4,<1.8.0"

# Install application requirements
COPY requirements/requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/python -m pip install -r /tmp/requirements.txt && \
  /opt/venv/bin/python -m pip install boxmot==15.0.2 --no-deps

# ---- Runtime Stage ----
FROM base AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libopenblas-base curl \
  && rm -rf /var/lib/apt/lists/*

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
ENV YOLO_MODEL_PATH="weights/yolo26n.pt"

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
