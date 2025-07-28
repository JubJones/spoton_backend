#!/bin/bash

# GPU Dependencies Installation Script
# Installs PyTorch, FAISS-GPU, and CUDA dependencies for SpotOn backend

set -e

echo "Installing GPU dependencies for SpotOn backend..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo "No NVIDIA GPU detected. Installing CPU versions."
    GPU_AVAILABLE=false
fi

# Install dependencies based on GPU availability
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Installing GPU dependencies..."
    
    # Install PyTorch with CUDA 12.1 support
    uv pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
    
    # Install FAISS-GPU
    uv pip install faiss-gpu==1.7.4
    
    # Install additional GPU-accelerated libraries
    uv pip install accelerate transformers clip-by-openai
    
    echo "GPU dependencies installed successfully!"
    
    # Verify CUDA installation
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
    
else
    echo "Installing CPU dependencies..."
    
    # Install PyTorch CPU version
    uv pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
        --index-url https://download.pytorch.org/whl/cpu
    
    # Install FAISS-CPU
    uv pip install faiss-cpu==1.7.4
    
    # Install additional libraries
    uv pip install accelerate transformers clip-by-openai
    
    echo "CPU dependencies installed successfully!"
    
    # Verify CPU installation
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
fi

echo "Dependencies installation completed!"