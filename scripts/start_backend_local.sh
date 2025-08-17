#!/bin/bash

# SpotOn Backend - Local Development Startup Script (Linux/macOS)
# This script starts the local backend with all dependencies

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo "🚀 Starting SpotOn Backend (Local Development)"
echo "============================================="

# Check if we're in the right directory
if [[ ! -f "app/main.py" ]]; then
    print_error "Please run this script from the spoton_backend directory"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d ".venv" ]]; then
    print_error "Virtual environment not found. Please run setup_local_environment.sh first"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate
print_status "Virtual environment activated"

# Check environment file
if [[ ! -f ".env.local" ]]; then
    if [[ -f ".env.local.example" ]]; then
        print_warning "No .env.local found. Copying from .env.local.example..."
        cp .env.local.example .env.local
        print_warning "Please edit .env.local with your S3 credentials"
    else
        print_error "No .env.local or .env.local.example found"
        exit 1
    fi
fi

# Function to check if service is running
check_service() {
    local service_name=$1
    local check_command=$2
    
    if eval "$check_command" >/dev/null 2>&1; then
        print_status "$service_name is running"
        return 0
    else
        print_warning "$service_name is not running"
        return 1
    fi
}

# Start dependencies
print_info "Checking and starting dependencies..."

# Check Redis
if ! check_service "Redis" "redis-cli ping"; then
    print_info "Starting Redis..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [[ -f "$SCRIPT_DIR/start_redis_local.sh" ]]; then
        bash "$SCRIPT_DIR/start_redis_local.sh"
    else
        print_warning "Redis startup script not found. Please start Redis manually"
        print_info "Try: redis-server --daemonize yes"
    fi
fi

# Check PostgreSQL
if ! check_service "PostgreSQL" "pg_isready -h localhost -U spoton_user"; then
    print_error "PostgreSQL is not running. Please start PostgreSQL service:"
    print_info "Linux: sudo systemctl start postgresql"
    print_info "macOS: brew services start postgresql@14"
    exit 1
fi

# Check database connection
if ! PGPASSWORD=spoton_password psql -h localhost -U spoton_user -d spotondb -c "SELECT 1" >/dev/null 2>&1; then
    print_error "Cannot connect to SpotOn database. Please run database setup:"
    print_info "psql -U postgres -f scripts/setup_database_local.sql"
    exit 1
fi

print_status "All dependencies are ready"

# Check required files
print_info "Checking required files..."

if [[ ! -f "weights/clip_market1501.pt" ]]; then
    print_warning "Model weights not found at weights/clip_market1501.pt"
    print_info "Please download from: https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view"
fi

if [[ ! -d "homography_points" ]] || [[ $(find homography_points -name "*.npz" 2>/dev/null | wc -l) -eq 0 ]]; then
    print_warning "Homography data not found in homography_points/"
    print_info "Some coordinate transformations may not work without this data"
fi

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export ENV_FILE=".env.local"

print_info "Environment variables:"
print_info "  PYTHONPATH: $PYTHONPATH"
print_info "  ENV_FILE: $ENV_FILE"

# Display system information
print_info "System information:"
python -c "
import torch
import sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
" 2>/dev/null || print_warning "Could not display PyTorch information"

# Start the backend
print_info "Starting SpotOn Backend server..."
print_info "Server will be available at: http://localhost:8000"
print_info "API documentation: http://localhost:8000/docs"
print_info "Health check: http://localhost:8000/health"
echo ""
print_info "Press Ctrl+C to stop the server"
echo ""

# Start uvicorn with local configuration
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --env-file .env.local \
    --log-level info