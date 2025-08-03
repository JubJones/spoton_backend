#!/bin/bash

# SpotOn Backend Quick Start Script
# Simple deployment script for CPU or GPU configurations

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [cpu|gpu] [options]"
    echo ""
    echo "Options:"
    echo "  cpu                Start CPU version"
    echo "  gpu                Start GPU version"
    echo "  -i, --interactive  Start in interactive mode (Ctrl+C to stop)"
    echo "  -d, --down         Stop services"
    echo "  -l, --logs         View logs"
    echo "  --health           Check health status"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 cpu -i          Start CPU version interactively"
    echo "  $0 gpu -i          Start GPU version interactively"
    echo "  $0 cpu --logs      Start CPU and show logs"
    echo "  $0 gpu --down      Stop GPU services"
}

check_weights() {
    if [ ! -f "weights/clip_market1501.pt" ]; then
        print_error "Model weights not found!"
        print_warning "Please download clip_market1501.pt to weights/ directory"
        print_info "Download from: https://drive.google.com/uc?id=1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7"
        exit 1
    fi
    print_success "Model weights found"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "GPU support available"
            return 0
        fi
    fi
    print_warning "GPU not available or NVIDIA drivers not installed"
    return 1
}

start_cpu() {
    INTERACTIVE=${2:-false}
    print_info "Starting SpotOn Backend (CPU version)..."
    
    if [ ! -f ".env" ]; then
        print_info "Copying CPU environment configuration..."
        cp .env.cpu .env
    fi
    
    check_weights
    
    if [ "$INTERACTIVE" = "true" ]; then
        print_info "Starting in interactive mode (Ctrl+C to stop)..."
        docker-compose -f docker-compose.cpu.yml up --build
    else
        docker-compose -f docker-compose.cpu.yml up --build -d
        print_success "SpotOn Backend (CPU) started successfully!"
        print_info "Access: http://localhost:8000"
        print_info "Health: http://localhost:8000/health"
    fi
}

start_gpu() {
    INTERACTIVE=${2:-false}
    print_info "Starting SpotOn Backend (GPU version)..."
    
    if ! check_gpu; then
        print_error "GPU support not available. Use 'cpu' mode instead."
        exit 1
    fi
    
    if [ ! -f ".env" ]; then
        print_info "Copying GPU environment configuration..."
        cp .env.gpu .env
    fi
    
    check_weights
    
    if [ "$INTERACTIVE" = "true" ]; then
        print_info "Starting in interactive mode (Ctrl+C to stop)..."
        docker-compose -f docker-compose.gpu.yml up --build
    else
        docker-compose -f docker-compose.gpu.yml up --build -d
        print_success "SpotOn Backend (GPU) started successfully!"
        print_info "Access: http://localhost:8000"
        print_info "Health: http://localhost:8000/health"
    fi
}

stop_services() {
    print_info "Stopping SpotOn Backend services..."
    docker-compose -f docker-compose.cpu.yml down 2>/dev/null || true
    docker-compose -f docker-compose.gpu.yml down 2>/dev/null || true
    print_success "Services stopped"
}

show_logs() {
    MODE=${1:-cpu}
    print_info "Showing logs for $MODE version..."
    docker-compose -f docker-compose.$MODE.yml logs -f backend
}

check_health() {
    print_info "Checking health status..."
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "Service is healthy"
        curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
    else
        print_error "Service is not responding"
        exit 1
    fi
}

# Main script logic
case "${1:-}" in
    "cpu")
        case "${2:-}" in
            "--down"|"-d") stop_services ;;
            "--logs"|"-l") start_cpu && show_logs cpu ;;
            "--interactive"|"-i") start_cpu cpu true ;;
            *) start_cpu ;;
        esac
        ;;
    "gpu")
        case "${2:-}" in
            "--down"|"-d") stop_services ;;
            "--logs"|"-l") start_gpu && show_logs gpu ;;
            "--interactive"|"-i") start_gpu gpu true ;;
            *) start_gpu ;;
        esac
        ;;
    "--down"|"-d")
        stop_services
        ;;
    "--health")
        check_health
        ;;
    "--help"|"-h"|"")
        show_usage
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac