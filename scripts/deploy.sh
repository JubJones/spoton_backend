#!/bin/bash

# SpotOn Backend Deployment Script
# This script handles deployment for different environments

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
GPU_ENABLED="false"
MONITORING_ENABLED="false"
BUILD_IMAGES="true"
PULL_IMAGES="false"
VERBOSE="false"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV    Target environment (production, staging, development)"
    echo "  -g, --gpu               Enable GPU support"
    echo "  -m, --monitoring        Enable monitoring stack"
    echo "  -p, --pull              Pull latest images instead of building"
    echo "  -v, --verbose           Verbose output"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e production -g -m   # Deploy to production with GPU and monitoring"
    echo "  $0 -e staging -p         # Deploy to staging with pulled images"
    echo "  $0 -e development        # Deploy to development"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ENABLED="true"
            shift
            ;;
        -m|--monitoring)
            MONITORING_ENABLED="true"
            shift
            ;;
        -p|--pull)
            PULL_IMAGES="true"
            BUILD_IMAGES="false"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if required configuration files exist
    if [[ ! -f "config/${ENVIRONMENT}.yaml" ]]; then
        print_error "Configuration file config/${ENVIRONMENT}.yaml not found"
        exit 1
    fi
    
    # Check if GPU is requested and available
    if [[ "$GPU_ENABLED" == "true" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            print_warning "nvidia-smi not found, GPU support may not work"
        else
            print_status "GPU support detected"
        fi
    fi
    
    print_status "Prerequisites check passed"
}

# Function to setup environment
setup_environment() {
    print_step "Setting up environment..."
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data
    mkdir -p models
    mkdir -p monitoring
    mkdir -p nginx/conf.d
    mkdir -p nginx/ssl
    mkdir -p scripts
    
    # Set environment variables
    export APP_ENV="$ENVIRONMENT"
    export COMPOSE_PROJECT_NAME="spoton_${ENVIRONMENT}"
    
    # Copy configuration files
    cp "config/${ENVIRONMENT}.yaml" "config/current.yaml"
    
    print_status "Environment setup completed"
}

# Function to build or pull images
handle_images() {
    if [[ "$BUILD_IMAGES" == "true" ]]; then
        print_step "Building Docker images..."
        
        if [[ "$VERBOSE" == "true" ]]; then
            docker build -t spoton-backend:latest .
        else
            docker build -t spoton-backend:latest . > /dev/null 2>&1
        fi
        
        print_status "Images built successfully"
    elif [[ "$PULL_IMAGES" == "true" ]]; then
        print_step "Pulling Docker images..."
        
        # Pull base images
        docker pull postgres:15-alpine
        docker pull timescale/timescaledb:latest-pg15
        docker pull redis:7-alpine
        docker pull nginx:alpine
        
        if [[ "$MONITORING_ENABLED" == "true" ]]; then
            docker pull prom/prometheus:latest
            docker pull grafana/grafana:latest
        fi
        
        print_status "Images pulled successfully"
    fi
}

# Function to start services
start_services() {
    print_step "Starting services..."
    
    # Build docker-compose command
    COMPOSE_CMD="docker-compose -f docker-compose.${ENVIRONMENT}.yml"
    
    # Add GPU profile if enabled
    if [[ "$GPU_ENABLED" == "true" ]]; then
        COMPOSE_CMD="$COMPOSE_CMD --profile gpu"
    fi
    
    # Add monitoring profile if enabled
    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        COMPOSE_CMD="$COMPOSE_CMD --profile monitoring"
    fi
    
    # Start services
    if [[ "$VERBOSE" == "true" ]]; then
        $COMPOSE_CMD up -d
    else
        $COMPOSE_CMD up -d > /dev/null 2>&1
    fi
    
    print_status "Services started successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for PostgreSQL..."
    while ! docker-compose -f docker-compose.${ENVIRONMENT}.yml exec postgres pg_isready -U spoton_user -d spoton_db; do
        sleep 2
    done
    
    # Wait for TimescaleDB
    print_status "Waiting for TimescaleDB..."
    while ! docker-compose -f docker-compose.${ENVIRONMENT}.yml exec timescale pg_isready -U spoton_user -d spoton_timescale; do
        sleep 2
    done
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    while ! docker-compose -f docker-compose.${ENVIRONMENT}.yml exec redis redis-cli ping; do
        sleep 2
    done
    
    # Wait for application
    print_status "Waiting for application..."
    while ! curl -f http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
    done
    
    print_status "All services are ready"
}

# Function to run database migrations
run_migrations() {
    print_step "Running database migrations..."
    
    # Run migrations
    docker-compose -f docker-compose.${ENVIRONMENT}.yml exec app python -m app.infrastructure.database.migrations.init_database
    
    print_status "Database migrations completed"
}

# Function to run health checks
run_health_checks() {
    print_step "Running health checks..."
    
    # Check application health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Application health check passed"
    else
        print_error "Application health check failed"
        exit 1
    fi
    
    # Check database health
    if docker-compose -f docker-compose.${ENVIRONMENT}.yml exec postgres pg_isready -U spoton_user -d spoton_db > /dev/null 2>&1; then
        print_status "Database health check passed"
    else
        print_error "Database health check failed"
        exit 1
    fi
    
    # Check Redis health
    if docker-compose -f docker-compose.${ENVIRONMENT}.yml exec redis redis-cli ping > /dev/null 2>&1; then
        print_status "Redis health check passed"
    else
        print_error "Redis health check failed"
        exit 1
    fi
    
    print_status "All health checks passed"
}

# Function to show deployment summary
show_summary() {
    print_step "Deployment Summary"
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "GPU Enabled: $GPU_ENABLED"
    echo "Monitoring Enabled: $MONITORING_ENABLED"
    echo ""
    echo "Services:"
    echo "  - Application: http://localhost:8000"
    echo "  - Health Check: http://localhost:8000/health"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo ""
    
    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        echo "Monitoring:"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000"
        echo ""
    fi
    
    echo "To view logs:"
    echo "  docker-compose -f docker-compose.${ENVIRONMENT}.yml logs -f"
    echo ""
    echo "To stop services:"
    echo "  docker-compose -f docker-compose.${ENVIRONMENT}.yml down"
    echo ""
}

# Function to cleanup on failure
cleanup_on_failure() {
    print_error "Deployment failed, cleaning up..."
    docker-compose -f docker-compose.${ENVIRONMENT}.yml down
    exit 1
}

# Main deployment function
main() {
    print_status "Starting SpotOn Backend deployment..."
    print_status "Target environment: $ENVIRONMENT"
    
    # Set up error handling
    trap cleanup_on_failure ERR
    
    # Run deployment steps
    check_prerequisites
    setup_environment
    handle_images
    start_services
    wait_for_services
    run_migrations
    run_health_checks
    
    print_status "Deployment completed successfully!"
    show_summary
}

# Run main function
main "$@"