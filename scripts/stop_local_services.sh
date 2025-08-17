#!/bin/bash

# SpotOn Backend - Stop Local Services Script (Linux/macOS)
# This script stops all local development services

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

echo "🛑 Stopping SpotOn Backend Local Services"
echo "========================================="

# Stop FastAPI backend
print_info "Stopping FastAPI backend processes..."
pkill -f "uvicorn app.main:app" && print_status "FastAPI backend stopped" || print_info "No FastAPI backend process found"

# Stop Redis (if started by our scripts)
if pgrep -x "redis-server" > /dev/null; then
    print_info "Stopping Redis server..."
    redis-cli shutdown >/dev/null 2>&1 && print_status "Redis stopped" || print_warning "Could not stop Redis"
else
    print_info "Redis is not running"
fi

# Note about PostgreSQL (don't stop system service)
if pgrep -x "postgres" > /dev/null; then
    print_info "PostgreSQL is still running (system service - not stopping)"
else
    print_info "PostgreSQL is not running"
fi

# Clean up any remaining Python processes related to our app
print_info "Cleaning up any remaining processes..."
pkill -f "spoton" 2>/dev/null || true
pkill -f "app.main" 2>/dev/null || true

print_status "Local services cleanup complete"

echo ""
echo "To restart services:"
echo "1. Start services: ./scripts/start_backend_local.sh"
echo "2. Or manually: source .venv/bin/activate && python -m uvicorn app.main:app --reload --env-file .env.local"