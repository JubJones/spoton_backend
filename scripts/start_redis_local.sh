#!/bin/bash

# SpotOn Backend - Start Local Redis Server
# This script starts Redis with the local configuration

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

echo "🚀 Starting Local Redis Server for SpotOn"
echo "========================================"

# Check if Redis is already running
if pgrep -x "redis-server" > /dev/null; then
    print_warning "Redis server is already running"
    redis-cli ping >/dev/null 2>&1 && print_status "Redis is responding to ping" || print_error "Redis not responding"
    exit 0
fi

# Find Redis configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDIS_CONF="$SCRIPT_DIR/setup_redis_local.conf"
CUSTOM_CONF="$SCRIPT_DIR/redis-local.conf"

if [[ -f "$CUSTOM_CONF" ]]; then
    CONF_FILE="$CUSTOM_CONF"
    print_info "Using custom Redis configuration: $CUSTOM_CONF"
elif [[ -f "$REDIS_CONF" ]]; then
    CONF_FILE="$REDIS_CONF"
    print_info "Using default Redis configuration: $REDIS_CONF"
else
    print_warning "No Redis configuration found, starting with default settings"
    CONF_FILE=""
fi

# Create Redis data directory
REDIS_DATA_DIR="$SCRIPT_DIR/../data/redis"
mkdir -p "$REDIS_DATA_DIR"

# Create Redis log directory  
REDIS_LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$REDIS_LOG_DIR"

print_info "Redis data directory: $REDIS_DATA_DIR"
print_info "Redis log directory: $REDIS_LOG_DIR"

# Start Redis server
print_info "Starting Redis server..."

if [[ -n "$CONF_FILE" ]]; then
    # Start with configuration file
    redis-server "$CONF_FILE" --dir "$REDIS_DATA_DIR" --logfile "$REDIS_LOG_DIR/redis.log" &
else
    # Start with default configuration
    redis-server --dir "$REDIS_DATA_DIR" --logfile "$REDIS_LOG_DIR/redis.log" --daemonize yes &
fi

REDIS_PID=$!

# Wait a moment for Redis to start
sleep 2

# Check if Redis started successfully
if redis-cli ping >/dev/null 2>&1; then
    print_status "Redis server started successfully"
    print_info "Redis PID: $REDIS_PID"
    print_info "Data directory: $REDIS_DATA_DIR"
    print_info "Log file: $REDIS_LOG_DIR/redis.log"
    
    # Display Redis info
    print_info "Redis server info:"
    redis-cli info server | grep "redis_version\|os\|process_id\|uptime_in_seconds" | sed 's/^/  /'
    
    print_info "Memory info:"
    redis-cli info memory | grep "used_memory_human\|maxmemory_human" | sed 's/^/  /'
    
    echo ""
    echo "Redis server is running at: redis://localhost:6379"
    echo "To stop Redis: redis-cli shutdown"
    echo "To monitor Redis: redis-cli monitor"
    echo "To view logs: tail -f $REDIS_LOG_DIR/redis.log"
    
else
    print_error "Failed to start Redis server"
    print_info "Check the log file: $REDIS_LOG_DIR/redis.log"
    exit 1
fi