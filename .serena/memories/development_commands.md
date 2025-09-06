# SpotOn Backend Development Commands

## Environment Setup

### Using uv (Recommended)
```bash
# Create virtual environment
uv venv .venv --python 3.9
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
uv pip install ".[dev]"

# For PyTorch CPU version (local development)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Docker Commands

### Development
```bash
# CPU version for development
docker-compose -f docker-compose.cpu.yml up --build -d

# GPU version for production (requires NVIDIA GPU)
docker-compose -f docker-compose.gpu.yml up --build -d

# Start only Redis and TimescaleDB for local development
docker-compose -f docker-compose.cpu.yml up -d redis timescaledb

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down

# Force rebuild without cache
docker-compose build --no-cache
```

## Running the Application

### Local Development
```bash
# With supporting services in Docker
docker-compose -f docker-compose.cpu.yml up -d redis timescaledb
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Full Docker deployment (recommended)
docker-compose -f docker-compose.cpu.yml up --build -d
```

### Health Check
```bash
# Check if system is running
curl http://localhost:3847/health

# Check API documentation
# Open http://localhost:3847/docs (Swagger UI)
# Open http://localhost:3847/redoc (ReDoc)
```

## Testing Commands

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/models/test_detectors.py

# Run with coverage
pytest --cov=app tests/

# Run specific test markers
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m security               # Security tests only
pytest -m performance            # Performance tests only

# Run specific test function
pytest tests/services/test_pipeline_orchestrator.py::test_initialize_task

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html tests/
# View coverage: open htmlcov/index.html
```

### Test Configuration
- Coverage requirement: 80% minimum
- Test timeout: 300 seconds
- Async test support enabled
- HTML/XML coverage reports generated

## Code Quality Commands

### Formatting
```bash
# Format all Python code
ruff format .

# Check formatting without making changes
ruff format . --check
```

### Linting
```bash
# Lint all code
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Lint specific file
ruff check app/main.py
```

### Combined Quality Check
```bash
# Format and lint in one command
ruff format . && ruff check . --fix
```

## Frontend Simulation Testing

### Quick Start
```bash
# 1. Start backend services
docker-compose -f docker-compose.cpu.yml up -d

# 2. Install dependencies (if needed)
python setup_frontend_simulation.py

# 3. Run comprehensive test
python frontend_simulation_client.py
```

### Manual Testing Scripts
```bash
# Simple WebSocket test
python simple_ws_test.py

# Detailed WebSocket test
python test_websocket_detailed.py

# Quick WebSocket test
python quick_ws_test.py
```

## Monitoring and Debugging

### System Status
```bash
# Check health endpoints
curl http://localhost:3847/health
curl http://localhost:3847/api/v1/auth/health
curl http://localhost:3847/ws/health

# Monitor Docker containers
docker stats

# View application logs
docker-compose logs -f backend
tail -f app.log
```

### Performance Monitoring
```bash
# System performance dashboard
curl http://localhost:3847/api/v1/system/performance/dashboard

# Real-time metrics
curl http://localhost:3847/api/v1/analytics/real-time/metrics

# WebSocket statistics
curl http://localhost:3847/ws/stats
```

## Database Operations

### Local Development
```bash
# Access TimescaleDB
docker exec -it spoton_backend_timescaledb_1 psql -U postgres -d spoton

# Access Redis CLI
docker exec -it spoton_backend_redis_1 redis-cli

# View database logs
docker-compose logs timescaledb
docker-compose logs redis
```

## Troubleshooting Commands

### Common Issues
```bash
# Clear Redis cache
docker exec spoton_backend_redis_1 redis-cli flushall

# Restart specific service
docker-compose restart backend

# View system resources
docker system df
docker system prune  # Clean unused resources

# Check port usage
lsof -i :3847  # Check if port is in use
netstat -an | grep 3847
```

### Model Debugging
```bash
# Verify model weights exist
ls -la weights/clip_market1501.pt

# Check homography data
ls -la homography_data/

# Test AI models individually
python -c "from app.models.detectors import RTDETRDetector; print('Detector loads successfully')"
```

## Deployment Commands

### Production Deployment
```bash
# GPU production build
docker-compose -f docker-compose.gpu.yml up --build -d

# Check production health
curl https://your-domain.com/health

# View production logs
docker-compose -f docker-compose.gpu.yml logs -f backend
```

### Environment Configuration
```bash
# Copy environment file
cp .env.example .env

# Edit configuration
nano .env

# Validate configuration
python -c "from app.core.config import settings; print(f'App: {settings.APP_NAME}')"
```