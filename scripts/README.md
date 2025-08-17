# SpotOn Backend - Local Development Scripts

This directory contains scripts for setting up and running SpotOn Backend locally without Docker.

## 🚀 Quick Start

```bash
# 1. Setup environment (run once)
./setup_local_environment.sh    # Linux/macOS
scripts\setup_local_environment.bat  # Windows

# 2. Validate setup
python validate_local_setup.py

# 3. Start backend
./start_backend_local.sh        # Linux/macOS  
scripts\start_backend_local.bat # Windows

# 4. Stop services when done
./stop_local_services.sh        # Linux/macOS
scripts\stop_local_services.bat # Windows
```

## 📁 Script Descriptions

### Setup Scripts
- **`setup_local_environment.sh`** (Linux/macOS) - Complete automated setup
- **`setup_local_environment.bat`** (Windows) - Complete automated setup
- **`setup_database_local.sql`** - PostgreSQL database schema and user setup
- **`setup_redis_local.conf`** - Redis configuration optimized for local development

### Service Management
- **`start_backend_local.sh`** (Linux/macOS) - Start backend with dependency checks
- **`start_backend_local.bat`** (Windows) - Start backend with dependency checks  
- **`start_redis_local.sh`** (Linux/macOS) - Start Redis server with local config
- **`stop_local_services.sh`** (Linux/macOS) - Stop all local services
- **`stop_local_services.bat`** (Windows) - Stop all local services

### Validation & Testing
- **`validate_local_setup.py`** - Comprehensive setup validation script

### Configuration
- **`.env.local.example`** - Local environment configuration template

## 🔧 Setup Process Overview

### 1. System Dependencies Installation
The setup scripts install required system packages:

**Linux (Ubuntu/Debian)**:
- PostgreSQL 14+ with TimescaleDB extension
- Redis server
- Python 3.9+ development tools
- Build essentials (gcc, cmake)
- GPU drivers (if available)

**macOS**:
- PostgreSQL via Homebrew
- Redis via Homebrew  
- Python 3.9+ via Homebrew
- Xcode command line tools

**Windows**:
- Manual installation guidance for PostgreSQL, Redis
- Python 3.9+ from python.org
- Visual Studio Build Tools (if needed)

### 2. Python Environment Setup
- Creates `.venv` virtual environment with Python 3.9+
- Installs all dependencies via `uv pip install ".[dev]"`
- Installs PyTorch with CUDA support (if GPU available)
- Downloads AI model weights (~600MB)

### 3. Database Configuration
- Creates PostgreSQL user `spoton_user` with password `spoton_password`
- Creates database `spotondb` with TimescaleDB extension
- Sets up tracking schema with hypertables for time-series data
- Creates indexes and materialized views for performance

### 4. Redis Configuration
- Configures Redis for local development
- Sets up memory limits and persistence options
- Optimized for development workloads

### 5. Environment Configuration
- Creates `.env.local` from template
- Configures local database and Redis connections
- Sets up S3 credentials (user must provide)
- Optimizes settings for local development

## 🧪 Validation Process

The `validate_local_setup.py` script performs comprehensive checks:

### System Requirements
- ✅ Python version (3.9+ required)
- ✅ Virtual environment setup and activation
- ✅ System commands availability (git, redis-cli, psql)

### Python Dependencies  
- ✅ All required packages installed
- ✅ PyTorch with/without CUDA support
- ✅ Database drivers (psycopg2, redis)
- ✅ Computer vision libraries (opencv, pillow)

### Environment Configuration
- ✅ Environment file exists and is readable
- ✅ Required environment variables are set
- ✅ Database connection strings are valid

### Required Files
- ✅ AI model weights (`weights/clip_market1501.pt`)
- ✅ Homography data (`homography_points/*.npz`)
- ✅ Application files (`app/main.py`, etc.)

### External Services
- ✅ Redis server running and accessible
- ✅ PostgreSQL server running with correct database
- ✅ Port 8000 available for backend

### GPU Support (Optional)
- ✅ CUDA availability and version
- ✅ GPU device count and names

### Application Modules
- ✅ Core application imports work correctly
- ✅ FastAPI application can be loaded
- ✅ Service and model modules are accessible

## 🐛 Troubleshooting

### Common Issues

1. **Virtual Environment Not Activated**
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate.bat  # Windows
   ```

2. **Missing Dependencies** 
   ```bash
   uv pip install ".[dev]"
   ```

3. **Database Connection Failed**
   ```bash
   # Check PostgreSQL is running
   sudo systemctl status postgresql  # Linux
   brew services list | grep postgres  # macOS
   
   # Run database setup
   psql -U postgres -f scripts/setup_database_local.sql
   ```

4. **Redis Connection Failed**
   ```bash
   # Start Redis
   ./scripts/start_redis_local.sh  # Linux/macOS
   redis-server  # Windows
   ```

5. **Model Weights Missing**
   - Download manually from Google Drive link
   - Place in `weights/clip_market1501.pt`
   - File should be ~600MB

### Complete Reset
```bash
# Stop all services
./scripts/stop_local_services.sh

# Remove virtual environment  
rm -rf .venv

# Re-run setup
./scripts/setup_local_environment.sh
```

## 📚 Additional Resources

- **Setup Guide**: `../docs/LOCAL_SETUP_GUIDE.md` - Detailed setup instructions
- **Troubleshooting**: `../docs/TROUBLESHOOTING.md` - Comprehensive issue resolution
- **Project README**: `../README.md` - Project overview and architecture

## 🎯 Script Usage Examples

### Development Workflow
```bash
# Daily development startup
./scripts/start_backend_local.sh

# Run validation after changes
python scripts/validate_local_setup.py

# Stop when done
./scripts/stop_local_services.sh
```

### CI/CD Integration
```bash
# In CI pipeline
python scripts/validate_local_setup.py
if [ $? -eq 0 ]; then
    echo "Setup validation passed"
else
    echo "Setup validation failed"
    exit 1
fi
```

### Manual Service Management
```bash
# Start individual services
./scripts/start_redis_local.sh
psql -U postgres -f scripts/setup_database_local.sql

# Validate specific components
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
redis-cli ping
psql -h localhost -U spoton_user -d spotondb -c "SELECT 1"
```