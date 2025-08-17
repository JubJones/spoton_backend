@echo off
REM SpotOn Backend - Local Environment Setup Script (Windows)
REM This script sets up the complete local development environment

setlocal enabledelayedexpansion

echo 🚀 SpotOn Backend - Local Environment Setup
echo ============================================

REM Color codes (limited in batch)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function equivalents for status messages
:print_status
echo %GREEN%✅ %~1%NC%
goto :eof

:print_error
echo %RED%❌ %~1%NC%
goto :eof

:print_warning
echo %YELLOW%⚠️  %~1%NC%
goto :eof

:print_info
echo %BLUE%ℹ️  %~1%NC%
goto :eof

REM Check if running with admin privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    call :print_warning "Running with administrator privileges"
) else (
    call :print_info "Running without administrator privileges (some features may require manual setup)"
)

call :print_info "Detected OS: Windows"

REM Check system requirements
call :print_info "Checking system requirements..."

REM Check Python
python --version >nul 2>&1
if %errorLevel% == 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    call :print_status "Python found: !PYTHON_VERSION!"
    set PYTHON_CMD=python
) else (
    call :print_error "Python not found. Please install Python 3.9+ from https://www.python.org/"
    pause
    exit /b 1
)

REM Check GPU
nvidia-smi >nul 2>&1
if %errorLevel% == 0 (
    for /f "skip=1 tokens=1,2" %%a in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader^,nounits') do (
        call :print_status "NVIDIA GPU detected: %%a %%b MB"
        set HAS_GPU=1
        goto :gpu_checked
    )
) else (
    call :print_warning "nvidia-smi not found. GPU acceleration may not work."
    set HAS_GPU=0
)
:gpu_checked

REM Check Git
git --version >nul 2>&1
if %errorLevel% == 0 (
    call :print_status "Git found"
) else (
    call :print_error "Git not found. Please install Git from https://git-scm.com/"
    pause
    exit /b 1
)

REM Install system dependencies (manual instructions)
call :print_info "System dependencies check..."

REM Check PostgreSQL
pg_config --version >nul 2>&1
if %errorLevel% == 0 (
    call :print_status "PostgreSQL found"
    set HAS_POSTGRES=1
) else (
    call :print_warning "PostgreSQL not found"
    call :print_info "Please install PostgreSQL 14+ from: https://www.postgresql.org/download/windows/"
    call :print_info "Also install TimescaleDB extension from: https://docs.timescale.com/install/latest/self-hosted/installation-windows/"
    set HAS_POSTGRES=0
)

REM Check Redis
redis-cli ping >nul 2>&1
if %errorLevel% == 0 (
    call :print_status "Redis found and running"
    set HAS_REDIS=1
) else (
    call :print_warning "Redis not found or not running"
    call :print_info "Please install Redis from: https://github.com/microsoftarchive/redis/releases"
    set HAS_REDIS=0
)

REM Setup Python virtual environment
call :print_info "Setting up Python virtual environment..."

if not exist ".venv" (
    %PYTHON_CMD% -m venv .venv
    call :print_status "Virtual environment created"
) else (
    call :print_info "Virtual environment already exists"
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip and install uv
python -m pip install --upgrade pip
pip install uv

call :print_status "Python virtual environment ready"

REM Install Python dependencies
call :print_info "Installing Python dependencies..."

REM Install project dependencies
if exist "pyproject.toml" (
    uv pip install ".[dev]"
) else if exist "requirements.txt" (
    uv pip install -r requirements.txt
) else (
    call :print_error "No pyproject.toml or requirements.txt found"
    pause
    exit /b 1
)

REM Install PyTorch with CUDA support (if GPU available)
if %HAS_GPU%==1 (
    call :print_info "Installing PyTorch with CUDA support..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    call :print_info "Installing PyTorch CPU version..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

call :print_status "Python dependencies installed"

REM Create weights directory and download model
call :print_info "Setting up model weights..."

if not exist "weights" mkdir weights

set WEIGHTS_FILE=weights\clip_market1501.pt

if not exist "%WEIGHTS_FILE%" (
    call :print_warning "Model weights not found. Please download manually:"
    call :print_info "1. Visit: https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view"
    call :print_info "2. Download clip_market1501.pt (~600MB)"
    call :print_info "3. Save to: %WEIGHTS_FILE%"
    call :print_info "Press any key after downloading the model weights..."
    pause >nul
) else (
    call :print_status "Model weights found"
)

REM Setup environment configuration
call :print_info "Setting up environment configuration..."

if not exist ".env.local" (
    if exist ".env.example" (
        copy ".env.example" ".env.local" >nul
        
        REM Update configuration for Windows local setup
        powershell -Command "(gc .env.local) -replace 'POSTGRES_SERVER=.*', 'POSTGRES_SERVER=localhost' | Out-File -encoding ASCII .env.local"
        powershell -Command "(gc .env.local) -replace 'REDIS_HOST=.*', 'REDIS_HOST=localhost' | Out-File -encoding ASCII .env.local"
        
        call :print_status "Environment configuration created (.env.local)"
    ) else (
        call :print_error ".env.example not found"
        pause
        exit /b 1
    )
) else (
    call :print_info "Environment configuration already exists"
)

REM Setup PostgreSQL database (if available)
if %HAS_POSTGRES%==1 (
    call :print_info "Setting up PostgreSQL database..."
    
    REM Create database and user
    psql -U postgres -c "CREATE USER spoton_user WITH PASSWORD 'spoton_password';" 2>nul
    psql -U postgres -c "CREATE DATABASE spotondb OWNER spoton_user;" 2>nul
    psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE spotondb TO spoton_user;" 2>nul
    
    REM Enable TimescaleDB extension
    psql -U postgres -d spotondb -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" 2>nul
    
    call :print_status "PostgreSQL database configured"
)

REM Verify installation
call :print_info "Verifying installation..."

REM Test Python environment
python -c "import torch; import sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>nul
if %errorLevel% == 0 (
    call :print_status "Python environment test passed"
) else (
    call :print_warning "Python environment test failed"
)

REM Test database connections
if %HAS_POSTGRES%==1 (
    psql -h localhost -U spoton_user -d spotondb -c "SELECT version();" >nul 2>&1
    if !errorLevel! == 0 (
        call :print_status "PostgreSQL connection successful"
    ) else (
        call :print_warning "PostgreSQL connection failed"
    )
)

if %HAS_REDIS%==1 (
    redis-cli ping >nul 2>&1
    if !errorLevel! == 0 (
        call :print_status "Redis connection successful"
    ) else (
        call :print_warning "Redis connection failed"
    )
)

REM Check required files
if exist "weights\clip_market1501.pt" (
    call :print_status "Model weights found"
) else (
    call :print_warning "Model weights not found"
)

if exist "homography_points\*.npz" (
    call :print_status "Homography data found"
) else (
    call :print_warning "Homography data not found"
)

REM Create startup script
call :print_info "Creating startup script..."

echo @echo off > start_backend_local.bat
echo echo Starting SpotOn Backend... >> start_backend_local.bat
echo call .venv\Scripts\activate.bat >> start_backend_local.bat
echo python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env.local >> start_backend_local.bat
echo pause >> start_backend_local.bat

call :print_status "Startup script created (start_backend_local.bat)"

echo.
echo 🎉 Setup Complete!
echo ==================
call :print_status "Local environment setup finished successfully"
echo.
echo Next steps:
echo 1. Activate Python environment: .venv\Scripts\activate.bat
echo 2. Start the backend: start_backend_local.bat
echo    OR: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env.local
echo 3. Test health endpoint: curl http://localhost:8000/health
echo 4. View API docs: http://localhost:8000/docs
echo.
echo Manual setup required if not already done:
if %HAS_POSTGRES%==0 echo - Install PostgreSQL 14+ and TimescaleDB
if %HAS_REDIS%==0 echo - Install and start Redis
if not exist "weights\clip_market1501.pt" echo - Download model weights (see instructions above)
echo.
echo For troubleshooting, see docs/LOCAL_SETUP_GUIDE.md

pause