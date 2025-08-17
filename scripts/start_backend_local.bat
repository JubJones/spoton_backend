@echo off
REM SpotOn Backend - Local Development Startup Script (Windows)
REM This script starts the local backend with all dependencies

setlocal enabledelayedexpansion

echo 🚀 Starting SpotOn Backend (Local Development)
echo =============================================

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

REM Check if we're in the right directory
if not exist "app\main.py" (
    call :print_error "Please run this script from the spoton_backend directory"
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv" (
    call :print_error "Virtual environment not found. Please run setup_local_environment.bat first"
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
call :print_status "Virtual environment activated"

REM Check environment file
if not exist ".env.local" (
    if exist ".env.local.example" (
        call :print_warning "No .env.local found. Copying from .env.local.example..."
        copy ".env.local.example" ".env.local" >nul
        call :print_warning "Please edit .env.local with your S3 credentials"
    ) else (
        call :print_error "No .env.local or .env.local.example found"
        pause
        exit /b 1
    )
)

REM Check dependencies
call :print_info "Checking dependencies..."

REM Check Redis
redis-cli ping >nul 2>&1
if !errorLevel! == 0 (
    call :print_status "Redis is running"
) else (
    call :print_warning "Redis is not running"
    call :print_info "Please start Redis manually or run Redis server"
)

REM Check PostgreSQL
pg_isready -h localhost -U spoton_user >nul 2>&1
if !errorLevel! == 0 (
    call :print_status "PostgreSQL is running"
) else (
    call :print_error "PostgreSQL is not running. Please start PostgreSQL service"
    call :print_info "Check Windows services or start PostgreSQL manually"
    pause
    exit /b 1
)

REM Check database connection
set PGPASSWORD=spoton_password
psql -h localhost -U spoton_user -d spotondb -c "SELECT 1" >nul 2>&1
if !errorLevel! == 0 (
    call :print_status "Database connection successful"
) else (
    call :print_error "Cannot connect to SpotOn database. Please run database setup:"
    call :print_info "psql -U postgres -f scripts\setup_database_local.sql"
    pause
    exit /b 1
)

call :print_status "All dependencies are ready"

REM Check required files
call :print_info "Checking required files..."

if not exist "weights\clip_market1501.pt" (
    call :print_warning "Model weights not found at weights\clip_market1501.pt"
    call :print_info "Please download from: https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view"
)

if not exist "homography_points\*.npz" (
    call :print_warning "Homography data not found in homography_points\"
    call :print_info "Some coordinate transformations may not work without this data"
)

REM Set up environment
set PYTHONPATH=%PYTHONPATH%;%cd%
set ENV_FILE=.env.local

call :print_info "Environment variables:"
call :print_info "  PYTHONPATH: %PYTHONPATH%"
call :print_info "  ENV_FILE: %ENV_FILE%"

REM Display system information
call :print_info "System information:"
python -c "import torch; import sys; print(f'  Python: {sys.version.split()[0]}'); print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); torch.cuda.is_available() and print(f'  CUDA version: {torch.version.cuda}') and print(f'  GPU count: {torch.cuda.device_count()}')" 2>nul || call :print_warning "Could not display PyTorch information"

REM Start the backend
call :print_info "Starting SpotOn Backend server..."
call :print_info "Server will be available at: http://localhost:8000"
call :print_info "API documentation: http://localhost:8000/docs"
call :print_info "Health check: http://localhost:8000/health"
echo.
call :print_info "Press Ctrl+C to stop the server"
echo.

REM Start uvicorn with local configuration
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env.local --log-level info

pause