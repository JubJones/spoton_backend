@echo off
REM SpotOn Backend - Simple Local Environment Setup Script (Windows)
REM This script sets up the complete local development environment

echo SpotOn Backend - Local Environment Setup
echo ============================================

REM Check if we're in the right directory
if not exist "app\main.py" (
    echo ERROR: Please run this script from the spoton_backend directory
    pause
    exit /b 1
)

REM Check if virtual environment is activated
python -c "import sys; exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" 2>nul
if %errorLevel% == 0 (
    echo SUCCESS: Virtual environment is activated
) else (
    echo WARNING: Virtual environment not activated. Please run:
    echo   .venv\Scripts\activate
    pause
    exit /b 1
)

echo Checking system requirements...

REM Check Python version
python --version
if %errorLevel% neq 0 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Check GPU
echo Checking GPU...
nvidia-smi >nul 2>&1
if %errorLevel% == 0 (
    echo SUCCESS: NVIDIA GPU detected
    set HAS_GPU=1
) else (
    echo WARNING: nvidia-smi not found. GPU acceleration may not work.
    set HAS_GPU=0
)

echo Installing Python dependencies...

REM Upgrade pip and install uv
python -m pip install --upgrade pip
pip install uv

echo Installing project dependencies...

REM Install project dependencies
if exist "pyproject.toml" (
    uv pip install ".[dev]"
) else if exist "requirements.txt" (
    uv pip install -r requirements.txt
) else (
    echo ERROR: No pyproject.toml or requirements.txt found
    pause
    exit /b 1
)

REM Install PyTorch
echo Installing PyTorch...
if %HAS_GPU%==1 (
    echo Installing PyTorch with CUDA support...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing PyTorch CPU version...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo Setting up model weights...
if not exist "weights" mkdir weights

set WEIGHTS_FILE=weights\clip_market1501.pt
if not exist "%WEIGHTS_FILE%" (
    echo WARNING: Model weights not found. Please download manually:
    echo 1. Visit: https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view
    echo 2. Download clip_market1501.pt (~600MB)
    echo 3. Save to: %WEIGHTS_FILE%
) else (
    echo SUCCESS: Model weights found
)

echo Setting up environment configuration...
if not exist ".env.local" (
    if exist ".env.local.example" (
        copy ".env.local.example" ".env.local" >nul
        echo SUCCESS: Environment configuration created (.env.local)
        echo WARNING: Please edit .env.local with your S3 credentials
    ) else (
        echo ERROR: .env.local.example not found
        pause
        exit /b 1
    )
) else (
    echo INFO: Environment configuration already exists
)

echo Verifying installation...

REM Test Python environment
echo Testing Python environment...
python -c "import torch; import sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>nul
if %errorLevel% == 0 (
    echo SUCCESS: Python environment test passed
) else (
    echo WARNING: Python environment test failed
)

REM Check required files
if exist "weights\clip_market1501.pt" (
    echo SUCCESS: Model weights found
) else (
    echo WARNING: Model weights not found
)

REM Create startup script
echo Creating startup script...
echo @echo off > start_backend_local.bat
echo echo Starting SpotOn Backend... >> start_backend_local.bat
echo call .venv\Scripts\activate.bat >> start_backend_local.bat
echo python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env.local >> start_backend_local.bat
echo pause >> start_backend_local.bat

echo SUCCESS: Startup script created (start_backend_local.bat)

echo.
echo Setup Complete!
echo ==================
echo SUCCESS: Local environment setup finished successfully
echo.
echo Next steps:
echo 1. Ensure PostgreSQL and Redis are installed and running
echo 2. Edit .env.local with your S3 credentials (if needed)
echo 3. Start the backend: start_backend_local.bat
echo 4. Test health endpoint: curl http://localhost:8000/health
echo 5. View API docs: http://localhost:8000/docs
echo.

if not exist "weights\clip_market1501.pt" (
    echo Manual setup still required:
    echo - Download model weights (see instructions above)
)

echo For detailed setup instructions, see docs\LOCAL_SETUP_GUIDE.md

pause