# SpotOn Backend - Local Environment Setup Script (PowerShell)
# This script sets up the complete local development environment

Write-Host "🚀 SpotOn Backend - Local Environment Setup" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "app\main.py")) {
    Write-Host "❌ ERROR: Please run this script from the spoton_backend directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment is activated
$inVenv = $false
try {
    $result = python -c "import sys; print(hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))" 2>$null
    if ($result -eq "True") {
        $inVenv = $true
        Write-Host "✅ Virtual environment is activated" -ForegroundColor Green
    }
} catch {
    # Ignore error
}

if (-not $inVenv) {
    Write-Host "⚠️  WARNING: Virtual environment not activated. Please run:" -ForegroundColor Yellow
    Write-Host "   .venv\Scripts\activate" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🔍 Checking system requirements..." -ForegroundColor Blue

# Check Python version
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check GPU
Write-Host "🎮 Checking GPU..." -ForegroundColor Blue
$hasGpu = $false
try {
    nvidia-smi 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ NVIDIA GPU detected" -ForegroundColor Green
        $hasGpu = $true
    }
} catch {
    Write-Host "⚠️  nvidia-smi not found. GPU acceleration may not work." -ForegroundColor Yellow
}

Write-Host "📦 Installing Python dependencies..." -ForegroundColor Blue

# Upgrade pip and install uv
python -m pip install --upgrade pip | Out-Null
pip install uv | Out-Null

Write-Host "📋 Installing project dependencies..." -ForegroundColor Blue

# Install project dependencies
if (Test-Path "pyproject.toml") {
    uv pip install ".[dev]"
} elseif (Test-Path "requirements.txt") {
    uv pip install -r requirements.txt
} else {
    Write-Host "❌ ERROR: No pyproject.toml or requirements.txt found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install PyTorch
Write-Host "🔥 Installing PyTorch..." -ForegroundColor Blue
if ($hasGpu) {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Installing PyTorch CPU version..." -ForegroundColor Cyan
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "🤖 Setting up model weights..." -ForegroundColor Blue
if (-not (Test-Path "weights")) {
    New-Item -ItemType Directory -Path "weights" | Out-Null
}

$weightsFile = "weights\clip_market1501.pt"
if (-not (Test-Path $weightsFile)) {
    Write-Host "⚠️  Model weights not found. Please download manually:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view" -ForegroundColor Yellow
    Write-Host "2. Download clip_market1501.pt (~600MB)" -ForegroundColor Yellow
    Write-Host "3. Save to: $weightsFile" -ForegroundColor Yellow
} else {
    Write-Host "✅ Model weights found" -ForegroundColor Green
}

Write-Host "⚙️  Setting up environment configuration..." -ForegroundColor Blue
if (-not (Test-Path ".env.local")) {
    if (Test-Path ".env.local.example") {
        Copy-Item ".env.local.example" ".env.local"
        Write-Host "✅ Environment configuration created (.env.local)" -ForegroundColor Green
        Write-Host "⚠️  Please edit .env.local with your S3 credentials" -ForegroundColor Yellow
    } else {
        Write-Host "❌ ERROR: .env.local.example not found" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "ℹ️  Environment configuration already exists" -ForegroundColor Cyan
}

Write-Host "🧪 Verifying installation..." -ForegroundColor Blue

# Test Python environment
Write-Host "Testing Python environment..." -ForegroundColor Cyan
try {
    $testResult = python -c "import torch; import sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python environment test passed" -ForegroundColor Green
        $testResult
    } else {
        Write-Host "⚠️  Python environment test failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Python environment test failed" -ForegroundColor Yellow
}

# Check required files
if (Test-Path "weights\clip_market1501.pt") {
    Write-Host "✅ Model weights found" -ForegroundColor Green
} else {
    Write-Host "⚠️  Model weights not found" -ForegroundColor Yellow
}

# Create startup script
Write-Host "📝 Creating startup script..." -ForegroundColor Blue
$startupScript = @"
@echo off
echo Starting SpotOn Backend...
call .venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env.local
pause
"@
$startupScript | Out-File -FilePath "start_backend_local.bat" -Encoding ASCII
Write-Host "✅ Startup script created (start_backend_local.bat)" -ForegroundColor Green

Write-Host ""
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host "✅ Local environment setup finished successfully" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Ensure PostgreSQL and Redis are installed and running" -ForegroundColor White
Write-Host "2. Edit .env.local with your S3 credentials (if needed)" -ForegroundColor White
Write-Host "3. Start the backend: .\start_backend_local.bat" -ForegroundColor White
Write-Host "4. Test health endpoint: curl http://localhost:8000/health" -ForegroundColor White
Write-Host "5. View API docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

if (-not (Test-Path "weights\clip_market1501.pt")) {
    Write-Host "Manual setup still required:" -ForegroundColor Yellow
    Write-Host "- Download model weights (see instructions above)" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "For detailed setup instructions, see docs\LOCAL_SETUP_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to continue"