# SpotOn Backend - Windows CUDA Setup Script
# This script installs everything in the correct order to preserve CUDA PyTorch

Write-Host "🚀 SpotOn Backend - Windows CUDA Setup" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

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
    Write-Host "⚠️  Please activate virtual environment first:" -ForegroundColor Yellow
    Write-Host "   .venv\Scripts\activate" -ForegroundColor Yellow
    exit 1
}

# Step 1: Install PyTorch with CUDA first
Write-Host "🔥 Step 1: Installing PyTorch with CUDA..." -ForegroundColor Blue
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA PyTorch
$cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaCheck -eq "True") {
    Write-Host "✅ CUDA PyTorch installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ CUDA PyTorch installation failed" -ForegroundColor Red
    exit 1
}

# Step 2: Install other dependencies (excluding PyTorch)
Write-Host "📦 Step 2: Installing other dependencies..." -ForegroundColor Blue

# Install dependencies manually to avoid PyTorch conflicts
$dependencies = @(
    "fastapi==0.110.3",
    "uvicorn[standard]==0.29.0",
    "pydantic==2.7.4",
    "pydantic-settings==2.2.1",
    "sqlalchemy==2.0.43",
    "psycopg2-binary==2.9.10",
    "redis==5.0.8",
    "opencv-python==4.11.0.86",
    "numpy==1.26.4",
    "pandas==2.3.1",
    "matplotlib==3.10.5",
    "seaborn==0.13.2",
    "scikit-learn==1.7.1",
    "scipy==1.11.4",
    "transformers==4.36.2",
    "huggingface-hub==0.34.4",
    "faiss-cpu==1.7.4",
    "boxmot==15.0.2",
    "ultralytics==8.0.239",
    "dagshub==0.3.47",
    "boto3==1.34.162",
    "python-dotenv==1.1.1",
    "loguru==0.7.3",
    "pytest==8.2.2",
    "pytest-asyncio==0.23.8",
    "pytest-mock==3.12.0",
    "ruff==0.4.10"
)

foreach ($dep in $dependencies) {
    Write-Host "Installing $dep..." -ForegroundColor Cyan
    uv pip install $dep
}

# Step 3: Verify CUDA is still working
Write-Host "🧪 Step 3: Verifying CUDA PyTorch..." -ForegroundColor Blue
$verification = python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CPU only')" 2>$null

if ($verification -like "*CUDA: True*") {
    Write-Host "✅ CUDA verification successful:" -ForegroundColor Green
    Write-Host $verification -ForegroundColor White
} else {
    Write-Host "❌ CUDA verification failed - PyTorch was overwritten" -ForegroundColor Red
    Write-Host "Reinstalling CUDA PyTorch..." -ForegroundColor Yellow
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
}

# Step 4: Set up configuration
Write-Host "⚙️  Step 4: Setting up configuration..." -ForegroundColor Blue

if (-not (Test-Path ".env.local")) {
    if (Test-Path ".env.local.example") {
        Copy-Item ".env.local.example" ".env.local"
        Write-Host "✅ Environment configuration created (.env.local)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  .env.local.example not found, copying from .env" -ForegroundColor Yellow
        if (Test-Path ".env") {
            Copy-Item ".env" ".env.local"
            
            # Update for local Windows setup
            $content = Get-Content ".env.local"
            $content = $content -replace "POSTGRES_SERVER=.*", "POSTGRES_SERVER=localhost"
            $content = $content -replace "REDIS_HOST=.*", "REDIS_HOST=localhost"
            $content | Set-Content ".env.local"
            
            Write-Host "✅ Environment configuration created and updated for local setup" -ForegroundColor Green
        }
    }
}

# Step 5: Create model weights directory
Write-Host "🤖 Step 5: Setting up model weights..." -ForegroundColor Blue
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
    $fileSize = (Get-Item $weightsFile).Length / 1MB
    Write-Host "✅ Model weights found (${fileSize:N0} MB)" -ForegroundColor Green
}

# Step 6: Final verification
Write-Host "🎯 Step 6: Final verification..." -ForegroundColor Blue

try {
    $importTest = python -c "import torch, fastapi, uvicorn, redis, cv2; print('All critical imports successful')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ All critical imports successful" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some imports failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Import test failed" -ForegroundColor Yellow
}

# Create startup script
$startupScript = @"
@echo off
echo Starting SpotOn Backend with CUDA support...
call .venv\Scripts\activate.bat
echo Testing CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo Starting server...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env.local
pause
"@
$startupScript | Out-File -FilePath "start_backend_cuda.bat" -Encoding ASCII

Write-Host ""
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host "=================" -ForegroundColor Green
Write-Host "✅ SpotOn Backend with CUDA support is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Download model weights (if not done): see instructions above" -ForegroundColor White
Write-Host "2. Start backend: .\start_backend_cuda.bat" -ForegroundColor White
Write-Host "3. Test health: curl http://localhost:8000/health" -ForegroundColor White
Write-Host "4. View docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Important: This setup preserves CUDA PyTorch for GPU acceleration" -ForegroundColor Yellow
Read-Host "Press Enter to continue"