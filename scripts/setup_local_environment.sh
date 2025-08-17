#!/bin/bash

# SpotOn Backend - Local Environment Setup Script (Linux/macOS)
# This script sets up the complete local development environment

set -e  # Exit on any error

echo "🚀 SpotOn Backend - Local Environment Setup"
echo "============================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [[ -f /etc/ubuntu-release ]] || [[ -f /etc/debian_version ]]; then
        DISTRO="debian"
    elif [[ -f /etc/redhat-release ]]; then
        DISTRO="redhat"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    print_error "Use setup_local_environment.bat for Windows"
    exit 1
fi

print_info "Detected OS: $OS"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python 3.9+
    if command_exists python3.9; then
        PYTHON_CMD="python3.9"
    elif command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        if [[ "$(echo "$PYTHON_VERSION >= 3.9" | bc -l)" -eq 1 ]]; then
            PYTHON_CMD="python3"
        else
            print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3.9+ not found. Please install Python 3.9 or higher."
        exit 1
    fi
    
    print_status "Python found: $PYTHON_CMD"
    
    # Check GPU
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_status "NVIDIA GPU detected: $GPU_INFO"
    else
        print_warning "nvidia-smi not found. GPU acceleration may not work."
    fi
    
    # Check Git
    if ! command_exists git; then
        print_error "Git not found. Please install Git."
        exit 1
    fi
    print_status "Git found"
}

# Install system dependencies
install_system_dependencies() {
    print_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        if [[ "$DISTRO" == "debian" ]]; then
            # Ubuntu/Debian
            print_info "Installing dependencies for Ubuntu/Debian..."
            
            sudo apt update
            
            # Development tools
            sudo apt install -y build-essential cmake git curl wget
            sudo apt install -y python3-dev python3-pip python3-venv
            
            # PostgreSQL with TimescaleDB
            print_info "Installing PostgreSQL with TimescaleDB..."
            sudo apt install -y postgresql postgresql-contrib
            
            # Add TimescaleDB repository
            if ! grep -q "timescaledb" /etc/apt/sources.list.d/timescaledb.list 2>/dev/null; then
                echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
                wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
                sudo apt update
            fi
            sudo apt install -y timescaledb-2-postgresql-14
            
            # Redis
            print_info "Installing Redis..."
            sudo apt install -y redis-server
            
            # Start and enable services
            sudo systemctl enable postgresql redis-server
            sudo systemctl start postgresql redis-server
            
        elif [[ "$DISTRO" == "redhat" ]]; then
            # CentOS/RHEL/Fedora
            print_info "Installing dependencies for RedHat/CentOS/Fedora..."
            
            if command_exists dnf; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
            
            sudo $PKG_MANAGER install -y gcc gcc-c++ cmake git curl wget
            sudo $PKG_MANAGER install -y python39 python39-devel python39-pip
            sudo $PKG_MANAGER install -y postgresql postgresql-server postgresql-contrib
            sudo $PKG_MANAGER install -y redis
            
            # Initialize PostgreSQL
            if [[ ! -d "/var/lib/pgsql/data" ]]; then
                sudo postgresql-setup initdb
            fi
            
            sudo systemctl enable postgresql redis
            sudo systemctl start postgresql redis
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        print_info "Installing dependencies for macOS..."
        
        # Check if Homebrew is installed
        if ! command_exists brew; then
            print_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install dependencies
        brew install postgresql@14 redis python@3.9 git cmake
        
        # Start services
        brew services start postgresql@14
        brew services start redis
        
        PYTHON_CMD="/opt/homebrew/bin/python3.9"  # Homebrew Python path
    fi
    
    print_status "System dependencies installed"
}

# Setup PostgreSQL database
setup_postgresql() {
    print_info "Setting up PostgreSQL database..."
    
    # Check if PostgreSQL is running
    if ! pgrep -x "postgres" > /dev/null; then
        print_error "PostgreSQL is not running. Please start PostgreSQL service."
        exit 1
    fi
    
    # Create database user and database
    sudo -u postgres psql -tc "SELECT 1 FROM pg_user WHERE usename = 'spoton_user'" | grep -q 1 || \
        sudo -u postgres psql -c "CREATE USER spoton_user WITH PASSWORD 'spoton_password';"
    
    sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'spotondb'" | grep -q 1 || \
        sudo -u postgres psql -c "CREATE DATABASE spotondb OWNER spoton_user;"
    
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE spotondb TO spoton_user;"
    
    # Enable TimescaleDB extension
    sudo -u postgres psql -d spotondb -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" || print_warning "TimescaleDB extension not installed"
    
    print_status "PostgreSQL database configured"
}

# Setup Redis
setup_redis() {
    print_info "Setting up Redis..."
    
    # Check if Redis is running
    if ! pgrep -x "redis-server" > /dev/null; then
        print_warning "Redis is not running. Attempting to start..."
        if [[ "$OS" == "linux" ]]; then
            sudo systemctl start redis-server
        elif [[ "$OS" == "macos" ]]; then
            brew services start redis
        fi
    fi
    
    # Test Redis connection
    if redis-cli ping >/dev/null 2>&1; then
        print_status "Redis configured and running"
    else
        print_warning "Redis connection test failed"
    fi
}

# Setup Python virtual environment
setup_python_environment() {
    print_info "Setting up Python virtual environment..."
    
    # Create virtual environment
    if [[ ! -d ".venv" ]]; then
        $PYTHON_CMD -m venv .venv
        print_status "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip and install uv
    pip install --upgrade pip
    pip install uv
    
    print_status "Python virtual environment ready"
}

# Install Python dependencies
install_python_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install project dependencies
    if [[ -f "pyproject.toml" ]]; then
        uv pip install ".[dev]"
    elif [[ -f "requirements.txt" ]]; then
        uv pip install -r requirements.txt
    else
        print_error "No pyproject.toml or requirements.txt found"
        exit 1
    fi
    
    # Install PyTorch with CUDA support (if GPU available)
    if command_exists nvidia-smi; then
        print_info "Installing PyTorch with CUDA support..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        uv pip install cupy-cuda11x
    else
        print_info "Installing PyTorch CPU version..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_status "Python dependencies installed"
}

# Download model weights
download_model_weights() {
    print_info "Downloading model weights..."
    
    mkdir -p weights
    
    WEIGHTS_FILE="weights/clip_market1501.pt"
    WEIGHTS_URL="https://drive.google.com/uc?id=1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7"
    
    if [[ ! -f "$WEIGHTS_FILE" ]]; then
        print_info "Downloading CLIP model weights (~600MB)..."
        
        # Try downloading with wget
        if command_exists wget; then
            wget --no-check-certificate "$WEIGHTS_URL" -O "$WEIGHTS_FILE" || {
                print_warning "Automatic download failed. Please download manually:"
                print_warning "1. Visit: https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view"
                print_warning "2. Download clip_market1501.pt"
                print_warning "3. Move to: $WEIGHTS_FILE"
            }
        elif command_exists curl; then
            curl -L "$WEIGHTS_URL" -o "$WEIGHTS_FILE" || {
                print_warning "Automatic download failed. Please download manually (see above)"
            }
        else
            print_warning "Neither wget nor curl found. Please download manually (see above)"
        fi
    else
        print_status "Model weights already exist"
    fi
    
    # Check file size
    if [[ -f "$WEIGHTS_FILE" ]]; then
        FILE_SIZE=$(stat -c%s "$WEIGHTS_FILE" 2>/dev/null || stat -f%z "$WEIGHTS_FILE" 2>/dev/null)
        if [[ $FILE_SIZE -gt 500000000 ]]; then  # ~500MB minimum
            print_status "Model weights downloaded ($(($FILE_SIZE / 1024 / 1024))MB)"
        else
            print_warning "Model weights file seems too small. Please verify download."
        fi
    fi
}

# Setup environment configuration
setup_environment_config() {
    print_info "Setting up environment configuration..."
    
    if [[ ! -f ".env.local" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env.local
            
            # Update database URLs for local setup
            if [[ "$OS" == "macos" ]]; then
                # macOS might have different socket paths
                sed -i '' 's/POSTGRES_SERVER=.*/POSTGRES_SERVER=localhost/' .env.local
                sed -i '' 's/REDIS_HOST=.*/REDIS_HOST=localhost/' .env.local
            else
                sed -i 's/POSTGRES_SERVER=.*/POSTGRES_SERVER=localhost/' .env.local
                sed -i 's/REDIS_HOST=.*/REDIS_HOST=localhost/' .env.local
            fi
            
            print_status "Environment configuration created (.env.local)"
        else
            print_error ".env.example not found. Please create environment configuration manually."
            exit 1
        fi
    else
        print_info "Environment configuration already exists"
    fi
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Test Python imports
    python -c "
import torch
import sys
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" || print_warning "Python environment test failed"
    
    # Test database connections
    print_info "Testing database connections..."
    
    # Test PostgreSQL
    if psql -h localhost -U spoton_user -d spotondb -c "SELECT version();" >/dev/null 2>&1; then
        print_status "PostgreSQL connection successful"
    else
        print_warning "PostgreSQL connection failed"
    fi
    
    # Test Redis
    if redis-cli ping >/dev/null 2>&1; then
        print_status "Redis connection successful"
    else
        print_warning "Redis connection failed"
    fi
    
    # Check required files
    print_info "Checking required files..."
    
    if [[ -f "weights/clip_market1501.pt" ]]; then
        print_status "Model weights found"
    else
        print_warning "Model weights not found"
    fi
    
    if [[ -d "homography_points" ]] && [[ $(ls homography_points/*.npz 2>/dev/null | wc -l) -gt 0 ]]; then
        print_status "Homography data found"
    else
        print_warning "Homography data not found"
    fi
}

# Main execution
main() {
    print_info "Starting local environment setup..."
    
    check_requirements
    install_system_dependencies
    setup_postgresql
    setup_redis
    setup_python_environment
    install_python_dependencies
    download_model_weights
    setup_environment_config
    verify_installation
    
    echo ""
    echo "🎉 Setup Complete!"
    echo "=================="
    print_status "Local environment setup finished successfully"
    echo ""
    echo "Next steps:"
    echo "1. Activate Python environment: source .venv/bin/activate"
    echo "2. Start the backend: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env.local"
    echo "3. Test health endpoint: curl http://localhost:8000/health"
    echo "4. View API docs: http://localhost:8000/docs"
    echo ""
    echo "For troubleshooting, see docs/LOCAL_SETUP_GUIDE.md"
}

# Run main function
main "$@"