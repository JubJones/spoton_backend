#!/usr/bin/env python3
"""
SpotOn Backend - Local Setup Validation Script
This script validates that the local development environment is set up correctly.
"""

import os
import sys
import subprocess
import platform
import socket
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    @classmethod
    def disable_colors(cls):
        """Disable colors for Windows compatibility."""
        cls.GREEN = cls.RED = cls.YELLOW = cls.BLUE = cls.BOLD = cls.END = ''


def print_status(message: str, success: bool = True) -> None:
    """Print a status message with color coding."""
    icon = "✅" if success else "❌"
    color = Colors.GREEN if success else Colors.RED
    print(f"{color}{icon} {message}{Colors.END}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")


def print_header(message: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{message}{Colors.END}")
    print("=" * len(message))


def check_command_exists(command: str) -> bool:
    """Check if a command exists in the system PATH."""
    try:
        subprocess.run([command, "--version"], 
                      capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_python_version() -> Tuple[bool, str]:
    """Check Python version requirements."""
    version = sys.version_info
    if version >= (3, 9):
        return True, f"{version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"{version.major}.{version.minor}.{version.micro} (requires 3.9+)"


def check_virtual_environment() -> Tuple[bool, str]:
    """Check if we're running in a virtual environment."""
    venv_path = Path(".venv")
    if venv_path.exists():
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        if in_venv:
            return True, f"Active virtual environment at {venv_path.absolute()}"
        else:
            return False, f"Virtual environment exists at {venv_path.absolute()} but not activated"
    else:
        return False, "Virtual environment not found (.venv directory missing)"


def check_python_packages() -> Dict[str, Tuple[bool, str]]:
    """Check if required Python packages are installed."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "torch",
        "torchvision", 
        "redis",
        "psycopg2-binary",
        "sqlalchemy",
        "pydantic",
        "opencv-python",
        "numpy",
        "pillow"
    ]
    
    results = {}
    for package in required_packages:
        try:
            if package == "opencv-python":
                import cv2
                results[package] = (True, f"OpenCV {cv2.__version__}")
            elif package == "psycopg2-binary":
                import psycopg2
                results[package] = (True, f"psycopg2 {psycopg2.__version__}")
            elif package == "torch":
                import torch
                cuda_info = f" (CUDA: {torch.cuda.is_available()})" if torch.cuda.is_available() else " (CPU only)"
                results[package] = (True, f"PyTorch {torch.__version__}{cuda_info}")
            else:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                results[package] = (True, f"Version {version}")
        except ImportError as e:
            results[package] = (False, f"Not installed: {e}")
    
    return results


def check_port_available(host: str, port: int, timeout: int = 5) -> bool:
    """Check if a port is available (not in use)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # Port is available if connection failed
    except Exception:
        return False


def check_service_running(service_name: str, host: str, port: int) -> Tuple[bool, str]:
    """Check if a service is running and responding."""
    if service_name.lower() == "redis":
        try:
            import redis
            r = redis.Redis(host=host, port=port, socket_timeout=5)
            r.ping()
            info = r.info()
            return True, f"Redis {info.get('redis_version', 'unknown')} running"
        except Exception as e:
            return False, f"Redis connection failed: {e}"
    
    elif service_name.lower() == "postgresql":
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=host,
                port=port,
                database="spotondb",
                user="spoton_user", 
                password="spoton_password",
                connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            conn.close()
            return True, f"PostgreSQL running: {version[:50]}..."
        except Exception as e:
            return False, f"PostgreSQL connection failed: {e}"
    
    else:
        # Generic port check
        if not check_port_available(host, port):
            return True, f"Service responding on {host}:{port}"
        else:
            return False, f"No service responding on {host}:{port}"


def check_environment_file() -> Tuple[bool, str, Dict[str, str]]:
    """Check if environment configuration exists and load variables."""
    env_files = [".env.local", ".env"]
    env_vars = {}
    
    for env_file in env_files:
        if Path(env_file).exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip().strip('"\'')
                return True, f"Environment file found: {env_file}", env_vars
            except Exception as e:
                return False, f"Error reading {env_file}: {e}", {}
    
    return False, "No environment file found (.env.local or .env)", {}


def check_required_files() -> Dict[str, Tuple[bool, str]]:
    """Check if required files and directories exist."""
    required_items = {
        "weights/clip_market1501.pt": "AI model weights",
        "homography_points/": "Homography transformation data",
        "app/main.py": "Main FastAPI application",
        "app/core/config.py": "Configuration module",
    }
    
    results = {}
    for item_path, description in required_items.items():
        path = Path(item_path)
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                results[item_path] = (True, f"{description} ({size_mb:.1f}MB)")
            else:
                # Directory
                file_count = len(list(path.glob("*")))
                results[item_path] = (True, f"{description} ({file_count} files)")
        else:
            results[item_path] = (False, f"{description} - Missing")
    
    return results


def check_gpu_support() -> Tuple[bool, str]:
    """Check for GPU support and CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            return True, f"CUDA available - {gpu_count} GPU(s): {', '.join(gpu_names[:2])}"
        else:
            return False, "CUDA not available (CPU-only mode)"
    except ImportError:
        return False, "PyTorch not installed - cannot check GPU"


def test_basic_imports() -> Dict[str, Tuple[bool, str]]:
    """Test importing core application modules."""
    test_imports = {
        "app.main": "Main FastAPI application",
        "app.core.config": "Configuration settings", 
        "app.models": "AI models package",
        "app.services": "Services package",
    }
    
    results = {}
    for module_name, description in test_imports.items():
        try:
            __import__(module_name)
            results[module_name] = (True, f"{description} - OK")
        except ImportError as e:
            results[module_name] = (False, f"{description} - Import error: {e}")
        except Exception as e:
            results[module_name] = (False, f"{description} - Error: {e}")
    
    return results


def main():
    """Main validation function."""
    print_header("🚀 SpotOn Backend - Local Setup Validation")
    
    # Detect Windows and disable colors if needed
    if platform.system() == "Windows":
        Colors.disable_colors()
    
    # Overall validation status
    all_checks_passed = True
    warnings = []
    
    # 1. System Requirements
    print_header("1. System Requirements")
    
    # Python version
    py_ok, py_version = check_python_version()
    print_status(f"Python version: {py_version}", py_ok)
    if not py_ok:
        all_checks_passed = False
    
    # Virtual environment
    venv_ok, venv_msg = check_virtual_environment()
    print_status(venv_msg, venv_ok)
    if not venv_ok:
        all_checks_passed = False
    
    # System commands
    commands = ["git", "redis-cli", "psql"]
    for cmd in commands:
        cmd_ok = check_command_exists(cmd)
        print_status(f"{cmd} command available", cmd_ok)
        if not cmd_ok:
            warnings.append(f"{cmd} command not found in PATH")
    
    # 2. Python Dependencies
    print_header("2. Python Dependencies")
    
    packages = check_python_packages()
    for package, (status, info) in packages.items():
        print_status(f"{package}: {info}", status)
        if not status:
            all_checks_passed = False
    
    # 3. Environment Configuration
    print_header("3. Environment Configuration")
    
    env_ok, env_msg, env_vars = check_environment_file()
    print_status(env_msg, env_ok)
    if not env_ok:
        all_checks_passed = False
    else:
        # Check important environment variables
        important_vars = [
            "POSTGRES_SERVER", "POSTGRES_USER", "POSTGRES_PASSWORD", 
            "REDIS_HOST", "S3_ENDPOINT_URL"
        ]
        for var in important_vars:
            if var in env_vars:
                value = env_vars[var][:20] + "..." if len(env_vars[var]) > 20 else env_vars[var]
                print_info(f"  {var}: {value}")
            else:
                print_warning(f"  {var}: Not set")
                warnings.append(f"Environment variable {var} not set")
    
    # 4. Required Files
    print_header("4. Required Files and Directories")
    
    files = check_required_files()
    for file_path, (status, info) in files.items():
        print_status(f"{file_path}: {info}", status)
        if not status:
            if "weights" in file_path:
                warnings.append("AI model weights missing - some features won't work")
            elif "homography" in file_path:
                warnings.append("Homography data missing - coordinate transformations disabled")
            else:
                all_checks_passed = False
    
    # 5. Services
    print_header("5. External Services")
    
    # Redis
    redis_ok, redis_msg = check_service_running("redis", "localhost", 6379)
    print_status(f"Redis: {redis_msg}", redis_ok)
    if not redis_ok:
        all_checks_passed = False
    
    # PostgreSQL
    pg_ok, pg_msg = check_service_running("postgresql", "localhost", 5432)
    print_status(f"PostgreSQL: {pg_msg}", pg_ok)
    if not pg_ok:
        all_checks_passed = False
    
    # Port availability
    port_8000_available = check_port_available("localhost", 8000)
    print_status(f"Port 8000 available for backend", port_8000_available)
    if not port_8000_available:
        warnings.append("Port 8000 is in use - backend may need different port")
    
    # 6. GPU Support (Optional)
    print_header("6. GPU Support (Optional)")
    
    gpu_ok, gpu_msg = check_gpu_support()
    print_status(f"GPU Support: {gpu_msg}", gpu_ok)
    if not gpu_ok:
        print_info("  GPU support is optional for local development")
    
    # 7. Application Imports
    print_header("7. Application Module Tests")
    
    imports = test_basic_imports()
    for module, (status, info) in imports.items():
        print_status(f"{module}: {info}", status)
        if not status:
            all_checks_passed = False
    
    # Final Summary
    print_header("🎯 Validation Summary")
    
    if all_checks_passed and not warnings:
        print_status("All checks passed! ✨ Your local environment is ready.", True)
        print_info("You can now start the backend with:")
        print_info("  ./scripts/start_backend_local.sh (Linux/macOS)")
        print_info("  scripts\\start_backend_local.bat (Windows)")
    elif all_checks_passed:
        print_status("Core requirements met with warnings ⚠️", True)
        print_info("Your environment should work, but consider addressing these warnings:")
        for warning in warnings:
            print_warning(f"  • {warning}")
    else:
        print_status("Validation failed ❌ - Setup incomplete", False)
        print_info("Please address the failed checks above.")
        print_info("See docs/TROUBLESHOOTING.md for help.")
    
    # Exit code
    sys.exit(0 if all_checks_passed else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {e}")
        sys.exit(1)