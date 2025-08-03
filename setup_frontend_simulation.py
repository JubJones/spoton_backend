#!/usr/bin/env python3
"""
Setup script for SpotOn Frontend Simulation Client
Automatically installs required dependencies if missing
"""

import subprocess
import sys
import importlib

def install_package(package_name, pip_name=None):
    """Install a package using pip if not already available"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"📦 Installing {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {pip_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {pip_name}: {e}")
            return False

def main():
    """Main setup function"""
    print("🚀 Setting up SpotOn Frontend Simulation Client...")
    
    # Required packages
    packages = [
        ("aiohttp", "aiohttp>=3.8.0"),
        ("websockets", "websockets>=10.0")
    ]
    
    all_installed = True
    for module_name, pip_name in packages:
        if not install_package(module_name, pip_name):
            all_installed = False
    
    if all_installed:
        print("\n✅ All dependencies are installed!")
        print("\n🎯 You can now run the frontend simulation:")
        print("   python frontend_simulation_client.py")
        
        # Test import to verify everything works
        try:
            import aiohttp
            import websockets
            print(f"\n📋 Installed versions:")
            print(f"   - aiohttp: {aiohttp.__version__}")
            print(f"   - websockets: {websockets.__version__}")
        except ImportError as e:
            print(f"\n⚠️  Import test failed: {e}")
            return False
    else:
        print("\n❌ Some dependencies failed to install")
        print("   Please install them manually:")
        print("   pip install aiohttp>=3.8.0 websockets>=10.0")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)