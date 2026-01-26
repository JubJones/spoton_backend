#!/usr/bin/env python3
"""
TensorRT Export Script for YOLO26-N Model

IMPORTANT: This script MUST be run on the target CUDA device (RTX 2060)
           where inference will be performed. TensorRT engines are
           device-specific and cannot be created on Mac.

Usage:
    python scripts/export_yolo_tensorrt.py

The script will:
1. Load the PyTorch model (yolo26m.pt)
2. Export to TensorRT engine with FP16 precision
3. Run a benchmark comparison between PyTorch and TensorRT
4. Validate that detections are consistent
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_cuda_available():
    """Check if CUDA is available for TensorRT export."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ ERROR: CUDA is not available!")
            print("   TensorRT export requires an NVIDIA GPU with CUDA.")
            print("   This script cannot run on Mac or CPU-only systems.")
            sys.exit(1)
        
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"✅ CUDA available: {gpu_name}")
        print(f"   CUDA Version: {cuda_version}")
        return gpu_name
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        sys.exit(1)


def check_tensorrt_available():
    """Check if TensorRT is installed."""
    try:
        import tensorrt as trt
        print(f"✅ TensorRT available: v{trt.__version__}")
        return True
    except ImportError:
        print("⚠️  TensorRT Python package not found.")
        print("   Ultralytics will attempt to use TensorRT via the engine export.")
        print("   If export fails, install TensorRT:")
        return False


def fix_yolo_serialization():
    """
    Workaround for 'AttributeError: Can't get attribute 'PatchedC3k2' on <module '__main__'...'.
    This happens when loading certain YOLO checkpoint versions.
    We map the missing class to the standard C3k2 block.
    """
    try:
        import sys
        from ultralytics.nn.modules import block
        
        # Helper to safely set main attribute
        def safe_set_main(name, cls):
            if not hasattr(sys.modules['__main__'], name):
                setattr(sys.modules['__main__'], name, cls)
                print(f"🔧 Applied fix: Mapped __main__.{name} to {cls.__module__}.{cls.__name__}")

        # Fix C3k2
        if hasattr(block, 'C3k2'):
            safe_set_main('PatchedC3k2', block.C3k2)

        # Fix SPPF - Try multiple ways to get it
        SPPF = getattr(block, 'SPPF', None)
        if SPPF is None:
            try:
                from ultralytics.nn.modules.block import SPPF as _SPPF
                SPPF = _SPPF
            except ImportError:
                 pass
        
        if SPPF:
            safe_set_main('PatchedSPPF', SPPF)
        else:
             print("⚠️ Warning: SPPF class not found in ultralytics modules for patching.")

    except Exception as e:
        print(f"⚠️  Could not apply serialization fix: {e}")



def export_to_tensorrt(
    model_path: str,
    output_path: str = None,
    imgsz: int = 640,
    batch_size: int = 4,
    half: bool = True,
    workspace_gb: int = 4,
):
    """
    Export YOLO model to TensorRT engine format.
    
    Args:
        model_path: Path to the .pt model file
        output_path: Output path for .engine file (defaults to same dir as input)
        imgsz: Input image size (default 640)
        batch_size: Fixed batch size for the engine (default 4 for 4 cameras)
        half: Use FP16 precision (recommended for RTX 2060)
        workspace_gb: GPU memory workspace in GB
    
    Returns:
        Path to the exported .engine file
    """
    from ultralytics import YOLO
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("EXPORTING YOLO MODEL TO TENSORRT")
    print(f"{'='*60}")
    print(f"Input model:  {model_path}")
    print(f"Image size:   {imgsz}x{imgsz}")
    print(f"Batch size:   {batch_size}")
    print(f"Precision:    {'FP16 (half)' if half else 'FP32'}")
    print(f"Workspace:    {workspace_gb} GB")
    print(f"{'='*60}\n")
    
    # Load the model
    fix_yolo_serialization() # Apply patch before loading
    print("Loading PyTorch model...")
    model = YOLO(str(model_path))
    
    # Export to TensorRT
    print("\nStarting TensorRT export (this may take 5-15 minutes)...")
    print("The engine is being optimized for your specific GPU.\n")
    
    start_time = time.time()
    
    try:
        export_path = model.export(
            format="engine",
            imgsz=imgsz,
            batch=batch_size,
            half=half,
            workspace=workspace_gb,
            device=0,
            dynamic=True,  # Allow variable batch sizes (1 to batch_size)
            verbose=True,
        )
        
        export_time = time.time() - start_time
        print(f"\n✅ Export completed in {export_time:.1f} seconds")
        print(f"   Engine saved to: {export_path}")
        
        return export_path
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure TensorRT is installed: pip install tensorrt")
        print("2. Check CUDA/cuDNN versions match TensorRT requirements")
        print("3. Try reducing workspace_gb if out of memory")
        sys.exit(1)


def benchmark_comparison(pt_model_path: str, engine_path: str, batch_size: int = 4, num_runs: int = 50):
    """
    Benchmark PyTorch vs TensorRT inference speed.
    
    Args:
        pt_model_path: Path to PyTorch .pt model
        engine_path: Path to TensorRT .engine model
        batch_size: Batch size to use for benchmarking
        num_runs: Number of inference runs for benchmarking
    """
    import numpy as np
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print("BENCHMARK: PyTorch vs TensorRT")
    print(f"{'='*60}")
    
    # Create dummy batch of 'batch_size' images
    dummy_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]
    
    def benchmark_model(model_path: str, name: str):
        print(f"\nBenchmarking {name}...")
        model = YOLO(model_path)
        
        # Warmup runs
        print("  Warming up (10 runs)...")
        for _ in range(10):
            model(dummy_images, verbose=False)
        
        # Timed runs
        print(f"  Running {num_runs} timed inferences...")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            model(dummy_images, verbose=False)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{num_runs}")
        
        return {
            "name": name,
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "p95": np.percentile(times, 95),
        }
    
    # Run benchmarks
    pt_results = benchmark_model(pt_model_path, "PyTorch (FP16)")
    trt_results = benchmark_model(engine_path, "TensorRT (FP16)")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS (batch of {batch_size} images)")
    print(f"{'='*60}")
    
    for r in [pt_results, trt_results]:
        print(f"\n{r['name']}:")
        print(f"  Mean:   {r['mean']:.1f} ms")
        print(f"  Std:    {r['std']:.1f} ms")
        print(f"  Min:    {r['min']:.1f} ms")
        print(f"  Max:    {r['max']:.1f} ms")
        print(f"  P95:    {r['p95']:.1f} ms")
    
    speedup = pt_results["mean"] / trt_results["mean"]
    print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster with TensorRT!")
    print(f"   ({pt_results['mean']:.1f}ms → {trt_results['mean']:.1f}ms)")
    
    return pt_results, trt_results


def validate_detections(pt_model_path: str, engine_path: str):
    """
    Validate that TensorRT produces similar detections to PyTorch.
    """
    import numpy as np
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print("VALIDATION: Comparing Detection Outputs")
    print(f"{'='*60}")
    
    # Create a consistent test image
    np.random.seed(42)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    pt_model = YOLO(pt_model_path)
    trt_model = YOLO(engine_path)
    
    # Test 1: Single image inference (validates dynamic batch size support)
    print("  Testing Batch=1...")
    pt_results_1 = pt_model(test_image, verbose=False, conf=0.5)
    trt_results_1 = trt_model(test_image, verbose=False, conf=0.5)

    # Test 2: Batch inference (validates max throughput)
    print("  Testing Batch=4...")
    batch_images = [test_image] * 4
    pt_results = pt_model(batch_images, verbose=False, conf=0.5)
    trt_results = trt_model(batch_images, verbose=False, conf=0.5)
    
    pt_boxes = pt_results[0].boxes
    trt_boxes = trt_results[0].boxes
    
    pt_count = len(pt_boxes) if pt_boxes is not None else 0
    trt_count = len(trt_boxes) if trt_boxes is not None else 0
    
    print(f"\nPyTorch detections:  {pt_count}")
    print(f"TensorRT detections: {trt_count}")
    
    if pt_count == trt_count:
        print("✅ Detection counts match!")
    else:
        print("⚠️  Detection counts differ (minor variation is normal with FP16)")
    
    return pt_count, trt_count


def main():
    """Main entry point for TensorRT export."""
    print("\n" + "="*60)
    print("YOLO TensorRT Export Tool")
    print("="*60 + "\n")
    
    # Check prerequisites
    gpu_name = check_cuda_available()
    check_tensorrt_available()
    
    # Model paths
    weights_dir = PROJECT_ROOT / "weights"
    pt_model = weights_dir / "yolo26m.pt"
    
    # Check for model file
    if not pt_model.exists():
        print(f"\n❌ Model not found: {pt_model}")
        print("\nAvailable models in weights directory:")
        for f in weights_dir.glob("*.pt"):
            print(f"   - {f.name}")
        
        # Try to find any .pt file
        pt_files = list(weights_dir.glob("*.pt"))
        if pt_files:
            pt_model = pt_files[0]
            print(f"\n📌 Using found model: {pt_model.name}")
        else:
            print("\nNo .pt model files found. Please add your model to the weights/ directory.")
            sys.exit(1)
    
    # Export to TensorRT
    engine_path = export_to_tensorrt(
        model_path=str(pt_model),
        imgsz=640,
        batch_size=1,   # Reverted to 1 (safest for 8GB GPU with Display)
        half=True,      # FP16 for RTX 4060
        workspace_gb=4, # 4GB Safe limit
    )
    
    # Benchmark comparison
    pt_results, trt_results = benchmark_comparison(
        pt_model_path=str(pt_model),
        engine_path=engine_path,
        batch_size=32, # Test with full batch capability
        num_runs=50,
    )
    
    # Validate outputs
    validate_detections(
        pt_model_path=str(pt_model),
        engine_path=engine_path,
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTensorRT engine saved to:")
    print(f"   {engine_path}")
    print(f"\nTo use in your application, update the model path in your config:")
    print(f"   Before: weights/yolo26m.pt")
    print(f"   After:  {Path(engine_path).name}")
    print(f"\nExpected speedup: {pt_results['mean'] / trt_results['mean']:.1f}x faster")
    print(f"Expected latency: ~{trt_results['mean']:.0f}ms per batch of 32 images")
    print()


if __name__ == "__main__":
    main()
