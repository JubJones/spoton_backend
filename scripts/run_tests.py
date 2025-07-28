#!/usr/bin/env python3
"""
Test runner script for SpotOn backend system.

Provides comprehensive test execution with different profiles:
- Unit tests only
- Integration tests only
- Security tests only
- Performance tests only
- Full test suite
- GPU tests (if CUDA available)
- Continuous integration profile
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
from datetime import datetime


class TestRunner:
    """Test runner for SpotOn backend system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.reports_dir = project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test profiles
        self.test_profiles = {
            "unit": {
                "markers": ["unit"],
                "description": "Unit tests only",
                "timeout": 60,
                "parallel": True
            },
            "integration": {
                "markers": ["integration"],
                "description": "Integration tests only",
                "timeout": 300,
                "parallel": False
            },
            "security": {
                "markers": ["security"],
                "description": "Security tests only",
                "timeout": 120,
                "parallel": True
            },
            "performance": {
                "markers": ["performance"],
                "description": "Performance tests only",
                "timeout": 600,
                "parallel": False
            },
            "gpu": {
                "markers": ["gpu"],
                "description": "GPU tests (requires CUDA)",
                "timeout": 180,
                "parallel": False
            },
            "quick": {
                "markers": ["unit", "not slow"],
                "description": "Quick tests for development",
                "timeout": 30,
                "parallel": True
            },
            "ci": {
                "markers": ["unit", "integration", "security", "not slow"],
                "description": "CI/CD pipeline tests",
                "timeout": 300,
                "parallel": True
            },
            "full": {
                "markers": [],
                "description": "Full test suite",
                "timeout": 900,
                "parallel": False
            }
        }
    
    def run_tests(self, profile: str = "unit", verbose: bool = False, 
                  coverage: bool = True, html_report: bool = False,
                  parallel: Optional[bool] = None) -> bool:
        """Run tests with specified profile."""
        if profile not in self.test_profiles:
            print(f"Error: Unknown test profile '{profile}'")
            print(f"Available profiles: {', '.join(self.test_profiles.keys())}")
            return False
        
        profile_config = self.test_profiles[profile]
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test directory
        cmd.append(str(self.test_dir))
        
        # Add markers
        if profile_config["markers"]:
            markers = " or ".join(profile_config["markers"])
            cmd.extend(["-m", markers])
        
        # Add timeout
        cmd.extend(["--timeout", str(profile_config["timeout"])])
        
        # Add verbosity
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=app",
                "--cov-report=term-missing",
                f"--cov-report=html:{self.reports_dir}/coverage_html"
            ])
        
        # Add parallel execution
        use_parallel = parallel if parallel is not None else profile_config.get("parallel", False)
        if use_parallel:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            cmd.extend(["-n", str(min(cpu_count, 4))])
        
        # Add HTML report
        if html_report:
            cmd.extend([
                "--html", str(self.reports_dir / f"{profile}_report.html"),
                "--self-contained-html"
            ])
        
        # Add JUnit XML report
        cmd.extend([
            "--junitxml", str(self.reports_dir / f"{profile}_junit.xml")
        ])
        
        # Add other options
        cmd.extend([
            "--tb=short",
            "--maxfail=10",
            "--durations=20"
        ])
        
        print(f"Running {profile_config['description']}...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        # Change to project root directory
        os.chdir(self.project_root)
        
        # Run tests
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False)
        end_time = time.time()
        
        # Report results
        duration = end_time - start_time
        success = result.returncode == 0
        
        print("-" * 60)
        print(f"Test execution completed in {duration:.2f} seconds")
        print(f"Status: {'PASSED' if success else 'FAILED'}")
        print(f"Exit code: {result.returncode}")
        
        # Generate summary report
        self._generate_summary_report(profile, success, duration, result.returncode)
        
        return success
    
    def run_multiple_profiles(self, profiles: List[str], stop_on_failure: bool = True) -> Dict[str, bool]:
        """Run multiple test profiles."""
        results = {}
        
        for profile in profiles:
            print(f"\n{'='*80}")
            print(f"Running test profile: {profile}")
            print(f"{'='*80}")
            
            success = self.run_tests(profile, verbose=True, coverage=True)
            results[profile] = success
            
            if not success and stop_on_failure:
                print(f"Tests failed for profile '{profile}', stopping execution")
                break
        
        # Print summary
        print(f"\n{'='*80}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        for profile, success in results.items():
            status = "PASSED" if success else "FAILED"
            print(f"{profile:20} {status}")
        
        return results
    
    def check_dependencies(self) -> bool:
        """Check if test dependencies are available."""
        dependencies = [
            ("pytest", "pytest"),
            ("pytest-cov", "coverage"),
            ("pytest-html", "HTML reports"),
            ("pytest-xdist", "parallel execution"),
            ("pytest-timeout", "test timeouts"),
            ("pytest-asyncio", "async test support")
        ]
        
        missing = []
        for package, description in dependencies:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append((package, description))
        
        if missing:
            print("Missing test dependencies:")
            for package, description in missing:
                print(f"  - {package} ({description})")
            print("\nInstall with: pip install " + " ".join(p[0] for p in missing))
            return False
        
        return True
    
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available for testing."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"GPU available: {device_count} CUDA device(s)")
                for i in range(device_count):
                    properties = torch.cuda.get_device_properties(i)
                    print(f"  Device {i}: {properties.name} ({properties.total_memory / 1024**3:.1f} GB)")
                return True
            else:
                print("GPU not available: CUDA not found")
                return False
        except ImportError:
            print("GPU not available: PyTorch not installed")
            return False
    
    def _generate_summary_report(self, profile: str, success: bool, duration: float, exit_code: int):
        """Generate summary report."""
        report = {
            "profile": profile,
            "description": self.test_profiles[profile]["description"],
            "success": success,
            "duration": duration,
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
            "reports_dir": str(self.reports_dir)
        }
        
        report_file = self.reports_dir / f"{profile}_summary.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Summary report saved to: {report_file}")
    
    def list_profiles(self):
        """List available test profiles."""
        print("Available test profiles:")
        print("-" * 40)
        
        for profile, config in self.test_profiles.items():
            markers = config["markers"] if config["markers"] else ["all"]
            print(f"{profile:12} {config['description']}")
            print(f"{'':12} Markers: {', '.join(markers)}")
            print(f"{'':12} Timeout: {config['timeout']}s")
            print(f"{'':12} Parallel: {config.get('parallel', False)}")
            print()
    
    def clean_reports(self):
        """Clean test reports directory."""
        import shutil
        
        if self.reports_dir.exists():
            shutil.rmtree(self.reports_dir)
            self.reports_dir.mkdir()
            print(f"Cleaned reports directory: {self.reports_dir}")
        else:
            print(f"Reports directory does not exist: {self.reports_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SpotOn Backend Test Runner")
    parser.add_argument("profile", nargs="?", default="unit", 
                       help="Test profile to run (default: unit)")
    parser.add_argument("--list", action="store_true", 
                       help="List available test profiles")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", 
                       help="Disable coverage reporting")
    parser.add_argument("--html-report", action="store_true", 
                       help="Generate HTML test report")
    parser.add_argument("--parallel", action="store_true", 
                       help="Force parallel execution")
    parser.add_argument("--no-parallel", action="store_true", 
                       help="Force sequential execution")
    parser.add_argument("--multiple", nargs="+", 
                       help="Run multiple profiles")
    parser.add_argument("--check-deps", action="store_true", 
                       help="Check test dependencies")
    parser.add_argument("--check-gpu", action="store_true", 
                       help="Check GPU availability")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean test reports")
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    runner = TestRunner(project_root)
    
    # Handle special commands
    if args.list:
        runner.list_profiles()
        return
    
    if args.check_deps:
        if runner.check_dependencies():
            print("All test dependencies are available")
        else:
            sys.exit(1)
        return
    
    if args.check_gpu:
        runner.check_gpu_availability()
        return
    
    if args.clean:
        runner.clean_reports()
        return
    
    # Check dependencies before running tests
    if not runner.check_dependencies():
        sys.exit(1)
    
    # Determine parallel execution
    parallel = None
    if args.parallel:
        parallel = True
    elif args.no_parallel:
        parallel = False
    
    # Run tests
    if args.multiple:
        results = runner.run_multiple_profiles(args.multiple)
        success = all(results.values())
    else:
        success = runner.run_tests(
            profile=args.profile,
            verbose=args.verbose,
            coverage=not args.no_coverage,
            html_report=args.html_report,
            parallel=parallel
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()