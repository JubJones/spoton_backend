#!/usr/bin/env python3
"""
Phase 11 test runner for comprehensive validation.

Runs all Phase 11 related tests and provides a summary report.
"""

import subprocess
import sys
import os
from typing import Dict, List, Any
import json
from pathlib import Path

def run_test_suite(test_path: str, markers: List[str] = None) -> Dict[str, Any]:
    """Run a test suite and return results."""
    cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
    
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add JSON report
    json_report_path = f"test_results_{test_path.replace('/', '_').replace('.py', '')}.json"
    cmd.extend(["--json-report", f"--json-report-file={json_report_path}"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Try to load JSON report if available
        test_results = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        
        if os.path.exists(json_report_path):
            try:
                with open(json_report_path, 'r') as f:
                    json_report = json.load(f)
                test_results["json_report"] = json_report
                os.remove(json_report_path)  # Clean up
            except Exception:
                pass
        
        return test_results
    except Exception as e:
        return {"returncode": -1, "error": str(e)}

def print_test_summary(results: Dict[str, Dict[str, Any]]):
    """Print a summary of test results."""
    print("\n" + "="*80)
    print("PHASE 11: Final Production Enablement - Test Results Summary")
    print("="*80)
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_suite, result in results.items():
        print(f"\n{test_suite}:")
        print("-" * 50)
        
        if result["returncode"] == 0:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
        
        # Extract test counts from JSON report if available
        if "json_report" in result:
            summary = result["json_report"].get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            skipped = summary.get("skipped", 0)
            total = summary.get("total", 0)
            
            print(f"   Total: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
            
            total_tests += total
            total_passed += passed
            total_failed += failed
            total_skipped += skipped
        
        # Show errors if any
        if result.get("stderr") and result["stderr"].strip():
            print(f"   Errors: {result['stderr'].strip()[:200]}...")
    
    print("\n" + "="*80)
    print(f"OVERALL SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_failed}")
    print(f"   Skipped: {total_skipped}")
    
    if total_failed == 0:
        print("✅ ALL PHASE 11 TESTS PASSED - PRODUCTION READY!")
    else:
        print(f"❌ {total_failed} TESTS FAILED - NEEDS ATTENTION")
    
    print("="*80)

def main():
    """Run Phase 11 comprehensive test suite."""
    print("Starting Phase 11: Final Production Enablement validation...")
    print("This will test all implemented features for production readiness.\n")
    
    # Define test suites
    test_suites = {
        "System Monitoring Endpoints": "tests/api/v1/test_system_monitoring_endpoints.py",
        "Security Hardening": "tests/test_security_hardening.py", 
        "Phase 11 End-to-End": "tests/test_phase11_e2e.py",
        "Export Endpoints": "tests/api/v1/test_export_endpoints.py",
        "Integration Tests": "tests/test_integration.py",
        "Security Tests": "tests/test_security.py",
    }
    
    results = {}
    
    # Run each test suite
    for suite_name, test_path in test_suites.items():
        print(f"Running {suite_name}...")
        
        # Check if test file exists
        full_path = Path(__file__).parent.parent / test_path
        if not full_path.exists():
            print(f"⚠️  Test file not found: {test_path}")
            results[suite_name] = {"returncode": -1, "error": "Test file not found"}
            continue
        
        result = run_test_suite(test_path)
        results[suite_name] = result
        
        if result["returncode"] == 0:
            print("✅ Passed")
        else:
            print("❌ Failed")
    
    # Run performance tests if requested
    if "--include-performance" in sys.argv:
        print("\nRunning Performance Tests...")
        perf_result = run_test_suite("tests/", ["performance"])
        results["Performance Tests"] = perf_result
    
    # Print comprehensive summary
    print_test_summary(results)
    
    # Exit with appropriate code
    failed_suites = [name for name, result in results.items() if result["returncode"] != 0]
    if failed_suites:
        print(f"\nFailed test suites: {', '.join(failed_suites)}")
        sys.exit(1)
    else:
        print("\n🎉 Phase 11 implementation validated successfully!")
        print("Backend is ready for production deployment.")
        sys.exit(0)

if __name__ == "__main__":
    main()