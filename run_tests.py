#!/usr/bin/env python3
"""
Test runner for SBCM integration tests.

Usage:
    python run_tests.py [test_type] [options]
    
Test types:
    unit        - Run unit tests only
    integration - Run integration tests only  
    validation  - Run validation tests only
    all         - Run all tests (default)
    performance - Run performance benchmarks
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return success status."""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.returncode == 0:
        print("✓ PASSED")
        if result.stdout:
            print(result.stdout)
    else:
        print("✗ FAILED")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    
    return result.returncode == 0


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'pytest', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def run_unit_tests(verbose=True, coverage=False):
    """Run unit tests."""
    cmd = ['python', '-m', 'pytest', 'tests/unit/', '-v']
    
    if coverage:
        cmd.extend(['--cov=agent_torch.examples.models.opinion_dynamics', '--cov-report=html'])
    
    if not verbose:
        cmd.remove('-v')
        cmd.append('-q')
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=True):
    """Run integration tests."""
    cmd = ['python', '-m', 'pytest', 'tests/integration/', '-v']
    
    if not verbose:
        cmd.remove('-v')
        cmd.append('-q')
    
    return run_command(cmd, "Integration Tests")


def run_validation_tests(verbose=True):
    """Run validation tests."""
    cmd = ['python', '-m', 'pytest', 'tests/validation/', '-v']
    
    if not verbose:
        cmd.remove('-v')
        cmd.append('-q')
    
    return run_command(cmd, "Validation Tests")


def run_performance_tests(verbose=True):
    """Run performance benchmark tests."""
    cmd = ['python', '-m', 'pytest', 'tests/performance/', '-v', '--benchmark-only']
    
    if not verbose:
        cmd.remove('-v')
        cmd.append('-q')
    
    return run_command(cmd, "Performance Tests")


def run_all_tests(verbose=True, coverage=False):
    """Run all test suites."""
    print(f"\n{'='*80}")
    print("SBCM INTEGRATION TEST SUITE")
    print(f"{'='*80}")
    
    overall_success = True
    results = {}
    
    # Unit tests
    results['unit'] = run_unit_tests(verbose, coverage)
    overall_success &= results['unit']
    
    # Integration tests
    results['integration'] = run_integration_tests(verbose)
    overall_success &= results['integration']
    
    # Validation tests
    results['validation'] = run_validation_tests(verbose)
    # Don't fail overall if validation tests fail (SINN might not be available)
    
    # Performance tests (optional)
    if Path('tests/performance').exists():
        results['performance'] = run_performance_tests(verbose)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for test_type, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_type.upper():15} : {status}")
    
    overall_status = "PASSED" if overall_success else "FAILED"
    print(f"{'OVERALL':15} : {overall_status}")
    
    return overall_success


def create_test_directories():
    """Create test directory structure if it doesn't exist."""
    test_dirs = [
        'tests',
        'tests/unit',
        'tests/integration', 
        'tests/validation',
        'tests/performance',
        'tests/fixtures'
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_file = Path(test_dir) / '__init__.py'
        if not init_file.exists():
            init_file.write_text('# Test directory\n')


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="SBCM Integration Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py unit              # Run unit tests only
  python run_tests.py all --coverage    # Run all tests with coverage
  python run_tests.py unit --quiet      # Run unit tests quietly
        """
    )
    
    parser.add_argument(
        'test_type',
        nargs='?',
        default='all',
        choices=['unit', 'integration', 'validation', 'performance', 'all'],
        help='Type of tests to run (default: all)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report for unit tests'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Run tests in quiet mode'
    )
    
    parser.add_argument(
        '--no-deps-check',
        action='store_true',
        help='Skip dependency checking'
    )
    
    args = parser.parse_args()
    
    # Setup
    create_test_directories()
    
    if not args.no_deps_check and not check_dependencies():
        print("\nDependency check failed. Use --no-deps-check to skip.")
        return 1
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    verbose = not args.quiet
    success = True
    
    # Run requested tests
    if args.test_type == 'unit':
        success = run_unit_tests(verbose, args.coverage)
    elif args.test_type == 'integration':
        success = run_integration_tests(verbose)
    elif args.test_type == 'validation':
        success = run_validation_tests(verbose)
    elif args.test_type == 'performance':
        success = run_performance_tests(verbose)
    elif args.test_type == 'all':
        success = run_all_tests(verbose, args.coverage)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())