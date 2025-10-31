#!/usr/bin/env python3
"""
Quick Test Script

Run a fast test with small dataset to verify everything works.
Use this before running the full benchmark suite.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[PASSED] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Run quick verification tests."""
    print("\n" + "=" * 70)
    print("Quick Test Suite - Verification")
    print("=" * 70)
    print("This will run a fast test with 1000 rows to verify setup.\n")
    
    src_dir = Path("src")
    test_data = Path("data/test_sample.jsonl")
    
    # Test 1: Generate small test dataset
    print("\nTest 1: Dataset Generation")
    success = run_command(
        ["python3", str(src_dir / "generate_dataset.py"),
         "--output", str(test_data),
         "--rows", "1000"],
        "Generate 1000-row test dataset"
    )
    if not success:
        print("\n[ERROR] Dataset generation failed. Check your setup.")
        sys.exit(1)
    
    # Test 2: Single-process pipeline
    print("\nTest 2: Single-Process Pipeline")
    success = run_command(
        ["python3", str(src_dir / "single_process.py"),
         "--input", str(test_data),
         "--output", "results/test_single.jsonl",
         "--max_rows", "500"],
        "Single-process with 500 rows"
    )
    
    # Test 3: Multiprocessing pipeline
    print("\nTest 3: Multiprocessing Pipeline")
    success = run_command(
        ["python3", str(src_dir / "multi_process.py"),
         "--input", str(test_data),
         "--output", "results/test_multi.jsonl",
         "--max_rows", "500",
         "--workers", "2"],
        "Multiprocessing with 500 rows"
    )
    
    # Test 4: Ray pipeline
    print("\nTest 4: Ray Distributed Pipeline")
    success = run_command(
        ["python3", str(src_dir / "ray_runner.py"),
         "--input", str(test_data),
         "--output", "results/test_ray.jsonl",
         "--max_rows", "500",
         "--actors", "2"],
        "Ray with 500 rows"
    )
    
    # Test 5: Check if metrics were created
    print("\nTest 5: Metrics Collection")
    if Path("results/metrics.csv").exists():
        print("[SUCCESS] Metrics file created successfully")
        
        # Show metrics
        with open("results/metrics.csv", 'r') as f:
            print("\nCurrent Metrics:")
            print(f.read())
    else:
        print("[WARNING] No metrics file found")
    
    # Summary
    print("\n" + "=" * 70)
    print("Quick Test Complete!")
    print("=" * 70)
    print("\nIf all tests passed, you're ready to run the full benchmark:")
    print("  python3 run_all.py")
    print("\nOr run individual pipelines:")
    print("  cd src")
    print("  python3 single_process.py")
    print("  python3 multi_process.py")
    print("  python3 ray_runner.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

