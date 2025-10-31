#!/usr/bin/env python3
"""
Master Runner Script

Run all benchmarks sequentially and generate visualizations.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str):
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command as list of strings
        description: Human-readable description
    """
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"[SUCCESS] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with code {e.returncode}")
        return False


def main():
    """Run complete benchmark suite."""
    print("\n" + "=" * 70)
    print("LLM Pipeline Benchmark - Complete Suite")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("[ERROR] 'src' directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    src_dir = Path("src")
    data_file = Path("data/sample_text.jsonl")
    
    # Step 1: Generate dataset if needed
    if not data_file.exists():
        print("\nDataset not found. Generating sample dataset...")
        success = run_command(
            ["python3", str(src_dir / "generate_dataset.py"), 
             "--output", str(data_file),
             "--rows", "50000"],
            "Dataset Generation"
        )
        if not success:
            print("[ERROR] Failed to generate dataset. Exiting.")
            sys.exit(1)
    else:
        print(f"\nDataset already exists: {data_file}")
    
    # Step 2: Run single-process benchmark
    success = run_command(
        ["python3", str(src_dir / "single_process.py"),
         "--input", str(data_file),
         "--output", "results/output_single.jsonl"],
        "Single-Process Pipeline"
    )
    if not success:
        print("[WARNING] Single-process benchmark failed, continuing...")
    
    # Step 3: Run multiprocessing benchmark
    success = run_command(
        ["python3", str(src_dir / "multi_process.py"),
         "--input", str(data_file),
         "--output", "results/output_multi.jsonl"],
        "Multiprocessing Pipeline"
    )
    if not success:
        print("[WARNING] Multiprocessing benchmark failed, continuing...")
    
    # Step 4: Run Ray benchmark
    success = run_command(
        ["python3", str(src_dir / "ray_runner.py"),
         "--input", str(data_file),
         "--output", "results/output_ray.jsonl"],
        "Ray Distributed Pipeline"
    )
    if not success:
        print("[WARNING] Ray benchmark failed, continuing...")
    
    # Step 5: Generate visualizations
    if Path("results/metrics.csv").exists():
        success = run_command(
            ["python3", str(src_dir / "plot_results.py"),
             "--metrics", "results/metrics.csv",
             "--output_dir", "results"],
            "Visualization Generation"
        )
        if not success:
            print("[WARNING] Visualization generation failed, continuing...")
    else:
        print("\n[WARNING] No metrics.csv found. Skipping visualization generation.")
    
    # Step 6: (Optional) Compare formats
    if Path("results/output_single.jsonl").exists():
        print("\n" + "=" * 70)
        print("Optional: Format Comparison")
        print("=" * 70)
        print("Run format comparison? This compares JSONL vs Parquet.")
        response = input("Continue? (y/n): ").lower().strip()
        
        if response == 'y':
            run_command(
                ["python3", str(src_dir / "compare_formats.py"),
                 "--input", "results/output_single.jsonl",
                 "--results_dir", "results"],
                "Format Comparison (JSONL vs Parquet)"
            )
    
    # Final summary
    print("\n" + "=" * 70)
    print("Benchmark Suite Complete!")
    print("=" * 70)
    print("\nCheck the following for results:")
    print("  - results/metrics.csv        (Performance metrics)")
    print("  - results/*.png              (Visualization plots)")
    print("  - results/output_*.jsonl     (Processed output files)")
    print("\nNext steps:")
    print("  1. Review metrics in results/metrics.csv")
    print("  2. Examine plots in results/ directory")
    print("  3. Run custom experiments with different parameters")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

