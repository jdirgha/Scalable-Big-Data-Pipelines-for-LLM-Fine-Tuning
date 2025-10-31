"""
Format Comparison Utility

"""

import json
import time
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from measure_utils import get_file_size_mb, print_metrics


def benchmark_jsonl_write(data: list, output_path: str) -> dict:
    """
    Benchmark JSONL write performance.
    
    Args:
        data: List of dictionaries to write
        output_path: Output file path
        
    Returns:
        Dictionary with metrics
    """
    start_time = time.time()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    write_time = time.time() - start_time
    file_size_mb = get_file_size_mb(output_path)
    
    return {
        'format': 'JSONL',
        'operation': 'write',
        'time_sec': write_time,
        'size_mb': file_size_mb,
        'rows': len(data)
    }


def benchmark_jsonl_read(input_path: str) -> dict:
    """
    Benchmark JSONL read performance.
    
    Args:
        input_path: Input file path
        
    Returns:
        Dictionary with metrics
    """
    start_time = time.time()
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    read_time = time.time() - start_time
    
    return {
        'format': 'JSONL',
        'operation': 'read',
        'time_sec': read_time,
        'rows': len(data)
    }


def benchmark_parquet_write(data: list, output_path: str) -> dict:
    """
    Benchmark Parquet write performance.
    
    Args:
        data: List of dictionaries to write
        output_path: Output file path
        
    Returns:
        Dictionary with metrics
    """
    start_time = time.time()
    
    # Convert to DataFrame and write
    df = pd.DataFrame(data)
    df.to_parquet(output_path, compression='snappy', index=False)
    
    write_time = time.time() - start_time
    file_size_mb = get_file_size_mb(output_path)
    
    return {
        'format': 'Parquet',
        'operation': 'write',
        'time_sec': write_time,
        'size_mb': file_size_mb,
        'rows': len(data)
    }


def benchmark_parquet_read(input_path: str) -> dict:
    """
    Benchmark Parquet read performance.
    
    Args:
        input_path: Input file path
        
    Returns:
        Dictionary with metrics
    """
    start_time = time.time()
    

    df = pd.read_parquet(input_path)
    data = df.to_dict('records')
    
    read_time = time.time() - start_time
    
    return {
        'format': 'Parquet',
        'operation': 'read',
        'time_sec': read_time,
        'rows': len(data)
    }


def run_format_comparison(input_jsonl: str, results_dir: str = 'results'):
    """
    Compare JSONL and Parquet formats.
    
    Args:
        input_jsonl: Path to input JSONL file
        results_dir: Directory for temporary output files
    """
    print("\n" + "=" * 70)
    print("Format Comparison: JSONL vs Parquet")
    print("=" * 70)
    
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data from input JSONL
    print(f"Loading data from: {input_jsonl}")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    print(f"Loaded {len(data):,} rows\n")
    
    
    jsonl_out = f"{results_dir}/temp_output.jsonl"
    parquet_out = f"{results_dir}/temp_output.parquet"
    
    results = []
    
    # Benchmark JSONL write
    print("Benchmarking JSONL write...")
    jsonl_write_metrics = benchmark_jsonl_write(data, jsonl_out)
    results.append(jsonl_write_metrics)
    print(f"  Time: {jsonl_write_metrics['time_sec']:.2f}s, "
          f"Size: {jsonl_write_metrics['size_mb']:.2f} MB")
    
    # Benchmark JSONL read
    print("Benchmarking JSONL read...")
    jsonl_read_metrics = benchmark_jsonl_read(jsonl_out)
    results.append(jsonl_read_metrics)
    print(f"  Time: {jsonl_read_metrics['time_sec']:.2f}s")
    
    # Benchmark Parquet write
    print("Benchmarking Parquet write...")
    parquet_write_metrics = benchmark_parquet_write(data, parquet_out)
    results.append(parquet_write_metrics)
    print(f"  Time: {parquet_write_metrics['time_sec']:.2f}s, "
          f"Size: {parquet_write_metrics['size_mb']:.2f} MB")
    
    # Benchmark Parquet read
    print("Benchmarking Parquet read...")
    parquet_read_metrics = benchmark_parquet_read(parquet_out)
    results.append(parquet_read_metrics)
    print(f"  Time: {parquet_read_metrics['time_sec']:.2f}s")
    
   
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print("\nWrite Performance:")
    print(f"  JSONL:   {jsonl_write_metrics['time_sec']:.2f}s "
          f"({jsonl_write_metrics['size_mb']:.2f} MB)")
    print(f"  Parquet: {parquet_write_metrics['time_sec']:.2f}s "
          f"({parquet_write_metrics['size_mb']:.2f} MB)")
    
    speedup = jsonl_write_metrics['time_sec'] / parquet_write_metrics['time_sec']
    compression = (1 - parquet_write_metrics['size_mb'] / jsonl_write_metrics['size_mb']) * 100
    print(f"  Parquet is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"  Parquet saves {compression:.1f}% space")
    
    print("\nRead Performance:")
    print(f"  JSONL:   {jsonl_read_metrics['time_sec']:.2f}s")
    print(f"  Parquet: {parquet_read_metrics['time_sec']:.2f}s")
    
    speedup = jsonl_read_metrics['time_sec'] / parquet_read_metrics['time_sec']
    print(f"  Parquet is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    print("\n" + "=" * 70 + "\n")
    
   
    results_df = pd.DataFrame(results)
    results_csv = f"{results_dir}/format_comparison.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")


def main():
    """Main entry point for format comparison."""
    parser = argparse.ArgumentParser(
        description="Compare JSONL vs Parquet format performance"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/output_single.jsonl',
        help='Input JSONL file to use for comparison'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory for output files'
    )
    
    args = parser.parse_args()
    
    run_format_comparison(
        input_jsonl=args.input,
        results_dir=args.results_dir
    )


if __name__ == '__main__':
    main()

