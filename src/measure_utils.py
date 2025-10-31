"""
Measurement Utilities for Pipeline Benchmarking

Provides helpers for timing, memory tracking, and metrics collection.
"""

import time
import psutil
import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, Iterable


class PerformanceTracker:
    """
    Track performance metrics during pipeline execution.
    
    Measures:
    - Execution time
    - Peak memory usage
    - Throughput (rows/sec)
    """
    
    def __init__(self, runner_name: str):
        """
        Initialize performance tracker.
        
        Args:
            runner_name: Name of the runner being tracked (e.g., 'single', 'multi', 'ray')
        """
        self.runner_name = runner_name
        self.start_time = None
        self.end_time = None
        self.peak_memory_mb = 0
        self.initial_memory_mb = 0
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        """Start tracking time and memory."""
        self.start_time = time.time()
        self.initial_memory_mb = self.get_current_memory_mb()
        self.peak_memory_mb = self.initial_memory_mb
        
    def update_peak_memory(self):
        """Update peak memory if current usage is higher."""
        current_mb = self.get_current_memory_mb()
        if current_mb > self.peak_memory_mb:
            self.peak_memory_mb = current_mb
            
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop tracking and return metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        self.end_time = time.time()
        self.update_peak_memory()
        
        elapsed_sec = self.end_time - self.start_time
        
        return {
            'runner': self.runner_name,
            'elapsed_sec': elapsed_sec,
            'peak_memory_mb': self.peak_memory_mb,
            'initial_memory_mb': self.initial_memory_mb
        }


def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in MB.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in MB
    """
    if not os.path.exists(filepath):
        return 0.0
    return os.path.getsize(filepath) / 1024 / 1024


def get_directory_size_mb(dirpath: str) -> float:
    """
    Get total size of directory in MB (recursive).
    
    Args:
        dirpath: Path to directory
        
    Returns:
        Total size in MB
    """
    if not os.path.exists(dirpath):
        return 0.0
    
    total_size = 0
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    
    return total_size / 1024 / 1024


def save_metrics(metrics: Dict[str, Any], output_path: str = "results/metrics.csv"):
    """
    Append metrics to CSV file.
    
    Args:
        metrics: Dictionary containing benchmark metrics
        output_path: Path to output CSV file
    """
    # Ensure results directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(output_path)
    
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)
    
    print(f"Metrics saved to {output_path}")


def print_metrics(metrics: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(f"Performance Metrics: {metrics.get('runner', 'Unknown')}")
    print("=" * 70)
    
    for key, value in metrics.items():
        if key == 'runner':
            continue
        
        if 'sec' in key or 'time' in key:
            print(f"  {key:.<30} {value:.2f} sec")
        elif 'mb' in key or 'memory' in key:
            print(f"  {key:.<30} {value:.2f} MB")
        elif 'rows' in key.lower() and 'per' not in key:
            print(f"  {key:.<30} {value:,}")
        elif 'per_sec' in key or 'throughput' in key:
            print(f"  {key:.<30} {value:,.2f} rows/sec")
        elif 'bytes' in key:
            print(f"  {key:.<30} {value / 1024 / 1024:.2f} MB")
        else:
            print(f"  {key:.<30} {value}")
    
    print("=" * 70 + "\n")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def save_jsonl(path: str, iterable: Iterable[Dict[str, Any]]):
    """
    Save an iterable of dictionaries to a JSONL file.
    
    Args:
        path: Output file path
        iterable: Iterable of dictionaries to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in iterable:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    
    return count

