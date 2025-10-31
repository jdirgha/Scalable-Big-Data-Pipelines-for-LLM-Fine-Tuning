"""
Results Visualization

Generate charts and plots from benchmark metrics for the term paper.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def set_plot_style():
    """Set consistent plot styling."""
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_throughput(df: pd.DataFrame, output_path: str):
    """
    Plot throughput comparison across runners.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path for plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    runners = df['runner'].tolist()
    throughput = df['rows_per_sec'].tolist()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(runners)]
    bars = ax.bar(runners, throughput, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\nrows/sec',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Throughput (rows/sec)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pipeline Runner', fontsize=12, fontweight='bold')
    ax.set_title('Pipeline Throughput Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Calculate speedup annotations
    if 'single' in runners:
        baseline_idx = runners.index('single')
        baseline_throughput = throughput[baseline_idx]
        
        for i, runner in enumerate(runners):
            if runner != 'single':
                speedup = throughput[i] / baseline_throughput
                ax.text(i, throughput[i] * 1.1, f'{speedup:.2f}x',
                       ha='center', fontsize=9, style='italic', color='darkred')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved throughput plot to: {output_path}")


def plot_memory(df: pd.DataFrame, output_path: str):
    """
    Plot memory usage comparison across runners.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path for plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    runners = df['runner'].tolist()
    memory = df['peak_mb'].tolist()
    
    colors = ['#9b59b6', '#f39c12', '#1abc9c'][:len(runners)]
    bars = ax.bar(runners, memory, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,} MB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pipeline Runner', fontsize=12, fontweight='bold')
    ax.set_title('Peak Memory Consumption Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved memory plot to: {output_path}")


def plot_execution_time(df: pd.DataFrame, output_path: str):
    """
    Plot execution time comparison across runners.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path for plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    runners = df['runner'].tolist()
    times = df['secs'].tolist()
    
    colors = ['#e67e22', '#16a085', '#c0392b'][:len(runners)]
    bars = ax.bar(runners, times, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        mins = int(height // 60)
        secs = int(height % 60)
        label = f'{mins}m {secs}s' if mins > 0 else f'{secs}s'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pipeline Runner', fontsize=12, fontweight='bold')
    ax.set_title('Total Execution Time Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved execution time plot to: {output_path}")


def plot_combined_metrics(df: pd.DataFrame, output_path: str):
    """
    Create a combined multi-metric comparison plot.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path for plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pipeline Performance Summary', fontsize=16, fontweight='bold', y=1.00)
    
    runners = df['runner'].tolist()
    
    # Plot 1: Throughput
    ax1 = axes[0, 0]
    throughput = df['rows_per_sec'].tolist()
    colors1 = ['#3498db', '#2ecc71', '#e74c3c'][:len(runners)]
    ax1.bar(runners, throughput, color=colors1, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Throughput (rows/sec)', fontweight='bold')
    ax1.set_title('Throughput Comparison', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Memory
    ax2 = axes[0, 1]
    memory = df['peak_mb'].tolist()
    colors2 = ['#9b59b6', '#f39c12', '#1abc9c'][:len(runners)]
    ax2.bar(runners, memory, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Peak Memory (MB)', fontweight='bold')
    ax2.set_title('Memory Usage Comparison', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Execution Time
    ax3 = axes[1, 0]
    times = df['secs'].tolist()
    colors3 = ['#e67e22', '#16a085', '#c0392b'][:len(runners)]
    ax3.bar(runners, times, color=colors3, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Execution Time (sec)', fontweight='bold')
    ax3.set_title('Execution Time Comparison', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Speedup relative to single
    ax4 = axes[1, 1]
    if 'single' in runners:
        baseline_idx = runners.index('single')
        baseline_throughput = throughput[baseline_idx]
        speedups = [t / baseline_throughput for t in throughput]
        
        colors4 = ['#95a5a6', '#27ae60', '#e74c3c'][:len(runners)]
        bars = ax4.bar(runners, speedups, color=colors4, alpha=0.8, edgecolor='black')
        
        # Add speedup labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x',
                    ha='center', va='bottom', fontweight='bold')
        
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax4.set_ylabel('Speedup (relative to single)', fontweight='bold')
        ax4.set_title('Speedup Analysis', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved combined metrics plot to: {output_path}")


def plot_efficiency_scatter(df: pd.DataFrame, output_path: str):
    """
    Create efficiency scatter plot (throughput vs memory).
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path for plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    runners = df['runner'].tolist()
    throughput = df['rows_per_sec'].tolist()
    memory = df['peak_mb'].tolist()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(runners)]
    
    for i, (runner, tp, mem) in enumerate(zip(runners, throughput, memory)):
        ax.scatter(mem, tp, s=300, alpha=0.6, color=colors[i], 
                  edgecolor='black', linewidth=2, label=runner.upper())
        ax.annotate(runner.upper(), (mem, tp), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    ax.set_xlabel('Peak Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (rows/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency Analysis: Throughput vs Memory', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved efficiency scatter plot to: {output_path}")


def generate_all_plots(metrics_csv: str, output_dir: str = 'results'):
    """
    Generate all visualization plots from metrics CSV.
    
    Args:
        metrics_csv: Path to metrics CSV file
        output_dir: Directory for output plots
    """
    print("\n" + "=" * 70)
    print(" Generating Visualization Plots")
    print("=" * 70)
    
    # Load metrics
    print(f"📂 Loading metrics from: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    print(f" Found {len(df)} benchmark results\n")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set plot style
    set_plot_style()
    
    # Generate individual plots
    print("🎨 Generating plots...")
    plot_throughput(df, f"{output_dir}/throughput.png")
    plot_memory(df, f"{output_dir}/memory.png")
    plot_execution_time(df, f"{output_dir}/execution_time.png")
    plot_combined_metrics(df, f"{output_dir}/combined_metrics.png")
    plot_efficiency_scatter(df, f"{output_dir}/efficiency_scatter.png")
    
    print("\n" + "=" * 70)
    print(" All plots generated successfully!")
    print("=" * 70 + "\n")


def main():
    """Main entry point for plotting."""
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from benchmark metrics"
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='results/metrics.csv',
        help='Path to metrics CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory for output plots'
    )
    
    args = parser.parse_args()
    
    generate_all_plots(
        metrics_csv=args.metrics,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

