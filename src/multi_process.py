"""
Multiprocessing text preprocessing pipeline
"""

import json
import re
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from measure_utils import (
    PerformanceTracker, save_metrics, print_metrics, 
    get_file_size_mb, save_jsonl
)


# Global tokenizer for worker processes
_tokenizer = None


def init_worker():
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    _tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing special characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_text(text: str, row_id: int) -> dict:
    """Process text through normalization and tokenization pipeline."""
    global _tokenizer
    
    normalized = normalize_text(text)
    tokens = _tokenizer.tokenize(normalized)
    formatted = " ".join(tokens)
    token_ids = _tokenizer.encode(normalized, add_special_tokens=False)
    
    return {
        'id': row_id,
        'original': text[:100],
        'normalized': normalized[:200],
        'tokens': tokens[:50],
        'formatted': formatted[:200],
        'token_ids': token_ids[:50],
        'token_count': len(tokens),
        'char_count': len(text)
    }


def process_batch(batch_data: list) -> list:
    """Process a batch of data in a worker process."""
    results = []
    
    for row_id, line in batch_data:
        try:
            data = json.loads(line.strip())
            text = data.get('text', '')
            
            if not text:
                continue
            
            processed = process_text(text, row_id)
            results.append(processed)
            
        except (json.JSONDecodeError, KeyError):
            continue
    
    return results


def run_multiprocess_pipeline(
    input_file: str,
    output_file: str,
    max_rows: int = None,
    num_workers: int = None
):
    """
    Run multiprocessing preprocessing pipeline with intermediate outputs.
    
    Uses Python's multiprocessing to parallelize work across CPU cores.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        max_rows: Maximum number of rows to process (None = all)
        num_workers: Number of worker processes (None = auto-detect)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print("\n" + "=" * 70)
    print(f"Running Multiprocessing Pipeline ({num_workers} workers)")
    print("=" * 70)
    
    # Setup performance tracking
    tracker = PerformanceTracker('multi')
    tracker.start()
    
    # Read input data
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if max_rows:
        lines = lines[:max_rows]
    
    total_lines = len(lines)
    print(f"Processing {total_lines:,} rows with {num_workers} workers...")
    
    # Create (row_id, line) tuples
    indexed_lines = list(enumerate(lines))
    
    # Split data into batches for parallel processing
    batch_size = max(100, total_lines // (num_workers * 10))
    batches = [indexed_lines[i:i + batch_size] for i in range(0, len(indexed_lines), batch_size)]
    
    print(f"Created {len(batches)} batches (size ~{batch_size} rows each)")
    
    # Storage for all results
    all_results = []
    
    # Process batches in parallel
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        
        # Process batches with progress bar
        for batch_results in tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="Processing batches"
        ):
            all_results.extend(batch_results)
            
            # Update memory tracking
            tracker.update_peak_memory()
    
    # Sort results by ID to maintain original order
    all_results.sort(key=lambda x: x['id'])
    
    # Extract intermediate outputs at each stage
    normalized_data = [{'id': r['id'], 'text': r['normalized']} for r in all_results]
    tokenized_data = [{'id': r['id'], 'tokens': r['tokens'], 'token_count': r['token_count']} for r in all_results]
    formatted_data = [{'id': r['id'], 'formatted': r['formatted'], 'token_ids': r['token_ids']} for r in all_results]
    
    # Save intermediate outputs
    print("\nSaving intermediate outputs...")
    intermediate_dir = "results/intermediate/multi"
    
    norm_count = save_jsonl(f"{intermediate_dir}/normalized.jsonl", normalized_data)
    tok_count = save_jsonl(f"{intermediate_dir}/tokenized.jsonl", tokenized_data)
    fmt_count = save_jsonl(f"{intermediate_dir}/formatted.jsonl", formatted_data)
    
    # Save final output
    final_count = save_jsonl(output_file, all_results)
    
    # Stop tracking and collect metrics
    perf_metrics = tracker.stop()
    
    # Calculate additional metrics
    output_size_mb = get_file_size_mb(output_file)
    throughput = len(all_results) / perf_metrics['elapsed_sec']
    
    metrics = {
        'runner': 'multi',
        'rows': len(all_results),
        'secs': perf_metrics['elapsed_sec'],
        'rows_per_sec': throughput,
        'peak_mb': perf_metrics['peak_memory_mb'],
        'bytes_out': int(output_size_mb * 1024 * 1024),
        'normalized_rows': norm_count,
        'tokenized_rows': tok_count,
        'formatted_rows': fmt_count
    }
    
    # Print and save metrics
    print_metrics(metrics)
    save_metrics(metrics)
    
    # Print intermediate outputs summary
    print("\n" + "=" * 70)
    print("Intermediate Outputs Saved")
    print("=" * 70)
    print(f"Normalized: {intermediate_dir}/normalized.jsonl ({norm_count:,} rows)")
    print(f"Tokenized:  {intermediate_dir}/tokenized.jsonl ({tok_count:,} rows)")
    print(f"Formatted:  {intermediate_dir}/formatted.jsonl ({fmt_count:,} rows)")
    print("=" * 70 + "\n")
    
    print(f"Final output saved to: {output_file}")
    print(f"Output size: {output_size_mb:.2f} MB\n")


def main():
    """Main entry point for multiprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Multiprocessing text preprocessing pipeline"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/sample_text.jsonl',
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/output_multi.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (default: all)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    run_multiprocess_pipeline(
        input_file=args.input,
        output_file=args.output,
        max_rows=args.max_rows,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
