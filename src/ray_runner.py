"""
Ray distributed text preprocessing pipeline
"""

import json
import re
import argparse
from pathlib import Path
import ray
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from measure_utils import (
    PerformanceTracker, save_metrics, print_metrics, 
    get_file_size_mb, save_jsonl
)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@ray.remote
class TokenizerActor:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    def process_text(self, text: str, row_id: int) -> dict:
        normalized = normalize_text(text)
        tokens = self.tokenizer.tokenize(normalized)
        formatted = " ".join(tokens)
        token_ids = self.tokenizer.encode(normalized, add_special_tokens=False)
        
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


@ray.remote
def process_batch(batch_data: list, actor) -> list:
    results = []
    
    for row_id, line in batch_data:
        try:
            data = json.loads(line.strip())
            text = data.get('text', '')
            
            if not text:
                continue
            
            processed = ray.get(actor.process_text.remote(text, row_id))
            results.append(processed)
            
        except (json.JSONDecodeError, KeyError):
            continue
    
    return results


def run_ray_pipeline(
    input_file: str,
    output_file: str,
    max_rows: int = None,
    num_actors: int = None
):
    """
    Run Ray distributed preprocessing pipeline with intermediate outputs.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        max_rows: Maximum number of rows to process (None = all)
        num_actors: Number of Ray actors (None = auto-detect)
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Determine number of actors
    if num_actors is None:
        num_actors = ray.available_resources().get('CPU', 4)
        num_actors = int(num_actors)
    
    print("\n" + "=" * 70)
    print(f"Running Running Ray Distributed Pipeline ({num_actors} actors)")
    print("=" * 70)
    
    # Setup performance tracking
    tracker = PerformanceTracker('ray')
    tracker.start()
    
    # Create tokenizer actors
    print(f"Creating Creating {num_actors} tokenizer actors...")
    actors = [TokenizerActor.remote() for _ in range(num_actors)]
    
    # Read input data
    print(f"Reading Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if max_rows:
        lines = lines[:max_rows]
    
    total_lines = len(lines)
    print(f"Processing Processing {total_lines:,} rows with {num_actors} actors...")
    
    # Create (row_id, line) tuples
    indexed_lines = list(enumerate(lines))
    
    # Split data into batches
    batch_size = max(100, total_lines // (num_actors * 10))
    batches = [indexed_lines[i:i + batch_size] for i in range(0, len(indexed_lines), batch_size)]
    
    print(f"Created Created {len(batches)} batches (size ~{batch_size} rows each)")
    
    # Distribute batches to actors in round-robin fashion
    futures = []
    for i, batch in enumerate(batches):
        actor = actors[i % num_actors]
        future = process_batch.remote(batch, actor)
        futures.append(future)
    
    # Collect results with progress bar
    all_results = []
    
    remaining_futures = futures
    pbar = tqdm(total=len(futures), desc="Processing batches")
    
    while remaining_futures:
        # Wait for at least one result
        ready_futures, remaining_futures = ray.wait(
            remaining_futures,
            num_returns=min(10, len(remaining_futures)),
            timeout=1.0
        )
        
        # Collect completed results
        for future in ready_futures:
            batch_results = ray.get(future)
            all_results.extend(batch_results)
            pbar.update(1)
            
            # Update memory tracking
            tracker.update_peak_memory()
    
    pbar.close()
    
    # Sort by ID to maintain order
    all_results.sort(key=lambda x: x['id'])
    
    # Prepare intermediate outputs
    normalized_data = [{'id': r['id'], 'text': r['normalized']} for r in all_results]
    tokenized_data = [{'id': r['id'], 'tokens': r['tokens'], 'token_count': r['token_count']} for r in all_results]
    formatted_data = [{'id': r['id'], 'formatted': r['formatted'], 'token_ids': r['token_ids']} for r in all_results]
    
    # Save intermediate outputs
    print("\nSaving Saving intermediate outputs...")
    intermediate_dir = "results/intermediate/ray"
    
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
        'runner': 'ray',
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
    print("=== Intermediate Outputs Saved ===")
    print("=" * 70)
    print(f"Normalized → {intermediate_dir}/normalized.jsonl ({norm_count:,} rows)")
    print(f"Tokenized  → {intermediate_dir}/tokenized.jsonl ({tok_count:,} rows)")
    print(f"Formatted  → {intermediate_dir}/formatted.jsonl ({fmt_count:,} rows)")
    print("=" * 70 + "\n")
    
    print(f"Saved Final output saved to: {output_file}")
    print(f"Created Output size: {output_size_mb:.2f} MB\n")
    
    # Shutdown Ray
    ray.shutdown()


def main():
    """Main entry point for Ray pipeline."""
    parser = argparse.ArgumentParser(
        description="Ray distributed text preprocessing pipeline"
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
        default='results/output_ray.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (default: all)'
    )
    parser.add_argument(
        '--actors',
        type=int,
        default=None,
        help='Number of Ray actors (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    run_ray_pipeline(
        input_file=args.input,
        output_file=args.output,
        max_rows=args.max_rows,
        num_actors=args.actors
    )


if __name__ == '__main__':
    main()
