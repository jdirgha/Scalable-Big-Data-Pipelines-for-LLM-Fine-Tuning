"""
Single-process text preprocessing pipeline
"""

import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from measure_utils import (
    PerformanceTracker, save_metrics, print_metrics, 
    get_file_size_mb, save_jsonl
)


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing special characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_text(text: str, tokenizer, row_id: int) -> dict:
    """Process text through normalization and tokenization pipeline."""
    normalized = normalize_text(text)
    tokens = tokenizer.tokenize(normalized)
    formatted = " ".join(tokens)
    token_ids = tokenizer.encode(normalized, add_special_tokens=False)
    
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


def run_single_process_pipeline(
    input_file: str,
    output_file: str,
    max_rows: int = None
):
    """
    Run single-process text preprocessing pipeline.
    
    Processes data sequentially in a single Python process.
    Good for small to medium datasets where parallelization overhead isn't worth it.
    """
    print("\n" + "=" * 70)
    print("Running Single-Process Pipeline")
    print("=" * 70)
    
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    tracker = PerformanceTracker('single')
    tracker.start()
    
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    if max_rows:
        total_lines = min(total_lines, max_rows)
    
    print(f"Processing {total_lines:,} rows...")
    
    # Store intermediate outputs at each pipeline stage
    normalized_data = []
    tokenized_data = []
    formatted_data = []
    final_data = []
    
    rows_processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        
        for line in tqdm(infile, total=total_lines, desc="Processing"):
            if max_rows and rows_processed >= max_rows:
                break
            
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                
                if not text:
                    continue
                
                # Process text through all pipeline stages
                processed = process_text(text, tokenizer, rows_processed)
                
                # Save intermediate outputs for analysis
                normalized_data.append({
                    'id': processed['id'],
                    'text': processed['normalized']
                })
                
                tokenized_data.append({
                    'id': processed['id'],
                    'tokens': processed['tokens'],
                    'token_count': processed['token_count']
                })
                
                formatted_data.append({
                    'id': processed['id'],
                    'formatted': processed['formatted'],
                    'token_ids': processed['token_ids']
                })
                
                final_data.append(processed)
                
                rows_processed += 1
                
                # Update memory tracking periodically
                if rows_processed % 1000 == 0:
                    tracker.update_peak_memory()
                
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    print("\nSaving intermediate outputs...")
    intermediate_dir = "results/intermediate/single"
    
    norm_count = save_jsonl(f"{intermediate_dir}/normalized.jsonl", normalized_data)
    tok_count = save_jsonl(f"{intermediate_dir}/tokenized.jsonl", tokenized_data)
    fmt_count = save_jsonl(f"{intermediate_dir}/formatted.jsonl", formatted_data)
    
    final_count = save_jsonl(output_file, final_data)
    
    perf_metrics = tracker.stop()
    
    output_size_mb = get_file_size_mb(output_file)
    throughput = rows_processed / perf_metrics['elapsed_sec']
    
    metrics = {
        'runner': 'single',
        'rows': rows_processed,
        'secs': perf_metrics['elapsed_sec'],
        'rows_per_sec': throughput,
        'peak_mb': perf_metrics['peak_memory_mb'],
        'bytes_out': int(output_size_mb * 1024 * 1024),
        'normalized_rows': norm_count,
        'tokenized_rows': tok_count,
        'formatted_rows': fmt_count
    }
    
    print_metrics(metrics)
    save_metrics(metrics)
    
    print("\n" + "=" * 70)
    print("Intermediate Outputs Saved")
    print("=" * 70)
    print(f"Normalized: {intermediate_dir}/normalized.jsonl ({norm_count:,} rows)")
    print(f"Tokenized:  {intermediate_dir}/tokenized.jsonl ({tok_count:,} rows)")
    print(f"Formatted:  {intermediate_dir}/formatted.jsonl ({fmt_count:,} rows)")
    print("=" * 70 + "\n")
    
    print(f"Output saved to: {output_file}")
    print(f"Output size: {output_size_mb:.2f} MB\n")


def main():
    """Main entry point for single-process pipeline."""
    parser = argparse.ArgumentParser(
        description="Single-process text preprocessing pipeline (baseline)"
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
        default='results/output_single.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    run_single_process_pipeline(
        input_file=args.input,
        output_file=args.output,
        max_rows=args.max_rows
    )


if __name__ == '__main__':
    main()
