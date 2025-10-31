"""
Dataset Generator

Generate a sample text dataset for benchmarking.
Uses Wikipedia text from Hugging Face datasets.
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def generate_sample_dataset(
    output_file: str,
    target_rows: int = 50000,
    source: str = 'wikitext'
):
    """
    Generate a sample JSONL dataset for benchmarking.
    
    Args:
        output_file: Path to output JSONL file
        target_rows: Target number of rows
        source: Source dataset ('wikitext' or 'openwebtext')
    """
    print("\n" + "=" * 70)
    print("Generating Generating Sample Dataset")
    print("=" * 70)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Loading {source} dataset from Hugging Face...")
    
    if source == 'wikitext':
        # Load WikiText dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    else:
        # Load a subset of OpenWebText or similar
        print("⚠️  OpenWebText is large. Loading a small subset...")
        dataset = load_dataset('openwebtext', split='train', streaming=True)
    
    print(f"Target Target rows: {target_rows:,}")
    print(f"Output Output file: {output_file}")
    print(f"\nGenerating Generating dataset...")
    
    rows_written = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        
        # If we need more rows than available, we'll cycle through the dataset
        cycles_needed = (target_rows // len(dataset)) + 1 if hasattr(dataset, '__len__') else 1
        
        pbar = tqdm(total=target_rows, desc="Writing rows")
        
        for cycle in range(cycles_needed):
            for item in dataset:
                if rows_written >= target_rows:
                    break
                
                # Extract text
                text = item.get('text', '')
                
                # Skip empty or very short texts
                if not text or len(text.strip()) < 50:
                    continue
                
                # Create JSON record
                record = {
                    'id': rows_written,
                    'text': text.strip(),
                    'source': source,
                    'cycle': cycle
                }
                
                # Write to file
                f.write(json.dumps(record) + '\n')
                rows_written += 1
                pbar.update(1)
                
                if rows_written >= target_rows:
                    break
            
            if rows_written >= target_rows:
                break
        
        pbar.close()
    
    # Calculate file size
    file_size_mb = Path(output_file).stat().st_size / 1024 / 1024
    
    print("\n" + "=" * 70)
    print("Complete Dataset Generation Complete")
    print("=" * 70)
    print(f"Target Total rows: {rows_written:,}")
    print(f" File size: {file_size_mb:.2f} MB")
    print(f"Output Saved to: {output_file}")
    print("=" * 70 + "\n")


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate sample text dataset for benchmarking"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/sample_text.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=50000,
        help='Number of rows to generate'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='wikitext',
        choices=['wikitext', 'openwebtext'],
        help='Source dataset'
    )
    
    args = parser.parse_args()
    
    generate_sample_dataset(
        output_file=args.output,
        target_rows=args.rows,
        source=args.source
    )


if __name__ == '__main__':
    main()

