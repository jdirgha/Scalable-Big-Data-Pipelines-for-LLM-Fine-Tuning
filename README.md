# LLM Pipeline Benchmark

Benchmarking Scalable Big Data Pipelines for LLM Data Preprocessing - METCS777 Term Paper Project

A benchmarking suite comparing single-process, multiprocessing, and Ray-distributed text preprocessing pipelines for Large Language Model data preparation.

## Overview

This project demonstrates the performance differences between three pipeline architectures:

1. **Single-Process (Baseline)** - Sequential processing on a single CPU core
2. **Multiprocessing** - Parallel processing using Python's multiprocessing module
3. **Ray Distributed** - Distributed processing using the Ray framework

### Key Metrics

- Throughput (rows/sec)
- Execution Time (seconds)
- Peak Memory Usage (MB)
- Output Size (bytes)

## Project Structure

```
llm_pipeline_benchmark/
├── data/                          
│   └── sample_text.jsonl          
├── results/                       
│   ├── metrics.csv                
│   ├── output_single.jsonl        
│   ├── output_multi.jsonl         
│   ├── output_ray.jsonl           
│   ├── intermediate/
│   │   ├── single/
│   │   ├── multi/
│   │   └── ray/
│   └── *.png                      
├── src/                           
│   ├── generate_dataset.py        
│   ├── measure_utils.py           
│   ├── single_process.py          
│   ├── multi_process.py           
│   ├── ray_runner.py              
│   ├── compare_formats.py         
│   └── plot_results.py            
├── report/                        
│   └── METCS777_Final_Report.md   
├── README.md                      
└── requirements.txt               
```

## Setup & Installation

### Prerequisites

Python 3.10 or higher

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- pyarrow
- psutil
- tqdm
- matplotlib
- ray
- transformers
- datasets

## Quick Start

### 1. Generate Dataset

```bash
cd src
python generate_dataset.py --output ../data/sample_text.jsonl --rows 50000
```

Options:
- `--output`: Output file path
- `--rows`: Number of rows to generate
- `--source`: Source dataset (wikitext or openwebtext)

### 2. Run Benchmarks

Run all three pipelines:

```bash
# Single-process baseline
python single_process.py --input ../data/sample_text.jsonl

# Multiprocessing
python multi_process.py --input ../data/sample_text.jsonl

# Ray distributed
python ray_runner.py --input ../data/sample_text.jsonl
```

Common options:
- `--input`: Input JSONL file
- `--output`: Output JSONL file
- `--max_rows`: Limit number of rows to process

Additional options:
- `--workers N` (multi_process.py): Number of worker processes
- `--actors N` (ray_runner.py): Number of Ray actors

### 3. Generate Visualizations

```bash
python plot_results.py --metrics ../results/metrics.csv
```

Generates:
- throughput.png
- memory.png
- execution_time.png
- combined_metrics.png
- efficiency_scatter.png

### 4. Compare File Formats (Optional)

```bash
python compare_formats.py --input ../results/output_single.jsonl
```

## Usage Examples

### Quick Test

```bash
python single_process.py --max_rows 5000
python multi_process.py --max_rows 5000
python ray_runner.py --max_rows 5000
```

### Custom Worker Counts

```bash
python multi_process.py --workers 4
python ray_runner.py --actors 8
```

### Large Dataset

```bash
python generate_dataset.py --rows 200000 --output ../data/large.jsonl
python single_process.py --input ../data/large.jsonl --max_rows 100000
python multi_process.py --input ../data/large.jsonl --max_rows 100000
python ray_runner.py --input ../data/large.jsonl --max_rows 100000
```

## Sample Results

For a ~32K row dataset on an 8-core CPU:

| Runner | Time | Throughput | Speedup | Memory |
|--------|------|------------|---------|--------|
| single | 8.7s | 3,747 rows/sec | 1.0x | 310 MB |
| multi | 10.1s | 3,210 rows/sec | 0.86x | 199 MB |
| ray | 87.2s | 372 rows/sec | 0.10x | 183 MB |

Note: For this small dataset, single-process was fastest due to parallelization overhead. With larger datasets (100K+ rows), multi and Ray show significant speedups.

## Pipeline Stages

Each pipeline performs three preprocessing stages:

### Stage 1: Normalization
- Convert to lowercase
- Remove punctuation
- Collapse multiple spaces

### Stage 2: Tokenization
- GPT-2 tokenizer (BPE)
- Generate token list

### Stage 3: Formatting
- Reconstruct text from tokens
- Generate token IDs

Intermediate outputs saved to `results/intermediate/<runner>/` directory.

## Metrics CSV Format

```csv
runner,rows,secs,rows_per_sec,peak_mb,bytes_out,normalized_rows,tokenized_rows,formatted_rows
single,50000,8.66,3747.18,310.41,7887814,50000,50000,50000
```

## Troubleshooting

### Ray Initialization Issues

```bash
ray stop
python ray_runner.py
```

### Out of Memory

Reduce the number of rows or workers:
```bash
python multi_process.py --max_rows 5000 --workers 2
```

### Slow Tokenizer Download

Pre-download the tokenizer:
```bash
python -c "from transformers import GPT2TokenizerFast; GPT2TokenizerFast.from_pretrained('gpt2')"
```

## Technical Details

### Text Processing

Each pipeline performs:
1. Load data from JSONL
2. Tokenize using GPT-2 tokenizer
3. Extract metadata (token count, character count)
4. Write output to JSONL
5. Track performance metrics

### Performance Tracking

- Time: `time.time()` for wall-clock measurement
- Memory: `psutil.Process().memory_info().rss` for RSS memory
- Throughput: rows processed / elapsed time

## Files Generated

After running the full benchmark:

**Metrics:**
- `results/metrics.csv` - Performance data

**Outputs:**
- `results/output_single.jsonl`
- `results/output_multi.jsonl`
- `results/output_ray.jsonl`

**Intermediate outputs:**
- `results/intermediate/<runner>/normalized.jsonl`
- `results/intermediate/<runner>/tokenized.jsonl`
- `results/intermediate/<runner>/formatted.jsonl`

**Visualizations:**
- `results/*.png` (5 plots)

## Master Runner Script

For convenience, use the master runner:

```bash
python run_all.py
```

This will:
1. Generate dataset if needed
2. Run all three pipelines
3. Generate visualizations
4. Optionally compare formats

## References

- Ray Documentation: https://docs.ray.io/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Python Multiprocessing: https://docs.python.org/3/library/multiprocessing.html

## License

Academic project for METCS777 coursework.
