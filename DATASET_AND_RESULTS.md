# Dataset and Results Documentation

## Overview
This document provides a detailed explanation of the dataset used for benchmarking LLM pipelines, the methodology, and the results obtained from running the benchmark suite.

---

## 1. Dataset Description

### 1.1 Dataset Source and Generation

The benchmark dataset is synthetically generated using the `generate_dataset.py` script, which creates realistic text data suitable for testing LLM preprocessing pipelines. The data simulates real-world text processing scenarios commonly encountered in machine learning and NLP applications.

**Dataset File**: `data/sample_500_rows.jsonl` (500 rows, ~357 KB)
- **Full dataset**: 50,000 rows (~22 MB) - generated locally, not committed to Git
- **Sample included**: 500 rows for demonstration and quick testing

### 1.2 Dataset Structure

Each record in the dataset is a JSON object with the following schema:

```json
{
  "id": <integer>,
  "text": "<string>",
  "source": "<string>",
  "cycle": <integer>
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier for each record (0-indexed) |
| `text` | string | Text content (100-500 characters), sourced from WikiText |
| `source` | string | Source identifier (e.g., "wikitext") |
| `cycle` | integer | Generation cycle identifier (typically 0) |

### 1.3 Sample Data Examples

**Example 1:**
```json
{
  "id": 0,
  "text": "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the \" Nameless \" , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit \" Calamaty Raven \" .",
  "source": "wikitext",
  "cycle": 0
}
```

**Example 2:**
```json
{
  "id": 1,
  "text": "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .",
  "source": "wikitext",
  "cycle": 0
}
```

### 1.4 Dataset Characteristics

- **Text Length**: Variable, typically 100-500 characters per record
- **Character Set**: Mixed ASCII and Unicode (includes special characters, punctuation)
- **Content Type**: Informational text (Wikipedia-style articles)
- **Language**: English
- **Total Records**: 
  - Sample dataset: 500 rows
  - Full dataset: 50,000 rows (32,432 rows used in benchmark tests)
- **File Format**: JSONL (JSON Lines) - one JSON object per line

### 1.5 Dataset Generation

To regenerate the full dataset:

```bash
cd src
python generate_dataset.py --rows 50000
```

**Generation Parameters:**
- `--rows`: Number of records to generate (default: 50000)
- Output: `data/sample_text.jsonl`
- Generation time: ~1-2 seconds
- Deterministic: Uses fixed random seed for reproducibility

---

## 2. Pipeline Processing

### 2.1 Processing Steps

The benchmark tests three different pipeline implementations, each applying the following transformations:

1. **Load** - Read data from JSONL format
2. **Tokenize** - Split text into words using whitespace tokenization
3. **Filter** - Remove records with fewer than 10 tokens
4. **Count** - Count tokens and add `token_count` field
5. **Save** - Write results to JSONL format

### 2.2 Processing Implementations

| Implementation | Description | Parallelism |
|----------------|-------------|-------------|
| **Single** | Single-process Python | None (sequential) |
| **Multi** | Python multiprocessing | CPU cores (8 workers) |
| **Ray** | Distributed Ray framework | Cluster/multi-core |

---

## 3. Benchmark Results

### 3.1 Full Dataset Results (32,432 rows)

| Runner | Rows | Time (s) | Throughput (rows/sec) | Peak Memory (MB) | Output Size (bytes) |
|--------|------|----------|----------------------|------------------|---------------------|
| **Single** | 32,432 | 8.66 | **3,747.18** | 310.41 | 7,887,814 |
| **Multi** | 32,432 | 10.10 | 3,210.12 | **199.20** | 7,887,814 |
| **Ray** | 32,432 | 87.24 | 371.77 | **182.84** | 7,887,814 |

### 3.2 Small Dataset Results (1,000 rows)

| Runner | Rows | Time (s) | Throughput (rows/sec) | Peak Memory (MB) | Output Size (bytes) |
|--------|------|----------|----------------------|------------------|---------------------|
| **Single** | 1,000 | 0.60 | **1,658.30** | 308.34 | 1,351,745 |
| **Multi** | 1,000 | 5.19 | 192.70 | 203.48 | 1,351,745 |
| **Ray** | 1,000 | 13.93 | 71.81 | **111.39** | 1,351,745 |

### 3.3 Micro Dataset Results (100 rows)

| Runner | Rows | Time (s) | Throughput (rows/sec) | Peak Memory (MB) |
|--------|------|----------|----------------------|------------------|
| **Single** | 100 | 0.13 | **776.43** | 357.88 |

---

## 4. Results Analysis

### 4.1 Key Findings

#### **Single-Process Implementation (Winner for Throughput)**
-  **Highest throughput** across all dataset sizes
-  **Best for small-to-medium datasets** (< 50K rows)
-  **No overhead** from parallelization
-  **Higher memory usage** (310 MB)
-  **Single CPU core utilization**

**Why it's fastest:**
- No inter-process communication overhead
- No serialization/deserialization costs
- Minimal context switching
- Direct memory access

#### **Multiprocessing Implementation**
-  **Better memory efficiency** than single (199 MB vs 310 MB)
-  **Multi-core utilization**
-  **Slower than single-process** due to overhead
-  **High startup cost** for small datasets

**Performance characteristics:**
- 15% slower than single-process (32K rows)
- 88% slower than single-process (1K rows)
- Overhead dominates for datasets < 100K rows

#### **Ray Implementation**
-  **Lowest memory footprint** (182 MB)
-  **Best scalability** for distributed computing
-  **Designed for clusters** and large-scale data
-  **Significant overhead** for small datasets
-  **90% slower** than single-process (32K rows)

**When to use Ray:**
- Datasets > 1M rows
- Distributed cluster environments
- Complex multi-stage pipelines
- Fault-tolerant processing required

### 4.2 Performance Trends

**Throughput by Dataset Size:**

| Dataset Size | Single (rows/s) | Multi (rows/s) | Ray (rows/s) |
|--------------|-----------------|----------------|--------------|
| 100 rows | 776.43 | N/A | N/A |
| 1,000 rows | 1,658.30 | 192.70 | 71.81 |
| 32,432 rows | 3,747.18 | 3,210.12 | 371.77 |

**Key Observations:**
1. Single-process throughput **increases** with dataset size (better amortization)
2. Multi-process shows improvement at scale but still loses to single
3. Ray's overhead is consistent (~10x slower) regardless of size

### 4.3 Memory Efficiency

**Peak Memory Usage:**
- **Ray**: 182 MB (most efficient)
- **Multi**: 199 MB (efficient)
- **Single**: 310 MB (higher due to full in-memory processing)

**Memory per 1,000 rows:**
- Single: 9.56 MB/1K rows
- Multi: 6.14 MB/1K rows
- Ray: 5.64 MB/1K rows

### 4.4 Efficiency Metrics

**Time Efficiency (seconds per 1,000 rows):**
- Single: 0.267s per 1K rows ⚡
- Multi: 0.311s per 1K rows
- Ray: 2.689s per 1K rows

**Memory-Time Product (MB·s, lower is better):**
- Single: 2,687 MB·s
- Multi: 2,012 MB·s  **(Best overall efficiency)**
- Ray: 15,952 MB·s

---

## 5. Visualizations

The benchmark generates three visualization plots:

### 5.1 Throughput Comparison (`results/throughput.png`)
- Bar chart showing rows/second for each implementation
- Clearly demonstrates single-process superiority for this dataset size

### 5.2 Combined Metrics (`results/combined_metrics.png`)
- Three subplots showing:
  - Execution time comparison
  - Memory usage comparison
  - Throughput comparison

### 5.3 Efficiency Scatter (`results/efficiency_scatter.png`)
- Scatter plot with memory (x-axis) vs time (y-axis)
- Shows trade-offs between memory and speed
- Ideal implementations are bottom-left (low memory, low time)

---

## 6. Intermediate Outputs

All intermediate pipeline stages are saved for inspection:

```
results/intermediate/
├── single/
│   ├── loaded.jsonl      # Raw data loaded
│   ├── tokenized.jsonl   # After tokenization
│   ├── filtered.jsonl    # After filtering (< 10 tokens removed)
│   └── counted.jsonl     # Final output with token counts
├── multi/
│   └── ... (same structure)
└── ray/
    └── ... (same structure)
```

See `INTERMEDIATE_OUTPUTS.md` for detailed inspection of intermediate stages.

---

## 7. Recommendations

### 7.1 For This Dataset Size (< 100K rows)
**Use Single-Process Implementation**
- Fastest processing time
- Simplest code
- Acceptable memory usage
- Best for interactive development

### 7.2 For Large Datasets (> 1M rows)
**Consider Distributed Processing (Ray)**
- Memory efficiency becomes critical
- Parallelization overhead amortizes
- Fault tolerance valuable
- Cluster deployment options

### 7.3 For Production Systems
**Hybrid Approach:**
- Use single-process for small batches
- Use multiprocessing for medium workloads (100K-1M rows)
- Use Ray for large-scale distributed processing
- Monitor memory constraints on target hardware

---

## 8. Reproducibility

### 8.1 Generate Full Dataset
```bash
cd src
python generate_dataset.py --rows 50000
```

### 8.2 Run Benchmarks
```bash
# Single test (1,000 rows)
python run_benchmark.py --runner single --rows 1000

# Full benchmark (all implementations, 32K rows)
python run_benchmark.py --runner single
python run_benchmark.py --runner multi
python run_benchmark.py --runner ray
```

### 8.3 Generate Visualizations
```bash
python plot_metrics.py
```

---

## 9. System Configuration

**Test Environment:**
- **OS**: macOS 14.6 (Darwin 24.6.0)
- **Python**: 3.x
- **CPU**: 8 cores (assumed for multiprocessing)
- **Ray Workers**: 4 (default configuration)

---

## 10. Conclusion

This benchmark demonstrates that **simple single-process implementations often outperform complex distributed frameworks** for small-to-medium datasets (<100K rows). The overhead of parallelization (serialization, IPC, scheduling) dominates processing time for datasets that fit comfortably in memory.

**Key Takeaways:**
1.  **Single-process is fastest** for this dataset size
2.  **Multiprocessing offers best memory-time trade-off**
3.  **Ray is ideal for distributed/large-scale scenarios**
4.  **Choose implementation based on dataset size and infrastructure**

For the 32,432-row dataset used in this benchmark, **single-process processing is 17% faster** than multiprocessing and **90% faster** than Ray, while consuming reasonable memory (310 MB).

---

## References

- Full benchmark code: `src/run_benchmark.py`
- Dataset generator: `src/generate_dataset.py`
- Visualization code: `src/plot_metrics.py`
- Intermediate outputs: `INTERMEDIATE_OUTPUTS.md`
- Project README: `README.md`

