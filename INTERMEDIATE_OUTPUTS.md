# Intermediate Outputs

The pipeline saves intermediate outputs at each preprocessing stage.

## Directory Structure

```
results/intermediate/
  single/
    normalized.jsonl
    tokenized.jsonl
    formatted.jsonl
  multi/
    normalized.jsonl
    tokenized.jsonl
    formatted.jsonl
  ray/
    normalized.jsonl
    tokenized.jsonl
    formatted.jsonl
```

## Preprocessing Stages

### 1. Normalization

Cleans the text:
- Lowercase
- Remove punctuation
- Collapse spaces

Output format:
```json
{"id": 0, "text": "cleaned text"}
```

### 2. Tokenization

Converts text to GPT-2 tokens using byte-pair encoding.

Output format:
```json
{"id": 0, "tokens": ["token1", "token2", ...], "token_count": 42}
```

### 3. Formatting

Prepares training-ready format with token IDs.

Output format:
```json
{"id": 0, "formatted": "reconstructed text", "token_ids": [123, 456, ...]}
```

## Metrics

Three new columns added to `metrics.csv`:
- `normalized_rows`
- `tokenized_rows`
- `formatted_rows`

## Inspecting Outputs

View normalized text:
```bash
head results/intermediate/single/normalized.jsonl
```

View tokens (formatted):
```bash
head -n 1 results/intermediate/single/tokenized.jsonl | python3 -m json.tool
```

View formatted output:
```bash
head -n 1 results/intermediate/single/formatted.jsonl | python3 -m json.tool
```

## File Sizes

For 1,000 rows:
- Normalized: ~200 KB
- Tokenized: ~500 KB
- Formatted: ~500 KB

