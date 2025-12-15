# Chunked Attention (ThunderKittens)

ThunderKittens implementations of chunked attention kernels from `chunk_attn/cpp/chunk_attn/kernel_cuda.cu`.

## Kernels

### attn_chunk_first

Computes partial (unnormalized) attention for shared-prefix chunks:

```
scores = Q @ K^T * scale       # [n_seqs, chunk_size]
maxs = row_max(scores)         # [n_seqs] - for numerical stability
exp_scores = exp(scores - maxs)
sums = row_sum(exp_scores)     # [n_seqs] - for later normalization
attns = exp_scores @ V         # [n_seqs, d_head] - unnormalized partial result
```

### attn_seq_first

Computes final normalized attention by combining chunks per sequence:

```
For each chunk:
    scores = Q @ K^T * scale
    (online softmax accumulation across chunks)

output = normalized(accumulated_attention)  # [n_seqs, n_heads, d_head]
```

## Build & Test

```bash
# Build both kernels
make clean && make

# --- attn_chunk_first ---
python gentests.py --quick              # Generate chunk_first tests
./run_tests.sh                          # Run all chunk_first tests
./attn_chunk_first tests/randn_s16_h4_c2.txt -v  # Single test

# --- attn_seq_first ---
python gentests_seq_first.py --quick    # Generate seq_first tests
./run_tests_seq_first.sh                # Run all seq_first tests
./attn_seq_first tests_seq_first/randn_s4_h4_c2.txt -v  # Single test
```

## Configuration

### attn_chunk_first

| Parameter  | Value | Notes                         |
| ---------- | ----- | ----------------------------- |
| MAX_N_SEQS | 64    | Max query rows (compile-time) |
| CHUNK_SIZE | 64    | K/V chunk size (compile-time) |
| D_HEAD     | 128   | Head dimension (compile-time) |
| NUM_WARPS  | 2     | Row-parallel warps per block  |

### attn_seq_first

| Parameter  | Value | Notes                          |
| ---------- | ----- | ------------------------------ |
| CHUNK_SIZE | 64    | K/V chunk size (compile-time)  |
| D_HEAD     | 128   | Head dimension (compile-time)  |
| NUM_WARPS  | 4     | Chunk-parallel warps per block |

## Architecture

### attn_chunk_first

- **Grid**: `(n_heads, n_chunks)` - one block per (head, chunk) pair
- **Block**: 64 threads (2 warps), each warp handles 16 query rows
- **Strategy**: K/V loaded cooperatively, then each warp processes its rows independently

### attn_seq_first

- **Grid**: `(n_heads, n_seqs)` - one block per (head, sequence) pair
- **Block**: 128 threads (4 warps), warps process chunks in round-robin
- **Strategy**: Each warp accumulates partial results using online softmax, then cross-warp reduction

## Differences from Original

| Aspect    | Original     | TK Version                            |
| --------- | ------------ | ------------------------------------- |
| Data type | half (fp16)  | bf16                                  |
| MMA       | WMMA         | TK warp::mma / matvec patterns        |
| Softmax   | Manual loops | TK primitives (row_max, exp, row_sum) |

## TK Primitives Used

### attn_chunk_first

- `warp::load` / `warp::store` - tile transfers
- `warp::mma_ABt` - Q @ K^T matrix multiply
- `warp::row_max` / `warp::sub_row` / `warp::exp` / `warp::row_sum` - softmax

### attn_seq_first

- `warp::load` / `warp::store` - tile/vector transfers
- `warp::broadcast_col` + `warp::mul` + `warp::row_sum` - Q @ K^T matvec
- `warp::broadcast_row` + `warp::mul` + `warp::col_sum` - scores @ V matvec
- `warp::max` / `warp::sum` - vector reductions
- `warp::add` / `warp::exp` - softmax computation

## Files

| File                     | Description                            |
| ------------------------ | -------------------------------------- |
| `attn_chunk_first.cu`    | Chunk-first kernel with inline harness |
| `attn_seq_first.cu`      | Seq-first kernel with inline harness   |
| `harness.impl`           | Test harness for chunk_first           |
| `harness_seq_first.impl` | Test harness for seq_first             |
| `gentests.py`            | Test generator for chunk_first         |
| `gentests_seq_first.py`  | Test generator for seq_first           |
| `run_tests.sh`           | Parallel test runner for chunk_first   |
| `run_tests_seq_first.sh` | Parallel test runner for seq_first     |
