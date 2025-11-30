# Chunked Attention (ThunderKittens)

ThunderKittens implementation of `attn_chunk_first_kernel_v2` from `chunk_attn/cpp/chunk_attn/kernel_cuda.cu`.

## What it computes

For shared-prefix chunked attention:
```
scores = Q @ K^T * scale       # [n_seqs, chunk_size]
maxs = row_max(scores)         # [n_seqs] - for numerical stability
exp_scores = exp(scores - maxs)
sums = row_sum(exp_scores)     # [n_seqs] - for later normalization
attns = exp_scores @ V         # [n_seqs, d_head] - unnormalized partial result
```

## Build & Test

```bash
make clean && make
python gentests.py --quick     # Generate test files
./run_tests.sh                 # Run all tests
./attn_chunk_first tests/randn_s16_h4_c2.txt -v  # Single test with verbose output
```

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| MAX_N_SEQS | 32 | Max query rows (compile-time) |
| CHUNK_SIZE | 64 | K/V chunk size (compile-time) |
| D_HEAD | 128 | Head dimension (compile-time) |
| NUM_WARPS | 2 | Row-parallel warps per block |

## Architecture

- **Grid**: `(n_heads, n_chunks)` - one block per (head, chunk) pair
- **Block**: 64 threads (2 warps), each warp handles 16 query rows
- **Row-parallel**: K/V loaded cooperatively, then each warp processes its rows independently

## Differences from original

| Aspect | Original | TK Version |
|--------|----------|------------|
| Data type | half (fp16) | bf16 |
| Threads | 128 (4 warps) | 64 (2 warps) |
| MMA | WMMA | TK warp::mma |
| Parallelism | Warp-level softmax | Row-parallel |

## Files

- `attn_chunk_first.cu` - Kernel with inline documentation
- `harness.impl` - Test harness (included by kernel)
- `gentests.py` - Reference computation & test generation
- `run_tests.sh` - Parallel test runner
