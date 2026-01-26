# ChunkAttention with ThunderKittens

Custom GPU kernels for prefix-aware attention using [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), a tile-based DSL for writing performant CUDA kernels. This was a course project for CMU 15-779 (Advanced Topics in Machine Learning Systems).

Full report: [`docs/MLSys_project_report.pdf`](docs/MLSys_project_report.pdf)

## Overview

[ChunkAttention](https://arxiv.org/abs/2402.15220) optimizes LLM inference when batches share common prefixes (system prompts, few-shot examples, etc.) by splitting attention into two phases:

1. **Chunk-first**: Process shared prefix chunks with batched queries from all sequences. This is matrix-matrix multiply and can use tensor cores.

2. **Sequence-first**: Process unique suffixes one sequence at a time, merging with cached prefix results. This is vector-matrix multiply and doesn't benefit from tensor cores.

We reimplemented both kernels using ThunderKittens to see if TK's MMA primitives and tile abstractions could improve on the original hand-written CUDA.

## Results

Benchmarked on H100 80GB, comparing our TK kernels against the original ChunkAttention CUDA kernels.

### Chunk-First Kernel

The TK kernel is **1.6-1.7x faster** than native across the board:

| Blocks | TK (TFLOPS) | Native (TFLOPS) | Speedup |
| ------ | ----------- | --------------- | ------- |
| 16     | 4.4         | 2.8             | 1.6x    |
| 256    | 61.1        | 31.1            | 2.0x    |
| 4096   | 68.5        | 43.6            | 1.6x    |
| 65536  | 74.3        | —               | —       |

The speedup comes from TK's MMA primitives providing better tensor core utilization on H100 than the original WMMA-based code.

### Sequence-First Kernel

The TK kernel is **slower** than native (~2.3 vs ~50 TFLOPS):

| Blocks | TK (TFLOPS) | Native (TFLOPS) |
| ------ | ----------- | --------------- |
| 16     | 1.8         | 8.5             |
| 4096   | 2.3         | 49.0            |
| 16384  | 2.3         | 53.6            |

This is expected. The sequence-first phase processes one query vector per sequence—it's vector-matrix products, not matrix-matrix. TK's tile abstractions add overhead (swizzled layouts, tile bookkeeping) without enabling tensor core use for this computation pattern. The native kernel's scalar dot products with manual shuffles are better suited here.

### Correctness

Both kernels pass all 36 test configurations (batch sizes 16-128, chunk counts 4-64, d_head=128).

## Repository Layout

```
├── ThunderKittens/                    # TK library (with our custom kernels)
│   └── kernels/chunked_attn/
│       ├── attn_chunk_first.cu        # Batched prefix attention (1.6x faster)
│       ├── attn_seq_first.cu          # Per-sequence suffix attention (slower)
│       ├── harness.impl               # Test harness for chunk_first
│       ├── harness_seq_first.impl     # Test harness for seq_first
│       └── Makefile
│
├── chunk_attn/                        # Original ChunkAttention (Microsoft)
│   └── cpp/chunk_attn/
│       └── kernel_cuda.cu             # Native CUDA kernels we compared against
│
├── benchmarks/                        # Benchmark code and results (submodule)
│   ├── bench/                         # Benchmark scripts
│   └── results/
│       └── chunk_attn_tk/
│           ├── chunk_first_vs_native_kernel_only/  # Kernel timing data
│           └── seq_first_vs_native_kernel_only/
│
└── docs/
    └── MLSys_project_report.pdf
```

## Implementation Notes

### Chunk-First Kernel (`attn_chunk_first.cu`)

Grid: `(n_heads, n_chunks)`, 4 warps per block (128 threads)

The kernel computes `softmax(Q @ K^T / sqrt(d)) @ V` for one chunk across all sequences in the batch. Key implementation choices:

- **Row-parallel design**: 4 warps each handle 16 rows of Q (for 64 total sequences).
- **bf16 throughout**: Better H100 support than fp16, and TK's MMA ops are optimized for it.
- **TK primitives for softmax**: `warp::row_max`, `warp::sub_row`, `warp::exp`, `warp::row_sum` replace manual shuffle-based reductions.
- **Warpgroup MMA**: `warpgroup::mm_ABt` for Q@K^T, `warpgroup::mma_AB` for scores@V.

Shared memory layout:

```
KV_s: [chunk_size, d_head]    # Reused for K then V (40% smem savings)
Q_s:  [max_n_seqs, d_head]    # Full Q tile, loaded via TMA
```

### Sequence-First Kernel (`attn_seq_first.cu`)

Grid: `(n_heads, n_seqs)`, 4 warps per block (128 threads)

This kernel is inherently harder to optimize. Each sequence has one query vector attending over all its chunks. For shared chunks, it merges cached results from chunk-first via online softmax. For unique chunks, it computes fresh attention.

The compute pattern:

- Q is `[d_head]`, K is `[chunk_size, d_head]`
- Score computation uses TK tile ops: broadcast Q across columns, element-wise multiply with K, then row-sum
- This avoids explicit matmuls but still can't use tensor cores (it's fundamentally vector-matrix)

TK's tile abstractions add overhead (swizzled layouts, tile bookkeeping) without enabling tensor core use for this computation pattern. The native kernel's scalar dot products with manual shuffles are more efficient here.

### Memory Loading (Chunk-First)

ChunkAttention stores KV cache as `void**` (array of pointers to per-chunk allocations). TK's TMA requires contiguous memory with compile-time strides, so chunk-first uses a hybrid approach:

- **Q uses TMA**: Hardware-accelerated async transfer directly into TK shared tiles
- **K/V use cp.async**: Manual async copies with pointer indirection, respecting TK's swizzled layout

Seq-first uses simpler manual loads since each warp processes chunks independently.

## Building

Requires CUDA 12.4+, H100 GPU, and ThunderKittens headers.

```bash
cd ThunderKittens/kernels/chunked_attn
make

# Run correctness tests
./run_tests.sh              # chunk_first
./run_tests_seq_first.sh    # seq_first
```

## Takeaways

ThunderKittens works well when your computation maps naturally to tiled matrix-matrix multiply. The chunk-first kernel's 1.6x speedup shows TK's WGMMA primitives can beat hand-written WMMA. But TK's abstractions add overhead for non-tiled patterns like vector-matrix products.

For prefix caching specifically: the chunk-first phase (shared prefix) benefits from TK, but the sequence-first phase (unique suffixes) doesn't. A production implementation might use TK for chunk-first and keep the native CUDA for sequence-first.

## Authors

Jackson Romero, Ayush Kumar, Jason Wei
CMU 15-779, Fall 2025
