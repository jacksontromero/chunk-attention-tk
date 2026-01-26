# Chunk Attention TK Optimization Plan

## Current Status (Updated)

### Completed Optimizations
- **Phase 1: Shared Memory Reuse** - K and V now share same buffer (`KV_s`), reducing smem by ~40%
- **Phase 2: Async Pipelining** - V load overlaps with Q@K^T compute using cp.async with proper swizzling

### Performance Results
- **50% improvement** from async pipelining
- Still ~1.6-2x slower than native kernel (kernel-only timing)

---

## Benchmark Validation

### Are the benchmarks fair?
**Yes, both use CUDA events for kernel-only timing.** Key differences:

| Metric | TK Kernel | Native Kernel |
|--------|-----------|---------------|
| n_seqs | 64 | 32 |
| FLOPs formula | Same: `4 * n_seqs * chunk_size * d_head * blocks` |
| Timing method | CUDA events | CUDA events |

**Important**: TK does 2x the work per block (n_seqs=64 vs 32), but this is correctly accounted for in TFLOPS calculation.

### Normalized Performance Comparison

| blocks | Native (0.0117ms) | TK (0.038ms) | TK work-normalized | Slowdown |
|--------|-------------------|--------------|-------------------|----------|
| 256 | 22.9 TFLOPS | ~14 TFLOPS | ~0.019ms equivalent | **~1.6x** |
| 512 | 31.3 TFLOPS | ~10 TFLOPS | ~0.055ms equivalent | **~3.2x** |

---

## Why TK is Still Slower: Root Cause Analysis

### 1. Swizzled Memory Access Overhead
**Impact: Medium**

TK tiles use swizzled layout for optimal MMA access patterns. Every cp.async requires computing the swizzled destination address:
```cpp
// TK requires this for every 16-byte copy:
uint32_t swizzled_addr = dst.idx(dst_ptr, {row, col});
cp_async_16_swizzled(swizzled_addr, &src[row * COLS + col]);
```

Native kernel uses flat shared memory with direct pointer arithmetic - no address computation overhead.

### 2. K-Dimension Tiling (Native does, TK doesn't)
**Impact: High**

Native kernel tiles along the K dimension (d_head=128 processed in mma_k=16 chunks):
```cpp
// Native pattern: K-tiled loop with prefetching
for (int k = 0; k < k_mma_count; k++) {
    // Prefetch next K tile while computing current
    if (k + pre_load < k_mma_count) {
        load_tile_col_partition(..., k + pre_load);  // cp.async
        commit_async_cp_group();
    }
    wait_async_cp_group<pre_load>();
    // Compute with current tile
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}
```

Our TK kernel loads K entirely (64×128 = 16KB) before computing. Native overlaps K loads with computation within a single Q@K^T operation.

### 3. Register Pressure
**Impact: Medium**

- TK kernel: 194 registers (high but no spills)
- May limit occupancy, especially at high block counts
- Native kernel likely uses fewer registers due to K-tiling (smaller working set)

### 4. Per-warp Tile Sizes
**Impact: Low-Medium**

With n_seqs=64 and 4 warps, each warp handles 16 rows:
- scores_r: 16×64 (ROWS_PER_WARP × chunk_size)
- This is a reasonable size, but still larger per-warp working set than native

### 5. Q Load Pattern
**Impact: Low**

Native loads Q with cp.async during the K-tiled loop (interleaved with K loads). Our TK kernel loads Q synchronously before K.

### 6. Scaling Issues at High Block Counts
**Impact: Observed**

At 512+ blocks, TK performance degrades more than native. Possible causes:
- Occupancy limited by smem/registers
- Memory controller contention
- L2 cache thrashing

---

## Remaining Optimization Opportunities

### Phase 3: K-Dimension Tiling (Highest Priority)
**Expected Improvement: 1.5-2x**

Restructure to tile along d_head dimension:
```cpp
// Tile K in chunks of 16-32 columns
for (int k_tile = 0; k_tile < D_HEAD; k_tile += TILE_K) {
    // Async load next K tile
    load_tile_async(K_tile_s, K_ptr + k_tile);
    cp_async_commit();

    // Wait for previous tile
    cp_async_wait<1>();

    // Partial MMA accumulation
    warp::mma_ABt(scores_r, Q_r_tile, K_r_tile, scores_r);
}
```

**Challenges**:
- TK's `warp::mma_ABt` expects full tiles
- Would need to restructure Q and K as column-subtiles
- Significant refactoring required

### Phase 4: Reduce n_seqs to Match Native
**Expected Improvement: Investigation needed**

Test with n_seqs=32 to match native kernel's configuration:
- Would reduce per-warp work
- May improve register usage
- Direct apples-to-apples comparison

### Phase 5: TMA for Q/K/V Loads
**Expected Improvement: Unknown**

Use hardware TMA instead of cp.async:
- Requires `gl<>` typed inputs (significant API change)
- May reduce address computation overhead
- Could enable warp specialization (producer/consumer pattern)

---

## Files Modified

- `attn_chunk_first.cu` - Main kernel with smem reuse + async V load
- `harness.impl` - Updated smem calculation (in 15-779-Project version)

## Architecture Summary

```
Current TK Flow (with optimizations):
  [Sync load K] -> [Sync load Q] -> [K to regs]
                                        |
                                  [Async load V]  (overlapped)
                                        |
  [Q@K^T compute] <--------------------+
        |
  [Wait V] -> [Softmax] -> [V to regs] -> [scores@V] -> [Store]

Native Flow (more aggressive pipelining):
  for k_tile in K_tiles:
    [Async K_tile] -> [MMA partial] -> [Prefetch next]
                           |
  [Async V] <-------------+
        |
  [Softmax] -> [scores@V with K-tiled V loads] -> [Store]
```

## Conclusion

The remaining ~1.6-2x gap is primarily due to:
1. **K-dimension tiling** - Native pipelines within the matmul, we don't
2. **Swizzle overhead** - Address computation for every async copy
3. **Working set size** - n_seqs=64 means larger per-warp tiles

The benchmark methodology is valid. To close the gap, K-dimension tiling is the most impactful optimization remaining.
