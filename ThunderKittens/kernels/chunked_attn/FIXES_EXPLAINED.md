# ThunderKittens Chunked Attention Build Errors - Analysis and Fixes

## Summary of Original Errors

The original implementation had several fundamental type system and API usage errors in ThunderKittens (TK). The errors fell into these categories:

1. **Incorrect vector types for row operations**
2. **Raw pointer usage instead of global layout wrappers**
3. **Wrong function names (mm_* vs mma_*)**
4. **Dimension mismatches in matrix operations**
5. **Missing proper warpgroup synchronization**

## Detailed Error Analysis and Fixes

### 1. Row Vector Type Errors

**Original Error:**
```
error: static assertion failed
static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
```

**Problem:**
The code was using `rv<float, 16, naive>` for row reductions, but TK requires specific layout types based on the register tile layout.

**Fix:**
Use the `col_vec` type helper:
```cpp
// WRONG:
rv<float, 16, naive> max_vec;

// CORRECT:
col_vec<rt_fl<TILE_DIM, TILE_DIM>> max_vec;
```

**Explanation:**
For a row-major register tile `rt<T, rows, cols, row>`, the `col_vec` type is `rv<T, rows, ortho>`, not `naive`. TK's type system enforces this through the `rt_base::col_vec_layout` typedef, which automatically selects the correct layout based on whether the tile is row-major or column-major.

### 2. Loading from Raw Pointers

**Original Error:**
```
error: no instance of overloaded function "kittens::group<_GROUP_WARPS>::load" matches the argument list
argument types are: (kittens::st<float, 64, 128>, float *__restrict__, {...})
```

**Problem:**
Attempting to load directly from raw `float*` pointers into shared tiles without wrapping them in a global layout.

**Fix:**
Create global layout wrappers:
```cpp
// WRONG:
auto* k = reinterpret_cast<float*>(g.gKeys[chunk_idx]);
warp::load(Ks, k, {head_idx, 0, 0});

// CORRECT:
auto* k_ptr = reinterpret_cast<scalar_t*>(g.gKeys[chunk_idx]);
using kv_gl = gl<scalar_t, 1, -1, chunk_size, d_head>;
kv_gl k_layout{k_ptr, nullptr, g.n_heads, chunk_size, nullptr};
warpgroup::load(K_smem, k_layout, {0, head_idx, k_tile * TILE_DIM, 0});
```

**Explanation:**
TK's load operations require global memory to be wrapped in a `gl<>` layout descriptor. This provides:
- Dimensionality information
- Stride calculation
- Bounds checking
- Proper TMA (Tensor Memory Accelerator) usage on Hopper GPUs

### 3. Wrong Function Names

**Original Error:**
```
error: no instance of overloaded function "kittens::group<_GROUP_WARPS>::mm_ABt" matches the argument list
```

**Problem:**
Using `mm_ABt` and `mm_AB` instead of the correct TK function names.

**Fix:**
```cpp
// WRONG:
kittens::warp::mm_ABt(output_reg, Qr, Kr);
kittens::warp::mm_AB(attn_result_reg, output_reg, Vr);

// CORRECT:
warpgroup::mm_ABt(attn_block, Q_smem, K_smem);
warpgroup::mma_AB(o_reg, attn_block, V_smem);
```

**Explanation:**
TK uses `mma_AB` and `mma_ABt` (not `mm_*`) for matrix multiply-accumulate operations. Additionally, for Hopper GPUs, these should typically be `warpgroup::` operations (not `warp::`) to leverage warpgroup-level matrix operations.

### 4. Dimension Mismatches

**Original Error:**
```
argument types are: (kittens::rt<float, 16, 64, ...>, kittens::rt<float, 16, 128, ...>, kittens::rt<float, 64, 128, ...>)
```

**Problem:**
Matrix multiply dimensions didn't align correctly.

**Fix:**
Ensure dimensions match:
```cpp
// For Q @ K^T:
// Q is [TILE_DIM, d_head]
// K is [TILE_DIM, d_head]
// Result is [TILE_DIM, TILE_DIM]
warpgroup::mm_ABt(attn_block, Q_smem, K_smem);

// For attn @ V:
// attn is [TILE_DIM, TILE_DIM]
// V is [TILE_DIM, d_head]
// Result is [TILE_DIM, d_head]
warpgroup::mma_AB(o_reg, attn_block, V_smem);
```

### 5. Additional Important Fixes

#### Proper Tile Types
```cpp
// Use specific tile types:
st_fl<TILE_DIM, d_head>  // Shared tiles of floats
rt_fl<TILE_DIM, TILE_DIM> // Register tiles of floats
```

#### Softmax Implementation
The corrected version implements numerically stable softmax with online normalization:
```cpp
// Track running max and normalizer
warp::neg_infty(max_vec);  // Initialize to -infinity
warp::zero(norm_vec);

// For each K/V tile:
warp::row_max(max_vec, attn_block, max_vec);  // Update max
warp::sub_row(attn_block, attn_block, max_vec);  // Subtract for stability
warp::exp2(attn_block, attn_block);  // Compute exp (using exp2 with log2e scaling)
warp::row_sum(norm_vec, attn_block, norm_vec);  // Update normalizer
```

#### Warpgroup Synchronization
```cpp
// Wait for async operations to complete
warpgroup::mma_async_wait();
```

## Key Takeaways

1. **Use TK's type helpers**: `col_vec<>`, `row_vec<>` instead of manually constructing vector types
2. **Wrap raw pointers**: Always use `gl<>` layouts for global memory access
3. **Use correct API names**: `mma_AB/mma_ABt` for matrix operations
4. **Leverage warpgroup operations**: On Hopper, use `warpgroup::` for better performance
5. **Follow working examples**: The ring_attn kernel demonstrates the proper patterns

## Testing Recommendation

To test this kernel on your system:
```bash
cd /home/ubuntu/chunk-attn/chunk-attention/ThunderKittens/kernels/chunked_attn
make clean && make
```

The corrected kernel should now compile without type errors. You may still need to:
1. Implement the output storage logic (marked with TODO)
2. Set up the harness for testing
3. Tune tile sizes and pipeline depth for your specific use case
