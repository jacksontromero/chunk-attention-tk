/**
 * ThunderKittens implementation of attn_chunk_first_kernel_v2
 *
 * Original: chunk_attn/cpp/chunk_attn/kernel_cuda.cu (lines 796-884)
 *
 * Computes chunked attention for shared prefixes:
 *   scores = Q @ K^T * scale           [n_seqs, chunk_size]
 *   maxs = row_max(scores)             [n_seqs]
 *   exp_scores = exp(scores - maxs)    [n_seqs, chunk_size]
 *   sums = row_sum(exp_scores)         [n_seqs]
 *   attns = exp_scores @ V             [n_seqs, d_head]  (unnormalized)
 *
 * Grid: (n_heads, n_chunks)
 * Block: 128 threads (4 warps)
 *
 * Performance optimizations:
 *   - 4 warps for better parallelism (matches original)
 *   - Shared memory reuse: K and V share same buffer (40% less smem)
 *   - TMA for Q loading (hardware-accelerated async transfer)
 *   - Warpgroup MMA (wgmma) for efficiency on H100
 *   - Efficient warp-level softmax primitives
 */

#include "kittens.cuh"
#include "../../include/kittens.cuh"
using namespace kittens;

// Forward declarations for tile types used in globals
template<int max_n_seqs, int d_head>
using Q_tile = st<bf16, max_n_seqs, d_head>;

// ============================================================================
// Async memory copy helpers (cp.async for H100)
// ============================================================================
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// 16-byte async copy (8 bf16 elements) using swizzled address
__device__ __forceinline__ void cp_async_16_swizzled(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// Sync load from raw bf16 pointer into TK shared tile (linear access, swizzled storage)
template<int ROWS, int COLS, int THREADS>
__device__ __forceinline__ void load_tile_sync(st<bf16, ROWS, COLS>& dst, const bf16* src) {
    constexpr int total_elements = ROWS * COLS;
    #pragma unroll
    for (int i = threadIdx.x; i < total_elements; i += THREADS) {
        dst[{i / COLS, i % COLS}] = src[i];
    }
}

// Async load from raw bf16 pointer into TK shared tile with proper swizzling
// Uses TK's idx() function to compute swizzled destination addresses
template<int ROWS, int COLS, int THREADS>
__device__ __forceinline__ void load_tile_async(st<bf16, ROWS, COLS>& dst, const bf16* src) {
    constexpr int elem_per_memcpy = 8;  // 16 bytes / 2 bytes per bf16
    constexpr int memcpy_per_row = COLS / elem_per_memcpy;
    constexpr int total_elements = ROWS * COLS;
    constexpr int total_memcpy = total_elements / elem_per_memcpy;
    constexpr int calls_per_thread = (total_memcpy + THREADS - 1) / THREADS;

    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    #pragma unroll
    for (int i = 0; i < calls_per_thread; i++) {
        int load_idx = i * THREADS + threadIdx.x;
        if (load_idx < total_memcpy) {
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            // Use TK's idx() for swizzled destination address
            uint32_t swizzled_addr = dst.idx(dst_ptr, {row, col});
            cp_async_16_swizzled(swizzled_addr, &src[row * COLS + col]);
        }
    }
}

// Configuration: 4 warps, each handling 16 rows
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_SIZE = NUM_WARPS * kittens::WARP_THREADS;

/**
 * Kernel globals - mirrors original kernel parameters
 */
template<int max_n_seqs, int d_head>
struct attn_chunk_first_globals {
    // Q tile type for TMA
    using Q_st = st<bf16, max_n_seqs, d_head>;
    // Q global layout: [1, n_heads, n_seqs, d_head] with TMA descriptor
    using Q_gl = gl<bf16, 1, -1, -1, d_head, Q_st>;

    // Outputs
    gl<float, -1, -1, -1, -1> attns;   // [total_seqs, d_head] partial attention
    gl<float, -1, -1, -1, -1> maxs;    // [total_seqs] row maxima
    gl<float, -1, -1, -1, -1> sums;    // [total_seqs] row sums

    // Chunk metadata
    gl<int, -1, -1, -1, -1> offsets;   // [n_chunks] output offset per chunk
    gl<int, -1, -1, -1, -1> begins;    // [n_chunks] seq range start
    gl<int, -1, -1, -1, -1> ends;      // [n_chunks] seq range end

    // Inputs - Q uses TMA, K/V use manual async
    Q_gl Q;                             // [1, n_heads, n_seqs, d_head] for TMA
    void** keys;                        // [n_chunks] -> [n_heads, chunk_size, d_head]
    void** values;                      // [n_chunks] -> [n_heads, chunk_size, d_head]

    float scale;                        // 1/sqrt(d_head)
    int n_heads;
};

template<int max_n_seqs, int chunk_size, int d_head>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
attn_chunk_first_tk(const __grid_constant__ attn_chunk_first_globals<max_n_seqs, d_head> g) {
    static_assert(max_n_seqs % NUM_WARPS == 0, "max_n_seqs must be divisible by NUM_WARPS");
    constexpr int ROWS_PER_WARP = max_n_seqs / NUM_WARPS;

    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int warp = kittens::warpid();
    const int lane = kittens::laneid();
    const int my_row_start = warp * ROWS_PER_WARP;

    // Get sequence range for this chunk
    const int seq_begin = g.begins[chunk];
    const int seq_end = g.ends[chunk];
    const int n = seq_end - seq_begin;
    if (n <= 0 || n > max_n_seqs) return;

    // =========================================================================
    // Shared memory allocation (TMA-compatible allocator)
    // =========================================================================
    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    // KV_s is reused: first for K, then for V (they're never needed simultaneously)
    st<bf16, chunk_size, d_head> &KV_s = al.allocate<st<bf16, chunk_size, d_head>>();

    // Q tile (full block, TMA will load it)
    st<bf16, max_n_seqs, d_head> &Q_s = al.allocate<st<bf16, max_n_seqs, d_head>>();

    // Per-warp output tiles
    auto &out_s = al.allocate<st<float, ROWS_PER_WARP, d_head>, NUM_WARPS>()[warp];

    // Shared vectors for max/sum
    auto &max_sv = al.allocate<sv<float, ROWS_PER_WARP>, NUM_WARPS>()[warp];
    auto &sum_sv = al.allocate<sv<float, ROWS_PER_WARP>, NUM_WARPS>()[warp];

    // Semaphore for Q TMA load
    __shared__ kittens::semaphore q_semaphore;

    // =========================================================================
    // Phase 1: Start TMA load for Q, async load K
    // =========================================================================
    const bf16* K_ptr = reinterpret_cast<const bf16*>(g.keys[chunk]) + head * chunk_size * d_head;
    const bf16* V_ptr = reinterpret_cast<const bf16*>(g.values[chunk]) + head * chunk_size * d_head;

    // Initialize semaphore and start Q TMA load (only thread 0)
    if (threadIdx.x == 0) {
        init_semaphore(q_semaphore, 0, 1);
        tma::expect_bytes(q_semaphore, sizeof(Q_s));
        // Q layout is [1, n_heads, n_seqs, d_head], load tile for this head starting at seq_begin
        coord<st<bf16, max_n_seqs, d_head>> q_coord = {0, head, seq_begin, 0};
        tma::load_async(Q_s, g.Q, q_coord, q_semaphore);
    }

    // Meanwhile, all threads load K asynchronously
    load_tile_async<chunk_size, d_head, BLOCK_SIZE>(KV_s, K_ptr);
    cp_async_commit();

    // Wait for both K (async) and Q (TMA) to complete
    cp_async_wait<0>();
    __syncthreads();
    wait(q_semaphore, 0);

    // =========================================================================
    // Compute scores = Q @ K^T using Warpgroup
    // =========================================================================
    // scores_r fragment holds 16 rows (per warp) of the 64x64 result
    rt<float, ROWS_PER_WARP, chunk_size> scores_r;
    
    // warpgroup::mm_ABt uses Q_s (whole 64x128) and KV_s (K 64x128).
    // Computes Q @ K^T. Result distributed across warps.
    warpgroup::mm_ABt(scores_r, Q_s, KV_s);
    
    warp::mul(scores_r, scores_r, g.scale);

    // =========================================================================
    // Warp-level softmax (Operates on per-warp fragment, which contains complete rows)
    // =========================================================================
    col_vec<decltype(scores_r)> max_vec;
    warp::row_max(max_vec, scores_r);

    warp::sub_row(scores_r, scores_r, max_vec);
    warp::exp(scores_r, scores_r);

    col_vec<decltype(scores_r)> sum_vec;
    warp::zero(sum_vec);
    warp::row_sum(sum_vec, scores_r, sum_vec);

    // =========================================================================
    // Store max/sum
    // =========================================================================
    warp::store(max_sv, max_vec);
    warp::store(sum_sv, sum_vec);
    
    // We can load V now. Ensure K usage in Q@K is done.
    // Important: wgmma instructions read from shared memory asynchronously.
    // We must ensure the wgmma is finished reading KV_s (K) before we overwrite it with V.
    warpgroup::mma_async_wait(); 
    __syncthreads();

    // =========================================================================
    // Phase 2: Load V (Async)
    // =========================================================================
    load_tile_async<chunk_size, d_head, BLOCK_SIZE>(KV_s, V_ptr);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int result_offset = g.offsets[chunk];
    const int out_base = result_offset * g.n_heads + head * n;
    float* maxs_out = g.maxs.raw_ptr + out_base;
    float* sums_out = g.sums.raw_ptr + out_base;

    for (int r = lane; r < ROWS_PER_WARP; r += 32) {
        int global_row = my_row_start + r;
        if (global_row < n) {
            maxs_out[global_row] = max_sv[r];
            sums_out[global_row] = sum_sv[r];
        }
    }

    // =========================================================================
    // Warp-level MMA: output = exp_scores @ V
    // =========================================================================
    // Convert scores to bf16
    rt<bf16, ROWS_PER_WARP, chunk_size> exp_scores_bf16;
    exp_scores_bf16 = scores_r;

    rt<float, ROWS_PER_WARP, d_head> out_r;
    warp::zero(out_r);
    
    // warpgroup::mma_AB
    // A: scores [64, 64] (distributed).
    // B: KV_s (V) [64, 128] (shared).
    // Out: [64, 128] (distributed).
    warpgroup::mma_AB(out_r, exp_scores_bf16, KV_s);
    warpgroup::mma_async_wait();

    warp::store(out_s, out_r);
    __syncwarp();

    // =========================================================================
    // Store output
    // =========================================================================
    float* attns_out = g.attns.raw_ptr + out_base * d_head;

    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = my_row_start + r;
        if (global_row >= n) continue;
        for (int c = lane; c < d_head; c += 32) {
            attns_out[global_row * d_head + c] = out_s[{r, c}];
        }
    }
}

#include "harness.impl"