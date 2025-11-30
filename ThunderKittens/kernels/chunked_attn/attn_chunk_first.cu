#include "../../include/kittens.cuh"
#include "kittens.cuh"

using namespace kittens;

// ===== Debug configuration =====
#define DEBUG_MODE 1          // Set to 1 to enable debug prints
#define DEBUG_CHECK_NAN 1     // Set to 1 to check for NaN/Inf values
#define DEBUG_STEP_BY_STEP 0  // Set to 1 to print after each step

// ===== Kernel configuration =====
// IMPORTANT: Using 32 threads (1 warp) because warp:: operations are designed for single-warp execution.
// Multi-warp execution causes each warp to compute row_offset = warpid * tile_rows, which goes OOB.
constexpr int BLOCK_SIZE = 32;  // 1 warp

template<int n_seqs, int d_head> struct attn_chunk_first_globals {
    using _gl_float = gl<float, -1, -1, -1, -1>;
    using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
    using _gl_int = gl<int, -1, -1, -1, -1>;

    _gl_float gAttns;           // Output attention values
    _gl_float gMaxs, gSums;     // Max/sum for online softmax
    _gl_int gOffsets, gBegins, gEnds;  // Chunk metadata

    // Q tensor: stored as [N_SEQS, N_HEADS, D_HEAD], accessed as [1, 1, N_SEQS, N_HEADS*D_HEAD] tiles
    // Row stride = N_HEADS * D_HEAD, tile cols = D_HEAD, so coord.c = head_idx
    gl<bf16, 1, 1, n_seqs, -1, st<bf16, n_seqs, d_head>> gQ;

    void** gKeys;     // Array of pointers to K chunks [n_heads, chunk_size, d_head] each
    void** gValues;   // Array of pointers to V chunks [n_heads, chunk_size, d_head] each

    float dim_scale;  // 1/sqrt(d_head)
    int n_heads;
};

template<int n_seqs, int chunk_size, int d_head>
__global__ void __launch_bounds__(BLOCK_SIZE) attn_chunk_first_tk(
    const __grid_constant__ attn_chunk_first_globals<n_seqs, d_head> g
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    const int head_idx = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    const int lane = laneid();

#if DEBUG_MODE
    if (lane == 0) {
        printf("[Block(%d,%d)] Starting: head=%d, chunk=%d\n",
               blockIdx.x, blockIdx.y, head_idx, chunk_idx);
    }
#endif

    // Read chunk metadata
    const int seq_begin = g.gBegins[chunk_idx];
    const int seq_end = g.gEnds[chunk_idx];
    const int n = seq_end - seq_begin;

#if DEBUG_MODE
    if (lane == 0) {
        printf("[Block(%d,%d)] seq_begin=%d, seq_end=%d, n=%d\n",
               blockIdx.x, blockIdx.y, seq_begin, seq_end, n);
    }
#endif

    // ===== STEP 1: Allocate shared memory =====
    st<bf16, n_seqs, d_head> &Qs = al.allocate<st<bf16, n_seqs, d_head>>();
    st<bf16, chunk_size, d_head> &Ks = al.allocate<st<bf16, chunk_size, d_head>>();
    st<bf16, chunk_size, d_head> &Vs = al.allocate<st<bf16, chunk_size, d_head>>();

#if DEBUG_STEP_BY_STEP
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 1: Shared memory allocated\n", blockIdx.x, blockIdx.y);
    }
#endif

    // ===== STEP 2: Load Q from global memory using TK's gl mechanism =====
    // Q is [N_SEQS, N_HEADS, D_HEAD], gl is [1, 1, N_SEQS, N_HEADS*D_HEAD]
    // coord.c = head_idx selects which head's D_HEAD columns to load
    // Row stride = N_HEADS * D_HEAD, so rows are correctly strided
    warp::load(Qs, g.gQ, {0, 0, 0, (int)head_idx});
    __syncwarp();

#if DEBUG_STEP_BY_STEP
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 2: Q loaded from global\n", blockIdx.x, blockIdx.y);
        bf16 q_val = Qs[{0, 0}];
        printf("[Block(%d,%d)]   Q[0,0] = %f\n", blockIdx.x, blockIdx.y, __bfloat162float(q_val));
    }
#endif

    // ===== STEP 3: Load K and V from global memory to shared memory =====
    // K/V are stored as [n_heads, chunk_size, d_head] per chunk
    const int kv_offset = head_idx * chunk_size * d_head;
    const bf16* k_ptr = reinterpret_cast<const bf16*>(g.gKeys[chunk_idx]) + kv_offset;
    const bf16* v_ptr = reinterpret_cast<const bf16*>(g.gValues[chunk_idx]) + kv_offset;

    // Load K into shared memory - use swizzled indexing via st[{row, col}]
    constexpr int total_kv_elems = chunk_size * d_head;
    constexpr int elems_per_thread = (total_kv_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int elem_idx = lane + i * BLOCK_SIZE;
        if (elem_idx < total_kv_elems) {
            int row = elem_idx / d_head;
            int col = elem_idx % d_head;
            Ks[{row, col}] = k_ptr[row * d_head + col];
        }
    }

    // Load V into shared memory
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int elem_idx = lane + i * BLOCK_SIZE;
        if (elem_idx < total_kv_elems) {
            int row = elem_idx / d_head;
            int col = elem_idx % d_head;
            Vs[{row, col}] = v_ptr[row * d_head + col];
        }
    }

    __syncwarp();

#if DEBUG_STEP_BY_STEP
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 3: K,V loaded to shared memory\n", blockIdx.x, blockIdx.y);
        bf16 k_val = Ks[{0, 0}];
        bf16 v_val = Vs[{0, 0}];
        printf("[Block(%d,%d)]   K[0,0] = %f, V[0,0] = %f\n",
               blockIdx.x, blockIdx.y, __bfloat162float(k_val), __bfloat162float(v_val));
    }
#endif

    // ===== STEP 4: Load shared tiles to register tiles =====
    rt<bf16, n_seqs, d_head> Qr;
    rt<bf16, chunk_size, d_head> Kr;
    rt<bf16, chunk_size, d_head, ducks::rt_layout::col> Vr;

    warp::load(Qr, Qs);
    warp::load(Kr, Ks);
    warp::load(Vr, Vs);

#if DEBUG_STEP_BY_STEP
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 4: Tiles loaded to registers\n", blockIdx.x, blockIdx.y);
    }
#endif

    // ===== STEP 5: Compute Q @ K^T (attention scores) =====
    rt<float, n_seqs, chunk_size> scores;
    warp::zero(scores);
    warp::mma_ABt(scores, Qr, Kr, scores);
    warp::mul(scores, scores, g.dim_scale);

#if DEBUG_STEP_BY_STEP
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 5: Computed attention scores\n", blockIdx.x, blockIdx.y);
    }
#endif

    // ===== STEP 6: Softmax - compute row max =====
    col_vec<rt<float, n_seqs, chunk_size>> maxes;
    warp::row_max(maxes, scores);

    // ===== STEP 7: Subtract max and exponentiate =====
    warp::sub_row(scores, scores, maxes);
    warp::exp(scores, scores);

    // ===== STEP 8: Compute row sums =====
    col_vec<rt<float, n_seqs, chunk_size>> sums;
    warp::row_sum(sums, scores);

#if DEBUG_STEP_BY_STEP
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 8: Computed softmax stats\n", blockIdx.x, blockIdx.y);
    }
#endif

    // ===== STEP 9: Store max and sum to global memory =====
    // Calculate output offsets (same as kernel_cuda.cu)
    const int result_offset = g.gOffsets[chunk_idx];
    const int max_sum_offset = result_offset * g.n_heads + head_idx * n;

#if DEBUG_MODE
    if (lane == 0) {
        printf("[Block(%d,%d)] result_offset=%d, max_sum_offset=%d\n",
               blockIdx.x, blockIdx.y, result_offset, max_sum_offset);
    }
#endif

    // Store max/sum directly to global memory
    // Vector store coord.c is in tile units (each tile = n_seqs elements)
    // So divide by n_seqs to get the correct tile index
    const int vec_tile_idx = max_sum_offset / n_seqs;
    warp::store(g.gMaxs, maxes, {0, 0, 0, vec_tile_idx});
    warp::store(g.gSums, sums, {0, 0, 0, vec_tile_idx});

#if DEBUG_STEP_BY_STEP
    __syncwarp();
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 9: Stored max/sum to global\n", blockIdx.x, blockIdx.y);
    }
#endif

    // ===== STEP 10: Compute attention output: softmax_scores @ V =====
    rt<float, n_seqs, d_head> attn_out;
    warp::zero(attn_out);

    // Convert scores to bf16 for mma
    rt<bf16, n_seqs, chunk_size> scores_bf16;
    scores_bf16 = scores;  // This copies and converts

    warp::mma_AB(attn_out, scores_bf16, Vr, attn_out);

#if DEBUG_STEP_BY_STEP
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 10: Computed attention output\n", blockIdx.x, blockIdx.y);
    }
#endif

    // ===== STEP 11: Store attention output to global memory =====
    const int attn_offset = max_sum_offset * d_head;
    float* attn_ptr = g.gAttns.raw_ptr + attn_offset;

    // Use shared memory as intermediate for the store
    st<float, n_seqs, d_head> &out_smem = al.allocate<st<float, n_seqs, d_head>>();
    warp::store(out_smem, attn_out);
    __syncwarp();

    // Copy to global - only copy n rows (might be partial)
    for (int row = 0; row < n; row++) {
        for (int col = lane; col < d_head; col += BLOCK_SIZE) {
            attn_ptr[row * d_head + col] = out_smem[{row, col}];
        }
    }

#if DEBUG_MODE
    __syncwarp();
    if (lane == 0) {
        printf("[Block(%d,%d)] Step 11: Kernel complete! attn_offset=%d\n",
               blockIdx.x, blockIdx.y, attn_offset);
        printf("[Block(%d,%d)]   attn_out[0,0]=%f\n",
               blockIdx.x, blockIdx.y, out_smem[{0, 0}]);
    }
#endif
}

#include "harness.impl"
