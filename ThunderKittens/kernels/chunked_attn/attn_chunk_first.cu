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
 * Block: 64 threads (2 warps), each warp handles n_seqs/2 rows
 *
 * Differences from original:
 *   - Uses bf16 instead of half (similar precision, better HW support)
 *   - Uses TK's warp:: MMA operations instead of WMMA
 *   - Row-parallel: 2 warps each handle half the query rows
 */

#include "kittens.cuh"
using namespace kittens;

// Configuration: 2 warps for row parallelism (TK tiles require rows ≥ 16)
constexpr int NUM_WARPS = 2;
constexpr int BLOCK_SIZE = NUM_WARPS * 32;

/**
 * Kernel globals - mirrors original kernel parameters
 */
template<int max_n_seqs, int d_head>
struct attn_chunk_first_globals {
    // Outputs
    gl<float, -1, -1, -1, -1> attns;   // [total_seqs, d_head] partial attention
    gl<float, -1, -1, -1, -1> maxs;    // [total_seqs] row maxima
    gl<float, -1, -1, -1, -1> sums;    // [total_seqs] row sums

    // Chunk metadata
    gl<int, -1, -1, -1, -1> offsets;   // [n_chunks] output offset per chunk
    gl<int, -1, -1, -1, -1> begins;    // [n_chunks] seq range start
    gl<int, -1, -1, -1, -1> ends;      // [n_chunks] seq range end

    // Inputs
    const bf16* Q;                      // [n_seqs, n_heads, d_head]
    void** keys;                        // [n_chunks] -> [n_heads, chunk_size, d_head]
    void** values;                      // [n_chunks] -> [n_heads, chunk_size, d_head]

    float scale;                        // 1/sqrt(d_head)
    int n_heads;
};

template<int max_n_seqs, int chunk_size, int d_head>
__global__ void __launch_bounds__(BLOCK_SIZE)
attn_chunk_first_tk(const __grid_constant__ attn_chunk_first_globals<max_n_seqs, d_head> g) {
    static_assert(max_n_seqs % NUM_WARPS == 0, "max_n_seqs must be divisible by NUM_WARPS");
    constexpr int ROWS_PER_WARP = max_n_seqs / NUM_WARPS;

    // Thread indexing
    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int my_row_start = warp * ROWS_PER_WARP;

    // Get sequence range for this chunk
    const int seq_begin = g.begins[chunk];
    const int seq_end = g.ends[chunk];
    const int n = seq_end - seq_begin;
    if (n <= 0 || n > max_n_seqs) return;

    // =========================================================================
    // Shared memory allocation
    // =========================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // K, V shared by all warps; Q, scores, output per-warp
    st<bf16, chunk_size, d_head> &K_s = al.allocate<st<bf16, chunk_size, d_head>>();
    st<bf16, chunk_size, d_head> &V_s = al.allocate<st<bf16, chunk_size, d_head>>();
    auto &Q_s = al.allocate<st<bf16, ROWS_PER_WARP, d_head>, NUM_WARPS>()[warp];
    auto &scores_s = al.allocate<st<float, ROWS_PER_WARP, chunk_size>, NUM_WARPS>()[warp];
    auto &out_s = al.allocate<st<float, ROWS_PER_WARP, d_head>, NUM_WARPS>()[warp];

    // =========================================================================
    // Load K, V (cooperative across all threads)
    // Original: loaded via matrix_multiply_gAB_sC's internal tiling
    // =========================================================================
    const bf16* K_ptr = reinterpret_cast<const bf16*>(g.keys[chunk]) + head * chunk_size * d_head;
    const bf16* V_ptr = reinterpret_cast<const bf16*>(g.values[chunk]) + head * chunk_size * d_head;

    for (int i = threadIdx.x; i < chunk_size * d_head; i += BLOCK_SIZE) {
        K_s[{i / d_head, i % d_head}] = K_ptr[i];
        V_s[{i / d_head, i % d_head}] = V_ptr[i];
    }

    // =========================================================================
    // Load Q (each warp loads its rows)
    // Original: Q layout is [n_seqs, n_heads, d_head], accessed as q + seq*n_heads*d_head + head*d_head
    // =========================================================================
    const bf16* Q_base = g.Q + seq_begin * g.n_heads * d_head + head * d_head;

    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = my_row_start + r;
        for (int c = lane; c < d_head; c += 32) {
            Q_s[{r, c}] = (global_row < n) ? Q_base[global_row * g.n_heads * d_head + c]
                                           : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // =========================================================================
    // Compute scores = Q @ K^T * scale
    // Original: matrix_multiply_gAB_sC then warp_vector_scale
    // =========================================================================
    rt<bf16, ROWS_PER_WARP, d_head> Q_r;
    rt<bf16, chunk_size, d_head> K_r;
    warp::load(Q_r, Q_s);
    warp::load(K_r, K_s);

    rt<float, ROWS_PER_WARP, chunk_size> scores_r;
    warp::zero(scores_r);
    warp::mma_ABt(scores_r, Q_r, K_r, scores_r);
    warp::mul(scores_r, scores_r, g.scale);
    warp::store(scores_s, scores_r);
    __syncwarp();

    // =========================================================================
    // Softmax: compute max, exp, sum per row
    // Original: warp_vector_max, warp_cal_exp, warp_vector_sum
    // =========================================================================
    const int result_offset = g.offsets[chunk];
    const int out_base = result_offset * g.n_heads + head * n;
    float* maxs_out = g.maxs.raw_ptr + out_base;
    float* sums_out = g.sums.raw_ptr + out_base;

    for (int r = lane; r < ROWS_PER_WARP; r += 32) {
        int global_row = my_row_start + r;
        if (global_row >= n) continue;

        // Row max
        float max_val = -INFINITY;
        for (int c = 0; c < chunk_size; c++)
            max_val = fmaxf(max_val, scores_s[{r, c}]);
        maxs_out[global_row] = max_val;

        // Exp and sum
        float sum_val = 0.0f;
        for (int c = 0; c < chunk_size; c++) {
            float v = expf(scores_s[{r, c}] - max_val);
            scores_s[{r, c}] = v;
            sum_val += v;
        }
        sums_out[global_row] = sum_val;
    }
    __syncwarp();

    // =========================================================================
    // Compute output = exp_scores @ V
    // Original: matrix_multiply_sA_gBC
    // =========================================================================
    rt<bf16, chunk_size, d_head, ducks::rt_layout::col> V_r;
    warp::load(V_r, V_s);

    rt<float, ROWS_PER_WARP, chunk_size> exp_scores_r;
    warp::load(exp_scores_r, scores_s);

    rt<bf16, ROWS_PER_WARP, chunk_size> exp_scores_bf16;
    exp_scores_bf16 = exp_scores_r;

    rt<float, ROWS_PER_WARP, d_head> out_r;
    warp::zero(out_r);
    warp::mma_AB(out_r, exp_scores_bf16, V_r, out_r);
    warp::store(out_s, out_r);
    __syncwarp();

    // =========================================================================
    // Store output
    // Original: direct copy or via shared_output for partial chunks
    // =========================================================================
    float* attns_out = g.attns.raw_ptr + out_base * d_head;

    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = my_row_start + r;
        if (global_row >= n) continue;
        for (int c = lane; c < d_head; c += 32)
            attns_out[global_row * d_head + c] = out_s[{r, c}];
    }
}

#include "harness.impl"
