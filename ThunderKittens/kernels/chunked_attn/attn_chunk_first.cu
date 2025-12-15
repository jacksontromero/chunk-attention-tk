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
 *   - Cooperative loading across all 128 threads
 *   - Warp-level MMA with fp32 accumulation
 *   - Efficient warp-level softmax primitives
 */

#include "kittens.cuh"
#include "../../include/kittens.cuh"
using namespace kittens;

// Configuration: 4 warps, each handling 16 rows
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_SIZE = NUM_WARPS * kittens::WARP_THREADS;

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
    // Shared memory allocation
    // =========================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // K, V shared by all warps
    st<bf16, chunk_size, d_head> &K_s = al.allocate<st<bf16, chunk_size, d_head>>();
    st<bf16, chunk_size, d_head> &V_s = al.allocate<st<bf16, chunk_size, d_head>>();

    // Per-warp Q and output tiles
    auto &Q_s = al.allocate<st<bf16, ROWS_PER_WARP, d_head>, NUM_WARPS>()[warp];
    auto &out_s = al.allocate<st<float, ROWS_PER_WARP, d_head>, NUM_WARPS>()[warp];

    // Shared vectors for max/sum
    auto &max_sv = al.allocate<sv<float, ROWS_PER_WARP>, NUM_WARPS>()[warp];
    auto &sum_sv = al.allocate<sv<float, ROWS_PER_WARP>, NUM_WARPS>()[warp];

    // =========================================================================
    // Load K, V cooperatively
    // =========================================================================
    const bf16* K_ptr = reinterpret_cast<const bf16*>(g.keys[chunk]) + head * chunk_size * d_head;
    const bf16* V_ptr = reinterpret_cast<const bf16*>(g.values[chunk]) + head * chunk_size * d_head;

    for (int i = threadIdx.x; i < chunk_size * d_head; i += BLOCK_SIZE) {
        K_s[{i / d_head, i % d_head}] = K_ptr[i];
        V_s[{i / d_head, i % d_head}] = V_ptr[i];
    }

    // =========================================================================
    // Load Q (each warp loads its rows)
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
    // Warp-level MMA: scores = Q @ K^T
    // =========================================================================
    rt<bf16, ROWS_PER_WARP, d_head> Q_r;
    rt<bf16, chunk_size, d_head> K_r;
    warp::load(Q_r, Q_s);
    warp::load(K_r, K_s);

    rt<float, ROWS_PER_WARP, chunk_size> scores_r;
    warp::zero(scores_r);
    warp::mma_ABt(scores_r, Q_r, K_r, scores_r);
    warp::mul(scores_r, scores_r, g.scale);

    // =========================================================================
    // Warp-level softmax
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
    __syncwarp();

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
    rt<bf16, chunk_size, d_head, ducks::rt_layout::col> V_r;
    warp::load(V_r, V_s);

    rt<bf16, ROWS_PER_WARP, chunk_size> exp_scores_bf16;
    exp_scores_bf16 = scores_r;

    rt<float, ROWS_PER_WARP, d_head> out_r;
    warp::zero(out_r);
    warp::mma_AB(out_r, exp_scores_bf16, V_r, out_r);

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
