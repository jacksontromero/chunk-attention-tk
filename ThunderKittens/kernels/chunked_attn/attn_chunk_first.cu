#include "../../include/kittens.cuh"
#include "kittens.cuh"

using namespace kittens;

// ===== Kernel configuration =====
constexpr int BLOCK_SIZE = 32;

// Globals structure
template<int max_n_seqs, int d_head>
struct attn_chunk_first_globals {
    using _gl_float = gl<float, -1, -1, -1, -1>;
    using _gl_int = gl<int, -1, -1, -1, -1>;

    _gl_float gAttns;
    _gl_float gMaxs, gSums;
    _gl_int gOffsets, gBegins, gEnds;

    const bf16* gQ;
    void** gKeys;
    void** gValues;

    float dim_scale;
    int n_heads;
};

template<int max_n_seqs, int chunk_size, int d_head>
__global__ void __launch_bounds__(BLOCK_SIZE) attn_chunk_first_tk(
    const __grid_constant__ attn_chunk_first_globals<max_n_seqs, d_head> g
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    const int head_idx = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    const int lane = laneid();

    const int seq_begin = g.gBegins[chunk_idx];
    const int seq_end = g.gEnds[chunk_idx];
    const int n = seq_end - seq_begin;

    if (n <= 0 || n > max_n_seqs) return;

    // ===== Allocate shared memory =====
    st<bf16, max_n_seqs, d_head> &Qs = al.allocate<st<bf16, max_n_seqs, d_head>>();
    st<bf16, chunk_size, d_head> &Ks = al.allocate<st<bf16, chunk_size, d_head>>();
    st<bf16, chunk_size, d_head> &Vs = al.allocate<st<bf16, chunk_size, d_head>>();
    st<float, max_n_seqs, chunk_size> &scores_smem = al.allocate<st<float, max_n_seqs, chunk_size>>();
    st<float, max_n_seqs, d_head> &out_smem = al.allocate<st<float, max_n_seqs, d_head>>();

    // ===== Load Q =====
    const bf16* q_ptr = g.gQ + seq_begin * g.n_heads * d_head + head_idx * d_head;
    for (int row = 0; row < n; row++) {
        for (int col = lane; col < d_head; col += BLOCK_SIZE) {
            Qs[{row, col}] = q_ptr[row * g.n_heads * d_head + col];
        }
    }
    for (int row = n; row < max_n_seqs; row++) {
        for (int col = lane; col < d_head; col += BLOCK_SIZE) {
            Qs[{row, col}] = __float2bfloat16(0.0f);
        }
    }
    __syncwarp();

    // ===== Load K and V =====
    const int kv_offset = head_idx * chunk_size * d_head;
    const bf16* k_ptr = reinterpret_cast<const bf16*>(g.gKeys[chunk_idx]) + kv_offset;
    const bf16* v_ptr = reinterpret_cast<const bf16*>(g.gValues[chunk_idx]) + kv_offset;

    constexpr int total_kv = chunk_size * d_head;
    for (int idx = lane; idx < total_kv; idx += BLOCK_SIZE) {
        int row = idx / d_head, col = idx % d_head;
        Ks[{row, col}] = k_ptr[idx];
        Vs[{row, col}] = v_ptr[idx];
    }
    __syncwarp();

    // ===== Load to registers and compute Q @ K^T =====
    rt<bf16, max_n_seqs, d_head> Qr;
    rt<bf16, chunk_size, d_head> Kr;
    rt<bf16, chunk_size, d_head, ducks::rt_layout::col> Vr;

    warp::load(Qr, Qs);
    warp::load(Kr, Ks);
    warp::load(Vr, Vs);

    rt<float, max_n_seqs, chunk_size> scores;
    warp::zero(scores);
    warp::mma_ABt(scores, Qr, Kr, scores);
    warp::mul(scores, scores, g.dim_scale);

    // Store scores to shared for softmax
    warp::store(scores_smem, scores);
    __syncwarp();

    // ===== Compute row max/sum from shared memory =====
    const int result_offset = g.gOffsets[chunk_idx];
    const int max_sum_offset = result_offset * g.n_heads + head_idx * n;
    float* maxs_ptr = g.gMaxs.raw_ptr + max_sum_offset;
    float* sums_ptr = g.gSums.raw_ptr + max_sum_offset;

    // Each thread handles different rows
    for (int row = lane; row < n; row += BLOCK_SIZE) {
        float max_val = -INFINITY;
        for (int col = 0; col < chunk_size; col++) {
            max_val = fmaxf(max_val, scores_smem[{row, col}]);
        }
        maxs_ptr[row] = max_val;

        // Subtract max and compute exp + sum
        float sum_val = 0.0f;
        for (int col = 0; col < chunk_size; col++) {
            float v = expf(scores_smem[{row, col}] - max_val);
            scores_smem[{row, col}] = v;
            sum_val += v;
        }
        sums_ptr[row] = sum_val;
    }
    __syncwarp();

    // ===== Load softmax scores back to registers and compute output =====
    rt<float, max_n_seqs, chunk_size> scores_exp;
    warp::load(scores_exp, scores_smem);

    rt<float, max_n_seqs, d_head> attn_out;
    warp::zero(attn_out);

    rt<bf16, max_n_seqs, chunk_size> scores_bf16;
    scores_bf16 = scores_exp;
    warp::mma_AB(attn_out, scores_bf16, Vr, attn_out);

    // ===== Store output =====
    warp::store(out_smem, attn_out);
    __syncwarp();

    const int attn_offset = max_sum_offset * d_head;
    float* attn_ptr = g.gAttns.raw_ptr + attn_offset;

    for (int row = 0; row < n; row++) {
        for (int col = lane; col < d_head; col += BLOCK_SIZE) {
            attn_ptr[row * d_head + col] = out_smem[{row, col}];
        }
    }
}

#include "harness.impl"
