#include "../../include/kittens.cuh"

using namespace kittens;

#define BLOCK_SIZE 32

// Template parameters for different head dimensions
template<int d_head> struct attn_chunk_first_globals {
    using _gl_float = gl<float, -1, -1, -1, -1>;
    using _gl_int = gl<int, -1, -1, -1, -1>;

    // Input/output arrays
    _gl_float gAttns, gMaxs, gSums;
    _gl_int gOffsets, gBegins, gEnds;

    // Q, K, V tensors
    gl<float, 1, -1, -1, d_head> gQ;  // [n_heads, n_seqs, d_head]
    void** gKeys;                      // Array of pointers to [n_heads, chunk_size, d_head]
    void** gValues;                    // Array of pointers to [n_heads, chunk_size, d_head]

    float dim_scale;
    int n_heads;
};

template<typename scalar_t, int n_seqs, int chunk_size, int d_head>
__global__ void attn_chunk_first_tk(const __grid_constant__ attn_chunk_first_globals<d_head> g) {
    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int head_idx = blockIdx.x;
    const int chunk_idx = blockIdx.y;

    const int seq_begin = g.gBegins[chunk_idx];
    const int seq_end = g.gEnds[chunk_idx];
    const int n = seq_end - seq_begin;

    // Get K and V pointers for this chunk and create global layouts
    auto* __restrict__ k_ptr = reinterpret_cast<scalar_t*>(g.gKeys[chunk_idx]);
    auto* __restrict__ v_ptr = reinterpret_cast<scalar_t*>(g.gValues[chunk_idx]);

    using kv_gl = gl<scalar_t, 1, -1, chunk_size, d_head>;
    kv_gl k_layout{k_ptr, nullptr, g.n_heads, chunk_size, nullptr};
    kv_gl v_layout{v_ptr, nullptr, g.n_heads, chunk_size, nullptr};

    // Allocate shared memory tiles (16x16 is standard TK subtile size)
    constexpr int TILE_DIM = 16;

    // Use st_fl for float shared tiles
    st_fl<TILE_DIM, d_head> &Q_smem = al.allocate<st_fl<TILE_DIM, d_head>>();
    st_fl<TILE_DIM, d_head> &K_smem = al.allocate<st_fl<TILE_DIM, d_head>>();
    st_fl<TILE_DIM, d_head> &V_smem = al.allocate<st_fl<TILE_DIM, d_head>>();

    // Register tiles for computation
    rt_fl<TILE_DIM, TILE_DIM> attn_block;      // Attention scores (Q @ K^T)
    rt_fl<TILE_DIM, d_head> o_reg;              // Output accumulator

    // Column vectors for softmax (use col_vec helper)
    col_vec<rt_fl<TILE_DIM, TILE_DIM>> max_vec;
    col_vec<rt_fl<TILE_DIM, TILE_DIM>> norm_vec;  // Sum for normalization

    // Process Q tiles (iterate over n_seqs in chunks of TILE_DIM)
    const int q_tiles = (n_seqs + TILE_DIM - 1) / TILE_DIM;

    for (int q_tile = 0; q_tile < q_tiles; q_tile++) {
        // Load Q tile from global memory
        // Q layout: [batch=1, head, seq, dim]
        warpgroup::load(Q_smem, g.gQ, {0, head_idx, seq_begin + q_tile * TILE_DIM, 0});

        // Initialize accumulator for this Q tile
        warpgroup::zero(o_reg);
        warp::neg_infty(max_vec);
        warp::zero(norm_vec);

        // Process K/V tiles (iterate over chunk_size)
        const int k_tiles = (chunk_size + TILE_DIM - 1) / TILE_DIM;

        for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
            // Load K and V tiles
            warpgroup::load(K_smem, k_layout, {0, head_idx, k_tile * TILE_DIM, 0});
            warpgroup::load(V_smem, v_layout, {0, head_idx, k_tile * TILE_DIM, 0});

            // Compute attention scores: Q @ K^T
            // Q_smem is [TILE_DIM, d_head], K_smem is [TILE_DIM, d_head]
            // Result is [TILE_DIM, TILE_DIM]
            warpgroup::mm_ABt(attn_block, Q_smem, K_smem);
            warpgroup::mma_async_wait();

            // Scale by 1/sqrt(d_head)
            warp::mul(attn_block, attn_block, g.dim_scale);

            // Softmax computation
            // 1. Find row max (for numerical stability)
            col_vec<rt_fl<TILE_DIM, TILE_DIM>> max_vec_last;
            warp::copy(max_vec_last, max_vec);
            warp::row_max(max_vec, attn_block, max_vec);  // Accumulate with previous max

            // 2. Subtract max from scores
            warp::sub_row(attn_block, attn_block, max_vec);

            // 3. Compute exp
            // Using exp2 with log2(e) scaling for better hardware support
            constexpr float log2e = 1.44269504089f;
            warp::mul(attn_block, attn_block, log2e);
            warp::exp2(attn_block, attn_block);

            // 4. Update normalizer
            // Rescale previous norm by exp(old_max - new_max)
            warp::sub(max_vec_last, max_vec_last, max_vec);
            warp::mul(max_vec_last, max_vec_last, log2e);
            warp::exp2(max_vec_last, max_vec_last);
            warp::mul(norm_vec, norm_vec, max_vec_last);

            // Add current tile's sum
            warp::row_sum(norm_vec, attn_block, norm_vec);

            // 5. Rescale output accumulator
            warp::mul_row(o_reg, o_reg, max_vec_last);

            // 6. Multiply attention weights by V and accumulate
            warpgroup::mma_AB(o_reg, attn_block, V_smem);
            warpgroup::mma_async_wait();
        }

        // Final normalization
        warp::div_row(o_reg, o_reg, norm_vec);

        // Store results
        // TODO: Set up output global layout and store
        // The output should go to gAttns at appropriate indices
        // Also store max_vec to gMaxs and norm_vec to gSums
    }
}

#include "harness.impl"
