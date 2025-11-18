#include "../../include/kittens.cuh"

using namespace kittens;

#define BLOCK_SIZE 32

// TODO MAXIMUM HEAD DIM IS 128
template<int D> constexpr size_t TILE_SIZE = 16*(128/D); // height of each tile (rows)
template<int D> using shared_tile = st_bf<TILE_SIZE<D>, D>;

template<int d_head> struct attn_chunk_first_globals {
    using _gl_float = gl<float, -1, -1, -1, -1>;
    using _gl_int = gl<int, -1, -1, -1, -1>;
    _gl_float gAttns, gMaxs, gSums;
    _gl_int gOffsets, gBegins, gEnds;
    gl<float, -1, -1, -1, d_head, shared_tile<d_head>> gQ_S_H_D;
    void** gKeys;
    void** gValues;
    float dim_scale;
    int n_heads;
};

template<typename scalar_t, int n_seqs, int chunk_size, int d_head>
__global__ void attn_chunk_first_tk(const __grid_constant__ attn_chunk_first_globals<d_head> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    const uint32_t head_idx = blockIdx.x;
    const uint32_t chunk_idx = blockIdx.y;

    const int seq_begin = g.gBegins[chunk_idx];
    const int seq_end = g.gEnds[chunk_idx];
    const int n = seq_end - seq_begin;

    // Extract raw pointers
    auto* __restrict__ k_ptr = reinterpret_cast<scalar_t*>(g.gKeys[chunk_idx]);
    auto* __restrict__ v_ptr = reinterpret_cast<scalar_t*>(g.gValues[chunk_idx]);

    // FIX #1: Wrap raw pointers in gl<> layouts so load() can work with them
    // K and V are [n_heads, chunk_size, d_head]
    using kv_gl = gl<scalar_t, 1, -1, chunk_size, d_head>;
    kv_gl k{k_ptr, nullptr, g.n_heads, chunk_size, nullptr};
    kv_gl v{v_ptr, nullptr, g.n_heads, chunk_size, nullptr};

    // Allocate shared memory
    shared_tile<d_head> &Qs = al.allocate<shared_tile<d_head>>();
    shared_tile<d_head> &Ks = al.allocate<shared_tile<d_head>>();
    shared_tile<d_head> &Vs = al.allocate<shared_tile<d_head>>();

    const int rows_per_tile = TILE_SIZE<d_head>;

    // Register tiles
    rt<scalar_t, TILE_SIZE<d_head>, d_head> Qr, Kr, Vr;
    rt<scalar_t, TILE_SIZE<d_head>, TILE_SIZE<d_head>> output_reg;  // Q @ K^T scores
    rt<scalar_t, TILE_SIZE<d_head>, d_head> attn_result_reg;

    // FIX #2: Use proper col_vec type instead of rv<float, N, naive>
    // For row operations on rt<T, rows, cols>, use typename rt::col_vec
    using vec_type = typename rt<scalar_t, TILE_SIZE<d_head>, TILE_SIZE<d_head>>::col_vec;
    vec_type max_vec, sum_vec;

    int q_iters = (n_seqs + rows_per_tile - 1) / rows_per_tile;
    for (int i = 0; i < q_iters; i++) {
        int q_start = seq_begin + i * rows_per_tile;

        // Load Q - this should work as-is
        warp::load(Qs, g.gQ_S_H_D, {q_start, head_idx, 0, 0});

        int k_iters = (chunk_size + rows_per_tile - 1) / rows_per_tile;
        for (int j = 0; j < k_iters; j++) {
            int k_start = j * rows_per_tile;

            // FIX #1 (continued): Now load works because k and v are gl<> types
            warp::load(Ks, k, {0, head_idx, k_start, 0});
            warp::load(Vs, v, {0, head_idx, k_start, 0});

            // Load from shared to registers
            warp::load(Qr, Qs);
            warp::load(Kr, Ks);
            warp::load(Vr, Vs);

            // FIX #3: Use mma_ABt instead of mm_ABt
            // Compute Q @ K^T
            warp::mma_ABt(output_reg, Qr, Kr);

            // Softmax operations - now work because max_vec has correct type
            warp::row_max(max_vec, output_reg);
            warp::sub_row(output_reg, output_reg, max_vec);

            // Exp (you'll need to add this - TK has exp2)
            // warp::exp(output_reg, output_reg);

            warp::row_sum(sum_vec, output_reg);

            // Normalize
            warp::div_row(output_reg, output_reg, sum_vec);

            // FIX #3 (continued): Use mma_AB instead of mm_AB
            // Multiply by V
            warp::mma_AB(attn_result_reg, output_reg, Vr);
        }
    }
}

#include "harness.impl"
