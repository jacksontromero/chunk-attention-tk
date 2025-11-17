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

template<int n_seqs, int chunk_size, int d_head>
__global__ void attn_chunk_first_tk(const __grid_constant__ attn_chunk_first_globals<d_head> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    const uint32_t head_idx = blockIdx.x;
    const uint32_t chunk_idx = blockIdx.y;

    const int seq_begin = g.gBegins[chunk_idx];
    const int seq_end = g.gEnds[chunk_idx];
    const int n = seq_end - seq_begin;

    // const int n_heads = g.n_heads;
    // const int* offsets = g.gOffsets;

    // const uint32_t q_row_offset = seq_begin * n_heads * d_head + head_idx * d_head;
    // const uint32_t kv_row_offset = head_idx * chunk_size * d_head;
    // const int result_offset = offsets[chunk_idx];
    // const uint32_t max_sum_offset = result_offset * n_heads + head_idx * n;
    // const uint32_t attn_offset = max_sum_offset * d_head;

    auto* __restrict__ k = reinterpret_cast<float*>(g.gKeys[chunk_idx]);

    shared_tile<d_head> &Qs = al.allocate<shared_tile<d_head>>();
    shared_tile<d_head> &Ks = al.allocate<shared_tile<d_head>>();
    const int rows_per_tile = TILE_SIZE<d_head>;

    // thread block process a [n_seqs, d_head] tile of Q starting from q+q_row_offset with stride n_heads * d_head
    // thread block processes a [chunk_size, d_head] tile of K starting from k + kv_row_offset with stride d_head

    int q_iters = (n_seqs + rows_per_tile - 1) / rows_per_tile;
    for (int i = 0; i < q_iters; i++) {
        int q_start = seq_begin + i * rows_per_tile;
        load(Qs, g.gQ_S_H_D, {q_start, head_idx, 0, 0});

        int k_iters = (chunk_size + rows_per_tile - 1) / rows_per_tile;
        for (int j = 0; j < k_iters; j++) {
            int k_start = j * rows_per_tile;
            load(Ks, k, {head_idx, k_start, 0});

        }
    }
}

// Q: [n_seqs, d_head] w/ stride n_heads * d_head. COULD get [tile, d_head]
// K: [chunk_size, d_head] w/ stride d_head. COULD get [tile, d_head]

// NOTES:
// - In kernel_cuda, it's assumed that n_seqs*chunk_size fits in shared memory. We should make the same assumption here.
// - Need to figure out if K is row or column layout - changes best loading strategy.
// - Maybe we should just get a working version that assumes everything fits in shared memory? Honestly a good idea ngl


#include "harness.impl"
