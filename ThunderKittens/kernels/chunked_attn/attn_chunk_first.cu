#include "../../include/kittens.cuh"
#include "kittens.cuh"

using namespace kittens;

// TODO MAXIMUM HEAD DIM IS 128
// template<int D> constexpr size_t TILE_SIZE = 16*(128/D); // height of each tile (rows)
// template<int D> using shared_tile = st_bf<TILE_SIZE<D>, D>;
// template<int D> using register_tile = rt_fl<TILE_SIZE<D>, D>;
template<typename scalar_t, int n_seqs, int d_head> struct attn_chunk_first_globals {
    using _gl_scalar_t = gl<scalar_t, -1, -1, -1, -1>;
    using _gl_float = gl<float, -1, -1, -1, -1>;
    using _gl_int = gl<int, -1, -1, -1, -1>;
    _gl_scalar_t gAttns;
    _gl_float gMaxs, gSums;
    _gl_int gOffsets, gBegins, gEnds;
    gl<scalar_t, 1, n_seqs, -1, d_head, st<scalar_t, n_seqs, d_head>> gQ_1_S_H_D;
    void** gKeys;
    void** gValues;
    float dim_scale;
    int n_heads;
};

// ONLY REALLY DESIGNED FOR SCALAR_T = float
template<typename scalar_t, int n_seqs, int chunk_size, int d_head>
__global__ void attn_chunk_first_tk(const __grid_constant__ attn_chunk_first_globals<scalar_t, n_seqs, d_head> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    const uint32_t head_idx = blockIdx.x;
    const uint32_t chunk_idx = blockIdx.y;

    const int seq_begin = g.gBegins[chunk_idx];
    const int seq_end = g.gEnds[chunk_idx];
    const int n = seq_end - seq_begin;

    auto* __restrict__ k = reinterpret_cast<scalar_t*>(g.gKeys[chunk_idx]);
    auto* __restrict__ v = reinterpret_cast<scalar_t*>(g.gValues[chunk_idx]);

    st<scalar_t, n_seqs, d_head> &Qs = al.allocate<st<scalar_t, n_seqs, d_head>>();
    st<scalar_t, chunk_size, d_head> &Ks = al.allocate<st<scalar_t, chunk_size, d_head>>();
    // todo this can reuse Ks
    st<scalar_t, chunk_size, d_head> &Vs = al.allocate<st<scalar_t, chunk_size, d_head>>();
    st<scalar_t, n_seqs, chunk_size> &output = al.allocate<st<scalar_t, n_seqs, chunk_size>>();

    // thread block process a [n_seqs, d_head] tile of Q starting from q+q_row_offset with stride n_heads * d_head
    // thread block processes a [chunk_size, d_head] tile of K starting from k + kv_row_offset with stride d_head

    warp::load(Qs, g.gQ_1_S_H_D, {0, seq_begin, head_idx, 0});
    warp::load(Ks, k, {head_idx, 0, 0});
    warp::load(Vs, v, {head_idx, 0, 0});

    __syncthreads();

    rt<scalar_t, n_seqs, d_head> Qr;
    rt<scalar_t, chunk_size, d_head> Kr;
    // todo reuse Kr
    rt<scalar_t, chunk_size, d_head> Vr;
    warp::load(Qr, Qs);
    warp::load(Kr, Ks);
    warp::load(Vr, Vs);

    __syncthreads();

    rt<scalar_t, n_seqs, chunk_size> output_reg;


    kittens::warp::mm_ABt(output_reg, Qr, Kr);
    warp::mul(output_reg, output_reg, g.dim_scale);

    rv<scalar_t, n_seqs> maxes;
    warp::zero(maxes);
    warp::row_max(maxes, output_reg);

    // not using double buffer with different precisions like in kernel_cuda, simplifying assumption for now
    warp::sub_row(output_reg, output_reg, maxes);
    warp::exp(output_reg, output_reg);

    const int result_offset = g.gOffsets[chunk_idx];
    const int max_sum_offset = result_offset * g.n_heads + head_idx * n;

    // write maxes back to global memory
    warp::store(g.gMaxs, maxes, {0, 0, 0, max_sum_offset});
    __syncthreads();

    // reuse maxes for sums
    auto sums = maxes;
    warp::row_sum(sums, output_reg);

    // write sums back to global memory
    warp::store(g.gSums, sums, {0, 0, 0, max_sum_offset});

    __syncthreads();

    const int attn_offset = max_sum_offset * d_head;
    rt<scalar_t, n_seqs, d_head> attn_result_reg;

    // now compute the attention weights
    if (n == n_seqs) {
        kittens::warp::mm_AB(attn_result_reg, output_reg, Vr);
        warp::store(g.gAttns, attn_result_reg, {0, 0, 0, attn_offset});
    } else {
        kittens::warp::mm_AB(attn_result_reg, output_reg, Vr);
        st<scalar_t, n_seqs, d_head> &shared_output = al.allocate<st<scalar_t, n_seqs, d_head>>();
        warp::store(shared_output, attn_result_reg);
        __syncthreads();

        scalar_t* __restrict__ attn_result_global = g.gAttns.raw_ptr + attn_offset;

        for (int row = 0; row < n; ++row) {
            for (int col = threadIdx.x; col < d_head; col += blockDim.x) {
                attn_result_global[row * d_head + col] = shared_output[{row, col}];
            }
        }
    }
}

#include "harness.impl"
