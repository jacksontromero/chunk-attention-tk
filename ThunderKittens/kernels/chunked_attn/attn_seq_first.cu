#include "kittens.cuh"
#include "../../include/kittens.cuh"
using namespace kittens;

constexpr int NUM_WARPS = 4;
constexpr int BLOCK_SIZE = NUM_WARPS * 32;

template<int chunk_size, int d_head>
struct attn_seq_first_globals {
    gl<bf16, -1, -1, -1, -1> output;
    gl<float, -1, -1, -1, -1> attns;
    gl<float, -1, -1, -1, -1> maxs;
    gl<float, -1, -1, -1, -1> sums;
    gl<int, -1, -1, -1, -1> offsets;
    gl<int, -1, -1, -1, -1> begins;
    gl<int, -1, -1, -1, -1> ends;
    const bf16* Q;
    void** keys;
    void** values;
    const int* seq_chunk_map;
    const int* seq_n_tokens;
    int seq_chunk_map_stride;
    float scale;
    int n_heads;
    int n_shared_chunks;
    int delta_tokens;
};

template<int chunk_size, int d_head>
__global__ void __launch_bounds__(BLOCK_SIZE)
attn_seq_first_tk(const __grid_constant__ attn_seq_first_globals<chunk_size, d_head> g) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int warp = kittens::warpid();
    const int lane = kittens::laneid();

    const int seq_length = g.seq_n_tokens[seq_idx] + g.delta_tokens;
    const int last_chunk_unmask_tokens = seq_length % chunk_size;
    const int chunk_num = (seq_length + chunk_size - 1) / chunk_size;

    const int q_row_offset = seq_idx * g.n_heads * d_head + head_idx * d_head;
    const int kv_row_offset = head_idx * chunk_size * d_head;

    const bf16* q_ptr = g.Q + q_row_offset;
    bf16* output_ptr = g.output.raw_ptr + q_row_offset;
    const int* seq_mapping = g.seq_chunk_map + seq_idx * g.seq_chunk_map_stride;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    sv<bf16, d_head> &Q_sv = al.allocate<sv<bf16, d_head>>();
    sv<float, d_head> *all_out_sv = al.allocate<sv<float, d_head>, NUM_WARPS>();
    sv<float, d_head> &out_sv = all_out_sv[warp];
    
    // KV tiles: [NUM_WARPS] of st
    st<bf16, chunk_size, d_head> *KV_tiles = al.allocate<st<bf16, chunk_size, d_head>, NUM_WARPS>();
    st<bf16, chunk_size, d_head> &KV_s = KV_tiles[warp];

    __shared__ float warp_max[NUM_WARPS];
    __shared__ float warp_sum[NUM_WARPS];
    __shared__ float warp_scale[NUM_WARPS];
    __shared__ float inv_sum_s;

    // Load Q (coalesced, all threads participate)
    // Q_sv is d_head=128. Block size=128.
    if (threadIdx.x < d_head) {
        Q_sv[threadIdx.x] = q_ptr[threadIdx.x];
    }

    // Init output
    warp::zero(out_sv);

    // Init max/sum
    if (lane == 0) {
        warp_max[warp] = -INFINITY;
        warp_sum[warp] = 0.0f;
    }
    __syncthreads();

    float score_max = -INFINITY;
    float score_sum = 0.0f;

    using kv_tile_t = rt<float, chunk_size, d_head>;
    using q_rv_t = row_vec<kv_tile_t>;
    using scores_cv_t = col_vec<kv_tile_t>;
    using out_rv_t = row_vec<kv_tile_t>;

    for (int i = warp; i < chunk_num; i += NUM_WARPS) {
        const int chunk_idx = seq_mapping[i];

        if (chunk_idx < g.n_shared_chunks) {
             // Path A: Cached
            const int result_offset = g.offsets[chunk_idx];
            const int seq_begin = g.begins[chunk_idx];
            const int seq_end = g.ends[chunk_idx];
            const int max_sum_offset = result_offset * g.n_heads +
                                       head_idx * (seq_end - seq_begin) +
                                       seq_idx - seq_begin;
            const int attn_offset = max_sum_offset * d_head;

            float cached_max = 0.0f, cached_sum = 0.0f;
            if (lane == 0) {
                cached_max = g.maxs.raw_ptr[max_sum_offset];
                cached_sum = g.sums.raw_ptr[max_sum_offset];
            }
            cached_max = __shfl_sync(0xffffffff, cached_max, 0);
            cached_sum = __shfl_sync(0xffffffff, cached_sum, 0);

            float new_score_max = fmaxf(score_max, cached_max);
            float cached_scale = __expf(cached_max - new_score_max);
            float scale_old = __expf(score_max - new_score_max);

            score_max = new_score_max;
            score_sum = cached_sum * cached_scale + score_sum * scale_old;

            const float* cached_attn = g.attns.raw_ptr + attn_offset;
            for (int j = lane; j < d_head; j += 32) {
                out_sv[j] = out_sv[j] * scale_old + cached_attn[j] * cached_scale;
            }
            __syncwarp();
            continue;
        }

        // Path B: Fresh
        const bf16* K_ptr = reinterpret_cast<const bf16*>(g.keys[chunk_idx]) + kv_row_offset;
        const bf16* V_ptr = reinterpret_cast<const bf16*>(g.values[chunk_idx]) + kv_row_offset;

        // Manual load K into KV_s (per warp)
        for(int j = lane; j < chunk_size * d_head; j += 32) {
            // kv_dst[j] = K_ptr[j]; // Linear write is WRONG for swizzled st
            KV_s[{j / d_head, j % d_head}] = K_ptr[j];
        }
        __syncwarp(); 
        
        kv_tile_t K_r;
        warp::load(K_r, KV_s);

        q_rv_t Q_rv;
        warp::load(Q_rv, Q_sv);

        kv_tile_t QK_tile;
        warp::broadcast_col(QK_tile, Q_rv);
        warp::mul(QK_tile, QK_tile, K_r);

        scores_cv_t scores_cv;
        warp::row_sum(scores_cv, QK_tile);
        warp::mul(scores_cv, scores_cv, g.scale);

        if (i == chunk_num - 1 && last_chunk_unmask_tokens != 0) {
            warp::apply(scores_cv, scores_cv,
                        [last_chunk_unmask_tokens] __device__ (int idx, float v) {
                            return (idx < last_chunk_unmask_tokens) ? v : -INFINITY;
                        });
        }

        float chunk_max;
        warp::max(chunk_max, scores_cv, score_max);
        warp::sub(scores_cv, scores_cv, chunk_max);
        warp::exp(scores_cv, scores_cv);

        float chunk_sum;
        warp::sum(chunk_sum, scores_cv);

        float scale_old = __expf(score_max - chunk_max);
        score_max = chunk_max;
        score_sum = score_sum * scale_old + chunk_sum;

        // Manual load V
        for(int j = lane; j < chunk_size * d_head; j += 32) {
            KV_s[{j / d_head, j % d_head}] = V_ptr[j];
        }
        __syncwarp(); 

        kv_tile_t V_r;
        warp::load(V_r, KV_s);

        kv_tile_t SV_tile;
        warp::broadcast_row(SV_tile, scores_cv);
        warp::mul(SV_tile, SV_tile, V_r);

        out_rv_t new_out_rv;
        warp::col_sum(new_out_rv, SV_tile);

        out_rv_t out_rv;
        warp::load(out_rv, out_sv);
        warp::mul(out_rv, out_rv, scale_old);
        warp::add(out_rv, out_rv, new_out_rv);
        warp::store(out_sv, out_rv);
    }

    if (lane == 0) {
        warp_max[warp] = score_max;
        warp_sum[warp] = score_sum;
    }
    __syncthreads();

    // Cross-warp reduction (same as before)
    if (warp == 0) {
        rv<float, 32, ducks::rv_layout::naive> max_rv;
        max_rv[0][0] = (lane < NUM_WARPS) ? warp_max[lane] : -INFINITY;
        float global_max;
        warp::max(global_max, max_rv);

        float my_scale = (lane < NUM_WARPS) ? __expf(warp_max[lane] - global_max) : 0.0f;
        if (lane < NUM_WARPS) warp_scale[lane] = my_scale;

        rv<float, 32, ducks::rv_layout::naive> sum_rv;
        sum_rv[0][0] = (lane < NUM_WARPS) ? (warp_sum[lane] * my_scale) : 0.0f;
        float total_sum;
        warp::sum(total_sum, sum_rv);

        if (lane == 0) {
            inv_sum_s = __fdividef(1.0f, total_sum + 1e-6f);
        }
    }
    __syncthreads();

    // Final normalization
    // Use manual loop carefully
    if (warp == 0) {
        for(int d = lane; d < d_head; d += 32) {
            float sum = 0.0f;
            #pragma unroll
            for(int w = 0; w < NUM_WARPS; w++) {
                sum += all_out_sv[w][d] * warp_scale[w];
            }
            float val = sum * inv_sum_s;
            // Write directly to global memory to avoid shared memory intermediate issues
            output_ptr[d] = __float2bfloat16(val);
        }
    }
    __syncthreads();
}

#include "harness_seq_first.impl"