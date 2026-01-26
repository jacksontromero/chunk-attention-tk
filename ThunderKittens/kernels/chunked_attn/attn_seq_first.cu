#include "kittens.cuh"
#include "../../include/kittens.cuh"
using namespace kittens;

constexpr int NUM_WARPS = 4;
constexpr int BLOCK_SIZE = NUM_WARPS * 32;

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

// 16-byte async copy (8 bf16 elements)
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// Warp-level async load helper (linear layout)
template<int ROWS, int COLS>
__device__ __forceinline__ void load_tile_async_warp_linear(st<bf16, ROWS, COLS>& dst, const bf16* src) {
    constexpr int elem_per_memcpy = 8;
    constexpr int memcpy_per_row = COLS / elem_per_memcpy;
    constexpr int total_elements = ROWS * COLS;
    constexpr int total_memcpy = total_elements / elem_per_memcpy;
    constexpr int WARP_SIZE = 32;
    constexpr int calls_per_thread = (total_memcpy + WARP_SIZE - 1) / WARP_SIZE;

    const int lane = kittens::laneid();
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    #pragma unroll
    for (int i = 0; i < calls_per_thread; i++) {
        int load_idx = i * WARP_SIZE + lane;
        if (load_idx < total_memcpy) {
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;
            uint32_t offset = (row * COLS + col) * sizeof(bf16);
            cp_async_16(dst_ptr + offset, &src[row * COLS + col]);
        }
    }
}

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

// ============================================================================
// Phase 2: Scalar dot product with TK primitives
// ============================================================================
template<int d_head>
__device__ __forceinline__ float compute_dot_product_tk(
    const sv<bf16, d_head>& Q_sv,
    const bf16* __restrict__ K_row_ptr,
    int lane
) {
    float sum = 0.0f;

    #pragma unroll
    for (int d = lane; d < d_head; d += 32) {
        float q_val = __bfloat162float(Q_sv[d]);
        float k_val = __bfloat162float(K_row_ptr[d]);
        sum += q_val * k_val;
    }

    // Warp reduction using shuffles
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    return sum;
}

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

    // ========================================================================
    // Shared memory allocation using TK's shared_allocator
    // ========================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Q vector in shared memory (TK type)
    sv<bf16, d_head> &Q_sv = al.allocate<sv<bf16, d_head>>();

    // Per-warp output vectors (TK type)
    sv<float, d_head> *all_out_sv = al.allocate<sv<float, d_head>, NUM_WARPS>();
    sv<float, d_head> &out_sv = all_out_sv[warp];

    // Per-warp scores (TK shared vector)
    sv<float, chunk_size> *all_scores_sv = al.allocate<sv<float, chunk_size>, NUM_WARPS>();
    sv<float, chunk_size> &scores_sv = all_scores_sv[warp];

    // Per-warp KV tiles (TK type)
    st<bf16, chunk_size, d_head> *all_KV_st = al.allocate<st<bf16, chunk_size, d_head>, NUM_WARPS>();
    st<bf16, chunk_size, d_head> &KV_st = all_KV_st[warp];

    // Cross-warp reduction variables
    __shared__ float warp_max[NUM_WARPS];
    __shared__ float warp_sum[NUM_WARPS];
    __shared__ float warp_scale[NUM_WARPS];
    __shared__ float inv_sum_s;

    // ========================================================================
    // Load Q to shared memory
    // ========================================================================
    if (threadIdx.x < d_head) {
        Q_sv[threadIdx.x] = q_ptr[threadIdx.x];
    }

    // Initialize output to zero
    for (int d = lane; d < d_head; d += 32) {
        out_sv[d] = 0.0f;
    }

    if (lane == 0) {
        warp_max[warp] = -INFINITY;
        warp_sum[warp] = 0.0f;
    }
    __syncthreads();

    // ========================================================================
    // TK Register types for softmax ops (16-element vectors)
    // ========================================================================
    constexpr int SCORE_CHUNK = 16;
    using score_rv_t = rv<float, SCORE_CHUNK, ducks::rv_layout::naive>;

    float score_max = -INFINITY;
    float score_sum = 0.0f;

    // Each warp processes chunks in round-robin
    for (int i = warp; i < chunk_num; i += NUM_WARPS) {
        const int chunk_idx = seq_mapping[i];

        // ====================================================================
        // Path A: Use cached results
        // ====================================================================
        if (chunk_idx < g.n_shared_chunks) {
            const int result_offset = g.offsets.raw_ptr[chunk_idx];
            const int seq_begin = g.begins.raw_ptr[chunk_idx];
            const int seq_end = g.ends.raw_ptr[chunk_idx];
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
            for (int d = lane; d < d_head; d += 32) {
                out_sv[d] = out_sv[d] * scale_old + cached_attn[d] * cached_scale;
            }
            __syncwarp();
            continue;
        }

        // ====================================================================
        // Path B: Fresh computation using TK primitives
        // ====================================================================
        const bf16* K_ptr = reinterpret_cast<const bf16*>(g.keys[chunk_idx]) + kv_row_offset;
        const bf16* V_ptr = reinterpret_cast<const bf16*>(g.values[chunk_idx]) + kv_row_offset;

        // Load K asynchronously using TK tile
        load_tile_async_warp_linear<chunk_size, d_head>(KV_st, K_ptr);
        cp_async_commit();
        cp_async_wait<0>();
        __syncwarp();

        const bf16* K_smem = &KV_st.data[0];

        // ================================================================
        // Compute Q @ K^T scores using scalar dot products
        // ================================================================
        #pragma unroll
        for (int r = 0; r < chunk_size; r++) {
            const bf16* K_row = K_smem + r * d_head;
            float score = compute_dot_product_tk<d_head>(Q_sv, K_row, lane);
            score = __shfl_sync(0xffffffff, score, 0);
            score *= g.scale;

            if (i == chunk_num - 1 && last_chunk_unmask_tokens != 0) {
                if (r >= last_chunk_unmask_tokens) {
                    score = -INFINITY;
                }
            }

            if (lane == 0) {
                scores_sv[r] = score;
            }
        }
        __syncwarp();

        // ================================================================
        // Softmax using TK register vectors
        // ================================================================
        float current_chunk_max = -INFINITY;

        #pragma unroll
        for (int sc = 0; sc < chunk_size / SCORE_CHUNK; sc++) {
            score_rv_t scores_rv;
            #pragma unroll
            for (int j = 0; j < SCORE_CHUNK; j++) {
                scores_rv[0][j] = scores_sv[sc * SCORE_CHUNK + j];
            }

            float chunk_max;
            warp::max(chunk_max, scores_rv);
            current_chunk_max = fmaxf(current_chunk_max, chunk_max);
        }

        float new_global_max = fmaxf(score_max, current_chunk_max);
        float scale_old = __expf(score_max - new_global_max);
        float scale_new = __expf(current_chunk_max - new_global_max);
        score_max = new_global_max;

        for (int d = lane; d < d_head; d += 32) {
            out_sv[d] = out_sv[d] * scale_old;
        }
        __syncwarp();

        float current_chunk_sum = 0.0f;

        #pragma unroll
        for (int sc = 0; sc < chunk_size / SCORE_CHUNK; sc++) {
            score_rv_t scores_rv;

            #pragma unroll
            for (int j = 0; j < SCORE_CHUNK; j++) {
                scores_rv[0][j] = scores_sv[sc * SCORE_CHUNK + j];
            }

            warp::sub(scores_rv, scores_rv, current_chunk_max);
            warp::exp(scores_rv, scores_rv);

            float chunk_sum;
            warp::sum(chunk_sum, scores_rv);
            current_chunk_sum += chunk_sum;

            #pragma unroll
            for (int j = 0; j < SCORE_CHUNK; j++) {
                scores_sv[sc * SCORE_CHUNK + j] = scores_rv[0][j];
            }
        }
        __syncwarp();

        // Load V asynchronously
        load_tile_async_warp_linear<chunk_size, d_head>(KV_st, V_ptr);
        cp_async_commit();
        cp_async_wait<0>();
        __syncwarp();

        const bf16* V_smem = &KV_st.data[0];

        // ================================================================
        // Compute weighted V sum
        // ================================================================
        for (int d = lane; d < d_head; d += 32) {
            float weighted_sum = 0.0f;

            #pragma unroll
            for (int r = 0; r < chunk_size; r++) {
                float exp_score = scores_sv[r];
                float v_val = __bfloat162float(V_smem[r * d_head + d]);
                weighted_sum += exp_score * v_val;
            }

            out_sv[d] = out_sv[d] + weighted_sum * scale_new;
        }
        __syncwarp();

        score_sum = score_sum * scale_old + current_chunk_sum * scale_new;
    }

    // Store per-warp results
    if (lane == 0) {
        warp_max[warp] = score_max;
        warp_sum[warp] = score_sum;
    }
    __syncthreads();

    // ========================================================================
    // Cross-warp reduction using TK register vectors
    // ========================================================================
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

    // Final normalization and output
    if (warp == 0) {
        for (int d = lane; d < d_head; d += 32) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                sum += all_out_sv[w][d] * warp_scale[w];
            }
            float val = sum * inv_sum_s;
            output_ptr[d] = __float2bfloat16(val);
        }
    }
    __syncthreads();
}

#include "harness_seq_first.impl"
